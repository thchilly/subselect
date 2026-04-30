"""Cache layer — parquet + zarr + sqlite catalog.

Three layers, each with one responsibility, per ``docs/refactor.md`` §
Caching strategy:

1. **Parquet** for tabular metric tables. Three path conventions:

   - per-country, per-(scenario, season):
     ``cache/parquet/<country>/<scenario>/<season>/<kind>__<crop_method>.parquet``
     (HPS, BVS, change-spread tables; the M7/M8 hot path).
   - per-country, multi-scenario time-series (M8 country-profile):
     ``cache/parquet/<country>/timeseries/<variable>__<crop_method>.parquet``
   - **global** (no country scope):
     ``cache/parquet/_global/<kind>.parquet`` — first user is M8's
     ``gwl_crossing_years.parquet``.

2. **Zarr** for fields and matrices:
   ``cache/zarr/<country>/<scenario>/<artefact>__<crop_method>.zarr``.
   ``zarr<3`` is pinned in ``environment.yml`` and ``pyproject.toml`` so the
   directory-store format stays stable across the refactor.

3. **SQLite catalog** at ``cache/catalog.sqlite``: one row per artefact with
   ``country, scenario, season, kind, crop_method, code_version, config_hash,
   path, format, created_at``. Catalog rows for global / time-series
   artefacts use the sentinel values ``_global`` (country) and ``n/a``
   (scenario / season / crop_method) so the natural-key ``UNIQUE`` constraint
   stays enforceable without nullable columns.

The catalog is an index, not the source of truth — the underlying parquet /
zarr files are. Throw the catalog away at any time and rebuild via
``Catalog.rescan(...)``; that helper is Phase 1+ scope and not implemented
here.
"""

from __future__ import annotations

import contextlib
import shutil
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import xarray as xr

from subselect.config import Config

GLOBAL_COUNTRY = "_global"
NOT_APPLICABLE = "n/a"
TIMESERIES_SEASON = "timeseries"

ArtefactFormat = Literal["parquet", "zarr"]
CATALOG_FILENAME = "catalog.sqlite"


CATALOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS artefacts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    country         TEXT NOT NULL,
    scenario        TEXT NOT NULL,
    season          TEXT NOT NULL,
    kind            TEXT NOT NULL,
    crop_method     TEXT NOT NULL,
    code_version    TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    path            TEXT NOT NULL,
    format          TEXT NOT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (country, scenario, season, kind, crop_method, config_hash)
);
CREATE INDEX IF NOT EXISTS idx_artefacts_country ON artefacts (country);
CREATE INDEX IF NOT EXISTS idx_artefacts_kind ON artefacts (kind);
"""


class CacheMiss(KeyError):
    """Raised when a requested cache artefact is not in the catalog or on disk."""


@dataclass(frozen=True)
class ArtefactRecord:
    """One row from the catalog. ``path`` is relative to the cache root."""

    country: str
    scenario: str
    season: str
    kind: str
    crop_method: str
    code_version: str
    config_hash: str
    path: str
    format: ArtefactFormat
    created_at: str


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class Catalog:
    """SQLite-backed index of cache artefacts.

    Operations: register / lookup / invalidate / list_all. The natural key is
    ``(country, scenario, season, kind, crop_method, config_hash)``; re-
    registering the same key updates ``path``, ``format``, ``code_version``,
    and ``created_at`` rather than inserting a duplicate.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextlib.contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(CATALOG_SCHEMA)

    def register(
        self,
        *,
        country: str,
        scenario: str,
        season: str,
        kind: str,
        crop_method: str,
        path: str,
        format: ArtefactFormat,
        code_version: str = "",
        config_hash: str = "",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO artefacts (country, scenario, season, kind, crop_method,
                                       code_version, config_hash, path, format)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (country, scenario, season, kind, crop_method, config_hash)
                DO UPDATE SET path = excluded.path,
                              format = excluded.format,
                              code_version = excluded.code_version,
                              created_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                """,
                (
                    country, scenario, season, kind, crop_method,
                    code_version, config_hash, path, format,
                ),
            )

    def lookup(
        self,
        *,
        country: str,
        scenario: str,
        season: str,
        kind: str,
        crop_method: str,
        config_hash: str = "",
    ) -> ArtefactRecord:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT country, scenario, season, kind, crop_method,
                       code_version, config_hash, path, format, created_at
                FROM artefacts
                WHERE country = ? AND scenario = ? AND season = ?
                  AND kind = ? AND crop_method = ? AND config_hash = ?
                """,
                (country, scenario, season, kind, crop_method, config_hash),
            ).fetchone()
        if row is None:
            raise CacheMiss(
                f"no catalog entry for country={country!r} scenario={scenario!r} "
                f"season={season!r} kind={kind!r} crop_method={crop_method!r} "
                f"config_hash={config_hash!r}"
            )
        return ArtefactRecord(*row)

    def invalidate(
        self,
        *,
        country: str | None = None,
        kind: str | None = None,
        crop_method: str | None = None,
        code_version: str | None = None,
    ) -> int:
        """Delete catalog rows matching the filters AND remove their files.

        Each provided filter narrows the deletion. Pass no filters to wipe
        the whole catalog (and every artefact file under the cache root).
        Files that are already missing are skipped silently — partial-state
        cleanup must not error out.

        Returns the number of catalog rows deleted.
        """
        clauses: list[str] = []
        params: list[str] = []
        if country is not None:
            clauses.append("country = ?")
            params.append(country)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if crop_method is not None:
            clauses.append("crop_method = ?")
            params.append(crop_method)
        if code_version is not None:
            clauses.append("code_version = ?")
            params.append(code_version)
        where = " AND ".join(clauses) if clauses else "1=1"

        cache_root = self.db_path.parent
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT path, format FROM artefacts WHERE {where}",
                params,
            ).fetchall()
            for rel_path, fmt in rows:
                _delete_artefact_at(cache_root / rel_path, fmt)
            conn.execute(f"DELETE FROM artefacts WHERE {where}", params)
            return len(rows)

    def list_all(self) -> list[ArtefactRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT country, scenario, season, kind, crop_method,
                       code_version, config_hash, path, format, created_at
                FROM artefacts
                ORDER BY country, scenario, season, kind, crop_method
                """
            ).fetchall()
        return [ArtefactRecord(*r) for r in rows]


def _delete_artefact_at(target: Path, fmt: str) -> None:
    if fmt == "parquet" and target.is_file():
        target.unlink()
    elif fmt == "zarr" and target.is_dir():
        shutil.rmtree(target)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _parquet_path_per_country(
    cache_root: Path, country: str, kind: str, scenario: str, season: str, crop_method: str
) -> Path:
    return (
        cache_root / "parquet" / country / scenario / season /
        f"{kind}__{crop_method}.parquet"
    )


def _parquet_path_timeseries(
    cache_root: Path, country: str, variable: str, crop_method: str
) -> Path:
    return (
        cache_root / "parquet" / country / "timeseries" /
        f"{variable}__{crop_method}.parquet"
    )


def _parquet_path_global(cache_root: Path, kind: str) -> Path:
    return cache_root / "parquet" / GLOBAL_COUNTRY / f"{kind}.parquet"


def _zarr_path(
    cache_root: Path, country: str, scenario: str, artefact: str, crop_method: str
) -> Path:
    return (
        cache_root / "zarr" / country / scenario /
        f"{artefact}__{crop_method}.zarr"
    )


def _resolve_config(config: Config | None) -> Config:
    return config if config is not None else Config.from_env()


def _catalog(config: Config) -> Catalog:
    return Catalog(config.cache_root / CATALOG_FILENAME)


# ---------------------------------------------------------------------------
# Parquet — per-country, per-(scenario, season)
# ---------------------------------------------------------------------------


def write_parquet(
    df: pd.DataFrame,
    *,
    country: str,
    kind: str,
    scenario: str,
    season: str,
    crop_method: str,
    config: Config | None = None,
    code_version: str = "",
    config_hash: str = "",
) -> Path:
    """Write a per-country, per-(scenario, season) tabular metric table."""
    config = _resolve_config(config)
    abs_path = _parquet_path_per_country(
        config.cache_root, country, kind, scenario, season, crop_method
    )
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(abs_path, compression="snappy", index=True)
    rel = abs_path.relative_to(config.cache_root)
    _catalog(config).register(
        country=country, scenario=scenario, season=season, kind=kind,
        crop_method=crop_method, path=str(rel), format="parquet",
        code_version=code_version, config_hash=config_hash,
    )
    return abs_path


def read_parquet(
    *,
    country: str,
    kind: str,
    scenario: str,
    season: str,
    crop_method: str,
    config: Config | None = None,
    config_hash: str = "",
) -> pd.DataFrame:
    config = _resolve_config(config)
    record = _catalog(config).lookup(
        country=country, scenario=scenario, season=season, kind=kind,
        crop_method=crop_method, config_hash=config_hash,
    )
    abs_path = config.cache_root / record.path
    if not abs_path.is_file():
        raise CacheMiss(
            f"catalog entry exists but parquet file is missing on disk: {abs_path}"
        )
    return pd.read_parquet(abs_path)


# ---------------------------------------------------------------------------
# Parquet — per-country time-series
# ---------------------------------------------------------------------------


def write_parquet_timeseries(
    df: pd.DataFrame,
    *,
    country: str,
    variable: str,
    crop_method: str,
    config: Config | None = None,
    code_version: str = "",
    config_hash: str = "",
) -> Path:
    """Write a per-country annual time-series (long-form: model, scenario, year, value)."""
    config = _resolve_config(config)
    abs_path = _parquet_path_timeseries(
        config.cache_root, country, variable, crop_method
    )
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(abs_path, compression="snappy", index=True)
    rel = abs_path.relative_to(config.cache_root)
    _catalog(config).register(
        country=country,
        scenario=NOT_APPLICABLE,
        season=TIMESERIES_SEASON,
        kind=variable,
        crop_method=crop_method,
        path=str(rel),
        format="parquet",
        code_version=code_version,
        config_hash=config_hash,
    )
    return abs_path


def read_parquet_timeseries(
    *,
    country: str,
    variable: str,
    crop_method: str,
    config: Config | None = None,
    config_hash: str = "",
) -> pd.DataFrame:
    config = _resolve_config(config)
    record = _catalog(config).lookup(
        country=country,
        scenario=NOT_APPLICABLE,
        season=TIMESERIES_SEASON,
        kind=variable,
        crop_method=crop_method,
        config_hash=config_hash,
    )
    abs_path = config.cache_root / record.path
    if not abs_path.is_file():
        raise CacheMiss(
            f"catalog entry exists but parquet file is missing on disk: {abs_path}"
        )
    return pd.read_parquet(abs_path)


# ---------------------------------------------------------------------------
# Parquet — global (ensemble-wide, no country scope)
# ---------------------------------------------------------------------------


def write_parquet_global(
    df: pd.DataFrame,
    *,
    kind: str,
    config: Config | None = None,
    code_version: str = "",
    config_hash: str = "",
) -> Path:
    """Write a global artefact (no country scope, e.g. GWL crossing years)."""
    config = _resolve_config(config)
    abs_path = _parquet_path_global(config.cache_root, kind)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(abs_path, compression="snappy", index=True)
    rel = abs_path.relative_to(config.cache_root)
    _catalog(config).register(
        country=GLOBAL_COUNTRY,
        scenario=NOT_APPLICABLE,
        season=NOT_APPLICABLE,
        kind=kind,
        crop_method=NOT_APPLICABLE,
        path=str(rel),
        format="parquet",
        code_version=code_version,
        config_hash=config_hash,
    )
    return abs_path


def read_parquet_global(
    *,
    kind: str,
    config: Config | None = None,
    config_hash: str = "",
) -> pd.DataFrame:
    config = _resolve_config(config)
    record = _catalog(config).lookup(
        country=GLOBAL_COUNTRY,
        scenario=NOT_APPLICABLE,
        season=NOT_APPLICABLE,
        kind=kind,
        crop_method=NOT_APPLICABLE,
        config_hash=config_hash,
    )
    abs_path = config.cache_root / record.path
    if not abs_path.is_file():
        raise CacheMiss(
            f"catalog entry exists but parquet file is missing on disk: {abs_path}"
        )
    return pd.read_parquet(abs_path)


# ---------------------------------------------------------------------------
# Zarr — per-country fields and matrices
# ---------------------------------------------------------------------------


def write_zarr(
    ds: xr.Dataset | xr.DataArray,
    *,
    country: str,
    scenario: str,
    artefact: str,
    crop_method: str,
    config: Config | None = None,
    code_version: str = "",
    config_hash: str = "",
) -> Path:
    """Write a per-country xarray field / matrix to a zarr directory store."""
    config = _resolve_config(config)
    abs_path = _zarr_path(config.cache_root, country, scenario, artefact, crop_method)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=ds.name or artefact)
    ds.to_zarr(abs_path, mode="w")
    rel = abs_path.relative_to(config.cache_root)
    _catalog(config).register(
        country=country,
        scenario=scenario,
        season=NOT_APPLICABLE,
        kind=artefact,
        crop_method=crop_method,
        path=str(rel),
        format="zarr",
        code_version=code_version,
        config_hash=config_hash,
    )
    return abs_path


def read_zarr(
    *,
    country: str,
    scenario: str,
    artefact: str,
    crop_method: str,
    config: Config | None = None,
    config_hash: str = "",
) -> xr.Dataset:
    config = _resolve_config(config)
    record = _catalog(config).lookup(
        country=country,
        scenario=scenario,
        season=NOT_APPLICABLE,
        kind=artefact,
        crop_method=crop_method,
        config_hash=config_hash,
    )
    abs_path = config.cache_root / record.path
    if not abs_path.is_dir():
        raise CacheMiss(
            f"catalog entry exists but zarr store is missing on disk: {abs_path}"
        )
    return xr.open_zarr(abs_path)


# ---------------------------------------------------------------------------
# Human-readable export
# ---------------------------------------------------------------------------


def export_to_xlsx(
    country: str,
    output_path: Path | str,
    *,
    crop_method: str | None = None,
    config: Config | None = None,
) -> Path:
    """Bundle every parquet artefact for a country into one xlsx workbook.

    For human inspection only — the parquet caches remain the source of truth.
    Each catalog row that matches becomes one sheet named
    ``<scenario>__<season>__<kind>``; sheet names get truncated to Excel's
    31-character limit if necessary. ``crop_method=None`` writes one sheet
    per (kind, crop_method) so different cropping choices stay distinguishable.
    """
    config = _resolve_config(config)
    output_path = Path(output_path)
    catalog = _catalog(config)

    rows = [
        r for r in catalog.list_all()
        if r.country == country
        and r.format == "parquet"
        and (crop_method is None or r.crop_method == crop_method)
    ]
    if not rows:
        raise CacheMiss(
            f"no parquet artefacts in catalog for country={country!r} "
            f"crop_method={crop_method!r}"
        )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for r in rows:
            df = pd.read_parquet(config.cache_root / r.path)
            sheet = f"{r.scenario}__{r.season}__{r.kind}__{r.crop_method}"[:31]
            df.to_excel(writer, sheet_name=sheet)
    return output_path
