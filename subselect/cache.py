"""Per-country, per-artefact cache for the L1 pipeline.

Each artefact (DataFrame, dict-of-DataFrame, xarray Dataset / DataArray) lives
in its own file under ``cache/<country>/``. A small JSON catalog
(``cache/<country>/catalog.json``) records what is cached and the maximum
input-file mtime at write time, so a stale upstream file invalidates only the
artefacts that consumed it.

Layout::

    cache/<country>/
        catalog.json
        parquet/
            <artefact_key>.parquet
            <artefact_key>/<sub_key>.parquet      # for dict[str, DataFrame]
            <artefact_key>__scalars.json          # for ProfileSignals scalars
        zarr/
            <artefact_key>.zarr
            <artefact_key>/<sub_key>.zarr         # for dict-of-arrays

The cache is a pure I/O layer: it does not know what an artefact *is*, only
how to round-trip it. The :class:`Cache` API exposes ``has``, ``is_fresh``,
``save``, ``load``, ``invalidate``, ``clear``.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import xarray as xr


CATALOG_FILENAME = "catalog.json"
GLOBAL_SCOPE = "_global"


class Cache:
    """Artefact cache rooted at ``cache_root / scope``.

    ``scope`` is either a country name (e.g. ``"greece"``) for per-country
    caches or the literal :data:`GLOBAL_SCOPE` (``"_global"``) for the
    country-independent cache shared across runs. Per-(model, var) historical
    climatologies, σ_obs grids, GWL crossing years and similar artefacts live
    in the global cache; country-mean reductions live in the per-country one.
    """

    def __init__(self, scope: str, cache_root: Path):
        self.scope = scope
        self.country = scope  # backwards-compatible alias for callers that named it
        self.root = Path(cache_root) / scope
        self.root.mkdir(parents=True, exist_ok=True)
        self._catalog_path = self.root / CATALOG_FILENAME
        self._catalog = self._load_catalog()

    @classmethod
    def global_cache(cls, cache_root: Path) -> "Cache":
        """Return the country-independent global-scope cache."""
        return cls(GLOBAL_SCOPE, cache_root)

    # -- catalog --------------------------------------------------------

    def _load_catalog(self) -> dict[str, dict[str, Any]]:
        if not self._catalog_path.is_file():
            return {}
        try:
            return json.loads(self._catalog_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_catalog(self) -> None:
        self._catalog_path.write_text(json.dumps(self._catalog, indent=2, sort_keys=True))

    def _max_dep_mtime(self, deps: Iterable[Path]) -> float:
        mtimes = [Path(p).stat().st_mtime for p in deps if Path(p).exists()]
        return max(mtimes) if mtimes else 0.0

    def has(self, key: str) -> bool:
        """Return ``True`` if the catalog records an artefact for ``key``."""
        return key in self._catalog

    def is_fresh(self, key: str, deps: Iterable[Path]) -> bool:
        """``True`` if the artefact exists, all its on-disk files are present,
        and no dependency mtime exceeds the recorded ``input_mtime``."""
        if key not in self._catalog:
            return False
        entry = self._catalog[key]
        path = self.root / entry["path"]
        if not path.exists():
            return False
        current = self._max_dep_mtime(deps)
        recorded = float(entry.get("input_mtime", 0.0))
        return current <= recorded + 1e-6  # tolerate fp jitter

    def invalidate(self, key: str) -> None:
        """Drop ``key`` from the catalog and remove its on-disk artefact."""
        entry = self._catalog.pop(key, None)
        if entry is None:
            self._save_catalog()
            return
        path = self.root / entry["path"]
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink(missing_ok=True)
        scalars = self.root / "parquet" / f"{key}__scalars.json"
        scalars.unlink(missing_ok=True)
        self._save_catalog()

    def clear(self) -> None:
        """Invalidate every artefact in the catalog and prune empty subdirs."""
        for key in list(self._catalog):
            self.invalidate(key)
        # remove any orphaned directories
        for sub in ("parquet", "zarr"):
            sub_dir = self.root / sub
            if sub_dir.is_dir() and not any(sub_dir.iterdir()):
                sub_dir.rmdir()

    # -- save -----------------------------------------------------------

    def save(
        self,
        key: str,
        value: Any,
        *,
        deps: Iterable[Path] = (),
        kind: str | None = None,
    ) -> None:
        """Persist *value* under *key*; record max(dep mtimes) for staleness checks.

        Supported value types:

        - :class:`pandas.DataFrame` / :class:`pandas.Series` → parquet
        - ``dict[str, pandas.DataFrame]`` → folder of parquets
        - ``dict[str, dict[str, pandas.DataFrame]]`` → nested folders (depth 2)
        - :class:`xarray.Dataset` / :class:`xarray.DataArray` → zarr
        - ``dict[str, xarray.Dataset]`` (or DataArray) → folder of zarrs
        - dataclass with pandas / scalar fields → folder of parquets + scalars.json
        """
        deps = list(deps)
        max_mtime = self._max_dep_mtime(deps)
        kind = kind or _infer_kind(value)
        path = self._write(key, value, kind)
        self._catalog[key] = {
            "path": str(path.relative_to(self.root)),
            "kind": kind,
            "computed_at": time.time(),
            "input_mtime": max_mtime,
            "deps": [str(p) for p in deps],
        }
        self._save_catalog()

    def _write(self, key: str, value: Any, kind: str) -> Path:
        parquet_dir = self.root / "parquet"
        zarr_dir = self.root / "zarr"

        if kind == "dataframe":
            parquet_dir.mkdir(parents=True, exist_ok=True)
            path = parquet_dir / f"{key}.parquet"
            _df_to_parquet(value, path)
            return path

        if kind == "series":
            parquet_dir.mkdir(parents=True, exist_ok=True)
            path = parquet_dir / f"{key}.parquet"
            _df_to_parquet(value.to_frame(name=value.name or key), path)
            return path

        if kind == "dict_of_dataframe":
            parquet_dir.mkdir(parents=True, exist_ok=True)
            sub = parquet_dir / key
            shutil.rmtree(sub, ignore_errors=True)
            sub.mkdir(parents=True)
            for sub_key, df in value.items():
                _df_to_parquet(df, sub / f"{sub_key}.parquet")
            return sub

        if kind == "nested_dict_of_dataframe":
            parquet_dir.mkdir(parents=True, exist_ok=True)
            sub = parquet_dir / key
            shutil.rmtree(sub, ignore_errors=True)
            sub.mkdir(parents=True)
            for outer, inner in value.items():
                inner_dir = sub / outer
                inner_dir.mkdir()
                for sub_key, df in inner.items():
                    _df_to_parquet(df, inner_dir / f"{sub_key}.parquet")
            return sub

        if kind == "dataset":
            zarr_dir.mkdir(parents=True, exist_ok=True)
            path = zarr_dir / f"{key}.zarr"
            _purge_path(path)
            _sanitize_attrs(value).to_zarr(path, mode="w-", consolidated=False)
            return path

        if kind == "dataarray":
            zarr_dir.mkdir(parents=True, exist_ok=True)
            path = zarr_dir / f"{key}.zarr"
            _purge_path(path)
            name = value.name or key
            ds = value.rename(name).to_dataset()
            _sanitize_attrs(ds).to_zarr(path, mode="w-", consolidated=False)
            return path

        if kind == "dict_of_dataset":
            zarr_dir.mkdir(parents=True, exist_ok=True)
            sub = zarr_dir / key
            _purge_path(sub)
            sub.mkdir(parents=True)
            for sub_key, ds in value.items():
                _safe = sub_key.replace("/", "_")
                _sanitize_attrs(ds).to_zarr(
                    sub / f"{_safe}.zarr", mode="w-", consolidated=False,
                )
            return sub

        if kind == "dataclass":
            parquet_dir.mkdir(parents=True, exist_ok=True)
            sub = parquet_dir / key
            shutil.rmtree(sub, ignore_errors=True)
            sub.mkdir(parents=True)
            scalars: dict[str, Any] = {}
            for f in fields(value):
                attr = getattr(value, f.name)
                if isinstance(attr, pd.DataFrame):
                    _df_to_parquet(attr, sub / f"{f.name}.parquet")
                elif isinstance(attr, pd.Series):
                    _df_to_parquet(
                        attr.to_frame(name=attr.name or f.name),
                        sub / f"{f.name}.parquet",
                    )
                else:
                    scalars[f.name] = attr
            (sub / "scalars.json").write_text(json.dumps(scalars))
            (sub / "_dataclass.json").write_text(
                json.dumps({"qualname": f"{type(value).__module__}.{type(value).__name__}"})
            )
            return sub

        raise ValueError(f"unsupported cache kind: {kind!r}")

    # -- load -----------------------------------------------------------

    def load(self, key: str) -> Any:
        """Load and return the artefact stored under ``key``.

        Raises :class:`KeyError` if the artefact is not in the catalog. The
        return type depends on the recorded ``kind`` (DataFrame, Series,
        Dataset, DataArray, or a nested dict of these).
        """
        if key not in self._catalog:
            raise KeyError(f"{key!r} not in cache catalog")
        entry = self._catalog[key]
        kind = entry["kind"]
        path = self.root / entry["path"]

        if kind == "dataframe":
            return pd.read_parquet(path)

        if kind == "series":
            df = pd.read_parquet(path)
            return df.iloc[:, 0]

        if kind == "dict_of_dataframe":
            return {p.stem: pd.read_parquet(p) for p in sorted(_iter_parquets(path))}

        if kind == "nested_dict_of_dataframe":
            return {
                outer.name: {
                    inner.stem: pd.read_parquet(inner)
                    for inner in sorted(_iter_parquets(outer))
                }
                for outer in sorted(path.iterdir())
                if outer.is_dir() and not outer.name.startswith("._")
            }

        if kind == "dataset":
            return xr.open_zarr(path).load()

        if kind == "dataarray":
            ds = xr.open_zarr(path).load()
            return ds[list(ds.data_vars)[0]]

        if kind == "dict_of_dataset":
            return {
                p.stem: xr.open_zarr(p).load()
                for p in sorted(_iter_zarrs(path))
            }

        if kind == "dataclass":
            qualname = json.loads((path / "_dataclass.json").read_text())["qualname"]
            cls = _import_qualname(qualname)
            scalars = json.loads((path / "scalars.json").read_text())
            kwargs: dict[str, Any] = {}
            for f in fields(cls):
                if f.name in scalars:
                    kwargs[f.name] = scalars[f.name]
                    continue
                file = path / f"{f.name}.parquet"
                if not file.exists():
                    continue
                df = pd.read_parquet(file)
                kwargs[f.name] = df.iloc[:, 0] if f.type is pd.Series else df
            return cls(**kwargs)

        raise ValueError(f"unsupported cache kind: {kind!r}")


def _purge_path(path: Path) -> None:
    """Delete a file or directory, including any macOS Apple Double sidecars
    (``._foo`` extended-attribute files) that ``shutil.rmtree`` can race with
    on volumes that synthesize them."""
    if not path.exists():
        # Apple Double sidecars can outlive their parent on these volumes;
        # explicitly remove the sibling sidecar of `path` if any.
        sidecar = path.parent / f"._{path.name}"
        if sidecar.exists():
            try:
                sidecar.unlink()
            except OSError:
                pass
        return
    if path.is_dir():
        # Walk + unlink Apple Double sidecars first so rmtree doesn't trip on
        # them mid-walk.
        for ad in path.rglob("._*"):
            try:
                ad.unlink()
            except OSError:
                pass
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except OSError:
            pass


def _df_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet (parquet requires string column names)."""
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    out.to_parquet(path)


def _sanitize_attrs(ds_or_da):
    """Strip / repair non-UTF-8 attribute values before zarr serialization.

    CMIP6 NetCDFs occasionally carry attributes encoded in Latin-1 (e.g. a
    bare ``°`` byte ``0xb0`` in a units string), which numcodecs' UTF-8
    encoder rejects. Rather than dropping the field, we re-decode bytes via
    Latin-1 then re-encode as UTF-8.
    """
    import xarray as xr

    if isinstance(ds_or_da, xr.Dataset):
        out = ds_or_da.copy()
        out.attrs = _clean_attr_dict(out.attrs)
        for var in list(out.variables):
            out[var].attrs = _clean_attr_dict(out[var].attrs)
        return out
    out = ds_or_da.copy()
    out.attrs = _clean_attr_dict(out.attrs)
    return out


def _clean_attr_dict(attrs: dict[str, Any]) -> dict[str, Any]:
    return {k: _clean_attr_value(v) for k, v in attrs.items()}


def _clean_attr_value(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("latin-1")
    if isinstance(value, str):
        # Encode to bytes then decode strictly to UTF-8; if it fails, treat
        # the original string as a Latin-1 byte sequence.
        try:
            value.encode("utf-8")
            return value
        except UnicodeEncodeError:
            return value.encode("latin-1", errors="replace").decode("utf-8", errors="replace")
    return value


def _iter_parquets(directory: Path):
    """Yield real parquet files under *directory*, skipping macOS Apple Double
    sidecars (``._foo``) that appear on filesystems mounted with extended-
    attribute synthesis."""
    return (p for p in directory.glob("*.parquet") if not p.name.startswith("._"))


def _iter_zarrs(directory: Path):
    """Yield real zarr stores under *directory*, skipping macOS Apple Double
    sidecars."""
    return (p for p in directory.glob("*.zarr") if not p.name.startswith("._"))


def _import_qualname(qualname: str) -> type:
    module_name, _, name = qualname.rpartition(".")
    import importlib
    return getattr(importlib.import_module(module_name), name)


def _infer_kind(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        return "dataframe"
    if isinstance(value, pd.Series):
        return "series"
    if isinstance(value, xr.Dataset):
        return "dataset"
    if isinstance(value, xr.DataArray):
        return "dataarray"
    if is_dataclass(value):
        return "dataclass"
    if isinstance(value, dict):
        if not value:
            return "dict_of_dataframe"
        first = next(iter(value.values()))
        if isinstance(first, pd.DataFrame):
            return "dict_of_dataframe"
        if isinstance(first, dict) and first and isinstance(next(iter(first.values())), pd.DataFrame):
            return "nested_dict_of_dataframe"
        if isinstance(first, (xr.Dataset, xr.DataArray)):
            return "dict_of_dataset"
    raise TypeError(f"cannot infer cache kind for {type(value).__name__}")
