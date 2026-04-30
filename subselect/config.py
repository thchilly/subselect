"""Path resolution and frozen project settings.

The `Config` dataclass holds every path and parameter that downstream modules
need to reach Data/, the GADM shapefile, the cache layer, plus the frozen
methodology constants (W5E5 reference, evaluation/future windows, HPS variable
list). It is the single replacement for the hardcoded Windows paths in the
legacy notebooks (see legacy/LEGACY.md for the originals).

Path resolution for `data_root` (highest priority first):

1. Environment variable ``SUBSELECT_DATA_ROOT``.
2. User TOML at ``~/.subselect.toml`` with a ``data_root`` key (and optionally
   ``cache_root`` / ``shapefile_path`` overrides).
3. Repo-relative default ``<repo>/Data``, derived from the package install
   location.

`cache_root` and `shapefile_path` default to ``data_root / "cache"`` and
``data_root / "shapefiles/gadm/gadm_410-levels.gpkg"`` per the layout in
CLAUDE.md, but can be overridden independently in the TOML.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

ENV_VAR_DATA_ROOT = "SUBSELECT_DATA_ROOT"
USER_CONFIG_PATH = Path.home() / ".subselect.toml"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "Data"

DEFAULT_SHAPEFILE_RELATIVE = Path("shapefiles") / "gadm" / "gadm_410-levels.gpkg"
DEFAULT_CACHE_RELATIVE = Path("cache")

HPS_VARIABLES: tuple[str, ...] = ("tas", "pr", "psl")
DIAGNOSTIC_VARIABLES: tuple[str, ...] = ("tasmax",)


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable bundle of paths and frozen methodology parameters.

    Construct via :meth:`from_env` for the standard resolution chain. Use
    :meth:`with_overrides` for tests or one-off scripts that need a tweaked
    copy without mutating the original.
    """

    data_root: Path
    cache_root: Path
    shapefile_path: Path
    reference_dataset: str = "W5E5"
    eval_window: tuple[int, int] = (1995, 2014)
    future_window: tuple[int, int] = (2081, 2100)
    pre_industrial: tuple[int, int] = (1850, 1900)
    hps_variables: tuple[str, ...] = HPS_VARIABLES
    diagnostic_variables: tuple[str, ...] = DIAGNOSTIC_VARIABLES

    @classmethod
    def from_env(cls) -> Config:
        data_root, overrides = _resolve_paths()
        return cls(
            data_root=data_root,
            cache_root=overrides.get("cache_root", data_root / DEFAULT_CACHE_RELATIVE),
            shapefile_path=overrides.get(
                "shapefile_path", data_root / DEFAULT_SHAPEFILE_RELATIVE
            ),
        )

    def with_overrides(self, **kwargs: object) -> Config:
        return replace(self, **kwargs)


def _resolve_paths() -> tuple[Path, dict[str, Path]]:
    """Resolve `data_root` and any TOML overrides for cache / shapefile paths."""
    env_value = os.environ.get(ENV_VAR_DATA_ROOT)
    if env_value:
        return _resolve_path(env_value), {}

    if USER_CONFIG_PATH.is_file():
        with USER_CONFIG_PATH.open("rb") as fh:
            toml_data = tomllib.load(fh)
        overrides: dict[str, Path] = {}
        for key in ("cache_root", "shapefile_path"):
            if key in toml_data:
                overrides[key] = _resolve_path(toml_data[key])
        if "data_root" in toml_data:
            return _resolve_path(toml_data["data_root"]), overrides
        # TOML present but no data_root key: fall through to the repo default,
        # but still honour cache/shapefile overrides if they were specified.
        return DEFAULT_DATA_ROOT, overrides

    return DEFAULT_DATA_ROOT, {}


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()
