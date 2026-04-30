"""Path resolution and frozen project settings.

The `Config` dataclass holds every path and parameter that downstream modules
need to reach Data/, the GADM shapefile, the cache layer, plus the frozen
methodology constants (W5E5 reference, evaluation/future windows, HPS variable
list). It is the single replacement for the hardcoded Windows paths in the
legacy notebooks (see legacy/LEGACY.md for the originals).

Path resolution for `data_root` (highest priority first):

1. Environment variable ``SUBSELECT_DATA_ROOT``.
2. User TOML at ``~/.subselect.toml`` with a ``data_root`` key (and optionally
   ``cache_root`` / ``shapefile_path`` / ``reference_root`` /
   ``single_grid_reference_root`` / ``cmip6_metadata_root`` / ``results_root``
   overrides).
3. Repo-relative default ``<repo>/Data``, derived from the package install
   location.

Default layout (post-restructure 2026-04-30):

- ``data_root``           → ``<repo>/Data``                 (input only)
- ``cache_root``          → ``<repo>/cache``                (sibling of Data/, derived artefacts)
- ``results_root``        → ``<repo>/results``              (sibling of Data/, legacy paper outputs)
- ``shapefile_path``              → ``<data_root>/shapefiles/gadm/gadm_410-levels.gpkg``
- ``reference_root``              → ``<data_root>/reference/monthly_cmip6_upscaled``
- ``single_grid_reference_root``  → ``<data_root>/reference/monthly_upscaled``
- ``cmip6_metadata_root``         → ``<data_root>/CMIP6/metadata``

Two reference products are kept side-by-side because the paper-era HPS
pipeline uses both. ``reference_root`` (per-CMIP6-model upscaled) feeds the
per-pixel model-vs-obs metrics (bias, RMSE, correlation, bias_score) where
obs must live on each model's own grid. ``single_grid_reference_root``
(single common grid) feeds the σ_obs scalar that goes into the TSS
denominator: using a common grid here decouples TSS values from
model-specific regridding variance and preserves cross-model comparability
for downstream min-max normalisation. See ``documentation/methods.tex`` §
Historical performance for the methodology entry.

`cache_root` and `results_root` live at the repo level (siblings of Data/),
not inside it, so changing `SUBSELECT_DATA_ROOT` does not relocate the cache
or the legacy paper outputs. Override either independently in the TOML.

Reference-dataset note: ``reference_dataset = "W5E5"`` is sourced from
ISIMIP3a's ``gswp3-w5e5_obsclim`` product. The 1995–2014 evaluation window
falls entirely inside the W5E5 portion of the merged GSWP3-W5E5 record. The
``psl`` files are W5E5-only at 1991–2019; ``tas``/``pr`` files cover
1901–2019. ``subselect.io`` handles the per-variable filename templates.
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
DEFAULT_CACHE_ROOT = REPO_ROOT / "cache"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"

DEFAULT_SHAPEFILE_RELATIVE = Path("shapefiles") / "gadm" / "gadm_410-levels.gpkg"
DEFAULT_REFERENCE_RELATIVE = Path("reference") / "monthly_cmip6_upscaled"
DEFAULT_SINGLE_GRID_REFERENCE_RELATIVE = Path("reference") / "monthly_upscaled"
DEFAULT_METADATA_RELATIVE = Path("CMIP6") / "metadata"

HPS_VARIABLES: tuple[str, ...] = ("tas", "pr", "psl")
DIAGNOSTIC_VARIABLES: tuple[str, ...] = ("tasmax",)

_PATH_OVERRIDE_KEYS: tuple[str, ...] = (
    "cache_root",
    "shapefile_path",
    "reference_root",
    "single_grid_reference_root",
    "cmip6_metadata_root",
    "results_root",
)


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
    reference_root: Path
    single_grid_reference_root: Path
    cmip6_metadata_root: Path
    results_root: Path
    reference_dataset: str = "W5E5"
    eval_window: tuple[int, int] = (1995, 2014)
    future_window: tuple[int, int] = (2081, 2100)
    # 1850–1899 inclusive (50 years) — matches the paper-era spread notebook's
    # `slice("1850-01-01", "1899-12-31")` (legacy GR_model_spread.ipynb line 190).
    # IPCC AR6 conventionally uses 1850–1900 (51 years); the off-by-one is an
    # original-work artefact pinned for regression-test parity.
    pre_industrial: tuple[int, int] = (1850, 1899)
    hps_variables: tuple[str, ...] = HPS_VARIABLES
    diagnostic_variables: tuple[str, ...] = DIAGNOSTIC_VARIABLES

    @classmethod
    def from_env(cls) -> Config:
        data_root, overrides = _resolve_paths()
        return cls(
            data_root=data_root,
            cache_root=overrides.get("cache_root", DEFAULT_CACHE_ROOT),
            shapefile_path=overrides.get(
                "shapefile_path", data_root / DEFAULT_SHAPEFILE_RELATIVE
            ),
            reference_root=overrides.get(
                "reference_root", data_root / DEFAULT_REFERENCE_RELATIVE
            ),
            single_grid_reference_root=overrides.get(
                "single_grid_reference_root",
                data_root / DEFAULT_SINGLE_GRID_REFERENCE_RELATIVE,
            ),
            cmip6_metadata_root=overrides.get(
                "cmip6_metadata_root", data_root / DEFAULT_METADATA_RELATIVE
            ),
            results_root=overrides.get("results_root", DEFAULT_RESULTS_ROOT),
        )

    def with_overrides(self, **kwargs: object) -> Config:
        return replace(self, **kwargs)


def _resolve_paths() -> tuple[Path, dict[str, Path]]:
    """Resolve `data_root` and any TOML overrides for the other path fields."""
    env_value = os.environ.get(ENV_VAR_DATA_ROOT)
    if env_value:
        return _resolve_path(env_value), {}

    if USER_CONFIG_PATH.is_file():
        with USER_CONFIG_PATH.open("rb") as fh:
            toml_data = tomllib.load(fh)
        overrides: dict[str, Path] = {
            key: _resolve_path(toml_data[key])
            for key in _PATH_OVERRIDE_KEYS
            if key in toml_data
        }
        if "data_root" in toml_data:
            return _resolve_path(toml_data["data_root"]), overrides
        # TOML present but no data_root key: fall through to the repo default,
        # but still honour the other overrides if they were specified.
        return DEFAULT_DATA_ROOT, overrides

    return DEFAULT_DATA_ROOT, {}


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()
