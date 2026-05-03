"""Data loaders for CMIP6, the W5E5 reference, country bboxes, and the model list.

The pipeline does no on-the-fly regridding: CMIP6 data lives on each model's
native grid, and observations are pre-upscaled to match each CMIP6 model's
grid for the per-pixel model-vs-obs metrics. ``subselect`` consumes both
as-is.

Two observation reference products are handled side-by-side:

1. **Per-CMIP6-model upscaled** (``reference_root``,
   ``Data/reference/monthly_cmip6_upscaled/<MODEL>/<file>.nc``): one file per
   CMIP6 model, on each model's native grid. Used for the per-pixel
   model-vs-obs comparison metrics — bias, RMSE, correlation, and the
   per-pixel bias_score whose spatial mean is BVS. Filenames:

   - ``tas``, ``pr``, ``tasmax`` →
     ``<var>_gswp3-w5e5_obsclim_mon_1901_2019_<MODEL>.nc``
   - ``psl`` → ``psl_w5e5_obsclim_mon_1991_2019_<MODEL>.nc``

2. **Single-grid upscaled** (``single_grid_reference_root``,
   ``Data/reference/monthly_upscaled/<file>.nc``): one file per variable on a
   common grid shared across all models. Used for the **σ_obs scalar that
   feeds the TSS denominator** — using a common grid here decouples TSS values
   from model-specific regridding variance and preserves cross-model
   comparability when the per-(variable, season) TSS values are min-max
   normalised across the 35-model ensemble. Filenames:

   - ``tas``, ``pr``, ``tasmax`` →
     ``<var>_gswp3-w5e5_obsclim_mon_1901_2019_cmip6_upscaled.nc``
   - ``psl`` → ``psl_w5e5_obsclim_mon_1991_2019_cmip6_upscaled.nc``

The ``psl`` template is the period-limiting one (1991–2019, W5E5-only); the
others span 1901–2019 from the merged GSWP3-W5E5 record. The 1995–2014
evaluation window per ``docs/historical_performance.md`` falls inside both.
"""

from __future__ import annotations

import json
from pathlib import Path

import xarray as xr

from subselect.config import Config

REFERENCE_FILENAME_TEMPLATES: dict[str, str] = {
    "tas": "tas_gswp3-w5e5_obsclim_mon_1901_2019_{model}.nc",
    "pr": "pr_gswp3-w5e5_obsclim_mon_1901_2019_{model}.nc",
    "tasmax": "tasmax_gswp3-w5e5_obsclim_mon_1901_2019_{model}.nc",
    "psl": "psl_w5e5_obsclim_mon_1991_2019_{model}.nc",
}

# Single-grid (common-grid) variant used for the σ_obs scalar in the TSS
# denominator. One file per variable, no per-model subdir. This is the
# cmip6-grid upscaled product (1.25° × 1.875°), not native W5E5; the native
# 0.5° W5E5 product is exposed separately via :func:`load_native_w5e5` for
# figure-display use (e.g. the observed-mean panel in the bias-map figure).
SINGLE_GRID_REFERENCE_FILENAME_TEMPLATES: dict[str, str] = {
    "tas": "tas_gswp3-w5e5_obsclim_mon_1901_2019_cmip6_upscaled.nc",
    "pr": "pr_gswp3-w5e5_obsclim_mon_1901_2019_cmip6_upscaled.nc",
    "tasmax": "tasmax_gswp3-w5e5_obsclim_mon_1901_2019_cmip6_upscaled.nc",
    "psl": "psl_w5e5_obsclim_mon_1991_2019_cmip6_upscaled.nc",
}

# Native 0.5° single-grid W5E5 — used for observed-mean display panels in
# the bias-map figure. Canonical location is ``Data/reference/monthly_05/``.
# ``psl`` is the W5E5 single-product (no GSWP3 merge), 1991–2019.
# ``tasmax`` has a hyphen-separated date range in its filename (ISIMIP3a
# upstream artefact — kept as-is to match the canonical filename).
NATIVE_REFERENCE_FILENAME_TEMPLATES: dict[str, str] = {
    "tas": "tas_gswp3-w5e5_obsclim_mon_1901_2019.nc",
    "pr": "pr_gswp3-w5e5_obsclim_mon_1901_2019.nc",
    "tasmax": "tasmax_gswp3-w5e5_obsclim_mon_1901-2019.nc",
    "psl": "psl_w5e5_obsclim_mon_1991_2019.nc",
}

CMIP6_DIR_TEMPLATE = "CMIP6/monthly/{variable}/{scenario}"
MODELS_CSV_NAME = "models_ordered.csv"
MODELS_PLACEHOLDER = "aaaa"


def load_models_list(config: Config | None = None) -> list[str]:
    """Return the canonical 1..35 model ordering from ``Data/models_ordered.csv``.

    The first line is an ``aaaa`` placeholder/header that is dropped here.
    The resulting 1..35 ordering is the index used throughout the project —
    every figure marker references it.
    """
    config = config or Config.from_env()
    path = config.data_root / MODELS_CSV_NAME
    with path.open() as fh:
        models = [line.strip() for line in fh if line.strip()]
    if models and models[0] == MODELS_PLACEHOLDER:
        models = models[1:]
    return models


def reference_path(variable: str, model: str, config: Config | None = None) -> Path:
    """Resolve the W5E5 reference NetCDF path for one (variable, model) pair."""
    config = config or Config.from_env()
    template = REFERENCE_FILENAME_TEMPLATES.get(variable)
    if template is None:
        raise ValueError(
            f"No reference filename template for variable={variable!r}; "
            f"known: {sorted(REFERENCE_FILENAME_TEMPLATES)}"
        )
    return config.reference_root / model / template.format(model=model)


def load_w5e5(variable: str, model: str, config: Config | None = None) -> xr.Dataset:
    """Open the per-CMIP6-model upscaled W5E5 reference dataset.

    Used for the per-pixel model-vs-obs comparison metrics (bias, RMSE,
    correlation, bias_score). For the σ_obs TSS-denominator scalar use
    :func:`load_single_grid_w5e5` instead — it returns the common-grid
    variant that decouples TSS from per-model regridding variance.
    """
    path = reference_path(variable, model, config=config)
    if not path.is_file():
        raise FileNotFoundError(f"W5E5 reference file not found: {path}")
    return xr.open_dataset(path)


def single_grid_reference_path(
    variable: str, config: Config | None = None
) -> Path:
    """Resolve the single-grid (common-grid) W5E5 reference path for one variable."""
    config = config or Config.from_env()
    template = SINGLE_GRID_REFERENCE_FILENAME_TEMPLATES.get(variable)
    if template is None:
        raise ValueError(
            f"No single-grid reference filename template for variable={variable!r}; "
            f"known: {sorted(SINGLE_GRID_REFERENCE_FILENAME_TEMPLATES)}"
        )
    return config.single_grid_reference_root / template


def load_single_grid_w5e5(
    variable: str, config: Config | None = None
) -> xr.Dataset:
    """Open the single-grid (common-grid) W5E5 reference dataset for one variable.

    Returns the **cmip6-grid upscaled** product (1.25° × 1.875°,
    ``Data/reference/monthly_upscaled/<var>_*_cmip6_upscaled.nc``). This is
    the σ_obs source for the TSS denominator. The per-CMIP6-model upscaled
    product (:func:`load_w5e5`) is the wrong source for σ_obs because each
    model's grid would yield a slightly different σ via regridding variance,
    breaking cross-model comparability when TSS values are min-max normalised
    across the 35-model ensemble.

    For the **native 0.5°** W5E5 product (used by figure-display code only —
    e.g. the observed-mean panel in the bias-map figure) call
    :func:`load_native_w5e5` instead.

    See ``documentation/methods.tex`` § Historical performance for the
    methodology entry.
    """
    path = single_grid_reference_path(variable, config=config)
    if not path.is_file():
        raise FileNotFoundError(
            f"Single-grid W5E5 reference file not found: {path}"
        )
    return xr.open_dataset(path)


def native_reference_path(
    variable: str, config: Config | None = None
) -> Path:
    """Resolve the native 0.5° W5E5 reference path for one variable.

    Canonical location: ``<data_root>/reference/monthly_05/<filename>``.
    """
    config = config or Config.from_env()
    template = NATIVE_REFERENCE_FILENAME_TEMPLATES.get(variable)
    if template is None:
        raise ValueError(
            f"No native reference filename template for variable={variable!r}; "
            f"known: {sorted(NATIVE_REFERENCE_FILENAME_TEMPLATES)}"
        )
    return config.data_root / "reference" / "monthly_05" / template


def load_native_w5e5(
    variable: str, config: Config | None = None
) -> xr.Dataset:
    """Open the native 0.5° single-grid W5E5 reference dataset for one variable.

    Returns the un-upscaled, native-grid GSWP3-W5E5 (or W5E5 for ``psl``)
    product at 0.5° × 0.5° resolution (360 × 720 global grid). Use **for
    figure display only** — the σ_obs scalar in the TSS denominator goes
    through :func:`load_single_grid_w5e5`, which returns the cmip6-grid
    upscaled variant.

    Canonical path: ``<data_root>/reference/monthly_05/<var>_*.nc``.
    """
    path = native_reference_path(variable, config=config)
    if not path.is_file():
        raise FileNotFoundError(
            f"Native W5E5 reference file not found: {path}. "
            f"Re-download from ISIMIP3a or restore from backup; the "
            f"refactor expects native 0.5° obs at <data_root>/reference/monthly_05/."
        )
    ds = xr.open_dataset(path)
    # Native ISIMIP3a files have descending latitude (90 → -90); xarray
    # ``slice`` requires monotonic ascending, so sort on read.
    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        ds = ds.sortby("lon")
    return ds


def cmip6_dir(variable: str, scenario: str, config: Config | None = None) -> Path:
    """Return the directory holding CMIP6 NetCDFs for ``(variable, scenario)``."""
    config = config or Config.from_env()
    return config.data_root / CMIP6_DIR_TEMPLATE.format(
        variable=variable, scenario=scenario
    )


def cmip6_path(
    variable: str, scenario: str, model: str, config: Config | None = None
) -> Path:
    """Resolve a single CMIP6 NetCDF for (variable, scenario, model).

    Uses a glob to discover the variant label (``r1i1p1f1``, ``r1i1p1f2``,
    ``r1i1p3f1``, etc.) embedded in filenames such as
    ``tas_GISS-E2-1-G_r1i1p3f1_mon_ssp126.nc``. Errors clearly if the
    directory is missing or the model has zero/multiple matches.
    """
    directory = cmip6_dir(variable, scenario, config=config)
    if not directory.is_dir():
        raise FileNotFoundError(f"CMIP6 directory not found: {directory}")
    matches = sorted(directory.glob(f"{variable}_{model}_*.nc"))
    if not matches:
        raise FileNotFoundError(
            f"No CMIP6 file for variable={variable!r} model={model!r} "
            f"scenario={scenario!r} under {directory}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple CMIP6 files matched variable={variable!r} model={model!r} "
            f"scenario={scenario!r}: {[p.name for p in matches]}. "
            f"Resolve by removing duplicates or refining the glob."
        )
    return matches[0]


def load_cmip6(
    variable: str, scenario: str, model: str, config: Config | None = None
) -> xr.Dataset:
    """Open a CMIP6 NetCDF on its native grid (no on-the-fly regridding)."""
    return xr.open_dataset(cmip6_path(variable, scenario, model, config=config))


def load_country_bboxes(config: Config | None = None) -> dict[str, dict]:
    """Read the bbox dictionary keyed by country code from country_codes.json."""
    config = config or Config.from_env()
    path = config.data_root / "country_codes" / "country_codes.json"
    with path.open() as fh:
        return json.load(fh)


def country_bbox(country: str, config: Config | None = None) -> dict[str, float]:
    """Return ``{lat_min, lat_max, lon_min, lon_max}`` for a country.

    Matches by ``name``, ``alpha-2``, or ``alpha-3`` (case-insensitive).
    """
    needle = country.lower()
    for entry in load_country_bboxes(config=config).values():
        for field in ("name", "alpha-2", "alpha-3"):
            value = entry.get(field, "")
            if value and value.lower() == needle:
                bbox = entry["boundingBox"]
                return {
                    "lat_min": float(bbox["sw"]["lat"]),
                    "lat_max": float(bbox["ne"]["lat"]),
                    "lon_min": float(bbox["sw"]["lon"]),
                    "lon_max": float(bbox["ne"]["lon"]),
                }
    raise ValueError(f"Country {country!r} not found in country_codes.json")
