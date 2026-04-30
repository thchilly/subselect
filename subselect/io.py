"""Data loaders for CMIP6, the W5E5 reference, country bboxes, and the model list.

The legacy paper-era pipeline did no on-the-fly regridding (verified by grep
of `legacy/cmip6-greece/GR_model_performance_HM.ipynb` during M3 prep): CMIP6
data lives on each model's native grid, observations are pre-upscaled to
match each CMIP6 model's grid, and `subselect` consumes both as-is. M7 will
verify this convention is the right one when it ports the HPS pipeline.

Reference filename templates (W5E5 from ISIMIP3a's gswp3-w5e5_obsclim product):

- ``tas``, ``pr``, ``tasmax`` → ``<var>_gswp3-w5e5_obsclim_mon_1901_2019_<MODEL>.nc``
- ``psl``                     → ``psl_w5e5_obsclim_mon_1991_2019_<MODEL>.nc``

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

CMIP6_DIR_TEMPLATE = "CMIP6/monthly/{variable}/{scenario}"
MODELS_CSV_NAME = "models_ordered.csv"
MODELS_PLACEHOLDER = "aaaa"


def load_models_list(config: Config | None = None) -> list[str]:
    """Return the canonical 1..35 model ordering from ``Data/models_ordered.csv``.

    The first line is the ``aaaa`` placeholder/header used by the paper-era code;
    it is dropped here. Athanasios is firm that the resulting 1..35 ordering is
    preserved throughout the project (every figure marker uses this index).
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
    """Open the W5E5 reference dataset for one (variable, model) pair."""
    path = reference_path(variable, model, config=config)
    if not path.is_file():
        raise FileNotFoundError(f"W5E5 reference file not found: {path}")
    return xr.open_dataset(path)


def cmip6_dir(variable: str, scenario: str, config: Config | None = None) -> Path:
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

    Matches by ``name``, ``alpha-2``, or ``alpha-3`` (case-insensitive), per the
    legacy ``extract_subset`` lookup convention.
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
