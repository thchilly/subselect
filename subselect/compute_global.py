"""Global L1 artefacts — country-independent precomputes shared across runs.

Implements the *compute-once-globally, reuse-per-country* convention. Each
artefact in this module depends only on (model, variable, scenario) and not
on country, so a single global cache serves every per-country compute that
follows.

Layout under ``cache/_global/``:

- ``zarr/historical_clim__<variable>__<model>.zarr`` —
  CMIP6 monthly climatology over the evaluation window (1995–2014) on the
  model's native grid. Inputs to the per-country HPS metric, monthly means,
  and bias-map pipelines.
- ``zarr/obs_clim__<variable>__<model>.zarr`` —
  W5E5 monthly climatology upscaled to the same model grid; pairs with the
  historical climatology for the per-pixel model-vs-obs metrics.
- ``zarr/sigma_ref__<variable>__<model>.zarr`` —
  per-pixel interannual σ map per period (annual + four seasons) on the
  model grid; feeds the bias_score formula.
- ``zarr/eoc_clim__<variable>__<model>__<scenario>.zarr`` —
  CMIP6 monthly climatology over the future window (2081–2100) on the
  model's native grid; one per scenario. Inputs to the spread pipeline.
- ``zarr/pi_clim__<variable>__<model>.zarr`` —
  CMIP6 monthly climatology over the pre-industrial window (1850–1899) on
  the model's native grid. Sourced from the historical+ssp585 file (the
  pre-2015 portion is identical across scenarios for the same model).
- ``zarr/annual_field__<variable>__<model>__<scenario>.zarr`` —
  per-pixel annual mean 1850–2100 (251 years × native grid). Per-country
  ``annual_timeseries`` becomes a small crop + spatial-mean of this field.
- ``zarr/native_sigma_obs__<variable>.zarr`` —
  native 0.5° W5E5 monthly climatology + per-period interannual σ maps;
  feeds the σ_obs scalar in the TSS denominator.

The :func:`compute_global` orchestrator builds every artefact in parallel
(joblib loky over (model, variable) and (variable, scenario) jobs).
Idempotent: per-artefact cache hits skip recomputation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from subselect import io
from subselect.cache import Cache
from subselect.config import Config


SCENARIOS: tuple[str, ...] = ("ssp126", "ssp245", "ssp370", "ssp585")
HPS_VARIABLES: tuple[str, ...] = ("tas", "pr", "psl")
ALL_VARIABLES: tuple[str, ...] = HPS_VARIABLES + ("tasmax",)
TIMESERIES_VARIABLES: tuple[str, ...] = ("tas", "pr", "psl")
DEFAULT_N_JOBS = -1
DEFAULT_BACKEND = "loky"


# ---------------------------------------------------------------------------
# Per-(model, variable) climatologies — historical + EoC + PI + obs + sigma_ref
# ---------------------------------------------------------------------------

def _build_one_model_variable_climatologies(
    model: str, variable: str, config: Config,
) -> dict[str, xr.DataArray] | None:
    """Compute every (model, variable) climatology in one NetCDF read pair.

    Returns a dict with keys ``hist_clim``, ``obs_clim``, ``sigma_ref`` (period
    × lat × lon), ``pi_clim``, plus per-scenario ``eoc_clim__<ssp>``. Returns
    ``None`` if the source files for this (model, variable) are missing.
    """
    from subselect.performance import (
        PERIODS, SEASON_MONTHS,
        _interannual_sigma_map, _monthly_climatology,
        _normalise_time_to_first_of_month, _slice_eval_window,
    )

    try:
        ssp585_ds = io.load_cmip6(variable, "ssp585", model, config=config)
    except FileNotFoundError:
        return None
    if "height" in ssp585_ds.coords:
        ssp585_ds = ssp585_ds.drop_vars("height")
    cmip6_var = ssp585_ds[variable]
    cmip6_var = _normalise_time_to_first_of_month(cmip6_var)

    # Historical eval-window climatology on the model's native grid
    hist_var = _slice_eval_window(cmip6_var, config.eval_window)
    hist_clim = _monthly_climatology(hist_var).compute()

    # Pre-industrial climatology — same NetCDF (historical+ssp585), 1850–1899
    pi_var = cmip6_var.sel(
        time=slice(f"{config.pre_industrial[0]}-01-01", f"{config.pre_industrial[1]}-12-31")
    )
    pi_clim = _monthly_climatology(pi_var).compute() if pi_var.sizes.get("time", 0) > 0 else None

    # End-of-century climatology — ssp585 directly
    eoc_clim_ssp585 = _monthly_climatology(
        cmip6_var.sel(
            time=slice(f"{config.future_window[0]}-01-01", f"{config.future_window[1]}-12-31")
        )
    ).compute()

    # Observation climatology + sigma_ref (upscaled to model grid)
    try:
        obs_var = io.load_w5e5(variable, model, config=config)[variable]
    except FileNotFoundError:
        ssp585_ds.close()
        return None
    obs_var = _normalise_time_to_first_of_month(obs_var)
    obs_var = _slice_eval_window(obs_var, config.eval_window)
    obs_clim = _monthly_climatology(obs_var).compute()
    sigma_per_period = []
    for period in PERIODS:
        sm = _interannual_sigma_map(obs_var, SEASON_MONTHS[period]).compute()
        sigma_per_period.append(sm.expand_dims({"period": [period]}))
    sigma_ref = xr.concat(sigma_per_period, dim="period")

    ssp585_ds.close()
    out = {
        "hist_clim": hist_clim,
        "obs_clim": obs_clim,
        "sigma_ref": sigma_ref,
        "eoc_clim__ssp585": eoc_clim_ssp585,
    }
    if pi_clim is not None:
        out["pi_clim"] = pi_clim
    return out


def _build_one_eoc_climatology(
    model: str, variable: str, scenario: str, config: Config,
) -> xr.DataArray | None:
    """End-of-century climatology for one (model, variable, scenario)."""
    from subselect.performance import (
        _monthly_climatology, _normalise_time_to_first_of_month,
    )
    try:
        ds = io.load_cmip6(variable, scenario, model, config=config)
    except FileNotFoundError:
        return None
    if "height" in ds.coords:
        ds = ds.drop_vars("height")
    da = _normalise_time_to_first_of_month(ds[variable])
    eoc_da = da.sel(
        time=slice(f"{config.future_window[0]}-01-01", f"{config.future_window[1]}-12-31")
    )
    if eoc_da.sizes.get("time", 0) == 0:
        ds.close()
        return None
    out = _monthly_climatology(eoc_da).compute()
    ds.close()
    return out


def _build_one_annual_field(
    model: str, variable: str, scenario: str, config: Config,
) -> xr.DataArray | None:
    """Per-pixel annual mean 1850–2100 for one (model, variable, scenario).

    Returned as a DataArray with dims (year, lat, lon) on the model's
    native grid. Per-country annual time-series consume this via crop +
    spatial-weighted mean — much faster than re-opening the NetCDF.
    """
    from subselect.performance import _normalise_time_to_first_of_month

    try:
        ds = io.load_cmip6(variable, scenario, model, config=config)
    except FileNotFoundError:
        return None
    if "height" in ds.coords:
        ds = ds.drop_vars("height")
    da = _normalise_time_to_first_of_month(ds[variable])
    annual = da.groupby("time.year").mean("time").compute()
    ds.close()
    return annual


# ---------------------------------------------------------------------------
# Native obs (σ_obs source)
# ---------------------------------------------------------------------------

def _build_native_obs_climatology(variable: str, config: Config) -> xr.Dataset | None:
    """Native 0.5° W5E5 climatology + per-period σ maps for σ_obs.

    Returned as a single xr.Dataset with two data variables: ``clim``
    (12-month means) and ``sigma`` (per-period interannual σ at native
    resolution). Per-country σ_obs scalars come from ``sigma`` after
    cos(lat)-weighted spatial mean over the country crop.
    """
    from subselect.performance import (
        PERIODS, SEASON_MONTHS,
        _interannual_sigma_map, _monthly_climatology,
        _normalise_time_to_first_of_month, _slice_eval_window,
    )
    try:
        obs_var = io.load_native_w5e5(variable, config=config)[variable]
    except FileNotFoundError:
        return None
    obs_var = _normalise_time_to_first_of_month(obs_var)
    obs_var = _slice_eval_window(obs_var, config.eval_window)
    obs_clim = _monthly_climatology(obs_var).compute()
    sigma_per_period = []
    for period in PERIODS:
        sm = _interannual_sigma_map(obs_var, SEASON_MONTHS[period]).compute()
        sigma_per_period.append(sm.expand_dims({"period": [period]}))
    sigma = xr.concat(sigma_per_period, dim="period")
    return xr.Dataset({"clim": obs_clim, "sigma": sigma})


# ---------------------------------------------------------------------------
# Cache key conventions
#
# Each helper returns the catalog key under which the corresponding global
# artefact is stored. Centralising the naming here keeps producer and
# consumer (per-country builders in :mod:`subselect.compute`) in sync.
# ---------------------------------------------------------------------------

def hist_clim_key(variable: str, model: str) -> str:
    """Catalog key for the historical (1995–2014) monthly climatology."""
    return f"historical_clim__{variable}__{model}"


def obs_clim_key(variable: str, model: str) -> str:
    """Catalog key for the observed monthly climatology on the model's grid."""
    return f"obs_clim__{variable}__{model}"


def sigma_ref_key(variable: str, model: str) -> str:
    """Catalog key for the per-period σ_ref interannual map."""
    return f"sigma_ref__{variable}__{model}"


def eoc_clim_key(variable: str, model: str, scenario: str) -> str:
    """Catalog key for the end-of-century (2081–2100) monthly climatology."""
    return f"eoc_clim__{variable}__{model}__{scenario}"


def pi_clim_key(variable: str, model: str) -> str:
    """Catalog key for the pre-industrial (1850–1899) monthly climatology."""
    return f"pi_clim__{variable}__{model}"


def annual_field_key(variable: str, model: str, scenario: str) -> str:
    """Catalog key for the per-(model, scenario) annual-mean field 1850–2100."""
    return f"annual_field__{variable}__{model}__{scenario}"


def native_sigma_obs_key(variable: str) -> str:
    """Catalog key for the native 0.5° W5E5 σ field for one variable."""
    return f"native_sigma_obs__{variable}"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _cmip6_dep(variable: str, scenario: str, model: str, config: Config) -> list[Path]:
    """Cache-staleness dependency: the CMIP6 NetCDF for this (var, ssp, model)."""
    try:
        return [io.cmip6_path(variable, scenario, model, config=config)]
    except FileNotFoundError:
        return []


def _native_obs_dep(variable: str, config: Config) -> list[Path]:
    try:
        return [io.native_reference_path(variable, config=config)]
    except FileNotFoundError:
        return []


def _upscaled_obs_dep(variable: str, model: str, config: Config) -> list[Path]:
    try:
        return [io.reference_path(variable, model, config=config)]
    except (FileNotFoundError, AttributeError):
        return []


def compute_global(
    *,
    config: Config | None = None,
    n_jobs: int = DEFAULT_N_JOBS,
    variables: Iterable[str] = ALL_VARIABLES,
    scenarios: Iterable[str] = SCENARIOS,
    timeseries_variables: Iterable[str] = TIMESERIES_VARIABLES,
    force: bool = False,
) -> Cache:
    """Populate the country-independent global cache.

    Builds every per-(model, variable, scenario) climatology + per-pixel
    annual-mean field + native-resolution observation climatology that
    downstream country runs consume. Idempotent: per-artefact cache hits
    skip recomputation; ``force=True`` rebuilds everything.

    Returns the global :class:`Cache` instance for inspection.
    """
    config = config or Config.from_env()
    cache = Cache.global_cache(config.cache_root)
    if force:
        cache.clear()

    variables = list(variables)
    scenarios = list(scenarios)
    timeseries_variables = list(timeseries_variables)
    models = io.load_models_list(config)

    parallel = Parallel(n_jobs=n_jobs, backend=DEFAULT_BACKEND)

    # ----- per-(model, variable) historical + obs + sigma_ref + pi + eoc(ssp585) -----
    pending_mv: list[tuple[str, str]] = []
    for variable in variables:
        for model in models:
            keys = [
                hist_clim_key(variable, model),
                obs_clim_key(variable, model),
                sigma_ref_key(variable, model),
                pi_clim_key(variable, model),
                eoc_clim_key(variable, model, "ssp585"),
            ]
            deps = (
                _cmip6_dep(variable, "ssp585", model, config)
                + _upscaled_obs_dep(variable, model, config)
            )
            if all(cache.is_fresh(k, deps) for k in keys):
                continue
            pending_mv.append((variable, model))

    if pending_mv:
        results = parallel(
            delayed(_build_one_model_variable_climatologies)(m, v, config)
            for v, m in pending_mv
        )
        for (variable, model), r in zip(pending_mv, results):
            if r is None:
                continue
            deps = (
                _cmip6_dep(variable, "ssp585", model, config)
                + _upscaled_obs_dep(variable, model, config)
            )
            cache.save(hist_clim_key(variable, model), r["hist_clim"], deps=deps)
            cache.save(obs_clim_key(variable, model), r["obs_clim"], deps=deps)
            cache.save(sigma_ref_key(variable, model), r["sigma_ref"], deps=deps)
            cache.save(eoc_clim_key(variable, model, "ssp585"), r["eoc_clim__ssp585"], deps=deps)
            if "pi_clim" in r:
                cache.save(pi_clim_key(variable, model), r["pi_clim"], deps=deps)

    # ----- per-(model, variable, scenario) EoC climatology for non-ssp585 scenarios -----
    pending_eoc: list[tuple[str, str, str]] = []
    for variable in variables:
        for model in models:
            for scenario in scenarios:
                if scenario == "ssp585":
                    continue
                key = eoc_clim_key(variable, model, scenario)
                deps = _cmip6_dep(variable, scenario, model, config)
                if cache.is_fresh(key, deps):
                    continue
                pending_eoc.append((variable, model, scenario))

    if pending_eoc:
        results = parallel(
            delayed(_build_one_eoc_climatology)(m, v, s, config)
            for v, m, s in pending_eoc
        )
        for (variable, model, scenario), r in zip(pending_eoc, results):
            if r is None:
                continue
            deps = _cmip6_dep(variable, scenario, model, config)
            cache.save(eoc_clim_key(variable, model, scenario), r, deps=deps)

    # ----- per-(model, variable, scenario) annual-mean fields -----
    # Each annual-mean field requires reading the FULL 1850–2100 NetCDF (50–
    # 300 MB) and computing a per-pixel annual mean, so the per-worker memory
    # footprint is large. We process in small batches to keep peak memory
    # bounded and stream saves as each batch returns, instead of holding all
    # 420 results in memory at once.
    pending_ann: list[tuple[str, str, str]] = []
    for variable in timeseries_variables:
        for model in models:
            for scenario in scenarios:
                key = annual_field_key(variable, model, scenario)
                deps = _cmip6_dep(variable, scenario, model, config)
                if cache.is_fresh(key, deps):
                    continue
                pending_ann.append((variable, model, scenario))

    if pending_ann:
        annual_n_jobs = min(8, n_jobs) if n_jobs > 0 else 8
        annual_parallel = Parallel(n_jobs=annual_n_jobs, backend=DEFAULT_BACKEND)
        batch_size = annual_n_jobs * 2
        for batch_start in range(0, len(pending_ann), batch_size):
            batch = pending_ann[batch_start:batch_start + batch_size]
            results = annual_parallel(
                delayed(_build_one_annual_field)(m, v, s, config)
                for v, m, s in batch
            )
            for (variable, model, scenario), r in zip(batch, results):
                if r is None:
                    continue
                deps = _cmip6_dep(variable, scenario, model, config)
                cache.save(annual_field_key(variable, model, scenario), r, deps=deps)
                del r

    # ----- native-resolution observation climatology + σ maps -----
    for variable in variables:
        key = native_sigma_obs_key(variable)
        deps = _native_obs_dep(variable, config)
        if cache.is_fresh(key, deps):
            continue
        ds = _build_native_obs_climatology(variable, config)
        if ds is None:
            continue
        cache.save(key, ds, deps=deps)

    return cache
