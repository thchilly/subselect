"""Future-response change signals, spread quadrants, country-profile artefacts.

Ports the paper-era spread pipeline from
``legacy/cmip6-greece/GR_model_spread.ipynb`` and adds the
country-profile / GWL-crossings layer the framework needs for the M9 viz
helpers (per ``docs/refactor.md`` § Country-profile / context outputs).

Three families of outputs:

1. **Change signals** ``compute_change_signals(country, scenario)`` →
   per-(model, variable, period) deltas of (2081–2100) − (1850–1899)
   spatially-averaged means. Mirrors
   ``results/<country>/assess_long_term_change_spread_<country>.xlsx``.
   Regression-pinned.
2. **Spread quadrants** ``compute_spread_quadrants(country, scenario)`` →
   discrete labels (``warm_wet`` / ``warm_dry`` / ``cool_wet`` /
   ``cool_dry``) per (model, period). Cutpoints are the seasonal medians
   of Δtas and Δpr across the 35 models. Regression-pinned.
3. **Country-profile artefacts** for the M9 figures:
   ``compute_country_timeseries(country, variable, scenario)`` →
   long-form (model, scenario, year, value) annual country-mean values
   1950–2100; ``compute_gwl_crossing_years()`` →
   (model, scenario, gwl_threshold, crossing_year) per the ETCCDI / IPCC
   AR6 smoothed-global-mean-tas convention. The first lives at
   ``cache/parquet/<country>/timeseries/...``, the second at
   ``cache/parquet/_global/gwl_crossing_years.parquet``.

Paper-era parity notes pinned by the regression contract:

- Spread pipeline uses **box_offset=1.5°** (legacy default for
  ``calculate_spatial_average``), inconsistent with the HPS pipeline's
  1.0° offset. Documented and matched.
- Pre-industrial window is **1850–1899 inclusive (50 years)**, not the
  IPCC AR6 conventional 1850–1900 (51 years). Documented in
  ``Config.pre_industrial`` and the regression test relies on this.
- Spatial mean is taken **before** period aggregation (the legacy order:
  groupby month → spatial mean → mean over months within season). When
  bbox cropping leaves no NaNs, this is mathematically identical to the
  alternative order; matches anyway for parity.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from subselect import geom, io
from subselect.config import Config
from subselect.geom import CropMethod
from subselect.performance import (
    DEFAULT_BACKEND,
    DEFAULT_N_JOBS,
    PERIODS,
    SEASON_MONTHS,
    _normalise_time_to_first_of_month,
    _select_months,
    _slice_eval_window,
)

# All variables tracked in the spread output xlsx.
SPREAD_VARIABLES: tuple[str, ...] = ("tas", "pr", "psl", "tasmax")

# Spread pipeline uses 1.5° (legacy default for calculate_spatial_average),
# different from the HPS pipeline's 1.0° (legacy extract_subset default).
SPREAD_BOX_OFFSET = 1.5

# Quadrant labels for the (Δtas, Δpr) spread plot per ``docs/future_spread.md``.
QuadrantLabel = Literal["warm_wet", "warm_dry", "cool_wet", "cool_dry"]
QUADRANT_LABELS: tuple[QuadrantLabel, ...] = (
    "warm_wet", "warm_dry", "cool_wet", "cool_dry",
)

# Default GWL thresholds (°C) for crossing-year computation.
DEFAULT_GWL_THRESHOLDS: tuple[float, ...] = (1.5, 2.0, 3.0, 4.0)

# Smoothing window for GWL crossing detection, per the ETCCDI / IPCC AR6
# convention (centred 20-year running mean of annual global-mean tas).
GWL_SMOOTHING_WINDOW = 20

# Country-profile time-series window.
TIMESERIES_START_YEAR = 1950
TIMESERIES_END_YEAR = 2100


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cos_lat_weighted_spatial_mean(da: xr.DataArray) -> xr.DataArray:
    """cos(lat)-weighted mean over (lat, lon). Preserves any other dims."""
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(dim=["lat", "lon"])


def _load_and_prepare(
    *,
    model: str,
    variable: str,
    scenario: str,
    country: str,
    crop_method: CropMethod,
    box_offset: float,
    config: Config,
) -> xr.DataArray | None:
    """Open + clean + crop + time-normalise a CMIP6 file.

    Returns the country-cropped, time-normalised DataArray, or ``None`` if
    the file is missing for this model / scenario combination (caller
    leaves NaNs in that row).
    """
    try:
        ds = io.load_cmip6(variable, scenario, model, config=config)
    except FileNotFoundError:
        return None
    if "height" in ds.coords:
        ds = ds.drop_vars("height", errors="ignore")
    if "file_qf" in ds.variables:
        ds = ds.drop_vars("file_qf", errors="ignore")
    da = ds[variable]
    da = geom.crop(
        da, country, method=crop_method, box_offset=box_offset, config=config
    ).data
    return _normalise_time_to_first_of_month(da)


def _slice_window(da: xr.DataArray, window: tuple[int, int]) -> xr.DataArray | None:
    """Slice a normalised-time DataArray to ``[start-01-01, end-12-01]``."""
    sliced = _slice_eval_window(da, window)
    if sliced.sizes.get("time", 0) == 0:
        return None
    return sliced


def _period_means_from_climatology(
    spatial_mean_clim: xr.DataArray,
) -> dict[str, float]:
    """Reduce a (month=12,) spatial-mean climatology to per-period scalars.

    Annual = mean over all 12 months; seasons = mean over the 3 months in
    the season. Matches the legacy spread-notebook order (spatial mean
    first, then period aggregation).
    """
    out: dict[str, float] = {}
    for period in PERIODS:
        months = list(SEASON_MONTHS[period])
        sel = spatial_mean_clim.sel(month=spatial_mean_clim["month"].isin(months))
        out[period] = float(sel.mean(dim="month").values)
    return out


# ---------------------------------------------------------------------------
# Change signals (Δ = long_term − pre_industrial)
# ---------------------------------------------------------------------------


def _per_model_variable_change(
    *,
    model: str,
    variable: str,
    scenario: str,
    country: str,
    crop_method: CropMethod,
    config: Config,
) -> tuple[str, str, dict[str, float] | None]:
    """One (model, variable) change-signal computation. Returns the period
    deltas dict or ``None`` if the file or a window is missing."""
    da = _load_and_prepare(
        model=model, variable=variable, scenario=scenario,
        country=country, crop_method=crop_method,
        box_offset=SPREAD_BOX_OFFSET, config=config,
    )
    if da is None:
        return model, variable, None
    lt = _period_means_for_window(da, config.future_window)
    pi = _period_means_for_window(da, config.pre_industrial)
    if lt is None or pi is None:
        return model, variable, None
    return model, variable, {p: lt[p] - pi[p] for p in PERIODS}


def compute_change_signals(
    country: str,
    *,
    scenario: str = "ssp585",
    crop_method: CropMethod = "bbox",
    config: Config | None = None,
    models: Iterable[str] | None = None,
    n_jobs: int = DEFAULT_N_JOBS,
) -> pd.DataFrame:
    """Per-(model, variable, period) end-of-century change signals.

    Returns a DataFrame indexed by the canonical 1..35 model order with
    columns ``<var>_<period>`` for every var ∈ {tas, pr, psl, tasmax} and
    period ∈ {annual, DJF, MAM, JJA, SON}. Mirrors
    ``results/<country>/assess_long_term_change_spread_<country>.xlsx``.

    Recipe (port of legacy GR_model_spread.ipynb):

    1. Per (variable, model): load the merged hist+ssp585 file (the same
       file covers both windows), drop ``height``/``file_qf``, normalise
       time to first-of-month, crop to country with box_offset=1.5° (the
       paper-era spread default).
    2. For each window — long_term=(2081, 2100), pre_industrial=(1850, 1899) —
       slice, monthly-climatology, cos(lat)-weighted spatial mean, then
       per-period reduction.
    3. Δ_<var>_<period> = long_term − pre_industrial.

    The (model × variable) outer product runs in parallel via
    ``joblib.Parallel`` with the loky (process) backend; pass ``n_jobs=1``
    for serial execution.
    """
    config = config or Config.from_env()
    models_list = list(models) if models is not None else io.load_models_list(config)

    columns = [f"{v}_{p}" for v in SPREAD_VARIABLES for p in PERIODS]
    out = pd.DataFrame(index=models_list, columns=columns, dtype=float)

    parallel = Parallel(n_jobs=n_jobs, backend=DEFAULT_BACKEND)
    results = parallel(
        delayed(_per_model_variable_change)(
            model=m, variable=v, scenario=scenario,
            country=country, crop_method=crop_method, config=config,
        )
        for v in SPREAD_VARIABLES
        for m in models_list
    )
    for model, variable, period_deltas in results:
        if period_deltas is None:
            continue
        for period, delta in period_deltas.items():
            out.loc[model, f"{variable}_{period}"] = delta
    return out


def _period_means_for_window(
    da: xr.DataArray, window: tuple[int, int]
) -> dict[str, float] | None:
    """One window of the change-signal pipeline: slice → climatology →
    spatial mean → per-period scalars. Returns ``None`` if the window
    has no data in the file."""
    sliced = _slice_window(da, window)
    if sliced is None:
        return None
    clim = sliced.groupby("time.month").mean("time")
    spatial_mean_clim = _cos_lat_weighted_spatial_mean(clim)
    return _period_means_from_climatology(spatial_mean_clim)


# ---------------------------------------------------------------------------
# Spread quadrants
# ---------------------------------------------------------------------------


def compute_spread_quadrants(
    country: str,
    *,
    scenario: str = "ssp585",
    crop_method: CropMethod = "bbox",
    config: Config | None = None,
    models: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Per-(model, period) quadrant labels in the (Δtas, Δpr) plane.

    Cutpoints are the seasonal medians of Δtas and Δpr across the 35
    models. ``warm`` = Δtas ≥ median(Δtas); ``wet`` = Δpr ≥ median(Δpr).
    Suffix order is ``<temp>_<precip>`` (e.g. ``warm_dry``).
    """
    deltas = compute_change_signals(
        country, scenario=scenario, crop_method=crop_method,
        config=config, models=models,
    )
    return _quadrants_from_deltas(deltas)


def _quadrants_from_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    quadrants: dict[str, pd.Series] = {}
    for period in PERIODS:
        tas = deltas[f"tas_{period}"]
        pr = deltas[f"pr_{period}"]
        warm = tas >= tas.median()
        wet = pr >= pr.median()
        labels = pd.Series(index=deltas.index, dtype="object")
        labels.loc[warm & wet] = "warm_wet"
        labels.loc[warm & ~wet] = "warm_dry"
        labels.loc[~warm & wet] = "cool_wet"
        labels.loc[~warm & ~wet] = "cool_dry"
        quadrants[period] = labels
    return pd.DataFrame(quadrants)


# ---------------------------------------------------------------------------
# Country-profile artefacts (M9 inputs)
# ---------------------------------------------------------------------------


def _per_model_scenario_timeseries(
    *,
    model: str,
    scenario: str,
    variable: str,
    country: str,
    crop_method: CropMethod,
    box_offset: float,
    config: Config,
    start_year: int,
    end_year: int,
) -> list[tuple[str, str, int, float]]:
    """One (model, scenario) annual time-series. Returns long-form rows."""
    da = _load_and_prepare(
        model=model, variable=variable, scenario=scenario,
        country=country, crop_method=crop_method,
        box_offset=box_offset, config=config,
    )
    if da is None:
        return []
    sliced = _slice_window(da, (start_year, end_year))
    if sliced is None:
        return []
    spatial_mean = _cos_lat_weighted_spatial_mean(sliced)
    annual = spatial_mean.groupby("time.year").mean("time")
    return [
        (model, scenario, int(y), float(annual.sel(year=y).values))
        for y in annual["year"].values
    ]


def compute_country_timeseries(
    country: str,
    *,
    variable: str,
    scenarios: Iterable[str] = ("ssp126", "ssp245", "ssp370", "ssp585"),
    crop_method: CropMethod = "bbox",
    box_offset: float = SPREAD_BOX_OFFSET,
    config: Config | None = None,
    models: Iterable[str] | None = None,
    start_year: int = TIMESERIES_START_YEAR,
    end_year: int = TIMESERIES_END_YEAR,
    n_jobs: int = DEFAULT_N_JOBS,
) -> pd.DataFrame:
    """Annual country-mean time-series, all models × scenarios × years.

    Long-form output: one row per (model, scenario, year), columns
    ``[model, scenario, year, value]``. Used by M9's
    ``subselect.viz.country_profile`` for the change-band / spaghetti /
    warming-level figures. Mirrors the cache convention in
    ``cache/parquet/<country>/timeseries/<variable>__<crop_method>.parquet``.

    Parallelized over (model, scenario) via ``joblib.Parallel``.
    """
    config = config or Config.from_env()
    models_list = list(models) if models is not None else io.load_models_list(config)
    scenarios_list = list(scenarios)

    parallel = Parallel(n_jobs=n_jobs, backend=DEFAULT_BACKEND)
    chunks = parallel(
        delayed(_per_model_scenario_timeseries)(
            model=m, scenario=s, variable=variable, country=country,
            crop_method=crop_method, box_offset=box_offset, config=config,
            start_year=start_year, end_year=end_year,
        )
        for m in models_list
        for s in scenarios_list
    )
    rows: list[tuple[str, str, int, float]] = []
    for chunk in chunks:
        rows.extend(chunk)
    return pd.DataFrame(rows, columns=["model", "scenario", "year", "value"])


def _per_model_scenario_gwl(
    *,
    model: str,
    scenario: str,
    thresholds: tuple[float, ...],
    smoothing_window: int,
    pi_start: int,
    pi_end: int,
    config: Config,
) -> list[tuple[str, str, float, float]]:
    """One (model, scenario) GWL crossing-year computation."""
    try:
        ds = io.load_cmip6("tas", scenario, model, config=config)
    except FileNotFoundError:
        return []
    if "height" in ds.coords:
        ds = ds.drop_vars("height", errors="ignore")
    tas = _normalise_time_to_first_of_month(ds["tas"])

    global_mean = _cos_lat_weighted_spatial_mean(tas)
    annual = global_mean.groupby("time.year").mean("time")
    smoothed = (
        annual.rolling(year=smoothing_window, center=True).mean().dropna("year")
    )

    baseline = smoothed.where(
        (smoothed["year"] >= pi_start) & (smoothed["year"] <= pi_end), drop=True
    )
    if baseline.size == 0:
        return []
    anomaly = smoothed - float(baseline.mean().values)

    rows: list[tuple[str, str, float, float]] = []
    for threshold in thresholds:
        crossing = anomaly.where(anomaly >= threshold, drop=True)
        if crossing.size == 0:
            crossing_year: float = float("nan")
        else:
            crossing_year = float(crossing["year"].values[0])
        rows.append((model, scenario, float(threshold), crossing_year))
    return rows


def compute_gwl_crossing_years(
    *,
    scenarios: Iterable[str] = ("ssp126", "ssp245", "ssp370", "ssp585"),
    thresholds: Iterable[float] = DEFAULT_GWL_THRESHOLDS,
    smoothing_window: int = GWL_SMOOTHING_WINDOW,
    config: Config | None = None,
    models: Iterable[str] | None = None,
    n_jobs: int = DEFAULT_N_JOBS,
) -> pd.DataFrame:
    """First year a model crosses each global-warming-level threshold.

    For each (model, scenario):

    1. Load the global ``tas`` field across the merged hist+ssp file.
    2. Compute annual global-mean tas with cos(lat) weighting.
    3. Centred ``smoothing_window``-year rolling mean (ETCCDI / IPCC AR6
       convention; default 20 years).
    4. Anomaly relative to the model's own pre-industrial baseline
       (1850–1899 mean of the smoothed series).
    5. Find the first year the smoothed anomaly ≥ each threshold.

    Returns a long-form DataFrame: ``(model, scenario, gwl_threshold,
    crossing_year)``. ``crossing_year`` is NaN if the threshold is never
    crossed within the available time series. Parallelised over
    (model, scenario) via ``joblib.Parallel``.
    """
    config = config or Config.from_env()
    models_list = list(models) if models is not None else io.load_models_list(config)
    scenarios_list = list(scenarios)
    thresholds_tuple = tuple(thresholds)
    pi_start, pi_end = config.pre_industrial

    parallel = Parallel(n_jobs=n_jobs, backend=DEFAULT_BACKEND)
    chunks = parallel(
        delayed(_per_model_scenario_gwl)(
            model=m, scenario=s, thresholds=thresholds_tuple,
            smoothing_window=smoothing_window, pi_start=pi_start,
            pi_end=pi_end, config=config,
        )
        for m in models_list
        for s in scenarios_list
    )
    rows: list[tuple[str, str, float, float]] = []
    for chunk in chunks:
        rows.extend(chunk)
    return pd.DataFrame(
        rows, columns=["model", "scenario", "gwl_threshold", "crossing_year"]
    )


# ---------------------------------------------------------------------------
# Phase 2 stub
# ---------------------------------------------------------------------------


def spread_coverage(
    subset_models: Iterable[str],
    full_ensemble_deltas: pd.DataFrame,
    method: Literal["quantile", "convex_hull", "wasserstein"] = "quantile",
) -> float:
    """Phase 2 stub. See docs/future_spread.md."""
    raise NotImplementedError(
        "spread_coverage is a Phase 2 deliverable; see docs/future_spread.md."
    )
