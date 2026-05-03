"""Historical Performance Score (HPS) pipeline.

Per-variable metrics live on each model's native grid (the per-CMIP6-model
upscaled obs is consumed as-is); the σ_obs scalar that feeds the Taylor
Skill Score (TSS) denominator comes from a separate single-grid obs
reference, decoupling TSS from per-model regridding variance — see
``documentation/methods.tex`` § Historical performance.

The HPS recipe:

1. For each (model, variable, season):
   - per-pixel metrics on the model's grid:
       std_dev  = std-over-months of the model climatology
       corr     = Pearson correlation of model-vs-obs climatology over months
       bias     = mean-over-months of (model - obs)
       rmse     = sqrt(mean-over-months of (obs - model)^2)
       crmse    = sqrt(mean-over-months of ((obs-obs̄) - (model-model̄))^2)
       bias_score = 1 / (1 + (|bias| / max(σ_ref, ε))^p)
   - spatially averaged with cos(lat) weighting → scalars.
2. TSS = 2(1+r) / (a + 1/a)^2 with a = σ_model / σ_obs (single-grid),
   R₀ = 1 (Taylor 2001 form).
3. Across {tas, pr, psl} → mean per (model, season) for both TSS and
   bias_score. tasmax is computed for diagnostics but excluded from HPS.
4. Min–max normalise the per-(model, season) values across the 35-model
   ensemble.
5. HMperf = 2 × TSS_mm × bias_score_mm / (TSS_mm + bias_score_mm + ε_hps).

The regression test pins HMperf and per-variable ``{TSS, bias_score}`` to
the published Greece artefacts in ``tests/fixtures/greece/`` within the
locked tolerance ladder (``1e-6`` target, ``1e-4`` fallback documented in
``methods.tex``).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from subselect import geom, io
from subselect.config import Config
from subselect.geom import CropMethod

# Default joblib parallelism for the embarrassingly-parallel per-model loops
# in compute_metrics and compute_change_signals. -1 uses every CPU core; tests
# can pass n_jobs=1 to keep things deterministic.
DEFAULT_N_JOBS = -1
DEFAULT_BACKEND = "loky"  # process-based; safest for xarray + netCDF4 reads

P_EXPONENT = 1.5  # bias-score nonlinearity
EPS_SIGMA = 1e-6  # σ floor for bias_score
EPS_DIVISION = 1e-12  # avoid div-by-zero in TSS std-ratio
EPS_HPS = 1e-12  # avoid div-by-zero in harmonic-mean composite

PERIODS: tuple[str, ...] = ("annual", "DJF", "MAM", "JJA", "SON")
SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "annual": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def taylor_skill_score(corr: float, std_ratio: float) -> float:
    """Taylor (2001) skill score with R₀=1: ``2(1+r) / (a + 1/a)^2``.

    Inputs are clipped to safe ranges: ``a`` is floored at ``EPS_DIVISION``
    and ``r`` is clipped to ``[-1, 1]``.
    """
    a = max(float(std_ratio), EPS_DIVISION)
    r = max(min(float(corr), 1.0), -1.0)
    return 2.0 * (1.0 + r) / ((a + 1.0 / a) ** 2)


def harmonic_mean(a: float, b: float, eps: float = EPS_HPS) -> float:
    """``HM = 2ab / (a + b + ε)``."""
    return 2.0 * (a * b) / (a + b + eps)


def minmax_normalize(values: pd.Series) -> pd.Series:
    """Min-max scale a Series to [0, 1]. Returns the input unchanged if the
    range is zero (degenerate case)."""
    lo = values.min(skipna=True)
    hi = values.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return values.copy()
    return (values - lo) / (hi - lo)


def bias_score_pixel(
    bias_map: xr.DataArray,
    sigma_ref_map: xr.DataArray,
    *,
    p: float = P_EXPONENT,
    eps: float = EPS_SIGMA,
) -> xr.DataArray:
    """Per-pixel ``1 / (1 + (|bias| / max(σ_ref, ε))^p)``.

    Pixels where the bias is NaN stay NaN in the score so the downstream
    area-weighted mean ignores them via xarray's NaN-aware weighted
    reductions.
    """
    sigma = sigma_ref_map.reindex_like(bias_map)
    sigma = xr.where(np.isfinite(sigma) & (sigma > 0), sigma, eps)
    abias = np.abs(bias_map)
    score = 1.0 / (1.0 + (abias / sigma) ** p)
    return score.where(np.isfinite(bias_map))


# ---------------------------------------------------------------------------
# Climatology + per-pixel metric helpers (operate on monthly-climatology data)
# ---------------------------------------------------------------------------


def _normalise_time_to_first_of_month(da: xr.DataArray) -> xr.DataArray:
    """Replace the time coord with day-1-of-month standard datetimes.

    CMIP6 models use a mix of 360-day, noleap, and gregorian calendars with
    timestamps at the start, middle, or end of each month. Converting every
    timestamp to ``(year, month, 1)`` on a standard pandas datetime axis
    gives a calendar-agnostic index so an eval-window slice catches exactly
    the intended 240 monthly steps for a 20-year window.
    """
    new_time = pd.to_datetime(
        {
            "year": da["time"].dt.year,
            "month": da["time"].dt.month,
            "day": 1,
        }
    )
    return da.assign_coords(time=("time", new_time.values))


def _slice_eval_window(da: xr.DataArray, eval_window: tuple[int, int]) -> xr.DataArray:
    """Slice a normalised-time DataArray to ``[start-01-01, end-12-01]``.

    The end-bound is ``<end>-12-01`` (not ``12-31``) to stay valid under
    360-day calendars (Dec 31 does not exist there). After
    :func:`_normalise_time_to_first_of_month` the boundary catches every
    Dec-1 timestamp and excludes nothing.
    """
    start, end = eval_window
    return da.sel(time=slice(f"{start}-01-01", f"{end}-12-01"))


def _monthly_climatology(da: xr.DataArray) -> xr.DataArray:
    """Group by time.month and mean. Returns a (month=12, lat, lon) DataArray."""
    return da.groupby("time.month").mean("time")


def _select_months(clim: xr.DataArray, months: Iterable[int]) -> xr.DataArray:
    return clim.sel(month=clim["month"].isin(list(months)))


def _season_year(time_index: xr.DataArray, months: Iterable[int]) -> xr.DataArray:
    """DJF-safe season-year coordinate (December rolls into next year)."""
    months = list(months)
    if set(months) == {12, 1, 2}:
        return (
            time_index.dt.year + xr.where(time_index.dt.month == 12, 1, 0)
        ).rename("season_year")
    return time_index.dt.year.rename("season_year")


def _interannual_sigma_map(da: xr.DataArray, months: Iterable[int]) -> xr.DataArray:
    """Per-pixel interannual σ for a season (used as σ_ref in bias_score).

    Selects the months from the full time series, groups by DJF-safe
    season-year, takes the per-season-year mean, then takes std across
    season-years.
    """
    months = list(months)
    sel = da.sel(time=da["time"].dt.month.isin(months))
    if sel.sizes.get("time", 0) == 0:
        return xr.full_like(sel.isel(time=0), np.nan, dtype=float)
    syear = _season_year(sel["time"], months)
    seasonal = sel.groupby(syear).mean("time")
    return seasonal.std(dim="season_year", skipna=True)


def _cos_lat_weights(da: xr.DataArray) -> xr.DataArray:
    return np.cos(np.deg2rad(da["lat"]))


def _spatial_weighted_mean(da: xr.DataArray) -> float:
    """Area-weighted mean over (lat, lon) with cos(lat) weights."""
    weights = _cos_lat_weights(da)
    return float(da.weighted(weights).mean(dim=["lat", "lon"]).values)


# ---------------------------------------------------------------------------
# Per-(model, variable) metric pipeline
# ---------------------------------------------------------------------------

PER_PERIOD_METRIC_COLUMNS: tuple[str, ...] = (
    "std_dev",
    "corr",
    "bias",
    "rmse",
    "crmse",
    "bias_score",
)


def _pixel_metrics_for_period(
    obs_clim: xr.DataArray,
    mod_clim: xr.DataArray,
    sigma_ref_map: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """All per-pixel metrics for one (model, variable, period). All over the
    'month' dim of the monthly-climatology subset.

    The model field is force-aligned to the obs field's coords via the
    ``obs*0 + mod`` legacy idiom (line 596 of the HPS notebook), which
    defends against tiny floating-point drift between the upscaled obs's
    coords and the model's native coords on the same nominal grid.
    """
    mod_aligned = obs_clim * 0 + mod_clim
    bias = (mod_aligned - obs_clim).mean(dim="month")
    return {
        "std_dev": mod_aligned.std(dim="month"),
        "corr": xr.corr(obs_clim, mod_aligned, dim="month"),
        "bias": bias,
        "rmse": np.sqrt(((obs_clim - mod_aligned) ** 2).mean(dim="month")),
        "crmse": np.sqrt(
            (
                (
                    (obs_clim - obs_clim.mean(dim="month"))
                    - (mod_aligned - mod_aligned.mean(dim="month"))
                )
                ** 2
            ).mean(dim="month")
        ),
        "bias_score": bias_score_pixel(bias, sigma_ref_map),
    }


def _scalar_obs_std(
    obs_clim: xr.DataArray, months: Iterable[int]
) -> float:
    """σ_obs scalar for the TSS denominator: cos(lat)-weighted spatial mean
    of the per-pixel std-over-months on the native-resolution observed
    climatology.

    Uses the country-cropped native-grid climatology so the spatial average
    runs over the same geographic footprint as the model side. The spatial
    mean is cos(lat)-weighted, consistent with every other spatial mean in
    the pipeline (per-model std_dev, bias_score, etc.).
    """
    sel = _select_months(obs_clim, months)
    return _spatial_weighted_mean(sel.std(dim="month"))


def _per_variable_period_row(
    *,
    obs_clim_full: xr.DataArray,
    mod_clim_full: xr.DataArray,
    obs_full_timeseries: xr.DataArray,
    obs_std_per_period: dict[str, float],
    period: str,
) -> dict[str, float]:
    """Compute every metric column the per-variable xlsx pins for one period.

    Output keys: {period}_std_dev, _corr, _bias, _rmse, _crmse, _bias_score,
    _tss, _tss_hirota. The ``_bias`` column matches the legacy xlsx
    convention of **mean absolute bias** (per legacy line 676:
    ``np.abs(metric_data) if metric_name == 'bias' else metric_data`` before
    the area-weighted mean), not signed bias. The pixelwise bias map itself
    keeps its sign and is what feeds bias_score (which uses ``|bias|``
    inside the formula).
    """
    months = SEASON_MONTHS[period]
    obs_clim = _select_months(obs_clim_full, months)
    mod_clim = _select_months(mod_clim_full, months)
    sigma_ref_map = _interannual_sigma_map(obs_full_timeseries, months)

    pixel_metrics = _pixel_metrics_for_period(obs_clim, mod_clim, sigma_ref_map)
    scalars: dict[str, float] = {}
    for name, da in pixel_metrics.items():
        spatial_input = np.abs(da) if name == "bias" else da
        scalars[f"{period}_{name}"] = _spatial_weighted_mean(spatial_input)

    # TSS variants (same form, different exponent).
    obs_std = obs_std_per_period[period]
    a = scalars[f"{period}_std_dev"] / max(obs_std, EPS_DIVISION)
    a = max(a, EPS_DIVISION)
    r = max(min(scalars[f"{period}_corr"], 1.0), -1.0)
    scalars[f"{period}_tss"] = 2.0 * (1.0 + r) / ((a + 1.0 / a) ** 2)
    scalars[f"{period}_tss_hirota"] = ((1.0 + r) ** 4) / (4.0 * (a + 1.0 / a) ** 2)
    return scalars


def _open_and_crop(
    da: xr.DataArray, country: str, crop_method: CropMethod, config: Config
) -> xr.DataArray:
    return geom.crop(da, country, method=crop_method, config=config).data


def _model_obs_climatologies(
    *,
    model: str,
    variable: str,
    scenario: str,
    country: str,
    crop_method: CropMethod,
    config: Config,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Load + crop + time-normalise + slice + climatology for one (model, variable).

    Returns ``(obs_clim_full, mod_clim_full, obs_full_timeseries_cropped)``.
    The full obs timeseries (1995–2014, country-cropped, monthly) is also
    returned so :func:`_interannual_sigma_map` can compute per-pixel σ_ref
    for the bias_score against the actual interannual variability rather
    than a smoothed climatology. Order: crop → time-normalise →
    eval-slice → climatology.
    """
    cmip6_ds = io.load_cmip6(variable, scenario, model, config=config)
    if "height" in cmip6_ds.coords:
        cmip6_ds = cmip6_ds.drop_vars("height")
    cmip6_var = cmip6_ds[variable]
    obs_var = io.load_w5e5(variable, model, config=config)[variable]

    cmip6_var = _open_and_crop(cmip6_var, country, crop_method, config)
    obs_var = _open_and_crop(obs_var, country, crop_method, config)

    cmip6_var = _normalise_time_to_first_of_month(cmip6_var)
    obs_var = _normalise_time_to_first_of_month(obs_var)
    cmip6_var = _slice_eval_window(cmip6_var, config.eval_window)
    obs_var = _slice_eval_window(obs_var, config.eval_window)

    return (
        _monthly_climatology(obs_var),
        _monthly_climatology(cmip6_var),
        obs_var,
    )


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


def _compute_obs_std_per_period(
    variable: str, country: str, crop_method: CropMethod, config: Config
) -> dict[str, float]:
    """σ_obs scalars per period from the native 0.5° W5E5 obs reference.

    Loaded once per variable, used as the TSS denominator across all 35
    models. See ``documentation/methods.tex`` § Methodology corrections
    (post-paper) for the native-grid + cos(lat)-weighted choice.
    """
    obs_full = io.load_native_w5e5(variable, config=config)[variable]
    obs_full = _open_and_crop(obs_full, country, crop_method, config)
    obs_full = _normalise_time_to_first_of_month(obs_full)
    obs_full = _slice_eval_window(obs_full, config.eval_window)
    obs_clim_full = _monthly_climatology(obs_full)
    return {
        period: _scalar_obs_std(obs_clim_full, SEASON_MONTHS[period])
        for period in PERIODS
    }


def _per_model_metrics(
    *,
    model: str,
    variable: str,
    scenario: str,
    country: str,
    crop_method: CropMethod,
    config: Config,
    obs_std_per_period: dict[str, float],
) -> tuple[str, dict[str, float] | None]:
    """Compute every metric column for one (model, variable). Returns
    ``(model, scalars)`` or ``(model, None)`` if the file is missing."""
    try:
        obs_clim_full, mod_clim_full, obs_full_ts = _model_obs_climatologies(
            model=model, variable=variable, scenario=scenario,
            country=country, crop_method=crop_method, config=config,
        )
    except FileNotFoundError:
        return model, None
    scalars: dict[str, float] = {}
    for period in PERIODS:
        scalars.update(
            _per_variable_period_row(
                obs_clim_full=obs_clim_full,
                mod_clim_full=mod_clim_full,
                obs_full_timeseries=obs_full_ts,
                obs_std_per_period=obs_std_per_period,
                period=period,
            )
        )
    return model, scalars


def compute_metrics(
    country: str,
    *,
    variable: str,
    scenario: str = "ssp585",
    crop_method: CropMethod = "bbox",
    config: Config | None = None,
    models: Iterable[str] | None = None,
    n_jobs: int = DEFAULT_N_JOBS,
) -> pd.DataFrame:
    """Per-(model, period) metric table for one variable.

    Returns a DataFrame whose index is the canonical 1..35 model ordering and
    whose columns are ``{period}_{metric}`` for each period in ``PERIODS``
    and each metric in ``PER_PERIOD_METRIC_COLUMNS`` plus ``tss`` and
    ``tss_hirota``. Mirrors
    ``results/<country>/assess_cmip6_<variable>_mon_perf_metrics_all_seasons_<country>.xlsx``.

    The 35-model loop runs in parallel via ``joblib.Parallel`` with the loky
    (process) backend by default — each worker opens its own NetCDF files,
    so there is no shared xarray state to coordinate. Pass ``n_jobs=1`` for
    serial execution (e.g. when debugging or when the joblib worker pool
    overhead would dominate for a small models list).
    """
    config = config or Config.from_env()
    if models is None:
        models = io.load_models_list(config)
    models = list(models)

    obs_std_per_period = _compute_obs_std_per_period(
        variable, country, crop_method, config
    )

    metric_columns = [
        f"{period}_{m}"
        for period in PERIODS
        for m in PER_PERIOD_METRIC_COLUMNS + ("tss", "tss_hirota")
    ]
    out = pd.DataFrame(index=models, columns=metric_columns, dtype=float)

    parallel = Parallel(n_jobs=n_jobs, backend=DEFAULT_BACKEND)
    results = parallel(
        delayed(_per_model_metrics)(
            model=m, variable=variable, scenario=scenario,
            country=country, crop_method=crop_method, config=config,
            obs_std_per_period=obs_std_per_period,
        )
        for m in models
    )
    for model, scalars in results:
        if scalars is None:
            continue
        for col, val in scalars.items():
            out.loc[model, col] = val
    return out


def compute_hps(
    country: str,
    *,
    scenario: str = "ssp585",
    crop_method: CropMethod = "bbox",
    config: Config | None = None,
    models: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Composite Historical Performance Score, per (model, period).

    Returns a DataFrame indexed by the canonical 1..35 model order with
    columns ``annual``, ``DJF``, ``MAM``, ``JJA``, ``SON``. Mirrors
    ``results/<country>/assess_cmip6_composite_HMperf_<country>.xlsx`` modulo
    the column-name suffix (the xlsx uses ``<period>_HMperf``; this returns
    just ``<period>``). The regression test renames before comparing.

    Recipe:

    1. Per-variable metric tables for ``{tas, pr, psl}`` (tasmax excluded).
    2. Composite TSS per period = mean across variables, then min-max
       scaled to [0, 1] within the 35-model ensemble.
    3. Composite bias_score per period = same.
    4. HMperf = ``2 × TSS_mm × bs_mm / (TSS_mm + bs_mm + ε)``.
    """
    config = config or Config.from_env()
    if models is None:
        models = io.load_models_list(config)
    models = list(models)

    per_var_tables = {
        var: compute_metrics(
            country, variable=var, scenario=scenario,
            crop_method=crop_method, config=config, models=models,
        )
        for var in config.hps_variables
    }

    hps = pd.DataFrame(index=models, columns=list(PERIODS), dtype=float)
    for period in PERIODS:
        comp_tss = pd.concat(
            [per_var_tables[v][f"{period}_tss"] for v in config.hps_variables],
            axis=1,
        ).mean(axis=1, skipna=True)
        comp_bs = pd.concat(
            [per_var_tables[v][f"{period}_bias_score"] for v in config.hps_variables],
            axis=1,
        ).mean(axis=1, skipna=True)

        tss_mm = minmax_normalize(comp_tss)
        bs_mm = minmax_normalize(comp_bs)

        hps[period] = 2.0 * (tss_mm * bs_mm) / (tss_mm + bs_mm + EPS_HPS)
    return hps
