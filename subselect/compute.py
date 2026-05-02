"""L1 orchestrator: compute every artefact a country needs for the figure set.

Single public entry point :func:`compute`. The result is a populated
:class:`subselect.state.SubselectState`; intermediate artefacts are persisted
to ``cache/<country>/`` so subsequent calls are cache-hit-fast (<30 s for a
country whose cache is already populated).

Each builder function in this module produces one logical artefact and is
guarded by a cache check at the top — recompute is skipped when the cached
file exists and no upstream input is newer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from subselect import io, profile_signals
from subselect.cache import Cache
from subselect.config import Config
from subselect.state import ProfileSignals, SubselectState


SCENARIOS: tuple[str, ...] = ("ssp126", "ssp245", "ssp370", "ssp585")
HPS_VARIABLES: tuple[str, ...] = ("tas", "pr", "psl")
ALL_VARIABLES: tuple[str, ...] = HPS_VARIABLES + ("tasmax",)
TIMESERIES_VARIABLES: tuple[str, ...] = ("tas", "pr", "psl")
GLOBAL_COUNTRY = "_global"
DEFAULT_N_JOBS = -1  # all CPUs; matches subselect.performance / subselect.spread
DEFAULT_BACKEND = "loky"


# ---------------------------------------------------------------------------
# Annual time series
# ---------------------------------------------------------------------------

def _annual_timeseries_one(
    variable: str, model: str, scenario: str, country: str, config: Config,
) -> tuple[str, pd.Series] | None:
    """Compute the annual country-mean time-series for one (variable, model,
    scenario). Returns ``(column_name, series)`` or ``None`` when the source
    NetCDF is missing for this combination."""
    from subselect.geom import crop
    from subselect.performance import (
        _normalise_time_to_first_of_month, _spatial_weighted_mean,
    )

    try:
        ds = io.load_cmip6(variable, scenario, model, config=config)
    except FileNotFoundError:
        return None
    if "height" in ds.coords:
        ds = ds.drop_vars("height")
    da = ds[variable]

    fpath = io.cmip6_path(variable, scenario, model, config=config)
    parts = fpath.stem.split("_")
    variant = parts[2] if len(parts) >= 4 else "r1i1p1f1"

    cropped = crop(da, country, method="bbox", config=config).data
    cropped = _normalise_time_to_first_of_month(cropped)
    annual = cropped.groupby("time.year").mean("time")
    series = pd.Series(
        {
            int(y): _spatial_weighted_mean(annual.sel(year=y))
            for y in annual.year.values
        }
    )
    col_name = f"{variable}_{model}_{variant}_{scenario}_yr"
    ds.close()
    return col_name, series


def _build_annual_timeseries(
    variable: str, country: str, config: Config, *, n_jobs: int = DEFAULT_N_JOBS,
) -> pd.DataFrame:
    """Annual country-mean time-series 1850–2100 across all SSPs and models.

    Columns: ``<variable>_<MODEL>_<variant>_<scenario>_yr``. Index: year (int).
    The 35-models × 4-scenarios loop runs in parallel via joblib loky.
    """
    models = io.load_models_list(config)
    jobs = [(model, scenario) for model in models for scenario in SCENARIOS]
    parallel = Parallel(n_jobs=n_jobs, backend=DEFAULT_BACKEND)
    results = parallel(
        delayed(_annual_timeseries_one)(variable, m, s, country, config)
        for m, s in jobs
    )
    cols: dict[str, pd.Series] = {}
    for r in results:
        if r is None:
            continue
        col_name, series = r
        cols[col_name] = series
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "time"
    return df


def annual_timeseries(country: str, config: Config, cache: Cache) -> dict[str, pd.DataFrame]:
    """Build (or load from cache) the annual country-mean time-series for the
    three time-series variables (``tas``, ``pr``, ``psl``).
    """
    out: dict[str, pd.DataFrame] = {}
    for variable in TIMESERIES_VARIABLES:
        key = f"annual_timeseries__{variable}"
        deps = _cmip6_inputs(variable, config)
        if cache.is_fresh(key, deps):
            out[variable] = cache.load(key)
            continue
        df = _build_annual_timeseries(variable, country, config)
        cache.save(key, df, deps=deps)
        out[variable] = df
    return out


def _cmip6_inputs(variable: str, config: Config) -> list[Path]:
    """Return the list of CMIP6 NetCDFs that feed an annual time-series.

    Used as the dependency set for cache staleness — when one of these files
    changes, the artefact must be recomputed. Limited to one per (variable,
    scenario) since per-model glob is expensive and all 35 share an mtime in
    practice.
    """
    paths: list[Path] = []
    for scenario in SCENARIOS:
        d = io.cmip6_dir(variable, scenario, config=config)
        if d.is_dir():
            paths.extend(sorted(d.glob("*.nc"))[:1])  # one file per dir is enough
    return paths


# ---------------------------------------------------------------------------
# Warming levels + future anomalies (cells 15, 19, 21)
# ---------------------------------------------------------------------------

def warming_levels(
    annual_temperature: pd.DataFrame, cache: Cache,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Crossing years per (model, scenario) and per-SSP medians (cell 15)."""
    deps_proxy = _annual_ts_proxy_path(cache, "tas")
    if cache.is_fresh("warming_levels", [deps_proxy]) and cache.is_fresh(
        "warming_level_medians", [deps_proxy]
    ):
        return (
            cache.load("warming_levels"),
            cache.load("warming_level_medians"),
            cache.load("tas_pi_baseline"),
            cache.load("tas_rp_baseline"),
        )
    crossings, medians, pi_b, rp_b = profile_signals.compute_warming_levels(
        annual_temperature
    )
    cache.save("warming_levels", crossings, deps=[deps_proxy])
    cache.save("warming_level_medians", medians, deps=[deps_proxy])
    cache.save("tas_pi_baseline", pi_b, deps=[deps_proxy])
    cache.save("tas_rp_baseline", rp_b, deps=[deps_proxy])
    return crossings, medians, pi_b, rp_b


def _annual_ts_proxy_path(cache: Cache, variable: str) -> Path:
    """Return the parquet path of the annual time-series for *variable* — used
    as the dependency-mtime proxy for downstream artefacts that consume it."""
    p = cache.root / "parquet" / f"annual_timeseries__{variable}.parquet"
    return p


def tas_future_anomalies(
    annual_temperature: pd.DataFrame,
    warming_levels_all_models: pd.DataFrame,
    pi_baseline: pd.Series,
    rp_baseline: pd.Series,
    cache: Cache,
) -> dict[str, pd.DataFrame]:
    """Per-SSP statistics over fixed future periods (cell 19)."""
    proxy = _annual_ts_proxy_path(cache, "tas")
    key = "future_anomalies__tas"
    if cache.is_fresh(key, [proxy]):
        return cache.load(key)
    out = profile_signals.compute_tas_future_anomalies(
        annual_temperature, warming_levels_all_models, pi_baseline, rp_baseline,
    )
    cache.save(key, out, deps=[proxy])
    return out


def pr_future_percent_anomalies(
    annual_precipitation: pd.DataFrame,
    warming_levels_all_models: pd.DataFrame,
    cache: Cache,
) -> dict[str, pd.DataFrame]:
    """Per-SSP percent-change statistics for precipitation (cell 21)."""
    proxy_pr = _annual_ts_proxy_path(cache, "pr")
    proxy_tas = _annual_ts_proxy_path(cache, "tas")
    key = "future_anomalies__pr_percent"
    if cache.is_fresh(key, [proxy_pr, proxy_tas]):
        return cache.load(key)
    out = profile_signals.compute_pr_future_percent_anomalies(
        annual_precipitation, warming_levels_all_models,
    )
    cache.save(key, out, deps=[proxy_pr, proxy_tas])
    return out


# ---------------------------------------------------------------------------
# Performance pipeline (M7)
# ---------------------------------------------------------------------------

def performance_metrics(
    country: str, config: Config, cache: Cache,
) -> dict[str, pd.DataFrame]:
    """Reload performance metrics from cache (assumes :func:`fused_performance_pass`
    has already populated them)."""
    out: dict[str, pd.DataFrame] = {}
    for variable in ALL_VARIABLES:
        key = f"performance_metrics__{variable}"
        if cache.has(key):
            out[variable] = cache.load(key)
    return out


def _fused_per_model(
    *,
    model: str,
    variable: str,
    scenario: str,
    country: str,
    crop_method: str,
    config: Config,
    obs_std_per_period: dict[str, float],
    bias_periods: list[str] | None,
) -> dict | None:
    """Single-pass per-(model, variable) worker.

    Loads model + observation climatologies *once* and computes every
    downstream artefact that depends on them — metric scalars, monthly
    spatial means, per-period bias fields. Returning all three from one
    NetCDF read pair is what brings the cold-cache pipeline runtime under
    the target.
    """
    from subselect.performance import (
        PERIODS, SEASON_MONTHS,
        _model_obs_climatologies, _per_variable_period_row,
        _select_months, _spatial_weighted_mean,
    )

    try:
        obs_clim_full, mod_clim_full, obs_full_ts = _model_obs_climatologies(
            model=model, variable=variable, scenario=scenario,
            country=country, crop_method=crop_method, config=config,
        )
    except FileNotFoundError:
        return None

    metric_scalars: dict[str, float] = {}
    for period in PERIODS:
        metric_scalars.update(_per_variable_period_row(
            obs_clim_full=obs_clim_full,
            mod_clim_full=mod_clim_full,
            obs_full_timeseries=obs_full_ts,
            obs_std_per_period=obs_std_per_period,
            period=period,
        ))

    months = list(range(1, 13))
    cmip_monthly = [_spatial_weighted_mean(mod_clim_full.sel(month=m)) for m in months]
    obs_monthly = [_spatial_weighted_mean(obs_clim_full.sel(month=m)) for m in months]

    bias_per_period: dict[str, xr.DataArray] | None = None
    if bias_periods is not None:
        bias_per_period = {}
        mod_aligned = obs_clim_full * 0 + mod_clim_full
        for period in bias_periods:
            obs_p = _select_months(obs_clim_full, SEASON_MONTHS[period])
            mod_p = _select_months(mod_aligned, SEASON_MONTHS[period])
            bias_per_period[period] = (mod_p - obs_p).mean(dim="month").compute()

    return {
        "model": model,
        "metrics": metric_scalars,
        "cmip_monthly": cmip_monthly,
        "obs_monthly": obs_monthly,
        "bias_per_period": bias_per_period,
    }


def fused_performance_pass(
    country: str,
    config: Config,
    cache: Cache,
    *,
    include_bias_maps: bool = True,
    include_seasonal_bias: bool = False,
    crop_method: str = "bbox",
    scenario: str = "ssp585",
    n_jobs: int = DEFAULT_N_JOBS,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[str, pd.DataFrame]],
    pd.DataFrame,
    dict[str, dict[str, xr.Dataset]],
    dict[str, dict[str, dict[str, xr.DataArray]]],
]:
    """Run every per-model climatology load exactly once across the
    performance, monthly-means, and bias-map artefacts.

    Returns ``(performance_metrics, monthly_means, observed_std_dev,
    observed_maps, bias_maps)``.
    """
    from subselect.performance import (
        PER_PERIOD_METRIC_COLUMNS, PERIODS, SEASON_MONTHS,
        _compute_obs_std_per_period, _monthly_climatology,
        _normalise_time_to_first_of_month, _open_and_crop,
        _select_months, _slice_eval_window,
    )

    months = list(range(1, 13))
    metric_columns = [
        f"{period}_{m}"
        for period in PERIODS
        for m in PER_PERIOD_METRIC_COLUMNS + ("tss", "tss_hirota")
    ]
    bias_periods = list(PERIODS) if include_seasonal_bias else ["annual"]

    perf: dict[str, pd.DataFrame] = {}
    mm: dict[str, dict[str, pd.DataFrame]] = {}
    obs_maps: dict[str, dict[str, xr.Dataset]] = (
        {p: {} for p in bias_periods} if include_bias_maps else {}
    )
    bias_maps_out: dict[str, dict[str, dict[str, xr.DataArray]]] = (
        {p: {v: {} for v in ALL_VARIABLES} for p in bias_periods}
        if include_bias_maps else {}
    )

    obs_std_columns: dict[str, list[float]] = {}
    parallel = Parallel(n_jobs=n_jobs, backend=DEFAULT_BACKEND)
    models = io.load_models_list(config)

    for variable in ALL_VARIABLES:
        deps = _cmip6_inputs(variable, config) + _reference_inputs(variable, config)

        perf_key = f"performance_metrics__{variable}"
        cmip_key = f"monthly_means__{variable}__cmip6"
        obs_key = f"monthly_means__{variable}__obs"
        tabular_cached = (
            cache.is_fresh(perf_key, deps)
            and cache.is_fresh(cmip_key, deps)
            and cache.is_fresh(obs_key, deps)
        )

        # σ_obs scalars from native obs — needed even on a tabular-cache hit
        # because :func:`observed_std_dev` is a top-level state field.
        obs_std_per_period = _compute_obs_std_per_period(
            variable, country, crop_method, config,
        )
        obs_std_columns[variable] = [obs_std_per_period[p] for p in PERIODS]

        # Observed-maps top panel (one native-obs read per variable)
        if include_bias_maps:
            try:
                obs_full = io.load_native_w5e5(variable, config=config)[variable]
                obs_full = _open_and_crop(obs_full, country, crop_method, config)
                obs_full = _normalise_time_to_first_of_month(obs_full)
                obs_full = _slice_eval_window(obs_full, config.eval_window)
                obs_clim_native = _monthly_climatology(obs_full)
                for period in bias_periods:
                    sel = _select_months(obs_clim_native, SEASON_MONTHS[period])
                    obs_maps[period][variable] = (
                        sel.mean(dim="month").to_dataset(name=variable)
                    )
            except FileNotFoundError:
                pass

        # Skip the per-model pass when every tabular artefact is cached and the
        # caller didn't ask for bias maps (which are not cached on disk).
        if tabular_cached and not include_bias_maps:
            perf[variable] = cache.load(perf_key)
            mm[variable] = {"cmip6": cache.load(cmip_key), "obs": cache.load(obs_key)}
            continue

        results = parallel(
            delayed(_fused_per_model)(
                model=m, variable=variable, scenario=scenario,
                country=country, crop_method=crop_method, config=config,
                obs_std_per_period=obs_std_per_period,
                bias_periods=bias_periods if include_bias_maps else None,
            )
            for m in models
        )

        perf_df = pd.DataFrame(index=models, columns=metric_columns, dtype=float)
        cmip_cols: dict[str, list[float]] = {}
        obs_means_first: list[float] | None = None
        for r in results:
            if r is None:
                continue
            model = r["model"]
            for col, val in r["metrics"].items():
                perf_df.loc[model, col] = val
            cmip_cols[model] = r["cmip_monthly"]
            if obs_means_first is None:
                obs_means_first = r["obs_monthly"]
            if r["bias_per_period"] is not None:
                for period, da in r["bias_per_period"].items():
                    bias_maps_out[period][variable][model] = da

        cache.save(perf_key, perf_df, deps=deps)
        cmip_df = pd.DataFrame(cmip_cols, index=pd.Index(months, name="month"))
        cache.save(cmip_key, cmip_df, deps=deps)
        obs_df = (
            pd.DataFrame(
                {variable: obs_means_first}, index=pd.Index(months, name="month"),
            ) if obs_means_first is not None else pd.DataFrame()
        )
        if not obs_df.empty:
            cache.save(obs_key, obs_df, deps=deps)

        perf[variable] = perf_df
        mm[variable] = {"cmip6": cmip_df, "obs": obs_df}

    obs_std_df = pd.DataFrame(obs_std_columns, index=list(PERIODS))
    obs_std_deps = sum((_reference_inputs(v, config) for v in ALL_VARIABLES), [])
    obs_std_key = "observed_std_dev"
    if cache.is_fresh(obs_std_key, obs_std_deps):
        obs_std_df = cache.load(obs_std_key)
    else:
        cache.save(obs_std_key, obs_std_df, deps=obs_std_deps)

    return perf, mm, obs_std_df, obs_maps, bias_maps_out


def _reference_inputs(variable: str, config: Config) -> list[Path]:
    """Return one observation reference NetCDF per variable (proxy for mtime
    tracking — all per-model upscaled files share a mtime in practice)."""
    paths: list[Path] = []
    p = io.single_grid_reference_path(variable, config=config)
    if p.exists():
        paths.append(p)
    ref_dir = config.reference_root
    if ref_dir.exists():
        for sub in sorted(ref_dir.iterdir())[:1]:
            if sub.is_dir():
                first = sorted(sub.glob(f"{variable}_*.nc"))
                if first:
                    paths.append(first[0])
    return paths


def composite_hps(
    country: str, perf_metrics: dict[str, pd.DataFrame], config: Config, cache: Cache,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from subselect.performance import (
        EPS_HPS, PERIODS, compute_hps, minmax_normalize,
    )

    deps = sum((_cmip6_inputs(v, config) + _reference_inputs(v, config) for v in HPS_VARIABLES), [])

    key_hps = "composite_hps"
    if cache.is_fresh(key_hps, deps):
        hps = cache.load(key_hps)
    else:
        hps = compute_hps(country, config=config).rename(
            columns={p: f"{p}_HMperf" for p in PERIODS}
        )
        cache.save(key_hps, hps, deps=deps)

    key_full = "composite_hps_full"
    if cache.is_fresh(key_full, deps):
        full = cache.load(key_full)
    else:
        full = pd.DataFrame()
        for period in PERIODS:
            comp_tss = pd.concat(
                [perf_metrics[v][f"{period}_tss"] for v in HPS_VARIABLES], axis=1,
            ).mean(axis=1, skipna=True)
            comp_bs_raw = pd.concat(
                [perf_metrics[v][f"{period}_bias_score"] for v in HPS_VARIABLES], axis=1,
            ).mean(axis=1, skipna=True)
            tss_mm = minmax_normalize(comp_tss)
            bs_mm = minmax_normalize(comp_bs_raw)
            hmperf = 2.0 * (tss_mm * bs_mm) / (tss_mm + bs_mm + EPS_HPS)
            full[f"{period}_rank"] = hmperf.rank(ascending=False).astype(int)
            full[f"{period}_HMperf"] = hmperf
            full[f"{period}_TSS_mm"] = tss_mm
            full[f"{period}_bias_score_mm"] = bs_mm
            full[f"{period}_bias_score_raw"] = comp_bs_raw
        full = full.sort_values("annual_HMperf", ascending=False)
        cache.save(key_full, full, deps=deps)
    return hps, full


def observed_std_dev(country: str, config: Config, cache: Cache) -> pd.DataFrame:
    from subselect.performance import PERIODS, _compute_obs_std_per_period

    deps = sum((_reference_inputs(v, config) for v in ALL_VARIABLES), [])
    key = "observed_std_dev"
    if cache.is_fresh(key, deps):
        return cache.load(key)
    sigmas = {
        v: _compute_obs_std_per_period(v, country, "bbox", config) for v in ALL_VARIABLES
    }
    df = pd.DataFrame(
        {v: [sigmas[v][p] for p in PERIODS] for v in ALL_VARIABLES},
        index=list(PERIODS),
    )
    cache.save(key, df, deps=deps)
    return df


def _monthly_means_one(
    model: str, variable: str, country: str, config: Config,
) -> tuple[str, list[float], list[float]] | None:
    """Per-(model, variable) monthly-cycle spatial means.

    Returns ``(model, cmip_means, obs_means)`` or ``None`` when the source
    files are missing for this model. Both lists have length 12.
    """
    from subselect.performance import (
        _model_obs_climatologies, _spatial_weighted_mean,
    )

    try:
        obs_clim, mod_clim, _ = _model_obs_climatologies(
            model=model, variable=variable, scenario="ssp585",
            country=country, crop_method="bbox", config=config,
        )
    except FileNotFoundError:
        return None
    months = list(range(1, 13))
    cmip_means = [_spatial_weighted_mean(mod_clim.sel(month=m)) for m in months]
    obs_means = [_spatial_weighted_mean(obs_clim.sel(month=m)) for m in months]
    return model, cmip_means, obs_means


def monthly_means(
    country: str, config: Config, cache: Cache, *, n_jobs: int = DEFAULT_N_JOBS,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Per-variable spatial-mean monthly-cycle climatology for CMIP6 models +
    observations. ``{var: {'cmip6': df, 'obs': df}}``.

    The 35-models × 4-vars loop runs in parallel via joblib loky.
    """
    months = list(range(1, 13))
    out: dict[str, dict[str, pd.DataFrame]] = {}
    parallel = Parallel(n_jobs=n_jobs, backend=DEFAULT_BACKEND)
    for variable in ALL_VARIABLES:
        deps = _cmip6_inputs(variable, config) + _reference_inputs(variable, config)
        cmip_key = f"monthly_means__{variable}__cmip6"
        obs_key = f"monthly_means__{variable}__obs"
        if cache.is_fresh(cmip_key, deps) and cache.is_fresh(obs_key, deps):
            out[variable] = {
                "cmip6": cache.load(cmip_key),
                "obs": cache.load(obs_key),
            }
            continue
        results = parallel(
            delayed(_monthly_means_one)(m, variable, country, config)
            for m in io.load_models_list(config)
        )
        cmip_cols: dict[str, list[float]] = {}
        obs_means_first: list[float] | None = None
        for r in results:
            if r is None:
                continue
            model, cmip_m, obs_m = r
            cmip_cols[model] = cmip_m
            if obs_means_first is None:
                obs_means_first = obs_m
        if obs_means_first is None:
            continue
        cmip_df = pd.DataFrame(cmip_cols, index=pd.Index(months, name="month"))
        obs_df = pd.DataFrame(
            {variable: obs_means_first}, index=pd.Index(months, name="month")
        )
        cache.save(cmip_key, cmip_df, deps=deps)
        cache.save(obs_key, obs_df, deps=deps)
        out[variable] = {"cmip6": cmip_df, "obs": obs_df}
    return out


# ---------------------------------------------------------------------------
# Spread pipeline (M8)
# ---------------------------------------------------------------------------

def spread(
    country: str, config: Config, cache: Cache,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from subselect.performance import PERIODS
    from subselect.spread import (
        SPREAD_BOX_OFFSET, SPREAD_VARIABLES,
        _load_and_prepare, _period_means_for_window, compute_change_signals,
    )

    deps = sum((_cmip6_inputs(v, config) for v in SPREAD_VARIABLES), [])

    key_change = "change_signals"
    if cache.is_fresh(key_change, deps):
        change_df = cache.load(key_change)
    else:
        change_df = compute_change_signals(country, scenario="ssp585", config=config)
        cache.save(key_change, change_df, deps=deps)

    key_long = "long_term_spread"
    key_pi = "pre_industrial_spread"
    if cache.is_fresh(key_long, deps) and cache.is_fresh(key_pi, deps):
        return change_df, cache.load(key_long), cache.load(key_pi)

    models = io.load_models_list(config)
    columns = [f"{v}_{p}" for v in SPREAD_VARIABLES for p in PERIODS]
    long_df = pd.DataFrame(index=models, columns=columns, dtype=float)
    pi_df = pd.DataFrame(index=models, columns=columns, dtype=float)

    def _spread_one(variable: str, model: str):
        da = _load_and_prepare(
            model=model, variable=variable, scenario="ssp585",
            country=country, crop_method="bbox",
            box_offset=SPREAD_BOX_OFFSET, config=config,
        )
        if da is None:
            return variable, model, None, None
        lt = _period_means_for_window(da, config.future_window)
        pi = _period_means_for_window(da, config.pre_industrial)
        return variable, model, lt, pi

    parallel = Parallel(n_jobs=DEFAULT_N_JOBS, backend=DEFAULT_BACKEND)
    results = parallel(
        delayed(_spread_one)(v, m) for v in SPREAD_VARIABLES for m in models
    )
    for variable, model, lt, pi in results:
        if lt is not None:
            for p in PERIODS:
                long_df.loc[model, f"{variable}_{p}"] = lt[p]
        if pi is not None:
            for p in PERIODS:
                pi_df.loc[model, f"{variable}_{p}"] = pi[p]
    cache.save(key_long, long_df, deps=deps)
    cache.save(key_pi, pi_df, deps=deps)
    return change_df, long_df, pi_df


# ---------------------------------------------------------------------------
# Bias maps (M9.2)
# ---------------------------------------------------------------------------

def bias_maps(
    country: str,
    *,
    include_seasonal: bool = False,
    crop_method: str = "bbox",
    scenario: str = "ssp585",
    config: Config,
    cache: Cache,
) -> tuple[
    dict[str, dict[str, xr.Dataset]],
    dict[str, dict[str, dict[str, xr.DataArray]]],
]:
    """Observed-mean fields + per-model bias fields.

    Returns ``(observed_maps, bias_maps)`` where:

    - ``observed_maps[period][variable]`` is an ``xr.Dataset`` (country-cropped,
      period-averaged) sourced from the native 0.5° W5E5 reference.
    - ``bias_maps[period][variable][model]`` is an ``xr.DataArray`` of
      ``model − upscaled_obs`` for the requested period.

    Period is one of ``annual / DJF / MAM / JJA / SON``. With
    ``include_seasonal=False`` (default) only ``annual`` is computed.

    The xarray fields are not cached (zarr round-trip is comparable to the
    compute cost at this scale). The per-model climatology pipeline they
    depend on is shared with the performance metrics, so wall-clock cost is
    dominated by I/O that happens anyway.
    """
    from subselect import io
    from subselect.performance import (
        PERIODS,
        SEASON_MONTHS,
        _model_obs_climatologies,
        _open_and_crop,
        _normalise_time_to_first_of_month,
        _slice_eval_window,
        _monthly_climatology,
        _select_months,
    )

    periods = list(PERIODS) if include_seasonal else ["annual"]
    variables = ALL_VARIABLES
    models = io.load_models_list(config)

    observed_maps: dict[str, dict[str, xr.Dataset]] = {p: {} for p in periods}
    bias_maps_out: dict[str, dict[str, dict[str, xr.DataArray]]] = {
        p: {v: {} for v in variables} for p in periods
    }

    for variable in variables:
        try:
            obs_full = io.load_native_w5e5(variable, config=config)[variable]
        except FileNotFoundError:
            continue
        obs_full = _open_and_crop(obs_full, country, crop_method, config)
        obs_full = _normalise_time_to_first_of_month(obs_full)
        obs_full = _slice_eval_window(obs_full, config.eval_window)
        obs_clim_native = _monthly_climatology(obs_full)
        for period in periods:
            sel = _select_months(obs_clim_native, SEASON_MONTHS[period])
            observed_maps[period][variable] = sel.mean(dim="month").to_dataset(name=variable)

        for model in models:
            try:
                obs_clim_full, mod_clim_full, _ = _model_obs_climatologies(
                    model=model, variable=variable, scenario=scenario,
                    country=country, crop_method=crop_method, config=config,
                )
            except FileNotFoundError:
                continue
            mod_aligned = obs_clim_full * 0 + mod_clim_full
            for period in periods:
                obs_p = _select_months(obs_clim_full, SEASON_MONTHS[period])
                mod_p = _select_months(mod_aligned, SEASON_MONTHS[period])
                bias_maps_out[period][variable][model] = (mod_p - obs_p).mean(dim="month")

    return observed_maps, bias_maps_out


# ---------------------------------------------------------------------------
# Global comparison artefacts (paper-era xlsx — read-only for now)
# ---------------------------------------------------------------------------

def global_comparison(config: Config) -> dict[str, dict[str, pd.DataFrame] | pd.DataFrame]:
    """Read the paper-era global-mean comparison artefacts.

    The country-profile figures compare the country's projections to the
    global ensemble. The global tables are paper-era and live under
    ``results/global/``; we read them directly. If a future framework
    iteration computes the global side from raw CMIP6, this function becomes
    a builder; for now it is a passthrough loader.
    """
    g = config.results_root / "global"
    return {
        "warming_level_medians": pd.read_excel(
            g / "cmip6_warming_levels_median_global.xlsx", index_col=0,
        ),
        "tas_future_anomalies": {
            "recent_past": pd.read_excel(
                g / "cmip6_tas_future_anomalies_recent_past_global.xlsx", index_col=0,
            ),
            "pre_industrial": pd.read_excel(
                g / "cmip6_tas_future_anomalies_pre_industrial_global.xlsx", index_col=0,
            ),
        },
        "pr_future_percent_anomalies": {
            "recent_past": pd.read_excel(
                g / "cmip6_pr_future_percent_anomalies_recent_past_global.xlsx", index_col=0,
            ),
            "pre_industrial": pd.read_excel(
                g / "cmip6_pr_future_percent_anomalies_pre_industrial_global.xlsx", index_col=0,
            ),
        },
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute(
    country: str,
    *,
    scenarios: tuple[str, ...] = SCENARIOS,
    only: Iterable[str] | None = None,
    force: bool = False,
    config: Config | None = None,
    include_bias_maps: bool = True,
    include_seasonal_bias: bool = False,
) -> SubselectState:
    """Run every L1 derivation for *country* and return a populated state.

    Parameters
    ----------
    country
        Lowercase country name (e.g. ``"greece"``, ``"sweden"``). Must be
        present in ``Data/country_codes/country_codes.json`` and the GADM
        polygon must be available.
    scenarios
        SSP scenarios to include. Default: all four.
    only
        Restrict computation to a subset of artefact-group names
        (``"performance"``, ``"spread"``, ``"profile"``, ``"bias"``). Any not
        listed are skipped (and the corresponding state fields are empty).
    force
        If True, ignore cache and recompute every artefact.
    config
        Optional :class:`Config` override (default: ``Config.from_env()``).
    include_bias_maps
        Whether to build the per-model bias-map fields (cheap if monthly
        climatologies are already cached, expensive on a cold start).
    include_seasonal_bias
        Whether bias maps include the four seasons in addition to ``annual``.
    """
    config = config or Config.from_env()
    cache = Cache(country, config.cache_root)
    if force:
        cache.clear()

    only_set = set(only) if only else None

    def _enabled(group: str) -> bool:
        return only_set is None or group in only_set

    # --- annual time series + warming levels ------------------------------
    annual_ts: dict[str, pd.DataFrame] = {}
    if _enabled("profile"):
        annual_ts = annual_timeseries(country, config, cache)

    wl_models = wl_medians = pi_b = rp_b = None
    if _enabled("profile") and "tas" in annual_ts:
        wl_models, wl_medians, pi_b, rp_b = warming_levels(annual_ts["tas"], cache)

    # --- future anomalies (per-country + global) --------------------------
    tas_future = pr_future_pct = {}
    if _enabled("profile") and wl_models is not None:
        tas_future = tas_future_anomalies(
            annual_ts["tas"], wl_models, pi_b, rp_b, cache,
        )
        if "pr" in annual_ts:
            pr_future_pct = pr_future_percent_anomalies(
                annual_ts["pr"], wl_models, cache,
            )

    global_artefacts = global_comparison(config) if _enabled("profile") else {
        "warming_level_medians": pd.DataFrame(),
        "tas_future_anomalies": {"recent_past": pd.DataFrame(), "pre_industrial": pd.DataFrame()},
        "pr_future_percent_anomalies": {"recent_past": pd.DataFrame(), "pre_industrial": pd.DataFrame()},
    }

    # --- profile signals (cells 24, 26, 39, 41) ---------------------------
    profile_sig: ProfileSignals | None = None
    if (
        _enabled("profile")
        and "tas" in annual_ts and "pr" in annual_ts
        and tas_future and pr_future_pct
    ):
        proxy = [
            _annual_ts_proxy_path(cache, "tas"),
            _annual_ts_proxy_path(cache, "pr"),
        ]
        if cache.is_fresh("profile_signals", proxy):
            profile_sig = cache.load("profile_signals")
        else:
            profile_sig = profile_signals.build_profile_signals(
                annual_temperature=annual_ts["tas"],
                annual_precipitation=annual_ts["pr"],
                country=country,
                tas_future_anomalies=tas_future,
                pr_future_percent_anomalies=pr_future_pct,
                tas_future_anomalies_global=global_artefacts["tas_future_anomalies"],
                pr_future_percent_anomalies_global=global_artefacts["pr_future_percent_anomalies"],
                warming_levels_all_models=wl_models,
            )
            cache.save("profile_signals", profile_sig, deps=proxy)

    # --- performance + bias-maps fused pass -------------------------------
    perf_metrics: dict[str, pd.DataFrame] = {}
    composite: pd.DataFrame = pd.DataFrame()
    composite_full: pd.DataFrame = pd.DataFrame()
    obs_std: pd.DataFrame = pd.DataFrame()
    mon_means: dict[str, dict[str, pd.DataFrame]] = {}
    obs_maps: dict[str, dict[str, xr.Dataset]] = {}
    bias_maps_dict: dict[str, dict[str, dict[str, xr.DataArray]]] = {}
    if _enabled("performance"):
        perf_metrics, mon_means, obs_std, obs_maps, bias_maps_dict = (
            fused_performance_pass(
                country, config, cache,
                include_bias_maps=include_bias_maps,
                include_seasonal_bias=include_seasonal_bias,
            )
        )
        composite, composite_full = composite_hps(country, perf_metrics, config, cache)

    # --- spread pipeline --------------------------------------------------
    change_df = long_df = pi_df = pd.DataFrame()
    if _enabled("spread"):
        change_df, long_df, pi_df = spread(country, config, cache)

    # --- assemble state ---------------------------------------------------
    return SubselectState(
        country=country,
        cache_dir=cache.root,
        performance_metrics=perf_metrics,
        composite_hps=composite,
        composite_hps_full=composite_full,
        observed_std_dev=obs_std,
        monthly_means=mon_means,
        change_signals=change_df,
        long_term_spread=long_df,
        pre_industrial_spread=pi_df,
        annual_timeseries=annual_ts,
        warming_levels=wl_models if wl_models is not None else pd.DataFrame(),
        warming_level_medians=wl_medians if wl_medians is not None else pd.DataFrame(),
        warming_level_medians_global=global_artefacts["warming_level_medians"],
        future_anomalies={
            "tas": tas_future,
            "pr": pr_future_pct,
        },
        future_anomalies_global={
            "tas": global_artefacts["tas_future_anomalies"],
            "pr": global_artefacts["pr_future_percent_anomalies"],
        },
        profile_signals=profile_sig if profile_sig is not None else _empty_profile_signals(),
        observed_maps=obs_maps,
        bias_maps=bias_maps_dict,
    )


def _empty_profile_signals() -> ProfileSignals:
    """Placeholder for callers that opt out of the country-profile group."""
    empty_df = pd.DataFrame()
    empty_s = pd.Series(dtype=float)
    return ProfileSignals(
        annual_temperature=empty_df, tas_pi_baseline=empty_s, tas_rp_baseline=empty_s,
        tas_baseline_offset=0.0, tas_anomaly_pi=empty_df, tas_anomaly_rp=empty_df,
        stats_tas_anomaly_pi=empty_df, stats_tas_anomaly_rp=empty_df,
        annual_precipitation=empty_df, pr_pi_baseline=empty_s, pr_rp_baseline=empty_s,
        pr_smoothed=empty_df, pr_anomaly_pi=empty_df, pr_anomaly_rp=empty_df,
        stats_pr_anomaly_pi=empty_df, stats_pr_anomaly_rp=empty_df,
        pr_baseline_offset=0.0, pr_baseline_offset_percent=0.0, pr_ax_ratio=1.0,
        pr_pi_percent_change=empty_df, pr_rp_percent_change=empty_df,
        stats_pr_pi_percent_change=empty_df, stats_pr_rp_percent_change=empty_df,
        tas_anomalies_table=empty_df, pr_percent_anom_table=empty_df,
    )
