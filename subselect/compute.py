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

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from subselect import compute_global as cg
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

def _annual_timeseries_one_from_global(
    variable: str, model: str, scenario: str, country: str,
    cache_root: Path, config: Config,
    crop_method: str = "bbox",
) -> tuple[str, pd.Series] | None:
    """Country-mean annual time-series for one (variable, model, scenario)
    by cropping the cached global annual-mean field.

    Falls back to a direct CMIP6 NetCDF read if the global cache lacks the
    field (e.g. first-time use on a system without the global cache built).
    """
    from subselect.geom import crop
    from subselect.performance import (
        _normalise_time_to_first_of_month, _spatial_weighted_mean,
    )

    global_cache = Cache.global_cache(cache_root)
    key = cg.annual_field_key(variable, model, scenario)

    try:
        fpath = io.cmip6_path(variable, scenario, model, config=config)
    except FileNotFoundError:
        return None
    parts = fpath.stem.split("_")
    variant = parts[2] if len(parts) >= 4 else "r1i1p1f1"
    col_name = f"{variable}_{model}_{variant}_{scenario}_yr"

    if global_cache.has(key):
        annual = global_cache.load(key)
        cropped = crop(annual, country, method=crop_method, config=config).data
        series = pd.Series(
            {
                int(y): _spatial_weighted_mean(cropped.sel(year=y))
                for y in cropped.year.values
            }
        )
        return col_name, series

    # Cold-cache fallback: direct NetCDF path (used when global cache hasn't
    # been built yet — e.g. the first cold-cache run before compute_global).
    try:
        ds = io.load_cmip6(variable, scenario, model, config=config)
    except FileNotFoundError:
        return None
    if "height" in ds.coords:
        ds = ds.drop_vars("height")
    da = _normalise_time_to_first_of_month(ds[variable])
    annual = da.groupby("time.year").mean("time")
    cropped = crop(annual, country, method=crop_method, config=config).data
    series = pd.Series(
        {
            int(y): _spatial_weighted_mean(cropped.sel(year=y))
            for y in cropped.year.values
        }
    )
    ds.close()
    return col_name, series


def _build_annual_timeseries(
    variable: str, country: str, config: Config,
    *, n_jobs: int = DEFAULT_N_JOBS,
    crop_method: str = "bbox",
) -> pd.DataFrame:
    """Annual country-mean time-series 1850–2100 across all SSPs and models.

    Columns: ``<variable>_<MODEL>_<variant>_<scenario>_yr``. Index: year (int).
    Reads from the cached global annual-mean fields when available; the
    35-models × 4-scenarios loop runs in parallel via joblib loky.
    """
    models = io.load_models_list(config)
    jobs = [(model, scenario) for model in models for scenario in SCENARIOS]
    parallel = Parallel(n_jobs=n_jobs, backend=DEFAULT_BACKEND)
    results = parallel(
        delayed(_annual_timeseries_one_from_global)(
            variable, m, s, country, config.cache_root, config,
            crop_method=crop_method,
        )
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


def annual_timeseries(
    country: str, config: Config, cache: Cache,
    *, crop_method: str = "bbox",
) -> dict[str, pd.DataFrame]:
    """Build (or load from cache) the annual country-mean time-series for the
    three time-series variables (``tas``, ``pr``, ``psl``)."""
    out: dict[str, pd.DataFrame] = {}
    for variable in TIMESERIES_VARIABLES:
        key = f"annual_timeseries__{variable}"
        # The country annual TS consumes the cached global annual_field zarr,
        # so any global-cache rebuild must invalidate this entry.
        deps = _cmip6_inputs(variable, config) + _global_catalog_dep(config)
        if cache.is_fresh(key, deps):
            out[variable] = cache.load(key)
            continue
        df = _build_annual_timeseries(variable, country, config, crop_method=crop_method)
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


def _global_catalog_dep(config: Config) -> list[Path]:
    """Cross-scope invalidation hook: every per-country artefact that derives
    from a global-cache artefact must include this in its dep list. The
    global catalog's mtime advances whenever ``compute_global`` writes any
    artefact (saves rewrite ``cache/_global/catalog.json``), so any global
    rebuild auto-invalidates downstream per-country derivations.
    """
    return [config.cache_root / "_global" / "catalog.json"]


# ---------------------------------------------------------------------------
# Warming levels + future anomalies (cells 15, 19, 21)
# ---------------------------------------------------------------------------

def warming_levels(
    annual_temperature: pd.DataFrame, cache: Cache,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Crossing years per (model, scenario) and per-SSP medians."""
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
    """Per-SSP statistics over fixed future periods."""
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
    """Per-SSP percent-change statistics for precipitation."""
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
    country: str,
    crop_method: str,
    cache_root: Path,
    config: Config,
    obs_std_per_period: dict[str, float],
    bias_periods: list[str] | None,
) -> dict | None:
    """Single-pass per-(model, variable) worker, consuming the global cache.

    Loads cached model + obs climatologies + sigma_ref maps from
    ``cache/_global/`` and computes country-mean metric scalars, monthly
    spatial means, and per-period bias fields. No NetCDF reads here.
    """
    from subselect.geom import crop as crop_fn
    from subselect.performance import (
        EPS_DIVISION, PERIODS, SEASON_MONTHS,
        _pixel_metrics_for_period, _select_months, _spatial_weighted_mean,
    )

    global_cache = Cache.global_cache(cache_root)
    hist_key = cg.hist_clim_key(variable, model)
    obs_key = cg.obs_clim_key(variable, model)
    sigma_key = cg.sigma_ref_key(variable, model)
    if not (
        global_cache.has(hist_key)
        and global_cache.has(obs_key)
        and global_cache.has(sigma_key)
    ):
        return None

    mod_clim_global = global_cache.load(hist_key)
    obs_clim_global = global_cache.load(obs_key)
    sigma_ref_global = global_cache.load(sigma_key)

    mod_clim_full = crop_fn(mod_clim_global, country, method=crop_method, config=config).data
    obs_clim_full = crop_fn(obs_clim_global, country, method=crop_method, config=config).data
    sigma_ref_full = crop_fn(sigma_ref_global, country, method=crop_method, config=config).data

    metric_scalars: dict[str, float] = {}
    for period in PERIODS:
        months = SEASON_MONTHS[period]
        obs_clim_p = _select_months(obs_clim_full, months)
        mod_clim_p = _select_months(mod_clim_full, months)
        sigma_map_p = sigma_ref_full.sel(period=period)
        pixel_metrics = _pixel_metrics_for_period(obs_clim_p, mod_clim_p, sigma_map_p)
        for name, da in pixel_metrics.items():
            spatial_input = abs(da) if name == "bias" else da
            metric_scalars[f"{period}_{name}"] = _spatial_weighted_mean(spatial_input)
        obs_std = obs_std_per_period[period]
        a = metric_scalars[f"{period}_std_dev"] / max(obs_std, EPS_DIVISION)
        a = max(a, EPS_DIVISION)
        r = max(min(metric_scalars[f"{period}_corr"], 1.0), -1.0)
        metric_scalars[f"{period}_tss"] = 2.0 * (1.0 + r) / ((a + 1.0 / a) ** 2)
        metric_scalars[f"{period}_tss_hirota"] = ((1.0 + r) ** 4) / (4.0 * (a + 1.0 / a) ** 2)

    months_idx = list(range(1, 13))
    cmip_monthly = [_spatial_weighted_mean(mod_clim_full.sel(month=m)) for m in months_idx]
    obs_monthly = [_spatial_weighted_mean(obs_clim_full.sel(month=m)) for m in months_idx]

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
    """Run every per-model country-mean derivation against the cached global
    climatologies. No NetCDF reads here — all input fields come from
    ``cache/_global/``.

    Returns ``(performance_metrics, monthly_means, observed_std_dev,
    observed_maps, bias_maps)``.
    """
    from subselect.geom import crop as crop_fn
    from subselect.performance import (
        PER_PERIOD_METRIC_COLUMNS, PERIODS, SEASON_MONTHS,
        _select_months, _spatial_weighted_mean,
    )

    global_cache = Cache.global_cache(config.cache_root)
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
        deps = [
            global_cache.root / "catalog.json",
        ]

        perf_key = f"performance_metrics__{variable}"
        cmip_key = f"monthly_means__{variable}__cmip6"
        obs_key = f"monthly_means__{variable}__obs"
        tabular_cached = (
            cache.is_fresh(perf_key, deps)
            and cache.is_fresh(cmip_key, deps)
            and cache.is_fresh(obs_key, deps)
        )

        # σ_obs scalars: cropped from cached native-resolution climatology +
        # std-over-months. The σ_obs scalar that feeds the TSS denominator is
        # the *climatology* std (variability across the 12 monthly means for a
        # given period), not the interannual std. Computing it from the cached
        # climatology preserves the legacy convention.
        nso_key = cg.native_sigma_obs_key(variable)
        obs_std_per_period: dict[str, float] = {}
        if global_cache.has(nso_key):
            from subselect.performance import _scalar_obs_std

            nso = global_cache.load(nso_key)
            clim_global = nso["clim"]
            clim_country = crop_fn(
                clim_global, country, method=crop_method, config=config,
            ).data
            for period in PERIODS:
                obs_std_per_period[period] = _scalar_obs_std(
                    clim_country, SEASON_MONTHS[period],
                )
            obs_std_columns[variable] = [obs_std_per_period[p] for p in PERIODS]

            # Observed-maps top panel: cropped from cached native climatology
            if include_bias_maps:
                for period in bias_periods:
                    sel = _select_months(clim_country, SEASON_MONTHS[period])
                    obs_maps[period][variable] = (
                        sel.mean(dim="month").to_dataset(name=variable)
                    )

        # Skip the per-model pass when every tabular artefact is cached and the
        # caller didn't ask for bias maps (which are not cached on disk).
        if tabular_cached and not include_bias_maps:
            perf[variable] = cache.load(perf_key)
            mm[variable] = {"cmip6": cache.load(cmip_key), "obs": cache.load(obs_key)}
            continue

        results = parallel(
            delayed(_fused_per_model)(
                model=m, variable=variable,
                country=country, crop_method=crop_method,
                cache_root=config.cache_root, config=config,
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
    # σ_obs is derived from the cached native_sigma_obs global artefact.
    # Including the global catalog as a dep means any global-cache rebuild
    # (or invalidation) auto-refreshes σ_obs and everything downstream of it.
    obs_std_deps = (
        sum((_reference_inputs(v, config) for v in ALL_VARIABLES), [])
        + _global_catalog_dep(config)
    )
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
    *, crop_method: str = "bbox",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Composite HPS table and its full ranked variant for one country.

    Returns ``(hps, full)`` where ``hps`` carries one ``<period>_HMperf``
    column per period in ``annual / DJF / MAM / JJA / SON`` and ``full``
    additionally exposes per-period rank, min-max-normalised TSS and
    bias-score, and the raw composite bias-score. Both tables are
    cached and only recomputed when the underlying CMIP6 / W5E5 inputs
    or the global catalog have changed.
    """
    from subselect.performance import (
        EPS_HPS, PERIODS, compute_hps, minmax_normalize,
    )

    deps = (
        sum((_cmip6_inputs(v, config) + _reference_inputs(v, config) for v in HPS_VARIABLES), [])
        + _global_catalog_dep(config)
    )

    key_hps = "composite_hps"
    if cache.is_fresh(key_hps, deps):
        hps = cache.load(key_hps)
    else:
        hps = compute_hps(country, config=config, crop_method=crop_method).rename(
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


def observed_std_dev(
    country: str, config: Config, cache: Cache,
    *, crop_method: str = "bbox",
) -> pd.DataFrame:
    """σ_obs (per variable, per period) for the country, on the single grid.

    Used as the σ in the TSS denominator. One row per variable, one column
    per period (``annual / DJF / MAM / JJA / SON``). Cached and refreshed
    when the W5E5 inputs change.
    """
    from subselect.performance import PERIODS, _compute_obs_std_per_period

    deps = (
        sum((_reference_inputs(v, config) for v in ALL_VARIABLES), [])
        + _global_catalog_dep(config)
    )
    key = "observed_std_dev"
    if cache.is_fresh(key, deps):
        return cache.load(key)
    sigmas = {
        v: _compute_obs_std_per_period(v, country, crop_method, config) for v in ALL_VARIABLES
    }
    df = pd.DataFrame(
        {v: [sigmas[v][p] for p in PERIODS] for v in ALL_VARIABLES},
        index=list(PERIODS),
    )
    cache.save(key, df, deps=deps)
    return df


def _monthly_means_one(
    model: str, variable: str, country: str, config: Config,
    *, crop_method: str = "bbox",
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
            country=country, crop_method=crop_method, config=config,
        )
    except FileNotFoundError:
        return None
    months = list(range(1, 13))
    cmip_means = [_spatial_weighted_mean(mod_clim.sel(month=m)) for m in months]
    obs_means = [_spatial_weighted_mean(obs_clim.sel(month=m)) for m in months]
    return model, cmip_means, obs_means


def monthly_means(
    country: str, config: Config, cache: Cache,
    *, n_jobs: int = DEFAULT_N_JOBS,
    crop_method: str = "bbox",
) -> dict[str, dict[str, pd.DataFrame]]:
    """Per-variable spatial-mean monthly-cycle climatology for CMIP6 models +
    observations. ``{var: {'cmip6': df, 'obs': df}}``.

    The 35-models × 4-vars loop runs in parallel via joblib loky.
    """
    months = list(range(1, 13))
    out: dict[str, dict[str, pd.DataFrame]] = {}
    parallel = Parallel(n_jobs=n_jobs, backend=DEFAULT_BACKEND)
    for variable in ALL_VARIABLES:
        deps = (
            _cmip6_inputs(variable, config)
            + _reference_inputs(variable, config)
            + _global_catalog_dep(config)
        )
        cmip_key = f"monthly_means__{variable}__cmip6"
        obs_key = f"monthly_means__{variable}__obs"
        if cache.is_fresh(cmip_key, deps) and cache.is_fresh(obs_key, deps):
            out[variable] = {
                "cmip6": cache.load(cmip_key),
                "obs": cache.load(obs_key),
            }
            continue
        results = parallel(
            delayed(_monthly_means_one)(m, variable, country, config, crop_method=crop_method)
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

def _period_means_from_cached_clim(
    clim: xr.DataArray, country: str, config: Config, *, crop_method: str = "bbox",
    box_offset: float = 1.5,
) -> dict[str, float]:
    """cos(lat)-weighted spatial mean of a cached climatology, then per-period
    reduction. Used by the spread pipeline against pre-cropped climatologies
    pulled from the global cache."""
    from subselect.geom import crop as crop_fn
    from subselect.performance import PERIODS, SEASON_MONTHS, _spatial_weighted_mean

    cropped = crop_fn(
        clim, country, method=crop_method, box_offset=box_offset, config=config,
    ).data
    monthly_means = {
        int(m): _spatial_weighted_mean(cropped.sel(month=m))
        for m in cropped.month.values
    }
    out: dict[str, float] = {}
    out["annual"] = float(np.mean([monthly_means[m] for m in range(1, 13)]))
    for period in ("DJF", "MAM", "JJA", "SON"):
        out[period] = float(
            np.mean([monthly_means[m] for m in SEASON_MONTHS[period]])
        )
    return out


def spread(
    country: str, config: Config, cache: Cache,
    *, crop_method: str = "bbox",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-of-century change signals + long-term + pre-industrial spread tables.

    Reads from the cached global EoC and PI climatologies (zarr per (variable,
    model[, scenario])). No NetCDF reads here.
    """
    from subselect.performance import PERIODS
    from subselect.spread import SPREAD_VARIABLES, SPREAD_BOX_OFFSET

    global_cache = Cache.global_cache(config.cache_root)
    deps = [global_cache.root / "catalog.json"]

    key_change = "change_signals"
    key_long = "long_term_spread"
    key_pi = "pre_industrial_spread"
    if (
        cache.is_fresh(key_change, deps)
        and cache.is_fresh(key_long, deps)
        and cache.is_fresh(key_pi, deps)
    ):
        return cache.load(key_change), cache.load(key_long), cache.load(key_pi)

    models = io.load_models_list(config)
    columns = [f"{v}_{p}" for v in SPREAD_VARIABLES for p in PERIODS]
    long_df = pd.DataFrame(index=models, columns=columns, dtype=float)
    pi_df = pd.DataFrame(index=models, columns=columns, dtype=float)
    change_df = pd.DataFrame(index=models, columns=columns, dtype=float)

    for variable in SPREAD_VARIABLES:
        for model in models:
            eoc_key = cg.eoc_clim_key(variable, model, "ssp585")
            pi_key = cg.pi_clim_key(variable, model)
            if not global_cache.has(eoc_key):
                continue
            eoc_clim = global_cache.load(eoc_key)
            lt = _period_means_from_cached_clim(
                eoc_clim, country, config,
                crop_method=crop_method, box_offset=SPREAD_BOX_OFFSET,
            )
            for p in PERIODS:
                long_df.loc[model, f"{variable}_{p}"] = lt[p]
            if global_cache.has(pi_key):
                pi_clim = global_cache.load(pi_key)
                pi = _period_means_from_cached_clim(
                    pi_clim, country, config,
                    crop_method=crop_method, box_offset=SPREAD_BOX_OFFSET,
                )
                for p in PERIODS:
                    pi_df.loc[model, f"{variable}_{p}"] = pi[p]
                    change_df.loc[model, f"{variable}_{p}"] = lt[p] - pi[p]

    cache.save(key_change, change_df, deps=deps)
    cache.save(key_long, long_df, deps=deps)
    cache.save(key_pi, pi_df, deps=deps)
    return change_df, long_df, pi_df


# ---------------------------------------------------------------------------
# Bias maps
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
# Global comparison artefacts (read-only; precomputed xlsx)
# ---------------------------------------------------------------------------

def global_comparison(config: Config) -> dict[str, dict[str, pd.DataFrame] | pd.DataFrame]:
    """Read the precomputed global-mean comparison artefacts.

    The country-profile figures compare a country's projections to the
    global ensemble. The global tables live under ``results/global/`` and
    are read directly. If a future iteration computes the global side from
    raw CMIP6, this function becomes a builder; for now it is a passthrough
    loader.
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

_FORCE_LITERALS = ("country", "global", "all")


def compute(
    country: str,
    *,
    scenarios: tuple[str, ...] = SCENARIOS,
    only: Iterable[str] | None = None,
    force: bool | str = False,
    config: Config | None = None,
    include_bias_maps: bool = True,
    include_seasonal_bias: bool = False,
    crop_method: str = "bbox",
) -> SubselectState:
    """Run every L1 derivation for *country* and return a populated state.

    Populates the global cache (`cache/_global/`) on first call, then
    consumes it for every per-country derivation. Subsequent country runs
    skip the global compute entirely.

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
        If ``True`` or ``"all"``, ignore both caches and recompute everything.
        ``"country"`` rebuilds only the per-country cache (global stays).
        ``"global"`` rebuilds only the global cache (per-country stays).
        ``False`` (default) consults both caches normally.
    config
        Optional :class:`Config` override (default: ``Config.from_env()``).
    include_bias_maps
        Whether to build the per-model bias-map fields (cheap if monthly
        climatologies are already cached, expensive on a cold start).
    include_seasonal_bias
        Whether bias maps include the four seasons in addition to ``annual``.
    crop_method
        Country-cropping rule. One of ``"bbox"`` (default),
        ``"shapefile_strict"``, ``"shapefile_lenient"``,
        ``"shapefile_fractional"``. Per-country cache entries are keyed by
        ``crop_method`` so two runs with different methods do not collide.
        The global cache is unaffected.

    Returns
    -------
    SubselectState
        Populated typed state. Hand it to :func:`subselect.render.render`
        to write the figure set under ``results/<country>/figures/``.

    Examples
    --------
    .. code:: python

        from subselect.compute import compute
        from subselect.render import render
        state = compute("greece")
        render(state)
    """
    config = config or Config.from_env()
    cache = Cache(country, config.cache_root, crop_method=crop_method)

    force_global = force is True or force == "all" or force == "global"
    force_country = force is True or force == "all" or force == "country"
    if force_country:
        cache.clear()
    cg.compute_global(config=config, force=force_global)

    only_set = set(only) if only else None

    def _enabled(group: str) -> bool:
        return only_set is None or group in only_set

    # --- annual time series + warming levels ------------------------------
    annual_ts: dict[str, pd.DataFrame] = {}
    if _enabled("profile"):
        annual_ts = annual_timeseries(country, config, cache, crop_method=crop_method)

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
                crop_method=crop_method,
            )
        )
        composite, composite_full = composite_hps(
            country, perf_metrics, config, cache, crop_method=crop_method,
        )

    # --- spread pipeline --------------------------------------------------
    change_df = long_df = pi_df = pd.DataFrame()
    if _enabled("spread"):
        change_df, long_df, pi_df = spread(country, config, cache, crop_method=crop_method)

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
