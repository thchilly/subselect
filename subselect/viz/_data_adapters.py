"""Adapters: cache → notebook-shaped variables.

Each adapter reproduces the in-cell namespace a legacy notebook builds by the
time the figure-generating cell runs. Verbatim cell ports (in
``performance_figs.py`` / ``spread_figs.py`` / ``country_profile.py``) consume
adapter outputs directly — same variable names, same shapes, same dtypes.

Adapters source data from, in order of preference:
1. M7/M8 cache parquets under ``cache/parquet/<country>/...`` (single source of
   truth post-refactor).
2. Paper-era xlsx artefacts under ``results/<country>/`` (canonical inputs the
   paper-era notebooks read).

The M9 ports route through (2) for any frame the M7/M8 cache doesn't yet
persist; this is the data-input deviation the brief permits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pandas as pd

from subselect.config import Config


# --------------------------------------------------------------------------
# Performance — ranked_full + model_ids + per-variable metric/cycle frames
# --------------------------------------------------------------------------

def load_ranked_full(country: str = "greece", *, config: Config | None = None) -> pd.DataFrame:
    """Return the per-(model, period) HPS frame the legacy HPS notebook calls
    ``ranked_full``.

    Columns: 21 = ``{period}_{rank,HMperf,TSS_mm,bias_score_mm,bias_score_raw}``
    for ``period ∈ {annual, DJF, MAM, JJA, SON}``. Index: model name.
    """
    config = config or Config.from_env()
    xlsx = Path(config.results_root) / country / f"assess_cmip6_composite_HMperf_full_{country}.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(
            f"{xlsx} missing — run the M7 pipeline (or copy the paper-era artefact) first"
        )
    return pd.read_excel(xlsx, index_col=0)


def load_model_ids(*, config: Config | None = None) -> dict:
    """Return the canonical ``{model_name: int_id}`` dict the legacy notebooks
    build via ``cmip6_models = pd.read_excel(... CMIP6_model_id.xlsx); model_ids = dict(zip(...))``.
    """
    config = config or Config.from_env()
    xlsx = Path(config.cmip6_metadata_root) / "CMIP6_model_id.xlsx"
    cmip6_models = pd.read_excel(xlsx)
    return dict(zip(cmip6_models["model"], cmip6_models["id"]))


def load_cmip6_models(*, config: Config | None = None) -> pd.DataFrame:
    """Return the raw ``CMIP6_model_id.xlsx`` DataFrame (columns ``id``, ``model``)."""
    config = config or Config.from_env()
    return pd.read_excel(Path(config.cmip6_metadata_root) / "CMIP6_model_id.xlsx")


def load_perf_metrics(variable: str, country: str = "greece", *, config: Config | None = None) -> pd.DataFrame:
    """Per-variable ``{<var>}_all_perf_metrics`` DataFrame.

    Columns include ``annual_std_dev``, ``annual_corr``, ``annual_bias``,
    ``annual_rmse``, ``annual_tss``, ``annual_bias_score`` and the same five
    per ``{DJF, MAM, JJA, SON}``. Index: model name.
    """
    config = config or Config.from_env()
    xlsx = Path(config.results_root) / country / f"assess_cmip6_{variable}_mon_perf_metrics_all_seasons_{country}.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(f"{xlsx} missing")
    return pd.read_excel(xlsx, index_col=0)


def load_perf_metrics_dict(variables=("tas", "pr", "psl", "tasmax"), country: str = "greece", *, config: Config | None = None) -> Dict[str, pd.DataFrame]:
    """Build the ``perf_metrics`` dict the legacy Taylor / bias cells expect."""
    return {var: load_perf_metrics(var, country, config=config) for var in variables}


def load_observed_std_dev_df(country: str = "greece", *, config: Config | None = None) -> pd.DataFrame:
    """Index=['annual','DJF','MAM','JJA','SON']; columns=['tas','pr','psl','tasmax']."""
    config = config or Config.from_env()
    xlsx = Path(config.results_root) / country / f"assess_observed_std_dev_{country}.xlsx"
    return pd.read_excel(xlsx, index_col=0)


def load_mon_means(variable: str, country: str = "greece", *, config: Config | None = None):
    """Return ``(observed_mon_means, cmip6_mon_means)`` DataFrames.

    Observed: index=months 1..12, columns=[variable].
    CMIP6:    index=months 1..12, columns=model names (35).
    """
    config = config or Config.from_env()
    base = Path(config.results_root) / country
    obs = pd.read_excel(base / f"assess_{variable}_observed_mon_means_{country}.xlsx", index_col=0)
    mod = pd.read_excel(base / f"assess_{variable}_cmip6_mon_means_{country}.xlsx", index_col=0)
    return obs, mod


# --------------------------------------------------------------------------
# Spread — change-signal + spread artefacts
# --------------------------------------------------------------------------

def load_long_term_spread(country: str = "greece", *, config: Config | None = None) -> pd.DataFrame:
    """End-of-century spatial-mean climatology per (model, variable, period)."""
    config = config or Config.from_env()
    return pd.read_excel(
        Path(config.results_root) / country / f"assess_long_term_spread_{country}.xlsx",
        index_col=0,
    )


def load_long_term_change_spread(country: str = "greece", *, config: Config | None = None) -> pd.DataFrame:
    """End-of-century change relative to recent past per (model, variable, period)."""
    config = config or Config.from_env()
    return pd.read_excel(
        Path(config.results_root) / country / f"assess_long_term_change_spread_{country}.xlsx",
        index_col=0,
    )


def load_pre_industrial_spread(country: str = "greece", *, config: Config | None = None) -> pd.DataFrame:
    config = config or Config.from_env()
    return pd.read_excel(
        Path(config.results_root) / country / f"assess_pre_industrial_spread_{country}.xlsx",
        index_col=0,
    )


# --------------------------------------------------------------------------
# Country profile — analysis_path + base_path resolution + warming-level frames
# --------------------------------------------------------------------------

def country_analysis_path(country: str = "greece", *, config: Config | None = None) -> str:
    """Return the absolute string path the legacy CLIMPACT_figures cells refer
    to as ``analysis_path``. Maps to ``<results_root>/<country>``.
    """
    config = config or Config.from_env()
    return str(Path(config.results_root) / country)


def country_base_path(*, config: Config | None = None) -> str:
    """Resolve the legacy ``base_path``. The legacy cells use it via
    ``os.path.join(base_path, 'analysis', 'global', ...)``; the post-refactor
    layout exposes the equivalent under ``<results_root>``. We return a path
    such that ``<base_path>/analysis/global/`` resolves to ``<results_root>/global``
    via a transparent symlink (created on first use).
    """
    config = config or Config.from_env()
    results = Path(config.results_root)
    # Create a transparent shim: results/analysis -> .  (so .../analysis/global → .../global)
    shim = results / "analysis"
    if not shim.exists():
        try:
            shim.symlink_to(".", target_is_directory=True)
        except (OSError, FileExistsError):
            pass
    # base_path is one level above 'analysis' — use results_root.parent
    # Actually, the legacy joins base_path/analysis/global/X.xlsx; with our
    # symlink, results_root/analysis -> results_root/, so the join becomes
    # results_root/analysis/global/X.xlsx == results_root/./global/X.xlsx,
    # which is correct. Return results_root.
    return str(results)


def load_warming_levels_all_models(country: str = "greece", *, config: Config | None = None) -> pd.DataFrame:
    config = config or Config.from_env()
    return pd.read_excel(
        Path(config.results_root) / country / f"cmip6_warming_levels_all_models_{country}.xlsx",
        index_col=0,
    )


def load_warming_level_medians(country: str = "greece", *, config: Config | None = None) -> pd.DataFrame:
    config = config or Config.from_env()
    return pd.read_excel(
        Path(config.results_root) / country / f"cmip6_warming_levels_median_{country}.xlsx",
        index_col=0,
    )


def load_warming_level_medians_global(*, config: Config | None = None) -> pd.DataFrame:
    config = config or Config.from_env()
    return pd.read_excel(
        Path(config.results_root) / "global" / "cmip6_warming_levels_median_global.xlsx",
        index_col=0,
    )


def load_yr_all_models(variable: str, country: str = "greece", *, config: Config | None = None) -> pd.DataFrame:
    """Annual time-series 1850–2100 across SSPs per model."""
    config = config or Config.from_env()
    return pd.read_excel(
        Path(config.results_root) / country / f"cmip6_{variable}_yr_all_models_{country}.xlsx",
        index_col=0,
    )


def load_future_anomalies(variable: str, baseline: str, country: str = "greece", *, config: Config | None = None) -> pd.DataFrame:
    """``baseline`` is one of ``pre_industrial`` / ``recent_past``."""
    config = config or Config.from_env()
    if variable in ("pr",):
        fname = f"cmip6_{variable}_future_percent_anomalies_{baseline}_{country}.xlsx"
    else:
        fname = f"cmip6_{variable}_future_anomalies_{baseline}_{country}.xlsx"
    return pd.read_excel(Path(config.results_root) / country / fname, index_col=0)


# --------------------------------------------------------------------------
# Country-profile state — runs CLIMPACT_figures derivation cells verbatim
# --------------------------------------------------------------------------

def build_climpact_state(country: str = "greece", *, config: Config | None = None) -> dict:
    """Return the namespace the CLIMPACT_figures.ipynb derivation cells build.

    Runs cells [13, 15, 19, 21, 24, 26, 39, 41] from
    ``legacy/climpact/CLIMPACT_figures.ipynb`` verbatim against the paper-era
    xlsx artefacts under ``results/<country>/``. Returns the resulting namespace
    so the figure-generating cells (29, 31, 32, 35, 42, 44, 46, 49, 50, 54) can
    consume the legacy-named variables: ``stats_tas_21yr_ma_rp_anomaly``,
    ``tas_21yr_ma_rp_anomaly``, ``stats_pr_21yr_ma_rp_anomaly``,
    ``pr_21yr_ma_pi_percent_change``, ``temperature_pi_baseline``,
    ``annual_precipitation``, ``tas_anomalies_table``, ``pr_percent_anom_table``,
    ``pr_baseline_offset``, ``ax_ratio``, etc.

    ``DataFrame.to_excel`` is monkey-patched to a no-op for the duration so the
    paper-era xlsx artefacts under ``results/<country>/`` are not clobbered.
    """
    import json
    import sys

    import numpy as np

    config = config or Config.from_env()
    repo_root = Path(config.results_root).parent

    # Make the legacy `from functions import *` resolvable
    sys.path.insert(0, str(repo_root / "legacy" / "climpact"))

    # Symlink: legacy `os.path.join(base_path, 'analysis', 'global', X)` resolves
    shim = Path(config.results_root) / "analysis"
    if not shim.exists():
        try:
            shim.symlink_to(".", target_is_directory=True)
        except (OSError, FileExistsError):
            pass

    analysis_path = country_analysis_path(country, config=config)
    base_path = str(Path(config.results_root)) + "/"

    ns = {
        "__builtins__": __builtins__,
        "pd": pd, "np": np, "os": os,
        "country": country,
        "analysis_path": analysis_path,
        "base_path": base_path,
        "ssp_scenarios": ['ssp126', 'ssp245', 'ssp370', 'ssp585'],
        "warming_levels": {'WL_+1.5°C': 1.5, 'WL_+2.0°C': 2.0, 'WL_+3.0°C': 3.0, 'WL_+4.0°C': 4.0},
        "variables": ['tas', 'pr', 'psl'],
    }

    _orig_to_excel = pd.DataFrame.to_excel
    pd.DataFrame.to_excel = lambda self, *a, **k: None
    try:
        nb_path = repo_root / "legacy" / "climpact" / "CLIMPACT_figures.ipynb"
        nb = json.loads(nb_path.read_text())
        for cell_idx in [13, 15, 19, 21, 24, 26, 39, 41]:
            src = "".join(nb["cells"][cell_idx].get("source", []))
            exec(compile(src, f"<climpact_cell_{cell_idx}>", "exec"), ns)
    finally:
        pd.DataFrame.to_excel = _orig_to_excel

    return ns


# --------------------------------------------------------------------------
# Bias maps — observed_maps + bias_maps adapter for cell 34
# --------------------------------------------------------------------------

def build_bias_maps_state(
    country: str = "greece",
    *,
    scenario: str = "ssp585",
    crop_method: str = "bbox",
    include_seasonal: bool = False,
    config: Config | None = None,
) -> dict:
    """Return ``{'observed_maps': {...}, 'bias_maps': {...}}`` matching the
    legacy notebook's cell 5 dicts.

    - ``observed_maps[period][variable]`` is an ``xr.Dataset`` with the variable
      as a DataArray (country-cropped, period-averaged). Sourced from
      ``load_single_grid_w5e5`` (the same single-grid obs that backs M7's σ_obs).
    - ``bias_maps[period][variable][model]`` is an ``xr.DataArray`` (country-
      cropped, period-averaged ``model − obs_upscaled_to_model_grid``). Built
      via the same per-model climatology pipeline M7 uses for HPS.

    ``period`` is one of ``annual / DJF / MAM / JJA / SON``. With
    ``include_seasonal=False`` (default) only ``annual`` is computed (matching
    cell 34's verbatim ``for period in ['annual']:`` loop). With
    ``include_seasonal=True`` all five periods are computed.

    Each (model, variable) is reused across periods, so a single
    ``_model_obs_climatologies`` call produces the 12-month climatology that
    every period selects from.
    """
    import xarray as xr

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

    config = config or Config.from_env()
    periods = list(PERIODS) if include_seasonal else ["annual"]
    variables = ["tas", "pr", "psl", "tasmax"]
    models = io.load_models_list(config)

    observed_maps: dict[str, dict[str, xr.Dataset]] = {p: {} for p in periods}
    bias_maps: dict[str, dict[str, dict[str, xr.DataArray]]] = {
        p: {v: {} for v in variables} for p in periods
    }

    for variable in variables:
        # observed_maps: native 0.5° W5E5 (NOT the cmip6-grid upscaled product;
        # the legacy cell 5 reads from `ISIMIP3a/monthly/<var>_*.nc` at native
        # resolution for the bias-map top-panel display, lines 50–60).
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

        # bias_maps: per-model (mod − upscaled_obs) climatology, period-averaged.
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
                bias_maps[period][variable][model] = (mod_p - obs_p).mean(dim="month")

    return {"observed_maps": observed_maps, "bias_maps": bias_maps}
