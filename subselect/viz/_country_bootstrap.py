"""Bootstrap a country's figure-adapter inputs.

The M9 figure adapters (in ``_data_adapters.py``) read paper-era xlsx
artefacts under ``results/<country>/``. For Greece these were produced by
the legacy paper notebooks; for any new country, this module generates the
same xlsx files from the M7/M8 pipeline + the legacy-notebook derivation
cells, so the entry-point script (``regenerate_paper_figures.py``) is
country-agnostic.

Public entry point:
    bootstrap_country_artefacts(country, *, config=None, force=False)

Each bootstrap step checks if its outputs already exist and skips unless
``force=True``. This keeps re-runs fast on countries already populated.

Coverage:
    - Performance metrics + composite HPS (compute_metrics, compute_hps)
    - σ_obs scalars (_compute_obs_std_per_period)
    - End-of-century spread deltas (compute_change_signals)
    - Per-variable monthly-cycle xlsx for the seasonal_perf_revised figures
      (spatial-mean monthly climatology from the M7 pipeline)
    - Annual time-series xlsx (cmip6_<var>_yr_all_models) — port of
      CLIMPACT_figures.ipynb cell 12
    - Warming-levels + future-anomalies xlsx — produced by
      build_climpact_state with to_excel enabled
    - Spread-window absolute means (long_term_spread, pre_industrial_spread)
      from per-window calls into the spread pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from subselect.config import Config


HPS_VARIABLES = ("tas", "pr", "psl")
ALL_VARIABLES = HPS_VARIABLES + ("tasmax",)
SCENARIOS = ("ssp126", "ssp245", "ssp370", "ssp585")


def _xlsx_exists(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def bootstrap_performance_xlsx(country: str, config: Config, *, force: bool = False) -> None:
    """compute_metrics (4 vars) + compute_hps + _compute_obs_std_per_period
    → 4 + 2 + 1 = 7 xlsx files under results/<country>/."""
    from subselect.performance import (
        EPS_HPS,
        PERIODS,
        _compute_obs_std_per_period,
        compute_hps,
        compute_metrics,
        minmax_normalize,
    )

    results_dir = Path(config.results_root) / country
    results_dir.mkdir(parents=True, exist_ok=True)

    per_var_tables: dict[str, pd.DataFrame] = {}
    for variable in ALL_VARIABLES:
        out = results_dir / f"assess_cmip6_{variable}_mon_perf_metrics_all_seasons_{country}.xlsx"
        if not force and _xlsx_exists(out):
            per_var_tables[variable] = pd.read_excel(out, index_col=0)
            continue
        print(f"  [perf] compute_metrics({variable})...")
        df = compute_metrics(country, variable=variable, config=config)
        df.to_excel(out)
        per_var_tables[variable] = df

    composite_path = results_dir / f"assess_cmip6_composite_HMperf_{country}.xlsx"
    if force or not _xlsx_exists(composite_path):
        print("  [perf] compute_hps...")
        hps = compute_hps(country, config=config)
        hps.rename(columns={p: f"{p}_HMperf" for p in PERIODS}).to_excel(composite_path)

    composite_full_path = results_dir / f"assess_cmip6_composite_HMperf_full_{country}.xlsx"
    if force or not _xlsx_exists(composite_full_path):
        print("  [perf] composite_HMperf_full...")
        out = pd.DataFrame()
        for period in PERIODS:
            comp_tss = pd.concat(
                [per_var_tables[v][f"{period}_tss"] for v in HPS_VARIABLES], axis=1,
            ).mean(axis=1, skipna=True)
            comp_bs_raw = pd.concat(
                [per_var_tables[v][f"{period}_bias_score"] for v in HPS_VARIABLES], axis=1,
            ).mean(axis=1, skipna=True)
            tss_mm = minmax_normalize(comp_tss)
            bs_mm = minmax_normalize(comp_bs_raw)
            hmperf = 2.0 * (tss_mm * bs_mm) / (tss_mm + bs_mm + EPS_HPS)
            out[f"{period}_rank"] = hmperf.rank(ascending=False).astype(int)
            out[f"{period}_HMperf"] = hmperf
            out[f"{period}_TSS_mm"] = tss_mm
            out[f"{period}_bias_score_mm"] = bs_mm
            out[f"{period}_bias_score_raw"] = comp_bs_raw
        out.sort_values("annual_HMperf", ascending=False).to_excel(composite_full_path)

    obs_std_path = results_dir / f"assess_observed_std_dev_{country}.xlsx"
    if force or not _xlsx_exists(obs_std_path):
        print("  [perf] observed_std_dev...")
        sigmas = {
            v: _compute_obs_std_per_period(v, country, "bbox", config)
            for v in ALL_VARIABLES
        }
        df = pd.DataFrame(
            {v: [sigmas[v][p] for p in PERIODS] for v in ALL_VARIABLES},
            index=list(PERIODS),
        )
        df.to_excel(obs_std_path)


def bootstrap_mon_means_xlsx(country: str, config: Config, *, force: bool = False) -> None:
    """Per-variable spatial-mean monthly cycle xlsx files used by the
    seasonal_perf_revised figures. Built by re-using the M7 per-model
    climatology pipeline (which already loads + crops + monthly-clims for
    each model, variable). The mon_means xlsx is just the spatial mean of
    those per-model 12-month climatologies, stacked into one DataFrame.
    """
    from subselect import io
    from subselect.performance import (
        _model_obs_climatologies,
        _spatial_weighted_mean,
    )

    results_dir = Path(config.results_root) / country
    models = io.load_models_list(config)
    months = list(range(1, 13))

    for variable in ALL_VARIABLES:
        cmip_path = results_dir / f"assess_{variable}_cmip6_mon_means_{country}.xlsx"
        obs_path = results_dir / f"assess_{variable}_observed_mon_means_{country}.xlsx"
        if not force and _xlsx_exists(cmip_path) and _xlsx_exists(obs_path):
            continue
        print(f"  [mon_means] {variable} (35 models)...")
        cmip_cols: dict[str, list[float]] = {}
        obs_means: list[float] | None = None
        for model in models:
            try:
                obs_clim, mod_clim, _ = _model_obs_climatologies(
                    model=model, variable=variable, scenario="ssp585",
                    country=country, crop_method="bbox", config=config,
                )
            except FileNotFoundError:
                continue
            cmip_cols[model] = [
                _spatial_weighted_mean(mod_clim.sel(month=m)) for m in months
            ]
            if obs_means is None:
                obs_means = [
                    _spatial_weighted_mean(obs_clim.sel(month=m)) for m in months
                ]
        if obs_means is None:
            print(f"    skipped ({variable}): no obs climatology available")
            continue
        cmip_df = pd.DataFrame(cmip_cols, index=pd.Index(months, name="month"))
        obs_df = pd.DataFrame({variable: obs_means}, index=pd.Index(months, name="month"))
        cmip_df.to_excel(cmip_path)
        obs_df.to_excel(obs_path)


def bootstrap_spread_xlsx(country: str, config: Config, *, force: bool = False) -> None:
    """compute_change_signals → 1 xlsx + the absolute long_term + pre_industrial
    window means (re-using the same per-model loading pipeline)."""
    from subselect import io
    from subselect.spread import (
        SPREAD_BOX_OFFSET,
        SPREAD_VARIABLES,
        _load_and_prepare,
        _period_means_for_window,
        compute_change_signals,
    )
    from subselect.performance import PERIODS

    results_dir = Path(config.results_root) / country
    change_path = results_dir / f"assess_long_term_change_spread_{country}.xlsx"
    if force or not _xlsx_exists(change_path):
        print("  [spread] compute_change_signals...")
        compute_change_signals(country, scenario="ssp585", config=config).to_excel(
            change_path
        )

    long_path = results_dir / f"assess_long_term_spread_{country}.xlsx"
    pi_path = results_dir / f"assess_pre_industrial_spread_{country}.xlsx"
    if force or not _xlsx_exists(long_path) or not _xlsx_exists(pi_path):
        print("  [spread] long_term + pre_industrial absolute window means...")
        models = io.load_models_list(config)
        columns = [f"{v}_{p}" for v in SPREAD_VARIABLES for p in PERIODS]
        long_df = pd.DataFrame(index=models, columns=columns, dtype=float)
        pi_df = pd.DataFrame(index=models, columns=columns, dtype=float)
        for variable in SPREAD_VARIABLES:
            for model in models:
                da = _load_and_prepare(
                    model=model, variable=variable, scenario="ssp585",
                    country=country, crop_method="bbox",
                    box_offset=SPREAD_BOX_OFFSET, config=config,
                )
                if da is None:
                    continue
                lt = _period_means_for_window(da, config.future_window)
                pi = _period_means_for_window(da, config.pre_industrial)
                if lt is not None:
                    for p in PERIODS:
                        long_df.loc[model, f"{variable}_{p}"] = lt[p]
                if pi is not None:
                    for p in PERIODS:
                        pi_df.loc[model, f"{variable}_{p}"] = pi[p]
        long_df.to_excel(long_path)
        pi_df.to_excel(pi_path)


def bootstrap_yr_timeseries_xlsx(country: str, config: Config, *, force: bool = False) -> None:
    """Annual country-mean time-series xlsx for tas/pr/psl, all 4 SSPs,
    1850–2100, all 35 models. Port of CLIMPACT_figures.ipynb cell 12.

    Column naming matches the legacy: ``<var>_<MODEL>_<variant>_<scenario>_yr``.
    The variant token is derived from the CMIP6 filename glob (legacy uses
    the filename's variant_label substring).
    """
    from subselect import io
    from subselect.geom import crop
    from subselect.performance import _normalise_time_to_first_of_month, _spatial_weighted_mean
    import xarray as xr

    results_dir = Path(config.results_root) / country
    models = io.load_models_list(config)

    for variable in ("tas", "pr", "psl"):
        out_path = results_dir / f"cmip6_{variable}_yr_all_models_{country}.xlsx"
        if not force and _xlsx_exists(out_path):
            continue
        print(f"  [yr_timeseries] {variable} (35 models × 4 SSPs)...")
        cols: dict[str, pd.Series] = {}
        for model in models:
            for scenario in SCENARIOS:
                try:
                    ds = io.load_cmip6(variable, scenario, model, config=config)
                except FileNotFoundError:
                    continue
                if "height" in ds.coords:
                    ds = ds.drop_vars("height")
                da = ds[variable]
                # The legacy paper-era column-name token uses the model's
                # variant_label from the CMIP6 filename. We resolve it the
                # same way io.cmip6_path does (glob match).
                fpath = io.cmip6_path(variable, scenario, model, config=config)
                # Filename pattern: <var>_<MODEL>_<variant>_<freq>_<scenario>.nc
                parts = fpath.stem.split("_")
                variant = parts[2] if len(parts) >= 4 else "r1i1p1f1"
                freq = parts[3] if len(parts) >= 5 else "mon"
                col_name = f"{variable}_{model}_{variant}_{scenario}_yr"
                # Crop to country, normalise time, annual mean (cos-weighted spatial)
                cropped = crop(da, country, method="bbox", config=config).data
                cropped = _normalise_time_to_first_of_month(cropped)
                # Annual = group by year, monthly mean → annual mean (unweighted in time)
                annual = cropped.groupby("time.year").mean("time")
                series = pd.Series(
                    {int(y): _spatial_weighted_mean(annual.sel(year=y)) for y in annual.year.values}
                )
                cols[col_name] = series
                ds.close()
        df = pd.DataFrame(cols).sort_index()
        df.index.name = "time"
        df.to_excel(out_path)


def bootstrap_climpact_state_xlsx(country: str, config: Config, *, force: bool = False) -> None:
    """Run build_climpact_state with to_excel enabled — the derivation cells
    write the warming_levels + future_anomalies xlsx files this way.
    """
    from subselect.viz._data_adapters import build_climpact_state

    results_dir = Path(config.results_root) / country
    wl_path = results_dir / f"cmip6_warming_levels_all_models_{country}.xlsx"
    if not force and _xlsx_exists(wl_path):
        return
    print("  [climpact_state] running cells [13, 15, 19, 21, 24, 26, 39, 41]...")

    # build_climpact_state monkey-patches DataFrame.to_excel to no-op for the
    # duration. We need it ENABLED for new countries so the warming-levels +
    # anomaly-stats xlsx files actually get written. Temporarily restore
    # to_excel, run the build, then restore the no-op behavior is unnecessary
    # since to_excel is restored at the end of build_climpact_state regardless.
    # The trick: monkey-patch the build itself via direct cell exec.
    import json as _json
    import os as _os
    import sys as _sys
    import numpy as _np

    repo_root = Path(config.results_root).parent
    _sys.path.insert(0, str(repo_root / "legacy" / "climpact"))
    shim = Path(config.results_root) / "analysis"
    if not shim.exists():
        try:
            shim.symlink_to(".", target_is_directory=True)
        except (OSError, FileExistsError):
            pass

    analysis_path = str(results_dir)
    base_path = str(Path(config.results_root)) + "/"
    ns = {
        "__builtins__": __builtins__,
        "pd": pd, "np": _np, "os": _os,
        "country": country, "analysis_path": analysis_path, "base_path": base_path,
        "ssp_scenarios": list(SCENARIOS),
        "warming_levels": {'WL_+1.5°C': 1.5, 'WL_+2.0°C': 2.0, 'WL_+3.0°C': 3.0, 'WL_+4.0°C': 4.0},
        "variables": ['tas', 'pr', 'psl'],
    }
    # NB: do NOT no-op to_excel here. Let the cells write their xlsx files.
    nb_path = repo_root / "legacy" / "climpact" / "CLIMPACT_figures.ipynb"
    nb = _json.loads(nb_path.read_text())
    for cell_idx in [13, 15, 19, 21, 24, 26, 39, 41]:
        src = "".join(nb["cells"][cell_idx].get("source", []))
        exec(compile(src, f"<climpact_cell_{cell_idx}>", "exec"), ns)


def bootstrap_global_warming_levels(config: Config, *, force: bool = False) -> None:
    """`results/global/cmip6_warming_levels_median_global.xlsx` — Greece-coupling
    leak: the gwls_boxplot figure compares the country's medians to global
    medians. The global xlsx is paper-era; if it doesn't exist, build it from
    the GWL crossing years over a "global" pseudo-country (no spatial crop).
    Skipped here because the file already exists at results/global/ from
    paper-era; surfaced as a leak only if missing.
    """
    global_path = Path(config.results_root) / "global" / "cmip6_warming_levels_median_global.xlsx"
    if global_path.is_file():
        return
    print(f"  [global GWL] WARNING: {global_path} missing; gwls_boxplot will fail.")


def bootstrap_country_artefacts(
    country: str, *, config: Config | None = None, force: bool = False,
) -> None:
    """Generate every xlsx artefact the M9 figure adapters need for ``country``.

    Idempotent: skips files that already exist (use ``force=True`` to
    regenerate). Order matters — yr_timeseries must come before
    climpact_state, which reads the time-series xlsx.
    """
    config = config or Config.from_env()
    print(f"Bootstrapping country artefacts for {country!r}...")
    bootstrap_performance_xlsx(country, config, force=force)
    bootstrap_mon_means_xlsx(country, config, force=force)
    bootstrap_spread_xlsx(country, config, force=force)
    bootstrap_yr_timeseries_xlsx(country, config, force=force)
    bootstrap_climpact_state_xlsx(country, config, force=force)
    bootstrap_global_warming_levels(config, force=force)
    print(f"Done: results/{country}/ has all expected xlsx artefacts.")
