"""Regenerate paper figures from cache (M9 entry-point).

Each ``fig_*`` function in ``subselect.viz.{performance_figs, spread_figs,
country_profile}`` returns a ``matplotlib.figure.Figure`` and carries a
module-level ``CATEGORY`` constant. This script routes each figure to
``results/<country>/figures/<CATEGORY>/<filename>.png`` at the cell's
paper-era DPI (300).

Coverage notes:
- Performance, spread, and the GWL country-profile triplet (WL_table,
  gwls_boxplot, gwls_boxplot_times) are wired here.
- The country-profile cells that depend on smoothed time-series derivations
  built upstream in ``CLIMPACT_figures.ipynb`` (cells 31, 32, 35, 44, 46, 49,
  50, 54 and the anomalies tables in 29, 42) are imported but left for a
  follow-up adapter that ports those derivations.
- Bias maps (cell 34) require per-pixel observed/bias map containers from M7's
  internals; deferred along with the smoothed-series cprof cells.

Usage:
    python scripts/regenerate_paper_figures.py --country greece [--include-seasonal-bias]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from subselect.config import Config
from subselect.viz import _country_bootstrap, _data_adapters as adapters
from subselect.viz import country_profile, performance_figs, spread_figs


def _save(fig: plt.Figure, country: str, category: str, filename: str, dpi: int = 300) -> Path:
    out_dir = REPO_ROOT / "results" / country / "figures" / category
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def regenerate(country: str, *, include_seasonal_bias: bool = False) -> list[Path]:
    config = Config.from_env()
    # Bootstrap any missing xlsx artefacts the figure adapters need (idempotent
    # — skips files that already exist). This makes the entry-point country-
    # agnostic: a fresh country only needs a `country_codes.json` entry and a
    # GADM polygon, plus the cached CMIP6 NetCDFs already on disk.
    _country_bootstrap.bootstrap_country_artefacts(country, config=config)

    written: list[Path] = []

    # Adapter pre-loads (shared inputs)
    ranked_full = adapters.load_ranked_full(country, config=config)
    model_ids = adapters.load_model_ids(config=config)
    cmip6_models = adapters.load_cmip6_models(config=config)
    perf_metrics = adapters.load_perf_metrics_dict(country=country, config=config)
    observed_std_dev_df = adapters.load_observed_std_dev_df(country, config=config)
    ordered_models = cmip6_models["model"].tolist()

    # ============================================================
    # Performance
    # ============================================================
    fig = performance_figs.fig_annual_HM_hist_perf(ranked_full, model_ids)
    written.append(_save(fig, country, performance_figs.CATEGORY,
                         f"{country}_annual_HM_hist_perf.png"))

    fig = performance_figs.fig_hps_rankings_annual_and_seasons(ranked_full, model_ids)
    written.append(_save(fig, country, performance_figs.CATEGORY,
                         f"{country}_HPS_rankings_annual_and_seasons.png"))

    # seasonal_perf_revised — per-variable
    seasonal_perf_specs = [
        ("tas", performance_figs.fig_seasonal_perf_revised_tas),
        ("pr", performance_figs.fig_seasonal_perf_revised_pr),
        ("psl", performance_figs.fig_seasonal_perf_revised_psl),
        ("tasmax", performance_figs.fig_seasonal_perf_revised_tasmax),
    ]
    for var, fn in seasonal_perf_specs:
        obs_mm, mod_mm = adapters.load_mon_means(var, country, config=config)
        kwargs = dict(
            ranked_full=ranked_full,
            model_ids=model_ids,
        )
        kwargs[f"{var}_all_perf_metrics"] = perf_metrics[var]
        kwargs[f"{var}_cmip6_mon_means"] = mod_mm
        kwargs[f"{var}_observed_mon_means"] = obs_mm
        if var == "tasmax":
            kwargs["ordered_models"] = ordered_models
        fig = fn(**kwargs)
        written.append(_save(fig, country, performance_figs.CATEGORY,
                             f"{country}_{var}_seasonal_perf_revised.png"))

    # annual Taylor diagrams — one per variable
    figs = performance_figs.fig_annual_taylor_per_variable(
        variables=["tas", "pr", "psl", "tasmax"],
        perf_metrics=perf_metrics,
        observed_std_dev_df=observed_std_dev_df,
        cmip6_models=cmip6_models,
        model_ids=model_ids,
    )
    for var, fig in figs.items():
        written.append(_save(fig, country, performance_figs.CATEGORY,
                             f"{country}_{var}_annual_taylor.png"))

    # 4-season Taylor diagrams — one per variable
    figs = performance_figs.fig_4season_taylor_per_variable(
        variables=["tas", "pr", "psl", "tasmax"],
        perf_metrics=perf_metrics,
        observed_std_dev_df=observed_std_dev_df,
        cmip6_models=cmip6_models,
        model_ids=model_ids,
    )
    for var, fig in figs.items():
        written.append(_save(fig, country, performance_figs.CATEGORY,
                             f"{country}_{var}_4season_taylor.png"))

    # Bias maps (M9.2) — observed_maps + bias_maps via per-model climatology adapter.
    bias_state = adapters.build_bias_maps_state(
        country, include_seasonal=include_seasonal_bias, config=config,
    )
    figs = performance_figs.fig_bias_maps_per_variable(
        observed_maps=bias_state["observed_maps"],
        bias_maps=bias_state["bias_maps"],
        perf_metrics=perf_metrics,
        model_ids=model_ids,
        country=country,
        shapefile_path=config.shapefile_path,
        include_seasonal_bias=include_seasonal_bias,
    )
    for key, fig in figs.items():
        # key is "<variable>_<period>"
        var, period = key.split("_", 1)
        if period == "annual":
            written.append(_save(fig, country, performance_figs.CATEGORY,
                                 f"{country}_{var}_annual_bias.png"))
        else:
            written.append(_save(fig, country, performance_figs.CATEGORY,
                                 f"{country}_{var}_{period}_bias.png"))

    # ============================================================
    # Spread
    # ============================================================
    long_term_df = adapters.load_long_term_spread(country, config=config).reindex(ordered_models)
    long_pi_change_df = adapters.load_long_term_change_spread(country, config=config).reindex(ordered_models)

    fig = spread_figs.fig_annual_spread_rev12(
        ranked_full=ranked_full,
        long_pi_change_df=long_pi_change_df,
        long_term_df=long_term_df,
        cmip6_models=cmip6_models,
        model_ids=model_ids,
        country=country,
    )
    written.append(_save(fig, country, spread_figs.CATEGORY,
                         f"{country}_annual_annual_spread_rev12.png"))

    fig = spread_figs.fig_seasonal_spread_perSeasonBars_right_named_rev1(
        ranked_full=ranked_full,
        long_pi_change_df=long_pi_change_df,
        long_term_df=long_term_df,
        cmip6_models=cmip6_models,
        model_ids=model_ids,
        country=country,
    )
    written.append(_save(fig, country, spread_figs.CATEGORY,
                         f"{country}_seasonal_spread_perSeasonBars_right_named_rev1.png"))

    # ============================================================
    # Country profile — GWL triplet (uses paper-era xlsx directly)
    # ============================================================
    warming_levels_all_models = adapters.load_warming_levels_all_models(country, config=config)
    warming_level_medians = adapters.load_warming_level_medians(country, config=config)
    global_warming_level_medians = adapters.load_warming_level_medians_global(config=config)

    fig = country_profile.fig_WL_table(
        warming_level_medians=warming_level_medians,
        country=country,
    )
    written.append(_save(fig, country, country_profile.CATEGORY,
                         f"{country}_WL_table.png"))

    fig = country_profile.fig_gwls_boxplot(
        warming_levels_all_models=warming_levels_all_models,
        warming_level_medians=warming_level_medians,
        global_warming_level_medians=global_warming_level_medians,
        country=country,
    )
    written.append(_save(fig, country, country_profile.CATEGORY,
                         f"{country}_gwls_boxplot.png"))

    # gwls_boxplot_times dropped per post-Phase-0 cleanup (CLIMPACT cell 61):
    # Times-font variant of the standard boxplot, redundant.

    # ============================================================
    # Country profile — M9.1 (smoothed-series + anomalies + change figures)
    # Built via build_climpact_state which runs CLIMPACT_figures derivation
    # cells [13, 15, 19, 21, 24, 26, 39, 41] verbatim against paper-era xlsx.
    # ============================================================
    state = adapters.build_climpact_state(country, config=config)

    fig = country_profile.fig_tas_anomalies_table(state, country=country)
    written.append(_save(fig, country, country_profile.CATEGORY,
                         f"{country}_tas_anomalies_table.png"))

    fig = country_profile.fig_tas_change(state, country=country)
    written.append(_save(fig, country, country_profile.CATEGORY,
                         f"{country}_tas_change.png"))

    # tas_change_all_shaded dropped per post-Phase-0 cleanup (CLIMPACT cell 32):
    # redundant variant of tas_change.

    fig = country_profile.fig_tas_change_spaghetti(state, country=country)
    written.append(_save(fig, country, country_profile.CATEGORY,
                         f"{country}_tas_change_spaghetti.png"))

    fig = country_profile.fig_pr_percent_anomalies_table(state, country=country)
    written.append(_save(fig, country, country_profile.CATEGORY,
                         f"{country}_pr_percent_anomalies_table.png"))

    # pr_change, pr_change_spaghetti, pr_percent_change_raw dropped per
    # post-Phase-0 cleanup (CLIMPACT cells 44, 46, 50): user prefers
    # percent-change variants over absolute pr; _raw is redundant.

    fig = country_profile.fig_pr_percent_change_ratio(state, country=country)
    written.append(_save(fig, country, country_profile.CATEGORY,
                         f"{country}_pr_percent_change_ratio.png"))

    fig = country_profile.fig_pr_percent_change_spaghetti(state, country=country)
    written.append(_save(fig, country, country_profile.CATEGORY,
                         f"{country}_pr_percent_change_spaghetti.png"))

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="greece")
    parser.add_argument(
        "--include-seasonal-bias", action="store_true",
        help="Render the 12 seasonal bias maps in addition to the 3 annual ones",
    )
    args = parser.parse_args()

    written = regenerate(args.country, include_seasonal_bias=args.include_seasonal_bias)
    for p in written:
        print(f"  → {p.relative_to(REPO_ROOT)}")
    print(f"\n{len(written)} figure(s) written to results/{args.country}/figures/")


if __name__ == "__main__":
    main()
