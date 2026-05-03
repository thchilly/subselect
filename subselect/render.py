"""L2 orchestrator: turn a :class:`SubselectState` into a folder of figures.

Single public entry point :func:`render`. Every figure function consumes
state attributes directly — there is no xlsx bridge.

The default writer puts each figure under
``results/<country>/figures/<category>/<filename>.png`` at the published
DPI (300). Matplotlib is used as the rendering backend; a future Phase 4
web-app branch can swap in a plotly backend without touching the figure
functions themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from subselect.config import Config
from subselect.state import SubselectState
from subselect.viz import country_profile, performance_figs, spread_figs


DEFAULT_DPI = 300


def _save(fig: plt.Figure, path: Path, dpi: int = DEFAULT_DPI) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def render(
    state: SubselectState,
    *,
    country: str | None = None,
    output_dir: Path | None = None,
    only: Iterable[str] | None = None,
    config: Config | None = None,
    include_seasonal_bias: bool = False,
) -> dict[str, Path]:
    """Render every figure for the country represented by *state*.

    Parameters
    ----------
    state
        A populated :class:`SubselectState` (as returned by
        :func:`subselect.compute.compute`).
    country
        Override the country name in the output folder structure. Defaults
        to ``state.country``.
    output_dir
        Override the output root. Default: ``results/<country>/figures/``
        relative to the resolved :class:`Config`.
    only
        Restrict rendering to a subset of figure groups. Choices:
        ``"performance"``, ``"spread"``, ``"country_profile"``.
    config
        Optional :class:`Config` (used only when ``output_dir`` is not given).
    include_seasonal_bias
        Whether to render the four seasonal bias maps in addition to the
        annual one (per variable).
    """
    config = config or Config.from_env()
    country = country or state.country
    if output_dir is None:
        output_dir = config.results_root / country / "figures"
    else:
        output_dir = Path(output_dir)

    only_set = set(only) if only else None

    def _enabled(group: str) -> bool:
        return only_set is None or group in only_set

    cmip6_models_meta = pd.read_excel(
        config.cmip6_metadata_root / "CMIP6_model_id.xlsx"
    )
    model_ids = dict(zip(cmip6_models_meta["model"], cmip6_models_meta["id"]))
    ordered_models = cmip6_models_meta["model"].tolist()
    ranked_full = state.composite_hps_full

    written: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # Performance figures
    # ------------------------------------------------------------------
    if _enabled("performance") and not ranked_full.empty:
        perf_dir = output_dir / performance_figs.CATEGORY

        fig = performance_figs.fig_hps_rankings_annual_and_seasons(ranked_full, model_ids)
        written["HPS_rankings_annual_and_seasons"] = _save(
            fig, perf_dir / f"{country}_HPS_rankings_annual_and_seasons.png",
        )

        # seasonal_perf_revised — per-variable
        seasonal_perf_specs = [
            ("tas", performance_figs.fig_seasonal_perf_revised_tas),
            ("pr", performance_figs.fig_seasonal_perf_revised_pr),
            ("psl", performance_figs.fig_seasonal_perf_revised_psl),
            ("tasmax", performance_figs.fig_seasonal_perf_revised_tasmax),
        ]
        for var, fn in seasonal_perf_specs:
            if var not in state.monthly_means or var not in state.performance_metrics:
                continue
            mm = state.monthly_means[var]
            kwargs = dict(
                ranked_full=ranked_full,
                model_ids=model_ids,
            )
            kwargs[f"{var}_all_perf_metrics"] = state.performance_metrics[var]
            kwargs[f"{var}_cmip6_mon_means"] = mm["cmip6"]
            kwargs[f"{var}_observed_mon_means"] = mm["obs"]
            if var == "tasmax":
                kwargs["ordered_models"] = ordered_models
            fig = fn(**kwargs)
            written[f"{var}_seasonal_perf_revised"] = _save(
                fig, perf_dir / f"{country}_{var}_seasonal_perf_revised.png",
            )

        # Composite Taylor (15 panels + shared legend) replaces the 12
        # individual annual + 4-season per-variable Taylor figures.
        # ``fig_annual_taylor_per_variable`` and ``fig_4season_taylor_per_variable``
        # remain in the module for backwards compatibility but are no longer
        # invoked by the entry point.
        fig = performance_figs.fig_composite_taylor(
            perf_metrics=state.performance_metrics,
            observed_std_dev_df=state.observed_std_dev,
            cmip6_models=cmip6_models_meta,
            model_ids=model_ids,
            country=country,
        )
        written["composite_taylor"] = _save(
            fig, perf_dir / f"{country}_composite_taylor.png",
        )

        # Bias maps
        has_bias = (
            state.observed_maps
            and state.bias_maps
            and any(state.observed_maps.get(p, {}) for p in state.observed_maps)
        )
        if has_bias:
            figs = performance_figs.fig_bias_maps_per_variable(
                observed_maps=state.observed_maps,
                bias_maps=state.bias_maps,
                perf_metrics=state.performance_metrics,
                model_ids=model_ids,
                country=country,
                shapefile_path=config.shapefile_path,
                include_seasonal_bias=include_seasonal_bias,
            )
            for key, fig in figs.items():
                var, period = key.split("_", 1)
                if period == "annual":
                    name = f"{country}_{var}_annual_bias.png"
                else:
                    name = f"{country}_{var}_{period}_bias.png"
                written[f"{var}_{period}_bias"] = _save(fig, perf_dir / name)

    # ------------------------------------------------------------------
    # Spread figures
    # ------------------------------------------------------------------
    if _enabled("spread") and not state.change_signals.empty:
        spread_dir = output_dir / spread_figs.CATEGORY
        long_term_df = state.long_term_spread.reindex(ordered_models)
        long_pi_change_df = state.change_signals.reindex(ordered_models)

        fig = spread_figs.fig_annual_spread_rev12(
            ranked_full=ranked_full,
            long_pi_change_df=long_pi_change_df,
            long_term_df=long_term_df,
            cmip6_models=cmip6_models_meta,
            model_ids=model_ids,
            country=country,
        )
        written["annual_spread_rev12"] = _save(
            fig, spread_dir / f"{country}_annual_annual_spread_rev12.png",
        )

        fig = spread_figs.fig_seasonal_spread_perSeasonBars_right_named_rev1(
            ranked_full=ranked_full,
            long_pi_change_df=long_pi_change_df,
            long_term_df=long_term_df,
            cmip6_models=cmip6_models_meta,
            model_ids=model_ids,
            country=country,
        )
        written["seasonal_spread_perSeasonBars_right_named_rev1"] = _save(
            fig, spread_dir / f"{country}_seasonal_spread_perSeasonBars_right_named_rev1.png",
        )

    # ------------------------------------------------------------------
    # Country profile figures
    # ------------------------------------------------------------------
    if _enabled("country_profile") and not state.warming_levels.empty:
        cprof_dir = output_dir / country_profile.CATEGORY
        fig_specs = [
            ("WL_table", country_profile.fig_WL_table),
            ("gwls_boxplot", country_profile.fig_gwls_boxplot),
            ("tas_anomalies_table", country_profile.fig_tas_anomalies_table),
            ("tas_change", country_profile.fig_tas_change),
            ("tas_change_spaghetti", country_profile.fig_tas_change_spaghetti),
            ("pr_percent_anomalies_table", country_profile.fig_pr_percent_anomalies_table),
            ("pr_percent_change_ratio", country_profile.fig_pr_percent_change_ratio),
            ("pr_percent_change_spaghetti", country_profile.fig_pr_percent_change_spaghetti),
        ]
        for name, fn in fig_specs:
            fig = fn(state, country=country)
            written[name] = _save(fig, cprof_dir / f"{country}_{name}.png")

    return written
