"""Step 4 — composite Taylor exploratory render. Iterated under
results/greece/figures/performance/composite_taylor_v1.png.

CHECKPOINT 2: annual row populated with actual Taylor diagrams (tas/pr/psl).
Legend slot + seasonal rows still placeholders. Verbatim per-panel drawing
via the existing TaylorDiagram class; new code is GridSpec orchestration.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, SubplotSpec

from subselect.config import Config
from subselect.viz.taylor import TaylorDiagram


COUNTRY = "greece"
_REPO_ROOT_FOR_OUT = Path(__file__).resolve().parents[1]
OUT = _REPO_ROOT_FOR_OUT / f"results/{COUNTRY}/figures/performance/composite_taylor_v1.png"

VAR_TITLE = {
    "tas": "Temperature",
    "pr": "Precipitation",
    "psl": "Sea-Level Pressure",
}
SEASONS = ("DJF", "MAM", "JJA", "SON")
PAD_HIGH = 0.10            # axis pad above max ratio (both annual + seasonal)
ANNUAL_INNER_CUT_MAX = 0.30  # annual donut: srange[0] capped at this (was 0.45 in legacy)
ANNUAL_INNER_CUT_PAD = 0.10  # annual donut: keep this much breathing room below min_ratio


def _data_driven_srange(
    *,
    variable: str,
    season_or_annual: str,
    perf_metrics: dict[str, pd.DataFrame],
    refstd: float,
    is_annual: bool,
) -> tuple[float, float]:
    """Choose ``srange = (low, high)`` for one Taylor panel.

    The upper bound is always data-driven (``max_ratio + PAD_HIGH``).
    The lower bound is split by panel type:

    - **Annual** keeps the legacy "annular" / donut look — ``srange[0] > 0``
      cuts an inner radius out of the polar plot. The cut is bounded both
      from above (max ``ANNUAL_INNER_CUT_MAX``, below the legacy 0.45) and
      from below (never go closer than ``ANNUAL_INNER_CUT_PAD`` to the
      smallest model ratio, so markers never sit on the inner edge).
    - **Seasonal** is always full-quadrant: ``srange[0] = 0``.
    """
    ref = max(refstd, 1e-12)
    std_series = pd.to_numeric(
        perf_metrics[variable][f"{season_or_annual}_std_dev"], errors="coerce",
    )
    ratios = std_series.values / ref
    finite = ratios[np.isfinite(ratios)]
    if finite.size == 0:
        return (0.0, 1.5)

    max_ratio = float(np.max(finite))
    high = max_ratio + PAD_HIGH

    if is_annual:
        min_ratio = float(np.min(finite))
        low = max(0.0, min(ANNUAL_INNER_CUT_MAX, min_ratio - ANNUAL_INNER_CUT_PAD))
    else:
        low = 0.0  # full quadrant from origin

    return (round(low, 2), round(high, 2))


# ---------------------------------------------------------------------------
# Per-panel Taylor drawing — verbatim per-cell logic, parameterised by
# (variable, season_or_annual). Caller passes a SubplotSpec from a master
# GridSpec; the helper attaches a FloatingSubplot inside that slot.
# ---------------------------------------------------------------------------

def draw_taylor_panel(
    fig: plt.Figure,
    subplotspec: SubplotSpec,
    variable: str,
    season_or_annual: str,
    *,
    perf_metrics: dict[str, pd.DataFrame],
    observed_std_dev_df: pd.DataFrame,
    cmip6_models: pd.DataFrame,
    model_ids: dict,
) -> TaylorDiagram:
    """Draw one Taylor panel inside *subplotspec*.

    Mirrors :func:`fig_annual_taylor_per_variable` (annual case) and
    :func:`fig_4season_taylor_per_variable` (seasonal case) verbatim, except
    the figure / rect arguments come from the caller's GridSpec.
    """
    is_annual = season_or_annual == "annual"
    refstd = float(observed_std_dev_df[variable][season_or_annual])

    # Data-driven srange: cover the actual model ratio range with ~5%
    # padding on each side. Hard-coded srange (0.45, 1.55) was Greece-tuned
    # and clipped markers for other countries where the ensemble spread is
    # wider. The lower-cut "stylistic" floor is preserved only when the data
    # genuinely has a gap below — see _data_driven_srange.
    srange = _data_driven_srange(
        variable=variable, season_or_annual=season_or_annual,
        perf_metrics=perf_metrics, refstd=refstd, is_annual=is_annual,
    )

    dia = TaylorDiagram(
        refstd=refstd, fig=fig, rect=subplotspec, label="Observed", srange=srange,
    )

    # Add the 35 numbered model markers
    for model_name in cmip6_models["model"].tolist():
        model_id = model_ids[model_name]
        std_val = perf_metrics[variable].loc[model_name, f"{season_or_annual}_std_dev"]
        r_val = perf_metrics[variable].loc[model_name, f"{season_or_annual}_corr"]
        if not (np.isfinite(std_val) and np.isfinite(r_val)):
            continue
        r_val = float(np.clip(r_val, -1.0, 1.0))
        dia.add_sample(
            stddev=float(std_val),
            corrcoef=r_val,
            marker=f"${model_id}$",
            ls="",
            ms=8 if model_id < 10 else 11,
            label=model_name,
        )

    # Declutter std-dev ticks. Annual panels also benefit (legacy
    # fig_annual_taylor lets matplotlib pick ~9 close-spaced ticks that
    # collide with each other in any composite layout). Use ~7 evenly-spaced
    # ticks at one decimal place.
    from mpl_toolkits.axisartist import grid_finder as GF

    n_ticks = 7
    gh = dia._ax.get_grid_helper()
    tick_vals = np.linspace(dia.smin, dia.smax, n_ticks)
    tick_vals = np.unique(np.round(tick_vals, 2))
    gh.grid_finder.grid_locator2 = GF.FixedLocator(tick_vals)
    gh.grid_finder.tick_formatter2 = GF.DictFormatter(
        {t: f"{t:.1f}" for t in tick_vals}
    )

    contours = dia.add_contours(levels=10 if is_annual else 6, colors="0.5")
    plt.clabel(contours, inline=1, fontsize=10 if is_annual else 9, fmt="%.1f")
    dia.add_grid()
    dia._ax.axis[:].major_ticks.set_tick_out(True)

    # Bump the "Standard deviation" + "Correlation" axis labels +2 units
    # over matplotlib default (10 → 12).
    for axis_name in ("left", "top"):
        dia._ax.axis[axis_name].label.set_fontsize(12)

    if is_annual:
        dia._ax.set_title(VAR_TITLE[variable], fontsize=14, pad=12)
    else:
        dia._ax.set_title(season_or_annual.upper(), fontsize=13)
    return dia


# ---------------------------------------------------------------------------
# Placeholders for not-yet-populated slots
# ---------------------------------------------------------------------------

def _placeholder(ax, label: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5, 0.5, label,
        transform=ax.transAxes, ha="center", va="center",
        fontsize=11, color="0.4",
    )
    for spine in ax.spines.values():
        spine.set_color("0.7")


def _row_label(ax, text: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.5, 0.5, text,
        transform=ax.transAxes, ha="center", va="center",
        fontsize=13, rotation=90,
    )


# ---------------------------------------------------------------------------
# State loading from cache (post-3.6 canonical inputs)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_inputs(country: str) -> dict:
    config = Config.from_env()
    cache_dir = REPO_ROOT / "cache" / country / "parquet"

    perf_metrics = {
        v: pd.read_parquet(cache_dir / f"performance_metrics__{v}.parquet")
        for v in ("tas", "pr", "psl", "tasmax")
    }
    observed_std_dev_df = pd.read_parquet(cache_dir / "observed_std_dev.parquet")
    cmip6_models = pd.read_excel(config.cmip6_metadata_root / "CMIP6_model_id.xlsx")
    model_ids = dict(zip(cmip6_models["model"], cmip6_models["id"]))
    return {
        "perf_metrics": perf_metrics,
        "observed_std_dev_df": observed_std_dev_df,
        "cmip6_models": cmip6_models,
        "model_ids": model_ids,
    }


# ---------------------------------------------------------------------------
# Composite render — CHECKPOINT 2
# ---------------------------------------------------------------------------

def render_checkpoint_2(out_path: Path, country: str) -> Path:
    inputs = load_inputs(country)

    fig = plt.figure(figsize=(18, 22))

    # Two-block outer layout so the annual↔legend↔seasonal gaps can be
    # tighter than the gaps between the three seasonal rows. left=0.05
    # gives the annual tas panel and its left axis label breathing room
    # away from the figure border (without it, the row label below has
    # nowhere to sit).
    upper_total = 7 + 2.5  # annual + legend
    lower_total = 5.0 * 3  # three seasonal rows — modest bump from 4.5×3
    gs_outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[upper_total, lower_total],
        hspace=0.08,  # legend ↔ first seasonal row: a bit of breathing room
        top=0.94,
        left=0.05,
    )
    gs_upper = gs_outer[0].subgridspec(
        2, 1, height_ratios=[7, 2.5], hspace=0.30,  # annual ↔ legend padding
    )
    gs_seasonal = gs_outer[1].subgridspec(
        3, 1, height_ratios=[1, 1, 1], hspace=0.30,  # more breathing room
                                                     # between the three seasonal rows
    )

    # Row 0 — annual: 3 Taylor panels. Capture first diagram so the shared
    # legend reuses its sample points (same matplotlib colour cycle as every
    # other panel — model_id N is the same colour everywhere because all
    # panels add samples in the same order).
    gs_annual = gs_upper[0].subgridspec(1, 3, wspace=0.12)
    first_dia = None
    for col, variable in enumerate(("tas", "pr", "psl")):
        dia = draw_taylor_panel(
            fig, gs_annual[0, col],
            variable, "annual",
            perf_metrics=inputs["perf_metrics"],
            observed_std_dev_df=inputs["observed_std_dev_df"],
            cmip6_models=inputs["cmip6_models"],
            model_ids=inputs["model_ids"],
        )
        if first_dia is None:
            first_dia = dia

    # Row 1 — shared legend block, 6 columns × 6 rows. mode="expand" makes
    # the legend stretch across the full slot width; markerscale enlarges
    # the per-panel markers so the model numbers read clearly.
    ax_legend = fig.add_subplot(gs_upper[1])
    ax_legend.axis("off")
    handles = list(first_dia.samplePoints)
    labels = [h.get_label() for h in handles]
    ax_legend.legend(
        handles, labels,
        ncol=6,
        loc="center",
        bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),  # confined to the legend slot
        mode="expand",
        fontsize=13,
        markerscale=1.6,
        frameon=False,
        handlelength=1.5,
        columnspacing=3.5,
        labelspacing=0.7,
        numpoints=1,
    )

    # Rows 2-4 — seasonal Taylor diagrams. CHECKPOINT 4 populates `tas`;
    # CHECKPOINTS 5/6 will populate `pr` and `psl`.
    seasonal_rows = [
        ("tas", "Seasonal Temperature", True),
        ("pr", "Seasonal Precipitation", True),
        ("psl", "Seasonal Sea-Level Pressure", True),
    ]
    for i, (var, row_label, populated) in enumerate(seasonal_rows):
        # Seasonal panels: no label column in the GridSpec — the row label
        # is placed below as a fig.text() at an absolute x close to the
        # figure left edge, INSIDE the gs_outer left=0.05 margin. This
        # gives the seasonal label visibly less white space to its left
        # than the annual panels (which start at gs_outer left=0.05).
        gs_seasons = gs_seasonal[i].subgridspec(1, 4, wspace=0.30)
        for col, season in enumerate(SEASONS):
            if populated:
                draw_taylor_panel(
                    fig, gs_seasons[0, col],
                    var, season,
                    perf_metrics=inputs["perf_metrics"],
                    observed_std_dev_df=inputs["observed_std_dev_df"],
                    cmip6_models=inputs["cmip6_models"],
                    model_ids=inputs["model_ids"],
                )
            else:
                ax = fig.add_subplot(gs_seasons[0, col])
                _placeholder(ax, f"{season} {var}")
        # Row label placed in the left margin via figure-coordinate text.
        slot_bbox = gs_seasonal[i].get_position(fig)
        y_center = (slot_bbox.y0 + slot_bbox.y1) / 2
        fig.text(
            0.015, y_center, row_label,
            rotation=90, ha="center", va="center", fontsize=14,
        )

    # bbox_inches="tight" crops asymmetrically (row labels extend left of
    # the panel area, nothing extends right), so suptitle's default x=0.5
    # ends up right of the saved-image centre. Anchor the title at the
    # actual content centre.
    fig.suptitle(
        f"Annual and seasonal Taylor diagrams over {country.capitalize()}",
        fontsize=16, x=0.45, y=0.99,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import sys
    country = sys.argv[1] if len(sys.argv) > 1 else COUNTRY
    out = _REPO_ROOT_FOR_OUT / f"results/{country}/figures/performance/composite_taylor_v1.png"
    p = render_checkpoint_2(out, country)
    print(f"wrote {p}")
