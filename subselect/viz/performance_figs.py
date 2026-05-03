"""Performance figures.

Each ``fig_*`` function consumes pre-computed performance artefacts (HPS
rankings, per-variable metric tables, monthly climatologies, σ_obs scalars,
bias-map fields) and returns a :class:`matplotlib.figure.Figure`. They are
called from :func:`subselect.render.render` against a
:class:`subselect.state.SubselectState`.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

CATEGORY = "performance"


# --------------------------------------------------------------------------
# Shared helpers: data-driven axis limits for the seasonal-performance
# scatter panels (DJF/MAM/JJA/SON × |bias| × annual correlation).
# --------------------------------------------------------------------------

def _robust_bound(
    values,
    *,
    side: str,
    iqr_mult: float = 3.0,
    max_outliers: int = 3,
) -> float:
    """Tukey-fence robust upper or lower bound for the inlier set.

    Parameters
    ----------
    values : array-like
        Sample values; NaNs and infinities are dropped.
    side : ``"upper"`` or ``"lower"``
        Which Tukey fence to apply.
    iqr_mult : float
        Fence multiplier on the IQR. Default 1.5 (standard outliers);
        2.0–3.0 catches only extreme outliers.
    max_outliers : int
        Cap on the number of points classified as outliers. When the
        Tukey rule flags more than this many, the bound backs off to
        retain (n_total − max_outliers) values inside. Prevents
        over-zooming when the distribution is heavy-tailed but not
        pathological.

    Returns
    -------
    float
        The largest (or smallest, depending on ``side``) value that the
        rule classifies as an inlier.
    """
    arr = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0

    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1

    if side == "upper":
        fence = q3 + iqr_mult * iqr
        inlier_mask = arr <= fence
        n_outliers = int(np.sum(~inlier_mask))
        if n_outliers > max_outliers:
            cut = np.partition(arr, -max_outliers - 1)[-max_outliers - 1]
            inlier_mask = arr <= cut
        inliers = arr[inlier_mask]
        return float(inliers.max()) if inliers.size > 0 else float(arr.max())

    if side == "lower":
        fence = q1 - iqr_mult * iqr
        inlier_mask = arr >= fence
        n_outliers = int(np.sum(~inlier_mask))
        if n_outliers > max_outliers:
            cut = np.partition(arr, max_outliers)[max_outliers]
            inlier_mask = arr >= cut
        inliers = arr[inlier_mask]
        return float(inliers.min()) if inliers.size > 0 else float(arr.min())

    raise ValueError(f"side must be 'upper' or 'lower', got {side!r}")


def _seasonal_performance_limits(
    perf_metrics_df: pd.DataFrame,
    *,
    bias_cols: tuple[str, ...] = ("DJF_bias", "MAM_bias", "JJA_bias", "SON_bias"),
    corr_col: str = "annual_corr",
    pad_frac: float = 0.05,
    iqr_mult: float = 3.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute ``(xlim, ylim)`` for the four seasonal-perf-revised scatter
    panels using Tukey-fence robust bounds.

    The four panels share xlim (cross-season comparison) and ylim (annual
    correlation is the same column for every panel). Outliers — models
    whose bias or correlation falls beyond the Tukey fence — fall outside
    the panel and trigger the docking + annotation machinery already in
    ``scatter_panel_optimized``. The intent is to keep the inlier cluster
    well-spread instead of letting one or two extreme models compress the
    whole panel.

    Bias columns are absolute values (per ``performance.py``: the
    per-pixel bias is taken in absolute value before the cos(lat)-weighted
    spatial mean when the metric name is ``"bias"``), so xlim floors at 0.
    Correlation upper bound caps at 1.0.
    """
    bias_values = np.concatenate([
        pd.to_numeric(perf_metrics_df[col], errors="coerce").to_numpy()
        for col in bias_cols
    ])
    bias_upper = _robust_bound(bias_values, side="upper", iqr_mult=iqr_mult)
    bias_pad = max(bias_upper, 1e-6) * pad_frac
    xlim = (0.0, bias_upper + bias_pad)

    corr = pd.to_numeric(perf_metrics_df[corr_col], errors="coerce")
    corr_lower = _robust_bound(corr, side="lower", iqr_mult=iqr_mult)
    corr_max = float(corr.max())
    corr_span = max(corr_max - corr_lower, 1e-3)
    corr_pad = corr_span * pad_frac
    ylim = (max(0.0, corr_lower - corr_pad), min(1.0, corr_max + corr_pad))
    return xlim, ylim


# --------------------------------------------------------------------------
# HPS rank plots: Annual (full-width) + 2×2 seasonal panels
# --------------------------------------------------------------------------

def fig_hps_rankings_annual_and_seasons(
    ranked_full: pd.DataFrame,
    model_ids: dict,
) -> plt.Figure:
    """Five-panel HPS ranking figure (annual full-width + 2×2 seasonal grid).

    Plots TSS, BVS, and the harmonic-mean HPS for every model, sorted
    descending by HPS, with the annual ranking spanning the top row at full
    figure width and the four seasonal rankings (DJF / MAM / JJA / SON)
    laid out below in a 2×2 grid using compact ID-only labels.

    Parameters
    ----------
    ranked_full
        Per-model HPS table with ``<period>_TSS_mm``, ``<period>_bias_score_mm``,
        and ``<period>_HMperf`` columns for every period.
    model_ids
        ``{model_name: integer_id}`` mapping used for marker labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # -------- configuration --------
    SEASONS = ['DJF', 'MAM', 'JJA', 'SON']
    ANNUAL  = 'annual'

    def col_TSS(season):  return f'{season}_TSS_mm'
    def col_BVS(season):  return f'{season}_bias_score_mm'   # BVS (bias-variability score), min–max normalized
    def col_HPS(season):  return f'{season}_HMperf'          # Harmonic-mean score (HPS), already min–max

    # sanity check
    needed = []
    for s in [ANNUAL] + SEASONS:
        needed += [col_TSS(s), col_BVS(s), col_HPS(s)]
    missing = [c for c in needed if c not in ranked_full.columns]
    if missing:
        raise KeyError(f"ranked_full is missing columns: {missing}")

    def prepare_for_season(df, season, sort_by_hps=True, top_n=None, use_names=True):
        df_s = df.copy()
        if sort_by_hps:
            df_s = df_s.sort_values(col_HPS(season), ascending=False)
        if top_n is not None:
            df_s = df_s.head(top_n)
        models = df_s.index.tolist()
        if use_names:
            xtick_labels = [f"({model_ids.get(m, 'NA')}) {m}" for m in models]
        else:
            xtick_labels = [f"{model_ids.get(m, 'NA')}" for m in models]  # compact: IDs only
        x = np.arange(len(models))
        tss = df_s[col_TSS(season)].to_numpy(dtype=float)
        bvs = df_s[col_BVS(season)].to_numpy(dtype=float)
        hps = df_s[col_HPS(season)].to_numpy(dtype=float)
        return x, xtick_labels, tss, bvs, hps

    def plot_rank_panel(ax, x, labels, tss, bvs, hps, title, compact=False, show_legend=False, rotate_labels=0, label_fontsize=9):
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.plot(x, tss, marker='o', linestyle='-', linewidth=1.1, markersize=5, alpha=0.8,
                label='TSS (min–max)', zorder=2)
        ax.plot(x, bvs, marker='s', linestyle='-', linewidth=1.1, markersize=5, alpha=0.8,
                label='BVS (min–max)', zorder=2)
        ax.plot(x, hps, marker='^', linestyle='-', linewidth=2.0, markersize=7,
                label='HPS (harmonic mean)', zorder=3)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score [0,1]', fontsize=11)
        ax.set_title(title, fontsize=13, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=rotate_labels, fontsize=label_fontsize)
    #    ax.set_xlabel('Model ID' if compact else 'Model (ID)', fontsize=10)
        ax.margins(x=0.01)
        if show_legend:
            ax.legend(loc='lower left', ncol=3, frameon=False, fontsize=10)

    # ----------------- build the full figure -----------------
    n_models = len(ranked_full.index)
    fig_w = max(14, n_models * 0.36)
    fig = plt.figure(figsize=(fig_w, 14.5))

    # Two-tier gridspec: outer splits Annual (with long rotated names) from the
    # 2x2 seasonal block (compact ID labels). Each tier gets its own hspace so
    # the gap between annual and seasons can be large without wasting space
    # between the two seasonal rows.
    outer = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[2.7, 3.6],
        hspace=0.50,
    )
    inner = outer[1].subgridspec(
        nrows=2, ncols=2,
        hspace=0.40,
        wspace=0.15,
    )

    # Top: ANNUAL — keep full model names, rotated 90°
    ax_top = fig.add_subplot(outer[0])
    x, labels, tss, bvs, hps = prepare_for_season(ranked_full, ANNUAL, sort_by_hps=True, top_n=None, use_names=True)
    plot_rank_panel(
        ax_top, x, labels, tss, bvs, hps,
        title='Annual Historical Performance — TSS, BVS, and HPS',
        compact=False, show_legend=True,
        rotate_labels=90, label_fontsize=9.5
    )

    # Seasons (compact: IDs only), rotate vertical to match style and save space
    axes_seasonal = [fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[0, 1]),
                     fig.add_subplot(inner[1, 0]), fig.add_subplot(inner[1, 1])]
    for ax, s in zip(axes_seasonal, SEASONS):
        x, labels, tss, bvs, hps = prepare_for_season(ranked_full, s, sort_by_hps=True, top_n=None, use_names=False)
        plot_rank_panel(
            ax, x, labels, tss, bvs, hps,
            title=f'{s} Historical Performance',
            compact=True, show_legend=False,
            rotate_labels=90, label_fontsize=9
        )

    return fig


# --------------------------------------------------------------------------
# Seasonal performance: per-variable annual cycle + four DJF/MAM/JJA/SON
# scatter panels (|bias| × annual correlation), coloured by the chosen
# annual metric. tas/pr/psl share this layout; tasmax has its own variant
# below because it carries extra missing-model bookkeeping.
# --------------------------------------------------------------------------

_SEASONAL_PERF_CONFIG = {
    "tas": dict(
        cmap_name="autumn_r",
        title="Annual Cycle of Temperature",
        ylabel="Temperature (°C)",
        xlabel="Absolute Bias",
        obs_legend="Observed (GSWP3-W5E5)",
        outlier_circle_size=350,
        bias_decimals=1,
        check_bias_lower_bound=False,
        forced_outlier_id=None,
        annot_offset_right=(-28, 22),
        annot_offset_left=(10, 17),
    ),
    "pr": dict(
        cmap_name="YlGnBu",
        title="Annual Cycle of Precipitation",
        ylabel="Precipitation (mm/day)",
        xlabel="Bias",
        obs_legend="Observed (GSWP3-W5E5)",
        outlier_circle_size=350,
        bias_decimals=2,
        check_bias_lower_bound=True,
        forced_outlier_id=26,
        annot_offset_right=(-42, 32),
        annot_offset_left=(8, 15.5),
    ),
    "psl": dict(
        cmap_name="winter_r",
        title="Annual Cycle of Sea-Level Pressure",
        ylabel="Sea-level pressure (hPa)",
        xlabel="Bias",
        obs_legend="Observed (W5E5)",
        outlier_circle_size=380,
        bias_decimals=2,
        check_bias_lower_bound=True,
        forced_outlier_id=None,
        annot_offset_right=(-42, 9),
        annot_offset_left=(8, 15.5),
    ),
}


def fig_seasonal_performance(
    *,
    variable: str,
    ranked_full: pd.DataFrame,
    all_perf_metrics: pd.DataFrame,
    cmip6_mon_means: pd.DataFrame,
    observed_mon_means: pd.DataFrame,
    model_ids: dict,
) -> plt.Figure:
    """Render the seasonal-performance figure for ``tas``, ``pr``, or ``psl``.

    The figure has a wide top panel with each model's annual cycle of
    monthly means against the observed reference, and four scatter panels
    (DJF, MAM, JJA, SON) of |bias| against annual correlation, with
    out-of-range models docked at the panel edge and annotated with their
    true coordinates. Colour encodes the annual RMSE of each model.

    Parameters
    ----------
    variable
        One of ``"tas"``, ``"pr"``, ``"psl"``. Selects the variable-specific
        cmap, axis labels, observation legend, and outlier-handling tweaks.
    ranked_full
        HPS ranking table (``annual_HMperf`` and ``<season>_HMperf`` columns).
    all_perf_metrics
        Per-model performance metrics for the chosen variable. Must include
        ``annual_rmse``, ``annual_corr``, and ``<season>_bias`` for the four
        seasons.
    cmip6_mon_means
        Monthly means per model (one column per model, rows indexed 1–12).
    observed_mon_means
        Monthly means of the observed reference (must contain a ``variable``-
        named column).
    model_ids
        Mapping ``{model_name: integer_id}`` used for marker labels.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.

    Raises
    ------
    ValueError
        If ``variable`` is not one of the supported keys.
    """
    if variable not in _SEASONAL_PERF_CONFIG:
        raise ValueError(
            f"variable must be one of {sorted(_SEASONAL_PERF_CONFIG)!r}, got {variable!r}"
        )
    cfg = _SEASONAL_PERF_CONFIG[variable]
    LIMIT_BIAS, LIMIT_CORR = _seasonal_performance_limits(all_perf_metrics)

    try:
        from adjustText import adjust_text
        HAS_ADJUST_TEXT = True
    except ImportError:
        HAS_ADJUST_TEXT = False
        print("Note: 'adjustText' not found. Labels may overlap slightly.")

    cmap = mpl.colormaps[cfg["cmap_name"]]
    series = all_perf_metrics["annual_rmse"]
    cbar_label = "Annual RMSE"

    valid = series.dropna()
    norm = Normalize(vmin=valid.min(), vmax=valid.max())

    def color_for(model):
        val = series.get(model, np.nan)
        return cmap(norm(val)) if np.isfinite(val) else (0.85, 0.85, 0.85, 1.0)

    model_colors = {m: color_for(m) for m in cmip6_mon_means.columns}

    forced_id = cfg["forced_outlier_id"]
    forced_outlier_name = None
    if forced_id is not None:
        for name, mid in model_ids.items():
            if mid == forced_id:
                forced_outlier_name = name
                break

    fig = plt.figure(figsize=(10, 11))
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.4, 1.0, 1.0],
        hspace=0.25, wspace=0.2,
    )
    ax_main = fig.add_subplot(gs[0, :])
    ax_djf = fig.add_subplot(gs[1, 0])
    ax_mam = fig.add_subplot(gs[1, 1])
    ax_jja = fig.add_subplot(gs[2, 0])
    ax_son = fig.add_subplot(gs[2, 1])
    for _ax in (ax_djf, ax_mam, ax_jja, ax_son):
        if hasattr(_ax, "set_box_aspect"):
            _ax.set_box_aspect(9 / 14)

    for col in cmip6_mon_means.columns:
        color = model_colors.get(col, "0.6")
        ax_main.plot(
            cmip6_mon_means.index,
            cmip6_mon_means[col].values,
            label=f"({model_ids[col]}) {col}",
            linewidth=0.8,
            color=color,
        )
    ax_main.plot(
        observed_mon_means.index,
        observed_mon_means[variable].values,
        label="Observed",
        color="black",
        linewidth=2.5,
        zorder=500,
    )
    ax_main.set_title(cfg["title"])
    ax_main.set_ylabel(cfg["ylabel"])
    ax_main.set_xlim(1, 12)
    months_ticks = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax_main.set_xticks(np.arange(1, 13))
    ax_main.set_xticklabels(months_ticks)

    bias_decimals = cfg["bias_decimals"]
    bias_fmt = f"B:{{:.{bias_decimals}f}}"
    check_bias_lower = cfg["check_bias_lower_bound"]
    circle_size = cfg["outlier_circle_size"]
    offset_right = cfg["annot_offset_right"]
    offset_left = cfg["annot_offset_left"]

    def scatter_panel(ax, xcol, title, xlabel):
        texts = []
        ax.set_xlim(LIMIT_BIAS)
        ax.set_ylim(LIMIT_CORR)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.axvline(x=0.0, color="black", linestyle=":", linewidth=1)

        for model in all_perf_metrics.index:
            true_x = all_perf_metrics.loc[model, xcol]
            true_y = all_perf_metrics.loc[model, "annual_corr"]
            mid = str(model_ids[model])
            color = model_colors.get(model, "0.6")

            if check_bias_lower:
                is_x_out = (true_x < LIMIT_BIAS[0]) or (true_x > LIMIT_BIAS[1])
            else:
                is_x_out = true_x > LIMIT_BIAS[1]
            is_y_out = (true_y < LIMIT_CORR[0]) or (true_y > LIMIT_CORR[1])
            is_forced = (forced_outlier_name is not None and model == forced_outlier_name)
            is_out = is_forced or is_x_out or is_y_out

            if is_out:
                plot_x = min(max(true_x, LIMIT_BIAS[0]), LIMIT_BIAS[1])
                plot_y = min(max(true_y, LIMIT_CORR[0]), LIMIT_CORR[1])

                ax.scatter(plot_x, plot_y, color="white", s=circle_size,
                           edgecolors="k", linewidth=1, zorder=10)
                t_out = ax.text(plot_x, plot_y, mid, fontsize=9,
                                color=color, ha="center", va="center",
                                fontweight="bold", zorder=11)
                t_out.set_path_effects([PathEffects.withStroke(linewidth=1.2, foreground="black")])

                parts = []
                if is_x_out:
                    parts.append(bias_fmt.format(true_x))
                if is_y_out:
                    parts.append(f"r:{true_y:.2f}")
                annot_text = "(" + ", ".join(parts) + ")" if parts else ""

                xytext_offset = offset_right if plot_x >= LIMIT_BIAS[1] else offset_left

                if annot_text:
                    ax.annotate(
                        annot_text, xy=(plot_x, plot_y), xytext=xytext_offset,
                        textcoords="offset points", fontsize=7.5, color="black",
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.8, ec="none"),
                    )
            else:
                ax.scatter(true_x, true_y, s=1, color=color, alpha=0)
                t = ax.text(true_x, true_y, mid, fontsize=9,
                            color=color, ha="center", va="center", fontweight="bold")
                t.set_path_effects([PathEffects.withStroke(linewidth=1.2, foreground="black")])
                texts.append(t)

        if HAS_ADJUST_TEXT and texts:
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.5),
                expand_points=(1.2, 1.2),
            )

        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)

    ax_djf.set_ylabel("Annual Correlation")
    scatter_panel(ax_djf, "DJF_bias", "DJF", "")
    scatter_panel(ax_mam, "MAM_bias", "MAM", "")
    ax_jja.set_ylabel("Annual Correlation")
    scatter_panel(ax_jja, "JJA_bias", "JJA", cfg["xlabel"])
    scatter_panel(ax_son, "SON_bias", "SON", cfg["xlabel"])

    legend_handles = [
        Line2D([0], [0], color=model_colors.get(col, "0.6"),
               marker="o", linestyle="", markersize=8,
               label=f"({model_ids[col]}) {col}")
        for col in cmip6_mon_means.columns
    ]
    legend_handles.insert(
        0,
        Line2D([0], [0], color="black", marker="o", linestyle="",
               markersize=6, label=cfg["obs_legend"]),
    )
    ax_main.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1.02),
        loc="upper left",
        borderaxespad=0.0,
        frameon=False,
        handlelength=0.5,
        fontsize=8.5,
        labelspacing=0.999,
    )

    fig.subplots_adjust(bottom=0.12)
    cax = fig.add_axes([0.12, 0.06, 0.783, 0.015])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(cbar_label)

    return fig


# --------------------------------------------------------------------------
# Seasonal performance — TASMAX
#
# Kept separate from :func:`fig_seasonal_performance` because the tasmax
# panel handles three CMIP6 models for which the variable is unavailable
# (extra missing-model bookkeeping, dedicated colormap, separate
# ``ordered_models`` argument). Folding it into the shared function via
# branches would be fragile, so it stays as its own implementation.
# --------------------------------------------------------------------------

def fig_seasonal_performance_tasmax(
    ranked_full: pd.DataFrame,
    tasmax_all_perf_metrics: pd.DataFrame,
    tasmax_cmip6_mon_means: pd.DataFrame,
    tasmax_observed_mon_means: pd.DataFrame,
    model_ids: dict,
    ordered_models: list = None,
) -> plt.Figure:
    """Seasonal-performance figure for ``tasmax`` (separate from tas/pr/psl).

    Three CMIP6 models in the canonical 35-model ordering do not provide
    ``tasmax``; this variant tracks them as missing and renders the
    available subset with its own colormap. Layout matches
    :func:`fig_seasonal_performance` (annual cycle on top, four seasonal
    scatter panels below).

    Parameters
    ----------
    ranked_full
        HPS ranking table.
    tasmax_all_perf_metrics
        Per-model performance metrics for ``tasmax``.
    tasmax_cmip6_mon_means, tasmax_observed_mon_means
        Monthly-mean tables for the model ensemble and the observed reference.
    model_ids
        ``{model_name: integer_id}`` mapping.
    ordered_models
        The canonical 1..35 model ordering (used to line up missing-model
        bookkeeping with the figure index).

    Returns
    -------
    matplotlib.figure.Figure
    """
    variable = 'tasmax'

    # Zoom limits (given by you)
    LIMIT_BIAS, LIMIT_CORR = _seasonal_performance_limits(tasmax_all_perf_metrics)

    # Try to import adjustText for smart label placement
    try:
        from adjustText import adjust_text
        HAS_ADJUST_TEXT = True
    except ImportError:
        HAS_ADJUST_TEXT = False
        print("Note: 'adjustText' not found. Labels may overlap slightly.")

    # --- 2. DATA PREPARATION ---
    color_metric = 'rmse'   # 'rmse' | 'corr' | 'bias' | 'tss' | 'hm'
    cmap = mpl.colormaps['copper_r']  # warm palette like tas

    if color_metric == 'rmse':
        series = tasmax_all_perf_metrics['annual_rmse']
        cbar_label = 'Annual RMSE'
    elif color_metric == 'corr':
        series = tasmax_all_perf_metrics['annual_corr']
        cbar_label = 'Annual Correlation'
    elif color_metric == 'bias':
        series = tasmax_all_perf_metrics['annual_bias'].abs()
        cbar_label = 'Annual |Bias|'
    elif color_metric == 'tss':
        series = tasmax_all_perf_metrics['annual_tss']
        cbar_label = 'Annual TSS'
    elif color_metric == 'hm':
        series = ranked_full['annual_HMperf']
        cbar_label = 'Annual HM performance (composite)'
    else:
        raise ValueError("color_metric must be one of: 'rmse','corr','bias','tss','hm'")

    valid = series.dropna()
    norm = Normalize(vmin=valid.min(), vmax=valid.max())

    def color_for(model):
        val = series.get(model, np.nan)
        return cmap(norm(val)) if np.isfinite(val) else (0.85, 0.85, 0.85, 1.0)

    # Present and missing models for tasmax
    present_models = list(tasmax_cmip6_mon_means.columns)
    all_models = ordered_models if 'ordered_models' in locals() and ordered_models is not None else list(model_ids.keys())
    missing_models = [m for m in all_models if m not in present_models]

    # Colors only for models we can actually draw in the cycle/scatter
    model_colors = {m: color_for(m) for m in present_models}

    # --- 3. FIGURE LAYOUT ---
    fig = plt.figure(figsize=(10, 11))
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.4, 1.0, 1.0],
        hspace=0.25, wspace=0.2
    )

    ax_main = fig.add_subplot(gs[0, :])
    ax_djf  = fig.add_subplot(gs[1, 0])
    ax_mam  = fig.add_subplot(gs[1, 1])
    ax_jja  = fig.add_subplot(gs[2, 0])
    ax_son  = fig.add_subplot(gs[2, 1])

    for _ax in (ax_djf, ax_mam, ax_jja, ax_son):
        if hasattr(_ax, "set_box_aspect"):
            _ax.set_box_aspect(9/14)

    # --- 4. MAIN PLOT (Annual Cycle Lines) ---
    for col in present_models:
        ax_main.plot(
            tasmax_cmip6_mon_means.index,
            tasmax_cmip6_mon_means[col].values,
            label=f"({model_ids[col]}) {col}",
            linewidth=0.8,
            color=model_colors.get(col, '0.6'),
        )

    ax_main.plot(
        tasmax_observed_mon_means.index,
        tasmax_observed_mon_means['tasmax'].values,
        label='Observed',
        color='black',
        linewidth=2.5,
        zorder=500
    )

    ax_main.set_title('Annual Cycle of Maximum Temperature')
    ax_main.set_ylabel('Temperature (°C)')
    ax_main.set_xlim(1, 12)
    months_ticks = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    ax_main.set_xticks(np.arange(1, 13))
    ax_main.set_xticklabels(months_ticks)

    # --- 5. SEASONAL SCATTER PANELS (Text markers + outlier docking) ---
    def scatter_panel_optimized(ax, xcol, title, xlabel):
        texts = []

        ax.set_xlim(LIMIT_BIAS)
        ax.set_ylim(LIMIT_CORR)

        ax.grid(True, linestyle='--', alpha=0.3)
        ax.axvline(x=0.0, color='black', linestyle=':', linewidth=1)

        for model in present_models:
            true_x = float(tasmax_all_perf_metrics.loc[model, xcol])
            true_y = float(tasmax_all_perf_metrics.loc[model, 'annual_corr'])
            if not (np.isfinite(true_x) and np.isfinite(true_y)):
                continue

            mid = str(model_ids[model])
            color = model_colors.get(model, '0.6')

            is_x_out = (true_x < LIMIT_BIAS[0]) or (true_x > LIMIT_BIAS[1])
            is_y_out = (true_y < LIMIT_CORR[0]) or (true_y > LIMIT_CORR[1])
            is_out = is_x_out or is_y_out

            if is_out:
                # Dock to nearest edge of the zoom box
                plot_x = min(max(true_x, LIMIT_BIAS[0]), LIMIT_BIAS[1])
                plot_y = min(max(true_y, LIMIT_CORR[0]), LIMIT_CORR[1])

                # White container circle at the edge
                ax.scatter(plot_x, plot_y, color="white", s=350,
                           edgecolors='k', linewidth=1, zorder=10)

                # Colored ID inside + black stroke
                t_out = ax.text(plot_x, plot_y, mid, fontsize=9,
                                color=color, ha='center', va='center',
                                fontweight='bold', zorder=11)
                t_out.set_path_effects([PathEffects.withStroke(linewidth=1.2, foreground='black')])

                # Annotation: ONLY show the out-of-bounds coordinate(s)
                parts = []
                if is_x_out:
                    parts.append(f"B:{true_x:.2f}")
                if is_y_out:
                    parts.append(f"r:{true_y:.3f}")
                annot_text = "(" + ", ".join(parts) + ")" if parts else ""

                xytext_offset = (-42, 9) if plot_x >= LIMIT_BIAS[1] else (8, 15.5)

                if annot_text:
                    ax.annotate(
                        annot_text, xy=(plot_x, plot_y), xytext=xytext_offset,
                        textcoords='offset points', fontsize=7.5, color='black',
                        arrowprops=dict(arrowstyle="->", color='black', lw=0.6),
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.8, ec='none')
                    )

            else:
                # Invisible anchor point
                ax.scatter(true_x, true_y, s=1, color=color, alpha=0)

                # Text marker (ID) with stroke
                t = ax.text(true_x, true_y, mid, fontsize=9,
                            color=color, ha='center', va='center',
                            fontweight='bold')
                t.set_path_effects([PathEffects.withStroke(linewidth=1.2, foreground='black')])
                texts.append(t)

        if HAS_ADJUST_TEXT and texts:
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle='-', color='dimgray', lw=0.5),
                expand_points=(1.2, 1.2)
            )

        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)

    # Execute the 4 panels
    ax_djf.set_ylabel('Annual Correlation')
    scatter_panel_optimized(ax_djf, 'DJF_bias', 'DJF', '')

    scatter_panel_optimized(ax_mam, 'MAM_bias', 'MAM', '')

    ax_jja.set_ylabel('Annual Correlation')
    scatter_panel_optimized(ax_jja, 'JJA_bias', 'JJA', 'Bias')

    scatter_panel_optimized(ax_son, 'SON_bias', 'SON', 'Bias')

    # --- 6. LEGEND & COLORBAR ---
    legend_handles = []
    legend_handles.append(
        Line2D([0], [0], color='black', marker='o', linestyle='',
               markersize=6, label='Observed (GSWP3-W5E5)')
    )

    # Present models (filled)
    for m in present_models:
        legend_handles.append(
            Line2D([0], [0], color=model_colors.get(m, '0.6'), marker='o', linestyle='',
                   markersize=6, label=f"({model_ids[m]}) {m}")
        )

    # Missing models (hollow, colored edge by chosen metric if available)
    for m in missing_models:
        edge_col = color_for(m)
        legend_handles.append(
            Line2D([0], [0],
                   marker='o', linestyle='',
                   markersize=6,
                   markerfacecolor='none',
                   markeredgewidth=1.5,
                   markeredgecolor=edge_col,
                   color='none',
                   label=f"({model_ids[m]}) {m}  [no tasmax]")
        )

    ax_main.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1.02),
        loc='upper left',
        borderaxespad=0.,
        frameon=False,
        handlelength=0.5,
        fontsize=8.5,
        labelspacing=0.999
    )

    fig.subplots_adjust(bottom=0.12)

    cax = fig.add_axes([0.12, 0.06, 0.783, 0.015])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label(cbar_label)

    return fig


# --------------------------------------------------------------------------
# Annual Taylor diagram (per variable). Superseded by
# :func:`fig_composite_taylor`; retained for direct callers that want a
# single-variable view.
# --------------------------------------------------------------------------

def fig_annual_taylor_per_variable(
    variables: Iterable[str],
    perf_metrics: Dict[str, pd.DataFrame],
    observed_std_dev_df: pd.DataFrame,
    cmip6_models: pd.DataFrame,
    model_ids: dict,
) -> Dict[str, plt.Figure]:
    """One annual Taylor diagram per variable. Returns ``{variable: Figure}``.

    Superseded by :func:`fig_composite_taylor` for the entry-point output
    (the composite renders all variables in a single figure); this helper
    is kept for callers that want each variable's diagram on its own.
    """
    from subselect.viz.taylor import TaylorDiagram
    figs: Dict[str, plt.Figure] = {}
    title_dict = {'tas': 'Temperature', 'pr': 'Precipitation', 'psl': 'Sea-Level Pressure', 'tasmax': 'Maximum Temperature'}
    #variable = 'tas'

    for variable in variables:
        fig = plt.figure(figsize=(8, 8))

        # Create a Taylor diagram and add the observed/reference point
        dia = TaylorDiagram(refstd=observed_std_dev_df[variable]['annual'], fig=fig, label='Observed', srange=(0.45, 1.55))

        # Add CMIP6 models to Taylor diagram with model ID as the marker
        for model_name in cmip6_models['model'].tolist():
            model_id = model_ids[model_name]  # Get the model ID
            dia.add_sample(perf_metrics[variable].loc[model_name, 'annual_std_dev'],
                        perf_metrics[variable].loc[model_name, 'annual_corr'],
                        marker=f'${model_id}$',  # Use the model ID as the marker
                        ls='', # No line connecting the marker to the point
                        #mfc='black', # Marker face color
                        ms=8 if model_id<10 else 11,
                        label=f'{model_name}') #f'({model_id}) {model_name}')


        # Add CRMSE contours, and label them
        contours = dia.add_contours(levels=10, colors='0.5')  # 5 levels in grey
        plt.clabel(contours, inline=1, fontsize=10, fmt='%.1f')

        # Add grid
        dia.add_grid()
        dia._ax.axis[:].major_ticks.set_tick_out(True)  # Put ticks outward

        # Add a figure legend and title
        fig.legend(dia.samplePoints,
                    [ p.get_label() for p in dia.samplePoints ],
                    numpoints=1, prop=dict(size='small'), loc='upper right',
                    bbox_to_anchor=(1.2, 0.96),
                    fancybox=False, edgecolor='white')

        fig.suptitle(f"Annual {title_dict[variable]}", size='x-large')  # Figure title

        figs[variable] = fig
    return figs


# --------------------------------------------------------------------------
# Four-season Taylor diagram (per variable). Superseded by
# :func:`fig_composite_taylor`; retained for callers that want a single-
# variable view across DJF/MAM/JJA/SON.
# --------------------------------------------------------------------------

def fig_4season_taylor_per_variable(
    variables: Iterable[str],
    perf_metrics: Dict[str, pd.DataFrame],
    observed_std_dev_df: pd.DataFrame,
    cmip6_models: pd.DataFrame,
    model_ids: dict,
) -> Dict[str, plt.Figure]:
    """One 4-season Taylor figure (DJF/MAM/JJA/SON) per variable.

    Superseded by :func:`fig_composite_taylor` for the entry-point output;
    kept for direct callers that want the per-variable view.
    """
    from subselect.viz.taylor import TaylorDiagram
    from mpl_toolkits.axisartist import grid_finder as GF
    figs: Dict[str, plt.Figure] = {}

    title_dict = {
        'tas': 'Temperature',
        'pr': 'Precipitation',
        'psl': 'Sea-Level Pressure',
        'tasmax': 'Maximum Temperature'
    }

    # soft caps to avoid absurdly compressed panels; raise if you truly need more
    SRANGE_CAP = {'tas': 3.0, 'tasmax': 3.0, 'psl': 5.0, 'pr': 20.0}

    def _round_up_to_half(x):
        return np.ceil(np.asarray(x) * 2.0) / 2.0

    for variable in variables:
        # observed std (per season) used as the Taylor reference radius
        stdrefs = dict(
            DJF=float(observed_std_dev_df[variable]['DJF']),
            MAM=float(observed_std_dev_df[variable]['MAM']),
            JJA=float(observed_std_dev_df[variable]['JJA']),
            SON=float(observed_std_dev_df[variable]['SON']),
        )

        # 1×4 layout
        rects = dict(DJF=141, MAM=142, JJA=143, SON=144)

        fig = plt.figure(figsize=(15, 4.6))
        fig.suptitle(title_dict[variable], size='x-large')

        for season in ['DJF', 'MAM', 'JJA', 'SON']:
            # guard: if the observed std is ~0, avoid divide-by-zero
            ref = max(stdrefs[season], 1e-12)

            # compute max σ ratio across models for this panel
            std_series = pd.to_numeric(perf_metrics[variable][f'{season}_std_dev'], errors='coerce')
            a_max = np.nanmax(std_series.values / ref) if np.isfinite(ref) else 1.5

            # choose σ-axis span in units of ref std (0 → srange_factor×ref)
            # at least 1.5, otherwise up to the rounded-up max ratio (with a soft cap)
            srange_factor = float(
                np.minimum(
                    SRANGE_CAP.get(variable, 5.0),
                    np.maximum(1.5, _round_up_to_half(1.05 * a_max))
                )
            )

            dia = TaylorDiagram(
                refstd=ref,
                fig=fig,
                rect=rects[season],
                label='Observed',
                srange=(0.0, srange_factor)
            )

            # --- declutter std-dev ticks ONLY (≈9 ticks, 1 decimal) ---
            gh = dia._ax.get_grid_helper()
            tick_vals = np.linspace(dia.smin, dia.smax, 9)        # endpoints included
            tick_vals = np.unique(np.round(tick_vals, 1))         # 1 decimal, drop dups
            gh.grid_finder.grid_locator2 = GF.FixedLocator(tick_vals)
            gh.grid_finder.tick_formatter2 = GF.DictFormatter(
                {t: f"{t:.1f}" for t in tick_vals}
            )
            # ---------------------------------------------------------

            # (optional) keep circular look
            try:
                dia.ax.set_aspect('equal', adjustable='box')
            except Exception:
                pass

            # add models
            for model_name in cmip6_models['model'].tolist():
                model_id = model_ids[model_name]
                std_val = perf_metrics[variable].loc[model_name, f'{season}_std_dev']
                r_val   = perf_metrics[variable].loc[model_name, f'{season}_corr']
                if not (np.isfinite(std_val) and np.isfinite(r_val)):
                    continue
                r_val = float(np.clip(r_val, -1.0, 1.0))
                dia.add_sample(
                    stddev=float(std_val),
                    corrcoef=r_val,
                    marker=f'${model_id}$',
                    ls='',
                    ms=(8 if model_id < 10 else 11),
                    label=model_name
                )

            # rms contours and cosmetics
            contours = dia.add_contours(levels=6, colors='0.5')
            dia.ax.clabel(contours, inline=1, fontsize=9, fmt='%.1f')
            dia._ax.set_title(season.upper())
            dia.add_grid()
            dia.ax.set_yticks([])

        figs[variable] = fig
    return figs


# --------------------------------------------------------------------------
# Composite Taylor figure — single 15-panel diagram replacing
# fig_annual_taylor_per_variable + fig_4season_taylor_per_variable.
# --------------------------------------------------------------------------

_COMPOSITE_VAR_TITLE = {
    "tas": "Temperature",
    "pr": "Precipitation",
    "psl": "Sea-Level Pressure",
}
_COMPOSITE_SEASONS = ("DJF", "MAM", "JJA", "SON")
_COMPOSITE_PAD_HIGH = 0.10
_COMPOSITE_ANNUAL_INNER_CUT_MAX = 0.30
_COMPOSITE_ANNUAL_INNER_CUT_PAD = 0.10


def _composite_data_driven_srange(
    *,
    variable: str,
    season_or_annual: str,
    perf_metrics: Dict[str, pd.DataFrame],
    refstd: float,
    is_annual: bool,
) -> tuple[float, float]:
    """Choose ``srange = (low, high)`` for one composite-Taylor panel.

    Upper bound is data-driven (``max_ratio + PAD_HIGH``).
    Lower bound: annual keeps a small donut cut, capped to leave room for
    the smallest model marker; seasonal is always full-quadrant from 0.
    """
    ref = max(refstd, 1e-12)
    std_series = pd.to_numeric(
        perf_metrics[variable][f"{season_or_annual}_std_dev"], errors="coerce",
    )
    ratios = std_series.values / ref
    finite = ratios[np.isfinite(ratios)]
    if finite.size == 0:
        return (0.0, 1.5)

    high = float(np.max(finite)) + _COMPOSITE_PAD_HIGH
    if is_annual:
        min_ratio = float(np.min(finite))
        low = max(
            0.0,
            min(_COMPOSITE_ANNUAL_INNER_CUT_MAX,
                min_ratio - _COMPOSITE_ANNUAL_INNER_CUT_PAD),
        )
    else:
        low = 0.0
    return (round(low, 2), round(high, 2))


def _draw_taylor_panel(
    fig: plt.Figure,
    subplotspec,
    variable: str,
    season_or_annual: str,
    *,
    perf_metrics: Dict[str, pd.DataFrame],
    observed_std_dev_df: pd.DataFrame,
    cmip6_models: pd.DataFrame,
    model_ids: dict,
):
    """Draw one Taylor panel inside *subplotspec*.

    Mirrors the per-panel logic of ``fig_annual_taylor_per_variable`` (annual)
    and ``fig_4season_taylor_per_variable`` (seasonal); the only difference is
    that the figure / rect arguments come from a master GridSpec slot.
    """
    from subselect.viz.taylor import TaylorDiagram
    from mpl_toolkits.axisartist import grid_finder as GF

    is_annual = season_or_annual == "annual"
    refstd = float(observed_std_dev_df[variable][season_or_annual])
    srange = _composite_data_driven_srange(
        variable=variable, season_or_annual=season_or_annual,
        perf_metrics=perf_metrics, refstd=refstd, is_annual=is_annual,
    )

    dia = TaylorDiagram(
        refstd=refstd, fig=fig, rect=subplotspec, label="Observed", srange=srange,
    )

    for model_name in cmip6_models["model"].tolist():
        model_id = model_ids[model_name]
        std_val = perf_metrics[variable].loc[model_name, f"{season_or_annual}_std_dev"]
        r_val = perf_metrics[variable].loc[model_name, f"{season_or_annual}_corr"]
        if not (np.isfinite(std_val) and np.isfinite(r_val)):
            continue
        r_val = float(np.clip(r_val, -1.0, 1.0))
        dia.add_sample(
            stddev=float(std_val), corrcoef=r_val,
            marker=f"${model_id}$", ls="",
            ms=8 if model_id < 10 else 11,
            label=model_name,
        )

    # Declutter std-dev ticks to ~7 evenly-spaced values, 1 decimal place,
    # so dense panels remain readable in the composite.
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
    for axis_name in ("left", "top"):
        dia._ax.axis[axis_name].label.set_fontsize(12)

    if is_annual:
        dia._ax.set_title(_COMPOSITE_VAR_TITLE[variable], fontsize=14, pad=12)
    else:
        dia._ax.set_title(season_or_annual.upper(), fontsize=13)
    return dia


def fig_composite_taylor(
    perf_metrics: Dict[str, pd.DataFrame],
    observed_std_dev_df: pd.DataFrame,
    cmip6_models: pd.DataFrame,
    model_ids: dict,
    country: str,
) -> plt.Figure:
    """Single composite Taylor figure: 15 panels (3 annual + 12 seasonal) +
    one shared 6×6 legend, replacing the 12 individual Taylor figures.

    Layout (top to bottom):
        Row 1  — Annual Taylor (tas, pr, psl) — donut quadrants.
        Row 2  — Shared legend (★ Observed + 35 numbered models, 6×6 grid).
        Rows 3-5 — Seasonal Taylor for tas / pr / psl (DJF, MAM, JJA, SON);
                   each row labelled on the left via ``fig.text``.
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 22))

    upper_total = 7 + 2.5
    lower_total = 5.0 * 3
    gs_outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[upper_total, lower_total],
        hspace=0.08, top=0.94, left=0.05,
    )
    gs_upper = gs_outer[0].subgridspec(2, 1, height_ratios=[7, 2.5], hspace=0.30)
    gs_seasonal = gs_outer[1].subgridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.30)

    # Annual row: 3 Taylor panels, capture first for shared-legend handles.
    gs_annual = gs_upper[0].subgridspec(1, 3, wspace=0.12)
    first_dia = None
    for col, variable in enumerate(("tas", "pr", "psl")):
        dia = _draw_taylor_panel(
            fig, gs_annual[0, col], variable, "annual",
            perf_metrics=perf_metrics, observed_std_dev_df=observed_std_dev_df,
            cmip6_models=cmip6_models, model_ids=model_ids,
        )
        if first_dia is None:
            first_dia = dia

    # Shared legend block (6 cols × 6 rows = 36 entries).
    ax_legend = fig.add_subplot(gs_upper[1])
    ax_legend.axis("off")
    handles = list(first_dia.samplePoints)
    labels = [h.get_label() for h in handles]
    ax_legend.legend(
        handles, labels, ncol=6, loc="center",
        bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), mode="expand",
        fontsize=13, markerscale=1.6, frameon=False,
        handlelength=1.5, columnspacing=3.5, labelspacing=0.7, numpoints=1,
    )

    # Seasonal rows — full-quadrant Taylor + figure-coord row label.
    seasonal_rows = [
        ("tas", "Seasonal Temperature"),
        ("pr", "Seasonal Precipitation"),
        ("psl", "Seasonal Sea-Level Pressure"),
    ]
    for i, (variable, row_label) in enumerate(seasonal_rows):
        gs_seasons = gs_seasonal[i].subgridspec(1, 4, wspace=0.30)
        for col, season in enumerate(_COMPOSITE_SEASONS):
            _draw_taylor_panel(
                fig, gs_seasons[0, col], variable, season,
                perf_metrics=perf_metrics, observed_std_dev_df=observed_std_dev_df,
                cmip6_models=cmip6_models, model_ids=model_ids,
            )
        slot_bbox = gs_seasonal[i].get_position(fig)
        y_center = (slot_bbox.y0 + slot_bbox.y1) / 2
        fig.text(
            0.015, y_center, row_label,
            rotation=90, ha="center", va="center", fontsize=14,
        )

    fig.suptitle(
        f"Annual and seasonal Taylor diagrams over {country.capitalize()}",
        fontsize=16, x=0.45, y=0.99,
    )
    return fig


# --------------------------------------------------------------------------
# Bias maps per variable. ``include_seasonal_bias`` toggles whether the
# four DJF/MAM/JJA/SON figures are produced in addition to the annual one.
# --------------------------------------------------------------------------

def fig_bias_maps_per_variable(
    observed_maps: dict,
    bias_maps: dict,
    perf_metrics: Dict[str, pd.DataFrame],
    model_ids: dict,
    country: str,
    shapefile_path,
    *,
    include_seasonal_bias: bool = False,
) -> Dict[str, plt.Figure]:
    """Per-(variable, period) bias-map figures for one country.

    Each figure shows the observed mean (top) followed by every model's
    bias-from-observed map. The grid column count adapts to the country's
    aspect ratio so panel titles stay legible. Returns a mapping
    ``{f"{variable}_{period}": Figure}``.

    Parameters
    ----------
    observed_maps
        Nested dict ``{period: {variable: xr.Dataset}}`` of country-cropped
        observed-mean fields (sourced from native 0.5° W5E5).
    bias_maps
        Nested dict ``{period: {variable: {model: xr.DataArray}}}`` of
        per-model ``model − observed`` bias fields.
    perf_metrics
        Per-variable performance-metric tables (used for the per-panel
        annotation showing each model's bias / RMSE).
    model_ids
        ``{model_name: integer_id}`` mapping for panel titles.
    country
        Country name (drives polygon lookup in the GADM shapefile and
        figure titles).
    shapefile_path
        Path to the GADM 4.1 GeoPackage.
    include_seasonal_bias
        When ``True``, render DJF/MAM/JJA/SON figures in addition to the
        annual one.

    Returns
    -------
    dict
        ``{f"{variable}_{period}": matplotlib.figure.Figure}``.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import cmocean.cm as cmo  # noqa: F401
    import cmcrameri.cm as cmc
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    import geopandas as gpd

    mpl.rcParams['figure.dpi'] = 100

    def get_bias_color_limits(bias_maps, variable, period):
        """Return symmetric ±limit where limit = ceil(max(|bias|)) across all models."""
        min_bias = np.inf
        max_bias = -np.inf
        for _, bias_map in bias_maps[period][variable].items():
            current_min = float(bias_map.min(skipna=True).values)
            current_max = float(bias_map.max(skipna=True).values)
            if current_min < min_bias:
                min_bias = current_min
            if current_max > max_bias:
                max_bias = current_max
        max_abs_bias = max(abs(min_bias), abs(max_bias))
        limit = int(np.ceil(max_abs_bias)) if np.isfinite(max_abs_bias) else 1
        return -limit, limit

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        """Create a truncated colormap from an existing colormap or name."""
        base = plt.get_cmap(cmap)
        colors = base(np.linspace(minval, maxval, n))
        return LinearSegmentedColormap.from_list(f'trunc({base.name},{minval:.2f},{maxval:.2f})', colors)

    def plot_variable_bias_maps(variable, period, observed_maps, bias_maps, perf_df, country):
        # Palettes (keep your choices; add tasmax)
        if variable == 'tas':
            truncated_cmap = truncate_colormap('afmhot_r', 0.00, 0.70)  # sequential for observed
            bias_cmap = 'RdBu_r'                                        # diverging for bias
            obs_title = f'Observed Mean Air Surface Temperature ({variable.lower()}) for {period} period [1995–2014]'
            obs_units = 'Air surface temperature (°C)'
            bias_label = 'Temperature Bias (°C)'
        elif variable == 'tasmax':
            truncated_cmap = truncate_colormap('afmhot_r', 0.00, 0.70)
            bias_cmap = 'RdBu_r'
            obs_title = f'Observed Mean Maximum Temperature ({variable.lower()}) for {period} period [1995–2014]'
            obs_units = 'Maximum temperature (°C)'
            bias_label = 'Max Temperature Bias (°C)'
        elif variable == 'pr':
            truncated_cmap = truncate_colormap(cmc.oslo_r, 0.05, 0.75)
            bias_cmap = 'PuOr'
            obs_title = f'Observed Mean Precipitation ({variable.lower()}) for {period} period [1995–2014]'
            obs_units = 'Precipitation (mm/d)'
            bias_label = 'Precipitation Bias (mm/d)'
        elif variable == 'psl':
            truncated_cmap = truncate_colormap(cmc.batlowW, 0.10, 0.50)
            bias_cmap = cmc.vik
            obs_title = f'Observed Mean Sea-Level Pressure ({variable.lower()}) for {period} period [1995–2014]'
            obs_units = 'Sea-Level Pressure (hPa)'
            bias_label = 'Sea-Level Pressure Bias (hPa)'
        else:
            truncated_cmap = plt.get_cmap('viridis')
            bias_cmap = 'RdBu_r'
            obs_title = f'Observed mean ({variable}) — {period}'
            obs_units = variable
            bias_label = f'{variable} Bias'

        # Filter the country polygon out of the GADM 4.1 GeoPackage.
        gdf = gpd.read_file(shapefile_path)
        country_boundaries = gdf[gdf["COUNTRY"].str.lower() == country.lower()]
        if country_boundaries.empty:
            raise FileNotFoundError(f"Country {country} not found in {shapefile_path}")

        minx, miny, maxx, maxy = country_boundaries.total_bounds
        aspect_ratio = (maxx - minx) / max(1e-6, (maxy - miny))

        # Grid sizing — adaptive to country bounding-box aspect ratio so the
        # per-model titles remain readable for tall/narrow countries (Sweden,
        # Chile, Norway, ...). Greece's 6-column default was Greece-tuned;
        # countries with aspect (width/height) < 1 produce panels too narrow
        # for typical CMIP6 model names. Two safeguards:
        #   1. Adaptive n_columns by aspect tier (fewer columns → each panel
        #      gets more horizontal space).
        #   2. Minimum panel width floor (MIN_PANEL_WIDTH_INCHES) so even
        #      extreme cases like Chile keep titles readable.
        # The map content inside each panel keeps its native aspect via
        # set_aspect('equal'); when base_width exceeds the natural width the
        # map sits centred with horizontal padding inside the panel slot.
        n_models = len(bias_maps[period][variable])
        if aspect_ratio >= 0.9:
            n_columns = 6
        elif aspect_ratio >= 0.6:
            n_columns = 5
        elif aspect_ratio >= 0.4:
            n_columns = 4
        elif aspect_ratio >= 0.25:
            n_columns = 3
        else:
            n_columns = 2
        n_rows = math.ceil(n_models / n_columns) + 2  # +2 rows for the observed panel

        base_height = 1.8
        MIN_PANEL_WIDTH_INCHES = 1.7  # fits typical CMIP6 model name as title
        natural_width = 0.7 * (base_height * aspect_ratio)
        base_width = max(MIN_PANEL_WIDTH_INCHES, natural_width)
        fig_width  = base_width * n_columns
        fig_height = base_height * n_rows - 2.5

        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(n_rows, n_columns, figure=fig)

        # Observed mean panel (spans first two rows)
        ax_mean = fig.add_subplot(gs[0:2, :], projection=ccrs.PlateCarree())
        obs_ds = observed_maps[period][variable]  # xr.Dataset with var named `variable`
        vmin = float(obs_ds[variable].min(dim=['lat', 'lon'], skipna=True).values)
        vmax = float(obs_ds[variable].max(dim=['lat', 'lon'], skipna=True).values)
        im_obs = obs_ds[variable].plot(
            ax=ax_mean, cmap=truncated_cmap, add_colorbar=False,
            vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree()
        )
        country_boundaries.boundary.plot(ax=ax_mean, color='black', linewidth=0.4)
        ax_mean.set_aspect('equal', adjustable='box')
        ax_mean.set_xlabel('Longitude')
        ax_mean.set_ylabel('Latitude')
        ax_mean.coastlines(linewidth=0.3)
        ax_mean.add_feature(cf.BORDERS, linewidth=0.3)
        ax_mean.set_title(obs_title)

        # Colorbar for observed panel (inset)
        cax_obs = inset_axes(ax_mean, width="5%", height="100%", loc='right',
                             bbox_to_anchor=(0.1, 0., 1, 1),
                             bbox_transform=ax_mean.transAxes, borderpad=0)
        cbar_mean = fig.colorbar(im_obs, cax=cax_obs, orientation='vertical')
        cbar_mean.set_label(obs_units, fontsize=11)

        # Bias color limits (symmetric) + centered norm (0 at the colormap center)
        bias_vmin, bias_vmax = get_bias_color_limits(bias_maps, variable, period)
        L = max(abs(bias_vmin), abs(bias_vmax))
        bias_norm = TwoSlopeNorm(vmin=-L, vcenter=0.0, vmax=L)

        # ======= MODEL PANELS =======
        # Order models by **decreasing** absolute bias (largest |bias| first)
        order = (
            perf_df[f'{period}_bias']
            .astype(float)
            .sort_values(ascending=True)
            .index
        )

        idx = 0
        for model_name in order:
            if model_name not in bias_maps[period][variable]:
                continue  # not available (e.g., missing tasmax)
            ax = fig.add_subplot(gs[2 + idx // n_columns, idx % n_columns], projection=ccrs.PlateCarree())

            # Signed bias map with centered norm
            bias_da = bias_maps[period][variable][model_name]
            im_bias = bias_da.plot(
                ax=ax, cmap=bias_cmap, norm=bias_norm,
                add_colorbar=False, transform=ccrs.PlateCarree()
            )

            ax.set_title(f'({model_ids[model_name]}) {model_name}', fontsize=11)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('', fontsize=1)
            ax.set_ylabel('', fontsize=1)
            ax.coastlines(linewidth=0.2)
            ax.add_feature(cf.BORDERS, linewidth=0.2)

            # Text annotation: ABS. BIAS (area-weighted; stored absolute in perf_df)
            abs_bias_val = float(perf_df.at[model_name, f'{period}_bias'])
            ax.text(0.03, 0.15, f'MAB: {abs_bias_val:.2f}',
                    transform=ax.transAxes, fontsize=9.5, color='black',
                    ha='left', va='top')

            idx += 1

        # Shared colorbar for bias maps using the same norm/cmap
        if idx > 0:
            cax = fig.add_axes([0.30, -0.03, 0.40, 0.02])  # [left, bottom, width, height]
            sm = mpl.cm.ScalarMappable(norm=bias_norm, cmap=bias_cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', extend='both')
            cbar.set_label(bias_label, fontsize=13)

        plt.tight_layout()
        return fig

    figs: Dict[str, plt.Figure] = {}
    periods = ['annual', 'DJF', 'MAM', 'JJA', 'SON'] if include_seasonal_bias else ['annual']
    # Run
    for variable in ['tas', 'pr', 'psl', 'tasmax']:
        for period in periods:
            fig = plot_variable_bias_maps(
                variable, period,
                observed_maps=observed_maps,
                bias_maps=bias_maps,
                perf_df=perf_metrics[variable],  # contains the ABS bias column
                country=country
            )
            figs[f"{variable}_{period}"] = fig
    return figs
