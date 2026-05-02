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
# Cell 13 — HPS rank plots: Annual (full-width) + 2×2 seasonal panels
# --------------------------------------------------------------------------

def fig_hps_rankings_annual_and_seasons(
    ranked_full: pd.DataFrame,
    model_ids: dict,
) -> plt.Figure:
    # HPS rank plots: Annual (full-width) + 2×2 seasonal panels

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
    # more total height to make room for rotated labels
    fig = plt.figure(figsize=(fig_w, 15.8))

    # bigger top panel + more vertical spacing between rows; a touch more col spacing too
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        height_ratios=[2.7, 1.8, 1.8],  # top panel taller
        hspace=1.1,                    # more vertical gap to avoid overlap
        wspace=0.15
    )

    # Top: ANNUAL — keep full model names, rotated 90°
    ax_top = fig.add_subplot(gs[0, :])
    x, labels, tss, bvs, hps = prepare_for_season(ranked_full, ANNUAL, sort_by_hps=True, top_n=None, use_names=True)
    plot_rank_panel(
        ax_top, x, labels, tss, bvs, hps,
        title='Annual Historical Performance — TSS, BVS, and HPS',
        compact=False, show_legend=True,
        rotate_labels=90, label_fontsize=9.5
    )

    # Seasons (compact: IDs only), rotate vertical to match style and save space
    axes_seasonal = [fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]),
                     fig.add_subplot(gs[2,0]), fig.add_subplot(gs[2,1])]
    for ax, s in zip(axes_seasonal, SEASONS):
        x, labels, tss, bvs, hps = prepare_for_season(ranked_full, s, sort_by_hps=True, top_n=None, use_names=False)
        plot_rank_panel(
            ax, x, labels, tss, bvs, hps,
            title=f'{s} Historical Performance',
            compact=True, show_legend=False,
            rotate_labels=90, label_fontsize=9
        )

    plt.tight_layout()

    return fig


# --------------------------------------------------------------------------
# Cell 12 — Annual HM historical performance (single-season variant of cell 13)
# --------------------------------------------------------------------------

def fig_annual_HM_hist_perf(
    ranked_full: pd.DataFrame,
    model_ids: dict,
) -> plt.Figure:
    # --- choose season and (optionally) limit to top N models ---
    season = 'annual'   # one of: 'annual','DJF','MAM','JJA','SON'
    top_n  = None       # e.g. set to 25 to show top 25

    # columns we need from ranked_full
    cols_needed = [f'{season}_TSS_mm', f'{season}_bias_score_mm', f'{season}_HMperf']
    missing = [c for c in cols_needed if c not in ranked_full.columns]
    if missing:
        raise KeyError(f"ranked_full is missing columns: {missing}")

    # Capitalize season nicely for the title
    season_label = season.upper() if season in ['DJF','MAM','JJA','SON'] else season.capitalize()

    # Data to plot (ranked_full is already sorted by annual HMperf in your code)
    df_plot = ranked_full.copy()
    if top_n is not None:
        df_plot = df_plot.head(top_n)

    models = df_plot.index.tolist()

    # Build labels "(ID) Model"
    # model_ids should already be defined: dict {model_name: int_id}
    xtick_labels = [f"({model_ids.get(m, 'NA')}) {m}" for m in models]

    tss   = df_plot[f'{season}_TSS_mm'].to_numpy()
    bias  = df_plot[f'{season}_bias_score_mm'].to_numpy()
    hm    = df_plot[f'{season}_HMperf'].to_numpy()

    x = np.arange(len(models))

    # Figure sizing that scales with number of models
    fig_w = max(12, len(models) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    # Plot (HM emphasized)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # TSS and Bias lines (context)
    ax.plot(x, tss,  marker='o', linestyle='-', linewidth=1.3, markersize=5,
            label='TSS (min–max)', alpha=0.9, zorder=2)
    ax.plot(x, bias, marker='s', linestyle='-', linewidth=1.3, markersize=5,
            label='Bias score (min–max)', alpha=0.9, zorder=2)

    # HM performance (dominant)
    ax.plot(x, hm,   marker='^', linestyle='-', linewidth=2.2, markersize=7,
            label='HM performance', zorder=3)

    # Axes & labels
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Historical Performance Score [0,1]', fontsize=12)
    ax.set_title(f'{season_label} Historical Performance — TSS, Bias Score, and Harmonic Mean',
                 fontsize=14, pad=8)
    ax.margins(x=0.01)

    # Legend
    ax.legend(loc='best', ncol=3, frameon=False)

    plt.tight_layout()

    return fig


# --------------------------------------------------------------------------
# Cell 25 — Seasonal performance (revised) — TAS
# --------------------------------------------------------------------------

def fig_seasonal_perf_revised_tas(
    ranked_full: pd.DataFrame,
    tas_all_perf_metrics: pd.DataFrame,
    tas_cmip6_mon_means: pd.DataFrame,
    tas_observed_mon_means: pd.DataFrame,
    model_ids: dict,
) -> plt.Figure:
    # --- 1. CONFIGURATION & IMPORTS ---
    variable = 'tas'
    LIMIT_BIAS = (0.0, 3.5)       # Zoom in on X-axis
    LIMIT_CORR = (0.978, 1.002)   # Zoom in on Y-axis

    # Try to import adjustText for smart label placement
    try:
        from adjustText import adjust_text
        HAS_ADJUST_TEXT = True
    except ImportError:
        HAS_ADJUST_TEXT = False
        print("Note: 'adjustText' not found. Labels may overlap slightly.")

    # --- 2. DATA PREPARATION ---
    # Choose metric used to color models & the colorbar label
    color_metric = 'rmse'   # 'rmse' | 'corr' | 'bias' | 'tss' | 'hm'
    cmap = mpl.colormaps['autumn_r']

    if color_metric == 'rmse':
        series = tas_all_perf_metrics['annual_rmse']
        cbar_label = 'Annual RMSE'
    elif color_metric == 'corr':
        series = tas_all_perf_metrics['annual_corr']
        cbar_label = 'Annual Correlation'
    elif color_metric == 'bias':
        series = tas_all_perf_metrics['annual_bias'].abs()  # use |bias|
        cbar_label = 'Annual |Bias|'
    elif color_metric == 'tss':
        series = tas_all_perf_metrics['annual_tss']
        cbar_label = 'Annual TSS'
    elif color_metric == 'hm':
        series = ranked_full['annual_HMperf']
        cbar_label = 'Annual HM performance (composite)'
    else:
        raise ValueError("color_metric must be one of: 'rmse','corr','bias','tss','hm'")

    # Normalize on available values; gray if a model is missing from the series
    valid = series.dropna()
    norm = Normalize(vmin=valid.min(), vmax=valid.max())

    def color_for(model):
        val = series.get(model, np.nan)
        return cmap(norm(val)) if np.isfinite(val) else (0.85, 0.85, 0.85, 1.0)

    model_colors = {m: color_for(m) for m in tas_cmip6_mon_means.columns}


    # --- 3. FIGURE LAYOUT ---
    fig = plt.figure(figsize=(10, 11))
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.4, 1.0, 1.0],   # main row taller, seasonal rows shorter
        hspace=0.25, wspace=0.2
    )

    # Big panel spanning first row
    ax_main = fig.add_subplot(gs[0, :])

    # Four seasonal panels
    ax_djf = fig.add_subplot(gs[1, 0])
    ax_mam = fig.add_subplot(gs[1, 1])
    ax_jja = fig.add_subplot(gs[2, 0])
    ax_son = fig.add_subplot(gs[2, 1])

    # Make seasonal axes ~3:2 (width:height)
    for _ax in (ax_djf, ax_mam, ax_jja, ax_son):
        if hasattr(_ax, "set_box_aspect"):
            _ax.set_box_aspect(9/14)


    # --- 4. MAIN PLOT (Annual Cycle Lines) ---
    for col in tas_cmip6_mon_means.columns:
        color = model_colors.get(col, '0.6')
        ax_main.plot(
            tas_cmip6_mon_means.index,
            tas_cmip6_mon_means[col].values,
            label=f"({model_ids[col]}) {col}",
            linewidth=0.8,
            color=color,
        )

    # Observed
    ax_main.plot(
        tas_observed_mon_means.index,
        tas_observed_mon_means['tas'].values,
        label='Observed',
        color='black',
        linewidth=2.5,
        zorder=500
    )

    ax_main.set_title('Annual Cycle of Temperature')
    ax_main.set_ylabel('Temperature (°C)')
    ax_main.set_xlim(1, 12)
    months_ticks = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    ax_main.set_xticks(np.arange(1, 13))
    ax_main.set_xticklabels(months_ticks)


    # --- 5. SEASONAL SCATTER PANELS (Fixed Outlier Style) ---
    def scatter_panel_optimized(ax, xcol, title, xlabel):
        texts = []

        # Apply strict zoom limits
        ax.set_xlim(LIMIT_BIAS)
        ax.set_ylim(LIMIT_CORR)

        # Grid for readability
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle=':', linewidth=1)

        for model in tas_all_perf_metrics.index:
            true_x = tas_all_perf_metrics.loc[model, xcol]
            true_y = tas_all_perf_metrics.loc[model, 'annual_corr']
            mid = str(model_ids[model])
            color = model_colors.get(model, '0.6')

            # Check if Outlier (outside the zoom box)
            is_x_out = true_x > LIMIT_BIAS[1]
            is_y_out = true_y < LIMIT_CORR[0]

            if is_x_out or is_y_out:
                # --- OUTLIER HANDLING ---
                # 1. Dock coordinates to the edge
                plot_x = min(true_x, LIMIT_BIAS[1])
                plot_y = max(true_y, LIMIT_CORR[0])

                # 2. Plot the "Container" Circle (White background with black edge)
                ax.scatter(plot_x, plot_y, color="white", s=350, edgecolors='k', linewidth=1, zorder=10, )

                # 3. Plot the ID Number INSIDE the circle
                # **CRITICAL FIX**: Use the model's color + black outline (same as main models)
                t_out = ax.text(plot_x, plot_y, mid, fontsize=9,
                                color=color,  # Use model color, NOT black
                                ha='center', va='center', fontweight='bold', zorder=11)

                # Add the consistent black outline
                t_out.set_path_effects([PathEffects.withStroke(linewidth=1.2, foreground='black')])

                # 4. Annotation Label (Small box with values)
                annot_text = "("
                if is_x_out: annot_text += f"B:{true_x:.1f}"
                if is_x_out and is_y_out: annot_text += ", "
                if is_y_out: annot_text += f"r:{true_y:.2f}"
                annot_text += ")"

                xytext_offset = (-28, 22) if is_x_out else (10, 17)

                ax.annotate(annot_text, xy=(plot_x, plot_y), xytext=xytext_offset,
                            textcoords='offset points', fontsize=7.5, color='black',
                            arrowprops=dict(arrowstyle="->", color='black', lw=0.6),
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.8, ec='none'))

            else:
                # --- MAIN CLUSTER (Text ID Markers) ---
                # Invisible dot for anchor
                ax.scatter(true_x, true_y, s=1, color=color, alpha=0)

                # Text ID with Outline
                t = ax.text(true_x, true_y, mid, fontsize=9,
                            color=color, ha='center', va='center', fontweight='bold')
                # Black outline around colored text
                t.set_path_effects([PathEffects.withStroke(linewidth=1.2, foreground='black')])
                texts.append(t)

        # Apply Repulsion (adjustText)
        if HAS_ADJUST_TEXT and texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='dimgray', lw=0.5), expand_points=(1.2, 1.2))

        ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)

    # Execute the 4 panels
    ax_djf.set_ylabel('Annual Correlation')
    scatter_panel_optimized(ax_djf, 'DJF_bias', 'DJF', '')

    scatter_panel_optimized(ax_mam, 'MAM_bias', 'MAM', '')

    ax_jja.set_ylabel('Annual Correlation')
    scatter_panel_optimized(ax_jja, 'JJA_bias', 'JJA', 'Absolute Bias')

    scatter_panel_optimized(ax_son, 'SON_bias', 'SON', 'Absolute Bias')


    # --- 6. LEGEND & COLORBAR ---
    # Create Legend Handles
    legend_handles = [
        Line2D([0], [0], color=model_colors.get(col, '0.6'), marker='o', linestyle='',
               markersize=8, label=f"({model_ids[col]}) {col}")
        for col in tas_cmip6_mon_means.columns
    ]
    legend_handles.insert(0, Line2D([0], [0], color='black', marker='o', linestyle='',
                                    markersize=6, label='Observed (GSWP3-W5E5)'))

    # Place Legend
    ax_main.legend(handles=legend_handles,
                   bbox_to_anchor=(1.02, 1.02), # Anchored top-left corner
                   loc='upper left',
                   borderaxespad=0.,
                   frameon=False,
                   handlelength=0.5,
                   fontsize=8.5,
                   labelspacing=0.999) # Adjusted to fill vertical space

    # Bottom margins for colorbar
    fig.subplots_adjust(bottom=0.12)

    # Colorbar
    cax = fig.add_axes([0.12, 0.06, 0.783, 0.015])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label(cbar_label)

    return fig


# --------------------------------------------------------------------------
# Cell 26 — Seasonal performance (revised) — PR
# --------------------------------------------------------------------------

def fig_seasonal_perf_revised_pr(
    ranked_full: pd.DataFrame,
    pr_all_perf_metrics: pd.DataFrame,
    pr_cmip6_mon_means: pd.DataFrame,
    pr_observed_mon_means: pd.DataFrame,
    model_ids: dict,
) -> plt.Figure:
    # --- 1. CONFIGURATION ---
    variable = 'pr'

    # Zoom limits for PR (given by you)
    LIMIT_BIAS = (-0.05, 1.28)
    LIMIT_CORR = (0.75, 0.95)

    # Only model 26 is an outlier (given by you)
    OUTLIER_MODEL_ID = 26

    # Try to import adjustText for smart label placement
    try:
        from adjustText import adjust_text
        HAS_ADJUST_TEXT = True
    except ImportError:
        HAS_ADJUST_TEXT = False
        print("Note: 'adjustText' not found. Labels may overlap slightly.")

    # --- 2. DATA PREPARATION ---
    color_metric = 'rmse'   # 'rmse' | 'corr' | 'bias' | 'tss' | 'hm'
    cmap = mpl.colormaps['YlGnBu']  # pr-specific palette

    if color_metric == 'rmse':
        series = pr_all_perf_metrics['annual_rmse']
        cbar_label = 'Annual RMSE'
    elif color_metric == 'corr':
        series = pr_all_perf_metrics['annual_corr']
        cbar_label = 'Annual Correlation'
    elif color_metric == 'bias':
        series = pr_all_perf_metrics['annual_bias'].abs()
        cbar_label = 'Annual |Bias|'
    elif color_metric == 'tss':
        series = pr_all_perf_metrics['annual_tss']
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

    model_colors = {m: color_for(m) for m in pr_cmip6_mon_means.columns}

    # Helper: find model name from a model ID number (e.g., 26)
    def model_name_from_id(target_id):
        for name, mid in model_ids.items():
            if mid == target_id:
                return name
        return None

    OUTLIER_MODEL_NAME = model_name_from_id(OUTLIER_MODEL_ID)

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
    for col in pr_cmip6_mon_means.columns:
        color = model_colors.get(col, '0.6')
        ax_main.plot(
            pr_cmip6_mon_means.index,
            pr_cmip6_mon_means[col].values,
            label=f"({model_ids[col]}) {col}",
            linewidth=0.8,
            color=color,
        )

    ax_main.plot(
        pr_observed_mon_means.index,
        pr_observed_mon_means['pr'].values,
        label='Observed',
        color='black',
        linewidth=2.5,
        zorder=500
    )

    ax_main.set_title('Annual Cycle of Precipitation')
    ax_main.set_ylabel('Precipitation (mm/day)')
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
        ax.axvline(x=0.0, color='black', linestyle=':', linewidth=1)  # <- optional reference for ratio bias
        # If your PR bias is absolute difference (not ratio), change this back to x=0.

        for model in pr_all_perf_metrics.index:
            true_x = pr_all_perf_metrics.loc[model, xcol]
            true_y = pr_all_perf_metrics.loc[model, 'annual_corr']
            mid = str(model_ids[model])
            color = model_colors.get(model, '0.6')

            # Outlier logic: enforce your statement "only Model 26 is outlier"
            is_outlier = (OUTLIER_MODEL_NAME is not None and model == OUTLIER_MODEL_NAME)

            # Also treat anything outside the clamp as out-of-range (safety)
            is_x_out = (true_x < LIMIT_BIAS[0]) or (true_x > LIMIT_BIAS[1])
            is_y_out = (true_y < LIMIT_CORR[0]) or (true_y > LIMIT_CORR[1])
            is_out = is_outlier or is_x_out or is_y_out

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
                if true_x < LIMIT_BIAS[0] or true_x > LIMIT_BIAS[1]:
                    parts.append(f"B:{true_x:.2f}")
                if true_y < LIMIT_CORR[0] or true_y > LIMIT_CORR[1]:
                    parts.append(f"r:{true_y:.2f}")
                annot_text = "(" + ", ".join(parts) + ")" if parts else ""
                # Offset: push left if docked on right edge, otherwise right
                xytext_offset = (-42, 32) if plot_x >= LIMIT_BIAS[1] else (8, 15.5)

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
    legend_handles = [
        Line2D([0], [0], color=model_colors.get(col, '0.6'),
               marker='o', linestyle='', markersize=8,
               label=f"({model_ids[col]}) {col}")
        for col in pr_cmip6_mon_means.columns
    ]
    legend_handles.insert(
        0,
        Line2D([0], [0], color='black', marker='o', linestyle='',
               markersize=6, label='Observed (GSWP3-W5E5)')
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
# Cell 27 — Seasonal performance (revised) — PSL
# --------------------------------------------------------------------------

def fig_seasonal_perf_revised_psl(
    ranked_full: pd.DataFrame,
    psl_all_perf_metrics: pd.DataFrame,
    psl_cmip6_mon_means: pd.DataFrame,
    psl_observed_mon_means: pd.DataFrame,
    model_ids: dict,
) -> plt.Figure:
    # --- 1. CONFIGURATION ---
    variable = 'psl'

    # Zoom limits for PSL (given by you)
    LIMIT_BIAS = (-0.1, 5.0)
    LIMIT_CORR = (0.8, 1.0)

    # Try to import adjustText for smart label placement
    try:
        from adjustText import adjust_text
        HAS_ADJUST_TEXT = True
    except ImportError:
        HAS_ADJUST_TEXT = False
        print("Note: 'adjustText' not found. Labels may overlap slightly.")

    # --- 2. DATA PREPARATION ---
    color_metric = 'rmse'   # 'rmse' | 'corr' | 'bias' | 'tss' | 'hm'
    cmap = mpl.colormaps['winter_r']  # psl-specific palette

    if color_metric == 'rmse':
        series = psl_all_perf_metrics['annual_rmse']
        cbar_label = 'Annual RMSE'
    elif color_metric == 'corr':
        series = psl_all_perf_metrics['annual_corr']
        cbar_label = 'Annual Correlation'
    elif color_metric == 'bias':
        series = psl_all_perf_metrics['annual_bias'].abs()
        cbar_label = 'Annual |Bias|'
    elif color_metric == 'tss':
        series = psl_all_perf_metrics['annual_tss']
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

    model_colors = {m: color_for(m) for m in psl_cmip6_mon_means.columns}

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
    for col in psl_cmip6_mon_means.columns:
        color = model_colors.get(col, '0.6')
        ax_main.plot(
            psl_cmip6_mon_means.index,
            psl_cmip6_mon_means[col].values,
            label=f"({model_ids[col]}) {col}",
            linewidth=0.8,
            color=color,
        )

    ax_main.plot(
        psl_observed_mon_means.index,
        psl_observed_mon_means['psl'].values,
        label='Observed',
        color='black',
        linewidth=2.5,
        zorder=500
    )

    ax_main.set_title('Annual Cycle of Sea-Level Pressure')
    ax_main.set_ylabel('Sea-level pressure (hPa)')
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

        for model in psl_all_perf_metrics.index:
            true_x = psl_all_perf_metrics.loc[model, xcol]
            true_y = psl_all_perf_metrics.loc[model, 'annual_corr']
            mid = str(model_ids[model])
            color = model_colors.get(model, '0.6')

            # Treat anything outside the clamp as out-of-range
            is_x_out = (true_x < LIMIT_BIAS[0]) or (true_x > LIMIT_BIAS[1])
            is_y_out = (true_y < LIMIT_CORR[0]) or (true_y > LIMIT_CORR[1])
            is_out = is_x_out or is_y_out

            if is_out:
                # Dock to nearest edge of the zoom box
                plot_x = min(max(true_x, LIMIT_BIAS[0]), LIMIT_BIAS[1])
                plot_y = min(max(true_y, LIMIT_CORR[0]), LIMIT_CORR[1])

                # White container circle at the edge
                ax.scatter(plot_x, plot_y, color="white", s=380,
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
                    parts.append(f"r:{true_y:.2f}")
                annot_text = "(" + ", ".join(parts) + ")" if parts else ""

                # Offset: push left if docked on right edge, otherwise right
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
    legend_handles = [
        Line2D([0], [0], color=model_colors.get(col, '0.6'),
               marker='o', linestyle='', markersize=8,
               label=f"({model_ids[col]}) {col}")
        for col in psl_cmip6_mon_means.columns
    ]
    legend_handles.insert(
        0,
        Line2D([0], [0], color='black', marker='o', linestyle='',
               markersize=6, label='Observed (W5E5)')
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
# Cell 28 — Seasonal performance (revised) — TASMAX
# --------------------------------------------------------------------------

def fig_seasonal_perf_revised_tasmax(
    ranked_full: pd.DataFrame,
    tasmax_all_perf_metrics: pd.DataFrame,
    tasmax_cmip6_mon_means: pd.DataFrame,
    tasmax_observed_mon_means: pd.DataFrame,
    model_ids: dict,
    ordered_models: list = None,
) -> plt.Figure:
    # --- 1. CONFIGURATION ---
    variable = 'tasmax'

    # Zoom limits (given by you)
    LIMIT_BIAS = (0.0, 6.5)
    LIMIT_CORR = (0.98, 1.00)

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
# Cell 29 — Annual Taylor diagram (per variable; verbatim cell loop)
# --------------------------------------------------------------------------

def fig_annual_taylor_per_variable(
    variables: Iterable[str],
    perf_metrics: Dict[str, pd.DataFrame],
    observed_std_dev_df: pd.DataFrame,
    cmip6_models: pd.DataFrame,
    model_ids: dict,
) -> Dict[str, plt.Figure]:
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
# Cell 32 — 4-season Taylor diagram (per variable; verbatim cell loop)
# --------------------------------------------------------------------------

def fig_4season_taylor_per_variable(
    variables: Iterable[str],
    perf_metrics: Dict[str, pd.DataFrame],
    observed_std_dev_df: pd.DataFrame,
    cmip6_models: pd.DataFrame,
    model_ids: dict,
) -> Dict[str, plt.Figure]:
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
# Cell 34 — Bias maps per variable (verbatim cell loop)
# loop period set is kwarg-controlled per user direction; cell 34 had seasonal
# entries commented out.
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
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import cmocean.cm as cmo  # noqa: F401  (kept verbatim from cell)
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

        # Shapefile — adapter passes a GADM 4.1 gpkg path; legacy cell hardcoded
        # `U:/OneDrive/Shapefiles/GADM_levels/countries`. We filter the country
        # polygon out of the GPKG instead of a per-country .shp file.
        gdf = gpd.read_file(shapefile_path)
        country_boundaries = gdf[gdf["COUNTRY"].str.lower() == country.lower()]
        if country_boundaries.empty:
            raise FileNotFoundError(f"Country {country} not found in {shapefile_path}")

        minx, miny, maxx, maxy = country_boundaries.total_bounds
        aspect_ratio = (maxx - minx) / max(1e-6, (maxy - miny))

        # Grid sizing
        n_models = len(bias_maps[period][variable])
        n_columns = 6
        n_rows = math.ceil(n_models / n_columns) + 2  # +2 rows for the observed panel

        base_height = 1.8
        base_width  = 0.7 * (base_height * aspect_ratio)
        fig_width   = base_width * n_columns
        fig_height  = base_height * n_rows - 2.5

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
