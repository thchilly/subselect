"""Spread figures — verbatim ports from
``legacy/cmip6-greece/GR_model_spread.ipynb``.

Cell-to-function map: see ``scripts/m9_cell_map.md``. Each function reproduces
its source cell byte-for-byte; the only deviations are documented in the cell
map (data inputs come from the M8 cache / paper-era xlsx via
``_data_adapters``, terminal ``save_figure(...)`` / ``plt.show()`` is replaced
with ``return fig``, and imports are hoisted to module level).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # noqa: F401  (kept verbatim)
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

CATEGORY = "spread"


# --------------------------------------------------------------------------
# Cell 13 — Annual future spread, rev12
# --------------------------------------------------------------------------

def fig_annual_spread_rev12(
    ranked_full: pd.DataFrame,
    long_pi_change_df: pd.DataFrame,
    long_term_df: pd.DataFrame,
    cmip6_models: pd.DataFrame,
    model_ids: dict,
    country: str = "greece",
) -> plt.Figure:
    # --- CONFIGURATION START ---

    # 1. SWITCH: Turn Euro-CORDEX labels ON or OFF
    SHOW_EURO_CORDEX_LABELS = True  # Set to True to see labels, False to hide them

    # 2. POSITION EDITING: Configure specific positions for labels here.
    #    Format: 'ModelName': (Angle_Degrees, Length_Factor)
    #    Angle: 0 is Right, 90 is Top, 180 is Left, 270 is Bottom
    #    Length_Factor: Distance from circle (0.6 is standard)
    euro_cordex_label_positions = {
        'MIROC6':        (45, 0.6),
        'MPI-ESM1-2-HR': (45, 0.6),
        'CNRM-ESM2-1':   (45, 0.6),
        'CMCC-CM2-SR5':  (25, 0.6),
        'NorESM2-MM':    (75, 0.4),
        'EC-Earth3-Veg': (90, 0.6)
    }

    # 3. Y-AXIS EXTENSION: Percent to extend the lower limit (0.05 = 5%)
    LOWER_YLIM_EXT_PCT = 0.03

    # --- CONFIGURATION END ---

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        base = plt.get_cmap(cmap)
        colors = base(np.linspace(minval, maxval, n))
        return LinearSegmentedColormap.from_list(
            f'trunc({base.name},{minval:.2f},{maxval:.2f})', colors
        )

    inferno_trunc = truncate_colormap('inferno', 0.15, 0.90)
    season = 'annual'

    # ---- Euro-CORDEX Driving Models (for Bold Legend & Continuous Outline) ----
    euro_cordex_models = [
        'CMCC-CM2-SR5', 'CNRM-ESM2-1', 'EC-Earth3-Veg', 'MIROC6',
        'MPI-ESM1-2-HR', 'NorESM2-MM'
    ]

    # ---- HM performance colors ----
    series = ranked_full[f'{season}_HMperf'].astype(float)
    cmap = inferno_trunc
    norm = Normalize(vmin=0.0, vmax=1.0)

    # ---- tasmax ensemble median ----
    tasmax_median = long_pi_change_df[f'tasmax_{season}'].median(skipna=True)

    # =========================
    # SIZE MAPPING
    S_BASE       = 420
    SIZE_SCALE   = 1.4
    GAMMA        = 2.25
    S_MIN, S_MAX = 80, 1800
    # =========================

    def size_map(ratio: float) -> float:
        if not np.isfinite(ratio):
            return S_BASE * SIZE_SCALE
        ratio = max(ratio, 0.0)
        s = S_BASE * SIZE_SCALE * (ratio ** GAMMA)
        return float(np.clip(s, S_MIN, S_MAX))

    # Outline geometry
    S_OUTLINE = size_map(1.0)

    # =========================
    # OUTLINE STYLE UPDATES
    # =========================
    OUTLINE_LW = 1.2
    OUTLINE_LS_DEFAULT = (0, (3, 1.6))
    OUTLINE_LS_CORDEX  = '-'
    Z_OUTLINE  = 12

    # =========================
    # LABEL GEOMETRY DEFAULTS
    # =========================
    DEFAULT_ANGLE_DEG   = 45.0
    DEFAULT_LEN_FACTOR  = 0.60
    TEXT_PAD_FACTOR     = 0.25
    LINE_LW             = 1.1

    def ring_label_points_angle(ax, center_xy, s_outline, angle_deg,
                                len_factor, text_pad_factor):
        """
        Compute three points in DATA coordinates for the label connector.
        """
        r_pts = np.sqrt(s_outline / np.pi)
        r_px  = r_pts * (ax.figure.dpi / 72.0)

        len_px      = max(1.0, r_px * float(len_factor))
        text_pad_px = r_px * float(text_pad_factor)

        theta = np.deg2rad(angle_deg)
        u = np.array([np.cos(theta), np.sin(theta)])
        n = np.array([-u[1], u[0]])

        to_disp = ax.transData.transform
        to_data = ax.transData.inverted().transform

        c_disp    = np.array(to_disp(center_xy))
        ring_disp = c_disp + u * r_px
        end_disp  = ring_disp + u * len_px
        text_disp = end_disp + n * text_pad_px

        return tuple(to_data(ring_disp)), tuple(to_data(end_disp)), tuple(to_data(text_disp))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Quadrant guides
    median_temp_change = long_pi_change_df[f'tas_{season}'].median(skipna=True)
    ax.axvline(median_temp_change, color='black', linestyle='--', linewidth=1, zorder=1)
    median_precip_change = long_term_df[f'pr_{season}'].median(skipna=True)
    ax.axhline(median_precip_change, color='black', linestyle='--', linewidth=1, zorder=1)

    legend_handles_map = {}
    last_colored_scatter = None

    for model_name in cmip6_models['model'].tolist():
        model_id = model_ids[model_name]

        tx = long_pi_change_df.loc[model_name, f'tas_{season}'] if model_name in long_pi_change_df.index else np.nan
        py = long_term_df.loc[model_name, f'pr_{season}']       if model_name in long_term_df.index else np.nan
        if not (np.isfinite(tx) and np.isfinite(py)):
            continue

        hm = series.get(model_name, np.nan)
        d_tasmax = long_pi_change_df.loc[model_name, f'tasmax_{season}'] if model_name in long_pi_change_df.index else np.nan

        has_tasmax = np.isfinite(d_tasmax)

        if np.isfinite(tasmax_median) and (tasmax_median > 0):
            ratio = d_tasmax / tasmax_median if has_tasmax else np.nan
        else:
            ratio = 1.0

        s_fill = size_map(ratio)
        alpha_fill = 1.0

        # Filled Bubble
        if np.isfinite(hm):
            scatter = ax.scatter(tx, py, marker='o', s=s_fill, c=[hm], cmap=cmap, norm=norm,
                                 zorder=8, alpha=alpha_fill)
            last_colored_scatter = scatter
            label_color = 'white'
            facecol = cmap(norm(hm)); edgecol = facecol
        else:
            ax.scatter(tx, py, marker='o', s=s_fill, facecolors='none', edgecolors='0.7',
                       zorder=8, alpha=alpha_fill)
            label_color = '0.4'
            facecol = 'none'; edgecol = '0.7'

        # Number inside
        ax.text(tx, py, model_id, fontsize=10, ha='center', va='center',
                color=label_color, fontweight='bold', zorder=9)

        # Outline Logic
        if has_tasmax:
            current_ls = OUTLINE_LS_CORDEX if model_name in euro_cordex_models else OUTLINE_LS_DEFAULT

            ax.scatter(tx, py, marker='o', s=S_OUTLINE,
                       facecolors='none', edgecolors='black',
                       linewidths=OUTLINE_LW, linestyles=current_ls,
                       zorder=Z_OUTLINE)

        # Label Logic (Using Switch and Dictionary)
        if SHOW_EURO_CORDEX_LABELS and (model_name in euro_cordex_label_positions):
            # Retrieve custom position if available, otherwise use defaults
            angle, len_fact = euro_cordex_label_positions.get(model_name, (DEFAULT_ANGLE_DEG, DEFAULT_LEN_FACTOR))

            ring_xy, end_xy, text_xy = ring_label_points_angle(
                ax, (tx, py), S_OUTLINE, angle,
                len_fact, TEXT_PAD_FACTOR
            )
            ax.annotate(
                model_name,
                xy=ring_xy, xytext=text_xy,
                fontsize=9, color='black',
                ha='left', va='bottom',
                zorder=Z_OUTLINE + 2,
                arrowprops=dict(arrowstyle='-', color='black', linewidth=LINE_LW,
                                shrinkA=0, shrinkB=0, connectionstyle="arc3")
            )

        # Legend Handle Logic
        font_props = dict(size=12)
        if model_name in euro_cordex_models:
            font_props['weight'] = 'bold'

        handle = Line2D([0], [0],
                        marker=f'${model_id}$', linestyle='', color='none',
                        markerfacecolor=(facecol if facecol != 'none' else 'white'),
                        markeredgecolor=edgecol,
                        markersize=(10 if model_id < 10 else 14),
                        label=model_name)

        legend_handles_map[model_name] = {'handle': handle, 'is_cordex': model_name in euro_cordex_models}

    # Colorbar
    if last_colored_scatter is not None:
        cbar = plt.colorbar(last_colored_scatter)
        cbar.set_label('HM Historical Performance Score', fontsize=13, labelpad=10)

    ax.yaxis.set_tick_params(labelsize=11)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.set_title(f'Annual Future Spread over {country.capitalize()} [2081–2100, SSP5-8.5]', fontsize=15)
    ax.set_xlabel('Temperature Change (°C)', fontsize=13)
    ax.set_ylabel('Precipitation (mm/day)', fontsize=13)

    # ----------------------------------------------------
    # APPLY Y-LIMIT EXTENSION (Before placing quadrant labels)
    # ----------------------------------------------------
    current_ymin, current_ymax = ax.get_ylim()
    y_range = current_ymax - current_ymin
    new_ymin = current_ymin - (y_range * LOWER_YLIM_EXT_PCT)
    ax.set_ylim(bottom=new_ymin)
    # ----------------------------------------------------

    # Quadrant labels (Calculate range based on NEW limits)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    x_range = xlims[1] - xlims[0]
    y_range = ylims[1] - ylims[0]

    pad_x = 0.03 * x_range
    pad_y = 0.03 * y_range

    ax.text(xlims[0] + pad_x, ylims[1] - pad_y, 'Cooler-Wetter',
            fontsize=12, fontstyle='italic', ha='left', va='top', alpha=0.8, zorder=-1000)
    ax.text(xlims[1] - pad_x, ylims[1] - pad_y, 'Warmer-Wetter',
            fontsize=12, fontstyle='italic', ha='right', va='top', alpha=0.8, zorder=-1000)
    ax.text(xlims[0] + pad_x, ylims[0] + pad_y, 'Cooler-Drier',
            fontsize=12, fontstyle='italic', ha='left', va='bottom', alpha=0.8, zorder=-1000)
    ax.text(xlims[1] - pad_x, ylims[0] + pad_y, 'Warmer-Drier',
            fontsize=12, fontstyle='italic', ha='right', va='bottom', alpha=0.8, zorder=-1000)

    # Legend Construction
    hm_order = series.sort_values(ascending=False).index.tolist()
    ordered_models = [m for m in hm_order if m in legend_handles_map]
    remaining = [m for m in legend_handles_map if m not in ordered_models]
    ordered_models += remaining

    ordered_handles = [legend_handles_map[m]['handle'] for m in ordered_models]
    ordered_labels  = [legend_handles_map[m]['handle'].get_label() for m in ordered_models]

    leg = fig.legend(handles=ordered_handles, labels=ordered_labels, title='CMIP6 Models',
               loc='upper right', bbox_to_anchor=(1.55, 0.9),
               numpoints=1, prop=dict(size=12), fancybox=False,
               edgecolor='white', ncol=2, title_fontsize=12)

    # APPLY BOLD TO LEGEND TEXT for CORDEX models
    for text in leg.get_texts():
        if text.get_text() in euro_cordex_models:
            text.set_weight('bold')

    # NB: legacy cell 13 calls plt.tight_layout() AFTER save_figure(...), so the
    # paper-era PNG is the pre-tight_layout figure. Returning fig here without
    # the tight_layout call matches that behaviour.
    return fig


# --------------------------------------------------------------------------
# Cell 21 — Seasonal future spread, perSeasonBars + named (rev1)
# --------------------------------------------------------------------------

def fig_seasonal_spread_perSeasonBars_right_named_rev1(
    ranked_full: pd.DataFrame,
    long_pi_change_df: pd.DataFrame,
    long_term_df: pd.DataFrame,
    cmip6_models: pd.DataFrame,
    model_ids: dict,
    country: str = "greece",
) -> plt.Figure:
    # ---------- helpers ----------
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        base = plt.get_cmap(cmap)
        colors = base(np.linspace(minval, maxval, n))
        return LinearSegmentedColormap.from_list(
            f'trunc({base.name},{minval:.2f},{maxval:.2f})', colors
        )

    def size_map_factory(S_BASE=420, SIZE_SCALE=1.4, GAMMA=2.25, S_MIN=80, S_MAX=1800):
        def size_map(ratio: float) -> float:
            if not np.isfinite(ratio):
                return S_BASE * SIZE_SCALE
            ratio = max(ratio, 0.0)
            s = S_BASE * SIZE_SCALE * (ratio ** GAMMA)
            return float(np.clip(s, S_MIN, S_MAX))
        return size_map

    # ---- EURO-CORDEX drivers ----
    EURO_CORDEX = {
        'CMCC-CM2-SR5', 'CNRM-ESM2-1', 'EC-Earth3-Veg',
        'MIROC6', 'MPI-ESM1-2-HR', 'NorESM2-MM'
    }

    # dashed ring style (Standard CMIP6)
    OUTLINE_LW = 0.8
    OUTLINE_LS = (0, (3, 1.6))
    Z_OUTLINE  = 12

    # solid ring style (EURO-CORDEX)
    EURO_LW = 1.2
    EURO_LS = 'solid'
    Z_EURO  = 12 # Same Z-order, just different style now

    # color map
    cmap = truncate_colormap('viridis', 0.15, 0.90)

    # ---------- one-axis plotter ----------
    def plot_season_panel_with_cbar(ax, season,
                                    ranked_full, long_pi_change_df, long_term_df,
                                    cmip6_models, model_ids, size_map, fig):

        # quadrant guides (season medians)
        med_dtas = long_pi_change_df[f'tas_{season}'].median(skipna=True)
        med_pr   = long_term_df[f'pr_{season}'].median(skipna=True)
        ax.axvline(med_dtas, color='black', linestyle='--', linewidth=1, zorder=1)
        ax.axhline(med_pr,   color='black', linestyle='--', linewidth=1, zorder=1)

        # per-season tasmax median for bubble size
        tasmax_median = long_pi_change_df[f'tasmax_{season}'].median(skipna=True)
        S_OUTLINE = size_map(1.0)

        # per-season HM values → per-season norm
        hm_vals = ranked_full[f'{season}_HMperf'].astype(float).values
        hm_vals = hm_vals[np.isfinite(hm_vals)]
        if hm_vals.size == 0:
            norm = Normalize(vmin=0, vmax=1)
        else:
            vmin, vmax = float(hm_vals.min()), float(hm_vals.max())
            if vmin == vmax:
                vmin -= 1e-6; vmax += 1e-6
            norm = Normalize(vmin=vmin, vmax=vmax)

        last_scatter = None

        for model_name in cmip6_models['model'].tolist():
            mid = model_ids[model_name]

            # coordinates
            tx = long_pi_change_df.loc[model_name, f'tas_{season}'] if model_name in long_pi_change_df.index else np.nan
            py = long_term_df.loc[model_name, f'pr_{season}']       if model_name in long_term_df.index else np.nan
            if not (np.isfinite(tx) and np.isfinite(py)):
                continue

            # color: HM for THIS season
            hm = ranked_full.loc[model_name, f'{season}_HMperf'] if model_name in ranked_full.index else np.nan

            # size: Δtasmax ratio vs seasonal median
            dtx   = long_pi_change_df.loc[model_name, f'tasmax_{season}'] if model_name in long_pi_change_df.index else np.nan

            has_tasmax = np.isfinite(dtx)

            ratio = (dtx / tasmax_median) if (has_tasmax and np.isfinite(tasmax_median) and tasmax_median > 0) else np.nan
            s_fill = size_map(ratio)

            # Transparency: Set to 1.0 (opaque) for everyone so colors are discernable
            alpha_fill = 1.0

            if np.isfinite(hm):
                sc = ax.scatter(tx, py, s=s_fill, c=[hm], cmap=cmap, norm=norm,
                                zorder=8, alpha=alpha_fill)
                last_scatter = sc
                num_color = 'white'
            else:
                ax.scatter(tx, py, s=s_fill, facecolors='none', edgecolors='0.7',
                           zorder=8, alpha=alpha_fill)
                num_color = '0.4'

            # ID inside
            ax.text(tx, py, mid, fontsize=9, ha='center', va='center',
                    color=num_color, fontweight='bold', zorder=9)

            # --- OUTLINE LOGIC ---
            if has_tasmax:
                if model_name in EURO_CORDEX:
                    # Solid ring for Euro-CORDEX
                    ax.scatter(tx, py, s=S_OUTLINE, facecolors='none', edgecolors='black', alpha=0.9,
                               linewidths=EURO_LW, linestyles=EURO_LS, zorder=Z_EURO)
                else:
                    # Dashed ring for standard CMIP6
                    ax.scatter(tx, py, s=S_OUTLINE, facecolors='none', edgecolors='black', alpha=0.7,
                               linewidths=OUTLINE_LW, linestyles=OUTLINE_LS, zorder=Z_OUTLINE)
            # else: NO OUTLINE if missing tasmax (Reviewer suggestion)

        # cosmetics
        ax.set_title(f'{season.upper()} Spread', fontsize=13)
        ax.set_xlabel('Temperature Change (°C)', fontsize=11)
        ax.set_ylabel('Precipitation (mm/day)', fontsize=11)
        ax.tick_params(labelsize=9)

        # --- AXIS LIMITS (Surgical Y-limit adjustments per season) ---
        # Get current auto-scaled limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]

        # Default: No change
        new_xlim = xlim
        new_ylim = ylim

        # Apply specific adjustments
        if season == 'DJF':
            new_ylim = (ylim[0], ylim[1] + y_range * 0.09)
        elif season == 'JJA':
            new_ylim = (ylim[0] - y_range * 0.06, ylim[1] + y_range * 0.12)
        elif season == 'MAM':
            new_ylim = (ylim[0] - y_range * 0.02, ylim[1])
        elif season == 'SON':
            new_ylim = (ylim[0] - y_range * 0.12, ylim[1])

        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)

        # --- QUADRANT LABELS (Positioned using new limits) ---
        # Adjust padding slightly for better placement
        pad_x = 0.03 * (new_xlim[1] - new_xlim[0])
        pad_y = 0.02 * (new_ylim[1] - new_ylim[0])

        ax.text(new_xlim[0] + pad_x, new_ylim[1] - pad_y,
                'Cooler-Wetter', fontsize=10, fontstyle='italic', ha='left', va='top', alpha=0.8, zorder=-1000)
        ax.text(new_xlim[1] - pad_x, new_ylim[1] - pad_y,
                'Warmer-Wetter', fontsize=10, fontstyle='italic', ha='right', va='top', alpha=0.8, zorder=-1000)
        ax.text(new_xlim[0] + pad_x, new_ylim[0] + pad_y,
                'Cooler-Drier',  fontsize=10, fontstyle='italic', ha='left',  va='bottom', alpha=0.8, zorder=-1000)
        ax.text(new_xlim[1] - pad_x, new_ylim[0] + pad_y,
                'Warmer-Drier',  fontsize=10, fontstyle='italic', ha='right', va='bottom', alpha=0.8, zorder=-1000)

        # --- vertical colorbar OUTSIDE axis ---
        if last_scatter is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.08)
            cb = fig.colorbar(last_scatter, cax=cax, orientation='vertical')
            cb.ax.tick_params(labelsize=8)
            cb.set_label(f"Seasonal Historical Performance Score for {season.upper()}", fontsize=9)

    # ---------- build the 2×2 figure ----------
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    size_map = size_map_factory(S_BASE=390, SIZE_SCALE=1.4, GAMMA=2.25, S_MIN=80, S_MAX=1800)

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    # leave room so outer colorbars don't get clipped
    plt.subplots_adjust(wspace=0.35, hspace=0.3, right=0.92)

    #title
    plt.suptitle(f"Seasonal Future Spread over {country.capitalize()} [2081–2100, SSP5-8.5]", fontsize=15)

    for ax, s in zip(axs.flatten(), seasons):
        plot_season_panel_with_cbar(ax, s,
                                    ranked_full, long_pi_change_df, long_term_df,
                                    cmip6_models, model_ids, size_map, fig)

    return fig
