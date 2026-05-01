"""Country-profile figures — verbatim ports from
``legacy/climpact/CLIMPACT_figures.ipynb``.

Cell-to-function map: see ``scripts/m9_cell_map.md``. Each function reproduces
its source cell byte-for-byte; the only deviations are documented in the cell
map: data inputs come from M8 cache / paper-era xlsx via ``_data_adapters``,
``save_figure(...)`` and the duplicate ``plt.savefig(...)`` calls are stripped
(Q4) so the entry-point script writes the canonical ``{country}_<name>.png``
target, ``plt.show()`` is dropped, and ``pd.read_excel(os.path.join(analysis_path, ...))``
calls are dropped (the values come from the state namespace instead).

State-spec cells (29, 31, 32, 35, 42, 44, 46, 49, 50, 54) take a
``state: dict`` parameter — the namespace built by
``_data_adapters.build_climpact_state``, which runs CLIMPACT_figures cells
[13, 15, 19, 21, 24, 26, 39, 41] verbatim against the paper-era xlsx
artefacts. The cell body is exec'd against a namespace containing every
climpact-state variable plus the stdlib/numpy/matplotlib symbols the cells
expect, so cell code references like ``stats_tas_21yr_ma_rp_anomaly``,
``anomalies_rp_table_df``, ``baseline_offset`` resolve directly.

# Q4: duplicate save_figure(file_name=...) call removed per user direction;
# psl_ prefix was stale.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

CATEGORY = "country_profile"

# Repo-relative legacy notebook path; resolved once at module load.
_REPO = Path(__file__).resolve().parents[2]
_CLIMPACT_NB = json.loads((_REPO / "legacy" / "climpact" / "CLIMPACT_figures.ipynb").read_text())


def _strip_terminal_calls(src: str) -> str:
    """Apply M9 verbatim deviations to a cell source (drop save_figure /
    plt.savefig / plt.show / output-path setup / pd.read_excel(analysis_path))."""
    lines = src.split("\n")
    out_lines = []
    i = 0
    import re as _re
    while i < len(lines):
        ln = lines[i]
        if _re.match(r"^(\s*)save_figure\(", ln):
            depth = 0; j = i
            while j < len(lines):
                depth += lines[j].count("(") - lines[j].count(")")
                if depth <= 0 and j > i: break
                if depth <= 0 and "(" in lines[j] and ")" in lines[j]: break
                j += 1
            i = j + 1; continue
        if _re.match(r"^(\s*)(plt|fig)\.savefig\(", ln):
            depth = 0; j = i
            while j < len(lines):
                depth += lines[j].count("(") - lines[j].count(")")
                if depth <= 0 and j > i: break
                if depth <= 0 and "(" in lines[j] and ")" in lines[j]: break
                j += 1
            i = j + 1; continue
        if _re.match(r"^\s*plt\.show\(\s*\)\s*$", ln):
            i += 1; continue
        if _re.match(r"^\s*(plots_output_path|svg_plots_output_path)\s*=", ln):
            i += 1; continue
        if _re.match(r"^\s*if not os\.path\.exists\((plots_output_path|svg_plots_output_path)\):", ln):
            i += 1
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].strip() == ""):
                if lines[i].strip() == "": i += 1; break
                if "os.makedirs" in lines[i]: i += 1; break
                i += 1
            continue
        if _re.match(r"^\s*os\.makedirs\((plots_output_path|svg_plots_output_path)", ln):
            i += 1; continue
        m_read = _re.match(r"^\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*pd\.read_excel\(\s*os\.path\.join\((analysis_path|base_path)", ln)
        if m_read:
            depth = 0; j = i
            while j < len(lines):
                depth += lines[j].count("(") - lines[j].count(")")
                if depth <= 0 and j > i: break
                if depth <= 0 and "(" in lines[j] and ")" in lines[j]: break
                j += 1
            i = j + 1; continue
        out_lines.append(ln)
        i += 1
    return "\n".join(out_lines).rstrip()


def _exec_state_cell(cell_idx: int, state: dict, country: str) -> plt.Figure:
    """Run a state-spec cell against `state` and return the resulting fig."""
    src = "".join(_CLIMPACT_NB["cells"][cell_idx].get("source", []))
    src = _strip_terminal_calls(src)
    ns = {
        "__builtins__": __builtins__,
        "pd": pd, "np": np, "os": os, "plt": plt, "mpl": mpl,
        "matplotlib": matplotlib, "mlines": mlines, "mpatches": mpatches,
        "ticker": ticker, "fm": fm, "HandlerPatch": HandlerPatch,
        "Normalize": Normalize, "LinearSegmentedColormap": LinearSegmentedColormap,
        "Line2D": Line2D, "FuncFormatter": FuncFormatter, "MaxNLocator": MaxNLocator,
        "make_axes_locatable": make_axes_locatable,
        "country": country,
    }
    ns.update(state)
    exec(compile(src, f"<climpact_cell_{cell_idx}>", "exec"), ns)
    # Some legacy cells (e.g. 31, 32, 35, 44, 46, 49, 50, 54) build the figure
    # via the global pyplot interface without binding it to a `fig` variable.
    # Fall back to `plt.gcf()` in that case.
    return ns.get("fig") or plt.gcf()


# --------------------------------------------------------------------------
# Cell 17 — fig_WL_table
# --------------------------------------------------------------------------

def fig_WL_table(
    warming_level_medians,
    country='greece',
) -> plt.Figure:
    # Reformatting the DataFrame for the table
    wl_table_df = pd.DataFrame(index=warming_level_medians.index)

    for ssp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
        # Create new column with formatted values
        wl_table_df[ssp.replace('ssp', 'SSP').replace('126', '1-2.6').replace('245', '2-4.5').replace('370', '3-7.0').replace('585', '5-8.5')] = \
            warming_level_medians[f'{ssp}_wl'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-") + \
            " (" + warming_level_medians[f'{ssp}_models'].astype(str) + ")"

    wl_table_df.index = [index.replace('WL_', '') for index in wl_table_df.index]
    wl_table_df.index.name = 'Warming Levels'

    col_label_colors = ['#5a6d85', '#f1bd60', '#e25f69', '#b5545e']

    # Create a figure and axis for the table
    fig, ax = plt.subplots(figsize=(9, 4))  # Adjust size as needed
    ax.axis('off')
    #ax.axis('tight')

    # Add a table at the bottom of the Axes
    table = ax.table(cellText=wl_table_df.values, colLabels=wl_table_df.columns, rowLabels=wl_table_df.index, loc='center', cellLoc='center', rowLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(13)  # Adjust font size as needed
    table.scale(1.2, 2.5)  # Adjust table scale if needed

    w, h = table[0,1].get_width(), table[0,1].get_height()
    table.add_cell(0, -1, w,h, text=wl_table_df.index.name)

    # Remove vertical borders from all cells below the header
    for key, cell in table.get_celld().items():
        if key[0] > 0:  # Apply only if the row index is greater than 0
            cell.visible_edges = 'horizontal'

    # Apply colors and settings to header cells
    col_label_colors = ['#5a6d85', '#f1bd60', '#e25f69', '#b5545e']
    for j, color in enumerate(col_label_colors):
        header_cell = table[0, j]
        header_cell.set_facecolor(color)
        header_cell.set_edgecolor('black')
        header_cell.set_linewidth(1)
        header_cell.set_text_props(fontsize=11, weight='bold')

    # Add a title cell for row labels
    w, h = table[0, 1].get_width(), table[0, 1].get_height()
    table.add_cell(0, -1, w, h, text=wl_table_df.index.name)


    # Create the directory if it does not exist

    # Create the directory if it does not exist



    # plt.savefig(f'{plots_output_path}/{country.lower()}_WL_table.png', bbox_inches='tight', dpi=300)
    # plt.savefig(f'{svg_plots_output_path}/{country.lower()}_WL_table.svg', bbox_inches='tight')







    # caption = (
    #     f"Table 1: The table outlines the median years when {country.capitalize()} is expected to reach global warming "
    #     f"levels of 1.5°C, 2.0°C, 3.0°C, and 4.0°C, under the Shared Socioeconomic Pathways (SSP1-2.6, SSP2-4.5, "
    #     f"SSP3-7.0, SSP5-8.5). The numbers in parentheses represent the count of CMIP6 models aligning with each "
    #     f"median projection, illustrating the scientific consensus on the timing of warming milestones for {country.capitalize()}."
    # )
    caption = (
        f"Table 1: Year (ensemble median) by which {country.capitalize()} is projected to reach temperature anomaly "
        f"thresholds of 1.5°C, 2.0°C, 3.0°C, and 4.0°C above pre-industrial levels (1850–1900), according to SSP scenarios ranging from "
        f"SSP1-2.6 to SSP5-8.5. Values in parentheses indicate the number of CMIP6 models predicting these thresholds within the century. "
        f"A lower count suggests a result more influenced by the warmer projections of specific models."
    )


    print(caption)
    return fig


# --------------------------------------------------------------------------
# Cell 58 — fig_gwls_boxplot
# --------------------------------------------------------------------------

def fig_gwls_boxplot(
    warming_levels_all_models,
    warming_level_medians,
    global_warming_level_medians,
    country='greece',
) -> plt.Figure:
    ### GWL HORIZONTAL BOXPLOT ###



    # Set the global font to be Roboto, if available
    #plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'sans-serif'

    # Plot settings for text and title
    title_fontsize = 17
    label_fontsize = 16  # Font size for axis labels
    legend_fontsize = 13
    tick_label_fontsize = 14  # Font size for tick labels
    shape_label_fontsize = 14  # Font size for shape labels
    major_tick_length = 10  # Length of major ticks
    minor_tick_length = 6  # Length of minor ticks


    # Set colors for each SSP scenario
    ssp_colors = {'ssp126': '#1D3758', 'ssp245': '#ECA525', 'ssp370': '#D72331', 'ssp585': '#991422'}

    # Create a figure with 4 subplots, sharing the x-axis
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # Define the WL labels
    wl_labels = ['WL_+1.5°C', 'WL_+2.0°C', 'WL_+3.0°C', 'WL_+4.0°C']
    ssp_scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

    # Iterate over each subplot to create boxplots for each WL
    for i, wl in enumerate(wl_labels):
        ax = axes[i]
        wl_data = warming_levels_all_models.loc[wl].dropna()
        ssp_data = [wl_data.filter(regex=ssp).values for ssp in ssp_scenarios]

        # Create boxplot
        bp = ax.boxplot(ssp_data, vert=False, patch_artist=True, positions=range(1, 5), showfliers=False)

 
        # Customizing boxplot colors and median markers, and setting whisker colors
        for patch, color in zip(bp['boxes'], ssp_colors.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
            patch.set_edgecolor(color)
        
    
        for whisker, color in zip(bp['whiskers'], [val for val in ssp_colors.values() for _ in (0, 1)]):
            whisker.set_color(color)
        for median, color in zip(bp['medians'], ssp_colors.values()):
            median.set_color(color)
            median.set_linewidth(2.5)  # Increase linewidth (adjust value as needed)


            # Set cap colors to match the SSP colors
        for cap, color in zip(bp['caps'], [val for val in ssp_colors.values() for _ in (0, 1)]):
            cap.set_color(color)

        # Add data points with opacity
        for j, scenario in enumerate(ssp_data):
            ax.scatter(scenario, [j+1]*len(scenario), alpha=0.2, color=ssp_colors[ssp_scenarios[j]])

        # Add labels with median year and count of models
        for j, ssp in enumerate(ssp_scenarios):
            median_year = warming_level_medians.loc[wl, f'{ssp}_wl']
            model_count = warming_level_medians.loc[wl, f'{ssp}_models']
            ax.text(2116, j+1, f'{median_year:.0f} ({model_count})', va='center', ha='right', fontsize=13)


        if country != 'global':
                # Add global median as a black marker
            for j, ssp in enumerate(ssp_scenarios):
                global_median_year = global_warming_level_medians.loc[wl, f'{ssp}_wl']
                ax.scatter(global_median_year, j+1, color='black', marker='D', s=20, zorder=10)  # Diamond shape
    

        # Set subplot title inside the plot area
        ax.text(0.01, 0.95, f'{wl[:2]} {wl[3:]}', transform=ax.transAxes, va='top', ha='left', fontsize=14)

        ax.set_yticks([])

    
        # Add dotted lines every 20 years
        for year in range(1980, 2101, 20):
            ax.axvline(x=year, color='grey', linestyle=':', linewidth=0.5)
    

    # Set common x-axis properties and adjust spacing between subplots
    axes[-1].set_xlim(1980, 2100)
    axes[-1].set_xticks(range(1980, 2101, 20))
    axes[-1].tick_params(axis='x', labelsize=tick_label_fontsize)  # Increase font size for x-axis tick labels
    plt.subplots_adjust(hspace=0)

    # Title 
    title_position_y = 0.91  
    if country == 'global':
        fig.suptitle(f'{country.capitalize()} warming levels', fontsize=title_fontsize, y=title_position_y)
    else:
        fig.suptitle(f'{country.capitalize()} warming levels', fontsize=title_fontsize, y=title_position_y)


    # Modify labels for SSP scenarios
    ssp_labels = {
        'ssp126': 'SSP1-2.6',
        'ssp245': 'SSP2-4.5',
        'ssp370': 'SSP3-7.0',
        'ssp585': 'SSP5-8.5'
    }

    # Adjust legend styling and position
    legend_handles = [matplotlib.patches.Patch(color=color, label=ssp_labels[ssp], alpha=0.75) for ssp, color in ssp_colors.items()]


    if country != 'global':
        # Define the legend handle for the global median marker
        global_median_handle = mlines.Line2D([], [], color='black', marker='D', linestyle='None',
                                             markersize=5, label='Global WL')
    
        # Add the global median handle to the existing list of handles for SSP scenarios
        legend_handles.append(global_median_handle)



    # Reverse the order of legend handles
    legend_handles = list(reversed(legend_handles))

    bottom_ax = axes[-1]  # The bottom subplot

    # Place the reversed legend in the bottom left corner of the bottom subplot
    bottom_ax.legend(handles=legend_handles, loc='lower left', fontsize=12, fancybox=False, edgecolor='white')




    # Create the directory if it does not exist
    
    # Create the directory if it does not exist




    # Show



    # Detailed caption for the figure
    caption = (
        f"Figure Z: Warming levels for {country.capitalize()}, showcasing the years each CMIP6 model projects crossing specific global temperature "
        f"thresholds (+1.5°C, +2.0°C, +3.0°C, and +4.0°C) with regard to the pre-industrial period (1850–1900). The box and whisker plots represent "
        f"the spread of years across models under different SSP scenarios, with the median value indicated by a horizontal bold line. Numbers to the "
        f"right of each warming level denote the median year (across all models) reaching the respective temperature threshold, with the count of "
        f"contributing models in parentheses. Black diamonds represent the global median year for each scenario and threshold, providing a reference "
        f"for {country.capitalize()}'s warming relative to global projections."
    )


    ssp_scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    wl_key_points = ['WL_+1.5°C', 'WL_+3.0°C']

    # Detailed paragraph describing the figure's findings
    paragraph = (
        f"As the global community navigates through the unfolding narrative of climate change, {country.capitalize()} stands as a poignant case study. "
        f"In the context of global warming, {country.capitalize()} provides a critical perspective on reaching significant temperature thresholds. "
        f"Under SSP1-2.6, the scenario that aligns with stringent mitigation strategies, the median projection for reaching the 1.5°C warming level occurs by "
        f"{int(warming_level_medians.at['WL_+1.5°C', 'ssp126_wl'])}, with {warming_level_medians.at['WL_+1.5°C', 'ssp126_models']} models concurring. "
        f"For the more severe warming level of 3.0°C under the same scenario, the median year is projected as {int(warming_level_medians.at['WL_+3.0°C', 'ssp126_wl'])} "
        f"by {warming_level_medians.at['WL_+3.0°C', 'ssp126_models']} models. On the other hand, under the high-emission scenario of SSP5-8.5, the median years for reaching "
        f"the 1.5°C and 3.0°C thresholds are {int(warming_level_medians.at['WL_+1.5°C', 'ssp585_wl'])} and {int(warming_level_medians.at['WL_+3.0°C', 'ssp585_wl'])}, "
        f"with the model consensus at {warming_level_medians.at['WL_+1.5°C', 'ssp585_models']} and {warming_level_medians.at['WL_+3.0°C', 'ssp585_models']} models respectively. "
        f"These projections illustrate the range of potential future climates for {country.capitalize()}, reflecting the critical impact of policy decisions and emission pathways. "
        f"Comparatively, the global median years, marked by the black diamonds, serve as a benchmark, indicating how the regional climate trajectory of {country.capitalize()} "
        f"aligns with global trends, thus highlighting the shared challenges and the necessity for global cooperation in climate change adaptation and mitigation."
    )

    # Country Warms Faster Than Global Average
    faster_wl = (
        f"The observed warming levels, manifesting earlier than the global median, signify an accelerated climate response within this region. "
        f"The rate of warming surpasses global projections, underscoring an imperative for immediate and decisive climate policies. "
        f"This trend not only underscores the vulnerability of {country.capitalize()} to a changing climate but also amplifies the call "
        f"for tailored adaptation measures that can mitigate the risks associated with such rapid environmental transformations."
    )

    # Country Warms Slower Than Global Average
    slower_wl = (
        f"The projection of warming levels occurring later than the global median suggests a comparatively moderate rate of climate change "
        f"for this region. While this may imply a temporal advantage, it is a reminder of the window of opportunity for {country.capitalize()} "
        f"to establish preemptive adaptation strategies. It provides a critical period for enhancing infrastructural resilience and ecological "
        f"safeguards to protect against future climate variability and extremes."
    )  

    print(caption + "\n\n" + paragraph)
    print(faster_wl)
    return fig


# --------------------------------------------------------------------------
# Cell 61 — fig_gwls_boxplot_times
# --------------------------------------------------------------------------

def fig_gwls_boxplot_times(
    warming_levels_all_models,
    warming_level_medians,
    global_warming_level_medians,
    country='greece',
) -> plt.Figure:
    ### GWL HORIZONTAL BOXPLOT ###



    # Set the global font to be Roboto, if available
    plt.rcParams['font.family'] = 'Times New Roman'
    #plt.rcParams['font.family'] = 'sans-serif'

    # Plot settings for text and title
    title_fontsize = 19
    label_fontsize = 16  # Font size for axis labels
    legend_fontsize = 13
    tick_label_fontsize = 15  # Font size for tick labels
    shape_label_fontsize = 14  # Font size for shape labels
    major_tick_length = 10  # Length of major ticks
    minor_tick_length = 6  # Length of minor ticks


    # Set colors for each SSP scenario
    ssp_colors = {'ssp126': '#1D3758', 'ssp245': '#ECA525', 'ssp370': '#D72331', 'ssp585': '#991422'}

    # Create a figure with 4 subplots, sharing the x-axis
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # Define the WL labels
    wl_labels = ['WL_+1.5°C', 'WL_+2.0°C', 'WL_+3.0°C', 'WL_+4.0°C']
    ssp_scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

    # Iterate over each subplot to create boxplots for each WL
    for i, wl in enumerate(wl_labels):
        ax = axes[i]
        wl_data = warming_levels_all_models.loc[wl].dropna()
        ssp_data = [wl_data.filter(regex=ssp).values for ssp in ssp_scenarios]

        # Create boxplot
        bp = ax.boxplot(ssp_data, vert=False, patch_artist=True, positions=range(1, 5), showfliers=False)

        # Customizing boxplot colors and median markers, and setting whisker colors
        for patch, color in zip(bp['boxes'], ssp_colors.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
            patch.set_edgecolor(color)
        
        for whisker, color in zip(bp['whiskers'], [val for val in ssp_colors.values() for _ in (0, 1)]):
            whisker.set_color(color)
        
        for median, color in zip(bp['medians'], ssp_colors.values()):
            median.set_color(color)
            median.set_linewidth(2.5)  # Increase linewidth (adjust value as needed)


            # Set cap colors to match the SSP colors
        for cap, color in zip(bp['caps'], [val for val in ssp_colors.values() for _ in (0, 1)]):
            cap.set_color(color)

        # Add data points with opacity
        for j, scenario in enumerate(ssp_data):
            ax.scatter(scenario, [j+1]*len(scenario), alpha=0.2, color=ssp_colors[ssp_scenarios[j]])

        # Add labels with median year and count of models
        for j, ssp in enumerate(ssp_scenarios):
            median_year = warming_level_medians.loc[wl, f'{ssp}_wl']
            model_count = warming_level_medians.loc[wl, f'{ssp}_models']
            ax.text(2114.5, j+1, f'{median_year:.0f} ({model_count})', va='center', ha='right', fontsize=15)

        if country != 'global':
                # Add global median as a black marker
            for j, ssp in enumerate(ssp_scenarios):
                global_median_year = global_warming_level_medians.loc[wl, f'{ssp}_wl']
                ax.scatter(global_median_year, j+1, color='black', marker='D', s=20, zorder=10)  # Diamond shape

        # Set subplot title inside the plot area
        ax.text(0.01, 0.95, f'{wl[:2]} {wl[3:]}', transform=ax.transAxes, va='top', ha='left', fontsize=15)

        ax.set_yticks([])

    
        # Add dotted lines every 20 years
        for year in range(1980, 2101, 20):
            ax.axvline(x=year, color='grey', linestyle=':', linewidth=0.5)
    

    # Set common x-axis properties and adjust spacing between subplots
    axes[-1].set_xlim(1980, 2100)
    axes[-1].set_xticks(range(1980, 2101, 20))
    axes[-1].tick_params(axis='x', labelsize=tick_label_fontsize)  # Increase font size for x-axis tick labels
    plt.subplots_adjust(hspace=0)

    # Title 
    title_position_y = 0.91  
    if country == 'global':
        fig.suptitle(f'{country.capitalize()} warming levels', fontsize=title_fontsize, y=title_position_y)
    else:
        fig.suptitle(f'{country.capitalize()} warming levels', fontsize=title_fontsize, y=title_position_y)


    # Modify labels for SSP scenarios
    ssp_labels = {
        'ssp126': 'SSP1-2.6',
        'ssp245': 'SSP2-4.5',
        'ssp370': 'SSP3-7.0',
        'ssp585': 'SSP5-8.5'
    }


    # Create a custom handler function
    class SquareHandler(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            p = mpatches.Rectangle(xy=(0, 0), width=height, height=height, angle=0)
            p.update_from(orig_handle)
            p.set_transform(trans)
            return [p]

    # Adjust legend styling and position
    legend_handles = [mpatches.Patch(color=color, label=ssp_labels[ssp], alpha=0.75) for ssp, color in ssp_colors.items()]

    if country != 'global':
        # Define the legend handle for the global median marker
        global_median_handle = mlines.Line2D([], [], color='black', marker='D', linestyle='None',
                                             markersize=5, label='Global WL')
    
        # Add the global median handle to the existing list of handles for SSP scenarios
        legend_handles.append(global_median_handle)

    # Reverse the order of legend handles
    legend_handles = list(reversed(legend_handles))

    bottom_ax = axes[-1]  # The bottom subplot

    # Place the reversed legend in the bottom left corner of the bottom subplot with custom handler
    bottom_ax.legend(handles=legend_handles, handler_map={mpatches.Patch: SquareHandler()}, loc='lower left', fontsize=13, fancybox=False, edgecolor='white', handlelength=0.5)





    # Create the directory if it does not exist
    
    # Create the directory if it does not exist





    # Show
    return fig


# --------------------------------------------------------------------------
# State-spec cells — exec'd against the climpact state namespace
# --------------------------------------------------------------------------


def fig_tas_anomalies_table(state: dict, country: str = 'greece') -> plt.Figure:
    """Cell 29 of CLIMPACT_figures.ipynb, executed verbatim against ``state``."""
    return _exec_state_cell(29, state, country)


def fig_tas_change(state: dict, country: str = 'greece') -> plt.Figure:
    """Cell 31 of CLIMPACT_figures.ipynb, executed verbatim against ``state``."""
    return _exec_state_cell(31, state, country)


def fig_tas_change_all_shaded(state: dict, country: str = 'greece') -> plt.Figure:
    """Cell 32 of CLIMPACT_figures.ipynb, executed verbatim against ``state``."""
    return _exec_state_cell(32, state, country)


def fig_tas_change_spaghetti(state: dict, country: str = 'greece') -> plt.Figure:
    """Cell 35 of CLIMPACT_figures.ipynb, executed verbatim against ``state``."""
    return _exec_state_cell(35, state, country)


def fig_pr_percent_anomalies_table(state: dict, country: str = 'greece') -> plt.Figure:
    """Cell 42 of CLIMPACT_figures.ipynb, executed verbatim against ``state``."""
    return _exec_state_cell(42, state, country)


def fig_pr_change(state: dict, country: str = 'greece') -> plt.Figure:
    """Cell 44 of CLIMPACT_figures.ipynb, executed verbatim against ``state``."""
    return _exec_state_cell(44, state, country)


def fig_pr_change_spaghetti(state: dict, country: str = 'greece') -> plt.Figure:
    """Cell 46 of CLIMPACT_figures.ipynb, executed verbatim against ``state``."""
    return _exec_state_cell(46, state, country)


def fig_pr_percent_change_ratio(state: dict, country: str = 'greece') -> plt.Figure:
    """Cell 49 of CLIMPACT_figures.ipynb, executed verbatim against ``state``."""
    return _exec_state_cell(49, state, country)


def fig_pr_percent_change_raw(state: dict, country: str = 'greece') -> plt.Figure:
    """Cell 50 of CLIMPACT_figures.ipynb, executed verbatim against ``state``."""
    return _exec_state_cell(50, state, country)


def fig_pr_percent_change_spaghetti(state: dict, country: str = 'greece') -> plt.Figure:
    """Cell 54 of CLIMPACT_figures.ipynb, executed verbatim against ``state``."""
    return _exec_state_cell(54, state, country)
