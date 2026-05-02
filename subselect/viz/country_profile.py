"""Country-profile figures.

Each ``fig_*`` function consumes a :class:`subselect.state.SubselectState` and
returns a :class:`matplotlib.figure.Figure`. The L1/L2 split means these
functions never load data themselves — every artefact they need is on the
state object.

Figure inventory:

- :func:`fig_WL_table` — warming-level crossing-year table.
- :func:`fig_gwls_boxplot` — per-SSP GWL boxplot with country vs global medians.
- :func:`fig_tas_anomalies_table` / :func:`fig_pr_percent_anomalies_table` —
  formatted anomaly tables (country + global rows interleaved).
- ``fig_tas_change`` / ``fig_tas_change_all_shaded`` /
  ``fig_tas_change_spaghetti`` — temperature change time-series figures with
  pre-industrial / recent-past dual y-axes.
- ``fig_pr_change`` / ``fig_pr_change_spaghetti`` /
  ``fig_pr_percent_change_ratio`` / ``fig_pr_percent_change_raw`` /
  ``fig_pr_percent_change_spaghetti`` — precipitation analogues.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerPatch

from subselect.state import SubselectState


CATEGORY = "country_profile"

SSP_COLORS = {
    "ssp126": "#1D3758",
    "ssp245": "#ECA525",
    "ssp370": "#D72331",
    "ssp585": "#991422",
}
SSP_SCENARIOS = list(SSP_COLORS.keys())
FUTURE_PERIODS_SHADING = {
    "Near-term": (2020, 2039.75),
    "Mid-term": (2040.25, 2060),
    "Long-term": (2080, 2100),
}


# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------

def _yticks_int(tick_val: float, _pos=None) -> str:
    if tick_val > 0:
        return f"+{int(tick_val)}"
    if tick_val < 0:
        return f"{int(tick_val)}"
    return "0"


def _yticks_decimal(tick_val: float, _pos=None) -> str:
    formatted = f"{tick_val:.2f}"
    if tick_val > 0:
        return f"+{formatted}"
    if tick_val < 0:
        return formatted
    return "0"


def _legend_position(ax, start_year: int, end_year: int) -> str:
    """Return upper-left vs lower-left depending on whether the leading line
    is positive or negative on average over [start_year, end_year]."""
    y_values = ax.lines[0].get_ydata()
    x_values = ax.lines[0].get_xdata()
    start_index = int(np.where(x_values == start_year)[0][0])
    end_index = int(np.where(x_values == end_year)[0][0])
    return "lower left" if np.mean(y_values[start_index:end_index]) > 0 else "upper left"


def _shade_future_periods(ax2, label_y: float, fontsize: int) -> None:
    for period, (start, end) in FUTURE_PERIODS_SHADING.items():
        plt.fill_betweenx(
            ax2.get_ylim(), start, end, color="black", alpha=0.06,
            edgecolor=None, zorder=1,
        )
        plt.text(
            (start + end) / 2, label_y, period,
            ha="center", va="top", fontsize=fontsize, color="black", zorder=-1,
        )


def _shade_future_periods_va_bottom(ax2, label_y: float, fontsize: int) -> None:
    for period, (start, end) in FUTURE_PERIODS_SHADING.items():
        plt.fill_betweenx(
            ax2.get_ylim(), start, end, color="black", alpha=0.06,
            edgecolor=None, zorder=1,
        )
        plt.text(
            (start + end) / 2, label_y, period,
            ha="center", va="bottom", fontsize=fontsize, color="black", zorder=3,
        )


def _model_counts(columns: Iterable[str]) -> dict[str, int]:
    cols = list(columns)
    return {ssp: sum(ssp in c for c in cols) for ssp in SSP_SCENARIOS}


# ---------------------------------------------------------------------------
# Warming-level table + GWL boxplot (already self-contained — kept as-is)
# ---------------------------------------------------------------------------

def fig_WL_table(state: SubselectState, country: str = "greece") -> plt.Figure:
    """Warming-level crossing-year table per SSP."""
    medians = state.warming_level_medians
    wl_table_df = pd.DataFrame(index=medians.index)
    label_map = {
        "ssp126": "SSP1-2.6",
        "ssp245": "SSP2-4.5",
        "ssp370": "SSP3-7.0",
        "ssp585": "SSP5-8.5",
    }
    for ssp, label in label_map.items():
        wl_table_df[label] = (
            medians[f"{ssp}_wl"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
            + " (" + medians[f"{ssp}_models"].astype(str) + ")"
        )
    wl_table_df.index = [i.replace("WL_", "") for i in wl_table_df.index]
    wl_table_df.index.name = "Warming Levels"

    col_label_colors = ["#5a6d85", "#f1bd60", "#e25f69", "#b5545e"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")

    table = ax.table(
        cellText=wl_table_df.values, colLabels=wl_table_df.columns,
        rowLabels=wl_table_df.index, loc="center", cellLoc="center", rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.2, 2.5)

    w, h = table[0, 1].get_width(), table[0, 1].get_height()
    table.add_cell(0, -1, w, h, text=wl_table_df.index.name)

    for (row, _col), cell in table.get_celld().items():
        if row > 0:
            cell.visible_edges = "horizontal"

    for j, color in enumerate(col_label_colors):
        header_cell = table[0, j]
        header_cell.set_facecolor(color)
        header_cell.set_edgecolor("black")
        header_cell.set_linewidth(1)
        header_cell.set_text_props(fontsize=11, weight="bold")

    table.add_cell(0, -1, w, h, text=wl_table_df.index.name)
    return fig


def fig_gwls_boxplot(state: SubselectState, country: str = "greece") -> plt.Figure:
    """GWL crossing-year boxplots for all 4 SSPs and 4 warming levels."""
    plt.rcParams["font.family"] = "sans-serif"

    title_fontsize = 17
    tick_label_fontsize = 14

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    wl_labels = ["WL_+1.5°C", "WL_+2.0°C", "WL_+3.0°C", "WL_+4.0°C"]

    warming_levels_all_models = state.warming_levels
    warming_level_medians = state.warming_level_medians
    global_warming_level_medians = state.warming_level_medians_global

    for i, wl in enumerate(wl_labels):
        ax = axes[i]
        wl_data = warming_levels_all_models.loc[wl].dropna()
        ssp_data = [wl_data.filter(regex=ssp).values for ssp in SSP_SCENARIOS]

        bp = ax.boxplot(
            ssp_data, vert=False, patch_artist=True,
            positions=range(1, 5), showfliers=False,
        )

        for patch, color in zip(bp["boxes"], SSP_COLORS.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
            patch.set_edgecolor(color)

        for whisker, color in zip(
            bp["whiskers"], [v for v in SSP_COLORS.values() for _ in (0, 1)],
        ):
            whisker.set_color(color)
        for median, color in zip(bp["medians"], SSP_COLORS.values()):
            median.set_color(color)
            median.set_linewidth(2.5)
        for cap, color in zip(
            bp["caps"], [v for v in SSP_COLORS.values() for _ in (0, 1)],
        ):
            cap.set_color(color)

        for j, scenario in enumerate(ssp_data):
            ax.scatter(
                scenario, [j + 1] * len(scenario),
                alpha=0.2, color=SSP_COLORS[SSP_SCENARIOS[j]],
            )

        for j, ssp in enumerate(SSP_SCENARIOS):
            median_year = warming_level_medians.loc[wl, f"{ssp}_wl"]
            model_count = warming_level_medians.loc[wl, f"{ssp}_models"]
            ax.text(
                2116, j + 1, f"{median_year:.0f} ({model_count})",
                va="center", ha="right", fontsize=13,
            )

        if country != "global":
            for j, ssp in enumerate(SSP_SCENARIOS):
                global_median = global_warming_level_medians.loc[wl, f"{ssp}_wl"]
                ax.scatter(
                    global_median, j + 1, color="black", marker="D", s=20, zorder=10,
                )

        ax.text(
            0.01, 0.95, f"{wl[:2]} {wl[3:]}",
            transform=ax.transAxes, va="top", ha="left", fontsize=14,
        )
        ax.set_yticks([])

        for year in range(1980, 2101, 20):
            ax.axvline(x=year, color="grey", linestyle=":", linewidth=0.5)

    axes[-1].set_xlim(1980, 2100)
    axes[-1].set_xticks(range(1980, 2101, 20))
    axes[-1].tick_params(axis="x", labelsize=tick_label_fontsize)
    plt.subplots_adjust(hspace=0)

    fig.suptitle(
        f"{country.capitalize()} warming levels", fontsize=title_fontsize, y=0.91,
    )

    ssp_labels = {
        "ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5",
        "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5",
    }
    legend_handles = [
        matplotlib.patches.Patch(color=color, label=ssp_labels[ssp], alpha=0.75)
        for ssp, color in SSP_COLORS.items()
    ]
    if country != "global":
        legend_handles.append(
            mlines.Line2D(
                [], [], color="black", marker="D", linestyle="None",
                markersize=5, label="Global WL",
            )
        )
    legend_handles = list(reversed(legend_handles))
    axes[-1].legend(
        handles=legend_handles, loc="lower left",
        fontsize=12, fancybox=False, edgecolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Anomaly tables (cells 29 and 42)
# ---------------------------------------------------------------------------

def _render_anomaly_table(
    table_df: pd.DataFrame, *, scale: tuple[float, float], subtitle_rows: list[int],
    horizontal_rows: list[int],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    ax.axis("tight")

    table = ax.table(
        cellText=table_df.values, colLabels=table_df.columns,
        rowLabels=table_df.index, loc="center", cellLoc="center", rowLoc="right",
    )
    table.auto_set_font_size(False)
    table.scale(*scale)

    w, h = table[0, 1].get_width(), table[0, 1].get_height()
    table.add_cell(0, -1, w, h, text=table_df.index.name)
    table.set_fontsize(15)

    for (row, col), cell in table.get_celld().items():
        if col == -1 and row in subtitle_rows:
            cell.set_text_props(fontsize=13, weight="bold")
        if row in horizontal_rows and col in (-1, 0, 1, 2, 3):
            cell.visible_edges = "horizontal"
            cell.visible_edges = "vertical"

    for c in table.get_children():
        c.set_linewidth(0.8)

    col_label_colors = ["#5a6d85", "#f1bd60", "#e25f69", "#b5545e"]
    for j, color in enumerate(col_label_colors):
        header_cell = table[0, j]
        header_cell.set_facecolor(color)
        header_cell.set_edgecolor("black")
        header_cell.set_linewidth(0.7)
        header_cell.set_text_props(fontsize=14, weight="bold")
    return fig


def fig_tas_anomalies_table(state: SubselectState, country: str = "greece") -> plt.Figure:
    """Country + global temperature anomalies, formatted as a table figure."""
    return _render_anomaly_table(
        state.profile_signals.tas_anomalies_table,
        scale=(1.2, 2.5),
        subtitle_rows=[1, 4, 7, 10, 13, 16],
        horizontal_rows=[2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18],
    )


def fig_pr_percent_anomalies_table(state: SubselectState, country: str = "greece") -> plt.Figure:
    """Country + global precipitation percent anomalies, formatted as a table figure."""
    plt.rcParams["font.family"] = "sans-serif"
    return _render_anomaly_table(
        state.profile_signals.pr_percent_anom_table,
        scale=(1.4, 2.5),
        subtitle_rows=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40],
        horizontal_rows=[
            2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21,
            23, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42,
        ],
    )


# ---------------------------------------------------------------------------
# Temperature-change time series (cells 31, 32, 35)
# ---------------------------------------------------------------------------

def _setup_change_axes(
    ylabel_left: str, ylabel_right: str,
    *, label_fontsize: int, tick_label_fontsize: int,
    major_tick_length: int, minor_tick_length: int,
    major_step_left: float, major_step_right: float,
    minor_step_left: float, minor_step_right: float,
    ytick_formatter, baseline_offset: float, ax_ratio: float | None = None,
) -> tuple[plt.Axes, plt.Axes]:
    plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
    plt.xlim(1950, 2100)
    plt.ylabel(ylabel_left, fontsize=label_fontsize)

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.set_ylabel(ylabel_right, fontsize=label_fontsize, rotation=-90, va="bottom")

    ax1_ymin, ax1_ymax = ax1.get_ylim()
    ax2_range = ax1_ymax - ax1_ymin
    ax2_ymax = ax1_ymax + baseline_offset
    ax2_ymin = ax2_ymax - ax2_range
    if ax_ratio is not None:
        ax2.set_ylim(ax2_ymin * ax_ratio, ax2_ymax * ax_ratio)
    else:
        ax2.set_ylim(ax2_ymin, ax2_ymax)
    ax2_yticks = np.arange(np.ceil(ax2_ymin), np.floor(ax2_ymax) + 1, step=1)
    ax2.set_yticks(ax2_yticks)

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(major_step_left))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(ytick_formatter))

    ax2.set_ylim(ax2_ymin, ax2_ymax)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(major_step_right))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(ytick_formatter))

    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(minor_step_left))
    ax1.tick_params(
        axis="both", which="major",
        labelsize=tick_label_fontsize, length=major_tick_length,
    )
    ax1.tick_params(axis="both", which="minor", length=minor_tick_length)
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(minor_step_right))
    ax2.tick_params(
        axis="both", which="major",
        labelsize=tick_label_fontsize, length=major_tick_length,
    )
    ax2.tick_params(axis="both", which="minor", length=minor_tick_length)

    ax2.axhline(y=0, color="black", linestyle=":", linewidth=1)
    ax2.axvline(x=2014, color="black", linestyle="-", linewidth=1)
    return ax1, ax2


def _percent_change_axes(
    ylabel_left: str, ylabel_right: str,
    *, label_fontsize: int, tick_label_fontsize: int,
    major_tick_length: int, minor_tick_length: int,
    pr_baseline_offset_percent: float, ax_ratio: float,
    ytick_formatter,
) -> tuple[plt.Axes, plt.Axes]:
    plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
    plt.xlim(1950, 2100)
    plt.ylabel(ylabel_left, fontsize=label_fontsize)

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.set_ylabel(ylabel_right, fontsize=label_fontsize, rotation=-90, va="bottom")

    ax1_ymin, ax1_ymax = ax1.get_ylim()
    ax2_ymax = (ax1_ymax + pr_baseline_offset_percent) * ax_ratio
    ax2_ymin = (ax1_ymin + pr_baseline_offset_percent) * ax_ratio
    ax2.set_ylim(ax2_ymin, ax2_ymax)
    ax2_yticks = np.arange(np.ceil(ax2_ymin), np.floor(ax2_ymax) + 1, step=1)
    ax2.set_yticks(ax2_yticks)

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(ytick_formatter))
    ax2.set_ylim(ax2_ymin, ax2_ymax)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(ytick_formatter))

    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax1.tick_params(
        axis="both", which="major",
        labelsize=tick_label_fontsize, length=major_tick_length,
    )
    ax1.tick_params(axis="both", which="minor", length=minor_tick_length)
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax2.tick_params(
        axis="both", which="major",
        labelsize=tick_label_fontsize, length=major_tick_length,
    )
    ax2.tick_params(axis="both", which="minor", length=minor_tick_length)

    ax2.axhline(y=0, color="black", linestyle=":", linewidth=1)
    ax2.axvline(x=2014, color="black", linestyle="-", linewidth=1)
    return ax1, ax2


def _draw_change_plot(
    stats_rp: pd.DataFrame,
    *,
    shade_ssps: tuple[str, ...],
    legend_labels: dict[str, str],
    line_width: float = 1.5,
) -> dict[str, plt.Line2D]:
    historical_plot, = plt.plot(
        stats_rp.loc[1870:2014, "ssp126_mean_rp"],
        label="Historical", color="black", linewidth=line_width,
    )
    plt.fill_between(
        stats_rp.loc[1870:2014].index,
        stats_rp.loc[1870:2014, "ssp126_5q_rp"],
        stats_rp.loc[1870:2014, "ssp126_95q_rp"],
        color="black", alpha=0.2, edgecolor=None,
    )

    plot_objects: dict[str, plt.Line2D] = {"Historical": historical_plot}

    for ssp, color in SSP_COLORS.items():
        if ssp in shade_ssps:
            plt.fill_between(
                stats_rp.loc[2014:].index,
                stats_rp.loc[2014:, f"{ssp}_5q_rp"],
                stats_rp.loc[2014:, f"{ssp}_95q_rp"],
                color=color, alpha=0.2, edgecolor=None,
            )

    for ssp, color in SSP_COLORS.items():
        plot_objects[legend_labels[ssp]] = plt.plot(
            stats_rp.loc[2014:, f"{ssp}_mean_rp"],
            color=color, linewidth=line_width,
        )[0]

    return plot_objects


def _draw_spaghetti_plot(
    stats_rp: pd.DataFrame, anomaly_rp: pd.DataFrame,
    *,
    legend_labels: dict[str, str],
    spaghetti_alpha: float = 0.20,
) -> dict[str, plt.Line2D]:
    historical_plot, = plt.plot(
        stats_rp.loc[1870:2014, "ssp126_mean_rp"],
        label="Historical", color="black", linewidth=2.0,
    )

    for ssp in ("ssp126",):
        ssp_data = anomaly_rp[[c for c in anomaly_rp.columns if ssp in c]]
        for col in ssp_data.columns:
            plt.plot(
                ssp_data.loc[1870:2014].index, ssp_data.loc[1870:2014][col],
                color="black", alpha=spaghetti_alpha, linewidth=0.5,
            )

    plot_objects: dict[str, plt.Line2D] = {"Historical": historical_plot}

    for ssp, color in SSP_COLORS.items():
        ssp_data = anomaly_rp[[c for c in anomaly_rp.columns if ssp in c]]
        for col in ssp_data.columns:
            plt.plot(
                ssp_data.loc[2014:].index, ssp_data.loc[2014:][col],
                color=color, alpha=spaghetti_alpha, linewidth=0.5,
            )

    for ssp, color in SSP_COLORS.items():
        plot_objects[legend_labels[ssp]] = plt.plot(
            stats_rp.loc[2014:, f"{ssp}_mean_rp"], color=color, linewidth=2.0,
        )[0]

    return plot_objects


def _change_plot_title(country: str, kind: str, fontsize: int) -> None:
    if country == "global":
        if kind == "tas":
            plt.title(f"{country.capitalize()} surface air temperature change", fontsize=fontsize)
        elif kind == "pr":
            plt.title(f"{country.capitalize()} precipitation change", fontsize=fontsize)
        else:
            plt.title(f"{country.capitalize()} precipitation percent change (%)", fontsize=fontsize)
    else:
        if kind == "tas":
            plt.title(f"Surface air temperature change over {country.capitalize()}", fontsize=fontsize)
        elif kind == "pr":
            plt.title(f"Precipitation change over {country.capitalize()}", fontsize=fontsize)
        else:
            plt.title(f"Precipitation percent change (%) over {country.capitalize()}", fontsize=fontsize)


def fig_tas_change(state: SubselectState, country: str = "greece") -> plt.Figure:
    """Temperature change relative to recent past + pre-industrial dual axis."""
    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(11, 8))

    title_fontsize = 17
    label_fontsize = 16
    legend_fontsize = 13
    tick_label_fontsize = 14
    shape_label_fontsize = 14
    major_tick_length = 10
    minor_tick_length = 6

    ps = state.profile_signals
    counts = _model_counts(ps.tas_anomaly_rp.columns)
    legend_labels = {
        "ssp126": f'SSP1-2.6; 5–95% range ({counts["ssp126"]})',
        "ssp245": f'SSP2-4.5 ({counts["ssp245"]})',
        "ssp370": f'SSP3-7.0; 5–95% range ({counts["ssp370"]})',
        "ssp585": f'SSP5-8.5 ({counts["ssp585"]})',
        "Historical": "Historical",
    }
    plot_objects = _draw_change_plot(
        ps.stats_tas_anomaly_rp,
        shade_ssps=("ssp126", "ssp370"),
        legend_labels=legend_labels,
    )

    ax1, ax2 = _setup_change_axes(
        ylabel_left="Relative to 1995–2014 (°C)",
        ylabel_right="Relative to 1850–1900 (°C)",
        label_fontsize=label_fontsize, tick_label_fontsize=tick_label_fontsize,
        major_tick_length=major_tick_length, minor_tick_length=minor_tick_length,
        major_step_left=1, major_step_right=1,
        minor_step_left=0.5, minor_step_right=0.5,
        ytick_formatter=_yticks_int, baseline_offset=ps.tas_baseline_offset,
    )
    _shade_future_periods(ax2, ax2.get_ylim()[0] * 0.7, shape_label_fontsize)

    ax1.set_zorder(ax1.get_zorder() + 1)
    ax1.patch.set_visible(False)

    legend_pos = _legend_position(ax1, 1900, 2014)
    plt.legend(
        plot_objects.values(), plot_objects.keys(),
        loc=legend_pos, fontsize=legend_fontsize, frameon=False,
    )
    _change_plot_title(country, "tas", title_fontsize)
    return fig


def fig_tas_change_all_shaded(state: SubselectState, country: str = "greece") -> plt.Figure:
    """Temperature change with 5–95% shading on every SSP (all four)."""
    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(11, 8))

    title_fontsize = 17
    label_fontsize = 16
    legend_fontsize = 13
    tick_label_fontsize = 14
    shape_label_fontsize = 14

    ps = state.profile_signals
    counts = _model_counts(ps.tas_anomaly_rp.columns)
    legend_labels = {
        ssp: f'{label}; 5–95% range ({counts[ssp]})'
        for ssp, label in zip(
            SSP_SCENARIOS,
            ("SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"),
        )
    }
    legend_labels["Historical"] = "Historical"
    plot_objects = _draw_change_plot(
        ps.stats_tas_anomaly_rp,
        shade_ssps=tuple(SSP_SCENARIOS),
        legend_labels=legend_labels,
    )

    ax1, ax2 = _setup_change_axes(
        ylabel_left="Relative to 1995–2014 (°C)",
        ylabel_right="Relative to 1850–1900 (°C)",
        label_fontsize=label_fontsize, tick_label_fontsize=tick_label_fontsize,
        major_tick_length=10, minor_tick_length=6,
        major_step_left=1, major_step_right=1,
        minor_step_left=0.5, minor_step_right=0.5,
        ytick_formatter=_yticks_int, baseline_offset=ps.tas_baseline_offset,
    )
    _shade_future_periods(ax2, ax2.get_ylim()[0] * 0.7, shape_label_fontsize)

    ax1.set_zorder(ax1.get_zorder() + 1)
    ax1.patch.set_visible(False)

    legend_pos = _legend_position(ax1, 1900, 2014)
    plt.legend(
        plot_objects.values(), plot_objects.keys(),
        loc=legend_pos, fontsize=legend_fontsize, frameon=False,
    )
    _change_plot_title(country, "tas", title_fontsize)
    return fig


def fig_tas_change_spaghetti(state: SubselectState, country: str = "greece") -> plt.Figure:
    """Temperature change spaghetti plot — every model trace + per-SSP medians."""
    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(11, 8))

    title_fontsize = 17
    label_fontsize = 16
    legend_fontsize = 13
    tick_label_fontsize = 14
    shape_label_fontsize = 14

    ps = state.profile_signals
    counts = _model_counts(ps.tas_anomaly_rp.columns)
    legend_labels = {
        ssp: f'{label} median; ({counts[ssp]} models)'
        for ssp, label in zip(
            SSP_SCENARIOS,
            ("SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"),
        )
    }
    legend_labels["Historical"] = "Historical"
    plot_objects = _draw_spaghetti_plot(
        ps.stats_tas_anomaly_rp, ps.tas_anomaly_rp,
        legend_labels=legend_labels, spaghetti_alpha=0.20,
    )

    ax1, ax2 = _setup_change_axes(
        ylabel_left="Relative to 1995–2014 (°C)",
        ylabel_right="Relative to 1850–1900 (°C)",
        label_fontsize=label_fontsize, tick_label_fontsize=tick_label_fontsize,
        major_tick_length=10, minor_tick_length=6,
        major_step_left=1, major_step_right=1,
        minor_step_left=0.5, minor_step_right=0.5,
        ytick_formatter=_yticks_int, baseline_offset=ps.tas_baseline_offset,
    )
    _shade_future_periods(ax2, ax2.get_ylim()[0] * 0.7, shape_label_fontsize)

    ax1.set_zorder(ax1.get_zorder() + 1)
    ax1.patch.set_visible(False)

    legend_pos = _legend_position(ax1, 1900, 2014)
    plt.legend(
        plot_objects.values(), plot_objects.keys(),
        loc=legend_pos, fontsize=legend_fontsize, frameon=False,
    )
    _change_plot_title(country, "tas", title_fontsize)
    return fig


# ---------------------------------------------------------------------------
# Precipitation absolute change (cells 44, 46)
# ---------------------------------------------------------------------------

def fig_pr_change(state: SubselectState, country: str = "greece") -> plt.Figure:
    """Precipitation change in absolute mm/day with dual baseline axes."""
    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(11, 8))

    title_fontsize = 17
    label_fontsize = 16
    legend_fontsize = 13
    tick_label_fontsize = 14
    shape_label_fontsize = 14

    ps = state.profile_signals
    counts = _model_counts(ps.pr_anomaly_rp.columns)
    legend_labels = {
        "ssp126": f'SSP1-2.6; 5–95% range ({counts["ssp126"]})',
        "ssp245": f'SSP2-4.5 ({counts["ssp245"]})',
        "ssp370": f'SSP3-7.0; 5–95% range ({counts["ssp370"]})',
        "ssp585": f'SSP5-8.5 ({counts["ssp585"]})',
        "Historical": "Historical",
    }
    plot_objects = _draw_change_plot(
        ps.stats_pr_anomaly_rp,
        shade_ssps=("ssp126", "ssp370"),
        legend_labels=legend_labels,
    )

    ax1, ax2 = _setup_change_axes(
        ylabel_left="Relative to 1995–2014 (mm/day)",
        ylabel_right="Relative to 1850–1900 (mm/day)",
        label_fontsize=label_fontsize, tick_label_fontsize=tick_label_fontsize,
        major_tick_length=10, minor_tick_length=6,
        major_step_left=0.1, major_step_right=0.1,
        minor_step_left=0.05, minor_step_right=0.05,
        ytick_formatter=_yticks_decimal, baseline_offset=ps.pr_baseline_offset,
    )
    lowest = ax2.get_ylim()[0]
    _shade_future_periods_va_bottom(
        ax2, lowest + 0.02 * abs(lowest), shape_label_fontsize,
    )

    ax1.set_zorder(ax1.get_zorder() + 1)
    ax1.patch.set_visible(False)

    legend_pos = _legend_position(ax1, 1900, 2014)
    plt.legend(
        plot_objects.values(), plot_objects.keys(),
        loc=legend_pos, fontsize=legend_fontsize, frameon=False,
    )
    _change_plot_title(country, "pr", title_fontsize)
    return fig


def fig_pr_change_spaghetti(state: SubselectState, country: str = "greece") -> plt.Figure:
    """Precipitation change spaghetti plot — every model trace."""
    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=(11, 8))

    title_fontsize = 17
    label_fontsize = 16
    legend_fontsize = 13
    tick_label_fontsize = 14
    shape_label_fontsize = 14

    ps = state.profile_signals
    counts = _model_counts(ps.pr_anomaly_rp.columns)
    legend_labels = {
        ssp: f'{label} median; ({counts[ssp]} models)'
        for ssp, label in zip(
            SSP_SCENARIOS,
            ("SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"),
        )
    }
    legend_labels["Historical"] = "Historical"
    plot_objects = _draw_spaghetti_plot(
        ps.stats_pr_anomaly_rp, ps.pr_anomaly_rp,
        legend_labels=legend_labels, spaghetti_alpha=0.25,
    )

    ax1, ax2 = _setup_change_axes(
        ylabel_left="Relative to 1995–2014 (mm/day)",
        ylabel_right="Relative to 1850–1900 (mm/day)",
        label_fontsize=label_fontsize, tick_label_fontsize=tick_label_fontsize,
        major_tick_length=10, minor_tick_length=6,
        major_step_left=0.1, major_step_right=0.1,
        minor_step_left=0.05, minor_step_right=0.05,
        ytick_formatter=_yticks_decimal,
        baseline_offset=ps.pr_baseline_offset, ax_ratio=ps.pr_ax_ratio,
    )
    lowest = ax2.get_ylim()[0]
    _shade_future_periods_va_bottom(
        ax2, lowest + 0.02 * abs(lowest), shape_label_fontsize,
    )

    ax1.set_zorder(ax1.get_zorder() + 1)
    ax1.patch.set_visible(False)

    legend_pos = _legend_position(ax1, 1900, 2014)
    plt.legend(
        plot_objects.values(), plot_objects.keys(),
        loc=legend_pos, fontsize=legend_fontsize, frameon=False,
    )
    _change_plot_title(country, "pr", title_fontsize)
    return fig


# ---------------------------------------------------------------------------
# Precipitation percent change (cells 49, 50, 54)
# ---------------------------------------------------------------------------

def fig_pr_percent_change_ratio(state: SubselectState, country: str = "greece") -> plt.Figure:
    """Smoothed precipitation percent change with dual axes (ratio-corrected)."""
    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(11, 8))

    title_fontsize = 17
    label_fontsize = 16
    legend_fontsize = 13
    tick_label_fontsize = 14
    shape_label_fontsize = 14

    ps = state.profile_signals
    counts = _model_counts(ps.pr_rp_percent_change.columns)
    legend_labels = {
        "ssp126": f'SSP1-2.6; 5–95% range ({counts["ssp126"]})',
        "ssp245": f'SSP2-4.5 ({counts["ssp245"]})',
        "ssp370": f'SSP3-7.0; 5–95% range ({counts["ssp370"]})',
        "ssp585": f'SSP5-8.5 ({counts["ssp585"]})',
        "Historical": "Historical",
    }
    plot_objects = _draw_change_plot(
        ps.stats_pr_rp_percent_change,
        shade_ssps=("ssp126", "ssp370"),
        legend_labels=legend_labels,
    )

    ax1, ax2 = _percent_change_axes(
        ylabel_left="Relative to 1995–2014 (%)",
        ylabel_right="Relative to 1850–1900 (%)",
        label_fontsize=label_fontsize, tick_label_fontsize=tick_label_fontsize,
        major_tick_length=10, minor_tick_length=6,
        pr_baseline_offset_percent=ps.pr_baseline_offset_percent,
        ax_ratio=ps.pr_ax_ratio,
        ytick_formatter=_yticks_int,
    )
    lowest = ax2.get_ylim()[0]
    _shade_future_periods_va_bottom(
        ax2, lowest + 0.02 * abs(lowest), shape_label_fontsize,
    )

    ax1.set_zorder(ax1.get_zorder() + 1)
    ax1.patch.set_visible(False)

    legend_pos = _legend_position(ax1, 1900, 2014)
    plt.legend(
        plot_objects.values(), plot_objects.keys(),
        loc=legend_pos, fontsize=legend_fontsize, frameon=False,
    )
    _change_plot_title(country, "pr_pct", title_fontsize)
    return fig


def fig_pr_percent_change_raw(state: SubselectState, country: str = "greece") -> plt.Figure:
    """Raw (unsmoothed) precipitation percent change time series.

    Computes percent change from the recent-past baseline directly on the
    annual precipitation series (no rolling-mean smoothing) so the historical
    interannual variability is visible.
    """
    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(11, 8))

    title_fontsize = 17
    label_fontsize = 16
    legend_fontsize = 13
    tick_label_fontsize = 14
    shape_label_fontsize = 14

    ps = state.profile_signals

    raw = (
        ps.annual_precipitation
        .subtract(ps.pr_rp_baseline, axis="columns")
        .div(ps.pr_rp_baseline) * 100
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    stats = pd.DataFrame(index=raw.index)
    for ssp in SSP_SCENARIOS:
        cols = [c for c in raw.columns if ssp in c]
        sub = raw[cols]
        stats[f"{ssp}_mean_rp"] = sub.mean(axis=1)
        stats[f"{ssp}_5q_rp"] = sub.quantile(0.05, axis=1)
        stats[f"{ssp}_95q_rp"] = sub.quantile(0.95, axis=1)

    counts = _model_counts(raw.columns)
    legend_labels = {
        "ssp126": f'SSP1-2.6; 5–95% range ({counts["ssp126"]})',
        "ssp245": f'SSP2-4.5 ({counts["ssp245"]})',
        "ssp370": f'SSP3-7.0; 5–95% range ({counts["ssp370"]})',
        "ssp585": f'SSP5-8.5 ({counts["ssp585"]})',
        "Historical": "Historical",
    }

    historical_plot, = plt.plot(
        stats.loc[1870:2014, "ssp126_mean_rp"],
        label="Historical", color="black", linewidth=1.0,
    )
    plt.fill_between(
        stats.loc[1870:2014].index,
        stats.loc[1870:2014, "ssp126_5q_rp"],
        stats.loc[1870:2014, "ssp126_95q_rp"],
        color="black", alpha=0.15, edgecolor=None,
    )
    plot_objects: dict[str, plt.Line2D] = {"Historical": historical_plot}
    for ssp, color in SSP_COLORS.items():
        if ssp in ("ssp126", "ssp370"):
            plt.fill_between(
                stats.loc[2014:].index,
                stats.loc[2014:, f"{ssp}_5q_rp"],
                stats.loc[2014:, f"{ssp}_95q_rp"],
                color=color, alpha=0.15, edgecolor=None,
            )
    for ssp, color in SSP_COLORS.items():
        plot_objects[legend_labels[ssp]] = plt.plot(
            stats.loc[2014:, f"{ssp}_mean_rp"], color=color, linewidth=1.0,
        )[0]

    ax1, ax2 = _percent_change_axes(
        ylabel_left="Relative to 1995–2014 (%)",
        ylabel_right="Relative to 1850–1900 (%)",
        label_fontsize=label_fontsize, tick_label_fontsize=tick_label_fontsize,
        major_tick_length=10, minor_tick_length=6,
        pr_baseline_offset_percent=ps.pr_baseline_offset_percent,
        ax_ratio=ps.pr_ax_ratio,
        ytick_formatter=_yticks_int,
    )
    lowest = ax2.get_ylim()[0]
    _shade_future_periods_va_bottom(
        ax2, lowest + 0.02 * abs(lowest), shape_label_fontsize,
    )

    ax1.set_zorder(ax1.get_zorder() + 1)
    ax1.patch.set_visible(False)

    legend_pos = _legend_position(ax1, 1900, 2014)
    plt.legend(
        plot_objects.values(), plot_objects.keys(),
        loc=legend_pos, fontsize=legend_fontsize, frameon=False,
    )

    if country == "global":
        plt.title(
            f"{country.capitalize()} precipitation percent change (%) - Raw Annual",
            fontsize=title_fontsize,
        )
    else:
        plt.title(
            f"Precipitation percent change (%) over {country.capitalize()} - Raw Annual",
            fontsize=title_fontsize,
        )
    return fig


def fig_pr_percent_change_spaghetti(state: SubselectState, country: str = "greece") -> plt.Figure:
    """Precipitation percent change spaghetti plot."""
    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=(11, 8))

    title_fontsize = 17
    label_fontsize = 16
    legend_fontsize = 13
    tick_label_fontsize = 14
    shape_label_fontsize = 14

    ps = state.profile_signals
    counts = _model_counts(ps.pr_anomaly_rp.columns)
    legend_labels = {
        ssp: f'{label} median; ({counts[ssp]} models)'
        for ssp, label in zip(
            SSP_SCENARIOS,
            ("SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"),
        )
    }
    legend_labels["Historical"] = "Historical"
    plot_objects = _draw_spaghetti_plot(
        ps.stats_pr_rp_percent_change, ps.pr_rp_percent_change,
        legend_labels=legend_labels, spaghetti_alpha=0.25,
    )

    ax1, ax2 = _percent_change_axes(
        ylabel_left="Relative to 1995-2014 (%)",
        ylabel_right="Relative to 1850–1900 (%)",
        label_fontsize=label_fontsize, tick_label_fontsize=tick_label_fontsize,
        major_tick_length=10, minor_tick_length=6,
        pr_baseline_offset_percent=ps.pr_baseline_offset_percent,
        ax_ratio=ps.pr_ax_ratio,
        ytick_formatter=_yticks_int,
    )
    lowest = ax2.get_ylim()[0]
    _shade_future_periods_va_bottom(
        ax2, lowest + 0.02 * abs(lowest), shape_label_fontsize,
    )

    ax1.set_zorder(ax1.get_zorder() + 1)
    ax1.patch.set_visible(False)

    legend_pos = _legend_position(ax1, 1900, 2014)
    plt.legend(
        plot_objects.values(), plot_objects.keys(),
        loc=legend_pos, fontsize=legend_fontsize, frameon=False,
    )
    _change_plot_title(country, "pr_pct", title_fontsize)
    return fig
