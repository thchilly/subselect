"""Country-profile signal derivations.

These functions consume an annual country-mean time-series DataFrame for a
given variable across all (model, scenario) combinations (1850–2100, columns
named ``<var>_<MODEL>_<variant>_<scenario>_yr``) and produce the derived
quantities the country-profile figures need: pre-industrial / recent-past
baselines, warming-level crossing years, future-period anomaly statistics,
smoothed trajectories, percentile bands, and the formatted anomaly tables.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from subselect.state import ProfileSignals


SCENARIOS: tuple[str, ...] = ("ssp126", "ssp245", "ssp370", "ssp585")
WARMING_LEVELS: dict[str, float] = {
    "WL_+1.5°C": 1.5,
    "WL_+2.0°C": 2.0,
    "WL_+3.0°C": 3.0,
    "WL_+4.0°C": 4.0,
}
PI_BASELINE_WINDOW: tuple[int, int] = (1850, 1899)
RP_BASELINE_WINDOW: tuple[int, int] = (1995, 2014)
FUTURE_PERIODS: dict[str, int] = {
    "Near-term [2021–2040]": 2030,
    "Mid-term [2041–2060]": 2050,
    "Long-term [2081–2100]": 2090,
}
PERIOD_HALF_WIDTH_YEARS = 10  # WL crossings ±10 (21-y window); future periods also ~21-y


def _baseline_mean(annual: pd.DataFrame, window: tuple[int, int]) -> pd.Series:
    start, end = window
    return annual.loc[start:end].mean()


def _ssp_columns(columns: Iterable[str], scenario: str) -> list[str]:
    return [c for c in columns if scenario in c]


# ---------------------------------------------------------------------------
# Warming-level crossings
# ---------------------------------------------------------------------------

def compute_warming_levels(
    annual_temperature: pd.DataFrame,
    *,
    warming_levels: dict[str, float] = WARMING_LEVELS,
    scenarios: Iterable[str] = SCENARIOS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Compute warming-level crossing years per (model, scenario) and per-SSP medians.

    Annual temperatures are smoothed with a centered 21-year rolling mean,
    referenced to the 1850–1899 pre-industrial baseline; the first year the
    smoothed anomaly exceeds each threshold (1.5 / 2.0 / 3.0 / 4.0 °C) is the
    crossing year.

    Returns
    -------
    warming_levels_all_models : DataFrame (warming_level × model_scenario_column)
    warming_level_medians     : DataFrame (warming_level × {ssp_wl, ssp_models})
    pi_baseline               : Series (per model_scenario_column)
    rp_baseline               : Series (per model_scenario_column)
    """
    pi_baseline = _baseline_mean(annual_temperature, PI_BASELINE_WINDOW)
    rp_baseline = _baseline_mean(annual_temperature, RP_BASELINE_WINDOW)

    centered_21yr = annual_temperature.rolling(window=21, min_periods=21, center=True).mean()
    anomaly = centered_21yr.subtract(pi_baseline, axis="columns")

    crossings = pd.DataFrame(
        index=list(warming_levels.keys()), columns=anomaly.columns, dtype=float,
    )
    for wl_label, threshold in warming_levels.items():
        for col in anomaly.columns:
            exceeded = anomaly.index[anomaly[col] > threshold]
            crossings.loc[wl_label, col] = exceeded.min() if len(exceeded) else np.nan

    medians = pd.DataFrame(index=list(warming_levels.keys()))
    for ssp in scenarios:
        ssp_cols = _ssp_columns(crossings.columns, ssp)
        ssp_data = crossings[ssp_cols]
        medians[f"{ssp}_wl"] = ssp_data.median(axis=1).apply(
            lambda x: int(round(x)) if pd.notna(x) else np.nan
        )
        medians[f"{ssp}_models"] = ssp_data.notna().sum(axis=1)

    return crossings, medians, pi_baseline, rp_baseline


# ---------------------------------------------------------------------------
# Future-period anomaly statistics — temperature
# ---------------------------------------------------------------------------

def _period_central_years(
    warming_levels_all_models: pd.DataFrame, columns: pd.Index,
) -> pd.DataFrame:
    """Build the future-period × model-scenario central-year table.

    Rows: three fixed future periods (2030/2050/2090 central years) followed
    by the four warming-level crossing years per model.
    """
    fixed = pd.DataFrame(
        {label: [year] * len(columns) for label, year in FUTURE_PERIODS.items()},
        index=columns,
    ).transpose()
    return pd.concat([fixed, warming_levels_all_models.reindex(columns=columns)])


def _period_window(period: str, central_year: float) -> tuple[int, int] | None:
    if pd.isna(central_year):
        return None
    central = int(central_year)
    if period in FUTURE_PERIODS:
        return central - 9, central + 10
    if period in WARMING_LEVELS:
        return central - PERIOD_HALF_WIDTH_YEARS, central + PERIOD_HALF_WIDTH_YEARS
    return None


def _period_means(
    annual_anomaly: pd.DataFrame, period_central_years: pd.DataFrame,
) -> pd.DataFrame:
    out = pd.DataFrame(
        index=period_central_years.index, columns=period_central_years.columns, dtype=float,
    )
    for period in period_central_years.index:
        for col in period_central_years.columns:
            window = _period_window(period, period_central_years.at[period, col])
            if window is None:
                continue
            start, end = window
            out.at[period, col] = annual_anomaly.loc[start:end, col].mean()
    return out


def _per_ssp_stats(
    period_means: pd.DataFrame, *, scenarios: Iterable[str] = SCENARIOS,
) -> pd.DataFrame:
    metrics = ("models", "mean", "median", "5th", "95th")
    columns = [f"{ssp}_{m}" for ssp in scenarios for m in metrics]
    out = pd.DataFrame(index=period_means.index, columns=columns)
    for ssp in scenarios:
        ssp_cols = _ssp_columns(period_means.columns, ssp)
        sub = period_means[ssp_cols]
        out[f"{ssp}_mean"] = sub.mean(axis=1)
        out[f"{ssp}_median"] = sub.median(axis=1)
        out[f"{ssp}_5th"] = sub.quantile(0.05, axis=1)
        out[f"{ssp}_95th"] = sub.quantile(0.95, axis=1)
        out[f"{ssp}_models"] = sub.count(axis=1)
    return out


def compute_tas_future_anomalies(
    annual_temperature: pd.DataFrame,
    warming_levels_all_models: pd.DataFrame,
    pi_baseline: pd.Series,
    rp_baseline: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Per-SSP mean / median / 5–95th-percentile temperature anomaly for each
    future period and warming level.

    Returns ``{'recent_past': df, 'pre_industrial': df}`` — only the three
    fixed future periods are kept (warming-level rows are redundant with the
    crossing-year table itself).
    """
    rp_anomaly = annual_temperature.subtract(rp_baseline, axis="columns")
    pi_anomaly = annual_temperature.subtract(pi_baseline, axis="columns")
    period_central = _period_central_years(warming_levels_all_models, annual_temperature.columns)
    rp_means = _period_means(rp_anomaly, period_central)
    pi_means = _period_means(pi_anomaly, period_central)
    return {
        "recent_past": _per_ssp_stats(rp_means).iloc[:3],
        "pre_industrial": _per_ssp_stats(pi_means).iloc[:3],
    }


# ---------------------------------------------------------------------------
# Future-period anomaly statistics — precipitation (percent change)
# ---------------------------------------------------------------------------

def compute_pr_future_percent_anomalies(
    annual_precipitation: pd.DataFrame,
    warming_levels_all_models: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Per-SSP mean / median / 5–95th-percentile percent-change for precipitation
    relative to the pre-industrial and recent-past baselines.

    The annual_precipitation columns are stripped of the leading ``pr_`` /
    ``tas_`` variable token so the warming-level crossings (built on tas) can
    be intersected with precipitation columns.
    """
    pr = annual_precipitation.copy()
    pr.columns = ["_".join(c.split("_")[1:]) for c in pr.columns]

    wl_pr = warming_levels_all_models.copy()
    wl_pr.columns = ["_".join(c.split("_")[1:]) for c in wl_pr.columns]

    common = wl_pr.columns.intersection(pr.columns)
    wl_pr = wl_pr[common]
    pr = pr[common]

    pi_baseline = _baseline_mean(pr, PI_BASELINE_WINDOW)
    rp_baseline = _baseline_mean(pr, RP_BASELINE_WINDOW)

    rp_pct = pr.subtract(rp_baseline, axis="columns").div(rp_baseline) * 100
    pi_pct = pr.subtract(pi_baseline, axis="columns").div(pi_baseline) * 100

    period_central = _period_central_years(wl_pr, pr.columns)
    rp_means = _period_means(rp_pct, period_central)
    pi_means = _period_means(pi_pct, period_central)
    return {
        "recent_past": _per_ssp_stats(rp_means),
        "pre_industrial": _per_ssp_stats(pi_means),
    }


# ---------------------------------------------------------------------------
# Per-year anomaly bands — temperature
# ---------------------------------------------------------------------------

def compute_tas_anomaly_stats(
    annual_temperature: pd.DataFrame,
    pi_baseline: pd.Series,
    rp_baseline: pd.Series,
    *,
    scenarios: Iterable[str] = SCENARIOS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Year-wise mean / median / 5th / 95th percentile bands of the temperature
    anomaly, per SSP, for both baselines.

    Returns
    -------
    tas_anomaly_pi : DataFrame (year × model_scenario_column)
    tas_anomaly_rp : DataFrame (year × model_scenario_column)
    stats_pi       : DataFrame (year × {ssp_mean_pi, ssp_median_pi, ssp_5q_pi, ssp_95q_pi})
    stats_rp       : DataFrame (same shape)
    baseline_offset: float (rp_baseline.mean() − pi_baseline.mean())
    """
    pi_anomaly = annual_temperature.subtract(pi_baseline, axis="columns")
    rp_anomaly = annual_temperature.subtract(rp_baseline, axis="columns")

    stats_rp = pd.DataFrame(index=rp_anomaly.index)
    stats_pi = pd.DataFrame(index=pi_anomaly.index)
    for ssp in scenarios:
        cols_rp = _ssp_columns(rp_anomaly.columns, ssp)
        sub_rp = rp_anomaly[cols_rp]
        stats_rp[f"{ssp}_mean_rp"] = sub_rp.mean(axis=1)
        stats_rp[f"{ssp}_median_rp"] = sub_rp.median(axis=1)
        stats_rp[f"{ssp}_5q_rp"] = sub_rp.quantile(0.05, axis=1)
        stats_rp[f"{ssp}_95q_rp"] = sub_rp.quantile(0.95, axis=1)

        cols_pi = _ssp_columns(pi_anomaly.columns, ssp)
        sub_pi = pi_anomaly[cols_pi]
        stats_pi[f"{ssp}_mean_pi"] = sub_pi.mean(axis=1)
        stats_pi[f"{ssp}_median_pi"] = sub_pi.median(axis=1)
        stats_pi[f"{ssp}_5q_pi"] = sub_pi.quantile(0.05, axis=1)
        stats_pi[f"{ssp}_95q_pi"] = sub_pi.quantile(0.95, axis=1)

    baseline_offset = float(rp_baseline.mean() - pi_baseline.mean())
    return pi_anomaly, rp_anomaly, stats_pi, stats_rp, baseline_offset


# ---------------------------------------------------------------------------
# Per-year anomaly bands — precipitation
# ---------------------------------------------------------------------------

PR_SMOOTHING_WINDOW = 5  # years; trailing rolling mean for precipitation


def compute_pr_anomaly_stats(
    annual_precipitation: pd.DataFrame,
    *,
    scenarios: Iterable[str] = SCENARIOS,
) -> dict:
    """Smoothed precipitation trajectories and per-year statistics.

    Applies a 5-year trailing rolling mean before computing anomalies and
    percent-change bands relative to the pre-industrial and recent-past
    baselines. The variable-token prefix in the column names is preserved
    here (callers do not strip it for the per-year bands; it is only
    stripped for the warming-level percent-anomaly tables).

    Returned mapping carries:

    - ``pr_pi_baseline``, ``pr_rp_baseline`` (Series)
    - ``pr_smoothed`` (rolling-mean trajectories)
    - ``pr_anomaly_pi`` / ``pr_anomaly_rp`` (per-year absolute anomalies)
    - ``stats_pr_anomaly_pi`` / ``stats_pr_anomaly_rp``
    - ``pr_pi_percent_change`` / ``pr_rp_percent_change``
    - ``stats_pr_pi_percent_change`` / ``stats_pr_rp_percent_change``
    - ``pr_baseline_offset`` (mean rp_baseline − mean pi_baseline)
    - ``pr_baseline_offset_percent`` (offset / mean rp_baseline × 100)
    - ``pr_ax_ratio`` (mean rp_baseline / mean pi_baseline)
    """
    pi_baseline = _baseline_mean(annual_precipitation, PI_BASELINE_WINDOW)
    rp_baseline = _baseline_mean(annual_precipitation, RP_BASELINE_WINDOW)

    smoothed = annual_precipitation.rolling(
        window=PR_SMOOTHING_WINDOW, min_periods=1,
    ).mean().dropna()

    pi_anomaly = smoothed.subtract(pi_baseline, axis="columns")
    rp_anomaly = smoothed.subtract(rp_baseline, axis="columns")

    pi_pct = (
        smoothed.subtract(pi_baseline, axis="columns").div(pi_baseline) * 100
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    rp_pct = (
        smoothed.subtract(rp_baseline, axis="columns").div(rp_baseline) * 100
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    stats_pi = pd.DataFrame(index=pi_anomaly.index)
    stats_rp = pd.DataFrame(index=rp_anomaly.index)
    stats_pi_pct = pd.DataFrame(index=pi_pct.index)
    stats_rp_pct = pd.DataFrame(index=rp_pct.index)

    for ssp in scenarios:
        cols_rp = _ssp_columns(rp_anomaly.columns, ssp)
        sub_rp = rp_anomaly[cols_rp]
        stats_rp[f"{ssp}_mean_rp"] = sub_rp.mean(axis=1)
        stats_rp[f"{ssp}_median_rp"] = sub_rp.median(axis=1)
        stats_rp[f"{ssp}_5q_rp"] = sub_rp.quantile(0.05, axis=1)
        stats_rp[f"{ssp}_95q_rp"] = sub_rp.quantile(0.95, axis=1)

        cols_pi = _ssp_columns(pi_anomaly.columns, ssp)
        sub_pi = pi_anomaly[cols_pi]
        stats_pi[f"{ssp}_mean_pi"] = sub_pi.mean(axis=1)
        stats_pi[f"{ssp}_median_pi"] = sub_pi.median(axis=1)
        stats_pi[f"{ssp}_5q_pi"] = sub_pi.quantile(0.05, axis=1)
        stats_pi[f"{ssp}_95q_pi"] = sub_pi.quantile(0.95, axis=1)

        cols_rp_pct = _ssp_columns(rp_pct.columns, ssp)
        sub_rp_pct = rp_pct[cols_rp_pct]
        stats_rp_pct[f"{ssp}_mean_rp"] = sub_rp_pct.mean(axis=1)
        stats_rp_pct[f"{ssp}_median_rp"] = sub_rp_pct.median(axis=1)
        stats_rp_pct[f"{ssp}_5q_rp"] = sub_rp_pct.quantile(0.05, axis=1)
        stats_rp_pct[f"{ssp}_95q_rp"] = sub_rp_pct.quantile(0.95, axis=1)

        cols_pi_pct = _ssp_columns(pi_pct.columns, ssp)
        sub_pi_pct = pi_pct[cols_pi_pct]
        stats_pi_pct[f"{ssp}_mean_pi"] = sub_pi_pct.mean(axis=1)
        stats_pi_pct[f"{ssp}_median_pi"] = sub_pi_pct.median(axis=1)
        stats_pi_pct[f"{ssp}_5q_pi"] = sub_pi_pct.quantile(0.05, axis=1)
        stats_pi_pct[f"{ssp}_95q_pi"] = sub_pi_pct.quantile(0.95, axis=1)

    pr_baseline_offset = float(rp_baseline.mean() - pi_baseline.mean())
    pr_baseline_offset_percent = float(
        (rp_baseline.mean() - pi_baseline.mean()) / rp_baseline.mean() * 100
    )
    pr_ax_ratio = float(rp_baseline.mean() / pi_baseline.mean())

    return {
        "pr_pi_baseline": pi_baseline,
        "pr_rp_baseline": rp_baseline,
        "pr_smoothed": smoothed,
        "pr_anomaly_pi": pi_anomaly,
        "pr_anomaly_rp": rp_anomaly,
        "stats_pr_anomaly_pi": stats_pi,
        "stats_pr_anomaly_rp": stats_rp,
        "pr_pi_percent_change": pi_pct,
        "pr_rp_percent_change": rp_pct,
        "stats_pr_pi_percent_change": stats_pi_pct,
        "stats_pr_rp_percent_change": stats_rp_pct,
        "pr_baseline_offset": pr_baseline_offset,
        "pr_baseline_offset_percent": pr_baseline_offset_percent,
        "pr_ax_ratio": pr_ax_ratio,
    }


# ---------------------------------------------------------------------------
# Anomaly tables (rendered string-formatted DataFrames)
# ---------------------------------------------------------------------------

_TAS_HEADER_REPLACEMENTS = (
    ("ssp", "SSP"),
    ("126", "1-2.6 (°C)"),
    ("245", "2-4.5 (°C)"),
    ("370", "3-7.0 (°C)"),
    ("585", "5-8.5 (°C)"),
)
_PR_HEADER_REPLACEMENTS = (
    ("ssp", "SSP"),
    ("126", "1-2.6 (%)"),
    ("245", "2-4.5 (%)"),
    ("370", "3-7.0 (%)"),
    ("585", "5-8.5 (%)"),
)


def _format_header(ssp: str, replacements: tuple[tuple[str, str], ...]) -> str:
    out = ssp
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def _format_anomaly_cell(row: pd.Series, ssp: str) -> str:
    if (
        pd.notna(row[f"{ssp}_mean"])
        and pd.notna(row[f"{ssp}_5th"])
        and pd.notna(row[f"{ssp}_95th"])
    ):
        return (
            f"{row[f'{ssp}_mean']:.1f} "
            f"({row[f'{ssp}_5th']:.1f}, {row[f'{ssp}_95th']:.1f}) "
            f"{int(row[f'{ssp}_models'])}"
        )
    return "- (-, -) 0"


def _build_formatted_table(
    stats: pd.DataFrame, replacements: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    out = pd.DataFrame(index=stats.index)
    for ssp in SCENARIOS:
        header = _format_header(ssp, replacements)
        out[header] = stats.apply(lambda row, _ssp=ssp: _format_anomaly_cell(row, _ssp), axis=1)
    return out


def _interleave_country_global_rows(
    country: str,
    rp_country: pd.DataFrame,
    pi_country: pd.DataFrame,
    rp_global: pd.DataFrame,
    pi_global: pd.DataFrame,
    *,
    rename_index_for_pr: bool = False,
) -> pd.DataFrame:
    """Interleave country and global anomaly rows in the order the legacy
    table figures expect:

        <country header row>
        <country: relative to 1995–2014>
        <country: relative to 1850–1900>
        <global header row>
        <global:  relative to 1995–2014>
        <global:  relative to 1850–1900>
    """
    rp_country = rp_country.copy()
    pi_country = pi_country.copy()
    rp_global = rp_global.copy()
    pi_global = pi_global.copy()

    if rename_index_for_pr:
        for df in (rp_country, pi_country, rp_global, pi_global):
            df.index = [i.replace("WL_", "Warming level ") for i in df.index]

    rp_country_indexes = rp_country.copy()
    rp_country_indexes.iloc[:, :] = " "
    rp_country_indexes.index = [f"{country.capitalize()}: {i}" for i in rp_country_indexes.index]
    rp_country.index = [f"{i} Relative to 1995–2014" for i in rp_country.index]
    pi_country.index = [f"{i} Relative to 1850–1900" for i in pi_country.index]

    rp_global_indexes = rp_global.copy()
    rp_global_indexes.iloc[:, :] = " "
    rp_global_indexes.index = [f"Global: {i}" for i in rp_global_indexes.index]
    rp_global.index = [f"{i} Relative to 1995–2014" for i in rp_global.index]
    pi_global.index = [f"{i} Relative to 1850–1900" for i in pi_global.index]

    dfs = [rp_country_indexes, rp_country, pi_country, rp_global_indexes, rp_global, pi_global]
    indices_to_slice = {1, 2, 4, 5}

    rows: list[pd.DataFrame] = []
    for i in range(len(rp_country_indexes.index)):
        block: list[pd.Series] = []
        for idx, df in enumerate(dfs):
            row = df.iloc[i].copy()
            if idx in indices_to_slice:
                row.name = row.name[-21:] if len(row.name) > 21 else row.name
            block.append(row)
        rows.append(pd.concat(block, axis=1).T)

    out = pd.concat(rows)
    out.index.name = "Time Period and Region"
    return out


def build_tas_anomalies_table(
    rp_country: pd.DataFrame,
    pi_country: pd.DataFrame,
    rp_global: pd.DataFrame,
    pi_global: pd.DataFrame,
    country: str,
) -> pd.DataFrame:
    """Country + global temperature anomalies, formatted for the table figure."""
    rp_c = _build_formatted_table(rp_country, _TAS_HEADER_REPLACEMENTS)
    pi_c = _build_formatted_table(pi_country, _TAS_HEADER_REPLACEMENTS)
    rp_g = _build_formatted_table(rp_global, _TAS_HEADER_REPLACEMENTS)
    pi_g = _build_formatted_table(pi_global, _TAS_HEADER_REPLACEMENTS)
    return _interleave_country_global_rows(country, rp_c, pi_c, rp_g, pi_g)


def build_pr_percent_anom_table(
    rp_country: pd.DataFrame,
    pi_country: pd.DataFrame,
    rp_global: pd.DataFrame,
    pi_global: pd.DataFrame,
    country: str,
) -> pd.DataFrame:
    """Country + global precipitation percent anomalies, formatted for the table figure."""
    rp_c = _build_formatted_table(rp_country, _PR_HEADER_REPLACEMENTS)
    pi_c = _build_formatted_table(pi_country, _PR_HEADER_REPLACEMENTS)
    rp_g = _build_formatted_table(rp_global, _PR_HEADER_REPLACEMENTS)
    pi_g = _build_formatted_table(pi_global, _PR_HEADER_REPLACEMENTS)
    return _interleave_country_global_rows(
        country, rp_c, pi_c, rp_g, pi_g, rename_index_for_pr=True,
    )


# ---------------------------------------------------------------------------
# All-in-one builder
# ---------------------------------------------------------------------------

def build_profile_signals(
    annual_temperature: pd.DataFrame,
    annual_precipitation: pd.DataFrame,
    *,
    country: str,
    tas_future_anomalies: dict[str, pd.DataFrame],
    pr_future_percent_anomalies: dict[str, pd.DataFrame],
    tas_future_anomalies_global: dict[str, pd.DataFrame],
    pr_future_percent_anomalies_global: dict[str, pd.DataFrame],
    warming_levels_all_models: pd.DataFrame,
) -> ProfileSignals:
    """Run every country-profile derivation against the annual time series and
    return a populated :class:`ProfileSignals`.
    """
    pi_baseline = _baseline_mean(annual_temperature, PI_BASELINE_WINDOW)
    rp_baseline = _baseline_mean(annual_temperature, RP_BASELINE_WINDOW)

    tas_pi_anomaly, tas_rp_anomaly, stats_tas_pi, stats_tas_rp, baseline_offset = (
        compute_tas_anomaly_stats(annual_temperature, pi_baseline, rp_baseline)
    )
    pr_signals = compute_pr_anomaly_stats(annual_precipitation)

    tas_table = build_tas_anomalies_table(
        rp_country=tas_future_anomalies["recent_past"],
        pi_country=tas_future_anomalies["pre_industrial"],
        rp_global=tas_future_anomalies_global["recent_past"],
        pi_global=tas_future_anomalies_global["pre_industrial"],
        country=country,
    )
    pr_table = build_pr_percent_anom_table(
        rp_country=pr_future_percent_anomalies["recent_past"],
        pi_country=pr_future_percent_anomalies["pre_industrial"],
        rp_global=pr_future_percent_anomalies_global["recent_past"],
        pi_global=pr_future_percent_anomalies_global["pre_industrial"],
        country=country,
    )

    return ProfileSignals(
        annual_temperature=annual_temperature,
        tas_pi_baseline=pi_baseline,
        tas_rp_baseline=rp_baseline,
        tas_baseline_offset=baseline_offset,
        tas_anomaly_pi=tas_pi_anomaly,
        tas_anomaly_rp=tas_rp_anomaly,
        stats_tas_anomaly_pi=stats_tas_pi,
        stats_tas_anomaly_rp=stats_tas_rp,
        annual_precipitation=annual_precipitation,
        pr_pi_baseline=pr_signals["pr_pi_baseline"],
        pr_rp_baseline=pr_signals["pr_rp_baseline"],
        pr_smoothed=pr_signals["pr_smoothed"],
        pr_anomaly_pi=pr_signals["pr_anomaly_pi"],
        pr_anomaly_rp=pr_signals["pr_anomaly_rp"],
        stats_pr_anomaly_pi=pr_signals["stats_pr_anomaly_pi"],
        stats_pr_anomaly_rp=pr_signals["stats_pr_anomaly_rp"],
        pr_baseline_offset=pr_signals["pr_baseline_offset"],
        pr_baseline_offset_percent=pr_signals["pr_baseline_offset_percent"],
        pr_ax_ratio=pr_signals["pr_ax_ratio"],
        pr_pi_percent_change=pr_signals["pr_pi_percent_change"],
        pr_rp_percent_change=pr_signals["pr_rp_percent_change"],
        stats_pr_pi_percent_change=pr_signals["stats_pr_pi_percent_change"],
        stats_pr_rp_percent_change=pr_signals["stats_pr_rp_percent_change"],
        tas_anomalies_table=tas_table,
        pr_percent_anom_table=pr_table,
    )
