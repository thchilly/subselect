"""Typed L1 state for the country pipeline.

:func:`subselect.compute.compute` returns a :class:`SubselectState`. Every
L2 figure function reads from this object — it is the single in-memory
representation of everything one country needs.

Adding new artefacts (e.g. model-independence distances, cost-function
results) is a one-line addition to :class:`SubselectState` plus the
corresponding builder in :mod:`subselect.compute`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr


@dataclass
class ProfileSignals:
    """Derivations the country-profile figures consume.

    The fields on this class come from the warming-level / anomaly-statistics
    derivations: smoothed time series, per-SSP per-year mean / median /
    percentile bands, baseline scalars, and the rendered anomaly tables.
    """

    # Annual-temperature derivations
    annual_temperature: pd.DataFrame
    tas_pi_baseline: pd.Series
    tas_rp_baseline: pd.Series
    tas_baseline_offset: float
    tas_anomaly_pi: pd.DataFrame
    tas_anomaly_rp: pd.DataFrame
    stats_tas_anomaly_pi: pd.DataFrame
    stats_tas_anomaly_rp: pd.DataFrame

    # Annual-precipitation derivations
    annual_precipitation: pd.DataFrame
    pr_pi_baseline: pd.Series
    pr_rp_baseline: pd.Series
    pr_smoothed: pd.DataFrame
    pr_anomaly_pi: pd.DataFrame
    pr_anomaly_rp: pd.DataFrame
    stats_pr_anomaly_pi: pd.DataFrame
    stats_pr_anomaly_rp: pd.DataFrame
    pr_baseline_offset: float
    pr_baseline_offset_percent: float
    pr_ax_ratio: float
    pr_pi_percent_change: pd.DataFrame
    pr_rp_percent_change: pd.DataFrame
    stats_pr_pi_percent_change: pd.DataFrame
    stats_pr_rp_percent_change: pd.DataFrame

    # Rendered anomaly tables (string-formatted, ready for table figures)
    tas_anomalies_table: pd.DataFrame
    pr_percent_anom_table: pd.DataFrame


@dataclass
class SubselectState:
    """Every artefact the L2 figure layer needs for one country.

    Constructed by :func:`subselect.compute.compute`. Adding new fields
    (e.g. model-independence distances, cost-function results) is a
    one-line change here plus the corresponding builder in
    :mod:`subselect.compute`.
    """

    country: str
    cache_dir: Path

    # Historical performance
    performance_metrics: dict[str, pd.DataFrame]
    composite_hps: pd.DataFrame
    composite_hps_full: pd.DataFrame
    observed_std_dev: pd.DataFrame
    monthly_means: dict[str, dict[str, pd.DataFrame]]

    # Future spread
    change_signals: pd.DataFrame
    long_term_spread: pd.DataFrame
    pre_industrial_spread: pd.DataFrame

    # Country profile — annual time series + warming levels
    annual_timeseries: dict[str, pd.DataFrame]
    warming_levels: pd.DataFrame
    warming_level_medians: pd.DataFrame
    warming_level_medians_global: pd.DataFrame
    future_anomalies: dict[str, dict[str, pd.DataFrame]]
    future_anomalies_global: dict[str, dict[str, pd.DataFrame]]
    profile_signals: ProfileSignals

    # Bias maps (xarray fields)
    observed_maps: dict[str, dict[str, xr.Dataset]] = field(default_factory=dict)
    bias_maps: dict[str, dict[str, dict[str, xr.DataArray]]] = field(default_factory=dict)

    # Independence / cost-function hooks — empty by default, populated when
    # those layers land.
    independence: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)

    @property
    def model_ids(self) -> dict[str, int]:
        """``{model_name: integer_id}`` mapping for the canonical 1..35 ensemble.

        Resolved by reading the canonical ``CMIP6_model_id.xlsx`` mapping
        from ``Data/CMIP6/metadata``.
        """
        from subselect.config import Config

        config = Config.from_env()
        meta = pd.read_excel(config.cmip6_metadata_root / "CMIP6_model_id.xlsx")
        return dict(zip(meta["model"], meta["id"]))
