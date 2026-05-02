"""Typed L1 state for the country pipeline.

`compute(country)` returns a `SubselectState`. Every L2 figure function reads
from this object — it is the single in-memory representation of everything the
country needs.

Phase 1 (model independence) and Phase 2 (cost function) extend the state by
adding fields here; existing callers are unaffected.
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

    Constructed by :func:`subselect.compute.compute`. Adding new fields (e.g.
    Phase 1 model-independence distances, Phase 2 cost-function results) is a
    one-line change here plus the corresponding builder in ``compute.py``.
    """

    country: str
    cache_dir: Path

    # Performance — M7
    performance_metrics: dict[str, pd.DataFrame]
    composite_hps: pd.DataFrame
    composite_hps_full: pd.DataFrame
    observed_std_dev: pd.DataFrame
    monthly_means: dict[str, dict[str, pd.DataFrame]]

    # Spread — M8
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

    # Bias maps — M9.2 (xarray fields)
    observed_maps: dict[str, dict[str, xr.Dataset]] = field(default_factory=dict)
    bias_maps: dict[str, dict[str, dict[str, xr.DataArray]]] = field(default_factory=dict)

    # Phase 1 / Phase 2 hooks — empty by default, populated when those phases land
    independence: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)

    @property
    def model_ids(self) -> dict[str, int]:
        """`{model_name: integer_id}` mapping (paper-era 1..35 fixed order).

        Resolved from `composite_hps_full`'s index by reading the canonical
        CMIP6_model_id mapping; cached on first access via the dataframe's own
        index (no separate field needed).
        """
        from subselect.io import load_models_list

        # Lazy: only the renderer needs this, and it should already be in
        # composite_hps_full.index — but expose canonical mapping here.
        from subselect.config import Config
        config = Config.from_env()
        meta = pd.read_excel(config.cmip6_metadata_root / "CMIP6_model_id.xlsx")
        return dict(zip(meta["model"], meta["id"]))
