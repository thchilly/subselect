"""Regression tests against the corrected-methodology pipeline snapshots.

Replaces ``tests/test_regression_greece.py`` (the M3 paper-xlsx regression
test) following the M-CORRECT step. The paper-era outputs are now archived
under ``documentation/historical_paper_outputs/``; the corrected pipeline
is the canonical contract.

Corrections applied at the ``methodology-corrected`` tag:

1. ``σ_obs`` spatial mean uses ``cos(latitude)`` weights, consistent with
   every other spatial mean in the pipeline.
2. ``σ_obs`` is computed on the native 0.5° W5E5 reference
   (``Data/reference/monthly_05/<var>_*.nc``) instead of the cmip6-grid
   upscaled product.

See ``documentation/methods.tex § Methodology corrections (post-paper)``.

Snapshots live under ``tests/fixtures/regression_corrected/``:

- ``per_variable_metrics_<var>.parquet`` for ``var ∈ {tas, pr, psl, tasmax}``
- ``composite_hps.parquet``        — output of ``compute_hps`` (renamed cols)
- ``composite_hps_full.parquet``   — full TSS_mm/bs_mm/HMperf/rank frame
- ``change_signals.parquet``       — output of ``compute_change_signals``
- ``observed_std_dev.parquet``     — σ_obs scalars per (variable, period)

Snapshots were generated via
``scripts/build_corrected_regression_fixtures.py``. Re-running that script
under the same code regenerates the fixtures byte-identically (the pipeline
is deterministic; ``joblib`` is used without across-iteration randomness).

Tolerance: ``1e-12`` (essentially bit-identity). Any drift larger than this
is a methodology change and should be a deliberate snapshot refresh, not a
test loosening.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "regression_corrected"
COUNTRY = "greece"
PERIODS = ("annual", "DJF", "MAM", "JJA", "SON")
HPS_VARIABLES = ("tas", "pr", "psl")
ALL_VARIABLES = HPS_VARIABLES + ("tasmax",)
ATOL = 1e-12


@pytest.mark.regression
def test_corrected_fixtures_present():
    expected = [
        f"per_variable_metrics_{v}.parquet" for v in ALL_VARIABLES
    ] + [
        "composite_hps.parquet",
        "composite_hps_full.parquet",
        "change_signals.parquet",
        "observed_std_dev.parquet",
    ]
    missing = [name for name in expected if not (FIXTURES / name).is_file()]
    assert not missing, f"Missing corrected-pipeline fixtures: {missing}"


@pytest.mark.regression
@pytest.mark.parametrize("variable", ALL_VARIABLES)
def test_per_variable_metrics_corrected(variable: str):
    """compute_metrics(<variable>, country='greece') must reproduce the
    snapshot for every (model, period, metric) cell."""
    from subselect.performance import compute_metrics

    expected = pd.read_parquet(FIXTURES / f"per_variable_metrics_{variable}.parquet")
    actual = compute_metrics(COUNTRY, variable=variable)
    pd.testing.assert_frame_equal(
        actual.reindex_like(expected),
        expected,
        atol=ATOL,
        rtol=0,
        check_dtype=False,
    )


@pytest.mark.regression
def test_composite_hps_corrected():
    """compute_hps(country='greece') must reproduce the snapshot."""
    from subselect.performance import compute_hps

    expected = pd.read_parquet(FIXTURES / "composite_hps.parquet")
    actual = compute_hps(COUNTRY).rename(columns={p: f"{p}_HMperf" for p in PERIODS})
    pd.testing.assert_frame_equal(
        actual.reindex_like(expected),
        expected,
        atol=ATOL,
        rtol=0,
        check_dtype=False,
    )


@pytest.mark.regression
def test_observed_std_dev_corrected():
    """σ_obs scalars per (variable, period) must reproduce the snapshot."""
    from subselect.config import Config
    from subselect.performance import _compute_obs_std_per_period

    expected = pd.read_parquet(FIXTURES / "observed_std_dev.parquet")
    config = Config.from_env()
    actual_columns = {}
    for variable in ALL_VARIABLES:
        sigmas = _compute_obs_std_per_period(variable, COUNTRY, "bbox", config)
        actual_columns[variable] = [sigmas[p] for p in PERIODS]
    actual = pd.DataFrame(actual_columns, index=list(PERIODS))
    pd.testing.assert_frame_equal(
        actual,
        expected,
        atol=ATOL,
        rtol=0,
        check_dtype=False,
    )


@pytest.mark.regression
def test_change_signals_corrected():
    """compute_change_signals(country='greece') must reproduce the snapshot."""
    from subselect.spread import compute_change_signals

    expected = pd.read_parquet(FIXTURES / "change_signals.parquet")
    actual = compute_change_signals(COUNTRY, scenario="ssp585")
    pd.testing.assert_frame_equal(
        actual.reindex_like(expected),
        expected,
        atol=ATOL,
        rtol=0,
        check_dtype=False,
    )
