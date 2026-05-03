"""Regression tests: pipeline outputs for Greece.

Pins the per-artefact outputs of the L1 pipeline at machine epsilon. The
fixtures are snapshots of the parquet files written to ``cache/greece/``
by ``subselect.compute.compute("greece")`` against the global cache
produced by ``subselect.compute_global.compute_global()``.

Cache filenames carry a ``__<crop_method>`` suffix; the regression
fixtures pin the current code default ``bbox``. Running the pipeline
with a different crop method writes to its own keyed entries and does
not affect this contract.

The tests read directly from ``cache/greece/parquet/`` rather than
calling :func:`subselect.compute.compute` so the test runtime stays
small. CI must populate the cache once via ``python -m subselect greece``
before running the regression suite. ``test_corrected_cache_present``
fails fast with a clear message when the cache is missing.

Tolerance: ``1e-12`` (essentially bit-identity). Any drift larger than
this is a methodology change and should be a deliberate snapshot
refresh, not a test loosening.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "regression_corrected"
CACHE = Path(__file__).resolve().parents[1] / "cache" / "greece" / "parquet"
COUNTRY = "greece"
PERIODS = ("annual", "DJF", "MAM", "JJA", "SON")
HPS_VARIABLES = ("tas", "pr", "psl")
ALL_VARIABLES = HPS_VARIABLES + ("tasmax",)
ATOL = 1e-12
# Cache filenames carry the crop-method suffix; the regression fixtures pin
# the current code default (bbox).
CROP_SUFFIX = "bbox"


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
def test_corrected_cache_present():
    """The Greece cache must be populated before regression assertions run.

    Run ``python -m subselect greece`` once on a fresh checkout to populate
    ``cache/_global/`` and ``cache/greece/`` before running the regression
    suite.
    """
    expected = [
        f"performance_metrics__{v}__{CROP_SUFFIX}.parquet" for v in ALL_VARIABLES
    ] + [
        f"composite_hps__{CROP_SUFFIX}.parquet",
        f"composite_hps_full__{CROP_SUFFIX}.parquet",
        f"change_signals__{CROP_SUFFIX}.parquet",
        f"observed_std_dev__{CROP_SUFFIX}.parquet",
    ]
    missing = [name for name in expected if not (CACHE / name).is_file()]
    assert not missing, (
        f"Missing per-country cache files: {missing}. "
        f"Run 'python -m subselect greece' to populate."
    )


@pytest.mark.regression
@pytest.mark.parametrize("variable", ALL_VARIABLES)
def test_per_variable_metrics_corrected(variable: str):
    expected = pd.read_parquet(FIXTURES / f"per_variable_metrics_{variable}.parquet")
    actual = pd.read_parquet(
        CACHE / f"performance_metrics__{variable}__{CROP_SUFFIX}.parquet"
    )
    pd.testing.assert_frame_equal(
        actual.reindex_like(expected),
        expected,
        atol=ATOL,
        rtol=0,
        check_dtype=False,
    )


@pytest.mark.regression
def test_composite_hps_corrected():
    expected = pd.read_parquet(FIXTURES / "composite_hps.parquet")
    actual = pd.read_parquet(CACHE / f"composite_hps__{CROP_SUFFIX}.parquet")
    pd.testing.assert_frame_equal(
        actual.reindex_like(expected),
        expected,
        atol=ATOL,
        rtol=0,
        check_dtype=False,
    )


@pytest.mark.regression
def test_composite_hps_full_corrected():
    expected = pd.read_parquet(FIXTURES / "composite_hps_full.parquet")
    actual = pd.read_parquet(CACHE / f"composite_hps_full__{CROP_SUFFIX}.parquet")
    pd.testing.assert_frame_equal(
        actual.reindex_like(expected),
        expected,
        atol=ATOL,
        rtol=0,
        check_dtype=False,
    )


@pytest.mark.regression
def test_observed_std_dev_corrected():
    expected = pd.read_parquet(FIXTURES / "observed_std_dev.parquet")
    actual = pd.read_parquet(CACHE / f"observed_std_dev__{CROP_SUFFIX}.parquet")
    pd.testing.assert_frame_equal(
        actual.reindex_like(expected),
        expected,
        atol=ATOL,
        rtol=0,
        check_dtype=False,
    )


@pytest.mark.regression
def test_change_signals_corrected():
    expected = pd.read_parquet(FIXTURES / "change_signals.parquet")
    actual = pd.read_parquet(CACHE / f"change_signals__{CROP_SUFFIX}.parquet")
    pd.testing.assert_frame_equal(
        actual.reindex_like(expected),
        expected,
        atol=ATOL,
        rtol=0,
        check_dtype=False,
    )
