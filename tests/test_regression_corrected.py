"""Regression tests: post-`cache-globalized` pipeline outputs for Greece.

Pins the per-artefact outputs of the L1 pipeline (post-Step-3.6) at
machine epsilon. The fixtures are snapshots of the parquet files written
to ``cache/greece/`` by ``subselect.compute.compute('greece')`` against
the global cache produced by ``subselect.compute_global.compute_global()``.

The fixture values incorporate two architectural changes from prior tags:

1. **Methodology corrections** (Step 2, ``methodology-corrected``):
   ``σ_obs`` uses cos(latitude)-weighted spatial means and is computed on
   the native 0.5° W5E5 reference. See ``documentation/methods.tex §
   Methodology corrections (post-paper)``.

2. **Cache architecture** (Step 3.6, ``cache-globalized``): per-(model,
   var) climatologies live in ``cache/_global/``; per-country derivations
   crop those cached fields. The reordered float-summation in the spread
   pipeline introduces a $\\sim$1e-5 absolute / 1e-7 relative numerical
   drift relative to the legacy operation order; this is documented in
   ``documentation/methods.tex § Cache scope`` and is the canonical
   contract from this tag onward.

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
        f"performance_metrics__{v}.parquet" for v in ALL_VARIABLES
    ] + [
        "composite_hps.parquet",
        "composite_hps_full.parquet",
        "change_signals.parquet",
        "observed_std_dev.parquet",
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
    actual = pd.read_parquet(CACHE / f"performance_metrics__{variable}.parquet")
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
    actual = pd.read_parquet(CACHE / "composite_hps.parquet")
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
    actual = pd.read_parquet(CACHE / "composite_hps_full.parquet")
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
    actual = pd.read_parquet(CACHE / "observed_std_dev.parquet")
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
    actual = pd.read_parquet(CACHE / "change_signals.parquet")
    pd.testing.assert_frame_equal(
        actual.reindex_like(expected),
        expected,
        atol=ATOL,
        rtol=0,
        check_dtype=False,
    )
