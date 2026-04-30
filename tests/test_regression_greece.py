"""Regression tests against the published Greece paper artefacts.

These tests pin the paper's HPS table, per-variable performance metrics
(TSS + BVS), end-of-century change signals, and spread-quadrant assignments
as fixtures under ``tests/fixtures/greece/``. They currently xfail(strict=True)
— they assert against the new pipeline (``subselect.performance``,
``subselect.spread``) that lands in M7 / M8. When those modules implement
the regression-target functions, the xfail markers must be removed so these
tests serve as the ongoing safety net per ``docs/refactor.md`` §
Regression test contract.

Tolerance ladder (frozen):

- Try ``ATOL_TARGET = 1e-6`` first.
- If the new pipeline fails to reproduce within 1e-6, fall back to
  ``ATOL_FALLBACK = 1e-4`` and log a methodology entry to
  ``documentation/methods.tex`` § Historical performance addendum explaining
  the float-arithmetic-order divergence. Root cause is vectorisation across
  the model dimension changing reduction order (sums of N floats are
  non-associative at machine precision); not regridder nondeterminism, since
  the paper-era pipeline does no on-the-fly regridding (see M7 plan in
  the agreed Phase 0 plan and ``docs/refactor.md``).
- If ``1e-4`` fails: stop. Do not loosen further. Investigate before merging.

Method pinned to ``crop_method="bbox"`` to match the paper-era setting per
``docs/refactor.md`` § Country cropping. The framework default
``shapefile_lenient`` is for new work; the paper used bbox cropping and the
regression test must reproduce that setting exactly.

Models are reindexed to the canonical 1..35 ordering from
``Data/models_ordered.csv`` (committed as ``tests/fixtures/greece/models35.txt``)
before comparison. Athanasios is firm that this 1..35 ordering is preserved
throughout the project — every figure marker uses it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "greece"

ATOL_TARGET = 1e-6
ATOL_FALLBACK = 1e-4  # only after the M7 mitigations from the plan are tried

PAPER_VARS: tuple[str, ...] = ("tas", "pr", "psl")
DIAGNOSTIC_VARS: tuple[str, ...] = ("tasmax",)
PERIODS: tuple[str, ...] = ("annual", "DJF", "MAM", "JJA", "SON")
QUADRANT_LABELS = frozenset({"warm_wet", "warm_dry", "cool_wet", "cool_dry"})


# ---------------------------------------------------------------------------
# Fixture loaders (always work; do not depend on M7/M8 implementation status)
# ---------------------------------------------------------------------------


def _load_models35() -> list[str]:
    with (FIXTURES / "models35.txt").open() as fh:
        return [line.strip() for line in fh if line.strip()]


def _load_excel_by_model(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, index_col=0)
    df.index.name = "model"
    return df.reindex(_load_models35())


def _load_expected_hps() -> pd.DataFrame:
    """35×5 DataFrame: rows = canonical model order, columns = the five periods."""
    df = _load_excel_by_model(FIXTURES / "assess_cmip6_composite_HMperf_greece.xlsx")
    df = df.rename(columns=lambda c: c.replace("_HMperf", ""))
    return df[list(PERIODS)]


def _load_expected_per_variable_metrics(variable: str) -> pd.DataFrame:
    return _load_excel_by_model(
        FIXTURES / f"assess_cmip6_{variable}_mon_perf_metrics_all_seasons_greece.xlsx"
    )


def _load_expected_change_signals() -> pd.DataFrame:
    """35×20 DataFrame: end-of-century minus pre-industrial deltas (SSP5-8.5)."""
    return _load_excel_by_model(
        FIXTURES / "assess_long_term_change_spread_greece.xlsx"
    )


def _expected_quadrants(deltas: pd.DataFrame) -> pd.DataFrame:
    """Derive per-(model, period) quadrant labels from Δtas, Δpr columns.

    Convention: for each period, the cutpoints are the seasonal medians of
    Δtas and Δpr across all 35 models. A model is ``warm_*`` if Δtas ≥ median,
    ``wet_*`` if Δpr ≥ median (suffix order: <temp>_<precip>).
    """
    quadrants: dict[str, pd.Series] = {}
    for period in PERIODS:
        tas = deltas[f"tas_{period}"]
        pr = deltas[f"pr_{period}"]
        warm = tas >= tas.median()
        wet = pr >= pr.median()
        labels = pd.Series(index=deltas.index, dtype="object")
        labels.loc[warm & wet] = "warm_wet"
        labels.loc[warm & ~wet] = "warm_dry"
        labels.loc[~warm & wet] = "cool_wet"
        labels.loc[~warm & ~wet] = "cool_dry"
        quadrants[period] = labels
    return pd.DataFrame(quadrants)


def _metric_columns(periods: tuple[str, ...] = PERIODS) -> list[str]:
    """The TSS + BVS columns that feed HPS, per-period."""
    out = []
    for period in periods:
        out.extend([f"{period}_tss", f"{period}_tss_hirota", f"{period}_bias_score"])
    return out


# ---------------------------------------------------------------------------
# Sanity test — must always pass; protects the fixtures themselves.
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_greece_fixtures_loadable():
    """Fixtures load with expected shapes; canonical 1..35 ordering preserved.

    Catches accidental fixture corruption / swapping / re-ordering during
    repo moves (e.g. the 2026-04-30 Data/ → results/ restructure).
    """
    models = _load_models35()
    assert len(models) == 35
    assert models[0] == "ACCESS-CM2"
    assert models[-1] == "UKESM1-0-LL"

    hps = _load_expected_hps()
    assert hps.shape == (35, 5)
    assert list(hps.columns) == list(PERIODS)
    assert list(hps.index) == models

    for variable in PAPER_VARS + DIAGNOSTIC_VARS:
        df = _load_expected_per_variable_metrics(variable)
        assert df.shape[0] == 35
        for col in _metric_columns():
            assert col in df.columns, f"{variable}: missing expected column {col!r}"

    deltas = _load_expected_change_signals()
    assert deltas.shape == (35, 20)
    for var in PAPER_VARS + DIAGNOSTIC_VARS:
        for period in PERIODS:
            assert f"{var}_{period}" in deltas.columns

    quadrants = _expected_quadrants(deltas)
    assert quadrants.shape == (35, 5)
    assert set(quadrants.values.ravel()) <= QUADRANT_LABELS


# ---------------------------------------------------------------------------
# Regression tests — xfail(strict=True) until M7 / M8 ship the pipeline.
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    reason="Awaiting M7 — subselect.performance.compute_hps not yet implemented",
)
def test_greece_hps_regression():
    """Per-model HPS_<period> matches paper xlsx within ATOL_TARGET (1e-6).

    crop_method='bbox' matches the paper-era setting. Models reindexed to the
    canonical 1..35 ordering before comparison.
    """
    from subselect.performance import compute_hps  # raises ImportError until M7

    expected = _load_expected_hps()
    actual = compute_hps("greece", scenario="ssp585", crop_method="bbox")
    pd.testing.assert_frame_equal(
        actual.reindex(expected.index)[list(expected.columns)],
        expected,
        atol=ATOL_TARGET,
        rtol=0,
        check_names=False,
    )


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    reason="Awaiting M7 — subselect.performance.compute_metrics not yet implemented",
)
@pytest.mark.parametrize("variable", PAPER_VARS + DIAGNOSTIC_VARS)
def test_greece_per_variable_metrics_regression(variable: str):
    """TSS (standard + Hirota) and bias_score (BVS) match paper within 1e-6.

    Pins both TSS variants because the paper-era code reports both; M7 will
    decide which one feeds HPS but the regression must reproduce both. tas/pr/psl
    feed HPS; tasmax is a diagnostic variable per docs/historical_performance.md.
    """
    from subselect.performance import compute_metrics  # raises ImportError until M7

    expected = _load_expected_per_variable_metrics(variable)
    actual = compute_metrics("greece", variable=variable, crop_method="bbox")
    cols = _metric_columns()
    pd.testing.assert_frame_equal(
        actual.reindex(expected.index)[cols],
        expected[cols],
        atol=ATOL_TARGET,
        rtol=0,
        check_names=False,
    )


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    reason="Awaiting M8 — subselect.spread.compute_change_signals not yet implemented",
)
def test_greece_change_signals_regression():
    """Per-(model, var, period) end-of-century deltas match paper within 1e-6.

    SSP5-8.5; future window 2081–2100; pre-industrial 1850–1900 — frozen from
    docs/future_spread.md. Covers tas/pr/psl/tasmax across annual + 4 seasons.
    """
    from subselect.spread import compute_change_signals  # raises ImportError until M8

    expected = _load_expected_change_signals()
    actual = compute_change_signals("greece", scenario="ssp585", crop_method="bbox")
    pd.testing.assert_frame_equal(
        actual.reindex(expected.index)[list(expected.columns)],
        expected,
        atol=ATOL_TARGET,
        rtol=0,
        check_names=False,
    )


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    reason="Awaiting M8 — subselect.spread.compute_spread_quadrants not yet implemented",
)
def test_greece_spread_quadrant_assignments():
    """Per-(model, period) quadrant assignment matches paper exactly.

    Quadrant cutpoints are the seasonal medians of Δtas and Δpr across all
    35 models — discrete labels, so no tolerance applies; all 175 (35×5)
    assignments must match.
    """
    from subselect.spread import compute_spread_quadrants  # raises ImportError until M8

    expected = _expected_quadrants(_load_expected_change_signals())
    actual = compute_spread_quadrants("greece", scenario="ssp585", crop_method="bbox")
    pd.testing.assert_frame_equal(
        actual.reindex(expected.index)[list(expected.columns)],
        expected,
        check_names=False,
    )
