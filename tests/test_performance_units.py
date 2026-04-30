"""Unit tests for the pure functions in `subselect.performance`.

Toy-data only — no CMIP6 / W5E5 reads. The end-to-end regression against
the published Greece paper is in `tests/test_regression_greece.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from subselect import performance


# ---------- taylor_skill_score -------------------------------------------


def test_tss_perfect_match_is_one():
    """Identical fields (r=1, a=1) yield TSS = 1."""
    assert performance.taylor_skill_score(corr=1.0, std_ratio=1.0) == pytest.approx(1.0)


def test_tss_anti_correlated_is_zero():
    """Anti-correlated fields (r=-1) yield TSS = 0 regardless of std ratio."""
    assert performance.taylor_skill_score(corr=-1.0, std_ratio=1.0) == 0.0
    assert performance.taylor_skill_score(corr=-1.0, std_ratio=2.0) == 0.0


def test_tss_decreases_as_std_ratio_diverges_from_one():
    base = performance.taylor_skill_score(corr=0.95, std_ratio=1.0)
    inflated = performance.taylor_skill_score(corr=0.95, std_ratio=2.0)
    deflated = performance.taylor_skill_score(corr=0.95, std_ratio=0.5)
    assert inflated < base
    assert deflated < base


def test_tss_clips_correlation_to_unit_interval():
    """Out-of-range r values must be clamped, not propagate as NaN."""
    assert performance.taylor_skill_score(corr=1.5, std_ratio=1.0) == pytest.approx(1.0)
    assert performance.taylor_skill_score(corr=-2.0, std_ratio=1.0) == 0.0


def test_tss_floors_zero_std_ratio():
    """std_ratio = 0 → handled (model is degenerate, TSS is tiny but finite)."""
    value = performance.taylor_skill_score(corr=0.5, std_ratio=0.0)
    assert np.isfinite(value)
    assert value > 0.0
    assert value < 1e-10


# ---------- harmonic_mean ------------------------------------------------


def test_harmonic_mean_of_equal_values_is_that_value():
    assert performance.harmonic_mean(0.5, 0.5) == pytest.approx(0.5, rel=1e-9)
    assert performance.harmonic_mean(1.0, 1.0) == pytest.approx(1.0, rel=1e-9)


def test_harmonic_mean_with_zero_collapses_to_zero():
    assert performance.harmonic_mean(0.0, 0.5) == pytest.approx(0.0, abs=1e-9)


def test_harmonic_mean_with_both_zero_does_not_explode():
    """The legacy formula adds ε to the denominator so HM(0, 0) is 0, not NaN."""
    assert performance.harmonic_mean(0.0, 0.0) == 0.0


def test_harmonic_mean_is_lower_than_arithmetic_mean():
    a, b = 0.2, 0.9
    hm = performance.harmonic_mean(a, b)
    am = (a + b) / 2.0
    assert hm < am


# ---------- minmax_normalize ---------------------------------------------


def test_minmax_normalize_maps_to_unit_interval():
    s = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0])
    out = performance.minmax_normalize(s)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_minmax_normalize_preserves_ordering():
    s = pd.Series([0.5, 0.1, 0.9, 0.3])
    out = performance.minmax_normalize(s)
    assert list(out.argsort()) == list(s.argsort())


def test_minmax_normalize_degenerate_constant_returns_input(caplog):
    """All-equal series: legacy notebook keeps the original (no rescale)."""
    s = pd.Series([0.42, 0.42, 0.42])
    out = performance.minmax_normalize(s)
    pd.testing.assert_series_equal(out, s)


def test_minmax_normalize_handles_nan():
    s = pd.Series([np.nan, 1.0, 3.0, np.nan, 5.0])
    out = performance.minmax_normalize(s)
    assert out.dropna().min() == pytest.approx(0.0)
    assert out.dropna().max() == pytest.approx(1.0)
    assert out.isna().sum() == 2


# ---------- bias_score_pixel ---------------------------------------------


def _toy_2d(values: np.ndarray) -> xr.DataArray:
    n_lat, n_lon = values.shape
    return xr.DataArray(
        values,
        coords={"lat": np.arange(n_lat, dtype=float), "lon": np.arange(n_lon, dtype=float)},
        dims=["lat", "lon"],
    )


def test_bias_score_one_when_bias_is_zero():
    bias = _toy_2d(np.zeros((3, 3)))
    sigma = _toy_2d(np.ones((3, 3)))
    out = performance.bias_score_pixel(bias, sigma)
    np.testing.assert_allclose(out.values, 1.0)


def test_bias_score_decays_to_zero_for_large_bias():
    bias = _toy_2d(np.full((3, 3), 1000.0))
    sigma = _toy_2d(np.ones((3, 3)))
    out = performance.bias_score_pixel(bias, sigma)
    assert out.max() < 1e-3


def test_bias_score_uses_eps_for_zero_or_negative_sigma():
    bias = _toy_2d(np.full((3, 3), 1e-3))
    sigma = _toy_2d(np.zeros((3, 3)))  # sigma=0 should be replaced by EPS_SIGMA=1e-6
    out = performance.bias_score_pixel(bias, sigma)
    # |bias| / eps = 1e-3/1e-6 = 1e3 → score ≈ 1/(1+(1e3)^1.5) ≈ tiny but not nan
    assert np.all(np.isfinite(out.values))
    assert out.max() < 1e-3


def test_bias_score_propagates_nan_in_bias():
    arr = np.array([[0.0, 0.5, np.nan], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    bias = _toy_2d(arr)
    sigma = _toy_2d(np.ones((3, 3)))
    out = performance.bias_score_pixel(bias, sigma)
    assert np.isnan(out.values[0, 2])
    np.testing.assert_allclose(out.values[0, 0], 1.0)


def test_bias_score_in_unit_interval():
    rng = np.random.default_rng(42)
    bias = _toy_2d(rng.standard_normal((4, 4)))
    sigma = _toy_2d(np.abs(rng.standard_normal((4, 4))) + 0.1)
    out = performance.bias_score_pixel(bias, sigma)
    assert out.min() >= 0.0 and out.max() <= 1.0


# ---------- season-month dictionary integrity ----------------------------


def test_season_months_cover_all_twelve_calendar_months():
    """DJF + MAM + JJA + SON must partition the 12 calendar months."""
    seasons = {"DJF", "MAM", "JJA", "SON"}
    union = sorted(
        m
        for s in seasons
        for m in performance.SEASON_MONTHS[s]
    )
    assert union == list(range(1, 13))


def test_season_annual_is_all_twelve_months():
    assert tuple(sorted(performance.SEASON_MONTHS["annual"])) == tuple(range(1, 13))
