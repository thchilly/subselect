"""Unit tests for `subselect.geom`.

Exercises the four crop methods against a synthetic 1°-grid DataArray and a
single-polygon ``TEST`` geopackage built in ``conftest.py``. Tests do not
require GADM or real CMIP6 data — fully offline-runnable.

Numerical checks per ``docs/refactor.md``:
- bbox returns a clean rectangular slice.
- shapefile_strict and shapefile_lenient differ by at most a one-pixel ring
  around the country boundary (centre-inside vs any-touch).
- shapefile_fractional weights live in [0, 1] and integrate to roughly the
  polygon's area in pixel units.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from subselect import geom
from subselect.geom import CROP_METHODS, ShapefileNotConfigured


# ---------- bbox -----------------------------------------------------------


def test_bbox_crops_to_country_plus_one_degree_ring(synthetic_data, synthetic_config):
    result = geom.crop(synthetic_data, "TEST", method="bbox", config=synthetic_config)
    # box_offset=1 → lat 34..43, lon 19..31 inclusive on the 1° grid
    assert float(result.data["lat"].min()) >= 34.0
    assert float(result.data["lat"].max()) <= 43.0
    assert float(result.data["lon"].min()) >= 19.0
    assert float(result.data["lon"].max()) <= 31.0
    assert result.weight is None
    assert result.metadata["crop_method"] == "bbox"
    assert result.metadata["country"] == "TEST"


def test_bbox_with_zero_offset_matches_country_exactly(synthetic_data, synthetic_config):
    result = geom.crop(
        synthetic_data, "TEST", method="bbox", box_offset=0.0, config=synthetic_config
    )
    assert float(result.data["lat"].min()) >= 35.0
    assert float(result.data["lat"].max()) <= 42.0
    assert float(result.data["lon"].min()) >= 20.0
    assert float(result.data["lon"].max()) <= 30.0


# ---------- shapefile binary methods --------------------------------------


def test_shapefile_lenient_is_a_superset_of_strict(synthetic_data, synthetic_config):
    """All-touched (lenient) cannot include fewer pixels than centre-inside (strict)."""
    strict = geom.crop(
        synthetic_data, "TEST", method="shapefile_strict", config=synthetic_config
    ).data
    lenient = geom.crop(
        synthetic_data, "TEST", method="shapefile_lenient", config=synthetic_config
    ).data
    # NaN positions: strict has at least as many NaNs as lenient.
    assert int(np.isnan(strict).sum()) >= int(np.isnan(lenient).sum())
    # Every non-NaN position in strict must also be non-NaN in lenient.
    strict_kept = ~np.isnan(strict.values)
    lenient_kept = ~np.isnan(lenient.values)
    assert np.all(lenient_kept[strict_kept])


def test_shapefile_lenient_records_metadata(synthetic_data, synthetic_config):
    result = geom.crop(
        synthetic_data, "TEST", method="shapefile_lenient", config=synthetic_config
    )
    assert result.metadata["crop_method"] == "shapefile_lenient"
    assert result.metadata["all_touched"] is True
    assert result.weight is None


def test_shapefile_strict_records_metadata(synthetic_data, synthetic_config):
    result = geom.crop(
        synthetic_data, "TEST", method="shapefile_strict", config=synthetic_config
    )
    assert result.metadata["crop_method"] == "shapefile_strict"
    assert result.metadata["all_touched"] is False


# ---------- shapefile_fractional ------------------------------------------


def test_fractional_returns_data_unmasked_with_weights_in_unit_interval(
    synthetic_data, synthetic_config
):
    result = geom.crop(
        synthetic_data,
        "TEST",
        method="shapefile_fractional",
        config=synthetic_config,
    )
    assert result.weight is not None
    # Data is the bbox-cropped DataArray, no NaN-masking applied.
    assert not np.any(np.isnan(result.data.values))
    weights = result.weight.values
    assert np.all((weights >= 0.0) & (weights <= 1.0))


def test_fractional_weight_total_matches_country_area_in_pixel_units(
    synthetic_data, synthetic_config
):
    """The TEST country polygon is 7° × 10° = 70 deg² on the 1° grid."""
    result = geom.crop(
        synthetic_data,
        "TEST",
        method="shapefile_fractional",
        config=synthetic_config,
    )
    weight_sum = float(result.weight.sum())
    assert weight_sum == pytest.approx(70.0, abs=1.5)  # mask_3D_frac_approx is approximate


# ---------- apply_weights -------------------------------------------------


def test_apply_weights_returns_cos_lat_when_weight_is_none(synthetic_data):
    weights = geom.apply_weights(synthetic_data, weight=None)
    expected = np.cos(np.deg2rad(synthetic_data["lat"]))
    np.testing.assert_allclose(weights.isel(lon=0).values, expected.values)


def test_apply_weights_composes_cos_lat_with_fractional_weight(
    synthetic_data, synthetic_config
):
    result = geom.crop(
        synthetic_data,
        "TEST",
        method="shapefile_fractional",
        config=synthetic_config,
    )
    composed = geom.apply_weights(result.data, weight=result.weight)
    cos_lat = np.cos(np.deg2rad(result.data["lat"]))
    expected = (cos_lat * result.weight).broadcast_like(result.data)
    np.testing.assert_allclose(composed.values, expected.values)


def test_apply_weights_zero_outside_country_for_fractional_path(
    synthetic_data, synthetic_config
):
    """Composing cos(lat) with frac weight zeros out non-country pixels."""
    result = geom.crop(
        synthetic_data,
        "TEST",
        method="shapefile_fractional",
        config=synthetic_config,
    )
    composed = geom.apply_weights(result.data, weight=result.weight)
    # Pixels with frac == 0 must have composed weight == 0 too.
    zero_frac = result.weight.values == 0.0
    assert np.all(composed.values[zero_frac] == 0.0)


# ---------- error paths ---------------------------------------------------


def test_invalid_method_raises(synthetic_data, synthetic_config):
    with pytest.raises(ValueError, match="Unknown crop method"):
        geom.crop(
            synthetic_data, "TEST", method="not_a_real_method", config=synthetic_config
        )


def test_unknown_country_raises(synthetic_data, synthetic_config):
    with pytest.raises(ValueError, match="not found in country_codes"):
        geom.crop(synthetic_data, "NOWHERE", method="bbox", config=synthetic_config)


def test_shapefile_method_without_configured_shapefile_raises(
    synthetic_data, synthetic_config, tmp_path
):
    bad_config = synthetic_config.with_overrides(
        shapefile_path=tmp_path / "nonexistent.gpkg"
    )
    with pytest.raises(ShapefileNotConfigured):
        geom.crop(
            synthetic_data, "TEST", method="shapefile_lenient", config=bad_config
        )


# ---------- canonical method list -----------------------------------------


def test_crop_methods_constant_lists_all_four():
    assert set(CROP_METHODS) == {
        "bbox",
        "shapefile_strict",
        "shapefile_lenient",
        "shapefile_fractional",
    }
