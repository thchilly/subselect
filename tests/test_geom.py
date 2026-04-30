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

import json

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Polygon

from subselect import geom
from subselect.config import Config
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
    """All-touched (lenient) cannot include fewer pixels than centre-inside (strict).

    Compare in the original (un-cropped) grid coordinate system because the
    mask-aware bbox can give the two methods different extents.
    """
    strict_data = geom.crop(
        synthetic_data, "TEST", method="shapefile_strict", config=synthetic_config
    ).data
    lenient_data = geom.crop(
        synthetic_data, "TEST", method="shapefile_lenient", config=synthetic_config
    ).data
    strict_kept = (~strict_data.isnull()).reindex_like(synthetic_data, fill_value=False)
    lenient_kept = (~lenient_data.isnull()).reindex_like(synthetic_data, fill_value=False)
    # Lenient cell count >= strict cell count.
    assert int(lenient_kept.sum()) >= int(strict_kept.sum())
    # Wherever strict includes a cell, lenient must include it too.
    assert int((strict_kept & ~lenient_kept).sum()) == 0


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


# ---------- degenerate-grid regression (M5 finding) -----------------------


def _build_tiny_country_config(tmp_path):
    """Build a Config + 1° polygon centred on a known 2.8°-grid cell centre.

    The 2.8°-grid case is what surfaced the M4 bug during M5 review: the
    bbox pre-crop reduced the grid to 1×1 cells, rioxarray could not infer
    the grid step from a 1-element coord, and rasterio.features.geometry_mask
    was fed a degenerate transform — returning all-False. The polygon is
    centred on cell-centre (34.4°N, 29.2°E) — both values appear in
    ``np.arange(-30, 50, 2.8)`` and ``np.arange(-10, 50, 2.8)`` respectively
    — so strict (centre-inside), lenient (any-touch), and fractional
    (area-fraction) all have a well-defined non-zero answer.
    """
    cell_lat, cell_lon = 34.4, 29.2  # known cell centres in the grid below
    half = 0.5
    poly = Polygon(
        [
            (cell_lon - half, cell_lat - half),
            (cell_lon + half, cell_lat - half),
            (cell_lon + half, cell_lat + half),
            (cell_lon - half, cell_lat + half),
        ]
    )
    gpkg = tmp_path / "tiny.gpkg"
    gpd.GeoDataFrame(
        {"COUNTRY": ["TINY"]}, geometry=[poly], crs="EPSG:4326"
    ).to_file(gpkg, driver="GPKG")

    data_root = tmp_path / "data"
    (data_root / "country_codes").mkdir(parents=True)
    (data_root / "country_codes" / "country_codes.json").write_text(
        json.dumps(
            {
                "TINY": {
                    "name": "TINY",
                    "alpha-2": "TI",
                    "alpha-3": "TIN",
                    "boundingBox": {
                        "sw": {"lat": cell_lat - half, "lon": cell_lon - half},
                        "ne": {"lat": cell_lat + half, "lon": cell_lon + half},
                    },
                }
            }
        )
    )
    config = Config(
        data_root=data_root,
        cache_root=tmp_path / "cache",
        shapefile_path=gpkg,
        reference_root=data_root / "reference" / "monthly_cmip6_upscaled",
        single_grid_reference_root=data_root / "reference" / "monthly_upscaled",
        cmip6_metadata_root=data_root / "CMIP6" / "metadata",
        results_root=tmp_path / "results",
    )
    return config


def _build_28deg_grid():
    """A 2.8° lat/lon grid that includes cell-centres (34.4, 29.2) used by the
    TINY-country fixture above."""
    lat = np.arange(-30.0, 50.0, 2.8)
    lon = np.arange(-10.0, 50.0, 2.8)
    # np.arange floating-point drift means `34.4 in lat` can be False; check
    # via tolerance instead.
    assert np.any(np.isclose(lat, 34.4)) and np.any(np.isclose(lon, 29.2)), (
        "fixture grid must include the cell centre"
    )
    return xr.DataArray(
        np.ones((len(lat), len(lon))),
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"],
        name="t",
    )


def test_lenient_includes_cell_when_polygon_lands_inside_a_single_cell(tmp_path):
    """Regression for the M4 degenerate-grid bug surfaced during M5 review.

    Pre-fix behaviour: ``shapefile_lenient`` returned 0 cells because the bbox
    pre-crop reduced the grid to 1×1 and the rasterio transform was degenerate.
    Post-fix: the binary mask is built on the full grid first and then bbox-
    cropped together with the data, so the cell containing the polygon is
    correctly included.
    """
    config = _build_tiny_country_config(tmp_path)
    da = _build_28deg_grid()

    result = geom.crop(da, "TINY", method="shapefile_lenient", config=config)
    n_included = int((~result.data.isnull()).sum())
    assert n_included >= 1, f"expected ≥1 cell, got {n_included}"


def test_strict_includes_cell_when_polygon_centre_lands_inside_a_single_cell(
    tmp_path,
):
    """Companion to the lenient regression — strict (centre-inside) on the
    same coarse-grid setup. The polygon's centre lies inside a 2.8° cell, so
    strict must include that cell too."""
    config = _build_tiny_country_config(tmp_path)
    da = _build_28deg_grid()

    result = geom.crop(da, "TINY", method="shapefile_strict", config=config)
    n_included = int((~result.data.isnull()).sum())
    assert n_included >= 1, f"expected ≥1 cell, got {n_included}"


def test_fractional_succeeds_on_degenerate_bbox_grid(tmp_path):
    """Regression for the M5 finding: ``shapefile_fractional`` previously
    raised ``regionmask.InvalidCoordsError`` when bbox pre-crop yielded a
    1-row or 1-col grid. Post-fix, the weight is computed on the full grid
    first, so regionmask sees uniform spacing and the small-country bbox-
    cropped weight has at least one positive cell."""
    config = _build_tiny_country_config(tmp_path)
    da = _build_28deg_grid()

    result = geom.crop(da, "TINY", method="shapefile_fractional", config=config)
    assert result.weight is not None
    weight_sum = float(result.weight.sum())
    assert weight_sum > 0, f"expected positive Σweight, got {weight_sum}"


def test_fractional_works_on_gaussian_grid(tmp_path):
    """Per-cell shapely intersection handles Gaussian-style non-uniform lat.

    Most CMIP6 atmosphere models use Gaussian latitudes (T63 ~2.8°, T127
    ~0.94°, T255 ~0.7°) where ``diff(lat)`` varies smoothly across the grid.
    Pre-fix, ``shapefile_fractional`` raised ``regionmask.InvalidCoordsError``
    on these grids — we replaced the regionmask path with a per-cell shapely
    intersection that accepts any monotone 1-D lat/lon. This regression
    locks the new behaviour: a non-uniform-lat grid plus the TINY country
    fixture must succeed and produce a non-zero weight.
    """
    config = _build_tiny_country_config(tmp_path)

    # Sin-spaced lat values — small steps near the poles, larger near the
    # equator (the same geometry as a true Gaussian grid).
    n_lat = 30
    sin_centers = np.linspace(-np.pi / 2 + 0.05, np.pi / 2 - 0.05, n_lat)
    lat = np.degrees(np.sin(sin_centers)) * 90.0 / np.degrees(np.sin(np.pi / 2 - 0.05))
    diff_lat = np.diff(lat)
    assert diff_lat.std() / abs(diff_lat.mean()) > 1e-3, (
        f"synthesised grid must be non-uniform; "
        f"got std/mean = {diff_lat.std() / abs(diff_lat.mean()):.6f}"
    )
    # Cover the TINY country bbox (lat 33.9–34.9, lon 28.95–29.45).
    lon = np.arange(0.0, 60.0, 2.0)
    da = xr.DataArray(
        np.ones((n_lat, len(lon))),
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"],
        name="t",
    )

    result = geom.crop(da, "TINY", method="shapefile_fractional", config=config)
    assert result.weight is not None
    weight_sum = float(result.weight.sum())
    assert weight_sum > 0, f"expected positive Σweight, got {weight_sum}"
    # Every weight must be in [0, 1].
    weights = result.weight.values
    assert weights.min() >= 0.0 and weights.max() <= 1.0


def test_lenient_includes_cell_where_polygon_overshoots_country_bbox_by_less_than_half_a_cell(
    tmp_path,
):
    """Regression for M5 issue 1 (mask-aware bbox).

    Pre-fix, ``crop()`` cropped the mask with ``country_bbox + box_offset=1°``,
    which is less than half a 2.8° cell. A polygon that extends past the
    stored country_bbox by less than half a cell on the north edge would
    have its northernmost touched cell silently dropped. Post-fix the crop
    is mask-aware: the smallest bbox enclosing all True cells, padded by one
    full cell. The dropped cell is now correctly retained.

    Setup: 2.8° grid; country_bbox.lat_max = 32.0; polygon extends to lat
    33.3 (1.3° past country_bbox, less than the 1.4° half-cell). The cell
    centred at lat 34.4 covers lat 33.0-35.8 and is touched by the polygon
    (lenient = any-touch). country_bbox+1° = lat 30.0-33.0 — exactly excludes
    the cell at lat 34.4.
    """
    cell_lon = 29.2  # known cell centre in the 2.8° grid below
    poly_north = 33.3  # 1.3° past country_bbox.lat_max = 32.0; < 1.4° = ½ cell

    poly = Polygon(
        [
            (cell_lon - 0.25, 31.0),
            (cell_lon + 0.25, 31.0),
            (cell_lon + 0.25, poly_north),
            (cell_lon - 0.25, poly_north),
        ]
    )
    gpkg = tmp_path / "overshoot.gpkg"
    gpd.GeoDataFrame(
        {"COUNTRY": ["OVERSHOOT"]}, geometry=[poly], crs="EPSG:4326"
    ).to_file(gpkg, driver="GPKG")

    data_root = tmp_path / "data"
    (data_root / "country_codes").mkdir(parents=True)
    (data_root / "country_codes" / "country_codes.json").write_text(
        json.dumps(
            {
                "OVERSHOOT": {
                    "name": "OVERSHOOT",
                    "alpha-2": "OV",
                    "alpha-3": "OVS",
                    "boundingBox": {
                        # Stored bbox stops short of the polygon on the north.
                        "sw": {"lat": 31.0, "lon": cell_lon - 0.25},
                        "ne": {"lat": 32.0, "lon": cell_lon + 0.25},
                    },
                }
            }
        )
    )
    config = Config(
        data_root=data_root,
        cache_root=tmp_path / "cache",
        shapefile_path=gpkg,
        reference_root=data_root / "reference" / "monthly_cmip6_upscaled",
        single_grid_reference_root=data_root / "reference" / "monthly_upscaled",
        cmip6_metadata_root=data_root / "CMIP6" / "metadata",
        results_root=tmp_path / "results",
    )
    da = _build_28deg_grid()

    result = geom.crop(da, "OVERSHOOT", method="shapefile_lenient", config=config)
    n_included = int((~result.data.isnull()).sum())
    assert n_included >= 2, (
        f"expected ≥2 cells (the country_bbox cell at lat 31.6 plus the "
        f"overshoot cell at lat 34.4); got {n_included}. The cell at lat 34.4 "
        f"is the one that pre-fix country_bbox + box_offset=1° silently dropped."
    )

    # Specifically assert the overshoot cell is in the result.
    lat_values = result.data["lat"].values
    assert any(np.isclose(lat_values, 34.4)), (
        f"the overshoot cell at lat 34.4 must be in the cropped result; "
        f"present lat values: {lat_values.tolist()}"
    )
