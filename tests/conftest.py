"""Shared test fixtures.

Synthetic Data/ tree builders so io and geom tests stay offline-runnable
(no GADM, no real CMIP6 / W5E5 NetCDFs needed). The synthetic country is a
square `TEST` polygon at lat 35–42°N, lon 20–30°E.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Polygon

from subselect.config import Config

TEST_COUNTRY_BOUNDS = {"lat_min": 35.0, "lat_max": 42.0, "lon_min": 20.0, "lon_max": 30.0}
TEST_MODELS = ["AAA-MOD", "BBB-MOD", "CCC-MOD"]  # synthetic; not real CMIP6 names


@pytest.fixture(scope="session")
def synthetic_gpkg(tmp_path_factory):
    """Single-feature GADM-style geopackage for the `TEST` country."""
    b = TEST_COUNTRY_BOUNDS
    poly = Polygon(
        [
            (b["lon_min"], b["lat_min"]),
            (b["lon_max"], b["lat_min"]),
            (b["lon_max"], b["lat_max"]),
            (b["lon_min"], b["lat_max"]),
        ]
    )
    gdf = gpd.GeoDataFrame({"COUNTRY": ["TEST"]}, geometry=[poly], crs="EPSG:4326")
    path = tmp_path_factory.mktemp("shapes") / "test_country.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


@pytest.fixture
def synthetic_data():
    """1° lat/lon DataArray covering the TEST country plus a buffer ring."""
    lat = np.arange(30.0, 47.0, 1.0)  # 30..46 inclusive
    lon = np.arange(15.0, 35.0, 1.0)  # 15..34 inclusive
    rng = np.random.default_rng(42)
    data = rng.standard_normal((len(lat), len(lon)))
    return xr.DataArray(
        data,
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"],
        name="testvar",
    )


@pytest.fixture
def synthetic_data_root(tmp_path):
    """A Data/ tree containing country_codes.json + models_ordered.csv for TEST."""
    data_root = tmp_path / "data"
    (data_root / "country_codes").mkdir(parents=True)
    b = TEST_COUNTRY_BOUNDS
    country_codes = {
        "TEST": {
            "name": "TEST",
            "alpha-2": "TE",
            "alpha-3": "TST",
            "boundingBox": {
                "sw": {"lat": b["lat_min"], "lon": b["lon_min"]},
                "ne": {"lat": b["lat_max"], "lon": b["lon_max"]},
            },
        }
    }
    (data_root / "country_codes" / "country_codes.json").write_text(
        json.dumps(country_codes)
    )
    (data_root / "models_ordered.csv").write_text(
        "aaaa\n" + "\n".join(TEST_MODELS) + "\n"
    )
    return data_root


@pytest.fixture
def synthetic_config(synthetic_data_root, synthetic_gpkg, tmp_path):
    """Config wired to the synthetic Data/ tree and TEST geopackage."""
    return Config(
        data_root=synthetic_data_root,
        cache_root=tmp_path / "cache",
        shapefile_path=synthetic_gpkg,
        reference_root=synthetic_data_root / "reference" / "monthly_cmip6_upscaled",
        single_grid_reference_root=synthetic_data_root / "reference" / "monthly_upscaled",
        cmip6_metadata_root=synthetic_data_root / "CMIP6" / "metadata",
        results_root=tmp_path / "results",
    )
