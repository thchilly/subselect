"""Unit tests for `subselect.cache`.

Covers all three path conventions (per-country / per-(scenario, season),
per-country time-series, global / no-country-scope), zarr round-trip, the
SQLite catalog (register / lookup / invalidate / list_all / ON CONFLICT
upsert), the ``_global`` sentinel handling, ``CacheMiss`` semantics, and the
xlsx human-export bundle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from subselect import cache
from subselect.cache import (
    CATALOG_FILENAME,
    GLOBAL_COUNTRY,
    NOT_APPLICABLE,
    TIMESERIES_SEASON,
    ArtefactRecord,
    CacheMiss,
    Catalog,
)
from subselect.config import Config


@pytest.fixture
def cache_config(tmp_path) -> Config:
    """Minimal Config wired only at cache_root for cache-layer tests."""
    return Config(
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        shapefile_path=tmp_path / "shape.gpkg",
        reference_root=tmp_path / "data" / "reference" / "monthly_cmip6_upscaled",
        single_grid_reference_root=tmp_path / "data" / "reference" / "monthly_upscaled",
        cmip6_metadata_root=tmp_path / "data" / "CMIP6" / "metadata",
        results_root=tmp_path / "results",
    )


def _hps_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"HPS_annual": [0.81, 0.42, 0.93], "BVS_annual": [0.55, 0.61, 0.78]},
        index=pd.Index(["ACCESS-CM2", "CanESM5", "MPI-ESM1-2-HR"], name="model"),
    )


def _timeseries_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for model in ["ACCESS-CM2", "CanESM5"]:
        for scenario in ["ssp245", "ssp585"]:
            for year in range(1995, 2001):
                rows.append((model, scenario, year, float(rng.standard_normal())))
    return pd.DataFrame(rows, columns=["model", "scenario", "year", "value"])


def _gwl_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"crossing_year": [2031, 2045, 2058]},
        index=pd.MultiIndex.from_tuples(
            [
                ("ACCESS-CM2", "ssp585", 1.5),
                ("ACCESS-CM2", "ssp585", 2.0),
                ("ACCESS-CM2", "ssp585", 3.0),
            ],
            names=["model", "scenario", "gwl_threshold"],
        ),
    )


def _toy_dataarray() -> xr.DataArray:
    lat = np.arange(30.0, 35.0)
    lon = np.arange(20.0, 26.0)
    return xr.DataArray(
        np.random.default_rng(42).standard_normal((len(lat), len(lon))),
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"],
        name="climatology",
    )


# ---------- Catalog: schema + register / lookup --------------------------


def test_catalog_initialises_schema_and_db_file(cache_config):
    catalog = Catalog(cache_config.cache_root / CATALOG_FILENAME)
    assert catalog.db_path.is_file()
    assert catalog.list_all() == []


def test_register_then_lookup_round_trip(cache_config):
    cat = Catalog(cache_config.cache_root / CATALOG_FILENAME)
    cat.register(
        country="greece", scenario="ssp585", season="annual", kind="hps",
        crop_method="bbox", path="parquet/greece/ssp585/annual/hps__bbox.parquet",
        format="parquet", code_version="0.1.0.dev0", config_hash="",
    )
    record = cat.lookup(
        country="greece", scenario="ssp585", season="annual", kind="hps",
        crop_method="bbox",
    )
    assert isinstance(record, ArtefactRecord)
    assert record.path == "parquet/greece/ssp585/annual/hps__bbox.parquet"
    assert record.format == "parquet"
    assert record.code_version == "0.1.0.dev0"
    assert record.created_at  # populated by SQLite default


def test_lookup_unknown_raises_cachemiss(cache_config):
    cat = Catalog(cache_config.cache_root / CATALOG_FILENAME)
    with pytest.raises(CacheMiss, match="no catalog entry"):
        cat.lookup(
            country="greece", scenario="ssp585", season="annual",
            kind="hps", crop_method="bbox",
        )


def test_register_same_key_updates_existing_row(cache_config):
    """ON CONFLICT must upsert path + created_at, not insert a duplicate."""
    cat = Catalog(cache_config.cache_root / CATALOG_FILENAME)
    natural_key = dict(
        country="greece", scenario="ssp585", season="annual", kind="hps",
        crop_method="bbox",
    )
    cat.register(**natural_key, path="old/path.parquet", format="parquet")
    cat.register(**natural_key, path="new/path.parquet", format="parquet")
    rows = cat.list_all()
    assert len(rows) == 1
    assert rows[0].path == "new/path.parquet"


def test_list_all_orders_by_natural_key(cache_config):
    cat = Catalog(cache_config.cache_root / CATALOG_FILENAME)
    cat.register(
        country="zenland", scenario="ssp585", season="annual", kind="hps",
        crop_method="bbox", path="z.parquet", format="parquet",
    )
    cat.register(
        country="atland", scenario="ssp585", season="annual", kind="hps",
        crop_method="bbox", path="a.parquet", format="parquet",
    )
    countries = [r.country for r in cat.list_all()]
    assert countries == ["atland", "zenland"]


# ---------- Catalog: invalidate ------------------------------------------


def test_invalidate_by_country_removes_rows_and_files(cache_config, tmp_path):
    cat = Catalog(cache_config.cache_root / CATALOG_FILENAME)
    file_a = cache_config.cache_root / "a.parquet"
    file_a.parent.mkdir(parents=True, exist_ok=True)
    file_a.write_text("payload")
    cat.register(
        country="greece", scenario="ssp585", season="annual", kind="hps",
        crop_method="bbox", path="a.parquet", format="parquet",
    )
    file_b = cache_config.cache_root / "b.parquet"
    file_b.write_text("payload")
    cat.register(
        country="cyprus", scenario="ssp585", season="annual", kind="hps",
        crop_method="bbox", path="b.parquet", format="parquet",
    )
    deleted = cat.invalidate(country="greece")
    assert deleted == 1
    assert not file_a.exists()  # removed
    assert file_b.exists()  # other country untouched
    assert [r.country for r in cat.list_all()] == ["cyprus"]


def test_invalidate_by_kind_filter(cache_config):
    cat = Catalog(cache_config.cache_root / CATALOG_FILENAME)
    # Two distinct hps rows (different countries) plus one spread row.
    for country in ("greece", "cyprus"):
        cat.register(
            country=country, scenario="ssp585", season="annual", kind="hps",
            crop_method="bbox", path=f"hps_{country}.parquet", format="parquet",
        )
    cat.register(
        country="greece", scenario="ssp585", season="annual", kind="spread",
        crop_method="bbox", path="spread_greece.parquet", format="parquet",
    )
    deleted = cat.invalidate(kind="hps")
    assert deleted == 2
    remaining = cat.list_all()
    assert len(remaining) == 1
    assert remaining[0].kind == "spread"


def test_invalidate_skips_already_missing_files(cache_config):
    """Partial-state cleanup must not error out."""
    cat = Catalog(cache_config.cache_root / CATALOG_FILENAME)
    cat.register(
        country="greece", scenario="ssp585", season="annual", kind="hps",
        crop_method="bbox", path="never_existed.parquet", format="parquet",
    )
    assert cat.invalidate(country="greece") == 1


def test_invalidate_no_filters_wipes_catalog(cache_config):
    cat = Catalog(cache_config.cache_root / CATALOG_FILENAME)
    cat.register(
        country="greece", scenario="ssp585", season="annual", kind="hps",
        crop_method="bbox", path="a.parquet", format="parquet",
    )
    cat.register(
        country="cyprus", scenario="ssp245", season="DJF", kind="spread",
        crop_method="bbox", path="b.parquet", format="parquet",
    )
    assert cat.invalidate() == 2
    assert cat.list_all() == []


def test_invalidate_no_match_returns_zero(cache_config):
    cat = Catalog(cache_config.cache_root / CATALOG_FILENAME)
    assert cat.invalidate(country="atlantis") == 0


# ---------- Parquet round-trip: per-country ------------------------------


def test_write_and_read_parquet_per_country(cache_config):
    df = _hps_frame()
    abs_path = cache.write_parquet(
        df, country="greece", kind="hps", scenario="ssp585", season="annual",
        crop_method="bbox", config=cache_config,
    )
    assert abs_path.is_file()
    assert abs_path.suffix == ".parquet"

    loaded = cache.read_parquet(
        country="greece", kind="hps", scenario="ssp585", season="annual",
        crop_method="bbox", config=cache_config,
    )
    pd.testing.assert_frame_equal(loaded, df)


def test_read_parquet_unknown_raises_cachemiss(cache_config):
    with pytest.raises(CacheMiss):
        cache.read_parquet(
            country="greece", kind="hps", scenario="ssp585", season="annual",
            crop_method="bbox", config=cache_config,
        )


def test_read_parquet_with_missing_file_raises_cachemiss(cache_config):
    df = _hps_frame()
    abs_path = cache.write_parquet(
        df, country="greece", kind="hps", scenario="ssp585", season="annual",
        crop_method="bbox", config=cache_config,
    )
    abs_path.unlink()  # simulate disk-cache drift
    with pytest.raises(CacheMiss, match="parquet file is missing"):
        cache.read_parquet(
            country="greece", kind="hps", scenario="ssp585", season="annual",
            crop_method="bbox", config=cache_config,
        )


def test_parquet_per_country_path_convention(cache_config):
    df = _hps_frame()
    abs_path = cache.write_parquet(
        df, country="greece", kind="hps", scenario="ssp585", season="annual",
        crop_method="shapefile_lenient", config=cache_config,
    )
    expected = (
        cache_config.cache_root / "parquet" / "greece" / "ssp585" / "annual"
        / "hps__shapefile_lenient.parquet"
    )
    assert abs_path == expected


def test_different_crop_methods_do_not_collide(cache_config):
    df_a = _hps_frame()
    df_b = _hps_frame() * 2.0
    cache.write_parquet(
        df_a, country="greece", kind="hps", scenario="ssp585", season="annual",
        crop_method="bbox", config=cache_config,
    )
    cache.write_parquet(
        df_b, country="greece", kind="hps", scenario="ssp585", season="annual",
        crop_method="shapefile_lenient", config=cache_config,
    )
    bbox_loaded = cache.read_parquet(
        country="greece", kind="hps", scenario="ssp585", season="annual",
        crop_method="bbox", config=cache_config,
    )
    lenient_loaded = cache.read_parquet(
        country="greece", kind="hps", scenario="ssp585", season="annual",
        crop_method="shapefile_lenient", config=cache_config,
    )
    pd.testing.assert_frame_equal(bbox_loaded, df_a)
    pd.testing.assert_frame_equal(lenient_loaded, df_b)


# ---------- Parquet round-trip: time-series ------------------------------


def test_write_and_read_parquet_timeseries(cache_config):
    df = _timeseries_frame()
    abs_path = cache.write_parquet_timeseries(
        df, country="greece", variable="tas", crop_method="bbox",
        config=cache_config,
    )
    expected = (
        cache_config.cache_root / "parquet" / "greece" / "timeseries"
        / "tas__bbox.parquet"
    )
    assert abs_path == expected

    loaded = cache.read_parquet_timeseries(
        country="greece", variable="tas", crop_method="bbox",
        config=cache_config,
    )
    pd.testing.assert_frame_equal(loaded, df)


def test_timeseries_catalog_row_uses_sentinels(cache_config):
    cache.write_parquet_timeseries(
        _timeseries_frame(), country="greece", variable="tas",
        crop_method="bbox", config=cache_config,
    )
    cat = Catalog(cache_config.cache_root / CATALOG_FILENAME)
    row = cat.list_all()[0]
    assert row.season == TIMESERIES_SEASON
    assert row.scenario == NOT_APPLICABLE
    assert row.kind == "tas"


# ---------- Parquet round-trip: global -----------------------------------


def test_write_and_read_parquet_global(cache_config):
    df = _gwl_frame()
    abs_path = cache.write_parquet_global(
        df, kind="gwl_crossing_years", config=cache_config,
    )
    expected = (
        cache_config.cache_root / "parquet" / GLOBAL_COUNTRY
        / "gwl_crossing_years.parquet"
    )
    assert abs_path == expected

    loaded = cache.read_parquet_global(
        kind="gwl_crossing_years", config=cache_config,
    )
    pd.testing.assert_frame_equal(loaded, df)


def test_global_catalog_row_uses_global_sentinel(cache_config):
    cache.write_parquet_global(
        _gwl_frame(), kind="gwl_crossing_years", config=cache_config,
    )
    cat = Catalog(cache_config.cache_root / CATALOG_FILENAME)
    row = cat.list_all()[0]
    assert row.country == GLOBAL_COUNTRY
    assert row.scenario == NOT_APPLICABLE
    assert row.season == NOT_APPLICABLE
    assert row.crop_method == NOT_APPLICABLE


def test_global_invalidate_via_country_sentinel(cache_config):
    cache.write_parquet_global(
        _gwl_frame(), kind="gwl_crossing_years", config=cache_config,
    )
    cat = Catalog(cache_config.cache_root / CATALOG_FILENAME)
    deleted = cat.invalidate(country=GLOBAL_COUNTRY)
    assert deleted == 1


# ---------- Zarr round-trip ----------------------------------------------


def test_write_and_read_zarr(cache_config):
    da = _toy_dataarray()
    abs_path = cache.write_zarr(
        da, country="greece", scenario="ssp585", artefact="climatology",
        crop_method="shapefile_lenient", config=cache_config,
    )
    assert abs_path.is_dir()  # zarr is a directory store
    expected = (
        cache_config.cache_root / "zarr" / "greece" / "ssp585"
        / "climatology__shapefile_lenient.zarr"
    )
    assert abs_path == expected

    loaded = cache.read_zarr(
        country="greece", scenario="ssp585", artefact="climatology",
        crop_method="shapefile_lenient", config=cache_config,
    )
    xr.testing.assert_allclose(loaded["climatology"], da)


def test_invalidate_removes_zarr_directory(cache_config):
    da = _toy_dataarray()
    abs_path = cache.write_zarr(
        da, country="greece", scenario="ssp585", artefact="climatology",
        crop_method="bbox", config=cache_config,
    )
    assert abs_path.is_dir()
    cat = Catalog(cache_config.cache_root / CATALOG_FILENAME)
    deleted = cat.invalidate(country="greece")
    assert deleted == 1
    assert not abs_path.exists()


# ---------- xlsx export ---------------------------------------------------


def test_export_to_xlsx_bundles_country_artefacts(cache_config, tmp_path):
    cache.write_parquet(
        _hps_frame(), country="greece", kind="hps", scenario="ssp585",
        season="annual", crop_method="bbox", config=cache_config,
    )
    cache.write_parquet(
        _hps_frame(), country="greece", kind="hps", scenario="ssp585",
        season="DJF", crop_method="bbox", config=cache_config,
    )
    cache.write_parquet(
        _hps_frame(), country="cyprus", kind="hps", scenario="ssp585",
        season="annual", crop_method="bbox", config=cache_config,
    )
    out = tmp_path / "greece_export.xlsx"
    cache.export_to_xlsx("greece", out, config=cache_config)
    assert out.is_file()
    sheets = pd.ExcelFile(out).sheet_names
    assert len(sheets) == 2  # two greece artefacts only; cyprus excluded


def test_export_to_xlsx_no_artefacts_raises_cachemiss(cache_config, tmp_path):
    with pytest.raises(CacheMiss, match="no parquet artefacts"):
        cache.export_to_xlsx(
            "greece", tmp_path / "x.xlsx", config=cache_config,
        )


# ---------- Cross-cutting: catalog index is durable ----------------------


def test_catalog_persists_across_instances(cache_config):
    """Two Catalog() instances against the same db file see the same rows."""
    db = cache_config.cache_root / CATALOG_FILENAME
    a = Catalog(db)
    a.register(
        country="greece", scenario="ssp585", season="annual", kind="hps",
        crop_method="bbox", path="x.parquet", format="parquet",
    )
    b = Catalog(db)
    assert len(b.list_all()) == 1
