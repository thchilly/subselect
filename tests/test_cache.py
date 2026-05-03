"""Unit tests for the L1 per-country cache layer.

Covers DataFrame round-trip, dict-of-DataFrame round-trip, dataset round-trip,
mtime-based staleness, invalidation, clear, and the dataclass round-trip
that backs ProfileSignals caching.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from subselect.cache import Cache


def test_dataframe_round_trip(tmp_path: Path) -> None:
    cache = Cache("test", tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.5, 5.5, 6.5]}, index=[10, 20, 30])
    df.index.name = "yr"

    cache.save("table", df, deps=[])
    assert cache.has("table")
    loaded = cache.load("table")
    pd.testing.assert_frame_equal(loaded, df)


def test_dict_of_dataframe_round_trip(tmp_path: Path) -> None:
    cache = Cache("test", tmp_path)
    d = {
        "tas": pd.DataFrame({"v": [1.0, 2.0]}),
        "pr": pd.DataFrame({"v": [10.0, 20.0]}),
    }
    cache.save("metrics", d, deps=[])
    loaded = cache.load("metrics")
    assert set(loaded) == {"tas", "pr"}
    pd.testing.assert_frame_equal(loaded["tas"], d["tas"])


def test_dataset_round_trip(tmp_path: Path) -> None:
    cache = Cache("test", tmp_path)
    ds = xr.Dataset({"tas": (("lat", "lon"), np.arange(6).reshape(2, 3) * 1.0)})
    cache.save("field", ds, deps=[])
    loaded = cache.load("field")
    xr.testing.assert_equal(loaded, ds)


def test_staleness(tmp_path: Path) -> None:
    cache = Cache("test", tmp_path)
    dep = tmp_path / "input.nc"
    dep.touch()

    df = pd.DataFrame({"a": [1, 2]})
    cache.save("artefact", df, deps=[dep])
    assert cache.is_fresh("artefact", [dep])

    time.sleep(0.05)
    dep.touch()
    assert not cache.is_fresh("artefact", [dep])


def test_invalidate(tmp_path: Path) -> None:
    cache = Cache("test", tmp_path)
    df = pd.DataFrame({"a": [1, 2]})
    cache.save("artefact", df, deps=[])
    assert cache.has("artefact")
    cache.invalidate("artefact")
    assert not cache.has("artefact")


def test_clear(tmp_path: Path) -> None:
    cache = Cache("test", tmp_path)
    cache.save("a", pd.DataFrame({"x": [1]}), deps=[])
    cache.save("b", pd.DataFrame({"y": [2]}), deps=[])
    cache.clear()
    assert not cache.has("a")
    assert not cache.has("b")


def test_cross_scope_invalidation(tmp_path: Path) -> None:
    """A per-country artefact whose dep set includes the global cache's
    ``catalog.json`` must auto-invalidate when the global cache mutates.

    This is the cross-scope invalidation rule: any per-country derivation
    that consumes a global-cache artefact (climatologies, σ_obs grids,
    annual fields, etc.) lists ``cache/_global/catalog.json`` as a dep so
    saves to the global cache propagate the invalidation downstream.
    """
    global_cache = Cache.global_cache(tmp_path)
    country_cache = Cache("greece", tmp_path)

    # Populate a global artefact, then a per-country artefact that depends
    # on the global catalog.
    global_cache.save("clim", pd.DataFrame({"v": [1.0, 2.0]}), deps=[])
    catalog_dep = [tmp_path / "_global" / "catalog.json"]
    country_cache.save("derived", pd.DataFrame({"v": [10.0, 20.0]}), deps=catalog_dep)
    assert country_cache.is_fresh("derived", catalog_dep)

    # Mutate the global cache by saving a new artefact — this rewrites
    # catalog.json and advances its mtime.
    time.sleep(0.05)
    global_cache.save("other_clim", pd.DataFrame({"v": [3.0, 4.0]}), deps=[])

    # The per-country derivation should now be stale.
    assert not country_cache.is_fresh("derived", catalog_dep)


def test_crop_method_invalidation(tmp_path: Path) -> None:
    """Two per-country caches keyed by different ``crop_method`` values
    must not collide.

    Concretely: writing the same logical key from two ``Cache`` instances
    (one with ``crop_method="bbox"``, one with ``crop_method="shapefile_lenient"``)
    should produce two distinct artefact entries; reading from each instance
    returns its own value; clearing one leaves the other intact.
    """
    bbox_cache = Cache("greece", tmp_path, crop_method="bbox")
    shp_cache = Cache("greece", tmp_path, crop_method="shapefile_lenient")

    df_bbox = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
    df_shp = pd.DataFrame({"v": [10.0, 20.0, 30.0]}, index=[10, 20, 30])

    bbox_cache.save("metric", df_bbox, deps=[])
    shp_cache.save("metric", df_shp, deps=[])

    # Both lookups succeed and return distinct values.
    assert bbox_cache.has("metric")
    assert shp_cache.has("metric")
    pd.testing.assert_frame_equal(bbox_cache.load("metric"), df_bbox)
    pd.testing.assert_frame_equal(shp_cache.load("metric"), df_shp)
    assert not df_bbox.equals(df_shp)

    # The catalog now records both — under suffixed keys.
    fresh = Cache("greece", tmp_path, crop_method="bbox")
    assert "metric__bbox" in fresh._catalog
    assert "metric__shapefile_lenient" in fresh._catalog

    # On-disk filenames carry the suffix and are distinct.
    assert (tmp_path / "greece" / "parquet" / "metric__bbox.parquet").is_file()
    assert (tmp_path / "greece" / "parquet" / "metric__shapefile_lenient.parquet").is_file()

    # Invalidating one method leaves the other intact.
    bbox_cache.invalidate("metric")
    assert not bbox_cache.has("metric")
    assert shp_cache.has("metric")
    pd.testing.assert_frame_equal(shp_cache.load("metric"), df_shp)


def test_dataclass_round_trip(tmp_path: Path) -> None:
    from subselect.state import ProfileSignals

    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=[2000, 2001, 2002])
    series = pd.Series([0.1, 0.2], index=["a", "b"], name="baseline")

    ps = ProfileSignals(
        annual_temperature=df, tas_pi_baseline=series, tas_rp_baseline=series,
        tas_baseline_offset=0.5,
        tas_anomaly_pi=df, tas_anomaly_rp=df,
        stats_tas_anomaly_pi=df, stats_tas_anomaly_rp=df,
        annual_precipitation=df, pr_pi_baseline=series, pr_rp_baseline=series,
        pr_smoothed=df, pr_anomaly_pi=df, pr_anomaly_rp=df,
        stats_pr_anomaly_pi=df, stats_pr_anomaly_rp=df,
        pr_baseline_offset=0.1, pr_baseline_offset_percent=2.5, pr_ax_ratio=1.05,
        pr_pi_percent_change=df, pr_rp_percent_change=df,
        stats_pr_pi_percent_change=df, stats_pr_rp_percent_change=df,
        tas_anomalies_table=pd.DataFrame({"col": ["a", "b"]}),
        pr_percent_anom_table=pd.DataFrame({"col": ["c", "d"]}),
    )

    cache = Cache("test", tmp_path)
    cache.save("ps", ps, deps=[])
    loaded = cache.load("ps")
    assert loaded.tas_baseline_offset == pytest.approx(0.5)
    assert loaded.pr_ax_ratio == pytest.approx(1.05)
    pd.testing.assert_frame_equal(loaded.annual_temperature, df)
