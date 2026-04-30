"""Unit tests for `subselect.io`.

Covers the model-list loader (placeholder-skipping, ordering preservation),
the per-variable reference filename templates (tas/pr use the merged
GSWP3-W5E5 product 1901–2019; psl is W5E5-only 1991–2019), and the CMIP6
glob-based file discovery (variant labels resolved automatically). Bbox
lookup is tested against a synthetic country_codes.json.
"""

from __future__ import annotations

import xarray as xr
import pytest

from subselect import io
from tests.conftest import TEST_COUNTRY_BOUNDS, TEST_MODELS


# ---------- load_models_list -----------------------------------------------


def test_load_models_list_skips_placeholder_and_preserves_order(synthetic_config):
    models = io.load_models_list(synthetic_config)
    assert models == TEST_MODELS  # placeholder dropped, order preserved


def test_load_models_list_handles_no_placeholder(synthetic_data_root, synthetic_config):
    csv = synthetic_data_root / "models_ordered.csv"
    csv.write_text("\n".join(TEST_MODELS) + "\n")  # no leading 'aaaa'
    assert io.load_models_list(synthetic_config) == TEST_MODELS


# ---------- reference_path / load_w5e5 -------------------------------------


@pytest.mark.parametrize(
    ("variable", "expected_filename"),
    [
        ("tas", "tas_gswp3-w5e5_obsclim_mon_1901_2019_AAA-MOD.nc"),
        ("pr", "pr_gswp3-w5e5_obsclim_mon_1901_2019_AAA-MOD.nc"),
        ("tasmax", "tasmax_gswp3-w5e5_obsclim_mon_1901_2019_AAA-MOD.nc"),
        ("psl", "psl_w5e5_obsclim_mon_1991_2019_AAA-MOD.nc"),
    ],
)
def test_reference_path_per_variable_filename_templates(
    synthetic_config, variable, expected_filename
):
    path = io.reference_path(variable, "AAA-MOD", config=synthetic_config)
    assert path.name == expected_filename
    assert path.parent == synthetic_config.reference_root / "AAA-MOD"


def test_reference_path_unknown_variable_raises(synthetic_config):
    with pytest.raises(ValueError, match="No reference filename template"):
        io.reference_path("hurs", "AAA-MOD", config=synthetic_config)


def test_load_w5e5_missing_file_raises(synthetic_config):
    with pytest.raises(FileNotFoundError, match="W5E5 reference file"):
        io.load_w5e5("tas", "AAA-MOD", config=synthetic_config)


# ---------- single_grid_reference_path / load_single_grid_w5e5 -------------


@pytest.mark.parametrize(
    ("variable", "expected_filename"),
    [
        ("tas", "tas_gswp3-w5e5_obsclim_mon_1901_2019_cmip6_upscaled.nc"),
        ("pr", "pr_gswp3-w5e5_obsclim_mon_1901_2019_cmip6_upscaled.nc"),
        ("tasmax", "tasmax_gswp3-w5e5_obsclim_mon_1901_2019_cmip6_upscaled.nc"),
        ("psl", "psl_w5e5_obsclim_mon_1991_2019_cmip6_upscaled.nc"),
    ],
)
def test_single_grid_reference_path_per_variable_filename_templates(
    synthetic_config, variable, expected_filename
):
    path = io.single_grid_reference_path(variable, config=synthetic_config)
    assert path.name == expected_filename
    # Single-grid files live directly under the reference root, no per-model dir.
    assert path.parent == synthetic_config.single_grid_reference_root


def test_single_grid_reference_path_unknown_variable_raises(synthetic_config):
    with pytest.raises(ValueError, match="No single-grid reference filename template"):
        io.single_grid_reference_path("hurs", config=synthetic_config)


def test_load_single_grid_w5e5_missing_file_raises(synthetic_config):
    with pytest.raises(FileNotFoundError, match="Single-grid W5E5 reference"):
        io.load_single_grid_w5e5("tas", config=synthetic_config)


# ---------- cmip6_path / load_cmip6 ----------------------------------------


def test_cmip6_path_resolves_variant_label(synthetic_config, synthetic_data_root):
    cmip6_dir = synthetic_data_root / "CMIP6" / "monthly" / "tas" / "ssp585"
    cmip6_dir.mkdir(parents=True)
    nc_file = cmip6_dir / "tas_AAA-MOD_r1i1p1f1_mon_ssp585.nc"
    # Write a minimal valid NetCDF so the path test is realistic.
    xr.Dataset({"tas": ("time", [0.0])}).to_netcdf(nc_file)

    resolved = io.cmip6_path("tas", "ssp585", "AAA-MOD", config=synthetic_config)
    assert resolved == nc_file


def test_cmip6_path_no_match_raises(synthetic_config, synthetic_data_root):
    (synthetic_data_root / "CMIP6" / "monthly" / "tas" / "ssp585").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No CMIP6 file"):
        io.cmip6_path("tas", "ssp585", "AAA-MOD", config=synthetic_config)


def test_cmip6_path_missing_dir_raises(synthetic_config):
    with pytest.raises(FileNotFoundError, match="CMIP6 directory not found"):
        io.cmip6_path("tas", "ssp585", "AAA-MOD", config=synthetic_config)


def test_cmip6_path_multiple_matches_raises(synthetic_config, synthetic_data_root):
    cmip6_dir = synthetic_data_root / "CMIP6" / "monthly" / "tas" / "ssp585"
    cmip6_dir.mkdir(parents=True)
    for variant in ("r1i1p1f1", "r1i1p1f2"):
        path = cmip6_dir / f"tas_AAA-MOD_{variant}_mon_ssp585.nc"
        xr.Dataset({"tas": ("time", [0.0])}).to_netcdf(path)
    with pytest.raises(ValueError, match="Multiple CMIP6 files"):
        io.cmip6_path("tas", "ssp585", "AAA-MOD", config=synthetic_config)


def test_load_cmip6_opens_dataset(synthetic_config, synthetic_data_root):
    cmip6_dir = synthetic_data_root / "CMIP6" / "monthly" / "tas" / "ssp585"
    cmip6_dir.mkdir(parents=True)
    nc_file = cmip6_dir / "tas_AAA-MOD_r1i1p1f1_mon_ssp585.nc"
    xr.Dataset({"tas": ("time", [1.0, 2.0, 3.0])}).to_netcdf(nc_file)

    ds = io.load_cmip6("tas", "ssp585", "AAA-MOD", config=synthetic_config)
    assert "tas" in ds
    assert ds["tas"].size == 3


# ---------- bbox lookup -----------------------------------------------------


def test_country_bbox_resolves_by_name(synthetic_config):
    box = io.country_bbox("TEST", config=synthetic_config)
    assert box == TEST_COUNTRY_BOUNDS


@pytest.mark.parametrize("alias", ["test", "TEST", "Te", "TST", "tst"])
def test_country_bbox_case_insensitive_aliases(synthetic_config, alias):
    box = io.country_bbox(alias, config=synthetic_config)
    assert box == TEST_COUNTRY_BOUNDS


def test_country_bbox_unknown_raises(synthetic_config):
    with pytest.raises(ValueError, match="not found in country_codes"):
        io.country_bbox("NOWHERE", config=synthetic_config)
