"""Unit tests for `subselect.config.Config` and the path-resolution chain.

Covers the three resolution branches (env var → ~/.subselect.toml → repo
default), TOML overrides for every path field, the cache/results sibling-
of-Data/ defaults, the frozen-dataclass contract, and the with_overrides
helper.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from subselect import config as cfg
from subselect.config import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_METADATA_RELATIVE,
    DEFAULT_REFERENCE_RELATIVE,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SHAPEFILE_RELATIVE,
    DEFAULT_SINGLE_GRID_REFERENCE_RELATIVE,
    DIAGNOSTIC_VARIABLES,
    ENV_VAR_DATA_ROOT,
    HPS_VARIABLES,
    Config,
)


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    """Strip env var and redirect the user-TOML path into a temp directory.

    Each test gets a fresh temp directory; the test decides whether to drop
    a `~/.subselect.toml` file there. This isolates from the developer's
    real `~/.subselect.toml` if one exists.
    """
    monkeypatch.delenv(ENV_VAR_DATA_ROOT, raising=False)
    fake_user_toml = tmp_path / ".subselect.toml"
    monkeypatch.setattr(cfg, "USER_CONFIG_PATH", fake_user_toml)
    return fake_user_toml


def test_default_paths_when_nothing_set(isolated_env):
    """Branch 3: no env var, no TOML → repo-relative defaults."""
    config = Config.from_env()
    assert config.data_root == DEFAULT_DATA_ROOT
    assert config.cache_root == DEFAULT_CACHE_ROOT
    assert config.results_root == DEFAULT_RESULTS_ROOT
    assert config.shapefile_path == DEFAULT_DATA_ROOT / DEFAULT_SHAPEFILE_RELATIVE
    assert config.reference_root == DEFAULT_DATA_ROOT / DEFAULT_REFERENCE_RELATIVE
    assert (
        config.single_grid_reference_root
        == DEFAULT_DATA_ROOT / DEFAULT_SINGLE_GRID_REFERENCE_RELATIVE
    )
    assert config.cmip6_metadata_root == DEFAULT_DATA_ROOT / DEFAULT_METADATA_RELATIVE


def test_cache_and_results_are_siblings_of_data_not_inside_it(isolated_env):
    """Restructure 2026-04-30 — cache/ and results/ live at REPO_ROOT, not under Data/."""
    config = Config.from_env()
    assert config.cache_root.parent == config.data_root.parent
    assert config.results_root.parent == config.data_root.parent
    assert "Data" not in config.cache_root.parts
    assert "Data" not in config.results_root.parts


def test_env_var_overrides_data_root_but_not_cache_or_results(isolated_env, monkeypatch, tmp_path):
    """Branch 1: env var moves data_root and the data_root-derived paths;
    cache_root and results_root stay at REPO_ROOT regardless."""
    custom_root = tmp_path / "alt_data"
    custom_root.mkdir()
    monkeypatch.setenv(ENV_VAR_DATA_ROOT, str(custom_root))
    isolated_env.write_text(f'data_root = "{tmp_path / "from_toml"}"\n')

    config = Config.from_env()
    assert config.data_root == custom_root.resolve()
    assert config.shapefile_path == custom_root.resolve() / DEFAULT_SHAPEFILE_RELATIVE
    assert config.reference_root == custom_root.resolve() / DEFAULT_REFERENCE_RELATIVE
    assert (
        config.single_grid_reference_root
        == custom_root.resolve() / DEFAULT_SINGLE_GRID_REFERENCE_RELATIVE
    )
    assert config.cmip6_metadata_root == custom_root.resolve() / DEFAULT_METADATA_RELATIVE
    # cache/results are pinned to REPO_ROOT, unaffected by SUBSELECT_DATA_ROOT
    assert config.cache_root == DEFAULT_CACHE_ROOT
    assert config.results_root == DEFAULT_RESULTS_ROOT


def test_toml_data_root_used_when_no_env_var(isolated_env, tmp_path):
    """Branch 2: TOML data_root wins over default; data_root-derived paths follow."""
    toml_root = tmp_path / "from_toml"
    toml_root.mkdir()
    isolated_env.write_text(f'data_root = "{toml_root}"\n')

    config = Config.from_env()
    assert config.data_root == toml_root.resolve()
    assert config.shapefile_path == toml_root.resolve() / DEFAULT_SHAPEFILE_RELATIVE
    assert config.reference_root == toml_root.resolve() / DEFAULT_REFERENCE_RELATIVE


def test_toml_independent_overrides_for_every_path_field(isolated_env, tmp_path):
    """Branch 2 + the TOML override keys for every path."""
    data_root = tmp_path / "data"
    cache_root = tmp_path / "scratch_cache"
    results_root = tmp_path / "scratch_results"
    shapefile_path = tmp_path / "custom" / "boundaries.gpkg"
    reference_root = tmp_path / "custom_ref"
    single_grid_reference_root = tmp_path / "custom_single_grid_ref"
    cmip6_metadata_root = tmp_path / "custom_meta"
    isolated_env.write_text(
        f'data_root = "{data_root}"\n'
        f'cache_root = "{cache_root}"\n'
        f'results_root = "{results_root}"\n'
        f'shapefile_path = "{shapefile_path}"\n'
        f'reference_root = "{reference_root}"\n'
        f'single_grid_reference_root = "{single_grid_reference_root}"\n'
        f'cmip6_metadata_root = "{cmip6_metadata_root}"\n'
    )

    config = Config.from_env()
    assert config.data_root == data_root.resolve()
    assert config.cache_root == cache_root.resolve()
    assert config.results_root == results_root.resolve()
    assert config.shapefile_path == shapefile_path.resolve()
    assert config.reference_root == reference_root.resolve()
    assert config.single_grid_reference_root == single_grid_reference_root.resolve()
    assert config.cmip6_metadata_root == cmip6_metadata_root.resolve()


def test_toml_with_only_partial_overrides(isolated_env, tmp_path):
    """TOML present, no data_root key, only a cache_root override."""
    cache_root = tmp_path / "cache_only"
    isolated_env.write_text(f'cache_root = "{cache_root}"\n')

    config = Config.from_env()
    assert config.data_root == DEFAULT_DATA_ROOT  # fell through to default
    assert config.cache_root == cache_root.resolve()
    assert config.results_root == DEFAULT_RESULTS_ROOT  # default still applies
    assert config.shapefile_path == DEFAULT_DATA_ROOT / DEFAULT_SHAPEFILE_RELATIVE


def test_env_var_supports_user_expansion(isolated_env, monkeypatch):
    """`~`-prefixed env-var values are expanded."""
    monkeypatch.setenv(ENV_VAR_DATA_ROOT, "~/some_data")
    config = Config.from_env()
    assert config.data_root == (Path.home() / "some_data").resolve()


def test_config_is_frozen(isolated_env):
    config = Config.from_env()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.data_root = Path("/tmp")  # type: ignore[misc]


def test_with_overrides_returns_new_instance_without_mutating(isolated_env, tmp_path):
    base = Config.from_env()
    override_root = tmp_path / "override"
    new = base.with_overrides(data_root=override_root)
    assert new.data_root == override_root
    assert base.data_root == DEFAULT_DATA_ROOT  # original unchanged
    assert new is not base


def test_frozen_methodology_constants_match_paper():
    """HPS uses {tas, pr, psl}; tasmax is diagnostic only. Frozen from paper."""
    config = Config(
        data_root=Path("/tmp/x"),
        cache_root=Path("/tmp/x/cache"),
        shapefile_path=Path("/tmp/x/shp.gpkg"),
        reference_root=Path("/tmp/x/reference"),
        single_grid_reference_root=Path("/tmp/x/reference_single_grid"),
        cmip6_metadata_root=Path("/tmp/x/CMIP6/metadata"),
        results_root=Path("/tmp/x/results"),
    )
    assert config.hps_variables == HPS_VARIABLES == ("tas", "pr", "psl")
    assert config.diagnostic_variables == DIAGNOSTIC_VARIABLES == ("tasmax",)
    assert config.reference_dataset == "W5E5"
    assert config.eval_window == (1995, 2014)
    assert config.future_window == (2081, 2100)
    assert config.pre_industrial == (1850, 1899)
