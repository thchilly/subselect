"""Unit tests for `subselect.config.Config` and the path-resolution chain.

Covers the three resolution branches (env var → ~/.subselect.toml → repo
default), TOML overrides for cache and shapefile paths, the frozen-dataclass
contract, and the with_overrides helper.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from subselect import config as cfg
from subselect.config import (
    DEFAULT_CACHE_RELATIVE,
    DEFAULT_DATA_ROOT,
    DEFAULT_SHAPEFILE_RELATIVE,
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


def test_default_data_root_when_nothing_set(isolated_env):
    """Branch 3: no env var, no TOML → repo-relative `<repo>/Data`."""
    config = Config.from_env()
    assert config.data_root == DEFAULT_DATA_ROOT
    assert config.cache_root == DEFAULT_DATA_ROOT / DEFAULT_CACHE_RELATIVE
    assert config.shapefile_path == DEFAULT_DATA_ROOT / DEFAULT_SHAPEFILE_RELATIVE


def test_env_var_overrides_default(isolated_env, monkeypatch, tmp_path):
    """Branch 1: env var wins over both TOML and default."""
    custom_root = tmp_path / "alt_data"
    custom_root.mkdir()
    monkeypatch.setenv(ENV_VAR_DATA_ROOT, str(custom_root))
    # Also drop a TOML to prove env var has priority.
    isolated_env.write_text(f'data_root = "{tmp_path / "from_toml"}"\n')

    config = Config.from_env()
    assert config.data_root == custom_root.resolve()
    assert config.cache_root == custom_root.resolve() / DEFAULT_CACHE_RELATIVE


def test_toml_data_root_used_when_no_env_var(isolated_env, tmp_path):
    """Branch 2: TOML wins over default when env var is unset."""
    toml_root = tmp_path / "from_toml"
    toml_root.mkdir()
    isolated_env.write_text(f'data_root = "{toml_root}"\n')

    config = Config.from_env()
    assert config.data_root == toml_root.resolve()
    assert config.shapefile_path == toml_root.resolve() / DEFAULT_SHAPEFILE_RELATIVE


def test_toml_independent_overrides_for_cache_and_shapefile(isolated_env, tmp_path):
    """Branch 2 + the optional cache_root / shapefile_path TOML overrides."""
    data_root = tmp_path / "data"
    cache_root = tmp_path / "scratch_cache"
    shapefile_path = tmp_path / "custom" / "boundaries.gpkg"
    isolated_env.write_text(
        f'data_root = "{data_root}"\n'
        f'cache_root = "{cache_root}"\n'
        f'shapefile_path = "{shapefile_path}"\n'
    )

    config = Config.from_env()
    assert config.data_root == data_root.resolve()
    assert config.cache_root == cache_root.resolve()
    assert config.shapefile_path == shapefile_path.resolve()


def test_toml_with_only_partial_overrides(isolated_env, tmp_path):
    """TOML present, no data_root key, only a cache_root override."""
    cache_root = tmp_path / "cache_only"
    isolated_env.write_text(f'cache_root = "{cache_root}"\n')

    config = Config.from_env()
    assert config.data_root == DEFAULT_DATA_ROOT  # fell through to default
    assert config.cache_root == cache_root.resolve()
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
    )
    assert config.hps_variables == HPS_VARIABLES == ("tas", "pr", "psl")
    assert config.diagnostic_variables == DIAGNOSTIC_VARIABLES == ("tasmax",)
    assert config.reference_dataset == "W5E5"
    assert config.eval_window == (1995, 2014)
    assert config.future_window == (2081, 2100)
    assert config.pre_industrial == (1850, 1900)
