"""Smoke test — package imports and exposes a version string.

This is the M1 acceptance check. M2+ add real tests.
"""

import subselect


def test_import_and_version():
    assert hasattr(subselect, "__version__")
    assert isinstance(subselect.__version__, str)
    assert subselect.__version__


def test_submodules_importable():
    # Every M1 skeleton module must be importable so M2+ can extend without
    # touching the package layout.
    import subselect.cache  # noqa: F401
    import subselect.config  # noqa: F401
    import subselect.geom  # noqa: F401
    import subselect.independence  # noqa: F401
    import subselect.io  # noqa: F401
    import subselect.optimize  # noqa: F401
    import subselect.performance  # noqa: F401
    import subselect.spread  # noqa: F401
    import subselect.viz  # noqa: F401
    import subselect.viz.country_profile  # noqa: F401
    import subselect.viz.independence_figs  # noqa: F401
    import subselect.viz.performance_figs  # noqa: F401
    import subselect.viz.spread_figs  # noqa: F401
    import subselect.viz.taylor  # noqa: F401
