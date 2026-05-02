"""Figure-generation submodule (L2 of the package architecture).

Each ``fig_*`` function returns a :class:`matplotlib.figure.Figure` and
carries a module-level ``CATEGORY`` constant used by
:func:`subselect.render.render` to route output to
``results/<country>/figures/<CATEGORY>/<filename>.png``.
"""
