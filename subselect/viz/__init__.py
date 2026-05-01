"""Figure-generation submodule.

Each ``fig_*`` function returns a ``matplotlib.figure.Figure`` and carries a
module-level ``CATEGORY`` constant used by ``scripts/regenerate_paper_figures.py``
to route output to ``results/<country>/figures/<CATEGORY>/<filename>.png``.

Cell ports are verbatim from the legacy ``cmip6-greece/`` and ``climpact/``
notebooks; permitted deviations are listed in ``scripts/m9_cell_map.md``.
"""
