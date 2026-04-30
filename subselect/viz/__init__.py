"""Figure-generation submodule. Ports paper-era figure code as-is in M9.

Every helper in this package gains a `backend: Literal["matplotlib", "plotly"]`
kwarg per docs/refactor.md line 259 so the same code paths feed the static
paper/poster figures and the Phase 4 web app.
"""
