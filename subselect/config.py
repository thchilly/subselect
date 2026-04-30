"""Path resolution and frozen project settings.

M2 fills in the `Config` dataclass: `data_root`, `cache_root`, `shapefile_path`,
the W5E5 reference handle, evaluation/future windows, and the frozen variable
lists. Resolution order: env var `SUBSELECT_DATA_ROOT` → `~/.subselect.toml` →
repo-relative `./Data`. See docs/refactor.md and CLAUDE.md.
"""
