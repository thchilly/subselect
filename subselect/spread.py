"""Future-response change signals, country-profile artefacts, and spread coverage.

M8 implements `compute_change_signals` (1995–2014 → 2081–2100 deltas, SSP5-8.5
default), `compute_country_timeseries` (annual country-mean 1950–2100 long-form
parquet per docs/refactor.md § Country-profile), and `compute_gwl_crossing_years`
(global artefact, ETCCDI / IPCC AR6 convention). `spread_coverage` is a stub for
Phase 2 — see docs/future_spread.md.
"""
