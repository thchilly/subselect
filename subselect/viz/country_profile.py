"""Country-profile context figures — change bands, spaghetti, warming-levels.

M9 implements `plot_change_band`, `plot_change_spaghetti`, and `plot_warming_levels`
per docs/refactor.md § Country-profile / context outputs. All three read from the
M8 cache parquets (`<country>/timeseries/<var>__<crop_method>.parquet` and
`_global/gwl_crossing_years.parquet`) and never touch raw data. Visual style
follows `subselection_paper/tas_pr_gwl/`.
"""
