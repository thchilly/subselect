"""Cache layer — parquet + zarr + sqlite catalog.

M6 implements the writers/readers and the `Catalog` over `Data/cache/catalog.sqlite`
per docs/refactor.md § Caching strategy. Path conventions cover three scopes:
per-country per-(scenario, season), per-country time-series, and `_global`
(country column nullable / sentinel) for ensemble-wide artefacts like
`_global/gwl_crossing_years.parquet`.
"""
