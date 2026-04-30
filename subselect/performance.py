"""Historical Performance Score (HPS) pipeline.

M7 implements `compute_hps(country, scenario, season, config, crop_method=...)`
plus the pure-function metrics (`tss`, `bvs`, `harmonic_mean`,
`minmax_normalize`). HPS recipe is frozen from the published Greece paper —
see docs/historical_performance.md. The regression test pins to crop_method='bbox'
to match the paper.
"""
