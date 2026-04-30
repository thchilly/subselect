"""Country cropping and area weighting.

M4 implements `crop(da, country, method=...)` with four rules — `bbox`,
`shapefile_strict`, `shapefile_lenient` (framework default per docs/refactor.md
§ Country cropping), `shapefile_fractional` (opt-in) — and `apply_weights` that
composes cos(lat) area weighting with the optional fractional weight. M5 is the
visual validation gate before the methodology log entry lands.
"""
