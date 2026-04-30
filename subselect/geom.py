"""Country cropping and area weighting.

`crop()` is the single entry point with four methods (per ``docs/refactor.md``
§ Country cropping):

- ``bbox`` — paper-era setting and regression-test pin.
- ``shapefile_strict`` — pixel-centre-inside polygon (binary, NaN outside).
- ``shapefile_lenient`` — any-touch (binary, NaN outside). **Framework default.**
- ``shapefile_fractional`` — area-fraction-inside as a weight; data un-masked,
  weight returned alongside. Opt-in for boundary-precision use cases. Computed
  via per-cell shapely intersection so it works on regular and Gaussian
  grids alike.

`apply_weights(da, weight=None)` composes ``cos(lat)`` area weighting with the
optional fractional weight so downstream metric code calls one helper for all
four methods.

Internal order (post-M5 fix, see docs/refactor.md § Country cropping for the
finding that motivated the change). For shapefile methods the mask is built
against the **full** input grid first — uniform spacing is then guaranteed for
``rasterio.features.geometry_mask`` (binary methods) and the per-cell shapely
intersection works on Gaussian grids too (fractional method). The bbox
pre-crop is applied to data and mask together afterwards, so it is a
post-mask optimisation rather than a precondition. This matches the legacy
prototype's *result* (the prototype bbox-cropped first, but only ever ran on
Greece where the bbox-cropped grid had ≥2 cells per axis); it removes the
degenerate-grid edge case the prototype's ordering had silently introduced.

Methodology decision pending M5: the framework default is committed to
``shapefile_lenient`` per ``docs/refactor.md`` line 161, but Athanasios reviews
it visually over Greece + a few contrasting countries before the methods.tex
entry lands.

Known coarse-grid limitation (Phase 1+ work, not fixed here). Even after the
M5 fix, a small country on a coarse grid (e.g. Cyprus on CanESM5 ≈ 2.8°) yields
≈1 included cell — statistically meaningless for HPS / spread metrics. Future
work is to emit a warning when the cropped grid has fewer than N cells
(proposal: N=4 for binary methods, N=2 for fractional weighted means); see
``docs/refactor.md`` § Country cropping → Known limitations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio xarray accessor)
import xarray as xr
from rasterio.features import geometry_mask
from shapely.geometry import box as shapely_box
from shapely.geometry import mapping

from subselect import io
from subselect.config import Config

CropMethod = Literal[
    "bbox", "shapefile_strict", "shapefile_lenient", "shapefile_fractional"
]
CROP_METHODS: tuple[CropMethod, ...] = (
    "bbox",
    "shapefile_strict",
    "shapefile_lenient",
    "shapefile_fractional",
)
COUNTRY_COLUMN = "COUNTRY"  # GADM 4.1 country-name column


class ShapefileNotConfigured(RuntimeError):
    """Raised when method=='shapefile_*' but no shapefile is available."""


class CropResult(NamedTuple):
    """Bundle returned by :func:`crop`.

    - ``data`` is the cropped DataArray (NaN outside country for binary
      methods; un-masked for ``bbox`` and ``shapefile_fractional``).
    - ``weight`` is None for ``bbox`` and the binary methods; a 2-D
      ``[0, 1]`` fractional-overlap array for ``shapefile_fractional``.
      Always pass it to :func:`apply_weights` together with the data so the
      downstream metric pipeline composes ``cos(lat) × weight`` correctly.
    - ``metadata`` includes ``crop_method``, the resolved bbox, and (for
      shapefile methods) the country polygon. The ``crop_method`` value
      lands in the SQLite cache catalog so different cropping choices do
      not collide (M6).
    """

    data: xr.DataArray
    weight: xr.DataArray | None
    metadata: dict


def crop(
    da: xr.DataArray,
    country: str,
    *,
    method: CropMethod = "shapefile_lenient",
    shapefile_path: Path | str | None = None,
    box_offset: float = 1.0,
    config: Config | None = None,
) -> CropResult:
    """Crop a DataArray to a country using one of the four supported methods.

    - ``bbox`` uses the country bounding box from ``country_codes.json`` plus
      ``box_offset`` degrees on each side (paper-era behaviour preserved for
      regression-test reproducibility).
    - The shapefile methods build the mask against the *full* input grid and
      then crop the data + mask to a **mask-aware** bbox: the smallest lat/lon
      box enclosing all selected cells, expanded by one cell on each side for
      visual context. The mask-aware crop cannot exclude any cell the mask
      included — fixes a degenerate-grid edge case discovered in M5 review
      where a small country on a coarse grid + the legacy ``country_bbox +
      box_offset=1°`` chopped off cells the lenient mask had correctly
      included (because 1° < ½ cell at ~2.8° native resolution).

    The ``box_offset`` parameter only affects the ``bbox`` method; shapefile
    methods always pad by one cell on each side.

    See the module docstring for the mask-first / bbox-after ordering and the
    Phase 1+ Gaussian-grid limitation on ``shapefile_fractional``.
    """
    if method not in CROP_METHODS:
        raise ValueError(f"Unknown crop method {method!r}; expected one of {CROP_METHODS}")
    config = config or Config.from_env()
    country_box = io.country_bbox(country, config=config)

    if method == "bbox":
        return CropResult(
            data=_apply_bbox(da, country_box, box_offset),
            weight=None,
            metadata={"crop_method": "bbox", "country": country, "bbox": country_box},
        )

    polygon = _load_country_polygon(country, shapefile_path, config)

    if method in ("shapefile_strict", "shapefile_lenient"):
        all_touched = method == "shapefile_lenient"
        mask_full = _binary_mask(da, polygon, all_touched=all_touched)
        crop_box = _maskaware_bbox(da, mask_full, fallback=country_box)
        da_cropped = _slice_to_box(da, crop_box)
        mask_cropped = _slice_to_box(mask_full, crop_box)
        return CropResult(
            data=da_cropped.where(mask_cropped),
            weight=None,
            metadata={
                "crop_method": method,
                "country": country,
                "bbox": crop_box,
                "all_touched": all_touched,
            },
        )

    # shapefile_fractional
    weight_full = _fractional_mask(da, polygon, country)
    crop_box = _maskaware_bbox(da, weight_full > 0, fallback=country_box)
    da_cropped = _slice_to_box(da, crop_box)
    weight_cropped = _slice_to_box(weight_full, crop_box)
    return CropResult(
        data=da_cropped,
        weight=weight_cropped,
        metadata={
            "crop_method": "shapefile_fractional",
            "country": country,
            "bbox": crop_box,
        },
    )


def apply_weights(
    da: xr.DataArray, weight: xr.DataArray | None = None
) -> xr.DataArray:
    """Compose ``cos(lat)`` area weighting with an optional fractional weight.

    Returns a DataArray of weights aligned to ``da``'s lat/lon dims. The caller
    composes a weighted spatial mean as ``(da * w).sum(dim) / w.sum(dim)``.
    """
    cos_lat = np.cos(np.deg2rad(da["lat"]))
    cos_lat = cos_lat.where(da["lat"].notnull(), 0.0)
    weights = cos_lat.broadcast_like(da)
    if weight is not None:
        weights = weights * weight
    return weights


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _apply_bbox(da: xr.DataArray, bbox: dict[str, float], box_offset: float) -> xr.DataArray:
    """Slice ``da`` to ``bbox`` ± offset, handling 0–360 vs −180/180 lon and
    Prime-Meridian crossing per the legacy `extract_subset` algorithm.

    Used for the ``bbox`` crop method only; shapefile methods route through
    :func:`_maskaware_bbox` and :func:`_slice_to_box` instead.
    """
    lat_min = max(bbox["lat_min"] - box_offset, -90.0)
    lat_max = min(bbox["lat_max"] + box_offset, 90.0)
    lon_min = bbox["lon_min"] - box_offset
    lon_max = bbox["lon_max"] + box_offset

    if float(da["lon"].min()) >= 0:
        if lon_min < 0:
            lon_min += 360
        if lon_max < 0:
            lon_max += 360

    if lon_min > lon_max:
        part1 = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, 360))
        part2 = da.sel(lat=slice(lat_min, lat_max), lon=slice(0, lon_max))
        return xr.concat([part1, part2], dim="lon")
    return da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))


def _maskaware_bbox(
    da: xr.DataArray, selected: xr.DataArray, *, fallback: dict[str, float]
) -> dict[str, float]:
    """Smallest lat/lon bbox enclosing every True cell in ``selected``.

    The bbox is expanded by one full cell on each side for visual context.
    By construction it cannot exclude any cell the mask included — fixes the
    M5-discovered failure where ``country_bbox + box_offset=1°`` was less
    than half a coarse-grid cell and chopped off cells the lenient mask had
    correctly included.

    If ``selected`` is empty (no cell overlaps the polygon), falls back to
    the country bbox so callers still get a coherent (all-NaN) slice rather
    than an empty array. Raises ``NotImplementedError`` if the selected
    region spans more than 180° lon, which suggests a Prime-Meridian-
    crossing country on a 0–360 lon grid (not yet supported).
    """
    if not bool(selected.any()):
        return fallback
    has_lat_strip = selected.any("lon")
    has_lon_strip = selected.any("lat")
    sel_lat = da["lat"].where(has_lat_strip, drop=True)
    sel_lon = da["lon"].where(has_lon_strip, drop=True)
    dlat = abs(float(np.diff(da["lat"].values).mean()))
    dlon = abs(float(np.diff(da["lon"].values).mean()))
    lat_min = float(sel_lat.min()) - dlat
    lat_max = float(sel_lat.max()) + dlat
    lon_min = float(sel_lon.min()) - dlon
    lon_max = float(sel_lon.max()) + dlon
    if lon_max - lon_min > 180:
        raise NotImplementedError(
            "Selected region spans more than 180° lon — likely a Prime-Meridian-"
            "crossing country on a 0–360 lon grid. Not yet supported by "
            "_maskaware_bbox; pass the data on a centred (-180, 180) grid or "
            "split the polygon manually."
        )
    return {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }


def _slice_to_box(da: xr.DataArray, box: dict[str, float]) -> xr.DataArray:
    """Slice ``da`` to a lat/lon box, robust to ascending or descending lat."""
    lat_values = da["lat"].values
    if len(lat_values) >= 2 and lat_values[0] > lat_values[-1]:
        lat_slice = slice(box["lat_max"], box["lat_min"])
    else:
        lat_slice = slice(box["lat_min"], box["lat_max"])
    return da.sel(lat=lat_slice, lon=slice(box["lon_min"], box["lon_max"]))


def _load_country_polygon(
    country: str, shapefile_path: Path | str | None, config: Config
) -> gpd.GeoDataFrame:
    path = Path(shapefile_path) if shapefile_path is not None else config.shapefile_path
    if not path.is_file():
        raise ShapefileNotConfigured(
            f"Shapefile not found at {path}. Place GADM 4.1 "
            f"(gadm_410-levels.gpkg) at config.shapefile_path or pass "
            f"shapefile_path= explicitly."
        )
    # GADM 4.1 ships six admin layers (ADM_0..ADM_5); ADM_0 is country-level
    # and is the first layer. layer=0 picks it without raising the
    # "More than one layer found" warning. Single-layer files (e.g. user-
    # supplied custom shapefiles, the synthetic test gpkg) also accept index 0.
    gdf = gpd.read_file(path, layer=0)
    if COUNTRY_COLUMN not in gdf.columns:
        raise ValueError(
            f"Shapefile {path} has no {COUNTRY_COLUMN!r} column; "
            f"present columns: {list(gdf.columns)}"
        )
    matched = gdf[gdf[COUNTRY_COLUMN].str.lower() == country.lower()]
    if matched.empty:
        raise ValueError(f"Country {country!r} not found in shapefile {path}")
    if matched.crs is None:
        matched = matched.set_crs("EPSG:4326")
    elif matched.crs.to_epsg() != 4326:
        matched = matched.to_crs("EPSG:4326")
    return matched


def _ensure_rio_crs(da: xr.DataArray) -> xr.DataArray:
    """Make sure rioxarray can read spatial dims and CRS off ``da``.

    Synthetic test arrays (and some CMIP6 NetCDFs that lack CF attrs) carry
    plain ``lat`` / ``lon`` dims without rio metadata. Explicitly bind them
    as the X / Y spatial dims and stamp EPSG:4326 if no CRS is set.
    """
    if "lat" in da.dims and "lon" in da.dims:
        da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326")
    return da


def _binary_mask(
    da: xr.DataArray, polygon: gpd.GeoDataFrame, *, all_touched: bool
) -> xr.DataArray:
    """Boolean True/False mask aligned to ``da``'s grid via ``rasterio``."""
    da = _ensure_rio_crs(da)
    transform = da.rio.transform()
    out_shape = (da.rio.height, da.rio.width)
    shapes = [mapping(geom) for geom in polygon.geometry]
    mask = geometry_mask(
        shapes,
        transform=transform,
        invert=True,
        out_shape=out_shape,
        all_touched=all_touched,
    )
    return xr.DataArray(mask, coords={"lat": da["lat"], "lon": da["lon"]}, dims=["lat", "lon"])


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Convert 1-D pixel centres on any grid to N+1 cell edges.

    Interior edges are midpoints between adjacent centres — handles Gaussian
    non-uniform spacing correctly. The first and last edges extrapolate using
    the *local* spacing of the adjacent cell pair (not the mean), so a
    Gaussian grid's pole-tightening is preserved at the boundaries.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        raise ValueError("centers must have at least 2 elements to derive edges")
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2.0
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2.0
    return edges


def _fractional_mask(
    da: xr.DataArray, polygon: gpd.GeoDataFrame, country: str
) -> xr.DataArray:
    """Per-pixel area-fraction-inside-country in [0, 1] via shapely intersection.

    Computes each cell's polygon-overlap fraction directly: cell rectangle
    (from edge midpoints between centroids) ∩ country polygon, divided by
    cell area. Works on regular and Gaussian grids alike — the latter is the
    norm for CMIP6 atmosphere models, where ``regionmask.mask_3D_frac_approx``
    raises because Gaussian latitudes are not equally spaced.

    Performance: bounded by a polygon-bounding-box early-out so the inner
    shapely intersection only runs for cells that could plausibly overlap
    the country. Sub-second per country at CMIP6 ensemble scale even on the
    finer T127/T255 grids.
    """
    polygon = polygon.reset_index(drop=True)
    if len(polygon) != 1 or polygon[COUNTRY_COLUMN].iloc[0].lower() != country.lower():
        raise ValueError(
            f"Expected single-row polygon for {country!r}, got {len(polygon)} rows"
        )
    poly_geom = polygon.geometry.iloc[0]
    poly_minx, poly_miny, poly_maxx, poly_maxy = poly_geom.bounds

    lat_centers = np.asarray(da["lat"].values, dtype=float)
    lon_centers = np.asarray(da["lon"].values, dtype=float)
    lat_edges = _centers_to_edges(lat_centers)
    lon_edges = _centers_to_edges(lon_centers)

    # Per-cell lat / lon bounds — handle ascending or descending coord axes.
    lat_lo = np.minimum(lat_edges[:-1], lat_edges[1:])
    lat_hi = np.maximum(lat_edges[:-1], lat_edges[1:])
    lon_lo = np.minimum(lon_edges[:-1], lon_edges[1:])
    lon_hi = np.maximum(lon_edges[:-1], lon_edges[1:])

    # Pre-filter rows / columns that cannot intersect the polygon's bbox.
    lat_candidates = np.where((lat_hi >= poly_miny) & (lat_lo <= poly_maxy))[0]
    lon_candidates = np.where((lon_hi >= poly_minx) & (lon_lo <= poly_maxx))[0]

    weight = np.zeros((lat_centers.size, lon_centers.size), dtype=float)
    for i in lat_candidates:
        for j in lon_candidates:
            cell = shapely_box(lon_lo[j], lat_lo[i], lon_hi[j], lat_hi[i])
            inter = cell.intersection(poly_geom)
            if inter.is_empty:
                continue
            weight[i, j] = inter.area / cell.area

    return xr.DataArray(
        weight,
        coords={"lat": da["lat"], "lon": da["lon"]},
        dims=["lat", "lon"],
        name="fractional_weight",
    )
