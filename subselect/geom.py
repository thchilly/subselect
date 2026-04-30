"""Country cropping and area weighting.

`crop()` is the single entry point with four methods (per ``docs/refactor.md``
§ Country cropping). All methods run a coarse bbox pre-crop first (matches the
legacy ``extract_subset`` prototype in ``legacy/climpact/shp_extraction.ipynb``):

- ``bbox`` — paper-era setting and regression-test pin.
- ``shapefile_strict`` — pixel-centre-inside polygon (binary, NaN outside).
- ``shapefile_lenient`` — any-touch (binary, NaN outside). **Framework default.**
- ``shapefile_fractional`` — area-fraction-inside as a weight; data un-masked,
  weight returned alongside. Opt-in for boundary-precision use cases.

`apply_weights(da, weight=None)` composes ``cos(lat)`` area weighting with the
optional fractional weight so downstream metric code calls one helper for all
four methods.

Methodology decision pending M5: the framework default is committed to
``shapefile_lenient`` per ``docs/refactor.md`` line 161, but Athanasios reviews
it visually over Greece + a few contrasting countries before the methods.tex
entry lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple

import geopandas as gpd
import numpy as np
import regionmask
import rioxarray  # noqa: F401  (registers the .rio xarray accessor)
import xarray as xr
from rasterio.features import geometry_mask
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

    A coarse bbox pre-crop runs first regardless of method (~100× shrink for
    small countries), so shapefile masking only operates on a small array.
    """
    if method not in CROP_METHODS:
        raise ValueError(f"Unknown crop method {method!r}; expected one of {CROP_METHODS}")
    config = config or Config.from_env()
    bbox = io.country_bbox(country, config=config)
    da_bbox = _apply_bbox(da, bbox, box_offset)

    if method == "bbox":
        return CropResult(
            data=da_bbox,
            weight=None,
            metadata={"crop_method": "bbox", "country": country, "bbox": bbox},
        )

    polygon = _load_country_polygon(country, shapefile_path, config)

    if method in ("shapefile_strict", "shapefile_lenient"):
        all_touched = method == "shapefile_lenient"
        mask = _binary_mask(da_bbox, polygon, all_touched=all_touched)
        return CropResult(
            data=da_bbox.where(mask),
            weight=None,
            metadata={
                "crop_method": method,
                "country": country,
                "bbox": bbox,
                "all_touched": all_touched,
            },
        )

    # shapefile_fractional
    weight = _fractional_mask(da_bbox, polygon, country)
    return CropResult(
        data=da_bbox,
        weight=weight,
        metadata={"crop_method": "shapefile_fractional", "country": country, "bbox": bbox},
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
    Prime-Meridian crossing per the legacy `extract_subset` algorithm."""
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


def _fractional_mask(
    da: xr.DataArray, polygon: gpd.GeoDataFrame, country: str
) -> xr.DataArray:
    """Per-pixel area-fraction-inside-country in [0, 1] via regionmask."""
    da = _ensure_rio_crs(da)
    # `names`/`abbrevs` are column-name lookups in regionmask.from_geopandas;
    # use the GADM COUNTRY column. country is a guard for the 1-row contract.
    polygon = polygon.reset_index(drop=True)
    if len(polygon) != 1 or polygon[COUNTRY_COLUMN].iloc[0].lower() != country.lower():
        raise ValueError(
            f"Expected single-row polygon for {country!r}, got {len(polygon)} rows"
        )
    regions = regionmask.from_geopandas(
        polygon, names=COUNTRY_COLUMN, abbrevs=COUNTRY_COLUMN
    )
    frac = regions.mask_3D_frac_approx(da)  # dims: (region, lat, lon)
    frac_2d = frac.isel(region=0)
    return frac_2d.drop_vars(["region", "abbrevs", "names"], errors="ignore")
