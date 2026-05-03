"""Diagnostic: visualise the four crop methods over a country.

Produces, per (country, model) pair, a 2×2 figure with one panel per crop
method (``bbox`` / ``shapefile_strict`` / ``shapefile_lenient`` /
``shapefile_fractional``). Pixels are coloured grey (excluded) →
orange (included); for the fractional panel, opacity encodes the
per-pixel area-fraction-inside-country.

Useful as a visual companion to ``tests/test_geom.py``: re-run after any
change to :mod:`subselect.geom` and check that the per-method pixel counts
look right for representative geometries (archipelagic, small-island,
mountainous-landlocked) on coarse and fine native grids.

Usage::

    python scripts/cropping_review.py

Outputs land in ``notebooks/exploratory/m5_cropping_review/`` as
``m5_cropping_<country>_<model>.{png,svg}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

from subselect import geom, io
from subselect.config import Config
from subselect.geom import CROP_METHODS, COUNTRY_COLUMN, _centers_to_edges

# Coarse model surfaces the strict zero-pixel risk on small countries;
# the finer model is a contrast point.
DEFAULT_MODELS = ("CanESM5", "MPI-ESM1-2-HR")
DEFAULT_COUNTRIES = ("Greece", "Cyprus", "Switzerland")
DEFAULT_VARIABLE = "tas"
DEFAULT_SCENARIO = "ssp585"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "exploratory" / "m5_cropping_review"

CMAP = LinearSegmentedColormap.from_list(
    "subselect_grey_orange", ["#d6d6d6", "#ff8c00"]
)
# Flat light-grey colormap for the full-grid context cells drawn behind the
# cropped result. Same hex on both endpoints so the colour is fixed regardless
# of the input array values; pcolormesh still draws cell edges for countability.
CONTEXT_FILL = "#f5f5f5"
CONTEXT_CMAP = LinearSegmentedColormap.from_list(
    "subselect_context_grey", [CONTEXT_FILL, CONTEXT_FILL]
)
GRID_COLOR = "#5a5a5a"
GRID_LW = 0.25
POLYGON_COLOR = "#1a1a1a"
POLYGON_LW = 1.4
# How many full-grid cells to show around the union of all four cropped extents.
CONTEXT_PADDING_CELLS = 2


def _edges_with_fallback(centers: np.ndarray, fallback_step: float) -> np.ndarray:
    """Render-side wrapper around `geom._centers_to_edges` for the 1-cell case.

    `geom._centers_to_edges` requires ≥2 centres so it can derive cell widths
    from the data itself — correct for the cropping logic. The render script
    sometimes plots a 1-cell cropped result (small country on a coarse grid)
    and needs a half-cell pad on each side; pass the full-grid step as the
    fallback in that case.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.size >= 2:
        return _centers_to_edges(centers)
    if centers.size == 1:
        return np.array([centers[0] - fallback_step / 2, centers[0] + fallback_step / 2])
    raise ValueError("centers must have at least 1 element")


def _load_country_polygon(country: str, config: Config) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(config.shapefile_path, layer=0)
    matched = gdf[gdf[COUNTRY_COLUMN].str.lower() == country.lower()]
    if matched.empty:
        raise ValueError(f"Country {country!r} not in GADM at {config.shapefile_path}")
    return matched.to_crs("EPSG:4326")


def _load_first_time_slice(model: str, config: Config) -> xr.DataArray:
    ds = io.load_cmip6(DEFAULT_VARIABLE, DEFAULT_SCENARIO, model, config=config)
    var = ds[DEFAULT_VARIABLE]
    if "time" in var.dims:
        var = var.isel(time=0)
    return var


def _panel_payload(
    method: str,
    result: geom.CropResult,
    *,
    lat_step: float,
    lon_step: float,
) -> tuple[xr.DataArray, str]:
    """Return the 2-D array to colour and the per-panel subtitle suffix.

    ``lat_step`` and ``lon_step`` come from the *full* grid so the deg²
    conversion in the fractional subtitle stays well-defined when the cropped
    slice has fewer than 2 cells on an axis.
    """
    if method == "bbox":
        included = xr.ones_like(result.data)
        count = int(included.size)
        return included, f"all {count} cells included"

    if method in ("shapefile_strict", "shapefile_lenient"):
        included = (~result.data.isnull()).astype(float)
        count = int(included.sum())
        total = int(included.size)
        return included, f"{count}/{total} cells included"

    # shapefile_fractional. weight_sum has units of "pixel-equivalents"
    # (Σ of per-pixel area-fractions); convert to deg² with the cell area, and
    # also report the bbox-coverage ratio for grid-agnostic interpretability.
    weight = result.weight
    weight_sum = float(weight.sum())
    nz = int((weight > 0).sum())
    total = int(weight.size)
    country_deg2 = weight_sum * lat_step * lon_step
    bbox_pct = 100.0 * weight_sum / total
    return weight, (
        f"Σweight = {weight_sum:.2f} cells "
        f"(≈{country_deg2:.1f} deg² country area;  {bbox_pct:.0f}% of bbox);  "
        f"{nz}/{total} cells with weight > 0"
    )


def render_one(country: str, model: str, *, config: Config, output_dir: Path) -> Path:
    da = _load_first_time_slice(model, config)
    polygon = _load_country_polygon(country, config)

    results: dict[str, geom.CropResult | Exception] = {}
    for method in CROP_METHODS:
        try:
            results[method] = geom.crop(
                da, country, method=method, config=config, box_offset=1.0
            )
        except Exception as exc:  # noqa: BLE001 — visualise any failure mode
            results[method] = exc

    # bbox is the safe reference; if even bbox fails, propagate.
    bbox_result = results["bbox"]
    if isinstance(bbox_result, Exception):
        raise bbox_result
    # Take grid spacing from the source data (always multi-cell), not the
    # cropped slice — small countries on coarse grids can produce a 1-cell
    # cropped result where np.diff(...).mean() is NaN.
    lat_step = abs(float(np.diff(da["lat"]).mean()))
    lon_step = abs(float(np.diff(da["lon"]).mean()))

    # Plot extent: union of all successful cropped extents, padded by
    # CONTEXT_PADDING_CELLS full-grid cells on each side. All four panels share
    # this extent so the visual comparison across methods is meaningful, and
    # single-cell countries still get visible neighbouring grey context cells.
    extents = [
        (
            float(r.data["lat"].min()),
            float(r.data["lat"].max()),
            float(r.data["lon"].min()),
            float(r.data["lon"].max()),
        )
        for r in results.values()
        if not isinstance(r, Exception)
    ]
    union_lat_min = min(e[0] for e in extents)
    union_lat_max = max(e[1] for e in extents)
    union_lon_min = min(e[2] for e in extents)
    union_lon_max = max(e[3] for e in extents)
    pad_lat = CONTEXT_PADDING_CELLS * lat_step
    pad_lon = CONTEXT_PADDING_CELLS * lon_step
    extent = (
        union_lon_min - pad_lon,
        union_lon_max + pad_lon,
        union_lat_min - pad_lat,
        union_lat_max + pad_lat,
    )

    # Slice the full grid to the plot extent for the context backdrop.
    lat_values = da["lat"].values
    if len(lat_values) >= 2 and lat_values[0] > lat_values[-1]:
        context_lat_slice = slice(extent[3], extent[2])
    else:
        context_lat_slice = slice(extent[2], extent[3])
    context = da.sel(lat=context_lat_slice, lon=slice(extent[0], extent[1]))
    context_lat_edges = _edges_with_fallback(
        np.asarray(context["lat"]), fallback_step=lat_step
    )
    context_lon_edges = _edges_with_fallback(
        np.asarray(context["lon"]), fallback_step=lon_step
    )
    context_array = np.zeros((context["lat"].size, context["lon"].size))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle(
        f"M5 cropping comparison — {country} on {model} ({DEFAULT_VARIABLE}, "
        f"{DEFAULT_SCENARIO}, native ~{abs(lat_step):.2f}° × {abs(lon_step):.2f}°)",
        fontsize=13,
        y=1.02,
    )

    for ax, method in zip(axes.flat, CROP_METHODS):
        # Context backdrop: every full-grid cell within the plot extent in
        # light grey, so single-cell countries still show their neighbours.
        ax.pcolormesh(
            context_lon_edges,
            context_lat_edges,
            context_array,
            cmap=CONTEXT_CMAP,
            vmin=0.0,
            vmax=1.0,
            edgecolors=GRID_COLOR,
            linewidth=GRID_LW,
            shading="flat",
        )

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_xlabel("lon (°E)")
        ax.set_ylabel("lat (°N)")

        result = results[method]
        if isinstance(result, Exception):
            polygon.boundary.plot(ax=ax, color=POLYGON_COLOR, linewidth=POLYGON_LW)
            ax.text(
                0.5, 0.5,
                f"{type(result).__name__}\n{result}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#9a3412",
                bbox={"boxstyle": "round", "fc": "#fff7ed", "ec": "#9a3412"},
                wrap=True,
            )
            ax.set_title(f"{method}\nUNAVAILABLE on this grid", fontsize=11)
            continue

        cells, subtitle = _panel_payload(
            method, result, lat_step=lat_step, lon_step=lon_step
        )
        lat_edges = _edges_with_fallback(np.asarray(cells["lat"]), fallback_step=lat_step)
        lon_edges = _edges_with_fallback(np.asarray(cells["lon"]), fallback_step=lon_step)

        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            cells.values,
            cmap=CMAP,
            vmin=0.0,
            vmax=1.0,
            edgecolors=GRID_COLOR,
            linewidth=GRID_LW,
            shading="flat",
        )

        polygon.boundary.plot(ax=ax, color=POLYGON_COLOR, linewidth=POLYGON_LW)
        ax.set_title(f"{method}\n{subtitle}", fontsize=11)

        if method == "shapefile_fractional":
            cb = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label("fractional weight ∈ [0, 1]", fontsize=9)

    out_stem = output_dir / f"m5_cropping_{country}_{model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.png", dpi=160, bbox_inches="tight")
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return out_stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--countries", nargs="+", default=list(DEFAULT_COUNTRIES),
        help="Country names matching the GADM COUNTRY column.",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(DEFAULT_MODELS),
        help="CMIP6 model IDs as they appear in Data/CMIP6/monthly/.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Where the m5_cropping_<country>_<model>.{png,svg} pairs land.",
    )
    args = parser.parse_args(argv)

    config = Config.from_env()
    written: list[Path] = []
    for country in args.countries:
        for model in args.models:
            stem = render_one(
                country, model, config=config, output_dir=args.output_dir
            )
            written.append(stem)
            print(f"wrote {stem}.png + .svg")

    print(f"\nWrote {len(written)} figure pairs under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
