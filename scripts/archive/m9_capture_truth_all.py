"""Capture truth-bytes PNGs for every in-scope M9 figure by re-executing the
legacy notebook cells against a pre-built namespace.

The namespace mirrors what the legacy notebooks build by the time the
figure-generating cell runs — same variable names, same shapes, same dtypes.
Each cell's ``save_figure(...)`` is monkey-patched to write to
``/tmp/m9_truth/<filename>.png``; ``plt.show()`` is a no-op (Agg backend).

Coverage notes:
- Performance, spread, and GWL country-profile cells run end-to-end (all
  upstream variables are recoverable from paper-era xlsx artefacts).
- Country-profile cells that depend on smoothed time-series derivations done
  upstream in CLIMPACT_figures (cells 31, 32, 35, 44, 46, 49, 50, 54, plus the
  anomalies tables in 29, 42) are skipped here and tagged "deferred — requires
  upstream pipeline port" in the visual-diff report.
"""

from __future__ import annotations

import json
import math as _math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # noqa: F401  — namespace
import matplotlib.lines as mlines  # noqa: F401  — namespace
import matplotlib.ticker as ticker  # noqa: F401  — namespace
import matplotlib.font_manager as fm  # noqa: F401  — namespace
from matplotlib.legend_handler import HandlerPatch  # noqa: F401  — namespace
import numpy as np
import pandas as pd
import xarray as xr  # noqa: F401  — namespace symbol for cells that import xr

# Make the legacy `from TaylorDiagram import TaylorDiagram` resolve to our port
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy" / "cmip6-greece"))

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from subselect.config import Config
from subselect.viz import _data_adapters as adapters

OUT_DIR = Path("/tmp/m9_truth")
OUT_DIR.mkdir(exist_ok=True)


def build_namespace(country: str = "greece") -> dict:
    config = Config.from_env()
    cmip6_models = adapters.load_cmip6_models(config=config)
    model_ids = adapters.load_model_ids(config=config)
    ranked_full = adapters.load_ranked_full(country, config=config)
    variables = ['tas', 'pr', 'psl', 'tasmax']
    perf_metrics = adapters.load_perf_metrics_dict(variables, country, config=config)
    observed_std_dev_df = adapters.load_observed_std_dev_df(country, config=config)
    ordered_models = cmip6_models["model"].tolist()

    long_term_df = adapters.load_long_term_spread(country, config=config).reindex(ordered_models)
    long_pi_change_df = adapters.load_long_term_change_spread(country, config=config).reindex(ordered_models)
    pre_industrial_df = adapters.load_pre_industrial_spread(country, config=config).reindex(ordered_models)

    warming_levels_all_models = adapters.load_warming_levels_all_models(country, config=config)
    warming_level_medians = adapters.load_warming_level_medians(country, config=config)
    global_warming_level_medians = adapters.load_warming_level_medians_global(config=config)

    analysis_path = adapters.country_analysis_path(country, config=config)
    base_path = adapters.country_base_path(config=config) + "/"  # legacy uses trailing slash

    ns = {
        "__builtins__": __builtins__,
        "pd": pd, "np": np, "plt": plt, "os": os, "math": _math, "xr": xr,
        "matplotlib": matplotlib, "mpatches": mpatches, "mlines": mlines,
        "ticker": ticker, "fm": fm, "HandlerPatch": HandlerPatch,
        "country": country, "variables": variables,
        "cmip6_models": cmip6_models, "model_ids": model_ids, "ordered_models": ordered_models,
        "ranked_full": ranked_full,
        "perf_metrics": perf_metrics,
        "observed_std_dev_df": observed_std_dev_df,
        "tas_all_perf_metrics": perf_metrics["tas"],
        "pr_all_perf_metrics": perf_metrics["pr"],
        "psl_all_perf_metrics": perf_metrics["psl"],
        "tasmax_all_perf_metrics": perf_metrics["tasmax"],
        "long_term_df": long_term_df,
        "long_pi_change_df": long_pi_change_df,
        "pre_industrial_df": pre_industrial_df,
        "warming_levels_all_models": warming_levels_all_models,
        "warming_level_medians": warming_level_medians,
        "global_warming_level_medians": global_warming_level_medians,
        "analysis_path": analysis_path,
        "base_path": base_path,
    }
    # per-variable cycle frames
    for var in variables:
        try:
            obs_mm, mod_mm = adapters.load_mon_means(var, country, config=config)
            ns[f"{var}_observed_mon_means"] = obs_mm
            ns[f"{var}_cmip6_mon_means"] = mod_mm
        except FileNotFoundError:
            pass
    # Bring TaylorDiagram into namespace under the legacy import name
    from subselect.viz.taylor import TaylorDiagram
    ns["TaylorDiagram"] = TaylorDiagram

    # save_figure monkey-patch — strips any stale `<var>_` prefix per Q4
    # cleanup so truth-bytes filenames match the M9 module outputs.
    def save_figure(country, png_dir, svg_dir, file_name, dpi=300, **_):
        clean = file_name.lstrip("_")
        path = OUT_DIR / f"{country.lower()}_{clean}.png"
        plt.savefig(path, dpi=dpi, bbox_inches="tight")

    ns["save_figure"] = save_figure
    ns["plots_output_path"] = str(OUT_DIR)
    svg_dump = Path("/tmp/m9_truth_svg")
    svg_dump.mkdir(exist_ok=True)
    ns["svg_plots_output_path"] = str(svg_dump)

    # Cell 34 calls `os.makedirs("H:/CLIMPACT/...", exist_ok=True)` for its
    # Windows-style hardcoded paths; monkey-patch makedirs to no-op for any
    # path starting with "H:/" so we don't pollute the cwd.
    _orig_makedirs = os.makedirs

    def _safe_makedirs(path, *a, **k):
        if isinstance(path, str) and path.startswith("H:/"):
            return None
        return _orig_makedirs(path, *a, **k)

    os.makedirs = _safe_makedirs

    # Cprof cells re-assign `plots_output_path = f'{base_path}analysis/{country}/plots'`
    # inside the cell body (overwriting any pre-set value). Redirect by globally
    # patching `Figure.savefig` to redirect any write that resolves to the
    # legacy `results/<country>/plots(_assess)?` directories into OUT_DIR.
    from matplotlib.figure import Figure as _Figure
    _orig_savefig = _Figure.savefig
    legacy_redirect_dirs = [
        str(Path(config.results_root) / country / "plots"),
        str(Path(config.results_root) / country / "plots_assess"),
        # Cprof cells build the path via `os.path.join(base_path, 'analysis', country, 'plots')`
        # which produces the literal `<results>/analysis/<country>/plots` form
        # (the on-disk symlink `<results>/analysis -> .` makes it resolve to
        # `<results>/<country>/plots`). Match the literal too.
        str(Path(config.results_root) / "analysis" / country / "plots"),
        str(Path(config.results_root) / "analysis" / country / "plots_assess"),
        # Cell 34 hardcodes a Windows-style path `H:/CLIMPACT/Data/analysis/<country>/plots_assess`.
        f"H:/CLIMPACT/Data/analysis/{country.lower()}/plots",
        f"H:/CLIMPACT/Data/analysis/{country.lower()}/plots_assess",
    ]

    def _redirected_savefig(self, fname, *args, **kwargs):
        if isinstance(fname, (str, Path)):
            fname_str = str(fname)
            for legacy_dir in legacy_redirect_dirs:
                if fname_str.startswith(legacy_dir):
                    rel = Path(fname_str).name
                    fname = OUT_DIR / rel
                    break
            else:
                # Drop .svg writes during truth capture (we only diff .png)
                if fname_str.endswith(".svg"):
                    fname = svg_dump / Path(fname_str).name
        return _orig_savefig(self, fname, *args, **kwargs)

    _Figure.savefig = _redirected_savefig
    return ns


# (notebook, cell_index, label, output_files-it-produces). The label is just for
# logging; the actual output filename(s) come from the cell's save_figure call(s).
PERFORMANCE_CELLS = [
    ("legacy/cmip6-greece/GR_model_performance_HM.ipynb", 12, "annual_HM_hist_perf"),
    ("legacy/cmip6-greece/GR_model_performance_HM.ipynb", 13, "HPS_rankings_annual_and_seasons"),
    ("legacy/cmip6-greece/GR_model_performance_HM.ipynb", 25, "tas_seasonal_perf_revised"),
    ("legacy/cmip6-greece/GR_model_performance_HM.ipynb", 26, "pr_seasonal_perf_revised"),
    ("legacy/cmip6-greece/GR_model_performance_HM.ipynb", 27, "psl_seasonal_perf_revised"),
    ("legacy/cmip6-greece/GR_model_performance_HM.ipynb", 28, "tasmax_seasonal_perf_revised"),
    ("legacy/cmip6-greece/GR_model_performance_HM.ipynb", 29, "annual_taylor"),
    ("legacy/cmip6-greece/GR_model_performance_HM.ipynb", 32, "4season_taylor"),
]
SPREAD_CELLS = [
    ("legacy/cmip6-greece/GR_model_spread.ipynb", 13, "annual_annual_spread_rev12"),
    ("legacy/cmip6-greece/GR_model_spread.ipynb", 21, "seasonal_spread_perSeasonBars_right_named_rev1"),
]
COUNTRY_PROFILE_GWL_CELLS = [
    ("legacy/climpact/CLIMPACT_figures.ipynb", 17, "WL_table"),
    ("legacy/climpact/CLIMPACT_figures.ipynb", 58, "gwls_boxplot"),
    ("legacy/climpact/CLIMPACT_figures.ipynb", 61, "gwls_boxplot_times"),
]
# M9.2 — bias-map cell. Cell 34 consumes observed_maps[period][variable] and
# bias_maps[period][variable][model] dicts the build_bias_maps_state adapter
# computes via M7's per-model climatology pipeline. Truth bytes come from
# running cell 34 against the same dicts.
BIAS_MAP_CELL = ("legacy/cmip6-greece/GR_model_performance_HM.ipynb", 34, "bias_maps")
# M9.1 — country-profile state-spec cells. They depend on the climpact state
# (smoothed series, anomalies tables, baselines, etc.) the build_climpact_state
# adapter computes. Truth bytes come from running each cell against the same
# state namespace.
COUNTRY_PROFILE_STATE_CELLS = [
    ("legacy/climpact/CLIMPACT_figures.ipynb", 29, "tas_anomalies_table"),
    ("legacy/climpact/CLIMPACT_figures.ipynb", 31, "tas_change"),
    ("legacy/climpact/CLIMPACT_figures.ipynb", 32, "tas_change_all_shaded"),
    ("legacy/climpact/CLIMPACT_figures.ipynb", 35, "tas_change_spaghetti"),
    ("legacy/climpact/CLIMPACT_figures.ipynb", 42, "pr_percent_anomalies_table"),
    ("legacy/climpact/CLIMPACT_figures.ipynb", 44, "pr_change"),
    ("legacy/climpact/CLIMPACT_figures.ipynb", 46, "pr_change_spaghetti"),
    ("legacy/climpact/CLIMPACT_figures.ipynb", 49, "pr_percent_change_ratio"),
    ("legacy/climpact/CLIMPACT_figures.ipynb", 50, "pr_percent_change_raw"),
    ("legacy/climpact/CLIMPACT_figures.ipynb", 54, "pr_percent_change_spaghetti"),
]


def _patch_cell_34_shapefile(src: str, gadm_gpkg: str) -> str:
    """Replace cell 34's hardcoded `U:/OneDrive/Shapefiles/...` walk with a
    direct read of the GADM 4.1 gpkg, filtered to the country polygon. This
    is the same data-input deviation the M9 module applies (it takes
    `shapefile_path` as a parameter)."""
    import re as _re
    new_block = (
        '    gdf = gpd.read_file(r"' + gadm_gpkg + '")\n'
        '    country_boundaries = gdf[gdf["COUNTRY"].str.lower() == country.lower()]\n'
        '    if country_boundaries.empty:\n'
        '        raise FileNotFoundError(f"Country {country} not found in GADM gpkg")\n'
    )
    pattern = _re.compile(
        r"^    shapefile_path = os\.path\.join\(\"U:/.*?(?=^    minx,)",
        _re.MULTILINE | _re.DOTALL,
    )
    return pattern.sub(new_block, src)


def run_cell(notebook_path: str, cell_idx: int, ns: dict, label: str) -> list[Path]:
    nb = json.loads(Path(notebook_path).read_text())
    src = "".join(nb["cells"][cell_idx].get("source", []))
    # Cell 34 hardcodes a Windows GADM path — redirect to our GADM gpkg.
    if "GR_model_performance_HM" in str(notebook_path) and cell_idx == 34:
        from subselect.config import Config
        cfg = Config.from_env()
        src = _patch_cell_34_shapefile(src, str(cfg.shapefile_path))
    # Track outputs by snapshotting OUT_DIR before/after exec
    before = set(p.name for p in OUT_DIR.glob("*.png"))
    try:
        exec(compile(src, f"<cell_{cell_idx}>", "exec"), ns)
    except Exception as e:
        print(f"  ✗ cell {cell_idx} ({label}): {type(e).__name__}: {e}")
        plt.close("all")
        return []
    plt.close("all")
    after = set(p.name for p in OUT_DIR.glob("*.png"))
    new = sorted(after - before)
    return [OUT_DIR / n for n in new]


def main() -> None:
    ns = build_namespace("greece")
    all_truth: list[Path] = []
    for notebook, idx, label in (PERFORMANCE_CELLS + SPREAD_CELLS):
        produced = run_cell(notebook, idx, ns, label)
        if produced:
            for p in produced:
                print(f"  ✓ truth → {p.name}")
            all_truth.extend(produced)
        else:
            print(f"  ✗ no output for cell {idx} ({label})")

    # For country-profile cells, blank out `variable` so save_figure produces
    # the canonical Q4 filename (no stale variable prefix).
    ns["variable"] = ""
    for notebook, idx, label in COUNTRY_PROFILE_GWL_CELLS:
        produced = run_cell(notebook, idx, ns, label)
        if produced:
            for p in produced:
                print(f"  ✓ truth → {p.name}")
            all_truth.extend(produced)
        else:
            print(f"  ✗ no output for cell {idx} ({label})")

    # M9.1 — country-profile state-spec cells. Build the climpact state and
    # run each figure cell against it.
    state = adapters.build_climpact_state("greece")
    ns_state = dict(ns)
    ns_state.update(state)
    ns_state["variable"] = ""
    for notebook, idx, label in COUNTRY_PROFILE_STATE_CELLS:
        produced = run_cell(notebook, idx, ns_state, label)
        if produced:
            for p in produced:
                print(f"  ✓ truth → {p.name}")
            all_truth.extend(produced)
        else:
            print(f"  ✗ no output for cell {idx} ({label})")

    # M9.2 — bias-map cell 34. Build observed_maps + bias_maps via the
    # M9.2 adapter, drop them into the namespace, run cell 34 against them.
    # Cell 34's loop saves directly via plt.savefig (already redirected by the
    # global Figure.savefig patch); the cell is structurally a function-def
    # plus a `for variable in [...]` loop, so re-using the existing run_cell
    # works.
    bias_state = adapters.build_bias_maps_state("greece", include_seasonal=False)
    ns_bias = dict(ns)
    ns_bias["observed_maps"] = bias_state["observed_maps"]
    ns_bias["bias_maps"] = bias_state["bias_maps"]
    # Inject perf_metrics for the legacy cell's `perf_df=perf_metrics[variable]` lookup
    notebook, idx, label = BIAS_MAP_CELL
    produced = run_cell(notebook, idx, ns_bias, label)
    if produced:
        for p in produced:
            print(f"  ✓ truth → {p.name}")
        all_truth.extend(produced)
    else:
        print(f"  ✗ no output for cell {idx} ({label})")

    print(f"\n{len(all_truth)} truth PNG(s) captured to {OUT_DIR}")


if __name__ == "__main__":
    main()
