"""Capture truth bytes for an M9 pilot figure by re-executing a legacy cell.

Loads `ranked_full` and `model_ids` from paper-era artefacts on disk, executes
the cell source verbatim against an Agg backend, and writes the resulting PNG
to a deterministic temp path. The cell's `save_figure(...)` call is monkey-
patched to a `plt.savefig` of the temp path; `plt.show()` is a no-op. No other
modification.

Usage:
    python scripts/m9_capture_truth.py <notebook> <cell_index> <out_png>

Example (HPS rankings pilot):
    python scripts/m9_capture_truth.py \
        legacy/cmip6-greece/GR_model_performance_HM.ipynb 13 \
        /tmp/m9_truth_HPS_rankings_annual_and_seasons.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd


HMPERF_FULL_XLSX = "results/greece/assess_cmip6_composite_HMperf_full_greece.xlsx"
MODEL_ID_XLSX = "Data/CMIP6/metadata/CMIP6_model_id.xlsx"


def load_legacy_inputs(country: str = "greece") -> dict:
    """Reproduce the in-cell namespace the legacy notebook builds upstream."""
    ranked_full = pd.read_excel(HMPERF_FULL_XLSX, index_col=0)
    cmip6_models = pd.read_excel(MODEL_ID_XLSX)
    model_ids = dict(zip(cmip6_models["model"], cmip6_models["id"]))
    return {
        "ranked_full": ranked_full,
        "model_ids": model_ids,
        "country": country,
        "cmip6_models": cmip6_models,
    }


def run_cell(notebook: Path, cell_index: int, out_png: Path) -> None:
    nb = json.loads(notebook.read_text())
    cell = nb["cells"][cell_index]
    if cell.get("cell_type") != "code":
        raise SystemExit(f"cell {cell_index} is not a code cell")
    src = "".join(cell.get("source", []))

    # Build the cell's expected namespace.
    ns = load_legacy_inputs()

    # Stand-in for the legacy `save_figure(...)` helper: capture every kwarg
    # call and write to our deterministic out_png path. Same dpi the cell asks
    # for is honoured.
    def save_figure(country, png_dir, svg_dir, file_name, dpi=300, **_):
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")

    ns["save_figure"] = save_figure

    # plt.show() is a no-op under Agg; keep the cell call intact.
    exec(compile(src, f"<cell_{cell_index}>", "exec"), ns)
    plt.close("all")
    print(f"truth bytes → {out_png}  ({out_png.stat().st_size} bytes)")


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        raise SystemExit(2)
    notebook = Path(sys.argv[1])
    cell_index = int(sys.argv[2])
    out_png = Path(sys.argv[3])
    run_cell(notebook, cell_index, out_png)


if __name__ == "__main__":
    main()
