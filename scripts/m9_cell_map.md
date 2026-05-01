# M9 — Cell map for verbatim paper-figure port

Each row maps a target figure filename to the legacy notebook cell that produces
it. Cell indices are 0-based (`nb["cells"][idx]`). All output paths are under
the new tree `results/<country>/figures/<category>/`.

Truth-bytes anchor (per user direction Q1): the SSIM target for every figure is
the **cell's own rendered output** captured by re-executing the cell in a
scratch script under the `subselect` conda env. The published-paper-figures
folder is **not** the reference (those were Photoshopped manually post-render
and are out of scope, including all composites and `_1pick`/`_4pick`/`_JJApick`
variants).

Selection rule (from brief): when multiple cells write the same filename, the
canonical cell is chosen by — in order — filename suffix
(`_rev12` > `_rev1` > `_revised` > `_named` > unsuffixed), markdown header
markers, then last cell in the notebook. Per user direction Q2,
`SN_GR_model_performance.ipynb` duplicates of `*_annual_taylor`,
`*_4season_taylor` and bias-map cells are **ignored entirely** in favour of
`GR_model_performance_HM.ipynb`.

Permitted verbatim deviations (only these; everything else stays byte-exact):
1. Replace `save_figure(...)` (and `plt.show()`) with `return fig` in each
   ported function. Adapter shim supplies the in-cell variables from the M7/M8
   cache instead of running the upstream pipeline cells.
2. **Q3** — bias maps gain a `include_seasonal_bias: bool = False` kwarg. Off
   by default → 3 annual outputs (tas/pr/psl). On → 15 outputs (3 vars × 5
   periods). Wired through `scripts/regenerate_paper_figures.py` as a CLI flag
   `--include-seasonal-bias`.
3. **Q4** — duplicate `save_figure(file_name=f"{variable}_<name>")` calls in
   every CLIMPACT_figures cell are stripped; only the
   `plt.savefig({country}_<name>.png)` call remains. Output filenames lose the
   stale `psl_` prefix entirely (target paths below reflect this).

---

## § Performance figures → `results/greece/figures/performance/`

| Target file | Notebook | Cell |
|---|---|---|
| `greece_annual_HM_hist_perf.png` | `GR_model_performance_HM.ipynb` | 12 |
| `greece_HPS_rankings_annual_and_seasons.png` | `GR_model_performance_HM.ipynb` | 13 |
| `greece_tas_seasonal_perf_revised.png` | `GR_model_performance_HM.ipynb` | 25 |
| `greece_pr_seasonal_perf_revised.png`  | `GR_model_performance_HM.ipynb` | 26 |
| `greece_psl_seasonal_perf_revised.png` | `GR_model_performance_HM.ipynb` | 27 |
| `greece_tasmax_seasonal_perf_revised.png` | `GR_model_performance_HM.ipynb` | 28 |
| `greece_tas_annual_taylor.png` | `GR_model_performance_HM.ipynb` | 29 (loop iter) |
| `greece_pr_annual_taylor.png` | `GR_model_performance_HM.ipynb` | 29 |
| `greece_psl_annual_taylor.png` | `GR_model_performance_HM.ipynb` | 29 |
| `greece_tasmax_annual_taylor.png` | `GR_model_performance_HM.ipynb` | 29 |
| `greece_tas_4season_taylor.png` | `GR_model_performance_HM.ipynb` | 32 (loop iter) |
| `greece_pr_4season_taylor.png` | `GR_model_performance_HM.ipynb` | 32 |
| `greece_psl_4season_taylor.png` | `GR_model_performance_HM.ipynb` | 32 |
| `greece_tasmax_4season_taylor.png` | `GR_model_performance_HM.ipynb` | 32 |
| `greece_tas_annual_bias.png` | `GR_model_performance_HM.ipynb` | 34 (default — annual only) |
| `greece_pr_annual_bias.png` | `GR_model_performance_HM.ipynb` | 34 |
| `greece_psl_annual_bias.png` | `GR_model_performance_HM.ipynb` | 34 |
| (12 seasonal bias maps when `--include-seasonal-bias`) | `GR_model_performance_HM.ipynb` | 34 |

`greece_tasmax_annual_bias.png`: the legacy cell loops over `['tas','pr','psl','tasmax']`. Tasmax is HPS-excluded but the cell still renders it — kept as part of verbatim port.

## § Spread figures → `results/greece/figures/spread/`

| Target file | Notebook | Cell |
|---|---|---|
| `greece_annual_annual_spread_rev12.png` | `GR_model_spread.ipynb` | 13 |
| `greece_seasonal_spread_perSeasonBars_right_named_rev1.png` | `GR_model_spread.ipynb` | 21 |

## § Country-profile figures → `results/greece/figures/country_profile/`

(Filename column reflects Q4 cleanup: `save_figure(file_name=...)` second call
removed, only `plt.savefig({country}_<name>.png)` remains. The stale `psl_`
prefix on `psl_tas_*` and `psl_pr_*` files is gone — these were never produced
by the `plt.savefig` pattern, only by the duplicate `save_figure` call.)

| Target file | Notebook | Cell |
|---|---|---|
| `greece_WL_table.png` | `CLIMPACT_figures.ipynb` | 17 |
| `greece_tas_anomalies_table.png` | `CLIMPACT_figures.ipynb` | 29 |
| `greece_tas_change.png` | `CLIMPACT_figures.ipynb` | 31 |
| `greece_tas_change_all_shaded.png` | `CLIMPACT_figures.ipynb` | 32 |
| `greece_tas_change_spaghetti.png` | `CLIMPACT_figures.ipynb` | 35 |
| `greece_pr_percent_anomalies_table.png` | `CLIMPACT_figures.ipynb` | 42 |
| `greece_pr_change.png` | `CLIMPACT_figures.ipynb` | 44 |
| `greece_pr_change_spaghetti.png` | `CLIMPACT_figures.ipynb` | 46 |
| `greece_pr_percent_change_ratio.png` | `CLIMPACT_figures.ipynb` | 49 |
| `greece_pr_percent_change_raw.png` | `CLIMPACT_figures.ipynb` | 50 |
| `greece_pr_percent_change_spaghetti.png` | `CLIMPACT_figures.ipynb` | 54 |
| `greece_gwls_boxplot.png` | `CLIMPACT_figures.ipynb` | 58 |
| `greece_gwls_boxplot_times.png` | `CLIMPACT_figures.ipynb` | 61 |

---

## § Out of scope (per Q1, Q5)

- All Photoshopped paper-folder targets: `1greece_allTaylorPlots.png`,
  `2greece_allTaylorPlots.png`, `*_anomalies_table_lined.png`,
  `*_1pick.png`, `*_4pick.png`, `*_JJApick.png`, `tasmax_annual4season_taylor`.
- Stale on-disk artefacts not produced by any current cell:
  `1greece_gwls_boxplot_times.png`, `greece_gwls_boxplot1.png`, all
  `greece_psl_*` country-profile duplicates, `_DJF`/`_JJA`/`_MAM` per-season
  spread variants, `seasonal_spread_absHM.png`, earlier rev annual spreads
  (`_rev1`, `_rev11` and unsuffixed when `_rev12` exists), all
  `*_seasonal_perf.png` (no suffix) when `_revised` exists.

---

## § In-scope cell count

- Performance: 9 cells (12, 13, 25, 26, 27, 28, 29, 32, 34) → 18 outputs (annual-only bias).
- Spread: 2 cells (13, 21) → 2 outputs.
- Country-profile: 13 cells → 13 outputs.

**Total: 24 cells; 33 figure outputs.** With `--include-seasonal-bias`, +12 more
bias maps → 45 outputs.

---

## § Bulk-port status (M9 run, 2026-05-01)

| Figure | SSIM vs cell-truth | Verdict |
|---|---|---|
| `greece_annual_HM_hist_perf.png` | 1.0000 | PASS |
| `greece_HPS_rankings_annual_and_seasons.png` | 1.0000 | PASS (pilot) |
| `greece_tas_seasonal_perf_revised.png` | 0.9768 | **FAIL — adjustText non-det.** |
| `greece_pr_seasonal_perf_revised.png` | 0.9827 | PASS |
| `greece_psl_seasonal_perf_revised.png` | 0.9863 | PASS |
| `greece_tasmax_seasonal_perf_revised.png` | 0.9798 | **FAIL — adjustText non-det.** |
| `greece_{tas,pr,psl,tasmax}_annual_taylor.png` | 1.0000 (×4) | PASS |
| `greece_{tas,pr,psl,tasmax}_4season_taylor.png` | 1.0000 (×4) | PASS |
| `greece_annual_annual_spread_rev12.png` | 1.0000 | PASS |
| `greece_seasonal_spread_perSeasonBars_right_named_rev1.png` | 1.0000 | PASS |
| `greece_WL_table.png` | 1.0000 | PASS |
| `greece_gwls_boxplot.png` | 1.0000 | PASS |
| `greece_gwls_boxplot_times.png` | 1.0000 | PASS |

**Total in scope: 19; pass: 17; fail: 2.**

The two FAILs are the `seasonal_perf_revised` panels for `tas` and `tasmax`. Root
cause **confirmed by independent re-capture**: re-running the legacy cell
verbatim produces a different PNG bit-for-bit each time
(md5(`greece_tas_seasonal_perf_revised.png`) differs across truth-capture
runs). The non-determinism is inside `adjustText.adjust_text(...)`, which
relies on a numpy random walk for label de-overlap. The same legacy cell on
the same data renders to SSIM ≈ 0.97–0.99 vs itself across runs — the M9 port
is therefore *as deterministic as the legacy cell*. **Surfaced for direction
per the brief; not patching.**

If the user wants a hard SSIM ≥ 0.98 gate for these two cells, the available
options are: (a) seed `np.random` immediately before each `adjust_text(...)`
call (one extra line; arguably stays "verbatim" because the legacy cell is
non-deterministic by accident, not by design); (b) drop `adjust_text` to its
fallback no-op branch and accept the slight label overlap; (c) accept SSIM
≈ 0.977 as the achievable upper bound.

### Deferred (14 figures, follow-up)

Cells that depend on smoothed time-series derivations done upstream in
``CLIMPACT_figures.ipynb`` cells 0–16 (annual_precipitation,
smoothed_temp_anomaly, smoothed_5th_temp_anomaly, smoothed_95th_temp_anomaly,
smoothed_median_temp_anomaly, tas_anomalies_table, pr_anomalies_table,
pr_percent_anom_table, smoothed_pr, smoothed_pr_pi_percent_change, …):

- `greece_tas_anomalies_table.png` — cell 29
- `greece_tas_change.png` — cell 31
- `greece_tas_change_all_shaded.png` — cell 32
- `greece_tas_change_spaghetti.png` — cell 35
- `greece_pr_percent_anomalies_table.png` — cell 42
- `greece_pr_change.png` — cell 44
- `greece_pr_change_spaghetti.png` — cell 46
- `greece_pr_percent_change_ratio.png` — cell 49
- `greece_pr_percent_change_raw.png` — cell 50
- `greece_pr_percent_change_spaghetti.png` — cell 54

Bias maps (cell 34 loop) deferred too — they need an `observed_maps` /
`bias_maps` adapter that runs M7's per-pixel observed and bias computation
for each (variable, period). The function ``fig_bias_maps_per_variable`` is
ported and wired through `--include-seasonal-bias`; only the adapter is
missing:

- `greece_tas_annual_bias.png`
- `greece_pr_annual_bias.png`
- `greece_psl_annual_bias.png`
- `greece_tasmax_annual_bias.png`

Both follow-ups land in M9.1 / M9.2 once the user signs off on the
SSIM-fail direction.

## § Module layout

```
subselect/viz/
  __init__.py
  taylor.py              ← Yannick Copin's TaylorDiagram class (verbatim from
                           legacy/cmip6-greece/TaylorDiagram.py)
  performance_figs.py    ← cells 12, 13, 25, 26, 27, 28, 29, 32, 34
                           (CATEGORY = "performance")
  spread_figs.py         ← cells 13, 21 from GR_model_spread
                           (CATEGORY = "spread")
  country_profile.py     ← cells from CLIMPACT_figures
                           (CATEGORY = "country_profile")
  _data_adapters.py      ← shims that produce notebook-shaped variables from
                           M7/M8 cache and paper-era xlsx artefacts
```

Each ported figure function:
- Module-level constant `CATEGORY` ("performance" / "spread" / "country_profile").
- Signature `def fig_<name>(*adapter_inputs, country: str = "greece") -> Figure`.
- Verbatim cell body, save_figure → `return fig`.
- Routed by `scripts/regenerate_paper_figures.py` to
  `results/<country>/figures/<CATEGORY>/<filename>.png`.
