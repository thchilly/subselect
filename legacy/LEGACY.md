# Legacy notebooks

This directory contains the two exploratory notebook collections that produced the published Greece paper (`subselection_paper/`). They are **frozen** — never edited, restyled, or deleted. They live here on disk for reference (porting code into the `subselect/` package, regenerating paper figures during M9 visual diffs, archaeological lookups) but are **not tracked by the parent `subselect/` git repository**.

Both subdirectories are themselves independent git repositories with GitHub remotes:

| Subdirectory | Origin | Role |
|---|---|---|
| `legacy/cmip6-greece/` | `github.com/thchilly/cmip6-greece` | Greece-specific code that produced the paper's HPS table and spread quadrants. Authoritative source for performance and spread figures in `subselection_paper/`. |
| `legacy/climpact/` | `github.com/thchilly/cmip6-subsampling` | Multi-country exploratory drafts. Contains lead-ins to Phase 1 (independence) and the prototype shapefile loader referenced by M4. |

Their inner `.git/` directories are preserved as-is; any uncommitted local state is intentionally left untouched. Refer to those repositories' own histories for their evolution.

## What each notebook backs and what supersedes it

### `legacy/cmip6-greece/`

| Notebook / module | What it does | Paper artefact | Supersedes |
|---|---|---|---|
| `GR_model_performance_HM.ipynb` | Annual + seasonal HPS pipeline (TSS, BVS, harmonic mean) for Greece. Reads CMIP6 from `Data/CMIP6/monthly/` and obs from `Data/reference/monthly_cmip6_upscaled/` (post-restructure 2026-04-30; the inner notebook still references the pre-restructure `monthly_new/` and `ISIMIP3a/monthly_cmip6_upscaled/` Windows paths). | `subselection_paper/performance_figures/` (Taylor diagrams, HPS tables) | `subselect.performance` (M7) + `subselect.viz.taylor`, `subselect.viz.performance_figs` (M9) |
| `SN_GR_model_performance.ipynb` | Seasonal variant of the HPS pipeline (DJF/MAM/JJA/SON). | Seasonal HPS tables in the paper | Same modules; orchestrator handles season kwarg |
| `GR_model_spread.ipynb` | End-of-century change signals (Δtas, pr, Δtasmax) and quadrant scatter plots for Greece. | `subselection_paper/spread_figures/`, `subselection_paper/tas_pr_gwl/` | `subselect.spread` (M8) + `subselect.viz.spread_figs`, `subselect.viz.country_profile` (M9) |
| `GR_cost.ipynb` | Composite scoring + ranking by combined HPS / spread. Phase 2 prototype, not in `docs/refactor.md`. | Not directly in the paper; informs Phase 2 cost-function design | `subselect.optimize` (Phase 2) — left as a stub in Phase 0 |
| `functions.py` | Shared helpers: `prepare_subset`, `extract_subset`, `calculate_spatial_average`, per-variable metrics (`std`, `corr`, `bias`, `rmse`, `crmse`, `r²`, `calculate_perf_metrics`), `save_figure`. Hardcoded `H:/CLIMPACT/Data/` path at line 30 and 187–189. | Backs every paper figure indirectly | Split into `subselect.io` (path resolution), `subselect.geom` (cropping, weighting), `subselect.performance` (metrics) |
| `TaylorDiagram.py` | Yannick Copin's TaylorDiagram class: polar plot, `add_sample`, `add_grid`, `add_contours`. | Taylor diagrams in `subselection_paper/performance_figures/` | `subselect.viz.taylor` (M9) — single consolidated copy |

### `legacy/climpact/`

| Notebook / module | What it does | Status | Supersedes / Phase |
|---|---|---|---|
| `shp_extraction.ipynb` | Prototype shapefile-based country cropping. Cells 11–14 (most complete in cell 14) load GADM 4.1 (`gadm_410-levels.gpkg`) and use `rasterio.features.geometry_mask(..., all_touched=True)` after a coarse bbox pre-crop. Returns `(ds_masked, bbox_coords, country_shp)`. **M4 reference implementation.** | Reference for M4 | `subselect.geom.crop` (M4), method=`shapefile_lenient` reproduces this binary rule |
| `model_independence.ipynb` | Pairwise model similarity / family clustering exploration. | Phase 1 lead-in; not implemented in Phase 0 | `subselect.independence` (Phase 1) |
| `model_similarity_mlds.ipynb` | Model similarity via MDS on regional climatology. Uses both `xesmf.Regridder(method='bilinear')` and `ds.interp(method='linear')` for regridding to a common 2° grid. | Phase 1 exploration; not used in the paper | `subselect.independence` (Phase 1) |
| `future_spread.ipynb` | Multi-country exploratory spread analysis. | Pre-paper exploration | `subselect.spread` (M8) |
| `CLIMPACT_figures.ipynb`, `CLIMPACT_figures_assessment.ipynb`, `CLIMPACT_scatter_plots.ipynb` | Multi-country summary figures and scatter templates. | Exploratory | `subselect.viz` family (M9) |
| `functions.py` | Byte-identical to `cmip6-greece/functions.py` except a one-character typo (`pfd` → `pdf`) at line 189. | Duplicate | Same split as the cmip6-greece copy; typo fixed during port |
| `TaylorDiagram.py` | Byte-identical to `cmip6-greece/TaylorDiagram.py`. | Duplicate | `subselect.viz.taylor` (M9) — single consolidated copy |
| `TaylorDiagramCRMSE.py` | Variant of `TaylorDiagram` declaring an unused `crmse` argument on `add_sample`. **Deprecation candidate** — the extra parameter is never consumed by the implementation. | Likely experimental dead code | M9 retains the class with a `DeprecationWarning` for now; remove in a future cleanup |
| `taylorDiagram_test.py` | Standalone tests / examples for the TaylorDiagram classes. | Demo only | Not ported; demos belong in the new `notebooks/exploratory/` |

## Reading from `legacy/` during the refactor

`.gitignore` excludes the two code subdirectories from parent-repo tracking but does **not** prevent reads. The refactor agent and Athanasios can open any file under `legacy/` while porting code in M4–M9. Edits, however, are not allowed — these are the canonical inputs to the paper figures, and any change risks breaking the regression test before it has a chance to validate the new pipeline.

If a new methodology question arises during the refactor that requires running a legacy notebook (e.g. to confirm a numeric result), do so in a fresh kernel with `Data/` paths resolved through the inner repo's own conventions; do not redirect those notebooks to the new `subselect.config` path resolver.
