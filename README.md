# subselect

`subselect` picks a small, representative subset of CMIP6 climate models for
country-scale impact assessments. Given a country and a target subset size *k*,
the framework recommends *k* models that jointly satisfy three criteria:

1. **Historical fidelity** — the subset reproduces observed climate well
   over the country.
2. **Future-response spread coverage** — the subset stays representative of
   the *full* CMIP6 spread of projected end-of-century changes.
3. **Model independence** — the subset avoids redundant models that share
   land/ocean/atmosphere components or institutional lineage.

The methodology was first published as a Greece-only study in April 2026
(`subselection_paper/`); this package generalises that pipeline to any
country with a GADM polygon and produces the same figure set automatically.

---

## Installation

`subselect` targets Python ≥ 3.11. The authoritative dependency pin source is
`environment.yml`; bootstrap the conda environment and install the package
in editable mode:

```bash
conda env create -f environment.yml
conda activate subselect
pip install -e .
```

A pure-pip install also works inside an existing scientific-Python environment
that satisfies the lower bounds in `pyproject.toml`:

```bash
pip install -e .
```

---

## Quick start

```bash
# First country: builds the global cache (one-time, ~5–7 min) plus the
# per-country derivations (~1–2 min). Renders the full figure set under
# results/greece/figures/.
python -m subselect greece

# Second and subsequent countries: ~30–60 s on a warm global cache.
python -m subselect sweden

# Re-running the same country: <30 s when nothing has changed.
python -m subselect sweden
```

The CLI also accepts:

- `--global-only` — populate `cache/_global/` without rendering for any country
- `--no-figures` — run the L1 compute pipeline only
- `--no-bias-maps` — skip the bias-map figures (useful for fast smoke tests)
- `--include-seasonal-bias` — render DJF/MAM/JJA/SON bias maps in addition to annual
- `--only performance,spread,country_profile` — restrict to a figure-group subset
- `--force {all,country,global}` — bypass the corresponding cache and recompute
- `--output-dir <path>` — write figures somewhere other than the default

The Python API matches the CLI:

```python
from subselect.compute import compute
from subselect.render import render

state = compute("greece")          # L1: build state, populate caches
paths = render(state)              # L2: write figure set
```

---

## How it works

The pipeline is a clean two-layer architecture with a two-scope cache:

- **L1 — `subselect.compute.compute(country)`** produces every artefact
  needed for the figure set (HPS metrics, σ\_obs, monthly climatologies,
  change signals, annual time series, warming-level crossings, future
  anomalies, country-profile signals, bias-map fields). Returns a typed
  `SubselectState`.
- **L2 — `subselect.render.render(state)`** consumes a `SubselectState`
  and writes the figure set under `results/<country>/figures/{performance,
  spread, country_profile}/`.
- **Cache — `cache/_global/`** holds country-independent artefacts (per-(model,
  variable) climatologies and annual fields, native-grid σ maps), built once
  and reused across countries; **`cache/<country>/`** holds country-mean
  reductions and country-specific tables.

Adding a new country requires only that the country has a row in
`Data/country_codes/country_codes.json` and a polygon in the GADM 4.1
GeoPackage at `Data/shapefiles/gadm/gadm_410-levels.gpkg`. The first call
populates the global cache; subsequent country calls reuse it.

---

## Output

`python -m subselect <country>` writes:

```
results/<country>/figures/
├── performance/
│   ├── <country>_HPS_rankings_annual_and_seasons.png
│   ├── <country>_tas_seasonal_performance.png
│   ├── <country>_pr_seasonal_performance.png
│   ├── <country>_psl_seasonal_performance.png
│   ├── <country>_tasmax_seasonal_performance.png
│   ├── <country>_composite_taylor.png
│   └── <country>_<var>_annual_bias.png            (one per variable)
├── spread/
│   ├── <country>_annual_spread.png
│   └── <country>_seasonal_spread.png
└── country_profile/
    ├── <country>_WL_table.png
    ├── <country>_gwls_boxplot.png
    ├── <country>_tas_anomalies_table.png
    ├── <country>_tas_change.png
    ├── <country>_tas_change_spaghetti.png
    ├── <country>_pr_percent_anomalies_table.png
    ├── <country>_pr_percent_change_ratio.png
    └── <country>_pr_percent_change_spaghetti.png
```

---

## Methodology

`subselect` evaluates models along three orthogonal dimensions:

- **Historical Performance Score (HPS).** Per-(variable, season) Taylor Skill
  Score and Bias-Variability Score on `{tas, pr, psl}`, harmonic-mean-combined
  and min-max normalised across the 35-model ensemble. Reference dataset:
  W5E5; evaluation window: 1995–2014.
- **Future spread.** End-of-century (2081–2100 vs. 1850–1899) Δtas, Δpr, and
  Δtasmax under SSP5-8.5; rendered as quadrant scatter coloured by HPS rank.
- **Model independence.** Two complementary methods (feature-space k-means
  on regional climatology, and pairwise-RMSE genealogy clustering) score
  the redundancy of a candidate subset.

The full methodology — definitions, equations, regression-test
contracts, design decisions — is logged in
`documentation/methods.tex` (build with `pdflatex methods.tex`).

The framework is inspired by ClimSIPS (Merrifield, Brunner, Lorenz, Humphrey,
Knutti, 2023; doi:10.5194/egusphere-2022-1520). Its contribution beyond
ClimSIPS is **explicit country-scale customisation** and **transparent
diagnostics for any user choice**.

---

## Citation

A peer-reviewed paper on the Greece-only application was published in April
2026; cite it as:

```bibtex
@article{tsilimigkras2026subselection,
  author  = {Tsilimigkras, A. and Lazaridis, M. and Voulgarakis, A. and others},
  title   = {Climate projections for {Greece}: Defining a regional sub-ensemble from the {CMIP6} landscape},
  journal = {Theoretical and Applied Climatology},
  volume  = {157},
  pages   = {123},
  year    = {2026},
  doi     = {10.1007/s00704-026-06029-w},
  url     = {https://doi.org/10.1007/s00704-026-06029-w},
}
```

---

## Contributing

Issues and pull requests are welcome at
<https://github.com/thchilly/subselect>. New countries, methodology
extensions, and figure-style options are all in scope; behaviour-changing
edits should reproduce the pinned regression test
(`pytest -m regression`) within the documented tolerance ladder.

---

## License

MIT. See the `license` field in `pyproject.toml` for the package metadata.
