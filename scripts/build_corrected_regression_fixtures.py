"""Build the regression fixtures for Greece.

Runs ``subselect.performance.compute_metrics`` (all 4 vars) +
``compute_hps`` + ``subselect.spread.compute_change_signals`` for
Greece and writes parquet snapshots under
``tests/fixtures/regression_corrected/``. ``tests/test_regression_corrected.py``
pins the live pipeline outputs against these snapshots at machine
epsilon. Re-run this script after a deliberate methodology change to
refresh the pinned values.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from subselect.config import Config
from subselect.performance import (
    PERIODS,
    EPS_HPS,
    compute_hps,
    compute_metrics,
    minmax_normalize,
)
from subselect.spread import compute_change_signals

REPO = Path(__file__).resolve().parents[1]
COUNTRY = "greece"
HPS_VARIABLES = ("tas", "pr", "psl")
ALL_VARIABLES = ("tas", "pr", "psl", "tasmax")

FIXTURES_DIR = REPO / "tests" / "fixtures" / "regression_corrected"
RESULTS_DIR = REPO / "results" / COUNTRY
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def _build_hps_full(per_var_tables: dict[str, pd.DataFrame], hps: pd.DataFrame) -> pd.DataFrame:
    """Build the per-(model, period) composite frame with TSS_mm, bs_mm,
    bias_score_raw, HMperf, and rank — matches the legacy
    ``assess_cmip6_composite_HMperf_full_<country>.xlsx`` schema.
    """
    out = pd.DataFrame(index=hps.index)
    for period in PERIODS:
        comp_tss = pd.concat(
            [per_var_tables[v][f"{period}_tss"] for v in HPS_VARIABLES],
            axis=1,
        ).mean(axis=1, skipna=True)
        comp_bs_raw = pd.concat(
            [per_var_tables[v][f"{period}_bias_score"] for v in HPS_VARIABLES],
            axis=1,
        ).mean(axis=1, skipna=True)
        tss_mm = minmax_normalize(comp_tss)
        bs_mm = minmax_normalize(comp_bs_raw)
        hmperf = 2.0 * (tss_mm * bs_mm) / (tss_mm + bs_mm + EPS_HPS)
        out[f"{period}_rank"] = hmperf.rank(ascending=False).astype(int)
        out[f"{period}_HMperf"] = hmperf
        out[f"{period}_TSS_mm"] = tss_mm
        out[f"{period}_bias_score_mm"] = bs_mm
        out[f"{period}_bias_score_raw"] = comp_bs_raw
    return out.sort_values("annual_HMperf", ascending=False)


def main() -> None:
    config = Config.from_env()

    print("Loading per_variable_metrics fixtures from prior partial run...")
    per_var_tables: dict[str, pd.DataFrame] = {}
    for variable in ALL_VARIABLES:
        cached = FIXTURES_DIR / f"per_variable_metrics_{variable}.parquet"
        if cached.is_file():
            print(f"  → {variable}: cached fixture")
            per_var_tables[variable] = pd.read_parquet(cached)
        else:
            print(f"  → {variable}: computing")
            df = compute_metrics(COUNTRY, variable=variable, config=config)
            per_var_tables[variable] = df
            df.to_parquet(cached)
            df.to_excel(
                RESULTS_DIR / f"assess_cmip6_{variable}_mon_perf_metrics_all_seasons_{COUNTRY}.xlsx"
            )

    print("Running compute_hps...")
    hps = compute_hps(COUNTRY, config=config)
    hps_renamed = hps.rename(columns={p: f"{p}_HMperf" for p in PERIODS})
    hps_renamed.to_parquet(FIXTURES_DIR / "composite_hps.parquet")
    hps_renamed.to_excel(
        RESULTS_DIR / f"assess_cmip6_composite_HMperf_{COUNTRY}.xlsx"
    )

    print("Building composite_HMperf_full...")
    hps_full = _build_hps_full(
        {v: per_var_tables[v] for v in HPS_VARIABLES}, hps
    )
    hps_full.to_parquet(FIXTURES_DIR / "composite_hps_full.parquet")
    hps_full.to_excel(
        RESULTS_DIR / f"assess_cmip6_composite_HMperf_full_{COUNTRY}.xlsx"
    )

    print("Running compute_change_signals...")
    # compute_change_signals returns the change deltas (long_term − pre_industrial).
    # Spread is not affected by the M-CORRECT σ_obs corrections (spread uses
    # cos(lat)-weighted spatial means already, and never touches the obs
    # reference); the long_term and pre_industrial absolute-window xlsx files
    # are paper-era artefacts left in place under results/greece/.
    change_df = compute_change_signals(COUNTRY, scenario="ssp585", config=config)
    change_df.to_parquet(FIXTURES_DIR / "change_signals.parquet")
    change_df.to_excel(
        RESULTS_DIR / f"assess_long_term_change_spread_{COUNTRY}.xlsx"
    )

    # Observed std xlsx — feeds the Taylor diagrams' refstd. Columns are
    # variables, rows are periods. Same shape as legacy.
    print("Building observed_std_dev xlsx...")
    from subselect.performance import _compute_obs_std_per_period
    sigmas = {
        var: _compute_obs_std_per_period(var, COUNTRY, "bbox", config)
        for var in ALL_VARIABLES
    }
    obs_std_df = pd.DataFrame(
        {var: [sigmas[var][p] for p in PERIODS] for var in ALL_VARIABLES},
        index=list(PERIODS),
    )
    obs_std_df.to_excel(RESULTS_DIR / f"assess_observed_std_dev_{COUNTRY}.xlsx")
    obs_std_df.to_parquet(FIXTURES_DIR / "observed_std_dev.parquet")

    print(f"\nWrote fixtures to {FIXTURES_DIR}")
    print(f"Refreshed xlsx artefacts under {RESULTS_DIR}")
    print("\nFixtures:")
    for p in sorted(FIXTURES_DIR.glob("*.parquet")):
        print(f"  {p.name}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
