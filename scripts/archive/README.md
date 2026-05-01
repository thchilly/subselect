# M9 SSIM forensics tooling — archived

These scripts powered the bit-identical / SSIM ≥ 0.95 verification gate during
M9 / M9.1 / M9.2 (April–May 2026 refactor session). They caught real bugs:
`tight_layout` ordering, adjustText non-determinism, the cmip6-grid /
native-0.5° W5E5 confusion in `load_single_grid_w5e5`. After Phase 0 closed,
figure renders are no longer regression-gated — visual sign-off is the user
eyeballing the file, not an SSIM number.

Per `docs/post_phase_0_cleanup.md`:

- These scripts are **not** part of the recurring test suite.
- `pytest` does not import or invoke them.
- `tests/test_regression_greece.py` (HPS bit-identity vs paper xlsx) is a
  separate functional regression test on the metric pipeline; it stays.

If a future port needs SSIM verification for a specific change, run these
scripts manually:

```bash
# Capture truth bytes by re-executing legacy cells against built namespaces
python scripts/archive/m9_capture_truth_all.py

# Render M9 outputs
python scripts/regenerate_paper_figures.py --country greece

# Compute SSIM and side-by-side panels
python scripts/archive/m9_visual_diff_report.py --country greece
# → opens scripts/archive/m9_visual_diff_report.html
```

`m9_capture_truth.py` is the single-cell variant (used during the M9 pilot
port of `greece_HPS_rankings_annual_and_seasons.png`).

`m9_ssim_diff.py` computes SSIM for one (truth, m9) pair and writes a
side-by-side PNG.
