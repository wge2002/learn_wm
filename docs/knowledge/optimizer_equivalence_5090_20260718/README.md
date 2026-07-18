# 5090 CEM selection / optimizer-equivalence audit

This directory contains the reproducible summaries and compact raw results for
the 2026-07-18 selection audit documented in
[`horizon_bundle_temporal.md` §13](../horizon_bundle_temporal.md).

## Contents

- `cem_round_*_summary/`: proposal/scoring decomposition, per-state
  strategies, and learned-vs-oracle CEM update equivalence.
- `cemcv_*_summary/`: paired end-to-end MPC comparisons and pooled
  three-seed controls.
- `e2e_results/`: append-only evaluator records, including exact episode
  success vectors and resolved Hydra configuration.
- `oracle_refits/`: compact executed top-k refit archives for the exact
  evaluator-matched 50 starts.
- `oracle_refit_summary.csv`: human-readable aggregate of those archives.

The three large complete-population archives stay on `5090lan` rather than in
Git:

| remote artifact | size | SHA-256 |
| --- | ---: | --- |
| `outputs/week1/selection_round_5090/cem_round_h5_off40_n12_full_v2.npz` | 12 MiB | `edfbf663ece346a8426bd67e91780f1a7aaa1c016809493fd9ab4847e0966825` |
| `outputs/week1/selection_round_5090/cem_round_h8_off60_n12_full_v2.npz` | 16 MiB | `7d0a3d04991047be1ddf217c20bb5418aaf858729e63188a059620cb89caff3a` |
| `outputs/week1/selection_round_5090/cem_round_h5_off40_n50_evalstarts_final_v2.npz` | 2.5 MiB | `36260cca882800009040c343b6fdacf4ffe85f5bb01a4d81c74d3f7cc4d917d1` |
| `outputs/week1/selection_round_5090/cem_round_h5_off40_n12_refit30_v1.npz` | 48 KiB | `83720897ff0fdd9b82edd7097508fc900e3ffa4113a4ad92c8df5f5e3a30b91b` |

All paths are relative to `/mnt/data/wge/learn_wm` on `5090lan`. The dataset
was `/mnt/data/wge/data/pusht_eval_state_only.h5`; checkpoints were resolved
through `STABLEWM_HOME=/mnt/data/wge/stablewm`.

## Main checks

The fastest entry points are:

- `cem_round_h5_off40_n12_full_v2_summary/report.md` for the late-round
  elite-update divergence.
- `cem_round_h5_off40_n50_evalstarts_final_v2_summary/report.md` and
  `oracle_refit_summary.csv` for candidate/refit headroom.
- `cemcv_h5off40_summary/report.md` for all paired variants.
- `cemcv_h5off40_model_vs_multistart_summary/report.md` for the equal-call
  model-diversity comparison.
- `cemcv_h5off40_vs_s100_summary/report.md` for the catastrophic
  shared-population ensemble control.

These are exploratory comparisons. The pooled bootstrap resamples paired
episodes; with only three seeds, the reports also retain the complete
between-seed delta range.
