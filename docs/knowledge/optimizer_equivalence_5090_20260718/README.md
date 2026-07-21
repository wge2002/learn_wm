# 5090 CEM selection / optimizer-equivalence audit

Current cross-machine conclusions and the stop/go ledger are consolidated in
[`../lewm_planning_status_20260721.md`](../lewm_planning_status_20260721.md).

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
- `oe_update_resample_*_summary/`: one-step oracle-update dose intervention.
- `oe_update_rollout_*_summary/`: recursive proposal-update interventions.
- `oe_ft12_crossfit_*_summary/`: state-held-out fixed-trace training controls;
  both recorded settings are negative results.
- `cem_round_h5_off40_k3_n60_seed20260719_v1_summary/`: independent
  60-state K3 trace and update-equivalence replication.
- `oe_ft60_crossfit_*_summary/`: the locked 60-state bridge and targeted-loss
  development follow-up; both miss the precommitted gate.

The three large complete-population archives stay on `5090lan` rather than in
Git:

| remote artifact | size | SHA-256 |
| --- | ---: | --- |
| `outputs/week1/selection_round_5090/cem_round_h5_off40_n12_full_v2.npz` | 12 MiB | `edfbf663ece346a8426bd67e91780f1a7aaa1c016809493fd9ab4847e0966825` |
| `outputs/week1/selection_round_5090/cem_round_h8_off60_n12_full_v2.npz` | 16 MiB | `7d0a3d04991047be1ddf217c20bb5418aaf858729e63188a059620cb89caff3a` |
| `outputs/week1/selection_round_5090/cem_round_h5_off40_n50_evalstarts_final_v2.npz` | 2.5 MiB | `36260cca882800009040c343b6fdacf4ffe85f5bb01a4d81c74d3f7cc4d917d1` |
| `outputs/week1/selection_round_5090/cem_round_h5_off40_n12_refit30_v1.npz` | 48 KiB | `83720897ff0fdd9b82edd7097508fc900e3ffa4113a4ad92c8df5f5e3a30b91b` |
| `outputs/week1/oe_update_5090_20260718/oe_update_rollout_k3_start4_r25_n12_n100_v2.npz` | 1.3 MiB | `1f29cbb123e338d3c48d634e6cbbebb176d0d2b2a8a36712a17f16ae99b73871` |
| `outputs/week1/selection_round_5090/cem_round_h5_off40_k3_n60_seed20260719_v1.npz` | 12 MiB | `fd8e3623e10c51db37ae14a2f42f6da29b656552079d9f090d13541c82ac1e07` |
| `outputs/week1/oe_update_5090_20260718/oe_update_rollout_k3_h8off60_start4_r25_n12_n100_v1.npz` | 984 KiB | `244533b036d2a8f153f485ee9731a15d86a68aae9114dd755537b7500ce7793a` |

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
- `oe_update_rollout_k3_start4_r25_n12_n100_v2_summary/report.md` for the
  strict recursive causal intervention (`3/12 → 7/12` final-mean success).
- `oe_update_rollout_k3_h8off60_start4_r25_n12_n100_v1_summary/report.md`
  for the independent H8/off60 pressure cell (`3/12 → 6/12`).
- `oe_ft12_crossfit_all_lr1e5_v1_summary/report.md` and
  `oe_ft12_crossfit_proj_lr2e6_a02_v1_summary/report.md` for the two rejected
  12-state fine-tuning settings.
- `oe_ft60_crossfit_all_lr2e6_a02_v1_summary/report.md` and
  `oe_ft60_crossfit_massrel_v2_summary/report.md` for the locked bridge failure
  and the non-confirmatory targeted-loss follow-up.

These are exploratory comparisons. The pooled bootstrap resamples paired
episodes; with only three seeds, the reports also retain the complete
between-seed delta range.
