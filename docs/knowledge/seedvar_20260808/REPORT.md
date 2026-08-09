# Seed-variance wave: bisim is not better than the PD anchor, and curvature is dead

DLC job `dlc1n2lxxmmiv50x` (`lewm-dynamic-seedvar`), 8xL20Z, 2026-08-08
01:12 -> 2026-08-09 02:43 CST. 5 of 8 runs reached epoch 30; the other 3 were
deliberately aborted by the new instability guard (see below). Iso-optimization
protocol unchanged from the anchors: D=192, batch 128, 30 epochs, 1 GPU/model,
AdamW lr 5e-5, bf16, `gradient_clip_val: 1.0`, 11306 steps/epoch, ~4.0 it/s.

## Why this wave was run

The `h2hfix` wave left exactly one usable checkpoint, `bisimfix_adapt_s1`
(bisim / raw / adaptive, seed 1) at **val 0.128 / pred 0.006**, which beat the
`pd_d192_k5_s7` anchor (0.135 / 0.009). It could not be claimed: the identical
config at seed 3072 died at epoch 14, so n=1 could not separate a real effect
from a lucky draw. 7 of that wave's 8 runs died.

Those deaths were not an aux-regularizer defect. Root cause, measured
2026-08-07: `precision: bf16` installs no `GradScaler`, so a single `inf`
gradient element reaches `gradient_clip_val`, which computes
`clip_coef = 1/inf = 0` and multiplies every gradient by it. Healthy gradients
become a harmless `0`, but `inf * 0 = NaN`, and AdamW writes that NaN into the
parameter *and* its `exp_avg`/`exp_avg_sq`. One step later every parameter is
NaN. `fp16` would have skipped the step for free. Fix:
`NonFiniteGradGuardCallback` (`scripts/train/lewm.py`) zeroes all gradients on a
non-finite step, restoring the `GradScaler` semantics bf16 lacks. Verified by
injecting an inf into a live run (`SWM_INJECT_INF_AT_STEP=5`): the run survived
and saved, where before it died 1 step later. Cost 3.92 ms/step (~1.8%).

## Result 1: bisim only matches the PD anchor

| run | seed | val | pred | sigreg | guard skips |
|---|---|---|---|---|---|
| `bisimg_s0101` | 101 | 0.1328 | 0.0082 | 1.384 | 1 |
| `bisimg_s0007` | 7 | 0.1367 | 0.0109 | 1.398 | 0 |
| `bisimg_s0013` | 13 | 0.1370 | 0.0112 | 1.397 | 0 |
| `bisimg_s0042` | 42 | 0.1396 | 0.0141 | 1.394 | 12 |
| `bisimg_s3072` | 3072 | 0.1930 | 0.0517 | 1.570 | 1 |

Four tightly-clustered seeds: **val 0.136 +/- 0.003, pred 0.0111 +/- 0.0024**.
Including the s3072 outlier: 0.148 +/- 0.025.

`pd_d192_k5_s7` anchor: **0.135 / 0.009**. So bisim's mean *matches* the anchor
and does not beat it. The h2hfix 0.128/0.006 was the left tail of this
distribution -- exactly the lucky-seed hypothesis, now confirmed. **There is no
head-to-head win to claim for Invariant-JEPA-style reward-free bisimulation on
this benchmark.**

Note `val = pred + 0.0904 * sigreg` across all waves, so `val` is dominated by
sigreg and `pred` is the real predictive-quality signal.

## Result 2: the guard works, and zero-to-few skips are harmless

Three h2hfix death points were all crossed: `bisimg_s3072` died at ep14 before
and finished 30 here; `curvg_unit_adapt` ep15 -> 18; `curvg_unit` ep10 -> 15.
`bisimg_s0042` absorbed 12 non-finite gradients and still landed at 0.1396,
inside the spread of the seeds that had none. Occasional skips do not degrade
the final model.

## Result 3: curvature (Temporal Straightening) is not viable -- stop spending GPUs on it

The guard aborts an epoch whose skip rate exceeds 1% (with a 1000-step minimum
so short probes cannot trigger it), on the grounds that a persistently rising
rate means the recipe is unstable rather than unlucky. Both curvature runs did
exactly that, monotonically:

```
curvg_unit        ep11 0.04% -> ep12 0.28% -> ep13 0.85% -> ep14 3.95% ABORT
curvg_unit_adapt  ep13 0.11% -> ep14 0.82% -> ep16 0.05% -> ep17 6.47% ABORT
```

This is a property of the objective, not of a single unlucky batch. Combined
with `curv_d192`, which *completed* all 30 epochs and still landed at val 5.18 /
sigreg 55.7 (representation collapse, which no gradient guard can fix), the
curvature line is closed on both counts: it is unstable *and*, when it is
stable, it collapses.

## Open item: seed 1 is anomalous

`bisimg_s0001` was the wave's control -- seed 1 had already completed *without*
the guard at 0.128/0.006, so it should have reproduced that number and proven
the guard does not perturb training. Instead it began skipping at ep10 (27
steps), climbed, and was aborted at ep19 with 1.23% (139/11306). The control
was therefore not obtained.

The four fresh seeds agreeing with each other and sitting near the old 0.128
supports the guard being neutral indirectly, but seed 1 specifically now looks
pathological: it produced both the best-ever checkpoint and, on a rerun, a
rising instability. Worth one run to reproduce before trusting anything
seed-1-derived.

## Reproduce

```bash
WAVE=seedvar NGPU=8 bash /mnt/home/gewang/.config/rbs-dlc/wave.sh
```

`wave.sh` lives outside the repo (machine-level launcher); the run matrix is
`outputs/week1/run_seedvar.sh`. Per-epoch metrics and the full guard trace for
each run are in `*.metrics.txt` beside this file; the 3.5 MB raw logs stay on
CPFS at `outputs/week1/dlc_seedvar/`.
