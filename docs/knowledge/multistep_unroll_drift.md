# Multi-step unroll training flattens LeWM drift (clean positive)

Spin-off from the regime direction (which closed): the one solid, gimmick-free anti-drift
lever that fell out of Step B. Direction-relevant because it directly attacks the diagnosed
disease (on-manifold compounding drift) with **no gate / no regime, one hyperparameter**.

## Idea

LeWM trains its predictor **single-step teacher-forced** (`num_preds=1`): predict the next
latent from true history. Multi-step drift (what hurts rollout/planning) is never directly
optimized. Fix: train with a **multi-step open-loop unroll loss** — seed `history_size`=3 true
frames, feed predictions back for `unroll`=5 steps, MSE vs the true future; encoder co-trained.
Implemented as a branch in `scripts/train/lewm.py:lejepa_forward` (cfg.wm.unroll), config
`scripts/train/config/lewm_multistep.yaml` (num_steps=8). Everything else identical to baseline.

## Result (from-scratch end-to-end, 30 epochs, full LeWM ~18M, 8-GPU DDP)

Fair eval: 3-frame-seed open-loop latent rollout for BOTH models (`regime_lewm_iter2_eval.py`),
normalized latent MSE@k vs each model's own true latents.

| k (steps) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single-step baseline | 0.01 | 0.03 | 0.06 | 0.10 | 0.15 | 0.20 | 0.26 | **0.315** |
| **multi-step (unroll=5)** | 0.02 | 0.04 | 0.05 | 0.07 | 0.09 | 0.12 | 0.15 | **0.177** |

**~44% lower drift at 8 steps (0.315 → 0.177), slope much flatter.** Confirms the iter1
frozen-encoder signal at full from-scratch + encoder-co-trained scale. The mechanism is textbook
(imagination/unroll training reduces compounding error, à la Dreamer) — not novel, but cleanly
demonstrated inside LeWM's own pipeline.

## Honest caveats

- **1 seed / 1 run each** — effect is large and the curve *shape* (flat vs compounding) is
  robust to normalization (raw mse@8: 0.153 vs 0.303), but seed-confirm still wanted.
- **Small 1-step cost:** multi-step is marginally worse at pure k=1 (0.02 vs 0.01) — expected
  trade (optimizes the horizon, not the single step).
- **Needs a 3-frame warmup history at inference** — it cannot cold-start from 1 frame (that was
  the earlier 1.22 "failure": `model.rollout`'s 1-frame seed mismatched the 3-frame training
  seed). Normal for history-conditioned WMs; just keep a 3-step buffer.
- **Not yet tested on planner cost-rank** (the deployment metric) — drift is the prerequisite
  and it's clearly improved; planning eval is the natural next step.
- Different latent spaces across runs (var 0.866 vs 0.961) — handled by normalizing; raw curves
  agree.

## Next

1. Seed-confirm (2-3 seeds) — but each from-scratch run is ~8h (multi-step) / ~3.4h (single).
2. Planner cost-rank: does flatter latent drift translate to better CEM action ranking?
3. Sweep `unroll` (3/5/8) and try a curriculum (single→multi) for stability/efficiency.

Scripts: `scripts/train/lewm.py` (+`--config-name lewm_multistep`),
`scripts/plan/regime_lewm_iter2_eval.py`. Checkpoints: `iter2_multistep`, `iter2_baseline`
(not in Git). Result: `outputs/regime_stepB2/multistep_eval.json`.
