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

## ⚠️ Planner result OVERTURNS the "positive": lower drift, MUCH worse control

Ran the actual PushT MPC/CEM benchmark (`scripts/plan/eval_wm.py`, 50 episodes, CEM
num_samples=300) on both from-scratch checkpoints:

| model | latent drift mse@8 | **PushT planning success (50 ep)** |
| --- | --- | --- |
| single-step baseline | 0.315 | **82% (41/50)** |
| multi-step (unroll=5) | **0.177** (−44%) | **22% (11/50)** |

**The multi-step model has far lower latent drift yet plans catastrophically worse.**
Lower self-consistency drift does NOT imply better control — here it's strongly inverted.

**Mechanism — first hypothesis REFUTED, real lesson is deeper.** I guessed multi-step collapses
action-sensitivity. A counterfactual probe (vary the action, measure the spread of the predicted
next latent) **refutes that**: multistep is *more* action-sensitive, not less
(action_spread 0.374 vs baseline 0.307). So the failure is not a collapse.

The real culprit is that **drift-MSE is *self-referential*: it compares the predictor's rollout
to the *same model's own encoder* outputs.** Encoder and predictor co-adapt, so a model can drive
self-drift down by reshaping its latent into a manifold that is easy to roll forward
*self-consistently* — without that manifold being **task/goal-aligned or physically accurate**.
CEM plans by ranking action sequences via `MSE(predicted final latent, goal latent)`; if the
self-consistent latent doesn't encode goal-discriminative / physically-faithful structure, the
cost landscape is uninformative and planning fails (→22%), even though the model is action-
sensitive and low-drift *in its own latent*. So **low self-referential drift ≠ a good world
model; it can be "gamed" by encoder+predictor co-adaptation into a self-consistent but
task-irrelevant latent.**

**Takeaways:**
- The multi-step "win" is dead — it's harmful for the real task. Honest negative.
- This is the cleanest evidence yet that **the training objective must target planning-relevant
  structure (counterfactual/action sensitivity, goal-aligned geometry), not self-consistent
  drift.** Directly motivates a theory-derived loss (see below).
- Caveat: 1 seed each (seed-1 reruns training to confirm 82 vs 22 isn't a fluke); the gap (60
  points) is far beyond plausible seed noise.
- Action-sensitivity probe done: multistep 0.374 > baseline 0.307 → action-insensitivity ruled
  out; the failure is task-misalignment of a self-consistent latent (above).

## Next

1. Confirm the planner gap with seed-1 models (training now); planner-eval the pretrained
   100-epoch LeWM as a reference anchor.
2. Verify the action-insensitivity mechanism (counterfactual sensitivity ‖∂z'/∂a‖).
3. Feed into the IDEA: a loss that combines LeWM's marginal structure (SIGReg) with a
   *transition* term that preserves action-discriminability / piecewise dynamics, instead of
   plain multi-step drift.

Scripts: `scripts/train/lewm.py` (+`--config-name lewm_multistep`),
`scripts/plan/regime_lewm_iter2_eval.py`. Checkpoints: `iter2_multistep`, `iter2_baseline`
(not in Git). Result: `outputs/regime_stepB2/multistep_eval.json`.
