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

- **Seed-confirm now done:** seed-1 repeats the same qualitative split: self-drift improves
  strongly, but planning remains much worse than baseline (details below).
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

Seed-1 confirms the same direction:

| model | norm drift mse@8 | **PushT planning success (50 ep)** |
| --- | ---: | ---: |
| single-step baseline s1 | 0.251 | **86%** |
| multi-step s1 | **0.130** | **40%** |

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
- The multi-step "win" is dead as a method — it is harmful for the real task despite lower
  self-drift. Honest negative.
- This is the cleanest evidence yet that **the training objective must target planning-relevant
  structure (counterfactual/action sensitivity, goal-aligned geometry), not self-consistent
  drift.** Directly motivates a theory-derived loss (see below).
- **Seed-confirmed (2 seeds, 2026-07-01):** the inversion is robust —
  baseline planning 82%/86% (drift 0.315/0.251) vs multistep planning 22%/40% (drift 0.177/0.130).
  Multistep is consistently *lower drift, much worse planning* (~53-pt gap).
- Action-sensitivity probe done: multistep 0.374 > baseline 0.307 → action-insensitivity ruled
  out; the failure is task-misalignment of a self-consistent latent (above).

## Stop-grad multistep follow-up (sgmulti): partial rescue, not enough

Follow-up from [theory_sufficiency_loss.md](theory_sufficiency_loss.md): keep the single-step
LeWM term shaping the encoder, add a multi-step open-loop term where encoder outputs are
stop-grad so the extra horizon loss is predictor-only. This tests the minimal hypothesis:
"multi-step is useful, but only if it cannot directly reshape the encoder."

Remote runs (2026-07-01→02) used `lewm_sgmulti` with `unroll_sg=5`, β=1 and β=2:

| model | loss setting | norm drift mse@8 | **PushT planning success (50 ep)** |
| --- | --- | ---: | ---: |
| baseline | single-step LeWM | 0.315 | **82%** |
| pure multi-step | encoder+predictor co-trained multi-step | **0.177** | 22% |
| `sgmulti_b1` | single-step + β=1 predictor-only multi-step | 0.358 | 50% |
| `sgmulti_b2` | single-step + β=2 predictor-only multi-step | 0.361 | 52% |

Readout:
- `sgmulti` improves planning over pure multi-step (22% → 50/52%), so cutting the direct
  multi-step gradient into the encoder does remove part of the damage.
- It does **not** beat baseline and does **not** reduce drift; in fact drift is worse than
  baseline. The original "drift and planning improve together" prediction is false.
- The remaining issue is likely that the target geometry is still moving: even with stop-grad
  on the multi-step branch, the encoder is still trained end-to-end by the single-step term.

## Next

The next mechanism test should freeze a planning-good baseline encoder `φ0` and train only the
predictor/action side `f` with one-step + multi-step losses in that fixed latent space. That is
cleaner than `sgmulti` because goal latents, rollout targets, and planning cost all live in the
same fixed metric. If fixed-`φ0` works, the final method should be an encoder anchor/EMA version;
if it fails, the problem is deeper than encoder erosion and points toward planning-rank or
control-sufficient geometry losses.

Scripts: `scripts/train/lewm.py` (+`--config-name lewm_multistep`),
`scripts/plan/regime_lewm_iter2_eval.py`. Checkpoints: `iter2_multistep`, `iter2_baseline`
(not in Git). Result: `outputs/regime_stepB2/multistep_eval.json`.
