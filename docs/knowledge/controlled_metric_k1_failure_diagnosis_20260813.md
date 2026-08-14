# Controlled-metric v2 K1 failure diagnosis

Date: 2026-08-13 (diagnosis closed 2026-08-15)

## Verdict

The two v2 K1 processes did **not** first reach NaN parameters or a NaN
forward. They were aborted by the strict v2 policy at the first non-finite
backward gradient:

| seed | epoch | global step | last complete validation |
| ---: | ---: | ---: | --- |
| 13 | 12 | 137496 | epoch 11: loss 2.1009, pred .4481, SIGReg 18.3658 |
| 42 | 10 | 115683 | epoch 9: loss 3.4999, pred .5672, SIGReg 32.5831 |

The root cause is now established: **a numerical pathology in the bf16 ViT
encoder backward.** The forward loss is finite and well behaved at both
failure points; the encoder backward produces gradients five orders of
magnitude above the healthy scale, and under `precision: bf16` that
magnitude is what eventually overflows.

Ruled out by direct evidence: a bad batch, forward-loss divergence, a
poisoned checkpoint, BatchNorm buffers, the DLC platform, and anything in
CEM (which never ran in these jobs).

## Reproduction

The corrected 30-epoch reproduction (`r2`, `lewm_nonfinite_v2_k1_repro`,
`max_epochs=30` with a callback-only diagnostic stop at step 138000) hit both
historical events **exactly**: seed 42 at epoch 10 / step 115683 and seed 13
at epoch 12 / step 137496, with finite forward losses in both cases. Same
seeds, same paired initializations, same step indices. The failure is
deterministic and reproducible, not a transient.

An earlier DSW attempt that set `max_epochs=13` completed with zero events,
but its epoch-based cosine LR had already decayed to zero (versus roughly
`3.3e-5`/`3.8e-5` near the historical failures). That is a different
optimization trajectory and is not stability evidence.

## The old guard measured post-clip gradients

`NonFiniteGradGuardCallback` originally hooked Lightning's
`on_before_optimizer_step`. Under `stable_pretraining.Module`'s **manual**
optimization that hook fires from inside `opt.step()` — after `training_step`
has already called `self.clip_gradients`. So the v2 evidence bundles recorded
the aftermath, not the cause: every offending tensor was reported with
`finite_max_abs=0` and a pure NaN count, which is the post-clip signature of
`inf * (1/inf) == inf * 0`, not the raw backward.

This corrects the earlier claim in this document that the guard ran before
clipping. It also means the propagation chain (one Inf → Inf global norm →
`clip_coef` 0 → `Inf * 0` = NaN → AdamW contaminates parameters and moments)
was being *observed at its second link*, and the saved zeros were mistaken
for the generating event. Because clipping ran first, the saved model and
AdamW state were still finite, and the job status `Failed` only means the
strict process exited nonzero.

The guard now dispatches from `after_manual_backward` (via
`RawGradientModule.on_raw_gradients`), the one window where an Inf is still
an Inf, and reports both a raw and an element-masked finite gradient norm.

## Replay measurements

With the exact failing step replayed under identical model state, RNG,
BatchNorm buffers and pixels:

| seed | bf16 raw grad norm | encoder-FP32 island | full FP32 |
| ---: | ---: | ---: | ---: |
| 42 | 13,982,356 | 220.2788 | 242.6049 |
| 13 | 1,907,000.875 | 14.9156 | — |

Three things follow. The bf16 norm exceeds the FP32 norm by roughly 6e4x
(seed 42) and 1.3e5x (seed 13), so the magnitude is an artifact of bf16
accumulation, not of the objective. The encoder-FP32 island lands within ~10%
of full FP32 (220.2788 vs 242.6049), so the remaining bf16 modules —
projector, predictor, action encoder — are well conditioned and need no
intervention. And the forward loss is finite and nearly identical in every
variant, confirming the defect is confined to the encoder backward.

### What the raw evidence cannot settle

The historical full runs reached Inf; in direct replay the raw bf16 gradients
were **huge but finite**. The saved evidence did not capture the true pre-clip
raw gradients of the failing step, nor the full CUDA backward execution state,
so a reconstructed replay cannot be guaranteed to reproduce the in-training
backward bitwise. Execution and reduction state that was never recorded may
therefore account for why the real run crossed the overflow threshold and the
replay landed below it. Consequently the pathology's *magnitude and location*
are established (encoder backward, five orders of magnitude, seed-independent),
but the specific emitting kernel remains unidentified and is deliberately not
named. That level of attribution is not required for the fix, since the fix
removes the low-precision accumulation the pathology depends on.

## Is skip part of standard LeWM?

No. Git commit `44c45bd` (“Adding LeWM (#161)”, 2026-03-23) introduced the
baseline with `precision: bf16` and `gradient_clip_val: 1.0`, but no
non-finite gradient callback and no skipped-update policy.
`NonFiniteGradGuardCallback` was added locally in commit `4b017b5` on
2026-08-09 after later experiments showed silent NaN contamination.

So the earlier wording “ordinary LeWM uses exact skip” was wrong. Accurately:

- original LeWM: bf16 + clip, with no protection against a first Inf;
- current instrumented code: detect raw non-finite gradients and fail;
- optional exact skip: diagnostic/operational only, never evidence that a
  formal run was healthy.

The distinction matters because a plain training objective producing an
organic Inf cannot be filed as “normal numerical noise.” The August guard
prevented silent checkpoint corruption, but its skip mode was an operational
workaround, not an explanation.

## Why this was not actually a standard-LeWM control

The v2 arm kept a one-step prefix prediction but changed several parts of the
original LeWM training construction:

- it encoded an 8-frame clip instead of the original 4-frame clip;
- SIGReg acted on all 8 frames, while prediction supervised only frames 0--3;
- frames 4--7 therefore supplied SIGReg-only encoder gradients in K1, whereas
  K5 used them as prediction targets;
- K1 made one predictor call while K5 made five;
- K1 and K5 did not use the same target positions;
- dropout was set to zero and the formal policy aborted on the first bad
  gradient, whereas the original implementation had no such guard at all.

The bitwise initialization, split and first-batch hashes were correctly
paired, but those checks do not repair this objective/coverage mismatch.
Consequently the v2 failures cannot be reported as “standard LeWM collapsed.”

## Supporting checks

- K1 seed 7 completed all 30 epochs, so failure was not deterministic by arm.
- Resuming the finite seed-13/42 checkpoints and crossing the original step
  locations with a fresh loader/RNG produced no non-finite gradient, ruling
  out an already-poisoned checkpoint.
- The two failing batches were reconstructed from the logged split and loader
  hashes. They shared dataset row `1023045`, but that row was not a high-loss
  outlier under either checkpoint (prefix-loss descending ranks 99/128 and
  117/128). Not a bad-batch failure.

## Minimal fix

Two changes, both scoped to the established cause:

1. **Opt-in encoder FP32 precision island.** `encoder_fp32` (default `false`,
   so every existing recipe keeps its numerics bit for bit) runs the ViT
   encoder under `torch.autocast(enabled=False)` in `LeWM.encode`. The
   projector, predictor and action encoder stay in bf16, which the replay
   above shows is safe.
2. **Raw guard placement.** `NonFiniteGradGuardCallback` and
   `DivergenceTraceCallback` now implement `on_raw_gradients`, dispatched from
   `RawGradientModule.after_manual_backward`, i.e. before `clip_gradients`.

v3 also replaces the prefix-only control with five teacher-forced one-step
LeWM transitions over the same target positions as recursive K5, so both arms
make five predictor calls and the only intervention after the first
transition is true versus predicted context.

### Production-shape preflight

Seed 42, production batch shape 128 x 8, encoder FP32 island enabled: all 297
gradients finite, largest element 5.5193, gradient norm **220.2788** —
matching the replay's encoder-FP32 value — with the projector confirmed still
in bf16. Cost: 0.4072 s per warm batch, peak allocated/reserved 38.34/39.90
GiB. The fix is numerically effective and fits the production footprint.

## Gate and remaining path

The gate is now the **two-seed step-138000 stability validation**
(`lewm_encoder_fp32_stability`, `MODE=stability`): seeds 13 and 42 on two
GPUs must both cross global step 138000 — past both historical failure points,
115683 and 137496 — with `nonfinite_grad_policy=error`, zero non-finite
events and a clean exit, writing `STABILITY_GATE_PASS.txt`.

**The formal wave remains blocked until that gate passes.** After it passes:

- the paired 3-seed K1-TF / K5 formal runs launch with the fix applied
  **symmetrically** to both arms;
- the six-model CEM conversion audit remains mandatory, not an optional
  appendix — a formal report is complete only with the training pairing
  proof, the representation audit and the CEM audit together.

Neither diagnostic mode is a formal control, and neither may be substituted
for a paired arm.

See [the v3 preregistration](controlled_metric_paired_protocol_v3_20260813.md).
