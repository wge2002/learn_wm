# Controlled-metric v2 K1 failure diagnosis

Date: 2026-08-13

## Verdict

The two K1 processes did **not** first reach NaN parameters or a NaN forward.
They were deliberately aborted by the v2 formal policy at the first
non-finite backward gradient:

| seed | epoch | global step | last complete validation |
| ---: | ---: | ---: | --- |
| 13 | 12 | 137496 | epoch 11: loss 2.1009, pred .4481, SIGReg 18.3658 |
| 42 | 10 | 115683 | epoch 9: loss 3.4999, pred .5672, SIGReg 32.5831 |

`NonFiniteGradGuardCallback` ran before gradient clipping, set all gradients to
`None`, and then `nonfinite_grad_policy=error` raised. Therefore the saved model
and AdamW state were still finite. The DLC job status `Failed` means the strict
formal process exited nonzero; it does not mean no usable checkpoint was made,
nor does it prove standard LeWM is intrinsically unstable.

Without the guard, the known bf16 **propagation chain** is: one Inf gradient
makes the global clip norm Inf; clipping multiplies it by zero; `Inf * 0`
becomes NaN; AdamW then permanently contaminates both parameters and moments.
This explains how one bad element kills the run, but it is not the generating
root cause of that first Inf. v2 did not record the offending tensor or the
pre-forward RNG/buffer state, so the exact operator that emitted it remains
unknown.

The distinction matters: a plain training objective producing an organic Inf
is not accepted as “normal numerical noise.” The August guard prevented silent
checkpoint corruption, but its skip mode was an operational workaround, not a
scientific explanation.

## Is skip part of standard LeWM?

No. Git commit `44c45bd` (“Adding LeWM (#161)”, 2026-03-23) introduced the
baseline with `precision: bf16` and `gradient_clip_val: 1.0`, but no non-finite
gradient callback and no skipped-update policy. `NonFiniteGradGuardCallback`
was added locally in commit `4b017b5` on 2026-08-09 after later experiments
showed silent NaN contamination.

Therefore the previous wording “ordinary LeWM uses exact skip” was wrong. The
accurate statement is:

- original LeWM: bf16 + clip, with no protection against a first Inf;
- current instrumented code: fail before clipping and save evidence;
- optional exact skip: diagnostic/operational only, never evidence that a
  formal run was healthy.

## Why this was not actually a standard-LeWM control

The v2 arm retained a one-step prefix prediction but changed several parts of
the original LeWM training construction:

- it encoded an 8-frame clip instead of the original 4-frame clip;
- SIGReg acted on all 8 frames, while prediction supervised only frames 0--3;
- frames 4--7 therefore supplied SIGReg-only encoder gradients in K1, whereas
  K5 used them as prediction targets;
- K1 made one predictor call while K5 made five;
- K1 and K5 did not use the same target positions;
- dropout was set to zero and the formal policy aborted on the first bad
  gradient, whereas the original implementation had no such guard at all.

The bitwise initialization, split and first-batch hashes were correctly paired,
but those checks do not repair this objective/coverage mismatch. Consequently,
the v2 failures cannot be reported as “standard LeWM collapsed.”

## Additional checks

- K1 seed 7 completed all 30 epochs, so failure was not deterministic by arm.
- Resuming the finite seed-13/42 checkpoints and crossing the original global
  step locations with a fresh loader/RNG produced no non-finite gradient. This
  rules out an already-poisoned checkpoint, but is not an exact replay because
  v2 checkpoints did not persist DataLoader/CUDA RNG state.
- The exact two failing batches were reconstructed from the logged split and
  loader hashes. They shared dataset row `1023045`, but that row was not a
  high-loss outlier under either saved checkpoint (prefix-loss descending ranks
  99/128 and 117/128). It is therefore not a supported root-cause attribution.
- v2 did not record the first offending tensor or raw pre-clip gradient. The
  exact CUDA operator that emitted the Inf is unknowable post hoc; claiming one
  would exceed the evidence.

## Fix

v3 replaces the prefix-only control with five teacher-forced one-step LeWM
transitions over the same target positions used by recursive K5. Both arms now
make five predictor calls; the only intervention after the first transition is
true versus predicted context.

Formal launch is paused until the first-Inf source is reproduced and fixed.
Both arms now use strict `nonfinite_grad_policy=error`: a formal pair requires
zero organic non-finite gradients. Every event writes the offending parameter,
NaN/Inf counts, losses, batch hash, model/optimizer state, and opt-in exact
pre-forward RNG/BatchNorm-buffer evidence. Skip mode remains available only for
explicit diagnostics and cannot pass the pairing verifier.

The next gate is a two-GPU expected-failure rerun of the exact superseded v2 K1
construction at seeds 13 and 42 (`lewm_nonfinite_v2_k1_repro`). It reuses the
original initialization artifacts and preserves the original
`trainer.max_epochs=30`, because that value controls the epoch-based cosine LR
schedule. The job counts as
diagnostically successful only if both strict guards reproduce and each writes
one replay bundle. The offending parameter identifies the branch; saved
pre-forward RNG and BatchNorm buffers then support an operator-level replay.
Only after a minimal numerical fix lets both seeds cross their old failure
points with zero events may the six-model v3 formal wave launch.

An initial DSW attempt on 2026-08-13 incorrectly set `max_epochs=13`. Both seeds
then completed with zero events, but their LR had already decayed to zero
(versus roughly `3.3e-5`/`3.8e-5` near the historical failures). That run is a
different optimization trajectory and is not stability evidence. The corrected
reproduction keeps `max_epochs=30` and uses a callback-only diagnostic stop at
global step 138000, just beyond both historical failure steps, so the scheduler
is unchanged without spending the remaining epochs if reproduction fails.

See [the v3 preregistration](controlled_metric_paired_protocol_v3_20260813.md).
