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

Without the guard, the known bf16 failure chain is: one Inf gradient makes the
global clip norm Inf; clipping multiplies it by zero; `Inf * 0` becomes NaN;
AdamW then permanently contaminates both parameters and moments. With exact
skip, that chain is cut before clipping.

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
  gradient, whereas ordinary runs use exact skip.

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
true versus predicted context. Both arms use exact bf16 skip with a locked
health gate (at most one skip per epoch and three total), and every event writes
the offending parameter, NaN/Inf counts, losses, batch hash, and model/optimizer
evidence bundle.

See [the v3 preregistration](controlled_metric_paired_protocol_v3_20260813.md).
