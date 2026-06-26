# Step C — regime-triggered re-grounding loses to uniform (RED, even with oracle)

Direction: [direction_discrete_regime_from_lewm.md](../direction_discrete_regime_from_lewm.md) ·
Step C(iii) "regime boundary = when to re-ground".
Run: 2026-06-26, `quentinll/lewm-pusht`, 1500 PushT expert windows, max_k=10, action_block=5.
Script: `scripts/plan/regime_reground_stepC.py`. Figure: `stepC_reground.png`.

## Question

Step B killed the regime-as-predictor form (specialized experts are brittle to routing
error). The regime's surviving proposed use is as a *monitoring signal*: re-ground (replace
the drifting latent with a fresh true encoding) AT regime boundaries instead of at fixed
intervals. Does regime-timed re-grounding beat **budget-matched uniform** re-grounding (same
number of re-grounds per trajectory)?

## Method

Re-grounding mechanism is identical to `phase3.regrounded_rollout` (reseed from a single true
frame, history rebuilds) — only the *schedule* of reseed points differs between arms, so the
comparison is apples-to-apples. Schedules:
- **open-loop**: reseed only at k=0.
- **regime @ boundary**: reseed at k where the binary contact label flips (onset/release).
- **regime before boundary**: reseed at k-1 (fresh anchor right before the hard transition).
- **uniform (budget-matched)**: per trajectory, the SAME number of reseeds as the regime arm,
  evenly spaced.
- **fixed every {1,2,3,5}**: reference ladder.

Metric: area = mean latent-MSE over interior steps k=1..10 (area under the drift sawtooth;
lower = better drift control). Significance: paired t-test across trajectories (oracle uses
true contact, so this is the *ceiling* for the monitoring idea — exactly the Step B logic).

## Result — RED

| schedule | area-MSE (mean over k) |
| --- | --- |
| open-loop (0 reground) | 0.2166 |
| **uniform (budget-matched ~3.5)** | **0.0652** |
| regime @ boundary (oracle) | 0.0951 |
| regime before boundary (oracle) | 0.0994 |
| fixed every 1 / 2 / 3 / 5 | 0.026 / 0.043 / 0.056 / 0.104 |

Paired, on trajectories where the schedules differ (n=1412): regime @ boundary is **worse**
than budget-matched uniform, Δ=+0.0317, **p<0.001**. Before-boundary helps a little (0.099 vs
0.095) but still loses to its matched uniform (Δ=+0.0227, p<0.001). See `stepC_reground.png`:
uniform's sawtooth stays capped; the regime curves let error accumulate in long un-anchored
gaps.

## Reading — why, and why it's fundamental

**Re-grounding controls error *accumulation* (drift), not the per-step difficulty of a
transition.** Within a segment, latent error grows monotonically with steps-since-reground, so
total drift is minimized by minimizing the gaps between re-grounds → **uniform spacing is
near-optimal**. Contact boundaries *cluster* (onset and release happen a few steps apart), so
spending the budget there leaves long un-anchored stretches elsewhere where drift balloons.
Knowing the regime tells you *where the dynamics are interesting*, but that is exactly NOT where
re-ground budget should go — it should go where the next gap would otherwise be largest, which
is regime-agnostic. Re-grounding also cannot reduce the error of the hard transition itself
(the model still must predict through the contact switch); it only resets what accumulates
after, which uniform already handles better.

This is the *oracle* schedule (true contact). A realizable test-time trigger could only be
worse, so no realizable round is warranted — same go/no-go logic that made Step B's oracle the
decisive test.

## Verdict for the direction

Combined with Steps A/B, the picture is complete and consistent:
- **Step A (GREEN, stands):** the discrete regime genuinely *exists* in the trained transition
  `f` (Jacobian↔contact, NMI 0.30) — a real, paper-worthy descriptive finding about LeWM.
- **Step B (RED):** regime is valuable only with oracle routing; no realizable gate captures it
  and specialized predictors are brittle to routing error.
- **Step C (RED):** regime-timed re-grounding loses to regime-agnostic uniform at equal budget.

**Unifying insight:** the regime is real and informative as an *analytical fact* about the
dynamics, but it does not convert into an *actionable control lever* — neither as a predictor
switch (brittle to routing) nor as a re-ground trigger (uniform dominates). Both failures share
a mechanism: knowing *where* the dynamics are special doesn't help you *act*, because the cost
structure (routing brittleness; drift accumulation) doesn't reward regime-localized action.
The honest landing: keep Step A as a standalone analysis result; the actionable direction is
exhausted.

## Caveats

PushT-only, latent-level, ID. Binary contact regime (Step A best-k=2). Metric is latent-MSE
area; a planner-cost-rank metric (Step C i) was not separately run, but the drift result is the
prerequisite and it is negative. Raw: `outputs/regime_stepC/full_20260626_1748/` (not in Git).
