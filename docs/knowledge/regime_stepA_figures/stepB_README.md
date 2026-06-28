# Step B decisive — regime helps (oracle), but the learned gate is blind

Direction: [direction_discrete_regime_from_lewm.md](../direction_discrete_regime_from_lewm.md) · Step B
(minimal training, original LeWM loss + multi-step unroll, no added clustering loss).
Run: 2026-06-25, `decisive_20260625_1913`, 5 configs × 3 seeds, 8000 contact-labeled
windows, K=2 experts, hist=3, unroll=5, 60 epochs (~11s/run).
Scripts: `scripts/plan/regime_moe_stepB.py` (trainer), `scripts/plan/regime_stepB_aggregate.py`
(stats + figure). **工程级复刻细节（训了什么、怎么训）见 [stepB_engineering.md](stepB_engineering.md)。**

## Question

Does conditioning the transition on a discrete regime flatten rollout drift (mse@k slope),
and does the regime emerge unsupervised from the LeWM rollout loss? Step A proved the regime
*exists* in the trained `f` (Jacobian→contact NMI 0.30). Step B asks whether it's *usable*.

## Method

Compare, at matched params, on open-loop latent rollout (mse@k, k=1..10):
- **mono-h512 / mono-h1024** — single continuous predictor (0.66M / 1.85M), the baselines.
- **moe-state / moe-both** — MoE `f = Σ g_k(·)·f_k`, K=2, soft gumbel train / hard argmax eval;
  gate conditioned on state only vs state+action (1.62M).
- **oracle** — MoE with the gate replaced by the ground-truth contact bin (upper bound: "if the
  regime were known perfectly, does anti-drift appear?").

Two metrics: rollout drift (mse@10 + slope) and gate→contact alignment (NMI of the learned
hard route vs the env contact label). Significance: 2-sample t-test over 3 seeds.

## Result

| config | mse@1 | mse@10 | slope | gate→contact NMI |
| --- | --- | --- | --- | --- |
| mono-h512 (0.66M) | 0.0525 | 0.430±0.013 | 0.0413 | — |
| mono-h1024 (1.85M) | 0.0393 | 0.368±0.010 | 0.0365 | — |
| moe-state (1.62M) | 0.0505 | 0.403±0.009 | 0.0392 | **0.008** |
| moe-both (1.62M) | 0.0499 | 0.392±0.015 | 0.0379 | **0.006** |
| **oracle (1.62M)** | 0.0439 | **0.320±0.018** | **0.0304** | (perfect by constr.) |

**Decisive test — oracle vs param-matched continuous (mono-wide 1.85M):**
mse@10 0.320 vs 0.368, Δ=−0.048, **t=3.29, p=0.030**; slope 0.030 vs 0.037.

See `stepB_decisive.png`.

## Reading — split verdict (NOT dead)

- **Hypothesis (b) "regime is worthless" is REFUTED.** With perfect contact routing, regime
  conditioning *significantly* flattens drift below the param-matched continuous baseline
  (p=0.030, ~13% lower mse@10, lower slope). The payoff the direction predicted is real.
- **The learned gate fails.** moe-state/both barely beat mono-narrow and *lose* to mono-wide
  at equal params; gate→contact NMI ≈ 0.006–0.008, i.e. the gate is contact-blind even with
  action input. The rollout loss alone does not make the gate discover the regime that Step A
  showed is sitting in the Jacobian (NMI 0.30, dashed line in figure).
- **Bottleneck = gate discovery, not regime value.** This is hypothesis (a): the regime is
  useful but the unsupervised emergence path (gumbel gate on rollout MSE) can't find it.
  Adding capacity to a blind gate just splits data ~50/50 with no contact meaning.

## Why so few epochs suffice

Both signals are visible at convergence of an 11s run because: (i) drift is measured open-loop
over 10 latent steps, so a 5% per-step operator difference compounds into a clean Δ at k=10;
(ii) gate-NMI is a property of the *routing*, which saturates early — the gate either latches
onto contact or it doesn't, and here it never does regardless of training length. Longer
training does not move NMI off ~0.

## Implication for next step

Oracle proves the ceiling exists, so the productive move is to *supervise / warm-start the gate*
toward the Step A regime (Jacobian-cluster or contact label) rather than hoping rollout MSE
discovers it — then test whether a learnable (not oracle) gate can close the 0.320↔0.403 gap.
This is consistent with the direction's "natural emergence" goal only if a weak prior (read
from the already-trained `f`, per Step A) counts as emergence rather than a new clustering loss
on `z`. Decision deferred to user.

## Round A (2026-06-26) — can a learnable gate close the gap? NO. Step B (MoE form) dies.

The decisive round left open whether the bottleneck was the regime (refuted) or the gate
(hypothesis a). Round A attacked the gate three ways and the answer is conclusive: **no
realizable gate recovers the oracle benefit; every realizable MoE variant is worse than the
plain continuous predictor.** See `stepB_roundA.png`. mse@10 (3 seeds each):

| variant | needs test-time label? | mse@10 | gate→contact NMI |
| --- | --- | --- | --- |
| **oracle** (GT route train+eval) | **yes** | **0.320** | 1.0 (given) |
| mono-wide (continuous baseline) | no | 0.368 | — |
| blind MoE (no sup) | no | 0.403 | 0.007 |
| clean-experts + learned gate, soft eval | no | 0.475 | 0.56 |
| clean-experts + learned gate, hard eval | no | 0.486 | 0.56 |
| supervised gate (gumbel), soft eval | no | 0.510 | 0.55 |
| supervised gate (gumbel), hard eval | no | 0.521 | 0.55 |

Three interventions, all fail to help:
1. **Weak gate supervision** (aux CE → contact, `--gate-sup`): the gate *does* find contact
   (NMI 0.008→0.55, purity 0.61→0.91), but drift gets *worse*, monotonically with the
   supervision weight (gs0.1→0.405, gs1.0→0.521, gs3.0→0.682). Routing correctly ≠ predicting
   better.
2. **Clean expert specialization** (`--train-route-gt`: GT routes experts in training like the
   oracle, learned gate only at eval): 0.486 — still worse than blind/mono. So the gap is NOT
   soft-gumbel expert blending.
3. **Graceful soft routing at eval** (`--eval-soft`): 0.475 / 0.510 — barely moves. So the gap
   is NOT hard-argmax brittleness either.

**Mechanism (corrected by the routing probe — `regime_stepB_routing_probe.py`).** Earlier we
guessed "the gate goes blind on drifted rollout states." **Direct measurement overturns that
framing:** the supervised gate routes at **0.91 accuracy on true states and still 0.887 on the
drifted open-loop rollout** (trivial floor 0.60) — routing barely degrades on drift. The real
mechanism is the *catastrophic cost per misroute* of specialized experts: with **100% routing
the oracle reaches mse@10 0.320, but ~89% routing (the supervised gate) lands at 0.506** — the
~11% misroute adds +0.18 drift and blows past the monolithic 0.368. So the specialization
benefit (~13%, oracle vs mono) is far smaller than the misroute penalty; you would need ~99%
routing for MoE to pay off, which no learnable gate reaches. Soft routing avoids brittleness but
blends specialists back into a generalist, forfeiting the benefit (hard = benefit-but-brittle;
soft = robust-but-≈monolithic). Separately, the **blind gate stays blind (NMI ~0) under BOTH the
multi-step and a single-step (LeWM-native `num_preds=1`) training objective**, so the
non-emergence is not an artifact of our multi-step loss. The oracle wins only because it *never*
misroutes — and it needs the contact label at inference, which is action-dependent and not worth
predicting from the latent given the brittleness.

**Verdict: Step B in the latent-MoE form is DEAD** (doc kill criterion "压不平→方法死": every
realizable variant has a steeper or equal slope than mono-wide). Step A's existence result
stands; the regime is real and informative, but conditioning the *predictor* on it does not buy
usable anti-drift. The regime's remaining viable use is as a *monitoring / re-grounding signal*
(Step C control layer: "regime boundary = when to re-ground"), which does not require routing a
brittle predictor — a different and weaker claim, deferred to user.

Scripts: `regime_moe_stepB.py` (+`--gate-sup`,`--train-route-gt`,`--eval-soft`),
`regime_stepB_roundA_figure.py`, `regime_stepB_routing_probe.py` (teacher-forced vs rollout
routing + single-step ablation). Raw: `outputs/regime_stepB/{gatesup_,trgt_,softeval_}*`,
`routing_probe.json` (not in Git).

### Scope caveat (train/eval differs from LeWM proper)
This whole Step B is a *controlled testbed on frozen LeWM latents*, not LeWM's own predictor:
(i) architecture — ours is an MLP on raw actions, LeWM's predictor is a Transformer (depth 6,
heads 16) with an action-encoder; (ii) objective — we train **multi-step open-loop unroll**
(U=5) while LeWM trains **single-step teacher-forced** (`num_preds=1`). The mono-vs-moe-vs-oracle
comparison is internally fair (all identical), but the negative result is scoped to "regime-as-
dense-MLP-MoE on frozen latents," and a working result would not have been drop-in to LeWM's
Transformer + single-step + CEM-planning pipeline. The faithful test (MoE inside LeWM's own
training) remains undone. See [stepB_engineering.md](stepB_engineering.md) Part 1.

## Caveats

PushT-only, latent-level, ID. K=2 (binary contact switch from Step A best-k). Oracle uses the
contact_frac>thresh bin; finer regimes (Step A k=3–4) untested here. 3 seeds (t-test, small n).
Raw runs: `outputs/regime_stepB/decisive_20260625_1913/` (not in Git).
