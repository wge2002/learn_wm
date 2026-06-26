# Step B decisive — regime helps (oracle), but the learned gate is blind

Direction: [direction_discrete_regime_from_lewm.md](../direction_discrete_regime_from_lewm.md) · Step B
(minimal training, original LeWM loss + multi-step unroll, no added clustering loss).
Run: 2026-06-25, `decisive_20260625_1913`, 5 configs × 3 seeds, 8000 contact-labeled
windows, K=2 experts, hist=3, unroll=5, 60 epochs (~11s/run).
Scripts: `scripts/plan/regime_moe_stepB.py` (trainer), `scripts/plan/regime_stepB_aggregate.py`
(stats + figure).

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

## Caveats

PushT-only, latent-level, ID. K=2 (binary contact switch from Step A best-k). Oracle uses the
contact_frac>thresh bin; finer regimes (Step A k=3–4) untested here. 3 seeds (t-test, small n).
Raw runs: `outputs/regime_stepB/decisive_20260625_1913/` (not in Git).
