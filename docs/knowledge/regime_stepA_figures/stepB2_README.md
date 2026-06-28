# Step B iteration 1 (faithful) — regime-MoE inside LeWM's REAL Transformer predictor

Direction: [direction_discrete_regime_from_lewm.md](../direction_discrete_regime_from_lewm.md) · Step B
faithful redo, closing the toy-MLP caveats. Run: 2026-06-28, frozen pretrained encoder +
trained predictor stage on 8000 contact-labeled latents, 3 seeds, 60 epochs.
Script: `scripts/plan/regime_lewm_predictor_stepB2.py`. Engineering: [stepB_engineering.md](stepB_engineering.md).

## What changed vs the toy MLP testbed

The earlier Step B used a 1.6M MLP predictor trained multi-step. Reviewer-raised caveats:
architecture (MLP≠LeWM's Transformer), scale (1.6M≪LeWM), objective (multi-step≠LeWM's
single-step). This iteration fixes 3 of them: **"mono" here IS LeWM's real predictor stage**
(`Embedder` action encoder + `Predictor` Transformer depth6/heads16 + `pred_proj`), at
**~11.8M params (LeWM scale)**, and we run both **multi-step** and **single-step (LeWM-native
`num_preds=1`)** training. Encoder stays frozen (predictor-only); end-to-end encoder
co-adaptation would be iteration 2.

## Result (multi-step training, 3 seeds) — core verdict CONFIRMED at LeWM scale

| config (~11.8M, param-matched) | mse@8 | vs mono | rollout gate acc |
| --- | --- | --- | --- |
| mono (LeWM predictor) | 0.233 ± 0.011 | — | — |
| moe-unsup (blind gate) | 0.229 ± 0.022 | Δ=−0.004, **p=0.84 (tie)** | 0.57 (≈floor 0.60) |
| moe-gatesup (contact gate) | 0.296 ± 0.011 | Δ=+0.064, **p=0.004 (worse)** | 0.91 |
| **oracle (true contact route)** | **0.202 ± 0.010** | Δ=−0.031, **p=0.041 (better)** | (given) |

## Reading

- **Regime is valuable with perfect routing — replicates at LeWM scale.** oracle beats mono
  significantly (0.233→0.202, p=0.041, ~13%), same size of gain as the toy MLP (0.368→0.320).
- **No realizable gate captures it — replicates.** The unsupervised gate is contact-blind (acc
  0.57) and lands exactly at mono (tie, p=0.84); supervising it to contact (acc 0.91)
  *significantly hurts* (p=0.004). Brittleness wall confirmed: oracle (100% routing) 0.202 vs
  contact gate (91%) 0.296 → the ~9% misroute costs +0.094, dwarfing oracle's 0.031 gain; you'd
  need ~99% routing to profit. Same conclusion as the toy MLP.
- **One thing the faithful test CORRECTS:** at LeWM scale MoE does **not actively hurt** — blind
  MoE ties mono, whereas the toy 1.6M MLP MoE was *worse* than mono (0.403 vs 0.368). So the
  toy "MoE harms" was a small-model artifact; the durable claim is "**no realizable regime gate
  beats mono; only the (unrealizable) oracle does**," which holds at LeWM architecture + scale.
- **Single-step (LeWM-native) training** (iteration-1 single-seed pass): all arms get great
  one-step MSE (~0.02) but worse multi-step drift (0.26–0.29) than multi-step-trained models
  (0.20–0.23), and oracle-single did not help drift. So multi-step training is the right
  objective for anti-drift, and switching to LeWM's single-step objective does not rescue the
  regime gate. (single-step seeds not yet run.)

## Verdict / next step

The faithful LeWM-scale test **confirms** the Step B verdict and removes its one overstatement:
regime-as-predictor-switch is valuable only with an oracle gate; every realizable gate ties or
loses to the monolithic LeWM predictor. **No "苗头" for a realizable regime predictor** ⇒
iteration 2 (full from-scratch retrain with encoder co-adaptation) is **not** triggered on these
results. The one thing iteration 2 could still test: whether co-training the encoder reshapes
latents so contact becomes ~99% routable — a long shot against the replicated brittleness wall;
deferred to user.

Raw: `outputs/regime_stepB2/iter1_*`, `iter1seeds_*` (not in Git).

## Iteration 2 (full from-scratch, encoder co-adapts) — regime still NOT routable

Run: 2026-06-28→29, baseline LeWM + MoE LeWM trained **from scratch end-to-end**
(encoder ViT + predictor + SIGReg, single-step LeWM-native loss, 30 epochs, 4-GPU DDP
each, ~9h). MoE predictor = drop-in `module.MoEPredictor` (gate→2 Predictor experts,
depth3 each, param-matched). This tests the one thing iteration 1 (frozen encoder) could
not: does letting the **encoder co-adapt** reshape the latent so the contact regime
becomes routable? Scripts: `scripts/train/lewm.py --config-name lewm_moe`,
`scripts/plan/regime_lewm_iter2_eval.py`.

| model (from scratch, ~18M, 30ep) | norm mse@10 | gate→contact NMI |
| --- | --- | --- |
| baseline LeWM | 0.401 | — |
| MoE LeWM | 0.369 | **0.003** (acc 0.556, floor 0.60) |

**Verdict: NO.** Even with full end-to-end encoder co-adaptation, the MoE gate is
**completely contact-blind** (NMI 0.003) — *lower* than the frozen-encoder iteration 1
(~0.08). SIGReg's pressure toward a structureless marginal wins; the contact regime cannot
be read from the latent by a learnable gate. The MoE's small drift edge (0.369 vs 0.401)
comes with a blind gate, so it is **not** a regime effect (and, like the iter-1 single-seed
edge that vanished at 3 seeds, likely noise / generic MoE capacity).

**This closes the regime-as-learned-gate path at every level:** toy MLP → faithful frozen
Transformer → faithful end-to-end with encoder co-adaptation. The keeper remains the
**oracle** result (regime carries real dynamical value if known); a learnable gate never
recovers it. Raw: `outputs/regime_stepB2/iter2_*`, `iter2_eval.json` (not in Git).
