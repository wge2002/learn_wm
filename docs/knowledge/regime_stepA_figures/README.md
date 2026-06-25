# Step A existence — discrete regime lives in f, not in z

Direction: [direction_discrete_regime_from_lewm.md](../direction_discrete_regime_from_lewm.md) · Step A (pure analysis, no training).
Run: 2026-06-25, `quentinll/lewm-pusht`, PushT expert (ID), 800 windows → 6400 transition
records, Jacobian on 4000, D=192, HS=3, elapsed 271s.
Script: `scripts/plan/regime_existence_stepA.py`.

## Question

Does a discrete **regime** structure exist in the already-trained LeWM transition `f`,
even though SIGReg forces the marginal `p(z)` to an isotropic Gaussian? Theory: discreteness
cannot live in the state `z` (structureless by construction) but can live in `f(z,a)` because
contact physics is piecewise.

## Method

Along expert trajectories, read three feature families per transition and compare:
`z` (state, **control**) · residual direction `f(z,a)-z` · **Jacobian** `df/dz` summarized by
its top-32 singular-value spectrum (the local operator). Each: standardize → PCA → KMeans
(k=2..8), best silhouette; align clusters to the env contact signal `n_contact_points`.

## Result — GREEN

| feature | best k | silhouette | contact NMI | ARI | purity |
| --- | --- | --- | --- | --- | --- |
| z (control) | 8 | 0.068 | 0.043 | 0.043 | 0.68 |
| residual dir | 8 | 0.037 | 0.097 | 0.068 | 0.69 |
| **Jacobian spectrum** | **2** | **0.396** | **0.298** | **0.371** | **0.81** |

Across all k, the Jacobian carries silhouette 0.23–0.40 and contact-NMI 0.20–0.30; `z` stays
at ~0.05 silhouette and ~0 NMI. See `stepA_silhouette_nmi.png`.

## Reading

- **`z` is structureless and contact-blind** — exactly the SIGReg isotropic-marginal prediction.
- **`f`'s local operator (Jacobian) is strongly clustered and contact-aligned** — the discrete
  regime lives in the transition, ~6× silhouette / ~7× NMI vs the `z` control.
- Best k=2 ⇒ the dominant axis is a binary **contact / non-contact** hybrid switch; k=3–4 stay
  high (purity ↑ 0.82) ⇒ finer regimes underneath.
- **Residual direction is weak** (silhouette 0.037): it is confounded by the action drive. The
  clean regime signal is in the autonomous operator. → Step B gating should condition on the
  state/operator, not on the raw residual.

## Caveats

Heuristic verdict (margins are large, not borderline). PushT-only, latent-level, ID condition;
cross-env / shift robustness is Step E. Raw per-record features in
`outputs/regime_stepA/full_20260625_1656/stepA_features.npz` (not in Git).
