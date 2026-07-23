# Tail-validity feedback-channel gate（2026-07-23）

**判决：`CLOSE`。** per-state positive-alpha fixed-pop oracle gain is < .05

## Protocol audit

- requested source states: `60`; valid next replans: `60`; prefix terminal/truncated: `0`;
- source SHA-256: `7e5a87cad0f8d87177ab2fc180dcf7211eea3c4e3951071f281346a48a7d0b0b`;
- collector SHA-256: `ee9655b4d79574d9fa216274dd8f50cf5469eb57cc2c84f4ee8b253514a0f15f`; summarizer SHA-256: `cf1c356cb3db761af931eb2be2328ad866caabb506333408a8a365874dbd3ca5`;
- alphas: `[-1.0, 0.0, 0.5, 1.0]`; state bootstrap: `20000` draws.

## Recursive next-replan arms

| alpha | top30 recall | Δ recall | returned true cost | Δ true cost | returned success |
|---:|---:|---:|---:|---:|---:|
| `-1` | 0.116 [0.081,0.154] | -0.002 [-0.032,0.027] | 57.08 [36.79,84.73] | 9.94 [-6.13,36.21] | 0.583 [0.450,0.700] |
| `0` | 0.118 [0.086,0.154] | 0.000 [0.000,0.000] | 47.14 [33.00,64.16] | 0.00 [0.00,0.00] | 0.533 [0.400,0.650] |
| `0.5` | 0.113 [0.082,0.146] | -0.006 [-0.033,0.021] | 55.40 [34.30,85.35] | 8.27 [-2.62,26.68] | 0.533 [0.400,0.650] |
| `1` | 0.103 [0.074,0.134] | -0.016 [-0.048,0.015] | 68.55 [36.08,114.03] | 21.41 [-6.09,61.30] | 0.567 [0.433,0.683] |

## Fixed baseline population

| alpha | top30 recall | Δ recall | elite true cost |
|---:|---:|---:|---:|
| `-1` | 0.123 [0.091,0.159] | 0.006 [-0.010,0.022] | 48.30 [34.32,64.51] |
| `0` | 0.118 [0.085,0.153] | 0.000 [0.000,0.000] | 48.64 [34.65,65.03] |
| `0.5` | 0.111 [0.080,0.144] | -0.007 [-0.018,0.002] | 48.90 [34.75,65.59] |
| `1` | 0.112 [0.082,0.144] | -0.006 [-0.021,0.008] | 49.23 [34.98,65.53] |

Per-state hindsight best of `{0,.5,1}`: top30-recall gain 0.015 [0.009,0.022] (locked CLOSE threshold: `.050`).

## State-held-out alpha selection

5-fold held-out state assignments: `{'0': 48, '0.5': 12}`.

| metric | baseline | cross-fit selected | selected − baseline |
|---|---:|---:|---:|
| topk_recall | 0.118 [0.086,0.155] | 0.112 [0.082,0.146] | -0.006 [-0.018,0.003] |
| support | 0.683 [0.567,0.800] | 0.700 [0.583,0.817] | 0.017 [0.000,0.050] |
| oracle_min | 31.69 [20.21,45.19] | 39.93 [20.96,67.72] | 8.25 [-1.27,25.60] |
| mean_true | 47.14 [32.98,63.87] | 55.07 [33.37,85.17] | 7.94 [-1.67,25.33] |
| mean_success | 0.533 [0.400,0.650] | 0.533 [0.400,0.667] | 0.000 [0.000,0.000] |

## Prefix residual predicts next-replan misranking?

- residual norm vs `1-recall`: Spearman `0.102` `[-0.175,0.358]`;
- fixed 5-fold OOF ridge: `R²=-0.479`; `MAE=0.121` vs constant `0.107`; OOF Spearman `-0.037`.

## Interpretation guardrail

The correction is applied inside all 30 CEM rounds, so recursive differences are not a final-selector result. Alpha is nevertheless a one-parameter persistent-residual family. This `CLOSE` rejects the locked optimistic additive-residual channel; it is not a proof that every nonlinear or history-conditioned feedback model is impossible.
