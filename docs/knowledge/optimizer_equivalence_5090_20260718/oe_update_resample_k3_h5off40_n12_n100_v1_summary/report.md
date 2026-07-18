# OE update one-step causal gate

- Generator: `pd_d192_k3_eval`
- Paired states: 12
- Source CEM rounds: [4, 9, 19, 29]
- Next-population samples per intervention: 100

## Aggregate over states and source rounds

| alpha | coverage | Δ coverage | min true | Δ min true | model-refit success | Δ model success | oracle-refit success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.438 | +0.000 | 80.57 | +0.00 [+0.00, +0.00] | 0.354 | +0.000 [+0.000, +0.000] | 0.354 |
| 0.25 | 0.438 | +0.000 | 76.57 | -4.01 [-7.30, -1.58] | 0.354 | +0.000 [+0.000, +0.000] | 0.396 |
| 0.50 | 0.438 | +0.000 | 74.91 | -5.66 [-9.49, -2.45] | 0.354 | +0.000 [+0.000, +0.000] | 0.396 |
| 0.75 | 0.438 | +0.000 | 73.15 | -7.43 [-12.39, -3.22] | 0.354 | +0.000 [+0.000, +0.000] | 0.396 |
| 1.00 | 0.438 | +0.000 | 71.11 | -9.47 [-16.38, -3.83] | 0.354 | +0.000 [+0.000, +0.000] | 0.396 |

Coverage asks whether the resampled next population contains any successful candidate. Model-refit uses the unchanged world model to select its top-k mean; oracle-refit is a ceiling.

## Linear dose slopes

| metric | slope per alpha | 95% paired bootstrap CI |
|---|---:|---:|
| `coverage` | +0.0000 | [+0.0000, +0.0000] |
| `min_true` | -8.9403 | [-15.6638, -3.6052] |
| `mean_success` | +0.0500 | [+0.0000, +0.1250] |
| `model_refit_true` | -9.2839 | [-15.4886, -3.8323] |
| `model_refit_success` | +0.0000 | [+0.0000, +0.0000] |
| `oracle_refit_true` | -13.7866 | [-21.3420, -6.8711] |
| `oracle_refit_success` | +0.0333 | [+0.0000, +0.0833] |

Negative slopes are favorable for true-cost metrics; positive slopes are favorable for success metrics.

## Integrity

- State mismatch: `0.000e+00`
- Goal mismatch: `0.000e+00`
- Action roundtrip error: `9.537e-07`
