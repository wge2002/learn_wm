# Recursive OE update intervention

- Generator: `pd_d192_k3_eval`
- Scorer: `pd_d192_k3_eval`
- Paired states: 12
- Start after source CEM step: 4
- Counterfactual rounds: 25
- Candidates per branch round: 100

| alpha | avg coverage | last coverage | last min true | final mean true | Δ final true | final mean success | Δ final success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.303 | 0.417 | 86.59 | 105.72 | +0.00 [+0.00, +0.00] | 0.250 | +0.000 [+0.000, +0.000] |
| 0.50 | 0.410 | 0.583 | 35.29 | 45.29 | -60.42 [-115.65, -14.06] | 0.583 | +0.333 [+0.083, +0.583] |
| 1.00 | 0.447 | 0.583 | 34.12 | 38.22 | -67.50 [-123.62, -20.26] | 0.583 | +0.333 [+0.083, +0.583] |

## Linear dose slopes

| metric | slope per alpha | 95% paired bootstrap CI |
|---|---:|---:|
| `average_coverage` | +0.1433 | [+0.0133, +0.3033] |
| `last_coverage` | +0.1667 | [+0.0000, +0.4167] |
| `last_min_true` | -52.4760 | [-104.5200, -10.3257] |
| `average_min_true` | -44.5317 | [-92.6218, -7.5192] |
| `mean_success` | +0.1567 | [+0.0467, +0.2800] |
| `final_mean_true` | -67.4987 | [-123.6158, -20.2574] |
| `final_mean_success` | +0.3333 | [+0.0833, +0.5833] |

Negative slopes are favorable for true-cost metrics; positive slopes are favorable for success metrics.

## Integrity

- State mismatch: `0.000e+00`
- Goal mismatch: `0.000e+00`
- Candidate quantization: `float16`, max error `3.906e-03`
- Action roundtrip error: `9.537e-07`
