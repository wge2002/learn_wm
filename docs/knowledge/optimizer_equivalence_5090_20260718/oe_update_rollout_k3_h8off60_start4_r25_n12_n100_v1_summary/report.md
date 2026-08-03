# Recursive OE update intervention

- Generator: `pd_d192_k3_eval`
- Scorer: `pd_d192_k3_eval`
- Paired states: 12
- Start after source CEM step: 4
- Counterfactual rounds: 25
- Candidates per branch round: 100

| alpha | avg coverage | last coverage | last min true | final mean true | Δ final true | final mean success | Δ final success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.443 | 0.500 | 73.12 | 97.27 | +0.00 [+0.00, +0.00] | 0.250 | +0.000 [+0.000, +0.000] |
| 1.00 | 0.487 | 0.583 | 44.02 | 47.94 | -49.32 [-101.39, -14.24] | 0.500 | +0.250 [+0.000, +0.500] |

## Linear dose slopes

| metric | slope per alpha | 95% paired bootstrap CI |
|---|---:|---:|
| `average_coverage` | +0.0433 | [-0.0200, +0.1267] |
| `last_coverage` | +0.0833 | [+0.0000, +0.2500] |
| `last_min_true` | -29.1036 | [-65.8697, -6.8786] |
| `average_min_true` | -17.2224 | [-34.1339, -4.5137] |
| `mean_success` | +0.1533 | [+0.0233, +0.3300] |
| `final_mean_true` | -49.3244 | [-101.3930, -14.2392] |
| `final_mean_success` | +0.2500 | [+0.0000, +0.5000] |

Negative slopes are favorable for true-cost metrics; positive slopes are favorable for success metrics.

## Integrity

- State mismatch: `0.000e+00`
- Goal mismatch: `0.000e+00`
- Candidate quantization: `float16`, max error `3.902e-03`
- Action roundtrip error: `9.537e-07`
