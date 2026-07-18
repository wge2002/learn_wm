# Recursive OE update intervention

- Generator: `pd_d192_k3_eval`
- Paired states: 12
- Start after source CEM step: 4
- Counterfactual rounds: 5
- Candidates per branch round: 100

| alpha | avg coverage | last coverage | last min true | final mean true | Δ final true | final mean success | Δ final success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.250 | 0.250 | 90.34 | 118.66 | +0.00 [+0.00, +0.00] | 0.167 | +0.000 [+0.000, +0.000] |
| 0.25 | 0.250 | 0.250 | 83.16 | 110.23 | -8.43 [-18.65, -0.58] | 0.167 | +0.000 [+0.000, +0.000] |
| 0.50 | 0.250 | 0.250 | 66.83 | 93.86 | -24.79 [-44.39, -8.90] | 0.250 | +0.083 [+0.000, +0.250] |
| 0.75 | 0.267 | 0.333 | 62.92 | 73.46 | -45.19 [-80.42, -15.55] | 0.250 | +0.083 [+0.000, +0.250] |
| 1.00 | 0.267 | 0.250 | 55.89 | 66.66 | -52.00 [-94.30, -17.60] | 0.250 | +0.083 [+0.000, +0.250] |

## Linear dose slopes

| metric | slope per alpha | 95% paired bootstrap CI |
|---|---:|---:|
| `average_coverage` | +0.0200 | [+0.0000, +0.0600] |
| `last_coverage` | +0.0333 | [+0.0000, +0.1000] |
| `last_min_true` | -35.6589 | [-74.4278, -4.7025] |
| `average_min_true` | -19.7335 | [-40.0031, -3.1179] |
| `mean_success` | +0.0467 | [+0.0000, +0.1400] |
| `final_mean_true` | -56.3060 | [-101.2124, -19.3184] |
| `final_mean_success` | +0.1000 | [+0.0000, +0.3000] |

Negative slopes are favorable for true-cost metrics; positive slopes are favorable for success metrics.

## Integrity

- State mismatch: `0.000e+00`
- Goal mismatch: `0.000e+00`
- Action roundtrip error: `9.537e-07`
