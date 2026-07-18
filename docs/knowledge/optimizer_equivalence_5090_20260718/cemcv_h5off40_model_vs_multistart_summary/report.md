# End-to-end CEM cross-validation audit

- Source: `/mnt/data/wge/stablewm/checkpoints/cemcv_*_h5_off40*_5090.txt`
- Paired bootstrap: 20,000 resamples

| seed | H | offset | variant | success | baseline | delta (pp) | 95% CI | wins/losses |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 7 | 5 | 40 | `baseline_k3` | 68.0% | 62.0% | +6.0 | [-8.0, +20.0] | 8/5 |
| 7 | 5 | 40 | `k3_s100` | 68.0% | 62.0% | +6.0 | [-6.0, +18.0] | 6/3 |
| 7 | 5 | 40 | `k3_s900` | 60.0% | 62.0% | -2.0 | [-18.0, +14.0] | 7/8 |
| 7 | 5 | 40 | `portfolio_equalcompute3x100` | 74.0% | 62.0% | +12.0 | [+0.0, +24.0] | 8/2 |
| 7 | 5 | 40 | `shared_rankensemble3x100` | 14.0% | 62.0% | -48.0 | [-62.0, -34.0] | 0/24 |
| 42 | 5 | 40 | `baseline_k10` | 52.0% | 66.0% | -14.0 | [-28.0, +0.0] | 4/11 |
| 42 | 5 | 40 | `baseline_k3` | 62.0% | 66.0% | -4.0 | [-14.0, +6.0] | 3/5 |
| 42 | 5 | 40 | `baseline_k5` | 56.0% | 66.0% | -10.0 | [-24.0, +4.0] | 4/9 |
| 42 | 5 | 40 | `k3_multistart3x7` | 70.0% | 66.0% | +4.0 | [-6.0, +14.0] | 4/2 |
| 42 | 5 | 40 | `k3_s100` | 58.0% | 66.0% | -8.0 | [-18.0, +0.0] | 1/5 |
| 42 | 5 | 40 | `k3_s900` | 72.0% | 66.0% | +6.0 | [-4.0, +16.0] | 5/2 |
| 42 | 5 | 40 | `k3_to_k10_finalpop` | 58.0% | 66.0% | -8.0 | [-20.0, +2.0] | 2/6 |
| 42 | 5 | 40 | `k3_to_k10_means` | 58.0% | 66.0% | -8.0 | [-20.0, +4.0] | 3/7 |
| 42 | 5 | 40 | `k3_to_k10_refit10` | 58.0% | 66.0% | -8.0 | [-18.0, +0.0] | 1/5 |
| 42 | 5 | 40 | `k3_to_k10_refit30` | 60.0% | 66.0% | -6.0 | [-16.0, +4.0] | 2/5 |
| 42 | 5 | 40 | `k3_to_k10_refit5` | 60.0% | 66.0% | -6.0 | [-16.0, +4.0] | 2/5 |
| 42 | 5 | 40 | `k3_to_k3_finalpop` | 60.0% | 66.0% | -6.0 | [-18.0, +6.0] | 3/6 |
| 42 | 5 | 40 | `k3_to_k3_means` | 62.0% | 66.0% | -4.0 | [-16.0, +6.0] | 3/5 |
| 42 | 5 | 40 | `k3_to_k3_refit30` | 62.0% | 66.0% | -4.0 | [-16.0, +6.0] | 3/5 |
| 42 | 5 | 40 | `k3_to_k5_finalpop` | 56.0% | 66.0% | -10.0 | [-22.0, +2.0] | 2/7 |
| 42 | 5 | 40 | `k3_to_k5_means` | 60.0% | 66.0% | -6.0 | [-18.0, +6.0] | 3/6 |
| 42 | 5 | 40 | `k3_to_k5_refit30` | 56.0% | 66.0% | -10.0 | [-20.0, +0.0] | 1/6 |
| 42 | 5 | 40 | `portfolio_equalcompute3x100` | 66.0% | 66.0% | +0.0 | [-8.0, +8.0] | 2/2 |
| 42 | 5 | 40 | `portfolio_rank3x1` | 66.0% | 66.0% | +0.0 | [-10.0, +10.0] | 3/3 |
| 42 | 5 | 40 | `portfolio_rank3x7` | 72.0% | 66.0% | +6.0 | [-2.0, +14.0] | 4/1 |
| 42 | 5 | 40 | `shared_rankensemble3x100` | 6.0% | 66.0% | -60.0 | [-74.0, -46.0] | 0/30 |
| 123 | 5 | 40 | `baseline_k3` | 58.0% | 60.0% | -2.0 | [-14.0, +10.0] | 4/5 |
| 123 | 5 | 40 | `k3_s100` | 52.0% | 60.0% | -8.0 | [-20.0, +4.0] | 3/7 |
| 123 | 5 | 40 | `k3_s900` | 66.0% | 60.0% | +6.0 | [-6.0, +18.0] | 6/3 |
| 123 | 5 | 40 | `portfolio_equalcompute3x100` | 58.0% | 60.0% | -2.0 | [-16.0, +10.0] | 5/6 |
| 123 | 5 | 40 | `shared_rankensemble3x100` | 4.0% | 60.0% | -56.0 | [-70.0, -42.0] | 0/28 |

## Pooled multi-seed comparisons

| H | offset | variant | seeds | success | baseline | delta (pp) | 95% CI | seed range | wins/losses |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | 40 | `baseline_k3` | 3 | 62.7% | 62.7% | +0.0 | [-7.3, +7.3] | [-4.0, +6.0] | 15/15 |
| 5 | 40 | `k3_s100` | 3 | 59.3% | 62.7% | -3.3 | [-10.0, +3.3] | [-8.0, +6.0] | 10/15 |
| 5 | 40 | `k3_s900` | 3 | 66.0% | 62.7% | +3.3 | [-4.0, +10.7] | [-2.0, +6.0] | 18/13 |
| 5 | 40 | `portfolio_equalcompute3x100` | 3 | 66.0% | 62.7% | +3.3 | [-3.3, +10.0] | [-2.0, +12.0] | 15/10 |
| 5 | 40 | `shared_rankensemble3x100` | 3 | 8.0% | 62.7% | -54.7 | [-62.7, -46.7] | [-60.0, -48.0] | 0/82 |

Wins/losses count paired episodes that flip relative to the configured baseline. Pooled confidence intervals resample episodes within the available equal-sized seed cells; the seed range is also shown because three seeds do not support a stable between-seed variance estimate. These are exploratory comparisons and were not adjusted for multiple testing.
