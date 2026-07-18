# End-to-end CEM cross-validation audit

- Source: `/mnt/data/wge/stablewm/checkpoints/cemcv_*_h5_off40*_5090.txt`
- Paired bootstrap: 20,000 resamples

| seed | H | offset | variant | success | baseline | delta (pp) | 95% CI | wins/losses |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 7 | 5 | 40 | `k3_multistart_equalcompute3x100` | 62.0% | 68.0% | -6.0 | [-20.0, +8.0] | 5/8 |
| 7 | 5 | 40 | `k3_s100` | 68.0% | 68.0% | +0.0 | [-12.0, +12.0] | 5/5 |
| 7 | 5 | 40 | `k3_s900` | 60.0% | 68.0% | -8.0 | [-20.0, +4.0] | 3/7 |
| 7 | 5 | 40 | `portfolio_equalcompute3x100` | 74.0% | 68.0% | +6.0 | [-8.0, +20.0] | 8/5 |
| 7 | 5 | 40 | `shared_rankensemble3x100` | 14.0% | 68.0% | -54.0 | [-68.0, -38.0] | 1/28 |
| 42 | 5 | 40 | `baseline_k10` | 52.0% | 62.0% | -10.0 | [-24.0, +4.0] | 5/10 |
| 42 | 5 | 40 | `baseline_k5` | 56.0% | 62.0% | -6.0 | [-20.0, +6.0] | 4/7 |
| 42 | 5 | 40 | `k3_multistart3x7` | 70.0% | 62.0% | +8.0 | [+2.0, +16.0] | 4/0 |
| 42 | 5 | 40 | `k3_multistart_equalcompute3x100` | 66.0% | 62.0% | +4.0 | [-6.0, +16.0] | 5/3 |
| 42 | 5 | 40 | `k3_s100` | 58.0% | 62.0% | -4.0 | [-14.0, +6.0] | 2/4 |
| 42 | 5 | 40 | `k3_s900` | 72.0% | 62.0% | +10.0 | [+2.0, +18.0] | 5/0 |
| 42 | 5 | 40 | `k3_to_k10_finalpop` | 58.0% | 62.0% | -4.0 | [-10.0, +0.0] | 0/2 |
| 42 | 5 | 40 | `k3_to_k10_means` | 58.0% | 62.0% | -4.0 | [-10.0, +0.0] | 0/2 |
| 42 | 5 | 40 | `k3_to_k10_refit10` | 58.0% | 62.0% | -4.0 | [-10.0, +0.0] | 0/2 |
| 42 | 5 | 40 | `k3_to_k10_refit30` | 60.0% | 62.0% | -2.0 | [-6.0, +0.0] | 0/1 |
| 42 | 5 | 40 | `k3_to_k10_refit5` | 60.0% | 62.0% | -2.0 | [-6.0, +0.0] | 0/1 |
| 42 | 5 | 40 | `k3_to_k3_finalpop` | 60.0% | 62.0% | -2.0 | [-6.0, +0.0] | 0/1 |
| 42 | 5 | 40 | `k3_to_k3_means` | 62.0% | 62.0% | +0.0 | [+0.0, +0.0] | 0/0 |
| 42 | 5 | 40 | `k3_to_k3_refit30` | 62.0% | 62.0% | +0.0 | [+0.0, +0.0] | 0/0 |
| 42 | 5 | 40 | `k3_to_k5_finalpop` | 56.0% | 62.0% | -6.0 | [-14.0, +0.0] | 0/3 |
| 42 | 5 | 40 | `k3_to_k5_means` | 60.0% | 62.0% | -2.0 | [-6.0, +0.0] | 0/1 |
| 42 | 5 | 40 | `k3_to_k5_refit30` | 56.0% | 62.0% | -6.0 | [-14.0, +0.0] | 0/3 |
| 42 | 5 | 40 | `portfolio_equalcompute3x100` | 66.0% | 62.0% | +4.0 | [-8.0, +16.0] | 6/4 |
| 42 | 5 | 40 | `portfolio_rank3x1` | 66.0% | 62.0% | +4.0 | [-6.0, +16.0] | 5/3 |
| 42 | 5 | 40 | `portfolio_rank3x7` | 72.0% | 62.0% | +10.0 | [+2.0, +18.0] | 5/0 |
| 42 | 5 | 40 | `shared_rankensemble3x100` | 6.0% | 62.0% | -56.0 | [-70.0, -42.0] | 0/28 |
| 123 | 5 | 40 | `k3_multistart_equalcompute3x100` | 60.0% | 58.0% | +2.0 | [-10.0, +14.0] | 5/4 |
| 123 | 5 | 40 | `k3_s100` | 52.0% | 58.0% | -6.0 | [-18.0, +6.0] | 3/6 |
| 123 | 5 | 40 | `k3_s900` | 66.0% | 58.0% | +8.0 | [+0.0, +18.0] | 5/1 |
| 123 | 5 | 40 | `portfolio_equalcompute3x100` | 58.0% | 58.0% | +0.0 | [-12.0, +12.0] | 5/5 |
| 123 | 5 | 40 | `shared_rankensemble3x100` | 4.0% | 58.0% | -54.0 | [-68.0, -40.0] | 1/28 |

## Pooled multi-seed comparisons

| H | offset | variant | seeds | success | baseline | delta (pp) | 95% CI | seed range | wins/losses |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | 40 | `k3_multistart_equalcompute3x100` | 3 | 62.7% | 62.7% | +0.0 | [-7.3, +7.3] | [-6.0, +4.0] | 15/15 |
| 5 | 40 | `k3_s100` | 3 | 59.3% | 62.7% | -3.3 | [-10.0, +3.3] | [-6.0, +0.0] | 10/15 |
| 5 | 40 | `k3_s900` | 3 | 66.0% | 62.7% | +3.3 | [-2.7, +9.3] | [-8.0, +10.0] | 13/8 |
| 5 | 40 | `portfolio_equalcompute3x100` | 3 | 66.0% | 62.7% | +3.3 | [-4.0, +10.7] | [+0.0, +6.0] | 19/14 |
| 5 | 40 | `shared_rankensemble3x100` | 3 | 8.0% | 62.7% | -54.7 | [-62.7, -46.0] | [-56.0, -54.0] | 2/84 |

Wins/losses count paired episodes that flip relative to the configured baseline. Pooled confidence intervals resample episodes within the available equal-sized seed cells; the seed range is also shown because three seeds do not support a stable between-seed variance estimate. These are exploratory comparisons and were not adjusted for multiple testing.
