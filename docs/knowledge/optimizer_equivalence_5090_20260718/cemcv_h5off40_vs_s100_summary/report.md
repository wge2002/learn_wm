# End-to-end CEM cross-validation audit

- Source: `/mnt/data/wge/stablewm/checkpoints/cemcv_*_h5_off40*_5090.txt`
- Paired bootstrap: 20,000 resamples

| seed | H | offset | variant | success | baseline | delta (pp) | 95% CI | wins/losses |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 7 | 5 | 40 | `baseline_k3` | 68.0% | 68.0% | +0.0 | [-12.0, +12.0] | 5/5 |
| 7 | 5 | 40 | `k3_multistart_equalcompute3x100` | 62.0% | 68.0% | -6.0 | [-18.0, +6.0] | 3/6 |
| 7 | 5 | 40 | `k3_s900` | 60.0% | 68.0% | -8.0 | [-26.0, +10.0] | 8/12 |
| 7 | 5 | 40 | `portfolio_equalcompute3x100` | 74.0% | 68.0% | +6.0 | [-6.0, +18.0] | 6/3 |
| 7 | 5 | 40 | `shared_rankensemble3x100` | 14.0% | 68.0% | -54.0 | [-68.0, -40.0] | 0/27 |
| 42 | 5 | 40 | `baseline_k10` | 52.0% | 58.0% | -6.0 | [-20.0, +8.0] | 5/8 |
| 42 | 5 | 40 | `baseline_k3` | 62.0% | 58.0% | +4.0 | [-6.0, +14.0] | 4/2 |
| 42 | 5 | 40 | `baseline_k5` | 56.0% | 58.0% | -2.0 | [-16.0, +12.0] | 6/7 |
| 42 | 5 | 40 | `k3_multistart3x7` | 70.0% | 58.0% | +12.0 | [+2.0, +24.0] | 7/1 |
| 42 | 5 | 40 | `k3_multistart_equalcompute3x100` | 66.0% | 58.0% | +8.0 | [+0.0, +18.0] | 5/1 |
| 42 | 5 | 40 | `k3_s900` | 72.0% | 58.0% | +14.0 | [+4.0, +26.0] | 8/1 |
| 42 | 5 | 40 | `k3_to_k10_finalpop` | 58.0% | 58.0% | +0.0 | [-10.0, +10.0] | 3/3 |
| 42 | 5 | 40 | `k3_to_k10_means` | 58.0% | 58.0% | +0.0 | [-10.0, +10.0] | 4/4 |
| 42 | 5 | 40 | `k3_to_k10_refit10` | 58.0% | 58.0% | +0.0 | [-10.0, +10.0] | 3/3 |
| 42 | 5 | 40 | `k3_to_k10_refit30` | 60.0% | 58.0% | +2.0 | [-6.0, +12.0] | 3/2 |
| 42 | 5 | 40 | `k3_to_k10_refit5` | 60.0% | 58.0% | +2.0 | [-6.0, +10.0] | 3/2 |
| 42 | 5 | 40 | `k3_to_k3_finalpop` | 60.0% | 58.0% | +2.0 | [-8.0, +12.0] | 4/3 |
| 42 | 5 | 40 | `k3_to_k3_means` | 62.0% | 58.0% | +4.0 | [-6.0, +14.0] | 4/2 |
| 42 | 5 | 40 | `k3_to_k3_refit30` | 62.0% | 58.0% | +4.0 | [-6.0, +14.0] | 4/2 |
| 42 | 5 | 40 | `k3_to_k5_finalpop` | 56.0% | 58.0% | -2.0 | [-14.0, +10.0] | 4/5 |
| 42 | 5 | 40 | `k3_to_k5_means` | 60.0% | 58.0% | +2.0 | [-8.0, +12.0] | 4/3 |
| 42 | 5 | 40 | `k3_to_k5_refit30` | 56.0% | 58.0% | -2.0 | [-12.0, +8.0] | 3/4 |
| 42 | 5 | 40 | `portfolio_equalcompute3x100` | 66.0% | 58.0% | +8.0 | [+0.0, +18.0] | 5/1 |
| 42 | 5 | 40 | `portfolio_rank3x1` | 66.0% | 58.0% | +8.0 | [-2.0, +20.0] | 6/2 |
| 42 | 5 | 40 | `portfolio_rank3x7` | 72.0% | 58.0% | +14.0 | [+4.0, +26.0] | 8/1 |
| 42 | 5 | 40 | `shared_rankensemble3x100` | 6.0% | 58.0% | -52.0 | [-66.0, -38.0] | 0/26 |
| 123 | 5 | 40 | `baseline_k3` | 58.0% | 52.0% | +6.0 | [-6.0, +18.0] | 6/3 |
| 123 | 5 | 40 | `k3_multistart_equalcompute3x100` | 60.0% | 52.0% | +8.0 | [-4.0, +20.0] | 7/3 |
| 123 | 5 | 40 | `k3_s900` | 66.0% | 52.0% | +14.0 | [+4.0, +26.0] | 8/1 |
| 123 | 5 | 40 | `portfolio_equalcompute3x100` | 58.0% | 52.0% | +6.0 | [-6.0, +18.0] | 6/3 |
| 123 | 5 | 40 | `shared_rankensemble3x100` | 4.0% | 52.0% | -48.0 | [-62.0, -34.0] | 1/25 |

## Pooled multi-seed comparisons

| H | offset | variant | seeds | success | baseline | delta (pp) | 95% CI | seed range | wins/losses |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | 40 | `baseline_k3` | 3 | 62.7% | 59.3% | +3.3 | [-3.3, +10.0] | [+0.0, +6.0] | 15/10 |
| 5 | 40 | `k3_multistart_equalcompute3x100` | 3 | 62.7% | 59.3% | +3.3 | [-3.3, +10.0] | [-6.0, +8.0] | 15/10 |
| 5 | 40 | `k3_s900` | 3 | 66.0% | 59.3% | +6.7 | [-1.3, +14.7] | [-8.0, +14.0] | 24/14 |
| 5 | 40 | `portfolio_equalcompute3x100` | 3 | 66.0% | 59.3% | +6.7 | [+0.7, +13.3] | [+6.0, +8.0] | 17/7 |
| 5 | 40 | `shared_rankensemble3x100` | 3 | 8.0% | 59.3% | -51.3 | [-59.3, -43.3] | [-54.0, -48.0] | 1/78 |

Wins/losses count paired episodes that flip relative to the configured baseline. Pooled confidence intervals resample episodes within the available equal-sized seed cells; the seed range is also shown because three seeds do not support a stable between-seed variance estimate. These are exploratory comparisons and were not adjusted for multiple testing.
