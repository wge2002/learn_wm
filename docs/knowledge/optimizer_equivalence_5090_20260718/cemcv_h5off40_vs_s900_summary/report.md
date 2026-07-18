# End-to-end CEM cross-validation audit

- Source: `/mnt/data/wge/stablewm/checkpoints/cemcv_*_h5_off40*_5090.txt`
- Paired bootstrap: 20,000 resamples

| seed | H | offset | variant | success | baseline | delta (pp) | 95% CI | wins/losses |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 7 | 5 | 40 | `baseline_k3` | 68.0% | 60.0% | +8.0 | [-4.0, +20.0] | 7/3 |
| 7 | 5 | 40 | `k3_multistart_equalcompute3x100` | 62.0% | 60.0% | +2.0 | [-14.0, +18.0] | 8/7 |
| 7 | 5 | 40 | `k3_s100` | 68.0% | 60.0% | +8.0 | [-10.0, +26.0] | 12/8 |
| 7 | 5 | 40 | `portfolio_equalcompute3x100` | 74.0% | 60.0% | +14.0 | [-2.0, +30.0] | 12/5 |
| 7 | 5 | 40 | `shared_rankensemble3x100` | 14.0% | 60.0% | -46.0 | [-60.0, -32.0] | 1/24 |
| 42 | 5 | 40 | `baseline_k10` | 52.0% | 72.0% | -20.0 | [-34.0, -6.0] | 2/12 |
| 42 | 5 | 40 | `baseline_k3` | 62.0% | 72.0% | -10.0 | [-18.0, -2.0] | 0/5 |
| 42 | 5 | 40 | `baseline_k5` | 56.0% | 72.0% | -16.0 | [-30.0, -4.0] | 2/10 |
| 42 | 5 | 40 | `k3_multistart3x7` | 70.0% | 72.0% | -2.0 | [-8.0, +4.0] | 1/2 |
| 42 | 5 | 40 | `k3_multistart_equalcompute3x100` | 66.0% | 72.0% | -6.0 | [-16.0, +4.0] | 2/5 |
| 42 | 5 | 40 | `k3_s100` | 58.0% | 72.0% | -14.0 | [-26.0, -4.0] | 1/8 |
| 42 | 5 | 40 | `k3_to_k10_finalpop` | 58.0% | 72.0% | -14.0 | [-24.0, -6.0] | 0/7 |
| 42 | 5 | 40 | `k3_to_k10_means` | 58.0% | 72.0% | -14.0 | [-24.0, -6.0] | 0/7 |
| 42 | 5 | 40 | `k3_to_k10_refit10` | 58.0% | 72.0% | -14.0 | [-24.0, -6.0] | 0/7 |
| 42 | 5 | 40 | `k3_to_k10_refit30` | 60.0% | 72.0% | -12.0 | [-22.0, -4.0] | 0/6 |
| 42 | 5 | 40 | `k3_to_k10_refit5` | 60.0% | 72.0% | -12.0 | [-22.0, -4.0] | 0/6 |
| 42 | 5 | 40 | `k3_to_k3_finalpop` | 60.0% | 72.0% | -12.0 | [-22.0, -4.0] | 0/6 |
| 42 | 5 | 40 | `k3_to_k3_means` | 62.0% | 72.0% | -10.0 | [-18.0, -2.0] | 0/5 |
| 42 | 5 | 40 | `k3_to_k3_refit30` | 62.0% | 72.0% | -10.0 | [-18.0, -2.0] | 0/5 |
| 42 | 5 | 40 | `k3_to_k5_finalpop` | 56.0% | 72.0% | -16.0 | [-26.0, -6.0] | 0/8 |
| 42 | 5 | 40 | `k3_to_k5_means` | 60.0% | 72.0% | -12.0 | [-22.0, -4.0] | 0/6 |
| 42 | 5 | 40 | `k3_to_k5_refit30` | 56.0% | 72.0% | -16.0 | [-26.0, -6.0] | 0/8 |
| 42 | 5 | 40 | `portfolio_equalcompute3x100` | 66.0% | 72.0% | -6.0 | [-18.0, +6.0] | 3/6 |
| 42 | 5 | 40 | `portfolio_rank3x1` | 66.0% | 72.0% | -6.0 | [-16.0, +4.0] | 2/5 |
| 42 | 5 | 40 | `portfolio_rank3x7` | 72.0% | 72.0% | +0.0 | [-8.0, +8.0] | 2/2 |
| 42 | 5 | 40 | `shared_rankensemble3x100` | 6.0% | 72.0% | -66.0 | [-78.0, -52.0] | 0/33 |
| 123 | 5 | 40 | `baseline_k3` | 58.0% | 66.0% | -8.0 | [-18.0, +0.0] | 1/5 |
| 123 | 5 | 40 | `k3_multistart_equalcompute3x100` | 60.0% | 66.0% | -6.0 | [-18.0, +6.0] | 3/6 |
| 123 | 5 | 40 | `k3_s100` | 52.0% | 66.0% | -14.0 | [-26.0, -4.0] | 1/8 |
| 123 | 5 | 40 | `portfolio_equalcompute3x100` | 58.0% | 66.0% | -8.0 | [-20.0, +2.0] | 2/6 |
| 123 | 5 | 40 | `shared_rankensemble3x100` | 4.0% | 66.0% | -62.0 | [-76.0, -48.0] | 1/32 |

## Pooled multi-seed comparisons

| H | offset | variant | seeds | success | baseline | delta (pp) | 95% CI | seed range | wins/losses |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | 40 | `baseline_k3` | 3 | 62.7% | 66.0% | -3.3 | [-9.3, +2.7] | [-10.0, +8.0] | 8/13 |
| 5 | 40 | `k3_multistart_equalcompute3x100` | 3 | 62.7% | 66.0% | -3.3 | [-10.7, +4.0] | [-6.0, +2.0] | 13/18 |
| 5 | 40 | `k3_s100` | 3 | 59.3% | 66.0% | -6.7 | [-14.7, +1.3] | [-14.0, +8.0] | 14/24 |
| 5 | 40 | `portfolio_equalcompute3x100` | 3 | 66.0% | 66.0% | +0.0 | [-7.3, +7.3] | [-8.0, +14.0] | 17/17 |
| 5 | 40 | `shared_rankensemble3x100` | 3 | 8.0% | 66.0% | -58.0 | [-66.0, -50.0] | [-66.0, -46.0] | 2/89 |

Wins/losses count paired episodes that flip relative to the configured baseline. Pooled confidence intervals resample episodes within the available equal-sized seed cells; the seed range is also shown because three seeds do not support a stable between-seed variance estimate. These are exploratory comparisons and were not adjusted for multiple testing.
