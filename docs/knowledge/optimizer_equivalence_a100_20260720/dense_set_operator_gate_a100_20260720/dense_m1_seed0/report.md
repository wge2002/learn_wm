# Set-valued optimizer operator probe

Top1 is a learned single route. Top2/all-mode rows use oracle selection only to measure retained branch coverage.

- Cell: `h5_off40`
- Features: `planner_dense`
- Retained modes including no-op: `1`

| output | metric | baseline | corrected | delta | paired 95% CI |
|---|---|---:|---:|---:|---:|
| top1 | update_cosine | 0.181 | 0.187 | +0.006 | [-0.012, +0.024] |
| top1 | relative_update_error | 1.141 | 1.124 | -0.017 | [-0.031, -0.003] |
| top2_coverage | update_cosine | 0.181 | 0.187 | +0.006 | [-0.012, +0.024] |
| top2_coverage | relative_update_error | 1.141 | 1.124 | -0.017 | [-0.031, -0.004] |
| all_mode_coverage | update_cosine | 0.181 | 0.187 | +0.006 | [-0.012, +0.024] |
| all_mode_coverage | relative_update_error | 1.141 | 1.124 | -0.017 | [-0.031, -0.004] |

- Top1 exact best-mode rate: `1.000`
- Top2 contains best-mode rate: `1.000`
