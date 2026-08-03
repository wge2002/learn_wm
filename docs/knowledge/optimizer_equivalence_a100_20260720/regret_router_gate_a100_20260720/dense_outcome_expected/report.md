# Set-valued optimizer operator probe

Top1 is a learned single route. Top2/all-mode rows use oracle selection only to measure retained branch coverage.

- Cell: `h5_off40`
- Features: `planner_dense_outcome`
- Retained modes including no-op: `5`

| output | metric | baseline | corrected | delta | paired 95% CI |
|---|---|---:|---:|---:|---:|
| top1 | update_cosine | 0.181 | 0.180 | -0.001 | [-0.013, +0.011] |
| top1 | relative_update_error | 1.141 | 1.122 | -0.018 | [-0.027, -0.010] |
| top2_coverage | update_cosine | 0.181 | 0.225 | +0.044 | [+0.030, +0.060] |
| top2_coverage | relative_update_error | 1.141 | 1.096 | -0.044 | [-0.055, -0.035] |
| all_mode_coverage | update_cosine | 0.181 | 0.318 | +0.137 | [+0.120, +0.154] |
| all_mode_coverage | relative_update_error | 1.141 | 1.044 | -0.097 | [-0.108, -0.086] |

- Top1 exact best-mode rate: `0.167`
- Top2 contains best-mode rate: `0.374`
