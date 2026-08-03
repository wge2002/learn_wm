# Set-valued optimizer operator probe

Top1 is a learned single route. Top2/all-mode rows use oracle selection only to measure retained branch coverage.

- Cell: `h5_off40`
- Features: `planner_outcome`
- Retained modes including no-op: `5`

| output | metric | baseline | corrected | delta | paired 95% CI |
|---|---|---:|---:|---:|---:|
| top1 | update_cosine | 0.181 | 0.208 | +0.027 | [+0.014, +0.041] |
| top1 | relative_update_error | 1.141 | 1.086 | -0.055 | [-0.063, -0.047] |
| top2_coverage | update_cosine | 0.181 | 0.248 | +0.067 | [+0.053, +0.080] |
| top2_coverage | relative_update_error | 1.141 | 1.058 | -0.083 | [-0.091, -0.075] |
| all_mode_coverage | update_cosine | 0.181 | 0.294 | +0.112 | [+0.100, +0.125] |
| all_mode_coverage | relative_update_error | 1.141 | 1.030 | -0.111 | [-0.119, -0.104] |

- Top1 exact best-mode rate: `0.243`
- Top2 contains best-mode rate: `0.456`
