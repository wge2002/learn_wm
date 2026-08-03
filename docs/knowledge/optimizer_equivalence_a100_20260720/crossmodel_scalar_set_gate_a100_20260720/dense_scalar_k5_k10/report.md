# Set-valued optimizer operator probe

Top1 is a learned single route. Top2/all-mode rows use oracle selection only to measure retained branch coverage.

- Cell: `h5_off40`
- Features: `planner_dense_outcome`
- Retained modes including no-op: `5`

| output | metric | baseline | corrected | delta | paired 95% CI |
|---|---|---:|---:|---:|---:|
| top1 | update_cosine | 0.181 | 0.195 | +0.014 | [+0.002, +0.026] |
| top1 | relative_update_error | 1.141 | 1.100 | -0.040 | [-0.049, -0.032] |
| top2_coverage | update_cosine | 0.181 | 0.246 | +0.065 | [+0.053, +0.077] |
| top2_coverage | relative_update_error | 1.141 | 1.065 | -0.076 | [-0.084, -0.068] |
| all_mode_coverage | update_cosine | 0.181 | 0.297 | +0.116 | [+0.104, +0.128] |
| all_mode_coverage | relative_update_error | 1.141 | 1.032 | -0.109 | [-0.117, -0.102] |

- Top1 exact best-mode rate: `0.256`
- Top2 contains best-mode rate: `0.470`
