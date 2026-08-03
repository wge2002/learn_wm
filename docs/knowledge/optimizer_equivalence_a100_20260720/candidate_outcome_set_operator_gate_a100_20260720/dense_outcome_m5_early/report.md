# Set-valued optimizer operator probe

Top1 is a learned single route. Top2/all-mode rows use oracle selection only to measure retained branch coverage.

- Cell: `h5_off40`
- Features: `planner_dense_outcome`
- Retained modes including no-op: `5`

| output | metric | baseline | corrected | delta | paired 95% CI |
|---|---|---:|---:|---:|---:|
| top1 | update_cosine | 0.181 | 0.201 | +0.020 | [+0.005, +0.035] |
| top1 | relative_update_error | 1.141 | 1.129 | -0.012 | [-0.023, -0.001] |
| top2_coverage | update_cosine | 0.181 | 0.254 | +0.072 | [+0.058, +0.088] |
| top2_coverage | relative_update_error | 1.141 | 1.087 | -0.054 | [-0.064, -0.044] |
| all_mode_coverage | update_cosine | 0.181 | 0.318 | +0.136 | [+0.122, +0.152] |
| all_mode_coverage | relative_update_error | 1.141 | 1.042 | -0.099 | [-0.108, -0.089] |

- Top1 exact best-mode rate: `0.207`
- Top2 contains best-mode rate: `0.411`
