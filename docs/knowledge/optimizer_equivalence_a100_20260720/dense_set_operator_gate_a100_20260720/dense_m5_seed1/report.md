# Set-valued optimizer operator probe

Top1 is a learned single route. Top2/all-mode rows use oracle selection only to measure retained branch coverage.

- Cell: `h5_off40`
- Features: `planner_dense`
- Retained modes including no-op: `5`

| output | metric | baseline | corrected | delta | paired 95% CI |
|---|---|---:|---:|---:|---:|
| top1 | update_cosine | 0.181 | 0.189 | +0.007 | [-0.005, +0.020] |
| top1 | relative_update_error | 1.141 | 1.102 | -0.039 | [-0.047, -0.030] |
| top2_coverage | update_cosine | 0.181 | 0.240 | +0.058 | [+0.046, +0.070] |
| top2_coverage | relative_update_error | 1.141 | 1.068 | -0.073 | [-0.081, -0.065] |
| all_mode_coverage | update_cosine | 0.181 | 0.298 | +0.116 | [+0.105, +0.128] |
| all_mode_coverage | relative_update_error | 1.141 | 1.029 | -0.112 | [-0.119, -0.104] |

- Top1 exact best-mode rate: `0.239`
- Top2 contains best-mode rate: `0.442`
