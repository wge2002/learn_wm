# Set-valued optimizer operator probe

Top1 is a learned single route. Top2/all-mode rows use oracle selection only to measure retained branch coverage.

- Cell: `h5_off40`
- Features: `planner_dense_outcome`
- Retained modes including no-op: `5`

| output | metric | baseline | corrected | delta | paired 95% CI |
|---|---|---:|---:|---:|---:|
| top1 | update_cosine | 0.181 | 0.206 | +0.025 | [+0.011, +0.039] |
| top1 | relative_update_error | 1.141 | 1.088 | -0.052 | [-0.062, -0.043] |
| top2_coverage | update_cosine | 0.181 | 0.247 | +0.066 | [+0.053, +0.079] |
| top2_coverage | relative_update_error | 1.141 | 1.060 | -0.081 | [-0.090, -0.073] |
| all_mode_coverage | update_cosine | 0.181 | 0.296 | +0.115 | [+0.103, +0.127] |
| all_mode_coverage | relative_update_error | 1.141 | 1.029 | -0.112 | [-0.120, -0.105] |

- Top1 exact best-mode rate: `0.286`
- Top2 contains best-mode rate: `0.469`
