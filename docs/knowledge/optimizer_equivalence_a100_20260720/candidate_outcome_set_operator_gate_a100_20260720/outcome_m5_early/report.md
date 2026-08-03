# Set-valued optimizer operator probe

Top1 is a learned single route. Top2/all-mode rows use oracle selection only to measure retained branch coverage.

- Cell: `h5_off40`
- Features: `planner_outcome`
- Retained modes including no-op: `5`

| output | metric | baseline | corrected | delta | paired 95% CI |
|---|---|---:|---:|---:|---:|
| top1 | update_cosine | 0.181 | 0.207 | +0.025 | [+0.012, +0.039] |
| top1 | relative_update_error | 1.141 | 1.093 | -0.047 | [-0.056, -0.039] |
| top2_coverage | update_cosine | 0.181 | 0.259 | +0.078 | [+0.065, +0.092] |
| top2_coverage | relative_update_error | 1.141 | 1.058 | -0.083 | [-0.091, -0.075] |
| all_mode_coverage | update_cosine | 0.181 | 0.303 | +0.122 | [+0.110, +0.134] |
| all_mode_coverage | relative_update_error | 1.141 | 1.029 | -0.112 | [-0.119, -0.104] |

- Top1 exact best-mode rate: `0.250`
- Top2 contains best-mode rate: `0.501`
