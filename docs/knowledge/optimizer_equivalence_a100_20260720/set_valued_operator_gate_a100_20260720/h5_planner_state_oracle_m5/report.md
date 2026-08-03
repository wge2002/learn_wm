# Set-valued optimizer operator probe

Top1 is a learned single route. Top2/all-mode rows use oracle selection only to measure retained branch coverage.

- Cell: `h5_off40`
- Features: `planner_state_oracle`
- Retained modes including no-op: `5`

| output | metric | baseline | corrected | delta | paired 95% CI |
|---|---|---:|---:|---:|---:|
| top1 | update_cosine | 0.181 | 0.240 | +0.058 | [+0.036, +0.082] |
| top1 | relative_update_error | 1.141 | 1.108 | -0.032 | [-0.049, -0.016] |
| top2_coverage | update_cosine | 0.181 | 0.294 | +0.113 | [+0.091, +0.135] |
| top2_coverage | relative_update_error | 1.141 | 1.060 | -0.081 | [-0.095, -0.066] |
| all_mode_coverage | update_cosine | 0.181 | 0.356 | +0.174 | [+0.155, +0.194] |
| all_mode_coverage | relative_update_error | 1.141 | 1.013 | -0.128 | [-0.140, -0.116] |

- Top1 exact best-mode rate: `0.283`
- Top2 contains best-mode rate: `0.498`
