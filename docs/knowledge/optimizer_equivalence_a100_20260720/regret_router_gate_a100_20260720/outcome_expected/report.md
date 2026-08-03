# Set-valued optimizer operator probe

Top1 is a learned single route. Top2/all-mode rows use oracle selection only to measure retained branch coverage.

- Cell: `h5_off40`
- Features: `planner_outcome`
- Retained modes including no-op: `5`

| output | metric | baseline | corrected | delta | paired 95% CI |
|---|---|---:|---:|---:|---:|
| top1 | update_cosine | 0.181 | 0.185 | +0.004 | [-0.005, +0.013] |
| top1 | relative_update_error | 1.141 | 1.099 | -0.042 | [-0.049, -0.036] |
| top2_coverage | update_cosine | 0.181 | 0.223 | +0.042 | [+0.030, +0.053] |
| top2_coverage | relative_update_error | 1.141 | 1.077 | -0.064 | [-0.072, -0.056] |
| all_mode_coverage | update_cosine | 0.181 | 0.304 | +0.123 | [+0.110, +0.135] |
| all_mode_coverage | relative_update_error | 1.141 | 1.028 | -0.112 | [-0.121, -0.104] |

- Top1 exact best-mode rate: `0.042`
- Top2 contains best-mode rate: `0.272`
