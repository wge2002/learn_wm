# Set-valued optimizer operator probe

Top1 is a learned single route. Top2/all-mode rows use oracle selection only to measure retained branch coverage.

- Cell: `h5_off40`
- Features: `planner_outcome`
- Retained modes including no-op: `1`

| output | metric | baseline | corrected | delta | paired 95% CI |
|---|---|---:|---:|---:|---:|
| top1 | update_cosine | 0.181 | 0.218 | +0.037 | [+0.019, +0.055] |
| top1 | relative_update_error | 1.141 | 1.096 | -0.045 | [-0.058, -0.033] |
| top2_coverage | update_cosine | 0.181 | 0.218 | +0.037 | [+0.019, +0.055] |
| top2_coverage | relative_update_error | 1.141 | 1.096 | -0.045 | [-0.058, -0.033] |
| all_mode_coverage | update_cosine | 0.181 | 0.218 | +0.037 | [+0.019, +0.055] |
| all_mode_coverage | relative_update_error | 1.141 | 1.096 | -0.045 | [-0.058, -0.033] |

- Top1 exact best-mode rate: `1.000`
- Top2 contains best-mode rate: `1.000`
