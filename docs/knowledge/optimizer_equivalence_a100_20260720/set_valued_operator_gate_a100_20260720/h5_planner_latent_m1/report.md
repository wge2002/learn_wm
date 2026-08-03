# Set-valued optimizer operator probe

Top1 is a learned single route. Top2/all-mode rows use oracle selection only to measure retained branch coverage.

- Cell: `h5_off40`
- Features: `planner_latent`
- Retained modes including no-op: `1`

| output | metric | baseline | corrected | delta | paired 95% CI |
|---|---|---:|---:|---:|---:|
| top1 | update_cosine | 0.181 | 0.210 | +0.028 | [+0.007, +0.050] |
| top1 | relative_update_error | 1.141 | 1.158 | +0.017 | [-0.001, +0.035] |
| top2_coverage | update_cosine | 0.181 | 0.210 | +0.028 | [+0.007, +0.050] |
| top2_coverage | relative_update_error | 1.141 | 1.158 | +0.017 | [-0.001, +0.035] |
| all_mode_coverage | update_cosine | 0.181 | 0.210 | +0.028 | [+0.007, +0.050] |
| all_mode_coverage | relative_update_error | 1.141 | 1.158 | +0.017 | [-0.001, +0.035] |

- Top1 exact best-mode rate: `1.000`
- Top2 contains best-mode rate: `1.000`
