# Candidate-level OE cost-head probe

Architecture, epoch, and residual blend are selected on inner state-held-out data. Metrics pool one outer-fold prediction per state.

- Cell: `h5_off40`
- Features: `planner_outcome`
- Hidden width: `128`

| metric | baseline | corrected | delta | paired 95% CI |
|---|---:|---:|---:|---:|
| update_cosine | 0.181 | 0.203 | +0.022 | [-0.005, +0.048] |
| relative_update_error | 1.141 | 1.194 | +0.054 | [+0.029, +0.077] |
| elite_overlap | 0.190 | 0.205 | +0.015 | [+0.001, +0.030] |
| selected_elite_true_cost | 100.883 | 100.668 | -0.215 | [-1.343, +0.837] |
