# Candidate-level OE cost-head probe

Architecture, epoch, and residual blend are selected on inner state-held-out data. Metrics pool one outer-fold prediction per state.

- Cell: `h5_off40`
- Features: `planner_outcome`
- Hidden width: `128`

| metric | baseline | corrected | delta | paired 95% CI |
|---|---:|---:|---:|---:|
| update_cosine | 0.181 | 0.206 | +0.025 | [-0.001, +0.051] |
| relative_update_error | 1.141 | 1.196 | +0.055 | [+0.031, +0.079] |
| elite_overlap | 0.190 | 0.204 | +0.014 | [+0.000, +0.029] |
| selected_elite_true_cost | 100.883 | 101.172 | +0.289 | [-0.692, +1.280] |
