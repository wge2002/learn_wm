# Structured counterfactual dynamics-head probe

Training epoch and frozen/structured cost fusion are selected on inner held-out states. Every reported prediction is from an outer state-held-out model.

- Cell: `h5_off40`
- Context: `dense_moment`
- Supervision: `relative`

| metric | frozen LeWM | structured | delta | paired 95% CI |
|---|---:|---:|---:|---:|
| update_cosine | 0.181 | 0.180 | -0.001 | [-0.037, +0.034] |
| relative_update_error | 1.141 | 1.288 | +0.147 | [+0.106, +0.190] |
| elite_overlap | 0.190 | 0.177 | -0.013 | [-0.032, +0.007] |
| selected_elite_true_cost | 100.883 | 102.899 | +2.016 | [+0.279, +3.721] |
