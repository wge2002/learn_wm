# Structured counterfactual dynamics-head probe

Training epoch and frozen/structured cost fusion are selected on inner held-out states. Every reported prediction is from an outer state-held-out model.

- Cell: `h5_off40`
- Context: `latent`
- Supervision: `terminal`

| metric | frozen LeWM | structured | delta | paired 95% CI |
|---|---:|---:|---:|---:|
| update_cosine | 0.181 | 0.162 | -0.019 | [-0.055, +0.017] |
| relative_update_error | 1.141 | 1.308 | +0.168 | [+0.126, +0.211] |
| elite_overlap | 0.190 | 0.175 | -0.015 | [-0.034, +0.005] |
| selected_elite_true_cost | 100.883 | 102.858 | +1.975 | [+0.243, +3.694] |
