# Structured counterfactual dynamics-head probe

Training epoch and frozen/structured cost fusion are selected on inner held-out states. Every reported prediction is from an outer state-held-out model.

- Cell: `h5_off40`
- Context: `state_oracle`
- Supervision: `relative`

| metric | frozen LeWM | structured | delta | paired 95% CI |
|---|---:|---:|---:|---:|
| update_cosine | 0.181 | 0.291 | +0.109 | [+0.068, +0.152] |
| relative_update_error | 1.141 | 1.230 | +0.090 | [+0.048, +0.131] |
| elite_overlap | 0.190 | 0.230 | +0.040 | [+0.016, +0.065] |
| selected_elite_true_cost | 100.883 | 98.957 | -1.926 | [-3.997, +0.132] |
