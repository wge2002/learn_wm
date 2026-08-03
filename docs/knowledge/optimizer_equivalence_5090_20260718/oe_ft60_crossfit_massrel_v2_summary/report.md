# OE fixed-trace state cross-fit

- Folds: 3
- Unique held-out states: 60
- Trainable modules: `action_encoder, pred_proj, predictor`
- Selected source steps: `[4, 9, 19, 29]`

Each row pools predictions made by a model that did not train on that row’s held-out states. This is a feasibility diagnostic, not a deployable single-checkpoint or closed-loop MPC result.

| epoch | update cosine | Δ cosine | relative error | Δ rel. error | elite overlap | Δ overlap | selected-elite true cost | Δ true cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.172 | +0.000 [+0.000, +0.000] | 1.152 | +0.000 [+0.000, +0.000] | 0.172 | +0.000 | 102.65 | +0.00 |
| 1 | 0.208 | +0.037 [+0.010, +0.063] | 1.138 | -0.014 [-0.034, +0.006] | 0.187 | +0.016 | 101.66 | -0.98 |
| 2 | 0.222 | +0.050 [+0.017, +0.084] | 1.135 | -0.017 [-0.043, +0.009] | 0.195 | +0.023 | 101.28 | -1.36 |
| 3 | 0.234 | +0.062 [+0.023, +0.102] | 1.135 | -0.017 [-0.046, +0.013] | 0.199 | +0.027 | 100.90 | -1.75 |
| 4 | 0.242 | +0.070 [+0.027, +0.114] | 1.143 | -0.009 [-0.044, +0.027] | 0.201 | +0.029 | 100.63 | -2.02 |
| 5 | 0.252 | +0.080 [+0.034, +0.128] | 1.147 | -0.005 [-0.043, +0.034] | 0.205 | +0.034 | 100.49 | -2.16 |
