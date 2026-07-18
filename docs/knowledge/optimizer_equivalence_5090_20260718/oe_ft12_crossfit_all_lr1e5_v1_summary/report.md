# OE fixed-trace state cross-fit

- Folds: 3
- Unique held-out states: 12
- Trainable modules: `action_encoder, pred_proj, predictor`
- Selected source steps: `[4, 9, 19, 29]`

Each row pools predictions made by a model that did not train on that row’s held-out states. This is a feasibility diagnostic, not a deployable single-checkpoint or closed-loop MPC result.

| epoch | update cosine | Δ cosine | relative error | Δ rel. error | elite overlap | Δ overlap | selected-elite true cost | Δ true cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.121 | +0.000 | 1.201 | +0.000 | 0.147 | +0.000 | 107.41 | +0.00 |
| 1 | 0.078 | -0.044 | 1.270 | +0.070 | 0.122 | -0.025 | 108.79 | +1.38 |
| 2 | 0.045 | -0.076 | 1.304 | +0.103 | 0.122 | -0.026 | 109.52 | +2.11 |
| 3 | 0.042 | -0.079 | 1.322 | +0.121 | 0.117 | -0.031 | 110.11 | +2.70 |
| 4 | 0.037 | -0.084 | 1.351 | +0.151 | 0.125 | -0.022 | 109.59 | +2.18 |
| 5 | 0.053 | -0.068 | 1.350 | +0.150 | 0.117 | -0.030 | 110.15 | +2.74 |
