# OE fixed-trace state cross-fit

- Folds: 3
- Unique held-out states: 12
- Trainable modules: `pred_proj`
- Selected source steps: `[4, 9, 19, 29]`

Each row pools predictions made by a model that did not train on that row’s held-out states. This is a feasibility diagnostic, not a deployable single-checkpoint or closed-loop MPC result.

| epoch | update cosine | Δ cosine | relative error | Δ rel. error | elite overlap | Δ overlap | selected-elite true cost | Δ true cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.121 | +0.000 | 1.201 | +0.000 | 0.147 | +0.000 | 107.41 | +0.00 |
| 1 | 0.119 | -0.002 | 1.208 | +0.007 | 0.141 | -0.006 | 107.60 | +0.19 |
| 2 | 0.105 | -0.016 | 1.221 | +0.020 | 0.141 | -0.006 | 107.79 | +0.37 |
| 3 | 0.102 | -0.019 | 1.224 | +0.023 | 0.141 | -0.006 | 107.87 | +0.46 |
| 4 | 0.093 | -0.028 | 1.227 | +0.027 | 0.139 | -0.008 | 107.96 | +0.55 |
| 5 | 0.087 | -0.034 | 1.239 | +0.038 | 0.134 | -0.013 | 108.06 | +0.65 |
| 6 | 0.093 | -0.028 | 1.231 | +0.031 | 0.136 | -0.011 | 108.12 | +0.71 |
| 7 | 0.083 | -0.039 | 1.241 | +0.041 | 0.136 | -0.011 | 108.23 | +0.82 |
| 8 | 0.083 | -0.038 | 1.247 | +0.047 | 0.133 | -0.014 | 108.38 | +0.96 |
| 9 | 0.078 | -0.043 | 1.252 | +0.051 | 0.133 | -0.015 | 108.36 | +0.94 |
| 10 | 0.075 | -0.046 | 1.259 | +0.058 | 0.132 | -0.015 | 108.37 | +0.96 |
