# CEM round selection audit

- Source: `/mnt/data/wge/learn_wm/outputs/week1/selection_round_5090/cem_round_h5_off40_n12_full_v2.npz`
- Cell: H=5, offset=40
- Paired split: dev=[0, 2, 4, 6, 8, 10], held-out=[1, 3, 5, 7, 9, 11]
- Primary test: proposer `pd_d192_k3_eval`, verifier `pd_d192_k10_eval`

## Held-out outcomes

| strategy | true cost | success | Δ cost vs final | Δ success | mean CEM step |
|---|---:|---:|---:|---:|---:|
| `pd_d192_k3_eval:returned:final` | 78.854 | 0.500 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:returned:dev_best_round` | 78.854 | 0.500 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:returned:score_by_pd_d192_k3_eval` | 78.854 | 0.500 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:returned:score_by_pd_d192_k10_eval` | 84.567 | 0.500 | +5.713 [-1.011, +17.107] | +0.000 | 16.5 |
| `pd_d192_k3_eval:returned:nonself_consensus` | 84.831 | 0.500 | +5.976 [-0.428, +17.325] | +0.000 | 17.8 |
| `pd_d192_k3_eval:population_elite:score_by_pd_d192_k10_eval` | 75.637 | 0.333 | -3.217 [-32.511, +19.219] | -0.167 | 12.8 |
| `pd_d192_k3_eval:population_all:score_by_pd_d192_k10_eval` | 72.681 | 0.667 | -6.173 [-27.505, +15.061] | +0.167 | 20.7 |
| `pd_d192_k3_eval:returned:oracle_round` | 61.065 | 0.500 | -17.790 [-50.444, -0.440] | +0.000 | 20.8 |
| `portfolio:returned:rank_consensus` | 33.768 | 0.500 | -45.086 [-101.617, -0.852] | +0.000 | 25.7 |
| `portfolio:returned:oracle` | 27.989 | 0.667 | -50.865 [-111.556, -2.494] | +0.167 | 14.7 |

Lower true cost and higher success are better. Oracle rows are ceilings, not deployable methods. The dev-best fixed round is chosen without inspecting held-out outcomes.

## CEM update equivalence

The table compares each generator's own learned top-k moment update with the top-k update induced by true simulator cost on the same population.

| generator | step | elite overlap | update cosine | relative update error |
|---|---:|---:|---:|---:|
| `pd_d192_k3_eval` | 0 | 0.342 | 0.525 | 0.922 |
| `pd_d192_k3_eval` | 1 | 0.336 | 0.390 | 0.983 |
| `pd_d192_k3_eval` | 2 | 0.214 | 0.219 | 1.129 |
| `pd_d192_k3_eval` | 4 | 0.131 | 0.104 | 1.249 |
| `pd_d192_k3_eval` | 9 | 0.200 | 0.133 | 1.150 |
| `pd_d192_k3_eval` | 19 | 0.161 | 0.098 | 1.229 |
| `pd_d192_k3_eval` | 29 | 0.097 | 0.150 | 1.175 |
| `iter2_multistep_eval` | 0 | 0.494 | 0.580 | 0.800 |
| `iter2_multistep_eval` | 1 | 0.406 | 0.431 | 0.905 |
| `iter2_multistep_eval` | 2 | 0.278 | 0.333 | 1.048 |
| `iter2_multistep_eval` | 4 | 0.192 | 0.158 | 1.213 |
| `iter2_multistep_eval` | 9 | 0.175 | 0.212 | 1.155 |
| `iter2_multistep_eval` | 19 | 0.136 | 0.089 | 1.215 |
| `iter2_multistep_eval` | 29 | 0.064 | 0.054 | 1.230 |
| `pd_d192_k10_eval` | 0 | 0.319 | 0.422 | 0.944 |
| `pd_d192_k10_eval` | 1 | 0.333 | 0.341 | 0.982 |
| `pd_d192_k10_eval` | 2 | 0.228 | 0.260 | 1.071 |
| `pd_d192_k10_eval` | 4 | 0.156 | 0.158 | 1.157 |
| `pd_d192_k10_eval` | 9 | 0.164 | 0.220 | 1.092 |
| `pd_d192_k10_eval` | 19 | 0.069 | -0.048 | 1.297 |
| `pd_d192_k10_eval` | 29 | 0.075 | 0.074 | 1.205 |

An overlap near zero or cosine near zero means that the learned cost and true cost would send the next CEM proposal in different directions. These are snapshot diagnostics, not deployable oracle interventions.
