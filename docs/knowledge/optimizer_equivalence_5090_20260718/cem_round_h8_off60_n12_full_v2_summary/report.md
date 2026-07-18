# CEM round selection audit

- Source: `/mnt/data/wge/learn_wm/outputs/week1/selection_round_5090/cem_round_h8_off60_n12_full_v2.npz`
- Cell: H=8, offset=60
- Paired split: dev=[0, 2, 4, 6, 8, 10], held-out=[1, 3, 5, 7, 9, 11]
- Primary test: proposer `pd_d192_k3_eval`, verifier `pd_d192_k10_eval`

## Held-out outcomes

| strategy | true cost | success | Δ cost vs final | Δ success | mean CEM step |
|---|---:|---:|---:|---:|---:|
| `pd_d192_k3_eval:returned:final` | 113.387 | 0.500 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:returned:dev_best_round` | 113.387 | 0.500 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:returned:score_by_pd_d192_k3_eval` | 113.387 | 0.500 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:returned:score_by_pd_d192_k10_eval` | 114.204 | 0.500 | +0.817 [-2.477, +4.928] | +0.000 | 24.0 |
| `pd_d192_k3_eval:returned:nonself_consensus` | 114.912 | 0.500 | +1.525 [-0.353, +4.928] | +0.000 | 23.2 |
| `pd_d192_k3_eval:population_elite:score_by_pd_d192_k10_eval` | 97.530 | 0.500 | -15.857 [-48.942, +7.106] | +0.000 | 15.3 |
| `pd_d192_k3_eval:population_all:score_by_pd_d192_k10_eval` | 92.229 | 0.500 | -21.158 [-66.419, +7.465] | +0.000 | 18.7 |
| `pd_d192_k3_eval:returned:oracle_round` | 79.998 | 0.500 | -33.389 [-68.600, -0.826] | +0.000 | 16.2 |
| `portfolio:returned:rank_consensus` | 91.234 | 0.500 | -22.153 [-72.172, +7.226] | +0.000 | 23.2 |
| `portfolio:returned:oracle` | 62.137 | 0.667 | -51.250 [-106.486, -2.413] | +0.167 | 20.7 |

Lower true cost and higher success are better. Oracle rows are ceilings, not deployable methods. The dev-best fixed round is chosen without inspecting held-out outcomes.

## CEM update equivalence

The table compares each generator's own learned top-k moment update with the top-k update induced by true simulator cost on the same population.

| generator | step | elite overlap | update cosine | relative update error |
|---|---:|---:|---:|---:|
| `pd_d192_k3_eval` | 0 | 0.481 | 0.608 | 0.830 |
| `pd_d192_k3_eval` | 1 | 0.339 | 0.449 | 0.981 |
| `pd_d192_k3_eval` | 2 | 0.272 | 0.349 | 1.064 |
| `pd_d192_k3_eval` | 4 | 0.258 | 0.273 | 1.114 |
| `pd_d192_k3_eval` | 9 | 0.275 | 0.176 | 1.237 |
| `pd_d192_k3_eval` | 19 | 0.286 | 0.304 | 1.038 |
| `pd_d192_k3_eval` | 29 | 0.219 | 0.311 | 1.047 |
| `iter2_multistep_eval` | 0 | 0.544 | 0.617 | 0.758 |
| `iter2_multistep_eval` | 1 | 0.406 | 0.494 | 0.920 |
| `iter2_multistep_eval` | 2 | 0.339 | 0.415 | 1.004 |
| `iter2_multistep_eval` | 4 | 0.314 | 0.273 | 1.109 |
| `iter2_multistep_eval` | 9 | 0.283 | 0.210 | 1.141 |
| `iter2_multistep_eval` | 19 | 0.258 | 0.315 | 1.050 |
| `iter2_multistep_eval` | 29 | 0.231 | 0.192 | 1.194 |
| `pd_d192_k10_eval` | 0 | 0.511 | 0.552 | 0.816 |
| `pd_d192_k10_eval` | 1 | 0.436 | 0.477 | 0.920 |
| `pd_d192_k10_eval` | 2 | 0.367 | 0.393 | 0.984 |
| `pd_d192_k10_eval` | 4 | 0.306 | 0.287 | 1.096 |
| `pd_d192_k10_eval` | 9 | 0.219 | 0.148 | 1.179 |
| `pd_d192_k10_eval` | 19 | 0.158 | 0.203 | 1.135 |
| `pd_d192_k10_eval` | 29 | 0.111 | 0.217 | 1.095 |

An overlap near zero or cosine near zero means that the learned cost and true cost would send the next CEM proposal in different directions. These are snapshot diagnostics, not deployable oracle interventions.
