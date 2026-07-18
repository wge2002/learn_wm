# CEM round selection audit

- Source: `/mnt/data/wge/learn_wm/outputs/week1/selection_round_5090/cem_round_h5_off40_n50_evalstarts_final_v2.npz`
- Cell: H=5, offset=40
- Paired split: dev=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48], held-out=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
- Primary test: proposer `pd_d192_k3_eval`, verifier `pd_d192_k10_eval`

## Held-out outcomes

| strategy | true cost | success | Δ cost vs final | Δ success | mean CEM step |
|---|---:|---:|---:|---:|---:|
| `pd_d192_k3_eval:returned:final` | 135.439 | 0.480 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:returned:dev_best_round` | 135.439 | 0.480 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:returned:score_by_pd_d192_k3_eval` | 135.439 | 0.480 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:returned:score_by_pd_d192_k10_eval` | 135.439 | 0.480 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:returned:nonself_consensus` | 135.439 | 0.480 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:population_elite:score_by_pd_d192_k10_eval` | 136.119 | 0.440 | +0.681 [-1.170, +2.623] | -0.040 | 29.0 |
| `pd_d192_k3_eval:population_all:score_by_pd_d192_k10_eval` | 137.302 | 0.480 | +1.863 [-1.482, +5.761] | +0.000 | 29.0 |
| `pd_d192_k3_eval:returned:oracle_round` | 135.439 | 0.480 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `portfolio:returned:rank_consensus` | 135.439 | 0.480 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `portfolio:returned:oracle` | 135.439 | 0.480 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |

Lower true cost and higher success are better. Oracle rows are ceilings, not deployable methods. The dev-best fixed round is chosen without inspecting held-out outcomes.

## CEM update equivalence

The table compares each generator's own learned top-k moment update with the top-k update induced by true simulator cost on the same population.

| generator | step | elite overlap | update cosine | relative update error |
|---|---:|---:|---:|---:|
| `pd_d192_k3_eval` | 29 | 0.111 | 0.073 | 1.202 |

An overlap near zero or cosine near zero means that the learned cost and true cost would send the next CEM proposal in different directions. These are snapshot diagnostics, not deployable oracle interventions.
