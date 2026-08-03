# CEM round selection audit

- Source: `/mnt/data/wge/learn_wm/outputs/week1/selection_round_5090/cem_round_h5_off40_k3_n60_seed20260719_v1.npz`
- Cell: H=5, offset=40
- Paired split: dev=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58], held-out=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59]
- Primary test: proposer `pd_d192_k3_eval`, verifier `pd_d192_k3_eval`

## Held-out outcomes

| strategy | true cost | success | Δ cost vs final | Δ success | mean CEM step |
|---|---:|---:|---:|---:|---:|
| `pd_d192_k3_eval:returned:final` | 126.522 | 0.467 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:returned:dev_best_round` | 126.522 | 0.467 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:returned:score_by_pd_d192_k3_eval` | 126.522 | 0.467 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:returned:score_by_pd_d192_k3_eval` | 126.522 | 0.467 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:returned:nonself_consensus` | 126.522 | 0.467 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `pd_d192_k3_eval:population_elite:score_by_pd_d192_k3_eval` | 127.647 | 0.433 | +1.125 [-8.195, +10.524] | -0.033 | 28.3 |
| `pd_d192_k3_eval:population_all:score_by_pd_d192_k3_eval` | 127.647 | 0.433 | +1.125 [-8.195, +10.524] | -0.033 | 28.3 |
| `pd_d192_k3_eval:returned:oracle_round` | 91.541 | 0.500 | -34.981 [-64.513, -12.005] | +0.033 | 14.7 |
| `portfolio:returned:rank_consensus` | 126.522 | 0.467 | +0.000 [+0.000, +0.000] | +0.000 | 29.0 |
| `portfolio:returned:oracle` | 91.541 | 0.500 | -34.981 [-64.513, -12.005] | +0.033 | 14.7 |

Lower true cost and higher success are better. Oracle rows are ceilings, not deployable methods. The dev-best fixed round is chosen without inspecting held-out outcomes.

## CEM update equivalence

The table compares each generator's own learned top-k moment update with the top-k update induced by true simulator cost on the same population.

| generator | step | elite overlap | update cosine | relative update error |
|---|---:|---:|---:|---:|
| `pd_d192_k3_eval` | 4 | 0.253 | 0.295 | 1.090 |
| `pd_d192_k3_eval` | 9 | 0.218 | 0.217 | 1.159 |
| `pd_d192_k3_eval` | 19 | 0.135 | 0.132 | 1.159 |
| `pd_d192_k3_eval` | 29 | 0.079 | 0.043 | 1.198 |

An overlap near zero or cosine near zero means that the learned cost and true cost would send the next CEM proposal in different directions. These are snapshot diagnostics, not deployable oracle interventions.
