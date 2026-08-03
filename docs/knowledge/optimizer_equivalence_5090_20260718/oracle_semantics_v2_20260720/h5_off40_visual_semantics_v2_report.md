# OE oracle-semantics audit

- Source: `/mnt/data/wge/learn_wm/outputs/week1/selection_round_5090/cem_round_h5_off40_n12_full_v2.npz`
- Policy / generator / scorer: `pd_d192_k3_eval` / `pd_d192_k3_eval` / `pd_d192_k3_eval`
- Cell: H5 / off40; 12 states; steps=[4, 9, 19, 29]; N=300; top-k=30

## Integrity checks

- initial-state reconstruction max abs: `0.000e+00`
- goal-state reconstruction max abs: `0.000e+00`
- recomputed physical-cost max abs: `0.000e+00`
- recomputed success disagreements: `0`
- exact render-visible terminal-state max abs: `2.842e-14`

## State-blocked aggregate

| metric | mean | paired-state 95% bootstrap CI |
| --- | ---: | ---: |
| learned update cosine | 0.121 | [0.046, 0.201] |
| visual update cosine | 0.426 | [0.303, 0.546] |
| visual - learned cosine | 0.305 | [0.212, 0.406] |
| learned elite overlap | 0.147 | [0.096, 0.199] |
| visual elite overlap | 0.374 | [0.258, 0.493] |
| visual/physical Spearman | 0.478 | [0.293, 0.657] |
| visual true-cost gain | 4.290 | [1.845, 6.705] |
| visual recovery fraction | 0.235 | [0.092, 0.444] |

The recovery fraction uses physical elite cost and is computed as `(learned - visual) / (learned - physical oracle)`. Positive values mean that true-terminal visual geometry closes part of the learned-rollout selection gap.

## By CEM step

| step | learned cos | visual cos | learned overlap | visual overlap | visual/physical rho | true-cost gain | recovery |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.104 | 0.240 | 0.131 | 0.208 | 0.397 | 1.41 | 0.052 |
| 9 | 0.133 | 0.452 | 0.200 | 0.344 | 0.489 | 2.83 | 0.163 |
| 19 | 0.098 | 0.542 | 0.161 | 0.467 | 0.622 | 6.56 | 0.466 |
| 29 | 0.150 | 0.471 | 0.097 | 0.475 | 0.402 | 6.36 | 0.446 |

## Predeclared cell gate

**MISS**: visual update cosine `0.426` (required `0.700`), visual elite overlap `0.374` (required `0.500`), and physical-gap recovery `0.235` (required `0.500`).

Passing this cell is not a method result: H5 and H8 must both pass, followed by a recursive proposal-resampling intervention. A miss means the raw LeWM terminal metric should not be used as the general oracle; it does not invalidate a task-aware physical teacher or a learned reachability-metric teacher.

Elapsed: `0.4` minutes.
