# OE oracle-semantics audit

- Source: `/mnt/data/wge/learn_wm/outputs/week1/selection_round_5090/cem_round_h8_off60_n12_full_v2.npz`
- Policy / generator / scorer: `pd_d192_k3_eval` / `pd_d192_k3_eval` / `pd_d192_k3_eval`
- Cell: H8 / off60; 12 states; steps=[4, 9, 19, 29]; N=300; top-k=30

## Integrity checks

- initial-state reconstruction max abs: `0.000e+00`
- goal-state reconstruction max abs: `0.000e+00`
- recomputed physical-cost max abs: `0.000e+00`
- recomputed success disagreements: `0`
- exact render-visible terminal-state max abs: `5.684e-14`

## State-blocked aggregate

| metric | mean | paired-state 95% bootstrap CI |
| --- | ---: | ---: |
| learned update cosine | 0.266 | [0.153, 0.377] |
| visual update cosine | 0.541 | [0.383, 0.681] |
| visual - learned cosine | 0.276 | [0.124, 0.431] |
| learned elite overlap | 0.260 | [0.158, 0.366] |
| visual elite overlap | 0.499 | [0.378, 0.608] |
| visual/physical Spearman | 0.613 | [0.423, 0.775] |
| visual true-cost gain | 8.684 | [0.161, 16.831] |
| visual recovery fraction | 0.385 | [0.009, 0.645] |

The recovery fraction uses physical elite cost and is computed as `(learned - visual) / (learned - physical oracle)`. Positive values mean that true-terminal visual geometry closes part of the learned-rollout selection gap.

## By CEM step

| step | learned cos | visual cos | learned overlap | visual overlap | visual/physical rho | true-cost gain | recovery |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.273 | 0.481 | 0.258 | 0.436 | 0.523 | 9.26 | 0.323 |
| 9 | 0.176 | 0.517 | 0.275 | 0.497 | 0.610 | 13.83 | 0.456 |
| 19 | 0.304 | 0.650 | 0.286 | 0.581 | 0.647 | 8.71 | 0.468 |
| 29 | 0.311 | 0.517 | 0.219 | 0.483 | 0.672 | 2.93 | 0.233 |

## Predeclared cell gate

**MISS**: visual update cosine `0.541` (required `0.700`), visual elite overlap `0.499` (required `0.500`), and physical-gap recovery `0.385` (required `0.500`).

Passing this cell is not a method result: H5 and H8 must both pass, followed by a recursive proposal-resampling intervention. A miss means the raw LeWM terminal metric should not be used as the general oracle; it does not invalidate a task-aware physical teacher or a learned reachability-metric teacher.

Elapsed: `0.4` minutes.
