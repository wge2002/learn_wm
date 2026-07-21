# Counterfactual refit audit

60 paired states; steps [4, 9, 19, 29].

## Final global-mean refits

| generator path | K3 elite mean | K10 elite mean | true elite mean | stored mean |
|---|---:|---:|---:|---:|
| pd_d192_k3_eval | 51.7% | 55.0% | 63.3% | 51.7% |
| pd_d192_k10_eval | 45.0% | 43.3% | 58.3% | 43.3% |

## Final component-wise refits

| path | elite source | global mean | own/true-selected component mean | any component succeeds |
|---|---|---:|---:|---:|
| pd_d192_k3_eval | k3 | 51.7% | 51.7% | 51.7% |
| pd_d192_k3_eval | k10 | 55.0% | 53.3% | 58.3% |
| pd_d192_k3_eval | true | 63.3% | 65.0% | 65.0% |
| pd_d192_k10_eval | k3 | 45.0% | 46.7% | 48.3% |
| pd_d192_k10_eval | k10 | 43.3% | 43.3% | 43.3% |
| pd_d192_k10_eval | true | 58.3% | 60.0% | 60.0% |

## Final K10-path decomposition

K3-global rescues K10-global failures: 4 states.

True-global fails but a true-elite component mean succeeds: 1 states.

Candidate population contains no success: 22 states.
