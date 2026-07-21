# Paired basin-lineage audit

Source: `../h5_off40_k3_k10_n60_seed20260735_v1.npz`; 60 states, steps [4, 9, 19, 29].

## Recursive outcomes

| generator | step | returned success | population any-success | population success rate | best true cost |
|---|---:|---:|---:|---:|---:|
| pd_d192_k3_eval | 4 | 0.200 | 0.433 | 0.054 | 48.17 |
| pd_d192_k3_eval | 9 | 0.317 | 0.617 | 0.106 | 38.72 |
| pd_d192_k3_eval | 19 | 0.533 | 0.667 | 0.246 | 36.65 |
| pd_d192_k3_eval | 29 | 0.517 | 0.667 | 0.386 | 42.44 |
| pd_d192_k10_eval | 4 | 0.150 | 0.367 | 0.047 | 61.91 |
| pd_d192_k10_eval | 9 | 0.383 | 0.517 | 0.097 | 57.09 |
| pd_d192_k10_eval | 19 | 0.417 | 0.650 | 0.213 | 54.49 |
| pd_d192_k10_eval | 29 | 0.433 | 0.633 | 0.323 | 58.05 |

## Fixed-population scorer fidelity

| generator path | step | scorer | true-elite recall | Spearman | selected success rate | selected any-success |
|---|---:|---|---:|---:|---:|---:|
| pd_d192_k3_eval | 4 | pd_d192_k3_eval | 0.262 | 0.441 | 0.139 | 0.350 |
| pd_d192_k3_eval | 4 | pd_d192_k10_eval | 0.411 | 0.657 | 0.152 | 0.433 |
| pd_d192_k3_eval | 9 | pd_d192_k3_eval | 0.234 | 0.431 | 0.252 | 0.533 |
| pd_d192_k3_eval | 9 | pd_d192_k10_eval | 0.362 | 0.607 | 0.289 | 0.583 |
| pd_d192_k3_eval | 19 | pd_d192_k3_eval | 0.184 | 0.392 | 0.438 | 0.600 |
| pd_d192_k3_eval | 19 | pd_d192_k10_eval | 0.286 | 0.525 | 0.461 | 0.667 |
| pd_d192_k3_eval | 29 | pd_d192_k3_eval | 0.096 | 0.271 | 0.492 | 0.567 |
| pd_d192_k3_eval | 29 | pd_d192_k10_eval | 0.272 | 0.444 | 0.498 | 0.667 |
| pd_d192_k10_eval | 4 | pd_d192_k3_eval | 0.285 | 0.365 | 0.129 | 0.317 |
| pd_d192_k10_eval | 4 | pd_d192_k10_eval | 0.302 | 0.507 | 0.133 | 0.317 |
| pd_d192_k10_eval | 9 | pd_d192_k3_eval | 0.266 | 0.387 | 0.234 | 0.500 |
| pd_d192_k10_eval | 9 | pd_d192_k10_eval | 0.277 | 0.499 | 0.252 | 0.467 |
| pd_d192_k10_eval | 19 | pd_d192_k3_eval | 0.208 | 0.314 | 0.339 | 0.550 |
| pd_d192_k10_eval | 19 | pd_d192_k10_eval | 0.183 | 0.404 | 0.379 | 0.600 |
| pd_d192_k10_eval | 29 | pd_d192_k3_eval | 0.197 | 0.247 | 0.409 | 0.583 |
| pd_d192_k10_eval | 29 | pd_d192_k10_eval | 0.079 | 0.246 | 0.402 | 0.583 |

## Final paired comparison

K10 − K3 returned-success delta: -0.083 (95% state bootstrap [-0.200, +0.033]).

Discordant states: K3-only 10, K10-only 5; both 21, neither 24.

Final true-elite-recall path interaction [(K10−K3 scorer on K10 path) − (K10−K3 scorer on K3 path)]: -0.293 (95% state bootstrap [-0.381, -0.207]).

## Elite-to-mean conversion at the final step

| generator | elite has success | returned success | witness but returned failure | conversion given witness |
|---|---:|---:|---:|---:|
| pd_d192_k3_eval | 0.567 | 0.517 | 0.050 | 0.912 |
| pd_d192_k10_eval | 0.583 | 0.433 | 0.150 | 0.743 |

## Interpretation guardrail

Connected-component results are local sampled-support diagnostics. Components were not matched across rounds, so their count trajectory alone is not a basin-identity claim.
