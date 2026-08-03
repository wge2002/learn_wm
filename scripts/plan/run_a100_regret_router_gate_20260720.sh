#!/usr/bin/env bash
set -euo pipefail

# A deployable-router diagnostic for the branch operator.  Exact mode
# classification penalizes every wrong branch equally; these arms instead
# train routing probabilities from the physical regret of each retained
# branch, while keeping the branch outputs on the same best-of-M objective.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/regret_router_gate_a100_20260720
PY="${ROOT}/.venv-clean/bin/python"
SOURCE=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_k3_n240_merged_v1.npz
DENSE=/225010117/logs/dense_token_gate_a100_20260720/h5_dense_tokens.npz
OUTCOME=/225010117/logs/candidate_outcome_gate_a100_20260720/h5_candidate_outcomes.npz
PROBE="${BUNDLE}/oe_set_valued_operator_probe.py"

export PYTHONPATH="${ROOT}/scripts/plan:${ROOT}:${BUNDLE}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "${OUT}"
cd "${ROOT}"

names=(
    dense_expected
    outcome_expected
    dense_outcome_expected
    dense_outcome_softmin
)
families=(
    planner_dense
    planner_outcome
    planner_dense_outcome
    planner_dense_outcome
)
kinds=(
    expected_regret
    expected_regret
    expected_regret
    softmin
)

pids=()
for gpu in 0 1 2 3; do
    name="${names[$gpu]}"
    family="${families[$gpu]}"
    run_dir="${OUT}/${name}"
    log="${OUT}/${name}.log"
    dense_args=()
    outcome_args=()
    if [[ "${family}" == *dense* ]]; then
        dense_args=(--dense-cache "${DENSE}")
    fi
    if [[ "${family}" == *outcome* ]]; then
        outcome_args=(--outcome-cache "${OUTCOME}")
    fi
    echo "start gpu=${gpu} arm=${name} at $(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PY}" "${PROBE}" \
            "${SOURCE}" \
            --out-dir "${run_dir}" \
            --family "${family}" \
            "${dense_args[@]}" \
            "${outcome_args[@]}" \
            --topk 30 \
            --modes 5 \
            --hidden 128 \
            --attention-heads 4 \
            --lr 3e-4 \
            --weight-decay 1e-4 \
            --max-epochs 20 \
            --batch-populations 8 \
            --router-weight 1.0 \
            --router-kind "${kinds[$gpu]}" \
            --router-temperature 0.1 \
            --delta-anchor-weight 1e-3 \
            --bootstrap 20000 \
            --seed "$((20260980 + gpu))" \
            >"${log}" 2>&1
        touch "${run_dir}/DONE"
    ) &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
if ((status != 0)); then
    echo "one or more regret-router probes failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "regret-router gate complete at $(date -Is)"
