#!/usr/bin/env bash
set -euo pipefail

# Gate for vector-valued candidate imagination.  LeWM normally collapses each
# predicted terminal embedding to one squared-distance scalar.  These arms
# preserve the signed terminal-minus-goal vector and test whether it resolves
# branch identity, with and without the frozen dense spatial read path.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/candidate_outcome_set_operator_gate_a100_20260720
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
    outcome_m1
    outcome_m5
    dense_outcome_m5
    dense_outcome_m5_router1
)
families=(
    planner_outcome
    planner_outcome
    planner_dense_outcome
    planner_dense_outcome
)
modes=(1 5 5 5)
router_weights=(0.1 0.1 0.1 1.0)

pids=()
for gpu in 0 1 2 3; do
    name="${names[$gpu]}"
    family="${families[$gpu]}"
    run_dir="${OUT}/${name}"
    log="${OUT}/${name}.log"
    dense_args=()
    if [[ "${family}" == *dense* ]]; then
        dense_args=(--dense-cache "${DENSE}")
    fi
    echo "start gpu=${gpu} arm=${name} at $(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PY}" "${PROBE}" \
            "${SOURCE}" \
            --out-dir "${run_dir}" \
            --family "${family}" \
            "${dense_args[@]}" \
            --outcome-cache "${OUTCOME}" \
            --topk 30 \
            --modes "${modes[$gpu]}" \
            --hidden 128 \
            --attention-heads 4 \
            --lr 3e-4 \
            --weight-decay 1e-4 \
            --max-epochs 20 \
            --batch-populations 8 \
            --router-weight "${router_weights[$gpu]}" \
            --delta-anchor-weight 1e-3 \
            --bootstrap 20000 \
            --seed "$((20260940 + gpu))" \
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
    echo "one or more candidate-outcome probes failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "candidate-outcome set-operator gate complete at $(date -Is)"
