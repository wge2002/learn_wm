#!/usr/bin/env bash
set -euo pipefail

# Gate for residual-conditioned optimizer equivalence.  The innovation feature
# is the frozen LeWM's last observable one-step latent error, available after a
# real transition without privileged state or simulator cost.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/innovation_set_operator_gate_a100_20260720
PY="${ROOT}/.venv-clean/bin/python"
SOURCE=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_k3_n240_merged_v1.npz
DENSE=/225010117/logs/dense_token_gate_a100_20260720/h5_dense_tokens.npz
OUTCOME=/225010117/logs/candidate_outcome_gate_a100_20260720/h5_candidate_outcomes.npz
INNOVATION=/225010117/logs/innovation_gate_a100_20260720/h5_innovation.npz
PROBE="${BUNDLE}/oe_set_valued_operator_probe.py"

export PYTHONPATH="${ROOT}/scripts/plan:${ROOT}:${BUNDLE}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "${OUT}"
cd "${ROOT}"

names=(
    innovation_m1
    innovation_m5
    innovation_outcome_m5
    dense_innovation_outcome_m5
)
families=(
    planner_innovation
    planner_innovation
    planner_innovation_outcome
    planner_dense_innovation_outcome
)
modes=(1 5 5 5)

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
            --innovation-cache "${INNOVATION}" \
            --topk 30 \
            --modes "${modes[$gpu]}" \
            --hidden 128 \
            --attention-heads 4 \
            --lr 3e-4 \
            --weight-decay 1e-4 \
            --max-epochs 20 \
            --batch-populations 8 \
            --router-weight 0.1 \
            --router-kind winner_ce \
            --delta-anchor-weight 1e-3 \
            --bootstrap 20000 \
            --seed "$((20261000 + gpu))" \
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
    echo "one or more innovation probes failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "innovation set-operator gate complete at $(date -Is)"
