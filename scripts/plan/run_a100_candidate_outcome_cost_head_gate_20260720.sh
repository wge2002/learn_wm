#!/usr/bin/env bash
set -euo pipefail

# Candidate-reranking ablation for vector-valued imagined outcomes.  This asks
# whether preserving terminal-minus-goal direction is sufficient without the
# explicit set-valued branch operator.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/candidate_outcome_cost_head_gate_a100_20260720
PY="${ROOT}/.venv-clean/bin/python"
SOURCE=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_k3_n240_merged_v1.npz
OUTCOME=/225010117/logs/candidate_outcome_gate_a100_20260720/h5_candidate_outcomes.npz
PROBE="${BUNDLE}/oe_candidate_cost_head_probe.py"

export PYTHONPATH="${ROOT}/scripts/plan:${ROOT}:${BUNDLE}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "${OUT}"
cd "${ROOT}"

pids=()
for gpu in 0 1 2 3; do
    seed="$((20260960 + gpu))"
    run_dir="${OUT}/planner_outcome_seed${seed}"
    log="${OUT}/planner_outcome_seed${seed}.log"
    echo "start gpu=${gpu} seed=${seed} at $(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PY}" "${PROBE}" \
            "${SOURCE}" \
            --out-dir "${run_dir}" \
            --family planner_outcome \
            --outcome-cache "${OUTCOME}" \
            --topk 30 \
            --hidden 128 \
            --lr 3e-4 \
            --weight-decay 1e-4 \
            --max-epochs 8 \
            --batch-populations 16 \
            --listwise-weight 0.2 \
            --anchor-weight 1e-3 \
            --bootstrap 20000 \
            --seed "${seed}" \
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
    echo "one or more candidate-outcome cost heads failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "candidate-outcome cost-head gate complete at $(date -Is)"
