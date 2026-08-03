#!/usr/bin/env bash
set -euo pipefail

# Clean disagreement ablation: retain each model's robust scalar cost/rank as
# separate candidate features, but remove high-dimensional latent vectors.
# The operator may route branches from disagreement; it never averages ranks.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/crossmodel_scalar_set_gate_a100_20260720
PY="${ROOT}/.venv-clean/bin/python"
SOURCE=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_k3_n240_merged_v1.npz
DENSE=/225010117/logs/dense_token_gate_a100_20260720/h5_dense_tokens.npz
K3=/225010117/logs/candidate_outcome_gate_a100_20260720/h5_candidate_outcomes.npz
K5=/225010117/logs/crossmodel_outcome_gate_a100_20260720/h5_k5_candidate_outcomes.npz
K10=/225010117/logs/crossmodel_outcome_gate_a100_20260720/h5_k10_candidate_outcomes.npz
PROBE="${BUNDLE}/oe_set_valued_operator_probe.py"

export PYTHONPATH="${ROOT}/scripts/plan:${ROOT}:${BUNDLE}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "${OUT}"
cd "${ROOT}"

names=(scalar_k10 scalar_k5 scalar_k5_k10 dense_scalar_k5_k10)

pids=()
for gpu in 0 1 2 3; do
    name="${names[$gpu]}"
    run_dir="${OUT}/${name}"
    log="${OUT}/${name}.log"
    extra_args=()
    dense_args=()
    case "${name}" in
        scalar_k10)
            extra_args=(--extra-outcome-cache "${K10}")
            ;;
        scalar_k5)
            extra_args=(--extra-outcome-cache "${K5}")
            ;;
        scalar_k5_k10|dense_scalar_k5_k10)
            extra_args=(
                --extra-outcome-cache "${K5}"
                --extra-outcome-cache "${K10}"
            )
            ;;
    esac
    family=planner_outcome
    if [[ "${name}" == dense_* ]]; then
        family=planner_dense_outcome
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
            --outcome-cache "${K3}" \
            "${extra_args[@]}" \
            --outcome-cost-only \
            --topk 30 \
            --modes 5 \
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
            --seed "$((20261060 + gpu))" \
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
    echo "one or more cross-model scalar probes failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "cross-model scalar set Gate complete at $(date -Is)"
