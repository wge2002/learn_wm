#!/usr/bin/env bash
set -euo pipefail

# Strict H5 fixed-bank Gate for the candidate-level cost-correction head.
# Every metric is computed from outer state-held-out predictions; epoch and
# residual blend are selected only on an inner held-out state split.
#
# The three deployable arms use planner signals available at inference.  The
# physical-state arm is privileged and serves only as a learnability ceiling.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/candidate_cost_head_gate_a100_20260720
PY="${ROOT}/.venv-clean/bin/python"
SOURCE=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_k3_n240_merged_v1.npz
LATENT=/225010117/logs/operator_head_gate_a100_20260720/h5_off40_k3_n240_latent.npz
PROBE="${BUNDLE}/oe_candidate_cost_head_probe.py"

export PYTHONPATH="${ROOT}/scripts/plan:${ROOT}:${BUNDLE}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "${OUT}"
cd "${ROOT}"

for required in "${PY}" "${SOURCE}" "${LATENT}" "${PROBE}" \
    "${BUNDLE}/oe_update_corrector_probe.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing required input: ${required}" >&2
        exit 1
    fi
done

families=(
    planner
    planner_latent
    planner_history_latent
    planner_state_oracle
)

pids=()
for gpu in 0 1 2 3; do
    family="${families[$gpu]}"
    run_dir="${OUT}/h5_${family}_mlp128"
    log="${OUT}/h5_${family}_mlp128.log"
    if [[ -s "${run_dir}/results.json" ]]; then
        echo "already complete: ${family}"
        continue
    fi
    latent_args=()
    if [[ "${family}" == *latent ]]; then
        latent_args=(--latent-cache "${LATENT}")
    fi
    echo "start gpu=${gpu} family=${family} at $(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PY}" "${PROBE}" \
            "${SOURCE}" \
            --out-dir "${run_dir}" \
            --family "${family}" \
            "${latent_args[@]}" \
            --topk 30 \
            --hidden 128 \
            --lr 3e-4 \
            --weight-decay 1e-4 \
            --max-epochs 8 \
            --batch-populations 16 \
            --listwise-weight 0.2 \
            --anchor-weight 1e-3 \
            --bootstrap 20000 \
            --seed "$((20260840 + gpu))" \
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
    echo "one or more candidate cost-head probes failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "candidate cost-head H5 Gate complete at $(date -Is)"
