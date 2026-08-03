#!/usr/bin/env bash
set -euo pipefail

# H5 Gate for the population-conditioned set-valued OE operator.  M=1 is the
# unimodal control; M=5 retains no-op plus four learned correction modes.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/set_valued_operator_gate_a100_20260720
PY="${ROOT}/.venv-clean/bin/python"
SOURCE=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_k3_n240_merged_v1.npz
LATENT=/225010117/logs/operator_head_gate_a100_20260720/h5_off40_k3_n240_latent.npz
PROBE="${BUNDLE}/oe_set_valued_operator_probe.py"

export PYTHONPATH="${ROOT}/scripts/plan:${ROOT}:${BUNDLE}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "${OUT}"
cd "${ROOT}"

families=(
    planner_latent
    planner_latent
    planner_state_oracle
    planner_state_oracle
)
modes=(1 5 1 5)

pids=()
for gpu in 0 1 2 3; do
    family="${families[$gpu]}"
    mode_count="${modes[$gpu]}"
    run_dir="${OUT}/h5_${family}_m${mode_count}"
    log="${OUT}/h5_${family}_m${mode_count}.log"
    latent_args=()
    if [[ "${family}" == *latent ]]; then
        latent_args=(--latent-cache "${LATENT}")
    fi
    echo "start gpu=${gpu} family=${family} modes=${mode_count} at $(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PY}" "${PROBE}" \
            "${SOURCE}" \
            --out-dir "${run_dir}" \
            --family "${family}" \
            "${latent_args[@]}" \
            --topk 30 \
            --modes "${mode_count}" \
            --hidden 128 \
            --attention-heads 4 \
            --lr 3e-4 \
            --weight-decay 1e-4 \
            --max-epochs 20 \
            --batch-populations 16 \
            --router-weight 0.1 \
            --delta-anchor-weight 1e-3 \
            --bootstrap 20000 \
            --seed "$((20260880 + gpu))" \
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
    echo "one or more set-valued operator probes failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "set-valued operator H5 Gate complete at $(date -Is)"
