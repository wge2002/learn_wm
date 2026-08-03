#!/usr/bin/env bash
set -euo pipefail

# Four-rung H5 information-sufficiency ladder for a structured dynamics
# sidecar.  The two latent arms are deployable.  The true-state arms are
# privileged learnability ceilings only.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/structured_dynamics_head_gate_a100_20260720
PY="${ROOT}/.venv-clean/bin/python"
SOURCE=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_k3_n240_merged_v1.npz
LATENT=/225010117/logs/operator_head_gate_a100_20260720/h5_off40_k3_n240_latent.npz
PROBE="${BUNDLE}/oe_structured_dynamics_head_probe.py"

export PYTHONPATH="${ROOT}/scripts/plan:${ROOT}:${BUNDLE}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "${OUT}"
cd "${ROOT}"

families=(latent latent state_oracle state_oracle)
targets=(terminal relative terminal relative)

pids=()
for gpu in 0 1 2 3; do
    family="${families[$gpu]}"
    target="${targets[$gpu]}"
    run_dir="${OUT}/h5_${family}_${target}_mlp128"
    log="${OUT}/h5_${family}_${target}_mlp128.log"
    latent_args=()
    if [[ "${family}" == "latent" ]]; then
        latent_args=(--latent-cache "${LATENT}")
    fi
    echo "start gpu=${gpu} family=${family} target=${target} at $(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PY}" "${PROBE}" \
            "${SOURCE}" \
            --out-dir "${run_dir}" \
            --family "${family}" \
            --target "${target}" \
            "${latent_args[@]}" \
            --topk 30 \
            --hidden 128 \
            --lr 3e-4 \
            --weight-decay 1e-4 \
            --max-epochs 20 \
            --batch-candidates 4096 \
            --goal-weight 1.0 \
            --anchor-weight 1e-7 \
            --bootstrap 20000 \
            --seed "$((20260860 + gpu))" \
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
    echo "one or more structured dynamics-head probes failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "structured dynamics-head H5 Gate complete at $(date -Is)"
