#!/usr/bin/env bash
set -euo pipefail

# Frozen dense-patch spatial readout + action-conditioned structured dynamics.
# This is the first training-based alternative after the frozen observable
# router audit: it predicts counterfactual terminal task geometry for every
# candidate, then fuses the resulting physical cost with LeWM's latent cost.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/dense_structured_dynamics_gate_a100_20260720
PY="${ROOT}/.venv-clean/bin/python"
SOURCE=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_k3_n240_merged_v1.npz
DENSE=/225010117/logs/dense_token_gate_a100_20260720/h5_dense_tokens.npz
PROBE="${BUNDLE}/oe_structured_dynamics_head_probe.py"

export PYTHONPATH="${ROOT}/scripts/plan:${ROOT}:${BUNDLE}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "${OUT}"
cd "${ROOT}"

names=(
    relative_h128_seed0
    relative_h128_seed1
    relative_h256_seed0
    terminal_h128_seed0
)
targets=(relative relative relative terminal)
hiddens=(128 128 256 128)
seeds=(20261020 20261021 20261020 20261020)

pids=()
for gpu in 0 1 2 3; do
    name="${names[$gpu]}"
    run_dir="${OUT}/${name}"
    log="${OUT}/${name}.log"
    echo "start gpu=${gpu} arm=${name} at $(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PY}" "${PROBE}" \
            "${SOURCE}" \
            --out-dir "${run_dir}" \
            --family dense_moment \
            --target "${targets[$gpu]}" \
            --dense-cache "${DENSE}" \
            --topk 30 \
            --hidden "${hiddens[$gpu]}" \
            --lr 3e-4 \
            --weight-decay 1e-4 \
            --max-epochs 20 \
            --batch-candidates 4096 \
            --goal-weight 1.0 \
            --anchor-weight 1e-7 \
            --bootstrap 20000 \
            --seed "${seeds[$gpu]}" \
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
    echo "one or more dense structured-dynamics probes failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "dense structured-dynamics gate complete at $(date -Is)"
