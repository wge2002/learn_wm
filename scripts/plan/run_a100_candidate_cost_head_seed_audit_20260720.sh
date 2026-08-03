#!/usr/bin/env bash
set -euo pipefail

# Training-seed audit for the two informative H5 cost-head arms:
# deployable frozen latents and the privileged physical-state ceiling.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/candidate_cost_head_seed_audit_a100_20260720
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

run_one() {
    local gpu="$1"
    local family="$2"
    local seed="$3"
    local run_dir="${OUT}/${family}_seed${seed}"
    local log="${OUT}/${family}_seed${seed}.log"
    local latent_args=()
    if [[ "${family}" == "planner_latent" ]]; then
        latent_args=(--latent-cache "${LATENT}")
    fi
    if [[ -s "${run_dir}/results.json" ]]; then
        echo "already complete: ${family} seed=${seed}"
        return
    fi
    echo "start gpu=${gpu} family=${family} seed=${seed} at $(date -Is)"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${PROBE}" \
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
        --seed "${seed}" \
        >"${log}" 2>&1
}

pids=()
run_one 0 planner_latent 20260850 & pids+=("$!")
run_one 1 planner_latent 20260851 & pids+=("$!")
run_one 2 planner_state_oracle 20260850 & pids+=("$!")
run_one 3 planner_state_oracle 20260851 & pids+=("$!")
for pid in "${pids[@]}"; do
    wait "${pid}"
done

pids=()
run_one 0 planner_latent 20260852 & pids+=("$!")
run_one 1 planner_state_oracle 20260852 & pids+=("$!")
for pid in "${pids[@]}"; do
    wait "${pid}"
done

touch "${OUT}/ALL.done"
echo "candidate cost-head seed audit complete at $(date -Is)"
