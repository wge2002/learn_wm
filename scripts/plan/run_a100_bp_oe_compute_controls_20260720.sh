#!/usr/bin/env bash
set -euo pipefail

# Equal-world-model-call controls for the 600-call BP mechanism ceiling.
# Results are paired offline with the CRN recursive Gate on the same rows.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/bp_oe_recursive_a100_20260720/formal_n12_r3_compute600_v1
PY="${ROOT}/.venv-clean/bin/python"
PROBE="${BUNDLE}/oe_branch_preserving_rollout.py"
DATA=/225010117/data/pusht_expert_train.h5
TRAIN=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_k3_n240_merged_v1.npz
EVAL=/225010117/logs/week1_selection/cem_round_h5_off40_n12_full_v2.npz
K3=/225010117/logs/candidate_outcome_gate_a100_20260720/h5_candidate_outcomes.npz
K10=/225010117/logs/crossmodel_outcome_gate_a100_20260720/h5_k10_candidate_outcomes.npz

export STABLEWM_HOME=/225010117/stablewm
export HF_HOME=/225010117/cache/huggingface
export PYTHONPATH="${ROOT}/scripts/plan:${ROOT}:${BUNDLE}"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "${OUT}"
cd "${ROOT}"

pids=()
for gpu in 0 1 2 3; do
    state_start=$((gpu * 3))
    result="${OUT}/shard_${state_start}_$((state_start + 3)).npz"
    log="${OUT}/shard_${state_start}_$((state_start + 3)).log"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PY}" "${PROBE}" \
            +plan_config.history_len=3 \
            plan_config.horizon=5 \
            plan_config.receding_horizon=5 \
            eval.goal_offset_steps=40 \
            eval.video=false \
            eval.dataset_name="${DATA}" \
            +branch_oe.train_source="${TRAIN}" \
            +branch_oe.eval_source="${EVAL}" \
            +branch_oe.k3_outcome="${K3}" \
            +branch_oe.k10_outcome="${K10}" \
            +branch_oe.out="${result}" \
            +branch_oe.state_start="${state_start}" \
            +branch_oe.num_states=3 \
            +branch_oe.num_rounds=3 \
            +branch_oe.start_step=4 \
            +branch_oe.train_epochs=10 \
            +branch_oe.compute_controls=true \
            "+branch_oe.methods='k3_1x600,k3_2x300'" \
            >"${log}" 2>&1
        touch "${OUT}/shard_${state_start}_$((state_start + 3)).done"
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
    echo "one or more 600-call controls failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "600-call controls complete at $(date -Is)"
