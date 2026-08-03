#!/usr/bin/env bash
set -euo pipefail

# Recursive, row-disjoint Gate for Branch-Preserving Optimizer Equivalence.
# Four three-state shards use all four A100s.  Each state compares:
#   K3 1x300, K3 2x150, BP 2x75 with K3+K10 (matched model calls),
#   BP 2x150 with K3+K10 (2x-compute mechanism ceiling), and sparse BP
#   with 272/136 K3 candidates plus 10% K10 queries (matched calls).

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/bp_oe_recursive_a100_20260720/formal_n12_r3_v3_sparse
PY="${ROOT}/.venv-clean/bin/python"
PROBE="${BUNDLE}/oe_branch_preserving_rollout.py"
DATA=/225010117/data/pusht_expert_train.h5
TRAIN=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_k3_n240_merged_v1.npz
EVAL=/225010117/logs/week1_selection/cem_round_h5_off40_n12_full_v2.npz
K3=/225010117/logs/candidate_outcome_gate_a100_20260720/h5_candidate_outcomes.npz
K10=/225010117/logs/crossmodel_outcome_gate_a100_20260720/h5_k10_candidate_outcomes.npz
K10_SPARSE=/225010117/logs/sparse_crossmodel_outcome_gate_a100_20260720_v2_strict/cache/elite30.npz

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

for required in \
    "${PY}" "${PROBE}" "${DATA}" "${TRAIN}" "${EVAL}" \
    "${K3}" "${K10}" "${K10_SPARSE}"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing required input: ${required}" >&2
        exit 1
    fi
done

pids=()
for gpu in 0 1 2 3; do
    state_start=$((gpu * 3))
    result="${OUT}/shard_${state_start}_$((state_start + 3)).npz"
    log="${OUT}/shard_${state_start}_$((state_start + 3)).log"
    echo "start gpu=${gpu} states=[${state_start},$((state_start + 3))) at $(date -Is)"
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
            +branch_oe.k10_sparse_outcome="${K10_SPARSE}" \
            +branch_oe.out="${result}" \
            +branch_oe.state_start="${state_start}" \
            +branch_oe.num_states=3 \
            +branch_oe.num_rounds=3 \
            +branch_oe.start_step=4 \
            +branch_oe.train_epochs=10 \
            +branch_oe.blend=0.5 \
            +branch_oe.diversity_threshold=0.25 \
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
    echo "one or more recursive BP-OE shards failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "recursive BP-OE Gate complete at $(date -Is)"
