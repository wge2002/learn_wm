#!/usr/bin/env bash
set -euo pipefail

# Generate a row-disjoint H5/off40 proposal source without executing all 300
# candidates in the simulator.  Only the saved K3 step-4 proposal is needed
# by the recursive branch Gate; final branch means are evaluated separately.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/bp_oe_recursive_a100_20260720/fresh_eval_source_h5_off40_n60_seed20260726_v1
PY="${ROOT}/.venv-clean/bin/python"
COLLECTOR="${BUNDLE}/cem_round_oracle.py"
DATA=/225010117/data/pusht_expert_train.h5
TRAIN=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_k3_n240_merged_v1.npz
OLD_EVAL=/225010117/logs/week1_selection/cem_round_h5_off40_n12_full_v2.npz
RESULT="${OUT}/source.npz"
LOG="${OUT}/source.log"

export STABLEWM_HOME=/225010117/stablewm
export HF_HOME=/225010117/cache/huggingface
export PYTHONPATH="${ROOT}/scripts/plan:${ROOT}:${BUNDLE}"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${OUT}"
cd "${ROOT}"

"${PY}" "${COLLECTOR}" \
    seed=20260726 \
    +plan_config.history_len=3 \
    plan_config.horizon=5 \
    plan_config.receding_horizon=5 \
    eval.goal_offset_steps=40 \
    eval.video=false \
    eval.dataset_name="${DATA}" \
    +audit.generators=pd_d192_k3_eval \
    +audit.scorers=pd_d192_k3_eval \
    +audit.num_states=60 \
    +audit.steps=4 \
    +audit.evaluate_simulator=false \
    "+audit.exclude_sources='${TRAIN},${OLD_EVAL}'" \
    +audit.out="${RESULT}" \
    >"${LOG}" 2>&1

touch "${OUT}/ALL.done"
echo "fresh model-only H5 source complete at $(date -Is)"
