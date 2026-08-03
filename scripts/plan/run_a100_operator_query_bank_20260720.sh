#!/usr/bin/env bash
set -euo pipefail

# Build the larger, cross-horizon planner-query bank required by the next
# Operator-only training stage.  Three independent H5 shards establish the
# main training distribution; one H8 shard is an out-of-horizon pressure set.
# Every saved population retains all 300 candidates and simulator labels.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_oe_replay_20260718
OUT=/225010117/logs/operator_query_bank_a100_20260720
PY="${ROOT}/.venv-clean/bin/python"
DATA=/225010117/data/pusht_expert_train.h5
POLICY=pd_d192_k3_eval

export STABLEWM_HOME=/225010117/stablewm
export HF_HOME=/225010117/cache/huggingface
export PYTHONPATH="${ROOT}"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

mkdir -p "${OUT}"
cd "${ROOT}"

for required in "${PY}" "${DATA}" "${BUNDLE}/cem_round_oracle.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing required input: ${required}" >&2
        exit 1
    fi
done

# gpu|cell|seed|horizon|goal_offset
shards=(
    "0|h5_off40_a|20260721|5|40"
    "1|h5_off40_b|20260722|5|40"
    "2|h5_off40_c|20260723|5|40"
    "3|h8_off60_a|20260724|8|60"
)

pids=()
for spec in "${shards[@]}"; do
    IFS='|' read -r gpu cell seed horizon goal_offset <<< "${spec}"
    archive="${OUT}/${cell}_k3_n60_seed${seed}_v1.npz"
    log="${OUT}/${cell}_k3_n60_seed${seed}_v1.log"

    if [[ -s "${archive}" ]] && grep -q '^elapsed=' "${log}"; then
        echo "already complete: ${cell}"
        continue
    fi

    echo "start gpu=${gpu} cell=${cell} seed=${seed} at $(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PY}" "${BUNDLE}/cem_round_oracle.py" \
            seed="${seed}" \
            +plan_config.history_len=3 \
            plan_config.horizon="${horizon}" \
            plan_config.receding_horizon="${horizon}" \
            eval.goal_offset_steps="${goal_offset}" \
            eval.video=false \
            eval.dataset_name="${DATA}" \
            "+audit.generators='${POLICY}'" \
            "+audit.scorers='${POLICY}'" \
            +audit.num_states=60 \
            "+audit.steps='4,9,19,29'" \
            +audit.max_candidates=300 \
            +audit.out="${archive}" \
            >"${log}" 2>&1
        touch "${archive}.done"
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
    echo "one or more query-bank shards failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "all operator query-bank shards complete at $(date -Is)"
