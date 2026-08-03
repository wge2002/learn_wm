#!/usr/bin/env bash
set -euo pipefail

# Collect paired K3/K10 on-policy CEM traces on one exact fresh-60 state set.
# Each generator gets its own adaptive query path, while common initial noise
# and identical rows make the first population and downstream outcomes paired.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/basin_lineage_pair_a100_20260720
PY="${ROOT}/.venv-clean/bin/python"
DATA=/225010117/data/pusht_expert_train.h5
STATE_SOURCE=/225010117/logs/bp_oe_recursive_a100_20260720/fresh_eval_source_h5_off40_n60_seed20260727_v1/source.npz
MODELS=pd_d192_k3_eval,pd_d192_k10_eval

export STABLEWM_HOME=/225010117/stablewm
export HF_HOME=/225010117/cache/huggingface
export PYTHONPATH="${ROOT}"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

mkdir -p "${OUT}"
cd "${ROOT}"

for required in \
    "${PY}" \
    "${DATA}" \
    "${STATE_SOURCE}" \
    "${BUNDLE}/cem_round_oracle.py" \
    "${BUNDLE}/oe_merge_query_banks.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing required input: ${required}" >&2
        exit 1
    fi
done

pids=()
for shard in 0 1 2 3; do
    start=$((shard * 15))
    archive="${OUT}/shard_${start}_$((start + 15)).npz"
    log="${OUT}/shard_${start}_$((start + 15)).log"
    if [[ -s "${archive}" ]] && grep -q '^elapsed=' "${log}"; then
        echo "already complete: shard=${shard}"
        continue
    fi
    echo "start gpu=${shard} rows=[${start},$((start + 15))) at $(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES="${shard}"
        "${PY}" "${BUNDLE}/cem_round_oracle.py" \
            seed=20260735 \
            +plan_config.history_len=3 \
            plan_config.horizon=5 \
            plan_config.receding_horizon=5 \
            eval.goal_offset_steps=40 \
            eval.video=false \
            eval.dataset_name="${DATA}" \
            "+audit.generators='${MODELS}'" \
            "+audit.scorers='${MODELS}'" \
            +audit.num_states=15 \
            "+audit.steps='4,9,19,29'" \
            +audit.max_candidates=300 \
            +audit.state_source="${STATE_SOURCE}" \
            +audit.state_start="${start}" \
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
    echo "one or more paired-lineage shards failed" >&2
    exit "${status}"
fi

"${PY}" "${BUNDLE}/oe_merge_query_banks.py" \
    "${OUT}"/shard_*.npz \
    --output "${OUT}/h5_off40_k3_k10_n60_seed20260735_v1.npz" \
    >"${OUT}/merge.log" 2>&1

touch "${OUT}/ALL.done"
echo "paired basin-lineage bank complete at $(date -Is)"
