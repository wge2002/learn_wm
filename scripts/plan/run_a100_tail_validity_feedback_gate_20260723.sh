#!/usr/bin/env bash
set -euo pipefail

# Four-shard PushT feedback-channel falsification gate.  The correction is
# active in every CEM round; simulator oracle labels are read only after each
# arm has produced its final population.

ROOT=/225010117/code/learn_wm
OUT=/225010117/logs/tail_validity_feedback_gate_a100_20260723
PY=/225010117/code/learn_wm/.venv-clean/bin/python
DATA=/225010117/data/pusht_expert_train.h5
SOURCE=/225010117/logs/basin_lineage_pair_a100_20260720/h5_off40_k3_k10_n60_seed20260735_v1.npz
COLLECTOR=${ROOT}/scripts/plan/tail_validity_feedback_gate.py
SUMMARIZER=${ROOT}/scripts/plan/summarize_tail_validity_feedback_gate.py

export STABLEWM_HOME=/225010117/stablewm
export HF_HOME=/225010117/cache/huggingface

mkdir -p "${OUT}/summary"
for required in "${PY}" "${DATA}" "${SOURCE}" "${COLLECTOR}" "${SUMMARIZER}"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing required input: ${required}" >&2
        exit 1
    fi
done

specs=(
    "0|0|15"
    "1|15|15"
    "2|30|15"
    "3|45|15"
)

pids=()
shards=()
for spec in "${specs[@]}"; do
    IFS='|' read -r gpu start count <<< "${spec}"
    stop=$((start + count))
    shard="${OUT}/formal_${start}_${stop}.npz"
    log="${OUT}/formal_${start}_${stop}.log"
    shards+=("${shard}")
    if [[ -s "${shard}" ]]; then
        echo "reuse completed shard: ${shard}"
        continue
    fi
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PY}" "${COLLECTOR}" \
            policy=pd_d192_k3_eval \
            +plan_config.history_len=3 \
            plan_config.horizon=5 \
            plan_config.receding_horizon=1 \
            eval.goal_offset_steps=40 \
            eval.video=false \
            eval.dataset_name="${DATA}" \
            seed=20260740 \
            +feedback.source="${SOURCE}" \
            +feedback.policy=pd_d192_k3_eval \
            "+feedback.alphas='-1,0,0.5,1'" \
            +feedback.state_start="${start}" \
            +feedback.num_states="${count}" \
            +feedback.prefix_blocks=1 \
            +feedback.out="${shard}" \
            >"${log}" 2>&1
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
    echo "one or more feedback-gate shards failed" >&2
    exit "${status}"
fi

"${PY}" "${SUMMARIZER}" "${shards[@]}" \
    --out-json "${OUT}/summary/report.json" \
    --out-md "${OUT}/summary/REPORT.md" \
    >"${OUT}/summary/summarize.log" 2>&1

touch "${OUT}/ALL.done"
echo "tail-validity feedback gate complete: ${OUT}/summary/REPORT.md"
