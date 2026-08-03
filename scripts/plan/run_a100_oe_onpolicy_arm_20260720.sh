#!/usr/bin/env bash
set -euo pipefail

# One formal planner-query aggregation arm. Environment variables select the
# causal intervention while the queried dataset rows, optimizer-update budget,
# fixed validation bank, and pressure bank remain matched across arms.

: "${ARM:?set ARM}"
: "${COLLECTOR_MODE:?set COLLECTOR_MODE to onpolicy or frozen}"
: "${BANK_MODE:?set BANK_MODE to cumulative or latest}"

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_oe_onpolicy_20260720
OUT_ROOT=/225010117/logs/oe_onpolicy_formal_a100_20260720
OUT="${OUT_ROOT}/${ARM}"
PY="${ROOT}/.venv-clean/bin/python"
DATA=/225010117/data/pusht_expert_train.h5
REPLAY=/225010117/logs/oe_replay_a100_20260718/dynamics_replay_k3_n2048_seed20260720_v1.npz
BASE_POLICY=pd_d192_k3_eval
START_POLICY="${START_POLICY:-oe_onpolicy_smoke_iter000_v1/weights_final.pt}"
QUERY_GENERATOR=onpolicy_queries
VALIDATION_GENERATOR=onpolicy_validation

COLLECTOR="${BUNDLE}/cem_round_oracle.py"
AUGMENTER="${BUNDLE}/oe_augment_trace_scorers.py"
MERGER="${BUNDLE}/oe_merge_query_banks.py"
TRAINER="${BUNDLE}/oe_fixed_trace_train.py"

OLD=/225010117/logs/oe_replay_a100_20260718/cem_round_h5_off40_k3_n60_seed20260720_v1.npz
A=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_a_k3_n60_seed20260721_v1.npz
B=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_b_k3_n60_seed20260722_v1.npz
VALIDATION=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_c_k3_n60_seed20260723_v1.npz
PRESSURE=/225010117/logs/operator_query_bank_a100_20260720/h8_off60_a_k3_n60_seed20260724_v1.npz

ROUNDS="${ROUNDS:-4}"
NEW_STATES="${NEW_STATES:-12}"
UPDATES_PER_ROUND="${UPDATES_PER_ROUND:-240}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.2}"
REPLAY_WEIGHT="${REPLAY_WEIGHT:-1}"
BASE_SEED="${BASE_SEED:-20260800}"

if [[ "${COLLECTOR_MODE}" != onpolicy && "${COLLECTOR_MODE}" != frozen ]]; then
    echo "invalid COLLECTOR_MODE=${COLLECTOR_MODE}" >&2
    exit 1
fi
if [[ "${BANK_MODE}" != cumulative && "${BANK_MODE}" != latest ]]; then
    echo "invalid BANK_MODE=${BANK_MODE}" >&2
    exit 1
fi
if ((ROUNDS < 1 || NEW_STATES < 1 || UPDATES_PER_ROUND < 1)); then
    echo "ROUNDS, NEW_STATES, and UPDATES_PER_ROUND must be positive" >&2
    exit 1
fi

export STABLEWM_HOME=/225010117/stablewm
export HF_HOME=/225010117/cache/huggingface
export PYTHONPATH="${ROOT}"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

mkdir -p "${OUT}"
cd "${ROOT}"

for required in \
    "${PY}" "${DATA}" "${REPLAY}" \
    "${COLLECTOR}" "${AUGMENTER}" "${MERGER}" "${TRAINER}" \
    "${OLD}" "${A}" "${B}" "${VALIDATION}" "${PRESSURE}"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing required input: ${required}" >&2
        exit 1
    fi
done

"${PY}" - "${OUT}/protocol.json" <<PY
import json
import sys

payload = {
    "version": 1,
    "arm": "${ARM}",
    "collector_mode": "${COLLECTOR_MODE}",
    "bank_mode": "${BANK_MODE}",
    "start_policy": "${START_POLICY}",
    "rounds": int("${ROUNDS}"),
    "new_states_per_round": int("${NEW_STATES}"),
    "updates_per_round": int("${UPDATES_PER_ROUND}"),
    "learning_rate": float("${LEARNING_RATE}"),
    "anchor_weight": float("${ANCHOR_WEIGHT}"),
    "replay_weight": float("${REPLAY_WEIGHT}"),
    "base_seed": int("${BASE_SEED}"),
    "fixed_validation": "${VALIDATION}",
    "pressure_test": "${PRESSURE}",
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\\n")
PY

augment_h5() {
    local source="$1"
    local output="$2"
    local policy="$3"
    local generator="$4"

    "${PY}" "${AUGMENTER}" \
        +plan_config.history_len=3 \
        plan_config.horizon=5 \
        plan_config.receding_horizon=5 \
        eval.goal_offset_steps=40 \
        eval.video=false \
        eval.dataset_name="${DATA}" \
        +augment.source="${source}" \
        +augment.out="${output}" \
        "+augment.scorers='${policy}'" \
        +augment.generator_name="${generator}"
}

train_round() {
    local source="$1"
    local validation="$2"
    local policy="$3"
    local run_name="$4"
    local run_dir="$5"
    local seed="$6"

    "${PY}" "${TRAINER}" \
        seed="${seed}" \
        +plan_config.history_len=3 \
        plan_config.horizon=5 \
        plan_config.receding_horizon=5 \
        eval.goal_offset_steps=40 \
        eval.video=false \
        eval.dataset_name="${DATA}" \
        +oe.source="${source}" \
        +oe.val_source="${validation}" \
        "+oe.policy='${policy}'" \
        "+oe.source_generator='${QUERY_GENERATOR}'" \
        "+oe.val_source_generator='${VALIDATION_GENERATOR}'" \
        +oe.run_name="${run_name}" \
        +oe.out_dir="${run_dir}" \
        +oe.epochs=1 \
        +oe.save_every=1 \
        +oe.save_weights=true \
        +oe.updates_per_epoch="${UPDATES_PER_ROUND}" \
        +oe.topk=30 \
        +oe.lr="${LEARNING_RATE}" \
        +oe.weight_decay=1e-4 \
        +oe.temperature=0.2 \
        +oe.boundary_weight=0 \
        +oe.mean_weight=1 \
        +oe.logstd_weight=0.25 \
        +oe.anchor_weight="${ANCHOR_WEIGHT}" \
        +oe.relative_update_weight=1 \
        +oe.calibrate_elite_mass=true \
        +oe.replay_path="${REPLAY}" \
        +oe.replay_weight="${REPLAY_WEIGHT}" \
        +oe.replay_batch_size=64 \
        +oe.replay_eval_batch_size=256 \
        "+oe.modules='action_encoder,predictor,pred_proj'" \
        "+oe.steps='4,9,19,29'"
}

raw_sources=("${OLD}" "${A}" "${B}")
current_policy="${START_POLICY}"

for ((round = 1; round <= ROUNDS; round++)); do
    printf -v round_tag '%03d' "${round}"
    round_dir="${OUT}/round_${round_tag}"
    mkdir -p "${round_dir}/rescored"
    seed=$((BASE_SEED + round))

    if [[ "${COLLECTOR_MODE}" == frozen ]]; then
        collector_policy="${BASE_POLICY}"
    else
        collector_policy="${current_policy}"
    fi

    exclusions=("${raw_sources[@]}" "${VALIDATION}" "${PRESSURE}")
    exclude_csv=""
    for item in "${exclusions[@]}"; do
        exclude_csv="${exclude_csv:+${exclude_csv},}${item}"
    done

    fresh="${round_dir}/fresh.npz"
    if [[ ! -e "${fresh}.done" ]]; then
        if [[ -e "${fresh}" ]]; then
            echo "partial fresh archive exists without done marker: ${fresh}" >&2
            exit 1
        fi
        "${PY}" "${COLLECTOR}" \
            seed="${seed}" \
            +plan_config.history_len=3 \
            plan_config.horizon=5 \
            plan_config.receding_horizon=5 \
            eval.goal_offset_steps=40 \
            eval.video=false \
            eval.dataset_name="${DATA}" \
            "+audit.generators='${collector_policy}'" \
            "+audit.scorers='${collector_policy}'" \
            +audit.num_states="${NEW_STATES}" \
            "+audit.steps='4,9,19,29'" \
            +audit.max_candidates=300 \
            "+audit.exclude_sources='${exclude_csv}'" \
            +audit.out="${fresh}" \
            >"${round_dir}/01_collect.log" 2>&1
        touch "${fresh}.done"
    fi
    raw_sources+=("${fresh}")

    if [[ "${BANK_MODE}" == cumulative ]]; then
        training_sources=("${raw_sources[@]}")
    else
        training_sources=("${fresh}")
    fi

    rescored_sources=()
    source_index=0
    for source in "${training_sources[@]}"; do
        output="${round_dir}/rescored/source_${source_index}.npz"
        augment_h5 \
            "${source}" \
            "${output}" \
            "${current_policy}" \
            "${QUERY_GENERATOR}" \
            >"${round_dir}/02_rescore_${source_index}.log" 2>&1
        rescored_sources+=("${output}")
        if [[ "${source}" == "${fresh}" ]]; then
            ln -sf "${output}" "${round_dir}/fresh_rescored.npz"
        fi
        source_index=$((source_index + 1))
    done

    validation_rescored="${round_dir}/validation_rescored.npz"
    augment_h5 \
        "${VALIDATION}" \
        "${validation_rescored}" \
        "${current_policy}" \
        "${VALIDATION_GENERATOR}" \
        >"${round_dir}/03_rescore_validation.log" 2>&1

    if ((${#rescored_sources[@]} == 1)); then
        training_bank="${rescored_sources[0]}"
    else
        training_bank="${round_dir}/training_bank.npz"
        "${PY}" "${MERGER}" "${rescored_sources[@]}" \
            --generator-name "${QUERY_GENERATOR}" \
            --output "${training_bank}" \
            >"${round_dir}/04_merge.log" 2>&1
    fi

    run_name="oe_onpolicy_${ARM}_iter${round_tag}_v1"
    train_round \
        "${training_bank}" \
        "${validation_rescored}" \
        "${current_policy}" \
        "${run_name}" \
        "${round_dir}/train" \
        "$((seed + 10000))" \
        >"${round_dir}/05_train.log" 2>&1
    current_policy="${run_name}/weights_final.pt"
    printf '%s\n' "${current_policy}" >"${round_dir}/next_policy.txt"
    touch "${round_dir}/DONE"
done

PRESSURE_OUT="${OUT}/pressure_final.npz"
"${PY}" "${AUGMENTER}" \
    +plan_config.history_len=3 \
    plan_config.horizon=8 \
    plan_config.receding_horizon=8 \
    eval.goal_offset_steps=60 \
    eval.video=false \
    eval.dataset_name="${DATA}" \
    +augment.source="${PRESSURE}" \
    +augment.out="${PRESSURE_OUT}" \
    "+augment.scorers='${current_policy}'" \
    +augment.generator_name=pressure_validation \
    >"${OUT}/pressure_final.log" 2>&1

printf '%s\n' "${current_policy}" >"${OUT}/final_policy.txt"
touch "${OUT}/ALL.done"
echo "formal arm ${ARM} complete at $(date -Is)"
