#!/usr/bin/env bash
set -euo pipefail

# End-to-end closure test for planner-query aggregation:
#   base K3 -> train -> collect fresh queries with the trained checkpoint
#   -> rescore the cumulative bank -> train the next checkpoint.
# The 60-state c shard is a fixed validation set and H8 is excluded from
# collection so it remains available as an untouched pressure test.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_oe_onpolicy_20260720
OUT=/225010117/logs/oe_onpolicy_smoke_a100_20260720
PY="${ROOT}/.venv-clean/bin/python"
DATA=/225010117/data/pusht_expert_train.h5
REPLAY=/225010117/logs/oe_replay_a100_20260718/dynamics_replay_k3_n2048_seed20260720_v1.npz
BASE_POLICY=pd_d192_k3_eval
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

export STABLEWM_HOME=/225010117/stablewm
export HF_HOME=/225010117/cache/huggingface
export PYTHONPATH="${ROOT}"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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

train_one_epoch() {
    local source="$1"
    local validation="$2"
    local policy="$3"
    local run_name="$4"
    local run_dir="$5"
    local val_generator="$6"
    local seed="$7"

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
        "+oe.val_source_generator='${val_generator}'" \
        +oe.run_name="${run_name}" \
        +oe.out_dir="${run_dir}" \
        +oe.epochs=1 \
        +oe.save_every=1 \
        +oe.save_weights=true \
        +oe.topk=30 \
        +oe.lr=2e-6 \
        +oe.weight_decay=1e-4 \
        +oe.temperature=0.2 \
        +oe.boundary_weight=0 \
        +oe.mean_weight=1 \
        +oe.logstd_weight=0.25 \
        +oe.anchor_weight=0.2 \
        +oe.relative_update_weight=1 \
        +oe.calibrate_elite_mass=true \
        +oe.replay_path="${REPLAY}" \
        +oe.replay_weight=1 \
        +oe.replay_batch_size=64 \
        +oe.replay_eval_batch_size=256 \
        "+oe.modules='action_encoder,predictor,pred_proj'" \
        "+oe.steps='4,9,19,29'"
}

augment_for_policy() {
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

BOOTSTRAP_BANK="${OUT}/bootstrap_k3_n180.npz"
"${PY}" "${MERGER}" "${OLD}" "${A}" "${B}" \
    --generator-name "${QUERY_GENERATOR}" \
    --output "${BOOTSTRAP_BANK}" \
    >"${OUT}/00_merge_bootstrap.log" 2>&1

ITER0_RUN=oe_onpolicy_smoke_iter000_v1
ITER0_POLICY="${ITER0_RUN}/weights_final.pt"
train_one_epoch \
    "${BOOTSTRAP_BANK}" \
    "${VALIDATION}" \
    "${BASE_POLICY}" \
    "${ITER0_RUN}" \
    "${OUT}/iter000_train" \
    "${BASE_POLICY}" \
    20260730 \
    >"${OUT}/01_train_iter000.log" 2>&1

FRESH="${OUT}/iter001_fresh_n2.npz"
"${PY}" "${COLLECTOR}" \
    seed=20260731 \
    +plan_config.history_len=3 \
    plan_config.horizon=5 \
    plan_config.receding_horizon=5 \
    eval.goal_offset_steps=40 \
    eval.video=false \
    eval.dataset_name="${DATA}" \
    "+audit.generators='${ITER0_POLICY}'" \
    "+audit.scorers='${ITER0_POLICY}'" \
    +audit.num_states=2 \
    "+audit.steps='4,9,19,29'" \
    +audit.max_candidates=300 \
    "+audit.exclude_sources='${OLD},${A},${B},${VALIDATION},${PRESSURE}'" \
    +audit.out="${FRESH}" \
    >"${OUT}/02_collect_iter001.log" 2>&1

RESCORED=()
source_index=0
for source in "${OLD}" "${A}" "${B}" "${FRESH}"; do
    output="${OUT}/iter001_rescored_${source_index}.npz"
    augment_for_policy \
        "${source}" \
        "${output}" \
        "${ITER0_POLICY}" \
        "${QUERY_GENERATOR}" \
        >"${OUT}/03_rescore_${source_index}.log" 2>&1
    RESCORED+=("${output}")
    source_index=$((source_index + 1))
done

VAL_RESCORED="${OUT}/iter001_validation_rescored.npz"
augment_for_policy \
    "${VALIDATION}" \
    "${VAL_RESCORED}" \
    "${ITER0_POLICY}" \
    "${VALIDATION_GENERATOR}" \
    >"${OUT}/04_rescore_validation.log" 2>&1

ITER1_BANK="${OUT}/iter001_cumulative_n182.npz"
"${PY}" "${MERGER}" "${RESCORED[@]}" \
    --generator-name "${QUERY_GENERATOR}" \
    --output "${ITER1_BANK}" \
    >"${OUT}/05_merge_iter001.log" 2>&1

ITER1_RUN=oe_onpolicy_smoke_iter001_v1
train_one_epoch \
    "${ITER1_BANK}" \
    "${VAL_RESCORED}" \
    "${ITER0_POLICY}" \
    "${ITER1_RUN}" \
    "${OUT}/iter001_train" \
    "${VALIDATION_GENERATOR}" \
    20260732 \
    >"${OUT}/06_train_iter001.log" 2>&1

touch "${OUT}/ALL.done"
echo "on-policy aggregation smoke complete at $(date -Is)"
