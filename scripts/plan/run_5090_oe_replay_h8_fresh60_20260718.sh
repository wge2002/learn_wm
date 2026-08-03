#!/usr/bin/env bash
set -euo pipefail

# 5090 exploratory bridge on the independent H8/off60 trace:
#   1. build an ordinary-dynamics replay cache disjoint from both 60-state
#      planner-query traces;
#   2. compare matched 3-fold OE fine-tuning with replay weights 0, 1, and 10;
#   3. summarize each arm without checkpoint picking.
#
# Replay weight 0 is the same-trace control, weight 1 is primary, and weight
# 10 is a preservation stress-control.  Epoch 5 is the predeclared endpoint.

ROOT=/mnt/data/wge/learn_wm
OUT="${ROOT}/outputs/week1/oe_replay_h8_5090_20260718"
PY="${ROOT}/.venv/bin/python"
DATA=/mnt/data/wge/data/pusht_eval_state_only.h5
POLICY=pd_d192_k3_eval
SOURCE_SEED=20260720
REPLAY_SEED=20260721
H8_TRACE="${ROOT}/outputs/week1/selection_round_5090/cem_round_h8_off60_k3_n60_seed${SOURCE_SEED}_v1.npz"
H5_TRACE="${ROOT}/outputs/week1/selection_round_5090/cem_round_h5_off40_k3_n60_seed20260719_v1.npz"
REPLAY="${OUT}/dynamics_replay_k3_n2048_seed${REPLAY_SEED}_v1.npz"

export CUDA_VISIBLE_DEVICES=0
export STABLEWM_HOME=/mnt/data/wge/stablewm
export HF_HOME=/mnt/data/wge/cache/huggingface
export PYTHONPATH="${ROOT}"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

mkdir -p "${OUT}"
cd "${ROOT}"

stage() {
    printf '[%s] %s\n' "$(date -Is)" "$*"
}

if [[ ! -s "${H8_TRACE}" ]]; then
    stage "missing H8 source trace: ${H8_TRACE}"
    exit 1
fi
if [[ ! -s "${H5_TRACE}" ]]; then
    stage "missing H5 exclusion trace: ${H5_TRACE}"
    exit 1
fi

if [[ ! -s "${REPLAY}" ]]; then
    stage "stage 1/2: independent 2048-window dynamics replay cache"
    "${PY}" scripts/plan/oe_build_replay_cache.py \
        seed="${REPLAY_SEED}" \
        +plan_config.history_len=3 \
        eval.video=false \
        eval.dataset_name="${DATA}" \
        +replay.out="${REPLAY}" \
        "+replay.policy='${POLICY}'" \
        +replay.num_windows=2048 \
        +replay.history_size=3 \
        +replay.action_block=5 \
        +replay.goal_offset=60 \
        +replay.exclusion_radius=60 \
        +replay.validation_fraction=0.2 \
        +replay.batch_size=64 \
        +replay.render_batch=64 \
        +replay.seed="${REPLAY_SEED}" \
        "+replay.exclude_sources='${H8_TRACE},${H5_TRACE}'" \
        2>&1 | tee -a "${OUT}/01_replay_cache.log"
else
    stage "stage 1/2 already complete: ${REPLAY}"
fi

fold_split() {
    local fold="$1"
    local train=""
    local val=""
    local index
    for ((index = 0; index < 60; index++)); do
        if ((index % 3 == fold)); then
            val="${val:+${val},}${index}"
        else
            train="${train:+${train},}${index}"
        fi
    done
    printf '%s|%s\n' "${train}" "${val}"
}

stage "stage 2/2: matched H8 OE cross-fit arms (replay weights 0, 1, 10)"
for replay_weight in 0 1 10; do
    arm="replay_w${replay_weight}"
    run_dirs=()
    for fold in 0 1 2; do
        IFS='|' read -r train_states val_states < <(fold_split "${fold}")
        run_name="oe_h8fresh60_${arm}_fold${fold}_v1"
        run_dir="${OUT}/${run_name}"
        run_dirs+=("${run_dir}")
        if [[ -s "${run_dir}/metrics.json" ]] \
            && grep -q '"epoch": 5' "${run_dir}/metrics.json"; then
            stage "already complete: ${run_name}"
            continue
        fi
        stage "train ${run_name}"
        "${PY}" scripts/plan/oe_fixed_trace_train.py \
            seed="$((SOURCE_SEED + fold))" \
            +plan_config.history_len=3 \
            plan_config.horizon=8 \
            plan_config.receding_horizon=8 \
            eval.goal_offset_steps=60 \
            eval.video=false \
            eval.dataset_name="${DATA}" \
            +oe.source="${H8_TRACE}" \
            "+oe.policy='${POLICY}'" \
            "+oe.source_generator='${POLICY}'" \
            +oe.run_name="${run_name}" \
            +oe.out_dir="${run_dir}" \
            +oe.epochs=5 \
            +oe.save_every=1 \
            +oe.save_weights=false \
            +oe.topk=30 \
            +oe.lr=2e-6 \
            +oe.weight_decay=1e-4 \
            +oe.temperature=0.2 \
            +oe.boundary_weight=1.0 \
            +oe.mean_weight=1.0 \
            +oe.logstd_weight=0.25 \
            +oe.anchor_weight=0.2 \
            +oe.relative_update_weight=1.0 \
            +oe.calibrate_elite_mass=true \
            +oe.replay_path="${REPLAY}" \
            +oe.replay_weight="${replay_weight}" \
            +oe.replay_batch_size=64 \
            +oe.replay_eval_batch_size=256 \
            "+oe.modules='action_encoder,predictor,pred_proj'" \
            "+oe.steps='4,9,19,29'" \
            "+oe.train_states='${train_states}'" \
            "+oe.val_states='${val_states}'" \
            2>&1 | tee -a "${OUT}/02_${run_name}.log"
    done

    summary_dir="${OUT}/oe_h8fresh60_${arm}_crossfit_v1_summary"
    stage "summarize ${arm}"
    "${PY}" scripts/plan/summarize_oe_fixed_trace.py \
        "${run_dirs[@]}" \
        --out-dir "${summary_dir}" \
        --bootstrap 20000 \
        --seed "$((REPLAY_SEED + replay_weight))" \
        2>&1 | tee -a "${OUT}/03_${arm}_summary.log"
done

stage "all 5090 H8 OE+replay exploratory arms complete"
touch "${OUT}/ALL.done"
