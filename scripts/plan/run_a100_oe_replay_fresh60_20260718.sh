#!/usr/bin/env bash
set -euo pipefail

# A100 GPU3 exploratory bridge:
#   1. collect a fresh 60-state K3 planner-query trace;
#   2. build an independent frozen-latent dynamics replay cache;
#   3. compare matched 3-fold OE fine-tuning with replay weights 0, 1, and 10.
#
# The trace seed is intentionally different from the 5090 bridge seed.  The
# replay-free arm is a matched control; weight 1 is primary and weight 10 is a
# preservation stress-control.  This is an exploratory method test, not an
# end-to-end MPC claim.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_oe_replay_20260718
OUT=/225010117/logs/oe_replay_a100_20260718
PY="${ROOT}/.venv-clean/bin/python"
DATA=/225010117/data/pusht_expert_train.h5
POLICY=pd_d192_k3_eval
SEED=20260720
TRACE="${OUT}/cem_round_h5_off40_k3_n60_seed${SEED}_v1.npz"
REPLAY="${OUT}/dynamics_replay_k3_n2048_seed${SEED}_v1.npz"

export CUDA_VISIBLE_DEVICES=3
export STABLEWM_HOME=/225010117/stablewm
export HF_HOME=/225010117/cache/huggingface
export PYTHONPATH="${ROOT}"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

mkdir -p "${OUT}"
cd "${ROOT}"

stage() {
    printf '[%s] %s\n' "$(date -Is)" "$*"
}

if [[ ! -s "${TRACE}" ]]; then
    stage "stage 1/3: fresh 60-state K3 H5/off40 planner-query trace"
    "${PY}" "${BUNDLE}/cem_round_oracle.py" \
        seed="${SEED}" \
        +plan_config.history_len=3 \
        plan_config.horizon=5 \
        plan_config.receding_horizon=5 \
        eval.goal_offset_steps=40 \
        eval.video=false \
        eval.dataset_name="${DATA}" \
        "+audit.generators='${POLICY}'" \
        "+audit.scorers='${POLICY}'" \
        +audit.num_states=60 \
        "+audit.steps='4,9,19,29'" \
        +audit.max_candidates=300 \
        +audit.out="${TRACE}" \
        2>&1 | tee -a "${OUT}/01_trace.log"
else
    stage "stage 1/3 already complete: ${TRACE}"
fi

if [[ ! -s "${REPLAY}" ]]; then
    stage "stage 2/3: independent 2048-window dynamics replay cache"
    "${PY}" "${BUNDLE}/oe_build_replay_cache.py" \
        seed="${SEED}" \
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
        +replay.seed="${SEED}" \
        +replay.exclude_sources="${TRACE}" \
        2>&1 | tee -a "${OUT}/02_replay_cache.log"
else
    stage "stage 2/3 already complete: ${REPLAY}"
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

stage "stage 3/3: matched OE cross-fit arms (replay weights 0, 1, 10)"
for replay_weight in 0 1 10; do
    arm="replay_w${replay_weight}"
    for fold in 0 1 2; do
        IFS='|' read -r train_states val_states < <(fold_split "${fold}")
        run_name="oe_fresh60_${arm}_fold${fold}_v1"
        run_dir="${OUT}/${run_name}"
        if [[ -s "${run_dir}/metrics.json" ]] \
            && grep -q '"epoch": 5' "${run_dir}/metrics.json"; then
            stage "already complete: ${run_name}"
            continue
        fi
        stage "train ${run_name}"
        "${PY}" "${BUNDLE}/oe_fixed_trace_train.py" \
            seed="$((SEED + fold))" \
            +plan_config.history_len=3 \
            plan_config.horizon=5 \
            plan_config.receding_horizon=5 \
            eval.goal_offset_steps=40 \
            eval.video=false \
            eval.dataset_name="${DATA}" \
            +oe.source="${TRACE}" \
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
            2>&1 | tee -a "${OUT}/03_${run_name}.log"
    done
done

stage "all A100 GPU3 OE+replay exploratory arms complete"
touch "${OUT}/ALL.done"
