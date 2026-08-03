#!/usr/bin/env bash
set -euo pipefail

# Locked 240-state Operator-only scale gate.  Four GPUs each own one held-out
# fold, so every state is evaluated exactly once by a model that did not train
# on it.  This is the final fixed-bank gate before any long/full training.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_oe_replay_20260718
BANK_ROOT=/225010117/logs/operator_query_bank_a100_20260720
OUT=/225010117/logs/operator_largebank_crossfit_a100_20260720
PY="${ROOT}/.venv-clean/bin/python"
DATA=/225010117/data/pusht_expert_train.h5
POLICY=pd_d192_k3_eval
SEED=20260725
MERGED="${BANK_ROOT}/h5_off40_k3_n240_merged_v1.npz"
REPLAY=/225010117/logs/oe_replay_a100_20260718/dynamics_replay_k3_n2048_seed20260720_v1.npz
MERGER="${BUNDLE}/oe_merge_query_banks.py"
TRAINER="${BUNDLE}/oe_fixed_trace_train.py"
SUMMARIZER="${BUNDLE}/summarize_oe_fixed_trace.py"

export STABLEWM_HOME=/225010117/stablewm
export HF_HOME=/225010117/cache/huggingface
export PYTHONPATH="${ROOT}"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

mkdir -p "${OUT}"
cd "${ROOT}"

sources=(
    /225010117/logs/oe_replay_a100_20260718/cem_round_h5_off40_k3_n60_seed20260720_v1.npz
    "${BANK_ROOT}/h5_off40_a_k3_n60_seed20260721_v1.npz"
    "${BANK_ROOT}/h5_off40_b_k3_n60_seed20260722_v1.npz"
    "${BANK_ROOT}/h5_off40_c_k3_n60_seed20260723_v1.npz"
)

for required in "${PY}" "${DATA}" "${REPLAY}" "${MERGER}" "${TRAINER}" \
    "${SUMMARIZER}" "${sources[@]}"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing required input: ${required}" >&2
        exit 1
    fi
done

if [[ ! -s "${MERGED}" ]]; then
    "${PY}" "${MERGER}" "${sources[@]}" --output "${MERGED}" \
        >"${OUT}/00_merge.log" 2>&1
fi

fold_split() {
    local fold="$1"
    local train=""
    local val=""
    local index
    for ((index = 0; index < 240; index++)); do
        if ((index % 4 == fold)); then
            val="${val:+${val},}${index}"
        else
            train="${train:+${train},}${index}"
        fi
    done
    printf '%s|%s\n' "${train}" "${val}"
}

pids=()
run_dirs=()
for gpu in 0 1 2 3; do
    fold="${gpu}"
    IFS='|' read -r train_states val_states < <(fold_split "${fold}")
    run_name="oe_operator_n240_fold${fold}_v1"
    run_dir="${OUT}/${run_name}"
    log="${OUT}/${run_name}.log"
    run_dirs+=("${run_dir}")

    if [[ -s "${run_dir}/metrics.json" ]] \
        && grep -q '"epoch": 5' "${run_dir}/metrics.json"; then
        echo "already complete: ${run_name}"
        continue
    fi

    echo "start gpu=${gpu} fold=${fold} at $(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PY}" "${TRAINER}" \
            seed="${SEED}" \
            +plan_config.history_len=3 \
            plan_config.horizon=5 \
            plan_config.receding_horizon=5 \
            eval.goal_offset_steps=40 \
            eval.video=false \
            eval.dataset_name="${DATA}" \
            +oe.source="${MERGED}" \
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
            "+oe.steps='4,9,19,29'" \
            "+oe.train_states='${train_states}'" \
            "+oe.val_states='${val_states}'" \
            >"${log}" 2>&1
        touch "${run_dir}/DONE"
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
    echo "one or more cross-fit folds failed" >&2
    exit "${status}"
fi

"${PY}" "${SUMMARIZER}" "${run_dirs[@]}" \
    --out-dir "${OUT}/summary" \
    --bootstrap 20000 \
    --seed "${SEED}" \
    >"${OUT}/summary.log" 2>&1

touch "${OUT}/ALL.done"
echo "operator large-bank cross-fit complete at $(date -Is)"
