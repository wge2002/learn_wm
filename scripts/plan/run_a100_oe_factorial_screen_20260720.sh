#!/usr/bin/env bash
set -euo pipefail

# Four-card loss-attribution screen on the independent A100 60-state trace.
#
# This is the locked, cheap precursor to full planner-query aggregation.  It
# separates preservation replay, oracle boundary ranking, CEM update matching,
# and their interaction while keeping the data, split, optimization budget,
# and initial K3 checkpoint identical.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_oe_replay_20260718
OUT=/225010117/logs/oe_factorial_a100_20260720_v2
PY="${ROOT}/.venv-clean/bin/python"
DATA=/225010117/data/pusht_expert_train.h5
POLICY=pd_d192_k3_eval
SEED=20260720
TRACE=/225010117/logs/oe_replay_a100_20260718/cem_round_h5_off40_k3_n60_seed20260720_v1.npz
REPLAY=/225010117/logs/oe_replay_a100_20260718/dynamics_replay_k3_n2048_seed20260720_v1.npz

export STABLEWM_HOME=/225010117/stablewm
export HF_HOME=/225010117/cache/huggingface
export PYTHONPATH="${ROOT}"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

mkdir -p "${OUT}"
cd "${ROOT}"

for required in "${PY}" "${TRACE}" "${REPLAY}" \
    "${BUNDLE}/oe_fixed_trace_train.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing required input: ${required}" >&2
        exit 1
    fi
done

fold=0
train_states=""
val_states=""
for ((index = 0; index < 60; index++)); do
    if ((index % 3 == fold)); then
        val_states="${val_states:+${val_states},}${index}"
    else
        train_states="${train_states:+${train_states},}${index}"
    fi
done

# arm|boundary|mean|logstd|relative|anchor
arms=(
    "data|0|0|0|0|0"
    "rank|1|0|0|0|0.2"
    "operator|0|1|0.25|1|0.2"
    "rank_operator|1|1|0.25|1|0.2"
)

pids=()
for gpu in 0 1 2 3; do
    IFS='|' read -r arm boundary mean logstd relative anchor \
        <<< "${arms[${gpu}]}"
    run_name="oe_factorial_${arm}_fold0_v2"
    run_dir="${OUT}/${run_name}"
    log="${OUT}/${run_name}.log"

    if [[ -s "${run_dir}/metrics.json" ]] \
        && grep -q '"epoch": 5' "${run_dir}/metrics.json"; then
        echo "already complete: ${run_name}"
        continue
    fi

    echo "start gpu=${gpu} arm=${arm} at $(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PY}" "${BUNDLE}/oe_fixed_trace_train.py" \
            seed="${SEED}" \
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
            +oe.save_weights=true \
            +oe.topk=30 \
            +oe.lr=2e-6 \
            +oe.weight_decay=1e-4 \
            +oe.temperature=0.2 \
            +oe.boundary_weight="${boundary}" \
            +oe.mean_weight="${mean}" \
            +oe.logstd_weight="${logstd}" \
            +oe.anchor_weight="${anchor}" \
            +oe.relative_update_weight="${relative}" \
            +oe.calibrate_elite_mass=true \
            +oe.replay_path="${REPLAY}" \
            +oe.replay_weight=1.0 \
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
    echo "one or more arms failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "all four factorial arms complete at $(date -Is)"
