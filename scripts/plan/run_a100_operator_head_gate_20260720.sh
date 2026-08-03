#!/usr/bin/env bash
set -euo pipefail

# Strict fixed-trace Gate for a frozen-WM planner-conditioned operator head.
# Two deployable feature families (current planner state and causal planner
# history) are tested with and without frozen LeWM state/goal latents.  The
# privileged PushT state family is only an upper-bound diagnostic.  Every
# reported prediction is nested state-cross-fitted.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/operator_head_gate_a100_20260720
PY="${ROOT}/.venv-clean/bin/python"
DATA=/225010117/data/pusht_expert_train.h5
H5=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_k3_n240_merged_v1.npz
H8=/225010117/logs/operator_query_bank_a100_20260720/h8_off60_a_k3_n60_seed20260724_v1.npz
H5_LATENT="${OUT}/h5_off40_k3_n240_latent.npz"
H8_LATENT="${OUT}/h8_off60_k3_n60_latent.npz"

export STABLEWM_HOME=/225010117/stablewm
export HF_HOME=/225010117/cache/huggingface
export PYTHONPATH="${ROOT}:${BUNDLE}"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "${OUT}"
cd "${ROOT}"

for required in \
    "${PY}" "${DATA}" "${H5}" "${H8}" \
    "${BUNDLE}/oe_build_trace_latent_cache.py" \
    "${BUNDLE}/oe_update_corrector_probe.py" \
    "${BUNDLE}/oe_update_mode_codebook_probe.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing required input: ${required}" >&2
        exit 1
    fi
done

pids=()

(
    export CUDA_VISIBLE_DEVICES=0
    "${PY}" "${BUNDLE}/oe_build_trace_latent_cache.py" \
        +plan_config.history_len=3 \
        plan_config.horizon=5 \
        plan_config.receding_horizon=5 \
        eval.goal_offset_steps=40 \
        eval.video=false \
        eval.dataset_name="${DATA}" \
        +latent.source="${H5}" \
        +latent.out="${H5_LATENT}" \
        "+latent.policy='pd_d192_k3_eval'" \
        >"${OUT}/00_h5_latent.log" 2>&1
) &
pids+=("$!")

(
    export CUDA_VISIBLE_DEVICES=1
    "${PY}" "${BUNDLE}/oe_build_trace_latent_cache.py" \
        +plan_config.history_len=3 \
        plan_config.horizon=8 \
        plan_config.receding_horizon=8 \
        eval.goal_offset_steps=60 \
        eval.video=false \
        eval.dataset_name="${DATA}" \
        +latent.source="${H8}" \
        +latent.out="${H8_LATENT}" \
        "+latent.policy='pd_d192_k3_eval'" \
        >"${OUT}/00_h8_latent.log" 2>&1
) &
pids+=("$!")

(
    "${PY}" "${BUNDLE}/oe_update_mode_codebook_probe.py" \
        "${H5}" \
        --out-dir "${OUT}/h5_codebook_top30" \
        --topk 30 \
        --clusters 4,8,16 \
        --bootstrap 20000 \
        --seed 20260820 \
        >"${OUT}/00_h5_codebook.log" 2>&1
) &
pids+=("$!")

(
    "${PY}" "${BUNDLE}/oe_update_mode_codebook_probe.py" \
        "${H8}" \
        --out-dir "${OUT}/h8_codebook_top30" \
        --topk 30 \
        --clusters 4,8,16 \
        --bootstrap 20000 \
        --seed 20260821 \
        >"${OUT}/00_h8_codebook.log" 2>&1
) &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
if ((status != 0)); then
    echo "latent-cache or codebook preflight failed" >&2
    exit "${status}"
fi

pids=()

(
    "${PY}" "${BUNDLE}/oe_update_corrector_probe.py" \
        "${H5}" \
        --out-dir "${OUT}/h5_planner" \
        --topk 30 \
        --families planner,planner_history,planner_state_oracle \
        --bootstrap 20000 \
        --seed 20260830 \
        >"${OUT}/01_h5_planner.log" 2>&1
) &
pids+=("$!")

(
    "${PY}" "${BUNDLE}/oe_update_corrector_probe.py" \
        "${H5}" \
        --latent-caches "${H5_LATENT}" \
        --out-dir "${OUT}/h5_latent" \
        --topk 30 \
        --families planner_latent,planner_history_latent \
        --bootstrap 20000 \
        --seed 20260831 \
        >"${OUT}/01_h5_latent.log" 2>&1
) &
pids+=("$!")

(
    "${PY}" "${BUNDLE}/oe_update_corrector_probe.py" \
        "${H8}" \
        --out-dir "${OUT}/h8_planner" \
        --topk 30 \
        --families planner,planner_history,planner_state_oracle \
        --bootstrap 20000 \
        --seed 20260832 \
        >"${OUT}/01_h8_planner.log" 2>&1
) &
pids+=("$!")

(
    "${PY}" "${BUNDLE}/oe_update_corrector_probe.py" \
        "${H8}" \
        --latent-caches "${H8_LATENT}" \
        --out-dir "${OUT}/h8_latent" \
        --topk 30 \
        --families planner_latent,planner_history_latent \
        --bootstrap 20000 \
        --seed 20260833 \
        >"${OUT}/01_h8_latent.log" 2>&1
) &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
if ((status != 0)); then
    echo "one or more operator-head probes failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "operator-head Gate complete at $(date -Is)"
