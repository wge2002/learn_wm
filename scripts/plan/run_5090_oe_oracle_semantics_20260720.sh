#!/usr/bin/env bash
set -euo pipefail

# Stage-0 oracle-semantics audit for Search/Operator-Aligned LeWM.
#
# It reuses stored terminal simulator states, so no candidate is executed.
# The frozen K3 encoder turns each true terminal observation into a visual
# oracle cost. H5/off40 and H8/off60 must both pass the predeclared cell gate
# before visual-oracle query-aggregation training is justified.

ROOT=/mnt/data/wge/learn_wm
OUT="${ROOT}/outputs/week1/oe_oracle_semantics_5090_20260720"
PY="${ROOT}/.venv/bin/python"
DATA=/mnt/data/wge/data/pusht_eval_state_only.h5
POLICY=pd_d192_k3_eval
TRACE_ROOT="${ROOT}/outputs/week1/selection_round_5090"
H5_TRACE="${TRACE_ROOT}/cem_round_h5_off40_n12_full_v2.npz"
H8_TRACE="${TRACE_ROOT}/cem_round_h8_off60_n12_full_v2.npz"

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

for source in "${H5_TRACE}" "${H8_TRACE}"; do
    if [[ ! -s "${source}" ]]; then
        stage "missing source trace: ${source}"
        exit 1
    fi
done

run_cell() {
    local label="$1"
    local horizon="$2"
    local offset="$3"
    local source="$4"
    # v2 explicitly reindexes Pymunk shapes after exact terminal-state
    # restoration. The preflight v1 omitted this and could render a stale
    # framebuffer, so v1 must not be used for scientific conclusions.
    local output="${OUT}/${label}_visual_semantics_v2.npz"
    local report="${OUT}/${label}_visual_semantics_v2_report.md"
    if [[ -s "${output}" && -s "${report}" ]]; then
        stage "already complete: ${label}"
        return
    fi
    stage "audit ${label}"
    "${PY}" scripts/plan/oe_oracle_semantics_audit.py \
        seed=20260720 \
        +plan_config.history_len=3 \
        plan_config.horizon="${horizon}" \
        plan_config.receding_horizon="${horizon}" \
        eval.goal_offset_steps="${offset}" \
        eval.video=false \
        eval.dataset_name="${DATA}" \
        +semantics.source="${source}" \
        +semantics.out="${output}" \
        "+semantics.policy='${POLICY}'" \
        "+semantics.generator='${POLICY}'" \
        "+semantics.scorer='${POLICY}'" \
        "+semantics.steps='4,9,19,29'" \
        +semantics.topk=30 \
        +semantics.batch_size=128 \
        +semantics.bootstrap=20000 \
        +semantics.seed=20260720 \
        +semantics.gate_update_cosine=0.7 \
        +semantics.gate_elite_overlap=0.5 \
        +semantics.gate_recovery_fraction=0.5 \
        2>&1 | tee -a "${OUT}/${label}.log"
}

run_cell h5_off40 5 40 "${H5_TRACE}"
run_cell h8_off60 8 60 "${H8_TRACE}"

stage "both oracle-semantics cells complete"
touch "${OUT}/ALL.done"
