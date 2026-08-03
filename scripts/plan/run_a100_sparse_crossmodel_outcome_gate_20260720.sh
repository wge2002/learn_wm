#!/usr/bin/env bash
set -euo pipefail

# Fixed-trace budget audit: retain full K3 outcomes but query K10 vectors on
# only a deployably selected subset of candidates.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/sparse_crossmodel_outcome_gate_a100_20260720_v2_strict
PY="${ROOT}/.venv-clean/bin/python"
SOURCE=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_k3_n240_merged_v1.npz
K3=/225010117/logs/candidate_outcome_gate_a100_20260720/h5_candidate_outcomes.npz
K10=/225010117/logs/crossmodel_outcome_gate_a100_20260720/h5_k10_candidate_outcomes.npz
MASK="${BUNDLE}/oe_mask_outcome_cache.py"
PROBE="${BUNDLE}/oe_set_valued_operator_probe.py"

export PYTHONPATH="${ROOT}/scripts/plan:${ROOT}:${BUNDLE}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "${OUT}/cache"
cd "${ROOT}"

arms=(elite30 elite60 stratified30 stratified60)
queries=(30 60 30 60)
strategies=(elite elite rank_stratified rank_stratified)

for index in 0 1 2 3; do
    "${PY}" "${MASK}" \
        "${SOURCE}" "${K10}" \
        --out "${OUT}/cache/${arms[$index]}.npz" \
        --queries "${queries[$index]}" \
        --strategy "${strategies[$index]}"
done

pids=()
for gpu in 0 1 2 3; do
    name="${arms[$gpu]}"
    run_dir="${OUT}/${name}"
    log="${OUT}/${name}.log"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PY}" "${PROBE}" \
            "${SOURCE}" \
            --out-dir "${run_dir}" \
            --family planner_outcome \
            --outcome-cache "${K3}" \
            --extra-outcome-cache "${OUT}/cache/${name}.npz" \
            --topk 30 \
            --modes 5 \
            --hidden 128 \
            --attention-heads 4 \
            --lr 3e-4 \
            --weight-decay 1e-4 \
            --max-epochs 20 \
            --batch-populations 8 \
            --router-weight 0.1 \
            --router-kind winner_ce \
            --delta-anchor-weight 1e-3 \
            --bootstrap 20000 \
            --seed "$((20261220 + gpu))" \
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
    echo "one or more sparse cross-model probes failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "sparse cross-model outcome Gate complete at $(date -Is)"
