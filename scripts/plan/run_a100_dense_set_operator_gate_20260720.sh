#!/usr/bin/env bash
set -euo pipefail

# H5 gate for the population-conditioned dense spatial read path.  The frozen
# LeWM patch grid is kept intact and queried jointly with the complete CEM
# population.  M=1 tests direct continuous correction; M=5 tests whether
# branch retention plus dense geometry makes the correction modes routable.

ROOT=/225010117/code/learn_wm
BUNDLE=/225010117/a100_operator_head_20260720
OUT=/225010117/logs/dense_set_operator_gate_a100_20260720
PY="${ROOT}/.venv-clean/bin/python"
SOURCE=/225010117/logs/operator_query_bank_a100_20260720/h5_off40_k3_n240_merged_v1.npz
DENSE=/225010117/logs/dense_token_gate_a100_20260720/h5_dense_tokens.npz
PROBE="${BUNDLE}/oe_set_valued_operator_probe.py"

export PYTHONPATH="${ROOT}/scripts/plan:${ROOT}:${BUNDLE}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "${OUT}"
cd "${ROOT}"

names=(
    dense_m1_seed0
    dense_m5_seed0
    dense_m5_seed1
    dense_m5_router1
)
modes=(1 5 5 5)
seeds=(20260920 20260920 20260921 20260920)
router_weights=(0.1 0.1 0.1 1.0)

pids=()
for gpu in 0 1 2 3; do
    name="${names[$gpu]}"
    run_dir="${OUT}/${name}"
    log="${OUT}/${name}.log"
    echo "start gpu=${gpu} arm=${name} at $(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PY}" "${PROBE}" \
            "${SOURCE}" \
            --out-dir "${run_dir}" \
            --family planner_dense \
            --dense-cache "${DENSE}" \
            --topk 30 \
            --modes "${modes[$gpu]}" \
            --hidden 128 \
            --attention-heads 4 \
            --lr 3e-4 \
            --weight-decay 1e-4 \
            --max-epochs 20 \
            --batch-populations 8 \
            --router-weight "${router_weights[$gpu]}" \
            --delta-anchor-weight 1e-3 \
            --bootstrap 20000 \
            --seed "${seeds[$gpu]}" \
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
    echo "one or more dense set-operator probes failed" >&2
    exit "${status}"
fi

touch "${OUT}/ALL.done"
echo "dense set-operator H5 gate complete at $(date -Is)"
