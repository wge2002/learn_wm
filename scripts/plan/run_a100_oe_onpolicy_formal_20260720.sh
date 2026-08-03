#!/usr/bin/env bash
set -euo pipefail

# Launch four matched causal arms after the end-to-end smoke test proves that
# descendant-checkpoint replay, fresh collection, cumulative rescoring, and
# checkpoint hand-off all close correctly.

BUNDLE=/225010117/a100_oe_onpolicy_20260720
OUT=/225010117/logs/oe_onpolicy_formal_a100_20260720
SMOKE=/225010117/logs/oe_onpolicy_smoke_a100_20260720
PY=/225010117/code/learn_wm/.venv-clean/bin/python
ARM_DRIVER="${BUNDLE}/run_a100_oe_onpolicy_arm_20260720.sh"
SUMMARIZER="${BUNDLE}/summarize_oe_onpolicy.py"
START_POLICY=oe_onpolicy_smoke_iter000_v1/weights_final.pt
START_CHECKPOINT="/225010117/stablewm/checkpoints/${START_POLICY}"

mkdir -p "${OUT}"

deadline=$((SECONDS + 3600))
while [[ ! -e "${SMOKE}/ALL.done" ]]; do
    if ((SECONDS >= deadline)); then
        echo "smoke test did not finish inside one hour" >&2
        exit 1
    fi
    sleep 15
done

for required in \
    "${PY}" "${ARM_DRIVER}" "${SUMMARIZER}" "${START_CHECKPOINT}"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing required input: ${required}" >&2
        exit 1
    fi
done

# gpu|arm|collector|bank|lr|anchor|replay
specs=(
    "0|frozen_cumulative|frozen|cumulative|2e-6|0.2|1"
    "1|onpolicy_latest|onpolicy|latest|2e-6|0.2|1"
    "2|onpolicy_cumulative|onpolicy|cumulative|2e-6|0.2|1"
    "3|onpolicy_trust|onpolicy|cumulative|1e-6|0.5|2"
)

pids=()
arm_dirs=()
for spec in "${specs[@]}"; do
    IFS='|' read -r gpu arm collector bank lr anchor replay <<< "${spec}"
    arm_dir="${OUT}/${arm}"
    mkdir -p "${arm_dir}"
    arm_dirs+=("${arm_dir}")
    echo "start gpu=${gpu} arm=${arm} at $(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export ARM="${arm}"
        export COLLECTOR_MODE="${collector}"
        export BANK_MODE="${bank}"
        export LEARNING_RATE="${lr}"
        export ANCHOR_WEIGHT="${anchor}"
        export REPLAY_WEIGHT="${replay}"
        export START_POLICY
        export ROUNDS=4
        export NEW_STATES=12
        export UPDATES_PER_ROUND=240
        export BASE_SEED=20260800
        bash "${ARM_DRIVER}" >"${arm_dir}/driver.log" 2>&1
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
    echo "one or more on-policy arms failed" >&2
    exit "${status}"
fi

"${PY}" "${SUMMARIZER}" "${arm_dirs[@]}" \
    --out-dir "${OUT}/summary" \
    >"${OUT}/summary.log" 2>&1

touch "${OUT}/ALL.done"
echo "formal on-policy experiment complete at $(date -Is)"
