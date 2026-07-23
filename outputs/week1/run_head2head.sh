#!/usr/bin/env bash
# Head-to-head: does one-step + geometry regularizer (Temporal Straightening
# curvature / Invariant-JEPA bisim) produce the far-goal composition gain that
# coupled multi-step (anchor+dose) does? Pre-registered: NO — geometry regs
# improve on-trajectory geometry but not far-goal open-loop composition.
# D=192, 2 seeds each, 30 epochs; then near/far-goal eval + certificates.
# Usage: NGPU=4 bash outputs/week1/run_head2head.sh   (set PY/DS for A100 box)
set -Eeuo pipefail
PY=${PY:-python}
DS=${DS:-$HOME/.stable_worldmodel/pusht_expert_train.h5}
PIXELS=${PIXELS:-${DS%.h5}_pixels.npy}
OUT=${OUT:-outputs/week1}
NGPU=${NGPU:-4}
WORKERS=${WORKERS:-2}
PREFETCH=${PREFETCH:-1}
PHASES=${PHASES:-train,eval}
CKPT_ROOT=${STABLEWM_HOME:-$HOME/.stable_worldmodel}/checkpoints
export SWM_TORCH_THREADS=${SWM_TORCH_THREADS:-2}
mkdir -p "$OUT"; test -f "$DS"; test "$NGPU" -ge 1
declare -a PIDS=() JOBS=()
I=0
wait_batch() { local f=0 i; for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then echo "DONE ${JOBS[$i]}"; else echo "FAILED ${JOBS[$i]}"; f=1; fi
  done; PIDS=(); JOBS=(); return "$f"; }
has() { case ",$PHASES," in *,"$1",*) return 0;; *) return 1;; esac; }

train() { # NAME SEED CONFIG
  [ -f "$CKPT_ROOT/$1/weights_epoch_30.pt" ] && { echo "SKIP $1"; return; }
  local gpu=$((I % NGPU))
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train/lewm.py --config-name "$3" \
    output_model_name="$1" subdir="$1" seed="$2" \
    trainer.max_epochs=30 trainer.devices=1 \
    loader.num_workers="$WORKERS" loader.prefetch_factor="$PREFETCH" \
    data.dataset.name="$DS" +data.dataset.pixels_path="$PIXELS" \
    > "$OUT/train_$1.log" 2>&1 &
  PIDS+=("$!"); JOBS+=("train_$1"); I=$((I+1))
  [ "${#PIDS[@]}" -eq "$NGPU" ] && wait_batch
}
evalm() { # NAME OFF SEED
  local pol gpu=$((I % NGPU))
  if [ -f "$CKPT_ROOT/$1/weights_epoch_30.pt" ]; then pol="$1/weights_epoch_30.pt";
  elif [ -d "$CKPT_ROOT/${1}_eval" ]; then pol="${1}_eval"; else echo "SKIP eval $1"; return; fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/plan/eval_wm.py \
    policy="$pol" seed="$3" +plan_config.history_len=3 \
    eval.goal_offset_steps="$2" eval.num_eval=50 eval.video=false \
    eval.dataset_name="$DS" output.filename="h2h_${1}_off${2}_seed${3}.txt" \
    > "$OUT/plan_${1}_off${2}_s${3}.log" 2>&1 &
  PIDS+=("$!"); JOBS+=("plan_${1}_off${2}_s${3}"); I=$((I+1))
  [ "${#PIDS[@]}" -eq "$NGPU" ] && wait_batch
}

if has train; then test -f "$PIXELS"
  train curv_d192    3072 lewm_curv
  train curv_d192_s1 1    lewm_curv
  train bisim_d192    3072 lewm_bisim
  train bisim_d192_s1 1    lewm_bisim
  [ "${#PIDS[@]}" -gt 0 ] && wait_batch; echo "H2H TRAININGS DONE $(date -Is)"
fi
if has eval; then
  for NAME in curv_d192 curv_d192_s1 bisim_d192 bisim_d192_s1; do
    for OFF in 25 40; do for SEED in 42 123 7; do evalm "$NAME" "$OFF" "$SEED"; done; done
  done
  [ "${#PIDS[@]}" -gt 0 ] && wait_batch; echo "H2H EVALS DONE $(date -Is)"
fi
echo "HEAD2HEAD DONE $(date -Is)"
