#!/usr/bin/env bash
set -Eeuo pipefail

EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QUEUE_LOG="$EXPERIMENT_ROOT/output/train/queue.log"
LOCK_FILE="$EXPERIMENT_ROOT/output/train/queue.lock"
NUM_GPUS="${NUM_GPUS:-4}"
MAX_MEMORY_MIB="${MAX_MEMORY_MIB:-1000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$(dirname -- "$QUEUE_LOG")"
exec 9>"$LOCK_FILE"
flock -n 9 || {
  echo "[ERROR] another queue watcher already holds $LOCK_FILE" >&2
  exit 1
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$QUEUE_LOG"
}

select_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F',' -v max_mem="$MAX_MEMORY_MIB" -v max_util="$MAX_UTILIZATION" '
      {
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3)
        if ($2 <= max_mem && $3 <= max_util) print $1
      }
    ' | head -n "$NUM_GPUS" | paste -sd, -
}

log "queue started: need $NUM_GPUS GPUs with memory <= ${MAX_MEMORY_MIB} MiB and utilization <= ${MAX_UTILIZATION}%"
previous=""
while true; do
  selected="$(select_gpus || true)"
  count=0
  if [[ -n "$selected" ]]; then
    IFS=',' read -r -a selected_array <<< "$selected"
    count="${#selected_array[@]}"
  fi
  if (( count == NUM_GPUS )) && [[ "$selected" == "$previous" ]]; then
    log "GPUs $selected passed two consecutive checks; launching training"
    exec env GPU_IDS="$selected" "$EXPERIMENT_ROOT/run_train.sh"
  fi
  if (( count == NUM_GPUS )); then
    log "candidate GPUs $selected; waiting one more poll for stability"
    previous="$selected"
  else
    log "waiting: found $count/$NUM_GPUS eligible GPUs"
    previous=""
  fi
  sleep "$POLL_SECONDS"
done
