#!/usr/bin/env bash
# v3 batch probe: find the largest micro-batch that fits ONE H100 80G (fp32).
# Run on the compute node (inside the platform task), after sootspike weights land.
# Usage: bash run_batch_probe.sh [sizes...]   (default sweep: 48 56 64 72)
# For each size: MAX_STEPS=30 training run + nvidia-smi polling -> peak VRAM + s/it.
set -uo pipefail

BASE=/data/260010028/dwh_vla
PKG=$BASE/v2_code_bundle_20260906/package
RES=$BASE/v2_code_bundle_20260906/resources
PY=$BASE/miniconda3/envs/turbovla/bin/python
TR=$BASE/miniconda3/envs/turbovla/bin/torchrun

SIZES=("$@")
if [ ${#SIZES[@]} -eq 0 ]; then SIZES=(48 56 64 72); fi
OUT=$PKG/output/batch_probe
mkdir -p "$OUT"

for B in "${SIZES[@]}"; do
  echo "=== probe batch_size=$B ==="
  GPU_IDS=0 \
  TEXT_ENCODER_TYPE=sootspike \
  TEXT_WEIGHTS_PATH=$RES/pretrained/SmoothSpike/smoothspike-bert-base-fused \
  TEXT_SOURCE_PATH=$RES/third_party/SmoothSpike \
  TEXT_MODE=frozen \
  BATCH_SIZE=$B \
  TARGET_GLOBAL_BATCH=$B \
  MAX_STEPS=30 \
  WARMUP_STEPS=8 \
  RESUME_MODE=none \
  CHECKPOINT_PREFIX=probe_b$B \
  OUTPUT_ROOT=$OUT/train_b$B \
  PYTHON_BIN=$PY \
  TORCHRUN_BIN=$TR \
  REPO_DIR=$RES \
  OFFICIAL_CKPT=$RES/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth \
  bash "$PKG/run_train_v2.sh" > "$OUT/probe_b$B.log" 2>&1 &
  TRAIN_PID=$!
  PEAK=0
  while kill -0 $TRAIN_PID 2>/dev/null; do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null || echo 0)
    if [ "$M" -gt "$PEAK" ]; then PEAK=$M; fi
    sleep 2
  done
  wait $TRAIN_PID
  RC=$?
  SPIT=$(grep -oE '[0-9.]+s/it' "$OUT/probe_b$B.log" | tail -1)
  echo "batch=$B rc=$RC peak=${PEAK}MiB speed=${SPIT:-n/a} log=$OUT/probe_b$B.log"
done
echo "=== probe done; pick the largest batch with rc=0 and peak < 78000 MiB ==="
