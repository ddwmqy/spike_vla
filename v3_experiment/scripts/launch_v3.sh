#!/usr/bin/env bash
# v3 (sootspike) launcher: ONE H100 80G (GPU 0), memory-filled micro-batch.
# Platform start command (one short line):
#   bash /data/260010028/dwh_vla/launch_v3.sh
# Cards 1-3 are reserved for other jobs - this launcher only touches GPU 0.
set -euo pipefail

BASE=/data/260010028/dwh_vla
PKG=$BASE/v2_code_bundle_20260906/package
RES=$BASE/v2_code_bundle_20260906/resources
PY=$BASE/miniconda3/envs/turbovla/bin/python
TR=$BASE/miniconda3/envs/turbovla/bin/torchrun

# sootspike (= SmoothSpike BERT) assets:
SOOT_WEIGHTS=$RES/pretrained/SmoothSpike/smoothspike-bert-base-fused
SOOT_SOURCE=$RES/third_party/SmoothSpike

mkdir -p "$PKG/output"
LOG=$PKG/output/launch_v3_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "[INFO] launch log: $LOG"

test -f "$PKG/run_train_v2.sh" || { echo "!!! wrong volume: run_train_v2.sh missing"; exit 1; }
test -d "$SOOT_WEIGHTS" || { echo "!!! sootspike weights not on disk yet: $SOOT_WEIGHTS"; exit 1; }
test -d "$SOOT_SOURCE" || { echo "!!! sootspike source not on disk yet: $SOOT_SOURCE"; exit 1; }
nvidia-smi || true

"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available to torch"
n = torch.cuda.device_count()
assert n > 0, "no GPU visible to torch"
print("GPUs visible to torch:", n)
EOF

# --- v3 training config ---
export GPU_IDS=0
export TEXT_ENCODER_TYPE=sootspike
export TEXT_WEIGHTS_PATH=$SOOT_WEIGHTS
export TEXT_SOURCE_PATH=$SOOT_SOURCE
export TEXT_MODE=frozen
# Memory-filled batch: global batch == micro-batch (accumulation 1).
# Probe result 2026-09-07 (full 80G H100, fp32): 48/56 OK, 64 OOM at step 1 -> 56 is the ceiling.
export BATCH_SIZE=56
export TARGET_GLOBAL_BATCH=$BATCH_SIZE
# Recipe-preserving alternative (global 256 via accumulation 4, ~4x slower wall time):
#   export TARGET_GLOBAL_BATCH=256
export RESUME_MODE=all
export PYTHON_BIN=$PY
export TORCHRUN_BIN=$TR
export REPO_DIR=$RES
export OFFICIAL_CKPT=$RES/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth
export CHECKPOINT_PREFIX=turbovla_v3_sootspike
export OUTPUT_ROOT=$PKG/output/train_v3

cd "$PKG"
exec bash run_train_v2.sh
