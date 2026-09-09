#!/usr/bin/env bash
# TurboVLA LIBERO 4b training launcher (4x H100).
# Platform start command should be exactly ONE short line:
#   bash /data/260010028/dwh_vla/launch_4b.sh
# All real logic lives here so the platform input box cannot truncate it.
set -euo pipefail

BASE=/data/260010028/dwh_vla
PKG=$BASE/v2_code_bundle_20260906/package
RES=$BASE/v2_code_bundle_20260906/resources
PY=$BASE/miniconda3/envs/turbovla/bin/python
TR=$BASE/miniconda3/envs/turbovla/bin/torchrun

mkdir -p "$PKG/output"
LOG=$PKG/output/launch_4b_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "[INFO] launch log: $LOG"

# --- sanity checks (short ASCII lines only) ---
test -f "$PKG/run_train_v2.sh" || { echo "!!! wrong volume: run_train_v2.sh missing"; exit 1; }
nvidia-smi || true

"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available to torch"
n = torch.cuda.device_count()
assert n > 0, "no GPU visible to torch"
print("GPUs visible to torch:", n)
EOF

# --- GPU ids: honor platform-assigned CUDA_VISIBLE_DEVICES when present ---
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_IDS="$CUDA_VISIBLE_DEVICES"
  echo "[INFO] GPU_IDS from platform: $GPU_IDS"
else
  GPU_IDS="0,1,2,3"
  echo "[INFO] GPU_IDS default: $GPU_IDS"
fi

# --- training config ---
export PYTHON_BIN=$PY
export TORCHRUN_BIN=$TR
export REPO_DIR=$RES
export OFFICIAL_CKPT=$RES/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth
export TEXT_MODE=frozen
export BATCH_SIZE=16
export TARGET_GLOBAL_BATCH=256
export RESUME_MODE=all
export GPU_IDS

cd "$PKG"
exec bash run_train_v2.sh
