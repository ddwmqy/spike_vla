#!/usr/bin/env bash
# v3 (sootspike) evaluation launcher - run this on the COMPUTE node.
# Usage:
#   bash /data/260010028/dwh_vla/eval_v3.sh                      # 4 GPUs, all 4 suites in parallel (~6h)
#   EVAL_GPU_IDS=0 bash /data/260010028/dwh_vla/eval_v3.sh       # single card, serial (~1 day)
#   EVAL_SUITES=libero_spatial bash /data/260010028/dwh_vla/eval_v3.sh   # subset
# Completed suites are fingerprint-cached: re-runs skip them automatically.
set -euo pipefail

BASE=/data/260010028/dwh_vla
PKG=$BASE/v2_code_bundle_20260906/package
RES=$BASE/v2_code_bundle_20260906/resources
PY=$BASE/miniconda3/envs/turbovla/bin/python
LIBERO_LOCAL=$BASE/LIBERO_eval          # on the shared /data volume, reused across pods
LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"

mkdir -p "$PKG/output/eval"
LOG=$PKG/output/eval/launch_eval_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "[INFO] launch log: $LOG"

test -f "$PKG/run_eval.sh" || { echo "!!! wrong volume: run_eval.sh missing"; exit 1; }
test -f "$PKG/output/train_v3/checkpoints/turbovla_v3_sootspike_80000.pth" \
  || { echo "!!! v3 80k checkpoint missing under $PKG/output/train_v3/checkpoints"; exit 1; }

# robosuite -> cv2 needs libGL; fresh platform containers ship without it.
if ! ldconfig -p | grep -q 'libGL.so.1'; then
  echo "[INFO] libGL missing -> installing via apt"
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgl1 libglib2.0-0 libegl1 libglx-mesa0
  ldconfig -p | grep -q 'libGL.so.1' || { echo "!!! libGL install failed - install it manually"; exit 1; }
fi

# LIBERO checkout at the pinned clean commit (clone via mirror if absent).
if [ ! -d "$LIBERO_LOCAL/libero/libero/assets" ]; then
  echo "[INFO] LIBERO checkout missing -> cloning via mirror"
  for M in "https://gitclone.com/github.com/Lifelong-Robot-Learning/LIBERO" \
           "https://gh-proxy.com/https://github.com/Lifelong-Robot-Learning/LIBERO" \
           "https://ghfast.top/https://github.com/Lifelong-Robot-Learning/LIBERO"; do
    git clone "$M" "$LIBERO_LOCAL" 2>/dev/null && [ -d "$LIBERO_LOCAL/.git" ] && break
    rm -rf "$LIBERO_LOCAL"
  done
  [ -d "$LIBERO_LOCAL/.git" ] || { echo "!!! LIBERO clone failed"; exit 1; }
fi
git config --global --add safe.directory "$LIBERO_LOCAL" 2>/dev/null || true
CUR=$(git -C "$LIBERO_LOCAL" rev-parse HEAD)
[ "$CUR" = "$LIBERO_COMMIT" ] || git -C "$LIBERO_LOCAL" checkout -q "$LIBERO_COMMIT"
DIRTY=$(git -C "$LIBERO_LOCAL" status --short | head -1)
[ -z "$DIRTY" ] || { echo "!!! LIBERO checkout is dirty: $DIRTY"; exit 1; }
echo "[INFO] LIBERO commit: $(git -C "$LIBERO_LOCAL" rev-parse HEAD)"

nvidia-smi || true
"$PY" - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available to torch"
print("GPUs visible to torch:", torch.cuda.device_count())
EOF

# --- v3 eval config ---
export EVAL_GPU_IDS="${EVAL_GPU_IDS:-0,1,2,3}"   # exactly 1 or exactly 4 GPUs (run_eval.sh enforces)
export EVAL_SUITES="${EVAL_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
export TRAIN_STEP=80000
export WEIGHT_SOURCE=raw                          # v3 ships no EMA; anchor comparison is raw<->raw
export LIBERO_CHECKOUT=$LIBERO_LOCAL
export SOURCE_CKPT=$PKG/output/train_v3/checkpoints/turbovla_v3_sootspike_${TRAIN_STEP}.pth
export BERT_PATH=$RES/pretrained/SmoothSpike/smoothspike-bert-base-fused
export TEXT_SOURCE_PATH=$RES/third_party/SmoothSpike
export EVAL_OUTPUT_ROOT=$PKG/output/eval/v3_sootspike_step_${TRAIN_STEP}_raw_clean_8f1084e
export PYTHON_BIN=$PY
export REPO_DIR=$RES
export MODEL_LABEL="libero_sdtv3_sootspike_snnfusion_20260907"

cd "$PKG"
exec bash run_eval.sh
