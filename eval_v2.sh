#!/usr/bin/env bash
# v2 (SpikingLM) evaluation launcher - run this on the COMPUTE node.
# D2 目的:v3-vs-v2 对照臂。与 v3 同协议同 evaluator;对比必须 raw<->raw。
# 用法:
#   V2_CKPT=/path/to/turbovla_sdtv3_bert_snnfusion_learned_80000.pth \
#     bash /data/260010028/dwh_vla/eval_v2.sh          # 4 GPUs, all 4 suites (~5h)
#   V2_CKPT=... EVAL_GPU_IDS=0 bash .../eval_v2.sh     # single card, serial
#   V2_CKPT=... EVAL_SUITES=libero_goal bash ...       # subset (e.g. task5 对照)
#   WEIGHT_SOURCE=ema V2_CKPT=... bash ...             # supplementary EMA eval (own dir)
# ckpt 路径来自 locate_v2_assets.sh 的清单;脚本只做只读使用,不修改 ckpt。
set -euo pipefail

BASE=/data/260010028/dwh_vla
PKG=$BASE/v2_code_bundle_20260906/package
RES=$BASE/v2_code_bundle_20260906/resources
PY=$BASE/miniconda3/envs/turbovla/bin/python
LIBERO_LOCAL=$BASE/LIBERO_eval
LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"

mkdir -p "$PKG/output/eval"
LOG=$PKG/output/eval/launch_eval_v2_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "[INFO] launch log: $LOG"

V2_CKPT="${V2_CKPT:-}"
[[ -n "$V2_CKPT" ]] || { echo "!!! set V2_CKPT to the located v2 checkpoint path (see v2_assets_manifest.txt)"; exit 1; }
[[ -f "$V2_CKPT" ]] || { echo "!!! v2 checkpoint not found: $V2_CKPT"; exit 1; }

if ! ldconfig -p | grep -q 'libGL.so.1'; then
  echo "[INFO] libGL missing -> installing via apt"
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgl1 libglib2.0-0 libegl1 libglx-mesa0
  ldconfig -p | grep -q 'libGL.so.1' || { echo "!!! libGL install failed"; exit 1; }
fi

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

# --- v2 eval config ---
export EVAL_GPU_IDS="${EVAL_GPU_IDS:-0,1,2,3}"
export EVAL_SUITES="${EVAL_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
export TRAIN_STEP=80000
export WEIGHT_SOURCE="${WEIGHT_SOURCE:-raw}"      # default raw: v3-vs-v2 comparison is raw<->raw
export LIBERO_CHECKOUT=$LIBERO_LOCAL
export SOURCE_CKPT="$V2_CKPT"
export BERT_PATH=$RES/pretrained/SpikingLM        # v2 text encoder weights (SpikingBERT)
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$PKG/output/eval/v2_spikinglm_step_${TRAIN_STEP}_${WEIGHT_SOURCE}_clean_8f1084e}"
export PYTHON_BIN=$PY
export REPO_DIR=$RES
export MODEL_LABEL="${MODEL_LABEL:-libero_sdtv3_spikinglm_snnfusion_20260831}"

cd "$PKG"
exec bash run_eval.sh
