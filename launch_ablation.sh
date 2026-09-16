#!/usr/bin/env bash
# 消融臂启动器(单卡;训练配方对齐 A4 = v3 sootspike 80k:batch 56 / accum 1 / fp32 / TEXT_MODE=frozen)
#
# 用法(平台任务命令填一行):
#   bash /data/260010028/dwh_vla/launch_ablation.sh a0p     # DINOv3+BERT+ANN
#   bash /data/260010028/dwh_vla/launch_ablation.sh a1      # SDT-V3+BERT+ANN
#   bash /data/260010028/dwh_vla/launch_ablation.sh a2      # DINOv3+sootspike+ANN
#   bash /data/260010028/dwh_vla/launch_ablation.sh a3      # SDT-V3+sootspike+ANN
#   bash /data/260010028/dwh_vla/launch_ablation.sh probe:a0p   # 只探 batch 上限(30 步),不正式训
#   bash /data/260010028/dwh_vla/launch_ablation.sh dry:a1      # 只打印命令,不训练
#
# ★ 四臂必须用同一个 batch 才能横向比。DINOv3 视觉(DINOv3-B 85.66M)比 SDT-V3(18.63M)大得多,
#   所以先跑 probe:a0p 定上限,再用同一个 BATCH_SIZE 跑其余三臂。
set -Eeuo pipefail

BASE=/data/260010028/dwh_vla
PKG=$BASE/v2_code_bundle_20260906/package
RES=$BASE/v2_code_bundle_20260906/resources
# 环境:显式给 PYTHON_BIN 就用它;否则**自动挑一个能在本机 GPU 上真算的环境** ——
# H100 会选中现成的 turbovla(与 A4 同栈);5090 上它跑不动,自动落到 turbovla-bw(Blackwell 栈)。
pick_py() {
  local c
  for c in "$BASE/miniconda3/envs/turbovla/bin/python" \
           "$BASE/miniconda3/envs/turbovla-bw/bin/python" "$BASE/miniconda3/envs/"*/bin/python; do
    [ -x "$c" ] || continue
    "$c" -c 'import torch;a=torch.randn(512,512,device="cuda");(a@a).sum()' >/dev/null 2>&1 && { echo "$c"; return; }
  done
}
PY="${PYTHON_BIN:-$(pick_py)}"
PY="${PY:-$BASE/miniconda3/envs/turbovla/bin/python}"   # 都不可用时给个默认,让后续报错更明确
TR="${TORCHRUN_BIN:-$(dirname "$PY")/torchrun}"
echo "[INFO] python 环境: $PY"

SOOT_W=$RES/pretrained/SmoothSpike/smoothspike-bert-base-fused
SOOT_S=$RES/third_party/SmoothSpike
DINOV3=$RES/pretrained/dinov3-vitb16-pretrain-lvd1689m
OFFICIAL=$RES/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth

MODE="${1:-}"
if [ -z "$MODE" ]; then
  echo "用法: $0 a0p|a1|a2|a3|probe:<arm>|dry:<arm>"
  exit 1
fi
PROBE=0
DRYRUN="${DRY_RUN:-0}"
case "$MODE" in
  probe:*) PROBE=1; ARM="${MODE#probe:}" ;;
  dry:*)   DRYRUN=1; ARM="${MODE#dry:}" ;;
  *)       ARM="$MODE" ;;
esac

# dry: 必须把 DRY_RUN 透传给内层脚本,否则会真的开训(13 小时)
export DRY_RUN="$DRYRUN"

case "$ARM" in
  a0p) VISION_TYPE=dinov3    ; TEXT_ENCODER_TYPE=bert      ; DOWNSTREAM_TYPE=ann
       PREFIX=turbovla_ablation_A0p_dinov3_bert_ann ;;
  a1)  VISION_TYPE=sdtv3_19m ; TEXT_ENCODER_TYPE=bert      ; DOWNSTREAM_TYPE=ann
       PREFIX=turbovla_ablation_A1_sdtv3_bert_ann ;;
  a2)  VISION_TYPE=dinov3    ; TEXT_ENCODER_TYPE=sootspike ; DOWNSTREAM_TYPE=ann
       PREFIX=turbovla_ablation_A2_dinov3_sootspike_ann ;;
  a3)  VISION_TYPE=sdtv3_19m ; TEXT_ENCODER_TYPE=sootspike ; DOWNSTREAM_TYPE=ann
       PREFIX=turbovla_ablation_A3_sdtv3_sootspike_ann ;;
  a4)  echo "[INFO] A4 已有(92.2%),不重跑;如需复核请直接用 launch_v3.sh"; exit 0 ;;
  *)   echo "!!! 未知臂: $ARM(可选 a0p|a1|a2|a3)"; exit 1 ;;
esac

# ---- 资产预检(带病开训=浪费 13 小时) ----
test -f "$PKG/run_train_v2_ablation.sh" || { echo "!!! 缺启动脚本: $PKG/run_train_v2_ablation.sh"; exit 1; }
test -f "$OFFICIAL" || { echo "!!! 缺官方 KD 教师 ckpt: $OFFICIAL"; exit 1; }
test -d "$DINOV3"   || { echo "!!! 缺 DINOv3 权重: $DINOV3"; exit 1; }
if [ "$TEXT_ENCODER_TYPE" = sootspike ]; then
  test -d "$SOOT_W" || { echo "!!! 缺 sootspike 权重: $SOOT_W"; exit 1; }
  test -d "$SOOT_S" || { echo "!!! 缺 sootspike 源码: $SOOT_S"; exit 1; }
fi

if [ "$DRYRUN" != "1" ] && [ "$PROBE" != "1" ]; then
  "$PY" - <<'EOF' || exit 1
import torch
assert torch.cuda.is_available(), "CUDA 不可见"
print("GPU:", torch.cuda.get_device_name(0))
EOF
fi

export GPU_IDS="${GPU_IDS:-0}"
export TEXT_MODE="${TEXT_MODE:-frozen}"
export VISION_TYPE TEXT_ENCODER_TYPE DOWNSTREAM_TYPE
export CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-$PREFIX}"
# 四臂共用父目录,ckpt 落在 $PKG/output/ablation/checkpoints/<prefix>_<step>.pth
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PKG/output/ablation}"
export PYTHON_BIN=$PY TORCHRUN_BIN=$TR REPO_DIR=$RES OFFICIAL_CKPT=$OFFICIAL
export RESUME_MODE="${RESUME_MODE:-all}"

if [ "$VISION_TYPE" = dinov3 ]; then
  # DINOv3 走 HF 本地目录;SDT-V3 的三个路径置空不让它被误用
  export VISION_MODEL_PATH="$DINOV3" VISION_PRETRAINED_PATH= VISION_SOURCE_PATH=
fi
if [ "$TEXT_ENCODER_TYPE" = sootspike ]; then
  export TEXT_WEIGHTS_PATH="$SOOT_W" TEXT_SOURCE_PATH="$SOOT_S"
fi

if [ "$PROBE" = "1" ]; then
  # 只探 batch 上限:逐档 30 步,报峰值显存与 s/it
  export OUTPUT_ROOT="$PKG/output/ablation_probe"
  mkdir -p "$OUTPUT_ROOT"
  # 阈值按**本机卡的实际容量**算,别用 H100 的 78GB 硬编码(5090 只有 32GB)
  CAP_MIB=$("$PY" - <<'EOF' 2>/dev/null || echo 0
import torch
print(int(torch.cuda.get_device_properties(0).total_memory / 2**20))
EOF
)
  LIMIT_MIB="${LIMIT_MIB:-$(( CAP_MIB * 95 / 100 ))}"
  echo "[INFO] 本机 GPU 容量 ${CAP_MIB} MiB,可用上限(95%) ${LIMIT_MIB} MiB"
  for B in "${BATCH_SIZES:-32 48 56 64}"; do
    echo "=== probe arm=$ARM batch=$B ==="
    BATCH_SIZE=$B TARGET_GLOBAL_BATCH=$B MAX_STEPS=30 WARMUP_STEPS=8 RESUME_MODE=none \
      CHECKPOINT_PREFIX="${PREFIX}_probe_b$B" OUTPUT_ROOT="$OUTPUT_ROOT/train_b$B" \
      bash "$PKG/run_train_v2_ablation.sh" > "$OUTPUT_ROOT/probe_${ARM}_b$B.log" 2>&1 &
    PID=$!
    PEAK=0
    while kill -0 $PID 2>/dev/null; do
      M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_IDS%%,*}" 2>/dev/null || echo 0)
      [ "${M:-0}" -gt "$PEAK" ] 2>/dev/null && PEAK=$M
      sleep 2
    done
    wait $PID; RC=$?
    SPIT=$(grep -oE '[0-9.]+s/it' "$OUTPUT_ROOT/probe_${ARM}_b$B.log" | tail -1)
    OK="✓"
    { [ "$RC" = "0" ] && [ "${PEAK:-0}" -le "$LIMIT_MIB" ]; } 2>/dev/null || OK="✗"
    echo "  arm=$ARM batch=$B rc=$RC peak=${PEAK}MiB$([ "$OK" = "✗" ] && echo '(超上限/失败)') speed=${SPIT:-n/a} log=$OUTPUT_ROOT/probe_${ARM}_b$B.log"
  done
  echo "=== probe 结束:取 rc=0 且峰值 ≤ ${LIMIT_MIB}MiB 的最大 batch,四臂统一使用 ==="
  exit 0
fi

export BATCH_SIZE="${BATCH_SIZE:-56}"
export TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-$BATCH_SIZE}"   # accum=1,同 A4

mkdir -p "$PKG/output"
LOG="$PKG/output/ablation_${ARM}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "[INFO] arm=$ARM vision=$VISION_TYPE text=$TEXT_ENCODER_TYPE fusion=$DOWNSTREAM_TYPE batch=$BATCH_SIZE"
echo "[INFO] log: $LOG"

cd "$PKG"
exec bash run_train_v2_ablation.sh
