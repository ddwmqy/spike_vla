#!/usr/bin/env bash
# 消融臂评测启动器(LIBERO,与 eval_v3.sh 同协议:fp32 · seed 7 · 50 trials · chunk 12 · open-loop 12 · LIBERO 8f1084e)
#
# 用法(在算力服务器上跑):
#   bash /data/260010028/dwh_vla/eval_ablation.sh a1
#   EVAL_SUITES=libero_spatial bash /data/260010028/dwh_vla/eval_ablation.sh a0   # 漂移核对(官方锚点)
#   WEIGHT_SOURCE=ema bash /data/260010028/dwh_vla/eval_ablation.sh a3
#   EVAL_GPU_IDS=0 bash /data/260010028/dwh_vla/eval_ablation.sh a2               # 单卡串行
#
# ★ 关键(容易踩):policy.py:411 会**无条件**用启动器给的 bert_path 覆盖 ckpt 里记的文本路径。
#   所以每个臂必须传对应的文本目录,传错会静默加载错误的文本编码器:
#     BERT 臂(A0'/A1)   → resources/pretrained/bert-base-uncased
#     sootspike 臂(A2/A3/A4) → resources/pretrained/SmoothSpike/smoothspike-bert-base-fused
#   (run_eval.sh 的指纹包含 BERT_PATH,所以不同臂不会互相命中缓存。)
set -Eeuo pipefail

BASE=/data/260010028/dwh_vla
PKG=$BASE/v2_code_bundle_20260906/package
RES=$BASE/v2_code_bundle_20260906/resources
PY="${PYTHON_BIN:-$BASE/miniconda3/envs/turbovla/bin/python}"
LIBERO_LOCAL=$BASE/LIBERO_eval
ABL_ROOT=$PKG/output/ablation/checkpoints

ARM="${1:-}"
[ -n "$ARM" ] || { echo "用法: $0 a0|a0p|a1|a2|a3|a4"; exit 1; }
export TRAIN_STEP="${TRAIN_STEP:-80000}"

case "$ARM" in
  # a0 = 官方发布 ckpt:外部锚点 + 漂移核对基准。
  #   漂移核对用 **raw**(归档基准 libero_spatial 468/500 = 93.6% 就是 raw 口径)
  a0)
    export SOURCE_CKPT=$RES/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth
    export BERT_PATH=$RES/pretrained/bert-base-uncased
    export TEXT_SOURCE_PATH=""
    export MODEL_LABEL="official_TurboVLA_LIBERO"
    export EVAL_OUTPUT_ROOT="$PKG/output/eval/ablation_anchor_official"
    WS_DEFAULT=raw
    APPEND_WS=1
    ;;
  # a0p/a1/a2/a3 = 消融主矩阵:主口径 **ema**(设计文档 §2.1;run_eval.sh 自身默认也是 ema)
  a0p)
    export SOURCE_CKPT=$ABL_ROOT/turbovla_ablation_A0p_dinov3_bert_ann_${TRAIN_STEP}.pth
    export BERT_PATH=$RES/pretrained/bert-base-uncased
    export TEXT_SOURCE_PATH=""
    export MODEL_LABEL="ablation_A0p_DINOv3-B_BERT_ANN-fusion_ACT"
    export EVAL_OUTPUT_ROOT="$PKG/output/eval/ablation_A0p_step_${TRAIN_STEP}"
    WS_DEFAULT=ema
    APPEND_WS=1
    ;;
  a1)
    export SOURCE_CKPT=$ABL_ROOT/turbovla_ablation_A1_sdtv3_bert_ann_${TRAIN_STEP}.pth
    export BERT_PATH=$RES/pretrained/bert-base-uncased
    export TEXT_SOURCE_PATH=""
    export MODEL_LABEL="ablation_A1_SDTV3_BERT_ANN-fusion_ACT"
    export EVAL_OUTPUT_ROOT="$PKG/output/eval/ablation_A1_step_${TRAIN_STEP}"
    WS_DEFAULT=ema
    APPEND_WS=1
    ;;
  a2)
    export SOURCE_CKPT=$ABL_ROOT/turbovla_ablation_A2_dinov3_sootspike_ann_${TRAIN_STEP}.pth
    export BERT_PATH=$RES/pretrained/SmoothSpike/smoothspike-bert-base-fused
    export TEXT_SOURCE_PATH=$RES/third_party/SmoothSpike
    export MODEL_LABEL="ablation_A2_DINOv3-B_sootspike_ANN-fusion_ACT"
    export EVAL_OUTPUT_ROOT="$PKG/output/eval/ablation_A2_step_${TRAIN_STEP}"
    WS_DEFAULT=ema
    APPEND_WS=1
    ;;
  a3)
    export SOURCE_CKPT=$ABL_ROOT/turbovla_ablation_A3_sdtv3_sootspike_ann_${TRAIN_STEP}.pth
    export BERT_PATH=$RES/pretrained/SmoothSpike/smoothspike-bert-base-fused
    export TEXT_SOURCE_PATH=$RES/third_party/SmoothSpike
    export MODEL_LABEL="ablation_A3_SDTV3_sootspike_ANN-fusion_ACT"
    export EVAL_OUTPUT_ROOT="$PKG/output/eval/ablation_A3_step_${TRAIN_STEP}"
    WS_DEFAULT=ema
    APPEND_WS=1
    ;;
  # a4 = 已有参考臂;EMA 92.20%(1844/2000)、raw 81.45%(1629/2000) 均已存在。
  #   默认 ema 且输出目录与既有 EMA 目录一致 → 重复调用会自动命中指纹缓存跳过。
  a4)
    export SOURCE_CKPT=$PKG/output/train_v3/checkpoints/turbovla_v3_sootspike_${TRAIN_STEP}.pth
    export BERT_PATH=$RES/pretrained/SmoothSpike/smoothspike-bert-base-fused
    export TEXT_SOURCE_PATH=$RES/third_party/SmoothSpike
    export MODEL_LABEL="libero_sdtv3_sootspike_snnfusion_20260907"
    export EVAL_OUTPUT_ROOT="$PKG/output/eval/v3_sootspike_step_${TRAIN_STEP}_${WEIGHT_SOURCE:-ema}_clean_8f1084e"
    WS_DEFAULT=ema
    APPEND_WS=0   # 该臂目录名把口径放在中间(..._ema_clean_...),不能再拼后缀
    ;;
  *) echo "!!! 未知臂: $ARM(可选 a0|a0p|a1|a2|a3|a4)"; exit 1 ;;
esac

# 口径:未显式指定时用该臂默认(a0=raw 对齐归档漂移基准;其余=ema 主口径)。
# 必须在 case 之后解析,且输出目录带口径后缀 —— 否则 raw/ema 两次评测会互相覆盖。
export WEIGHT_SOURCE="${WEIGHT_SOURCE:-$WS_DEFAULT}"
if [ "${APPEND_WS:-0}" = "1" ]; then
  export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}_${WEIGHT_SOURCE}"
fi

# DRY_RUN=1 只打印解析结果(可在 ckpt 还没训出来时先验证路径映射)
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY] arm=$ARM"
  for v in SOURCE_CKPT BERT_PATH TEXT_SOURCE_PATH MODEL_LABEL EVAL_OUTPUT_ROOT TRAIN_STEP; do
    printf '  %-18s %s\n' "$v" "${!v:-}"
  done
  printf '  %-18s %s\n' "ckpt exists" "$([ -f "$SOURCE_CKPT" ] && echo yes || echo NO)"
  printf '  %-18s %s\n' "text dir exists" "$([ -d "$BERT_PATH" ] && echo yes || echo NO)"
  exit 0
fi

test -f "$SOURCE_CKPT" || { echo "!!! 找不到 ckpt: $SOURCE_CKPT"; exit 1; }
test -d "$BERT_PATH"   || { echo "!!! 找不到文本目录: $BERT_PATH"; exit 1; }

export EVAL_GPU_IDS="${EVAL_GPU_IDS:-0,1,2,3}"
export EVAL_SUITES="${EVAL_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
# ★ git 的 dubious ownership:/data 上的仓库属主是 UID 11192,容器里是别的用户 →
#   run_eval.sh 里的 `git -C $LIBERO_CHECKOUT rev-parse HEAD` 会 exit 128
#   (2026-09-17 实测:四臂评测全部启动即挂)。eval_v3.sh 里有这行,我之前漏了。
for d in "$LIBERO_LOCAL" "$PKG" "$PKG/code"; do
  git config --global --add safe.directory "$d" 2>/dev/null || true
done
# ★ 图形库:cv2 要 libGL.so.1、MuJoCo 的 EGL 后端要 libEGL.so.1 —— 容器(尤其新拉起的)
#   两个通常都没有。eval_v3.sh 用 apt 解决,这里照抄那一套(经 v3 2000 局评测验证过);
#   apt 不通的集群(如 5090)才退回共享卷的 glvnd(只兜底 libGL,不遮蔽系统库)。
need_gl=0
ldconfig -p 2>/dev/null | grep -q "libGL.so.1"  || need_gl=1
ldconfig -p 2>/dev/null | grep -q "libEGL.so.1" || need_gl=1
if [ "$need_gl" = "1" ]; then
  echo "[INFO] 缺 libGL/libEGL → apt 安装(限时 120s)"
  timeout 120 bash -c 'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgl1 libglib2.0-0 libegl1 libglx-mesa0' || \
    echo "[WARN] apt 安装失败/超时(某些集群 apt 不通)"
fi
SYSLIBS=$BASE/v4_assets/syslibs
if ! ldconfig -p 2>/dev/null | grep -q "libGL.so.1" && [ -e "$SYSLIBS/libGL.so.1" ]; then
  export LD_LIBRARY_PATH="$SYSLIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "[WARN] 系统无 libGL,退回共享卷: $SYSLIBS(注意:卷上没有 libEGL)"
fi
export LIBERO_CHECKOUT=$LIBERO_LOCAL
export PYTHON_BIN=$PY
export REPO_DIR=$RES

echo "[INFO] arm=$ARM ckpt=$SOURCE_CKPT"
echo "[INFO] text=$BERT_PATH  source=${TEXT_SOURCE_PATH:-<none>}  label=$MODEL_LABEL"
echo "[INFO] suites=$EVAL_SUITES gpus=$EVAL_GPU_IDS source=$WEIGHT_SOURCE"
echo "[INFO] out=$EVAL_OUTPUT_ROOT"

cd "$PKG"
exec bash run_eval.sh
