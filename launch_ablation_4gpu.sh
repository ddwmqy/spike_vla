#!/usr/bin/env bash
# 消融四臂**并行**跑满 4 张 H100:每卡一个臂,每臂 batch 56 / accum 1 / fp32 / TEXT_MODE=frozen
#   —— 与 A4(v3 sootspike,92.2%)完全同配方,可比性不打折。
#
# 为什么不把 4 卡 DDP 到单臂:
#   4 卡 DDP 时若维持全局 56 就得每卡 micro-batch 14,而 SDT-V3 主干用 BatchNorm,
#   micro-batch 变了 BN 统计量就变 → 与 A4 不可比。四臂并行则每臂都是"1 卡 × batch 56",
#   与 A4 逐项一致,且总墙钟同为 ~13 h(四臂同时出)。
#
# 用法(平台任务命令填这一行,前台跑,勿 nohup):
#   bash /data/260010028/dwh_vla/launch_ablation_4gpu.sh
#   DRY=1 bash /data/260010028/dwh_vla/launch_ablation_4gpu.sh        # 只打印四臂命令
#   ARMS="a1,a3" GPUS="0,1" bash ...                                  # 只用 2 张卡
#   MAX_STEPS=2000 bash ...                                           # 短跑自检
#
# 说明:
#   - 每臂独立 TMPDIR(共享临时目录是并行跑的经典坑),日志各自一份到 parallel_<ts>/
#   - 启动错开 STAGGER 秒,避免四个进程同时压共享盘的 TFDS 初始化
#   - RESUME_MODE=all:任务被抢占后重启会从最近 ckpt 续训
set -Eeuo pipefail

BASE=/data/260010028/dwh_vla
PKG=$BASE/v2_code_bundle_20260906/package
LAUNCHER=$BASE/launch_ablation.sh

ARMS_STR="${ARMS:-a0p,a1,a2,a3}"
GPUS_STR="${GPUS:-0,1,2,3}"
STAGGER="${STAGGER:-45}"
DRY="${DRY:-0}"
PROBE="${PROBE:-0}"          # PROBE=1 先用一张卡探 batch 上限(56 已在 40G MIG 上验过,默认跳过)
export BATCH_SIZE="${BATCH_SIZE:-56}"
export MAX_STEPS="${MAX_STEPS:-80000}"

IFS=',' read -r -a ARMS <<< "$ARMS_STR"
IFS=',' read -r -a GPUS <<< "$GPUS_STR"
(( ${#ARMS[@]} == ${#GPUS[@]} )) || { echo "!!! ARMS 与 GPUS 数量不一致"; exit 1; }

# GPU 不能重复(两条臂挤同一张卡会 OOM)
dup=$(printf '%s\n' "${GPUS[@]}" | sort | uniq -d | tr '\n' ' ')
[ -z "$dup" ] || { echo "!!! GPUS 有重复: $dup —— 四臂必须各占一卡"; exit 1; }

# 臂名不能重复(否则 Checkpoint 前缀相同会互相覆盖)
dup_arm=$(printf '%s\n' "${ARMS[@]}" | sort | uniq -d | tr '\n' ' ')
[ -z "$dup_arm" ] || { echo "!!! ARMS 有重复: $dup_arm"; exit 1; }

test -x "$LAUNCHER" || { echo "!!! 缺 $LAUNCHER"; exit 1; }

echo "==================== 四臂并行 ===================="
echo "臂 × GPU : $(for i in "${!ARMS[@]}"; do printf '%s→GPU%s ' "${ARMS[$i]}" "${GPUS[$i]}"; done)"
echo "配方     : batch=$BATCH_SIZE accum=1 fp32 TEXT_MODE=frozen max_steps=$MAX_STEPS (对齐 A4)"
echo "TRAIN_STEP/口径: 训练无口径之分;评测用 eval_ablation.sh(默认 ema)"
echo "================================================="

if [ "$DRY" = "1" ]; then
  for i in "${!ARMS[@]}"; do
    echo "--- ${ARMS[$i]} @ GPU${GPUS[$i]}"
    GPU_IDS="${GPUS[$i]}" BATCH_SIZE="$BATCH_SIZE" MAX_STEPS="$MAX_STEPS" \
      DRY_RUN=1 bash "$LAUNCHER" "dry:${ARMS[$i]}" 2>&1 | sed 's/^/    /'
  done
  exit 0
fi

# 环境自动选择(与 launch_ablation.sh 同一策略):挑一个能在本机 GPU 上真算的
pick_py() {
  local c
  for c in "$BASE/miniconda3/envs/turbovla/bin/python" \
           "$BASE/miniconda3/envs/turbovla-bw/bin/python" "$BASE/miniconda3/envs/"*/bin/python; do
    [ -x "$c" ] || continue
    "$c" -c 'import torch;a=torch.randn(512,512,device="cuda");(a@a).sum()' >/dev/null 2>&1 && { echo "$c"; return; }
  done
}
PY="${PYTHON_BIN:-$(pick_py)}"
PY="${PY:-$BASE/miniconda3/envs/turbovla/bin/python}"
echo "[INFO] python 环境: $PY"

# 卡数与显存预检:见不到 4 张卡就不要静默降级
"$PY" - "${#GPUS[@]}" <<'PY' || exit 1
import sys, torch
need = int(sys.argv[1])
n = torch.cuda.device_count()
print(f"可见 GPU: {n}(需要 {need})")
assert n >= need, f"可见 {n} 张卡,少于需要的 {need} 张 —— 别静默降级,先查任务配额"
PY
# 显存也要够:实测最重的 A0'(DINOv3-B,fp32,batch56)训练峰值 22.84GB,单卡需 ≥32GB
"$PY" - <<'PY' || exit 1
import torch
for i in range(torch.cuda.device_count()):
    gb = torch.cuda.get_device_properties(i).total_memory / 2**30
    print(f"  GPU{i}: {torch.cuda.get_device_name(i)} {gb:.0f} GiB")
    assert gb >= 30, f"GPU{i} 只有 {gb:.0f} GiB —— 实测 batch56 训练峰值 22.84GB,余量不足"
PY
# 若卡上已有别的进程,提醒(共享节点上会互相抢显存)
busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | head -3)
[ -z "$busy" ] || echo "[WARN] 检测到卡上已有进程,确认不是别人的任务: $busy"

if [ "$PROBE" = "1" ]; then
  echo "=== 先用 GPU${GPUS[0]} 探 a0p 的 batch 上限 ==="
  GPU_IDS="${GPUS[0]}" bash "$LAUNCHER" probe:a0p || true
  echo "=== probe 结束;若最大可用 batch < $BATCH_SIZE,请四臂一起改 BATCH_SIZE 后重跑 ==="
fi

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="$PKG/output/ablation/parallel_$TS"
mkdir -p "$LOGDIR"
echo "[INFO] 日志目录: $LOGDIR"

PIDS=(); NAMES=()
for i in "${!ARMS[@]}"; do
  arm="${ARMS[$i]}"; gpu="${GPUS[$i]}"
  # ★ TMPDIR 必须**短**:Python multiprocessing 的 AF_UNIX socket 路径上限 107 字符,
  #   放共享卷长路径(如 $PKG/output/.../tmp_a0p = 79 字符)会在 resource_sharer.DupFd
  #   上崩 "OSError: AF_UNIX path too long"(2026-09-16 实测踩到,四臂全崩)。
  tmpdir="${TMPDIR_ROOT:-/tmp}/abl_$arm"
  (( ${#tmpdir} < 60 )) || { echo "!!! TMPDIR 太长(${#tmpdir} 字符): $tmpdir —— AF_UNIX 上限 107,会崩"; exit 1; }
  mkdir -p "$tmpdir"
  [ "$i" -eq 0 ] || sleep "$STAGGER"
  echo "[START] arm=$arm gpu=$gpu (日志 $LOGDIR/$arm.log)"
  GPU_IDS="$gpu" TMPDIR="$tmpdir" bash "$LAUNCHER" "$arm" > "$LOGDIR/$arm.log" 2>&1 &
  PIDS+=($!); NAMES+=("$arm")
done

echo "[INFO] 四臂已全部启动;前台等待(勿中断,容器随主进程结束被回收)"
FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then rc=0; else rc=$?; FAIL=1; fi
  printf '[DONE] %-4s rc=%s\n' "${NAMES[$i]}" "$rc"
done

echo "==================== 汇总 ===================="
declare -A PREFIX=(
  [a0p]=turbovla_ablation_A0p_dinov3_bert_ann
  [a1]=turbovla_ablation_A1_sdtv3_bert_ann
  [a2]=turbovla_ablation_A2_dinov3_sootspike_ann
  [a3]=turbovla_ablation_A3_sdtv3_sootspike_ann
)
for arm in "${ARMS[@]}"; do
  log="$LOGDIR/$arm.log"
  last=$(tr '\r' '\n' < "$log" 2>/dev/null | grep -oE "loss=[0-9.]+.*gacc=[0-9.]+" | tail -1)
  pfx="${PREFIX[$arm]:-}"
  nck=$(ls "$PKG/output/ablation/checkpoints/" 2>/dev/null | grep -c "^${pfx}_" || true)
  nsave=$(grep -c "^saved:" "$log" 2>/dev/null || true)
  printf '%-4s | ckpt %s 个(落盘 %s 次) | 最新: %s\n' "$arm" "$nck" "$nsave" "${last:0:100}"
done
echo "[INFO] 完整日志: $LOGDIR/*.log"
echo "[INFO] 评测:bash /data/260010028/dwh_vla/eval_ablation.sh <arm>(默认 ema 口径)"
if [ "$FAIL" = "0" ]; then
  echo "[RESULT] 四臂全部正常退出"
else
  echo "[RESULT] 有臂非零退出,见上方 rc —— 平台会如实显示失败"
fi
exit "$FAIL"
