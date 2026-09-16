#!/usr/bin/env bash
# 消融四臂**并行**评测:每卡一个臂、各自把 4 个 suite 串行跑完(≈6 h 全部出结果)。
# 与"单臂占 4 卡并行跑 4 suite"(≈1.5 h/臂、四臂要串行 6 h)墙钟相同,但四臂同时产出。
#
# 赛后必做(**漂移核对**):用官方锚点重跑 libero_spatial,与归档 468/500 = 93.6% 对齐;
# 对得上才允许复用 A4 的既有数字。命令见脚本末尾提示。
#
# 用法(在算力服务器上跑,前台按住):
#   bash /data/260010028/dwh_vla/eval_ablation_4gpu.sh
#   ARMS="a1,a3" GPUS="0,1" bash ...                      # 只用两卡
#   WEIGHT_SOURCE=raw bash ...                            # 改口径(默认 ema 主口径)
set -Eeuo pipefail

BASE=/data/260010028/dwh_vla
PKG=$BASE/v2_code_bundle_20260906/package
EVAL=$BASE/eval_ablation.sh

ARMS_STR="${ARMS:-a0p,a1,a2,a3}"
GPUS_STR="${GPUS:-0,1,2,3}"
IFS=',' read -r -a ARMS <<< "$ARMS_STR"
IFS=',' read -r -a GPUS <<< "$GPUS_STR"
(( ${#ARMS[@]} == ${#GPUS[@]} )) || { echo "!!! ARMS 与 GPUS 数量不一致"; exit 1; }
test -x "$EVAL" || { echo "!!! 缺 $EVAL"; exit 1; }

gen="$BASE/miniconda3/envs/turbovla/bin/python"
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="$PKG/output/eval/parallel_$TS"
mkdir -p "$LOGDIR"

echo "==================== 四臂并行评测 ===================="
echo "臂 × GPU : $(for i in "${!ARMS[@]}"; do printf '%s→GPU%s ' "${ARMS[$i]}" "${GPUS[$i]}"; done)"
echo "口径     : ${WEIGHT_SOURCE:-ema(默认)} | suites: ${EVAL_SUITES:-全部4个}"
echo "日志     : $LOGDIR"
echo "====================================================="

# 先确认每个臂的 ckpt 都在,避免跑一半才发现缺
missing=0
for arm in "${ARMS[@]}"; do
  if ! DRY_RUN=1 bash "$EVAL" "$arm" 2>/dev/null | grep -q "ckpt exists *yes"; then
    echo "!!! 缺 ckpt: $arm(先确认训练已完成)"; missing=1
  fi
done
(( missing == 0 )) || exit 1

PIDS=(); NAMES=()
for i in "${!ARMS[@]}"; do
  arm="${ARMS[$i]}"; gpu="${GPUS[$i]}"
  echo "[START] arm=$arm gpu=$gpu"
  EVAL_GPU_IDS="$gpu" bash "$EVAL" "$arm" > "$LOGDIR/$arm.log" 2>&1 &
  PIDS+=($!); NAMES+=("$arm")
  sleep 10   # 轻微错开,避免同时初始化 LIBERO/EGL
done

FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then rc=0; else rc=$?; FAIL=1; fi
  printf '[DONE] %-4s rc=%s\n' "${NAMES[$i]}" "$rc"
  eval "rc_${NAMES[$i]}=$rc"
done

echo "==================== 各臂结果(从明细重算) ===================="
"$gen" - "${ARMS[@]}" <<'PY'
import json, os, sys
from pathlib import Path
PKG = Path("/data/260010028/dwh_vla/v2_code_bundle_20260906/package")
ws = os.environ.get("WEIGHT_SOURCE", "ema")   # 与 eval_ablation.sh 的口径后缀保持一致
ROOT = {a: PKG/f"output/eval/ablation_{a.upper().replace('A0P','A0p')}_step_80000_{ws}"
        for a in ("a0p", "a1", "a2", "a3")}
for arm in sys.argv[1:]:
    d = ROOT.get(arm, PKG/"output/eval")/"results"
    tot_s = tot_e = 0; n = 0
    for f in sorted(d.glob("libero_*.json")) if d.is_dir() else []:
        j = json.loads(f.read_text()); tot_s += j["total_successes"]; tot_e += j["total_episodes"]; n += 1
    print(f"  {arm:<4} {tot_s}/{tot_e} = {100*tot_s/tot_e:.1f}%" if tot_e else f"  {arm:<4} (无结果)")
PY

echo "[INFO] 出主表:python /data/260010028/dwh_vla/v3_experiment/scripts/report_main_table.py"
echo "[INFO] 漂移核对(官方锚点,4 卡并行,~1.5 h):"
echo "       EVAL_SUITES=libero_spatial bash /data/260010028/dwh_vla/eval_ablation.sh a0"
echo "       基准:归档 468/500 = 93.6%;±1pp 内则复用 A4 既有数字成立"
[ "$FAIL" = "0" ] && echo "[RESULT] 四臂评测全部正常退出" || echo "[RESULT] 有臂非零退出,见 $LOGDIR"
exit 0
