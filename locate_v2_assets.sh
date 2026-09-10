#!/usr/bin/env bash
# v2/v3 资产只读定位 —— 在算力服务器上运行;不复制、不修改、不删除任何文件。
# 目的:D2/A1 前置的"v2 实际资产清单"(checkpoint、启动日志、训练配置、已有评测结果),
#       以及官方锚点逐任务 json(对比 MD 的对齐剔除口径需要)。
# 用法:
#   bash /data/260010028/dwh_vla/locate_v2_assets.sh              # 默认搜索根
#   ROOTS="/home/dwh /data" bash /data/260010028/dwh_vla/locate_v2_assets.sh
#   HASH=1 bash ...        # 额外对找到的 .pth 计算 sha256(每个 ~1GB,约几秒/个)
# 产物:manifest 写到 /data(两台机器都可见),dev pod 侧可直接读取核对。
set -uo pipefail
MANIFEST="${MANIFEST:-/data/260010028/dwh_vla/v2_assets_manifest.txt}"
ROOTS=${ROOTS:-"/home /root /data /mnt /workspace /tmp"}

section() { echo; echo "=== $1 ==="; }

{
echo "=== v2/v3 asset locate manifest (READ-ONLY) ==="
echo "host: $(hostname)  date: $(date -Is)"
echo "search roots: $ROOTS"

section "[1] checkpoints: turbovla_*.pth"
for r in $ROOTS; do
  [ -d "$r" ] || continue
  find "$r" -xdev -maxdepth 8 -name 'turbovla_*.pth' -exec ls -la --time-style=long-iso {} \; 2>/dev/null
done | sort -k9 | uniq
if [ "${HASH:-0}" = "1" ]; then
  echo "--- sha256 (HASH=1) ---"
  for r in $ROOTS; do
    [ -d "$r" ] || continue
    find "$r" -xdev -maxdepth 8 -name 'turbovla_*train_v2*' -name '*.pth' -exec sha256sum {} \; 2>/dev/null
  done
fi

section "[2] train_v2 / train_v3 directories"
for r in $ROOTS; do
  [ -d "$r" ] || continue
  find "$r" -xdev -maxdepth 8 -type d \( -name 'train_v2' -o -name 'train_v3' \) -exec ls -lad {} \; 2>/dev/null
done | sort -k9 | uniq

section "[3] existing eval outputs: summary.json / suite json under *eval*"
for r in $ROOTS; do
  [ -d "$r" ] || continue
  find "$r" -xdev -maxdepth 10 -type f -name 'summary.json' -path '*eval*' -exec ls -la --time-style=long-iso {} \; 2>/dev/null
done | sort -k9 | uniq | head -50

section "[4] official anchor eval output dir (official_clean_8f1084e) — per-task json needed"
for r in $ROOTS; do
  [ -d "$r" ] || continue
  find "$r" -xdev -maxdepth 9 -type d -name 'official_clean_8f1084e*' -exec ls -lad {} \; 2>/dev/null
  find "$r" -xdev -maxdepth 10 -type f -name 'libero_*.json' -path '*official*' -exec ls -la --time-style=long-iso {} \; 2>/dev/null
done | sort -k9 | uniq | head -50

section "[5] v2 launch/training logs (launch_v2*, nohup, train_v2/logs)"
for r in $ROOTS; do
  [ -d "$r" ] || continue
  find "$r" -xdev -maxdepth 8 -type f \( -name 'launch_v2*.log' -o -name 'launch_v2*.sh' -o -path '*train_v2/logs*' -o -name 'nohup*.out' \) -exec ls -la --time-style=long-iso {} \; 2>/dev/null
done | sort -k9 | uniq | head -50

section "[6.5] original v2-era workspace: libero_sdtv3_bert_snnfusion_20260831 (dir may sit on a mount find -xdev skips)"
ORIG_PARENT="${ORIG_PARENT:-/home/dwh/work/vla/libero_vla}"
ORIG_NAME="libero_sdtv3_bert_snnfusion_20260831"
ORIG=""
if [ -d "$ORIG_PARENT/$ORIG_NAME" ]; then
  ORIG="$ORIG_PARENT/$ORIG_NAME"
else
  for r in $ROOTS /home/dwh; do
    [ -d "$r" ] || continue
    hit=$(find "$r" -xdev -maxdepth 5 -type d -name "$ORIG_NAME" -print -quit 2>/dev/null)
    [ -n "$hit" ] && { ORIG="$hit"; break; }
  done
fi
if [ -n "$ORIG" ]; then
  echo "ORIG PKG = $ORIG"
  ls -la --time-style=long-iso "$ORIG" 2>/dev/null | head -25
  echo "--- expected v2 ckpt: output/train_v2/checkpoints/ ---"
  ls -la --time-style=long-iso "$ORIG/output/train_v2/checkpoints/" 2>/dev/null || echo "[miss] $ORIG/output/train_v2/checkpoints/"
  echo "--- expected anchor per-task json: output/eval/official_clean_8f1084e/results/ ---"
  ls -la --time-style=long-iso "$ORIG/output/eval/official_clean_8f1084e/results/" 2>/dev/null || echo "[miss] $ORIG/output/eval/official_clean_8f1084e/results/"
  echo "--- any eval output dirs under ORIG ---"
  find "$ORIG" -xdev -maxdepth 4 -type d \( -name '*clean_8f1084e*' -o -name 'eval' \) 2>/dev/null
  echo "--- v2-era launch/train logs under ORIG ---"
  find "$ORIG" -xdev -maxdepth 5 -type f \( -name '*.log' -path '*train*' -o -name 'nohup*.out' \) -exec ls -la --time-style=long-iso {} \; 2>/dev/null | head -20
  echo "--- historical snnve 80k (Aug-30 generation, 历史参考) ---"
  ls -la --time-style=long-iso "$ORIG/results/libero_sdtv3_spikinglm_80k_ema_clean_8f1084e/checkpoints/" 2>/dev/null || echo "[miss] $ORIG/results/libero_sdtv3_spikinglm_80k_ema_clean_8f1084e/checkpoints/"
else
  echo "[miss] $ORIG_NAME not under $ORIG_PARENT nor within maxdepth 5 of roots: $ROOTS /home/dwh"
fi

section "[7] training configs near checkpoints (config.yaml / dataset_statistics.json / *.yaml)"
for r in $ROOTS; do
  [ -d "$r" ] || continue
  find "$r" -xdev -maxdepth 9 -type f \( -name 'config.yaml' -o -name 'dataset_statistics.json' \) 2>/dev/null \
    | grep -Ei 'v2|ckpt|checkpoint|output' | head -30
done

echo
echo "=== end of manifest ==="
} | tee "$MANIFEST"
echo
echo "[INFO] manifest written to: $MANIFEST (on /data, readable from the dev pod)"
