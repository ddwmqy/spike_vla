#!/usr/bin/env bash
# RTX 5090 机器探针 + 自动建环境(一条命令跑完,不需要任何参数)。
#
# 运行:
#   bash /data/260010028/dwh_vla/probe_5090.sh
#
# 它会:
#   1) 扫描本机候选 python,逐个打分(有没有 torch/spikingjelly/cupy/transformers/timm)
#   2) 用最优的试一次真算子(matmul+conv+backward)——**这一步决定它能不能跑 5090**
#   3) 若跑不了(典型:torch 2.6+cu124 无 Blackwell 内核)→ **自动在 /data 上建一个
#      python3.10 的新环境**并装好全套依赖,再重测
#   4) 测 spikingjelly+cupy 的 LIF 前向、真实模型训练显存(batch 56)
#   5) 最后跑 30 步真训练冒烟(跳过:SKIP_SMOKE=1)
#
# 日志(共享卷,dev pod 可直接读):/data/260010028/dwh_vla/v4_code/probe_5090.log
set -uo pipefail

BASE=/data/260010028/dwh_vla
PKG=$BASE/v2_code_bundle_20260906/package
RES=$BASE/v2_code_bundle_20260906/resources
LOG="${PROBE_LOG:-$BASE/v4_code/probe_5090.log}"   # 可覆盖:PROBE_LOG=/data/.../other.log
BW_ENV="${BW_ENV:-$BASE/miniconda3/envs/turbovla-bw}"
BASE_PY="${BASE_PY:-$BASE/miniconda3/envs/turbovla/bin/python}"   # py3.10(必须,TF 2.15 不支持 3.14)
mkdir -p "$(dirname "$LOG")"

# NGC 镜像会设 PIP_CONSTRAINT 把 torch 锁死(导致 PyPI 解析必然失败),必须清掉。
# 这是 2026-09-14 给服务器装模拟器环境时踩过的同一个坑。
unset PIP_CONSTRAINT
export PIP_CONFIG_FILE=/dev/null
export PIP_NO_INPUT=1
# 这个集群会抢占任务 → pip 缓存放共享卷,重跑时已下好的 wheel 不用再下(省 3GB+)
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$BASE/.pip-cache}"
mkdir -p "$PIP_CACHE_DIR"

PASS=(); FAIL=(); WARN=()
ok()   { PASS+=("$1"); echo "  ✅ $1"; }
bad()  { FAIL+=("$1"); echo "  ❌ $1"; }
warn() { WARN+=("$1"); echo "  ⚠️  $1"; }

gpu_ok() {  # 本机 GPU 上真算一次;能出数=内核可用
  "$1" -c '
import torch
a = torch.randn(2048, 2048, device="cuda", requires_grad=True)
(a @ a).sum().backward()
torch.nn.Conv2d(8,16,3,padding=1).cuda()(torch.randn(4,8,32,32,device="cuda"))
' >/dev/null 2>&1
}

scan_envs() {  # 打分选最优,回显 BESTS
  local cands=() best="" best_s=0
  [ -n "${PYTHON_BIN:-}" ] && cands+=("$PYTHON_BIN")
  cands+=("$(command -v python3 2>/dev/null)")
  for p in "$BASE/miniconda3/envs"/*/bin/python "$BW_ENV/bin/python" \
           /home/dwh/venvs/*/bin/python "$BASE/v4_assets/envs"/*/bin/python; do
    [ -x "$p" ] && cands+=("$p")
  done
  for c in "${cands[@]}"; do
    [ -x "$c" ] || continue
    info=$("$c" - <<'PE' 2>/dev/null
import importlib.util as u
mods = {m: u.find_spec(m) is not None for m in ("torch","spikingjelly","cupy","transformers","timm")}
try:
    import torch
    s = "torch=%s,cuda=%s,avail=%s" % (torch.__version__, torch.version.cuda, torch.cuda.is_available())
except Exception as e:
    s = "torch不可用"
print("%s | " % s + " ".join(f"{k}={'Y' if v else 'N'}" for k, v in mods.items()))
PE
)
    echo "--- $c"; echo "    $info"
    local score=0
    echo "$info" | grep -q "avail=True"     && score=$((score+1))
    echo "$info" | grep -q "spikingjelly=Y" && score=$((score+2))
    echo "$info" | grep -q "cupy=Y"         && score=$((score+1))
    echo "$info" | grep -q "transformers=Y" && score=$((score+1))
    if [ "$score" -gt "$best_s" ]; then best_s=$score; best="$c"; fi
  done
  echo "  → 初选: ${best:-无}(得分 $best_s/5)"
  BESTS="$best"
}

build_env() {
  echo; echo "=== [B] 自动新建 Blackwell 环境: $BW_ENV ==="
  if [ -x "$BW_ENV/bin/python" ]; then
    echo "  已存在,仅补装依赖"
  else
    [ -x "$BASE_PY" ] || { bad "找不到 python3.10 基底: $BASE_PY(新环境需要它)"; return 1; }
    "$BASE_PY" -m venv "$BW_ENV" || { bad "venv 创建失败"; return 1; }
    "$BW_ENV/bin/pip" install -q -U pip 2>&1 | tail -2
  fi
  local py="$BW_ENV/bin/python"
  echo "  基底版本: $($py -V 2>&1)"
  # torch:优先官方 cu128 索引(Blackwell 必需);装完**必须再验一次真算子**,
  # 不通过才继续换源(否则会停在"pip 成功但内核仍不可用"的状态)
  local TORCH_OK=0
  # 源顺序:清华/阿里优先(download.pytorch.org 在该集群被墙 403,放最后只用一次尝试)
  for IDX in "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/" \
             "https://mirrors.aliyun.com/pypi/simple/" \
             "https://download.pytorch.org/whl/cu128"; do
    echo "  --- 尝试 torch 源: $IDX"
    if ! "$BW_ENV/bin/pip" install torch torchvision --index-url "$IDX" > /tmp/pip_torch.log 2>&1; then
      echo "      pip 失败,错误尾部:"; tail -6 /tmp/pip_torch.log | sed 's/^/        /'
    fi
    v=$("$py" -c 'import torch;print(torch.__version__,"cuda",torch.version.cuda)' 2>&1 | tail -1)
    echo "      装到: $v"
    if gpu_ok "$py"; then TORCH_OK=1; echo "      ✅ 该源装出的 torch 能跑本机 GPU"; break; fi
    echo "      ✗ 该源装出的 torch 仍跑不了本机 GPU,换下一个源"
  done
  # 保险:上面按"最新兼容版"装,万一解析到不支持 Blackwell 的旧版,再显式指定几组已知支持 sm_120 的版本
  if [ "$TORCH_OK" != "1" ]; then
    echo "  --- 兜底:显式指定已知支持 sm_120 的 torch 版本"
    for SPEC in "torch==2.9.0 torchvision==0.24.0" "torch==2.8.0 torchvision==0.23.0"; do
      echo "      试 $SPEC"
      if ! "$BW_ENV/bin/pip" install $SPEC -i "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/" \
           > /tmp/pip_torch2.log 2>&1; then
        tail -4 /tmp/pip_torch2.log | sed 's/^/        /'
      fi
      v=$("$py" -c 'import torch;print(torch.__version__,"cuda",torch.version.cuda)' 2>&1 | tail -1)
      echo "      装到: $v"
      if gpu_ok "$py"; then TORCH_OK=1; echo "      ✅ 可用"; break; fi
    done
  fi
  [ "$TORCH_OK" = "1" ] || bad "所有源/版本都装不出能在本机跑的 torch(需要人工判断)"

  # 其余依赖(版本对齐 A4 那次训练的环境;镜像逐个回退)
  local MIRRORS=("https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/"
                 "https://mirrors.aliyun.com/pypi/simple/"
                 "https://pypi.org/simple/")
  local PKGS=( "numpy<2" "tensorflow-cpu==2.15.1" "tensorflow-datasets==4.9.4" "protobuf==3.20.3"
    "transformers==4.57.6" "timm==1.0.28" "safetensors" "huggingface_hub" "tokenizers" "regex"
    "einops" "omegaconf" "pyyaml" "tqdm" "scipy" "pillow" "opencv-python-headless"
    "accelerate" "wandb" "cupy-cuda12x" "spikingjelly==0.0.0.0.14" )
  for M in "${MIRRORS[@]}"; do
    echo "  --- 依赖源: $M"
    if ! "$BW_ENV/bin/pip" install -i "$M" "${PKGS[@]}" > /tmp/pip_deps.log 2>&1; then
      echo "      pip 失败,错误尾部:"; tail -8 /tmp/pip_deps.log | sed 's/^/        /'
    fi
    if "$py" -c 'import spikingjelly, cupy, tensorflow_datasets, transformers, timm' 2>/dev/null; then
      echo "      ✅ 依赖齐了"; break
    fi
    echo "      ✗ 该源不完整,换下一个"
  done
  echo "  装完。torch 版本: $(  $py -c 'import torch;print(torch.__version__, "cuda", torch.version.cuda)' 2>&1 | tail -1)"
}

{
echo "############ RTX 5090 探针 $(date) ############"

echo; echo "=== [0] 机器与共享卷 ==="
uname -srm; echo "nproc: $(nproc)"; free -g 2>/dev/null | head -2
df -h /data 2>/dev/null | tail -1
df -h /data >/dev/null 2>&1 && ok "/data 已挂载" || bad "/data 未挂载(资产全在共享卷上)"
miss=0
for p in "$RES/data/libero" "$RES/pretrained/V3_19.0M_1x4.pth" \
         "$RES/pretrained/SmoothSpike/smoothspike-bert-base-fused" \
         "$RES/pretrained/bert-base-uncased" "$RES/pretrained/dinov3-vitb16-pretrain-lvd1689m" \
         "$RES/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth" \
         "$BASE/LIBERO_eval" "$PKG/run_train_v2_ablation.sh" "$PKG/code/turbovla" "$BASE_PY"; do
  [ -e "$p" ] && echo "  ✓ $p" || { echo "  ✗ 缺 $p"; miss=1; }
done
[ "$miss" = "0" ] && ok "全部资产可见" || bad "有资产缺失(见上)"

echo; echo "=== [1] GPU 与驱动 ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>&1 | head -6
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l); echo "  可见 GPU 数: $ngpu"
[ "$ngpu" -ge 4 ] && ok "≥4 张卡(够四臂并行)" || warn "只看到 $ngpu 张卡(四臂并行需要 4 张)"

echo; echo "=== [2] 候选 python 环境 ==="
BESTS=""; scan_envs

echo; echo "=== [3] GPU 内核可用性(决定性;5090 需 sm_120)==="
if [ -n "$BESTS" ] && gpu_ok "$BESTS"; then
  cap=$("$BESTS" -c 'import torch;c=torch.cuda.get_device_capability(0);print(f"{c[0]}{c[1]}")' 2>/dev/null)
  echo "  ✅ $BESTS 在本机(sm_$cap)能跑真算子"
  ok "GPU 内核可用: $BESTS (sm_$cap)"
  PY="$BESTS"
else
  echo "  ❌ 初选环境跑不了本机 GPU(5090 上常见原因:torch 2.6+cu124 无 sm_120 内核)"
  warn "初选环境不可用,触发自动建环境"
  build_env
  PY="$BW_ENV/bin/python"
  if gpu_ok "$PY"; then
    ok "新建环境 GPU 内核可用: $PY"
  else
    bad "新建环境仍跑不了本机 GPU —— 请把本日志给我,需要进一步排查"
  fi
fi

echo; echo "=== [4] SNN 栈(spikingjelly + cupy)==="
out=$(PYTHONPATH="$PKG/code:$RES" "$PY" -c '
import torch
import turbovla.models.components.spiking as S      # 触发 spikingjelly 需要的 np.int shim
from spikingjelly.activation_based import neuron
n = neuron.LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode="m").cuda()
print("  ✅ LIF(cupy)前向 OK:", tuple(n(torch.rand(4,8,device="cuda")).shape))' 2>&1 | tail -5)
echo "$out"
echo "$out" | grep -q "✅" && ok "spikingjelly(cupy 后端)前向" || bad "spikingjelly(cupy 后端)前向(cupy 在 sm_120 上可能编不出 kernel)"

echo; echo "=== [5] 训练显存(真实模型,batch 56,最重的 A0')==="
out=$(cd "$BASE/v3_experiment/scripts" && V3CODE=$PKG/code PYTHONPATH="$PKG/code:$RES" \
      CUDA_VISIBLE_DEVICES=0 "$PY" train_mem_probe.py --arm A0p --batch 56 --steps 3 2>&1 | tail -20)
echo "$out"
if echo "$out" | grep -q '"fits_32GB"'; then
  peak=$(echo "$out" | grep -o '"peak_reserved_GB": [0-9.]*' | grep -o '[0-9.]*' | head -1)
  echo "$out" | grep -q '"fits_32GB": true' \
    && ok "训练显存 batch56 峰值 ${peak}GB → 32GB 卡可跑" \
    || bad "训练显存 batch56 峰值 ${peak}GB → 32GB 装不下,四臂需一起降 batch"
else
  bad "模型构建/前向失败(见上;通常是缺包或 GPU 内核问题)"
fi

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo; echo "=== [6] 30 步真训练冒烟(A1)==="
  GPU_IDS=0 MAX_STEPS=30 WARMUP_STEPS=8 BATCH_SIZE=56 TARGET_GLOBAL_BATCH=56 \
  RESUME_MODE=none CHECKPOINT_PREFIX=probe5090_A1 \
  PYTHON_BIN="$PY" TORCHRUN_BIN="$(dirname "$PY")/torchrun" \
  timeout 2400 bash "$BASE/launch_ablation.sh" a1 > /tmp/probe5090_smoke.log 2>&1
  rc=$?; tail -5 /tmp/probe5090_smoke.log
  if [ "$rc" = "0" ] && grep -q "saved:" /tmp/probe5090_smoke.log; then
    ok "30 步训练冒烟通过(ckpt 已落盘)"
  else
    bad "训练冒烟失败 rc=$rc(见 /tmp/probe5090_smoke.log)"
  fi
fi

echo; echo "=== [SUMMARY] ==="
echo "PASS (${#PASS[@]}):"; printf '  - %s\n' "${PASS[@]}"
[ ${#WARN[@]} -gt 0 ] && { echo "WARN (${#WARN[@]}):"; printf '  - %s\n' "${WARN[@]}"; }
[ ${#FAIL[@]} -gt 0 ] && { echo "FAIL (${#FAIL[@]}):"; printf '  - %s\n' "${FAIL[@]}"; }
if [ ${#FAIL[@]} -eq 0 ]; then
  echo "==> 结论: 可以直接开四臂(用环境 $PY):"
  echo "    PYTHON_BIN=$PY bash $BASE/launch_ablation_4gpu.sh"
else
  echo "==> 结论: 先解决 FAIL 项。日志已在共享卷,交给 Claude 看。"
fi
echo "############ DONE $(date) ############"
} 2>&1 | tee "$LOG"

echo; echo "日志: $LOG"

# 退出码(方便只看任务状态时判断):
#   0 = 全绿,可直接开四臂
#   2 = 有 FAIL(见日志的 [SUMMARY])
#   3 = 自动建环境后 GPU 仍不可用(需要人工介入)
if [ ${#FAIL[@]} -eq 0 ]; then exit 0; fi
for f in "${FAIL[@]}"; do case "$f" in *"GPU 内核"*) exit 3;; esac; done
exit 2
