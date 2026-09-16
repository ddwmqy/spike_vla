# 算力服务器运行手册(2026-09-13)

> 配套 `PLAN_CN.md` v7、`ASSETS.md`、`v4_code/experiments/robotwin/ARMS_CN.md`。
> 全部资产在共享卷 `/data` 上,算力服务器直接可见,无需重下。
> **原则**:每一步都有"预期输出";不符就先停下排查,不要带病开训。

## 阶段 0:建两个 env(约 30–60 分钟,仅首次)

训练/评测要**两个互相独立的 env**,不要合并(模拟器固定 torch 2.4.1,策略侧要 torch≥2.6):

```bash
# ---- A. 策略 env(turbovla-robotwin) ----
conda create -y -n turbovla-robotwin python=3.10 && conda activate turbovla-robotwin
cd /data/260010028/dwh_vla/v4_code
pip install -e ".[robotwin]"
pip install flash-attn                      # C1/B 的视觉走 flash_attention_2,必需
python -c "import torch, flash_attn; print('策略 env OK', torch.__version__, torch.cuda.device_count(), 'GPUs')"
# 预期:策略 env OK 2.x.x 4 GPUs

# ---- B. 模拟器 env(RoboTwin) ----
conda create -y -n robotwin python=3.10 && conda activate robotwin
cd /data/260010028/dwh_vla/RoboTwin
bash scripts/_install.sh                    # 装 requirements + pytorch3d + XPolicyLab + curobo
python -c "import sapien, sapien.render; print('sim env OK')"
```

**环境变量**(两个 env 都要,建议写进 `env.sh` 再 `source`):

```bash
export ROBOTWIN_DATA_ROOT=/data/260010028/dwh_vla/v4_assets/robotwin_data/RoboTwin
export DINOV3_MODEL_PATH=/data/260010028/dwh_vla/v4_assets/dinov3-vitl16-pretrain-lvd1689m
export BERT_MODEL_PATH=/data/260010028/dwh_vla/v2_code_bundle_20260906/resources/pretrained/bert-base-uncased
export TURBOVLA_INIT_CKPT=/data/260010028/dwh_vla/v4_assets/groundingdino/groundingdino_swint_ogc.pth
export SMOOTHSPIKE_MODEL_PATH=/data/260010028/dwh_vla/v2_code_bundle_20260906/resources/pretrained/SmoothSpike/smoothspike-bert-base-fused
export SDTV3_WEIGHTS_PATH=/data/260010028/dwh_vla/v2_code_bundle_20260906/resources/pretrained/V3_19.0M_1x4.pth
export SDTV3_PROCESSOR_PATH=/data/260010028/dwh_vla/v2_code_bundle_20260906/resources/pretrained/dinov3-vitb16-pretrain-lvd1689m
export STARVLA_PYTHON=$(which python)       # 策略侧 python(train.sh 默认用系统 python,必须显式指定)
```

## 阶段 1:三项冒烟(约 1 小时;**评测栈优先**)

### 1.1 渲染/Vulkan(最可能出意外,先做)

```bash
conda activate robotwin
vulkaninfo --summary | grep -E "deviceName|driverName"
# 预期:deviceName = NVIDIA ...(不是 llvmpipe!)
# 若是 llvmpipe 或报错 → 服务器的 NVIDIA Vulkan 用户态不全,评测跑不了,先解决驱动
python -c "
import sapien, sapien.render, sapien.physx
s = sapien.Scene(); c = s.add_camera('c', 224, 224, 1.0, 0.1, 10); s.step(); c.take_picture()
print('SAPIEN 渲染 OK', c.get_picture('Color').shape)"
# 预期:SAPIEN 渲染 OK (224, 224, 4)
```

> 参考:pod(dev 容器)**过不了这一步**——只有软件 Vulkan(lavapipe),而 SAPIEN 的渲染器
> 硬性要求光追扩展(`VK_KHR_acceleration_structure` 等)。这也是评测只能在服务器跑的原因。

### 1.2 策略侧真卡前向 + 训练循环计数

```bash
conda activate turbovla-robotwin && cd /data/260010028/dwh_vla/v4_code && source env.sh
python scripts/gate3_gpu_check.py
# 预期:device: NVIDIA ...;fp32/bf16 均 finite;GATE #3 PASS

# §5.4 计数短跑(单卡即可;本地已验,这里确认服务器 env 行为一致)
python scripts/robotwin/verify_54_counting.py \
  --config_yaml experiments/robotwin/configs/clean50.yaml \
  --max-steps 1100 --warmup 1000 --eval-interval 25 --save-interval 200 \
  --set trainer.use_deepspeed=false --set datasets.vla_data.per_device_batch_size=8
# 预期:结尾 "§5.4 VERIFY PASS";报告落 results/Checkpoints/clean50_54verify_*/54_verify_report.json
```

### 1.3 评测栈端到端(官方 ckpt,1 任务 × 5 局)

```bash
export ROBOTWIN_PATH=/data/260010028/dwh_vla/RoboTwin
export ROBOTWIN_PYTHON=/path/to/envs/robotwin/bin/python
ROBOTWIN_TEST_NUM=5 bash scripts/robotwin/evaluate.sh \
  /data/260010028/dwh_vla/v4_assets/TurboVLA_robotwin/checkpoints/robotwin/steps_55000_ema_model.safetensors \
  beat_block_hammer
# 预期:5 局里有成功/失败记录,解析器产出 summary;分母硬校验通过
```

## 阶段 2:A-S 冒烟(1 卡,约 0.5 天;可跳过——本地 MIG 已验流水线健康)

```bash
conda activate turbovla-robotwin && cd /data/260010028/dwh_vla/v4_code && source env.sh
CONFIG_YAML=experiments/robotwin/configs/clean50_a.yaml \
RUN_ID=asmoke_$(date +%Y%m%d) MAX_TRAIN_STEPS=1000 NUM_PROCESSES=1 \
PER_DEVICE_BATCH_SIZE=48 GRADIENT_ACCUMULATION_STEPS=4 \
nohup bash scripts/robotwin/train.sh > asmoke.log 2>&1 &
```
**启动后 5 分钟内必须核对**:
- `grep "loaded 0 initialization\|load_pretrained" asmoke.log` → A 臂是 `from-scratch`,**不应**出现初始化加载
- loss 下降、无 NaN;`results/Checkpoints/asmoke_*/checkpoints/` 出现 ckpt + EMA

**注意**:训练必须**前台**跑(`server_train.sh` 已如此)——后台 `nohup` 会在任务主进程退出时被平台**回收容器**一并杀掉(2026-09-13 实测)。

## 阶段 3:C1(4 卡,约 1 天)—— 官方配方锚点

```bash
bash /data/260010028/dwh_vla/v4_assets/server_train.sh c1
```

**启动后必须核对**(用真实 ckpt 实测的期望值):
```bash
grep "initialization tensors" c1.log   # 预期:loaded 381 initialization tensors(2026-09-13 实测)
```
- 每 1000 步看一次 `Step N, Loss:`(logging_frequency=50),应有下降趋势
- 参考规模:`434.036M` 总参 / `324.553M` 可训练;**实测 ~1.9 s/step**(4×H100,全局 192;含周期性数据加载停顿)→ **全程约 29 小时**
- **55k 不得中断**(§10-4);建议 `tmux`/`nohup` + `ssh -o ServerAliveInterval=60`
- 到 55k 时:EMA ckpt = `checkpoints/steps_55000_ema_pytorch_model.pt`(+ 同目录 `config.yaml`/`dataset_statistics.json`,评测栈要求)

## 阶段 4:B(4 卡,约 1 天)—— 文本 spike 化

```bash
bash /data/260010028/dwh_vla/v4_assets/server_train.sh b
```
**启动后必须核对**(§5.2 防覆盖):
```bash
grep "initialization tensors" b.log    # 预期:loaded 182 initialization tensors(不含 200 个普通 BERT!)
```
- 若显示 331 → **立刻停**:说明 `load_bert` 没生效,sootspike 权重会被覆盖
- 事后可复核:`python scripts/check_b_init_hash.py`(需同一份 init ckpt)

## 阶段 5:评测(两臂统一协议)

```bash
# C0-20 筛查(官方 ckpt × 50 任务 × 20 局 = 1000 局)
ROBOTWIN_TEST_NUM=20 bash scripts/robotwin/evaluate.sh <官方 ckpt>
# C0-100(协议校准,5000 局)
ROBOTWIN_TEST_NUM=100 bash scripts/robotwin/evaluate.sh <官方 ckpt>
# C1 / B 各 5000 局(EMA-55k 口径,判定用)
ROBOTWIN_TEST_NUM=100 bash scripts/robotwin/evaluate.sh results/Checkpoints/<c1_run>/checkpoints/steps_55000_ema_pytorch_model.pt
ROBOTWIN_TEST_NUM=100 bash scripts/robotwin/evaluate.sh results/Checkpoints/<b_run>/checkpoints/steps_55000_ema_pytorch_model.pt
```
判定按 `PLAN_CN.md` §5.8 的预注册规则(B−C1 ≥ −2pp 成立 / ≤ −4pp 不成立 / 中间不定)。

## 常见坑速查

| 症状 | 原因 | 处置 |
|---|---|---|
| `train.sh` 报 `accelerate.commands.launch` 找不到 | 默认用系统 python | 设 `STARVLA_PYTHON=$(which python)` |
| 数据找不到 | `ROBOTWIN_DATA_ROOT` 没指向含 `Clean/` 的那层 | 指到 `.../robotwin_data/RoboTwin` |
| 覆盖参数不生效 | 少了 `--` 前缀(被静默丢弃) | 用 `--key value` 或 `--key=value` |
| `libGL.so.1` 缺失 | 容器没装 | `apt-get install -y libgl1`(容器重建后要重装) |
| `UnpicklingError` | torch≥2.6 的 `weights_only` 默认值 | 已修(走 `load_checkpoint_file`);别回退该改动 |
| `run_id` 目录已存在 | train.sh 拒绝覆盖 | 换 RUN_ID 或删目录 |
| 评测报 `failed to find a rendering device` | 无可用 Vulkan/光追 | 见阶段 1.1;pod 不可行,服务器需驱动完整 |
| SAPIEN 导入即 `FileNotFoundError: '/etc/glvnd/egl_vendor.d'` | `sapien/_vulkan_tricks.py:56` 会逐个 listdir 两个 glvnd 目录,容器里 `/etc/glvnd/egl_vendor.d` 不存在 | 已修(2026-09-15):`server_vulkan_test.sh` **无条件**导出 `__EGL_VENDOR_LIBRARY_FILENAMES`(该变量在函数开头就提前返回),并补齐两个标准目录 + `libvulkan1` |
