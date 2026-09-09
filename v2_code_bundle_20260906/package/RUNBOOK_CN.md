# LIBERO 新实验迁移与运行手册

## 1. 要跑什么

本实验训练并评测以下学生模型：

| 模块 | 配置 |
|---|---|
| 视觉编码器 | SDT-V3 19M，SNN，可训练 |
| 语言编码器 | BERT-base-uncased，ANN，完全冻结 |
| 特征融合 | 6 层双向 SNN fusion，`T=4` |
| SNN attention | binary Q/K + exact Spike2Max normalization，优先准确率 |
| 时间读出 | 对连续膜电位/残差流学习 4 个 softmax 权重 |
| 动作解码器 | ANN ACT，12 步 action chunk |

时间读出读取的是连续 fusion membrane/residual，不是最终二值脉冲。初始 4 个权重完全均匀，因此第 0 步严格等价于原来的 `mean(T)`，训练后再自行选择时间步。

训练使用强 SDT-V3+BERT 80k checkpoint 的 EMA：

- warm-start 视觉编码器、ANN action head 和形状兼容的投影层；预期映射 `603 + 70 + 11 = 684` 个张量；
- 同一 EMA 作为 teacher，feature/action 蒸馏权重均为 `0.1`；
- SNN fusion 本身不从 ANN fusion 错配加载，而是重新初始化。

正式训练为 FP32、全局 batch 256、80k optimizer steps、每 5k 保存一次 checkpoint，同时保存 online 和 EMA 权重。

## 2. 包里有什么

```text
code/turbovla/             隔离的学生模型和训练代码
third_party/vla_adapter/   固定版本 LIBERO rollout adapter
run_train.sh               正式训练入口
queue_train.sh             等待空闲 GPU 后自动训练
evaluate.py                强制使用包内代码的评测入口
run_eval.sh                clean LIBERO 四套件正式评测
tests/                     SNN 控制和 learned readout 测试
```

包内不包含 LIBERO RLDS 数据、BERT/DINO/SDT 权重、teacher checkpoint 和 Python 环境。

## 3. 新服务器必须准备的内容

假设新服务器的 TurboVLA 资源根目录为 `$REPO_DIR`，以下路径必须存在：

```text
$REPO_DIR/data/libero/libero_10_no_noops/1.0.0/dataset_info.json
$REPO_DIR/data/libero/libero_goal_no_noops/1.0.0/dataset_info.json
$REPO_DIR/data/libero/libero_object_no_noops/1.0.0/dataset_info.json
$REPO_DIR/data/libero/libero_spatial_no_noops/1.0.0/dataset_info.json
$REPO_DIR/pretrained/dinov3-vitb16-pretrain-lvd1689m/
$REPO_DIR/pretrained/bert-base-uncased/
$REPO_DIR/pretrained/V3_19.0M_1x4.pth
$REPO_DIR/third_party/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Base/models.py
$REPO_DIR/experiments/libero/configs/libero_all4_stats.json
$REPO_DIR/experiments/libero/configs/online_text_layout.json
$REPO_DIR/outputs/checkpoints/libero_sdtv3_19m_2gpu/turbovla_sdtv3_19m_80000.pth
```

最后一个文件是 SDT-V3+BERT 强基线 teacher，必须包含：

```text
model_state_dict
ema_model_state_dict
model_config
global_step = 80000
```

Python 环境应与原 `turbovla-libero` 环境一致，至少需要 PyTorch 2.3/CUDA、Transformers、SpikingJelly、CuPy、TensorFlow/RLDS 依赖和训练数据读取依赖。

## 4. 解压与环境变量

```bash
tar -xzf libero_sdtv3_bert_snnfusion_20260831.tar.gz
cd libero_sdtv3_bert_snnfusion_20260831

export REPO_DIR=/path/to/TurboVLA
export PYTHON_BIN=/path/to/conda/envs/turbovla-libero/bin/python
export TORCHRUN_BIN=/path/to/conda/envs/turbovla-libero/bin/torchrun
```

先检查 GPU 和资源：

```bash
nvidia-smi
test -f "$REPO_DIR/pretrained/V3_19.0M_1x4.pth"
test -d "$REPO_DIR/pretrained/bert-base-uncased"
test -f "$REPO_DIR/outputs/checkpoints/libero_sdtv3_19m_2gpu/turbovla_sdtv3_19m_80000.pth"
```

## 5. 代码测试与命令预检

```bash
export PYTHONPATH="$PWD/code:$REPO_DIR"
"$PYTHON_BIN" -m unittest discover -s tests -v

GPU_IDS=0,1,2,3 DRY_RUN=1 \
  REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" TORCHRUN_BIN="$TORCHRUN_BIN" \
  ./run_train.sh
```

预期单元测试为 `7 tests, OK`。dry-run 应显示：

```text
SDT-V3 + frozen BERT + SNN fusion
--text_encoder_type bert
--freeze_text_encoder
--downstream_type spiking
--temporal_readout learned
--action_head_type ann
--teacher_weight_source ema
--student_init_weight_source ema
```

## 6. 建议先跑 2-step GPU smoke test

使用一张完全空闲的 24 GB GPU：

```bash
GPU_IDS=0 \
BATCH_SIZE=1 \
TARGET_GLOBAL_BATCH=2 \
MAX_STEPS=2 \
WARMUP_STEPS=1 \
SAVE_STEPS=1 \
OUTPUT_ROOT="$PWD/output/smoke" \
REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" TORCHRUN_BIN="$TORCHRUN_BIN" \
./run_train.sh
```

必须确认日志中出现：

```text
text_encoder_type=bert
freeze_text_encoder=True
downstream_type=spiking, T=4
action_head_type=ann
temporal_readout=learned
student ... report={'mapped': 684, ...}
teacher ... SDT-V3
```

并确认 `output/smoke/checkpoints/` 生成 1-step 和 2-step checkpoint。smoke 目录不能作为正式训练的 resume 来源。

## 7. 正式训练

四张空闲 4090：

```bash
GPU_IDS=0,1,2,3 \
REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" TORCHRUN_BIN="$TORCHRUN_BIN" \
./run_train.sh
```

默认输出：

```text
output/train/logs/train_4gpu_<timestamp>.log
output/train/checkpoints/turbovla_sdtv3_bert_snnfusion_learned_5000.pth
...
output/train/checkpoints/turbovla_sdtv3_bert_snnfusion_learned_80000.pth
```

四卡配置为每卡 batch 16、梯度累积 4、有效全局 batch 256。不要在有其他用户模型占显存的 GPU 上启动。参考既有同规模蒸馏训练，80k 约需 35-55 小时，实际以新服务器吞吐为准。

长期托管可用：

```bash
systemd-run --user \
  --unit=libero-sdt-bert-snnfusion-train \
  --collect \
  --setenv=GPU_IDS=0,1,2,3 \
  --setenv=REPO_DIR="$REPO_DIR" \
  --setenv=PYTHON_BIN="$PYTHON_BIN" \
  --setenv=TORCHRUN_BIN="$TORCHRUN_BIN" \
  "$PWD/run_train.sh"

systemctl --user status libero-sdt-bert-snnfusion-train.service
journalctl --user -u libero-sdt-bert-snnfusion-train.service -f
```

如果暂时没有四张空闲卡：

```bash
REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" TORCHRUN_BIN="$TORCHRUN_BIN" \
NUM_GPUS=4 POLL_SECONDS=60 ./queue_train.sh
```

队列要求 GPU 连续两次满足显存占用不超过 1 GB、利用率不超过 10% 才会启动。

## 8. 中断后续训

正式输出目录保持不变：

```bash
GPU_IDS=0,1,2,3 \
RESUME_MODE=all \
REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" TORCHRUN_BIN="$TORCHRUN_BIN" \
./run_train.sh
```

`RESUME_MODE=all` 会恢复最新 5k checkpoint 的模型、EMA、optimizer、scheduler 和 global step。不要使用 `resume_mode=model` 代替正式续训。

## 9. 训练过程检查

```bash
tail -f output/train/logs/train_4gpu_*.log
ls -lh output/train/checkpoints/
nvidia-smi
```

重点观察：

- loss、action loss、feature KD、action KD 是否有限且下降；
- firing rate 不是全 0 或全 1；
- `temporal_readout_logits` 存在于 checkpoint；
- BERT 参数没有进入 trainable parameter 列表；
- 每 5k checkpoint 同时含 online 和 EMA state。

检查 5k checkpoint 元数据：

```bash
"$PYTHON_BIN" - <<'PY'
import torch
p = "output/train/checkpoints/turbovla_sdtv3_bert_snnfusion_learned_5000.pth"
c = torch.load(p, map_location="cpu", weights_only=False)
print(c["global_step"], c["loss"])
print(c["model_config"]["text"]["encoder_type"])
print(c["model_config"]["text"]["frozen"])
print(c["model_config"]["spike"]["downstream_type"])
print(c["model_config"]["spike"]["temporal_readout"])
print(c["model_config"]["action"]["decoder_type"])
print([k for k in c["model_state_dict"] if "temporal_readout_logits" in k])
PY
```

预期输出依次包含 `bert / True / spiking / learned / ann`。

## 10. 准备 clean LIBERO 评测环境

必须另建干净 checkout，不覆盖服务器原来的 LIBERO：

```bash
git clone <LIBERO_REPOSITORY_URL> /path/to/LIBERO-8f1084e-clean
git -C /path/to/LIBERO-8f1084e-clean checkout 8f1084e
git -C /path/to/LIBERO-8f1084e-clean status --short
git -C /path/to/LIBERO-8f1084e-clean rev-parse HEAD
```

要求：

```text
HEAD = 8f1084e3132a39270c3a13ebe37270a43ece2a01
git status --short 为空
```

## 11. 正式评测

训练完成后使用 80k EMA、FP32、seed 7、每任务 50 trials、chunk/open-loop 12：

```bash
export LIBERO_CHECKOUT=/path/to/LIBERO-8f1084e-clean

EVAL_GPU_IDS=0,1,2,3 \
REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" \
LIBERO_CHECKOUT="$LIBERO_CHECKOUT" \
./run_eval.sh | tee output/eval_runner.log
```

评测会先执行三项强校验：

1. LIBERO commit 精确等于 `8f1084e...` 且 worktree 干净；
2. 从训练 checkpoint 导出并标注 EMA 权重，同时把资源路径重写为新服务器路径；
3. 在 rollout 前进行 checkpoint strict-load dry run。

四个 suite 各占一张 GPU，并行完成 40 个任务、总计 2000 episodes。已有同协议运行的墙钟时间约 15-16 小时。

最终结果：

```text
output/eval/step_80000_ema_clean_8f1084e/results/libero_spatial.json
output/eval/step_80000_ema_clean_8f1084e/results/libero_object.json
output/eval/step_80000_ema_clean_8f1084e/results/libero_goal.json
output/eval/step_80000_ema_clean_8f1084e/results/libero_10.json
output/eval/step_80000_ema_clean_8f1084e/results/summary.json
```

## 12. 对比原则

只和完全相同协议的结果比较：相同 clean LIBERO commit、FP32、seed 7、50 trials/task、open-loop 12、相同相机和 action normalization。建议至少报告：

| 模型 | 环境 | 权重 | 成功率 |
|---|---|---|---|
| 官方 TurboVLA | clean `8f1084e` | 官方指定 checkpoint | 同脚本重跑 |
| SDT-V3+BERT ANN fusion | clean `8f1084e` | 80k EMA | 需要补跑干净环境 |
| 本实验 SNN fusion | clean `8f1084e` | 80k EMA | `summary.json` |

此前 SDT-V3+BERT 的 93.95% 来自旧的纹理损坏环境，不能直接作为本次 clean 环境的严格对照。

## 13. 停止任务

前台或 nohup 任务先查精确 PID/PGID，再只停止本实验：

```bash
ps -eo pid,ppid,pgid,etime,cmd | grep 'turbovla.training.train_mixed'
kill -TERM -- -<本实验PGID>
```

systemd 托管任务：

```bash
systemctl --user stop libero-sdt-bert-snnfusion-train.service
```

不要结束其他用户的 Python、torchrun 或 GPU 进程。
