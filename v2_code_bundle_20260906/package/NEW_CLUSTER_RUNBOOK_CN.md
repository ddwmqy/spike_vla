# v2 正式训练(④b)新集群部署说明

打包时间 2026-09-06。来源机器:amax(本包 ④a smoke 已通过;③ 官方锚点评测仍在源机跑)。

## 1. 包内容

```
v2_train_bundle_20260906/
├── package/                  # 实验包(代码 + 启动脚本 + 文档)
│   ├── run_train_v2.sh       # ④b 训练启动器(TEXT_MODE 门禁、warmup 自动钳制)
│   ├── preflight_v2.py       # Part A 学生构建 / Part B teacher 严格加载
│   ├── convert_official_ckpt.py / evaluate.py / run_eval*.sh / run_eval.sh
│   ├── code/turbovla/...     # 训练与模型代码
│   ├── third_party/vla_adapter/  # 评测 rollout 适配层
│   ├── V2_PLAN_CN.md / V2_FLOW_AND_COMPARISON_CN.md   # 计划与对齐审计(先读 §4/§6.1)
│   └── NEW_CLUSTER_RUNBOOK_CN.md(本文件)
├── resources/                # 全部资产(布局与源机 resources/ 一致,REPO_DIR 指向这里即可)
│   ├── data/libero/{libero_10,libero_goal,libero_object,libero_spatial}_no_noops/1.0.0  # RLDS 9.8G
│   ├── pretrained/{dinov3-vitb16-pretrain-lvd1689m, bert-base-uncased, SpikingLM,
│   │              V3_19.0M_1x4.pth, TurboVLA/checkpoints/libero/turbovla_libero.pth}
│   ├── third_party/{SpikingLM, Spike-Driven-Transformer-V3/.../models.py}
│   └── experiments/libero/configs/{libero_all4_stats.json, online_text_layout.json}
└── environment/{pyvenv.cfg, pip-freeze.txt}   # 源机环境快照
```

## 2. 环境要求(以 environment/pip-freeze.txt 为准)

- Python 3.10;torch **2.6.0+cu124**(新集群驱动需支持 CUDA 12.4;换 torch 版本未验证,自行承担)
- **numpy 必须 1.26.4(<2)**:tensorflow 2.15.1 与 numpy 2.x 不兼容(`_ARRAY_API not found`),源机踩过
- 关键包:tensorflow 2.15.1、tfds 4.9.4、cupy 13.6、transformers/timm 按 freeze 版本
- 离线运行:脚本内部已设 `HF_HUB_OFFLINE=1`,所有权重均为本地路径,不触网

## 3. 启动前预检(建议,~2 分钟)

```bash
cd <bundle>/package
python preflight_v2.py <bundle>/resources/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth
# 期望:Part A "student built OK: 1421 state tensors";Part B "teacher built OK: 216.1M parameters"
# 再跑 DRY_RUN 看完整命令:
DRY_RUN=1 GPU_IDS=0,1,2,3 TEXT_MODE=frozen BATCH_SIZE=16 TARGET_GLOBAL_BATCH=256 \
  REPO_DIR=<bundle>/resources \
  OFFICIAL_CKPT=<bundle>/resources/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth \
  bash run_train_v2.sh
```

## 4. 正式训练启动

```bash
cd <bundle>/package
GPU_IDS=0,1,2,3 TEXT_MODE=frozen BATCH_SIZE=16 TARGET_GLOBAL_BATCH=256 \
  REPO_DIR=<bundle>/resources \
  OFFICIAL_CKPT=<bundle>/resources/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth \
  nohup bash run_train_v2.sh > train_4b.log 2>&1 &
```

- **TEXT_MODE=frozen 是定案**,不要改:官方 ckpt 实测 `text.frozen=True`;若误传 online,
  启动器门禁会自动拒绝(`FORCE_TEXT_MODE=1` 可强制,但不要用)。
- **BATCH_SIZE 选择**:单卡 fp32 batch16 ≈ **21.5GB**。4 卡均为空(≥24GB 空闲)→ 用 16(官方档位);
  任何卡有共存进程 ≥2GB → 改 `BATCH_SIZE=8`(显存 ~13GB,全局 batch 仍 256,优化数学不变,
  慢 ~1.5 天,源机因此被迫用 8;新集群若干净就直接 16)。
- PYTHON_BIN/TORCHRUN_BIN:默认 `/home/dwh/venvs/turbovla-libero/bin/{python,torchrun}`,
  新集群路径不同必须显式传。
- 预计时长:80k 步,batch16 ≈ 3 天 / batch8 ≈ 4.5 天(4×4090 基准;更快卡按比例)。
- 产物:`package/output/train_v2/checkpoints/turbovla_v2_snnfusion_{5000..80000}.pth`
  与 `output/train_v2/logs/train_v2_*.log`。

## 5. 跑完拷回

```
output/train_v2/checkpoints/turbovla_v2_snnfusion_80000.pth(主)
output/train_v2/checkpoints/*_50000.pth 等中间档(备用)
output/train_v2/logs/*.log(诊断)
```

⑤ v2 评测(clean 协议,WEIGHT_SOURCE=raw 主对比)需要 LIBERO 8f1084e 环境 + 评测代码,
**不在本包内**,回源机或另配环境后按 `V2_PLAN_CN.md` §4 ⑤ 执行。
