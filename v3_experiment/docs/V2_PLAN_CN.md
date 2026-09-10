# v2 实验计划:双端 SNN 前端 + 官方模型蒸馏(LIBERO)

> 2026-09-05 准备就绪,等官方 checkpoint 到货即可启动。
> 所有脚本已通过预检/dry-run,**尚未启动任何训练**(遵守"先不跑实验")。

## 1. 结论回顾

- **实验目标(用户 2026-09-05 明确):做出与官方模型同台对比的双 SNN 模型。对比基准 = 官方模型
  (clean 锚点评测,第④步)。此前跑出的 17.4%(BERT teacher)/ 6.75%(BERT 学生)是错误实验
  (错误 teacher + 冻结 BERT + KD 0.1/0.1),只作诊断结论,不作基准。**
- clean 协议下:SpikingLM 架构模型 **55.95%**(唯一确证的真 SpikingLM 架构成绩,仅作参考)、
  BERT teacher 17.4%、BERT 学生 6.75%。
  **93.95% 历史成绩归因更正(2026-09-05 二审,基于旧评测日志硬证据)**:旧日志明确构造的是
  **标准 BertModel**("Some weights of BertModel were not initialized ... pooler.dense.*"),
  且旧训练脚本未传 `--text_encoder_type spikinglm`;SpikingLM 权重目录本身是 BERT 命名格式
  (model_type=bert,无 pooler)。故 93.95% = **BERT 架构(权重取自 SpikingLM 目录,pooler 全新初始化)**,
  不是 SpikingLM 架构成绩。RUNBOOK 的 BERT 描述更符合证据。
  **"文本编码器是决定性变量"不作因果结论**:55.95% 与 17.4%/93.95% 之间混有训练谱系、KD teacher、
  fusion 配置、评测代码版本差异,不能单变量归因。
  **55.95% 不再作为热启动来源:2026-09-05 决定 v2 从头训,初始化对齐官方。**
- v2 = SDT-V3(SNN 视觉)+ SpikingLM(SNN 文本,**冻结 —— TEXT_MODE 已按官方 ckpt 记录仲裁为
  frozen**,2026-09-05;官方 trainer 默认 freeze 且发布 ckpt 记录 text.frozen=True,README "online"
  表述与实物不符)+ SNN 融合,KD 从**官方模型**蒸馏。成败判据 = 在 clean 协议下逼近官方锚点的程度。
  冻结 SpikingLM + 全新 SNN 融合 + vision lr 5e-5 在 LIBERO 线仍是首次组合 → **正式训练前先跑
  1-2k 步 smoke**。

## 2. v2 结构

| 部件 | 来源 |
|---|---|
| 视觉 | SDT-V3 19M(SNN),从分类预训练权重出发(对齐官方 pretrained-backbone 起点,lr 同官方 5e-5) |
| 文本 | **SpikingLM 冻结**(`TEXT_MODE=frozen`,与官方 ckpt 记录 text.frozen=True 一致;从 GLUE 预训练权重出发,不加载 pooler) |
| 融合 | **SNN 6 层(Spike2Max exact,learned temporal readout)**,全新初始化(486 张量/14.3M;官方可从 GroundingDINO 预加载,SNN 架构装不进) |
| 动作头 | ANN ACT,全新初始化(官方动作头同样全新) |
| KD teacher | 官方 TurboVLA LIBERO ckpt(DINOv3 ViT-B + BERT + ANN),**feature 0.25 / action 0.5** |

> **无学生热启动(2026-09-05 决定)**:官方没有 student_init 概念(README 只传
> `--pretrained_init_ckpt` GroundingDINO 给 enhancer/text_projection 预加载),v2 删除
> `--student_init_checkpoint`,与官方一样从头训 80k。

KD 通路已核实(`code/turbovla/training/trainer.py`):
- teacher 由 ckpt 内 `model_config` 重建,vision 支持 `dinov3`,text 强制 BERT(对官方模型恰好正确),随后严格加载;
- feature 蒸馏在 `vision_projection` 之后的 256 维 token 上做 MSE(两架构同维,shape 天然对齐);
- 数据管线自动产出 `teacher_pixel_values`(DINOv3 预处理),即"官方 teacher → SNN 学生"本就是这个包设计好的通路。

## 3. snnve 80k 配方(仅作历史参考,v2 已不使用其热启动)

(dump 自 checkpoint,EMA 导出于 2026-08-30)

模型侧:`model_config` 见 `results/libero_sdtv3_spikinglm_80k_ema_clean_8f1084e/checkpoints/turbovla_snnve_spikebert_ann_distill_80000_ema.pth`
(text: spikinglm/frozen/T=4;vision: sdtv3_19m@224;interaction 256×6;action ann;spike downstream ann;loss 0.1905;ema_decay 0.999;distillation={teacher libero_2gpu/turbovla_libero_55000.pth, feature 0.25, action 0.5})。

训练侧(同家族 2gpu 配方,dump 自 `resources/outputs/checkpoints/libero_sdtv3_19m_2gpu/turbovla_sdtv3_19m_80000.pth` 的 args):
batch 32/GPU × accum 2、lr 5e-5(head 同)、vision_encoder 1e-5、wd 1e-10、80k 步、warmup 10k、min_lr_ratio 1.0、FP32、seed 42、4 个 no_noops 套件、stats_key libero_all4_no_noops。

## 4. 到货后的操作顺序

```bash
# ① 有网机器下载官方发布(几个 GB)
#   【已完成 2026-09-05】本机经 hf-mirror 直下成功(1.7G):
#   HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('H-EmbodVis/TurboVLA', local_dir='.../resources/pretrained/TurboVLA')"
#   OFFICIAL_LIBERO_CKPT=/home/dwh/work/vla/libero_vla/resources/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth

# ② 预检 Part B:验证官方 ckpt 能否严格加载为 KD teacher(需 GPU 亦可在 CPU 跑)
cd /home/dwh/work/vla/libero_vla/libero_sdtv3_bert_snnfusion_20260831
/home/dwh/venvs/turbovla-libero/bin/python preflight_v2.py <OFFICIAL_LIBERO_CKPT>
#   -> "teacher built OK: ~216M parameters" 即通过;
#      若 STRICT-LOAD MISMATCH,按打印的 missing/unexpected 写 key 映射进 convert_official_ckpt.py
#   同时运行转换器查看 ckpt 自述信息(决定 TEXT_MODE 与发布物结构):
/home/dwh/venvs/turbovla-libero/bin/python convert_official_ckpt.py \
  --src <OFFICIAL_LIBERO_CKPT> --dst /tmp/probe.pth --weight_source model | head -20
#   -> 打印 ckpt 的 text.frozen / text.encoder_type / vision.compute_precision / args.dataset_dirs
#   **TEXT_MODE 仲裁【已完成 2026-09-05】:官方 ckpt 记录 text.frozen=True → TEXT_MODE=frozen 定案**
#   (官方 trainer 默认 freeze_text_encoder=True、README 称 online BERT,两处矛盾;
#    实物 ckpt 与 trainer 默认一致,即官方训练时文本是冻结的。README 表述与实物不符。)
#   ② 步其余结论:发布物为单 ckpt(turbovla_libero.pth,672 张量,216.1M,严格加载零缺漏);
#   顶层仅 model_state_dict + model_config(无 EMA、无 args、无 global_step);
#   vision.compute_precision=bf16_autocast(证实 teacher 强制 fp32 的必要性);
#   padding_length_by_instruction 本包原生支持(configuration.py:16)。
#   若发布物是每套件一个 ckpt(README 示例 checkpoints/libero/libero_object.pth),
#   评测时用 OFFICIAL_CKPT_SPATIAL/OBJECT/GOAL/LIBERO_10 分别指定

# ③ 官方模型 clean 评测(补上从未跑成的官方锚点;GPU 3/5 等空闲卡)
OFFICIAL_CKPT=<OFFICIAL_LIBERO_CKPT> EVAL_GPU_IDS=<4块空闲卡> ./run_eval_official.sh
#   或 per-suite 发布物时:
#   OFFICIAL_CKPT_SPATIAL=... OFFICIAL_CKPT_OBJECT=... OFFICIAL_CKPT_GOAL=... \
#   OFFICIAL_CKPT_LIBERO_10=... EVAL_GPU_IDS=<4块空闲卡> ./run_eval_official.sh
#   产物:output/eval/official_clean_8f1084e/results/summary.json

# ④ v2 训练(TEXT_MODE 已仲裁为 frozen;启动器门禁会自动核对,不符即拒启)
#   4a) 从头训练 smoke(必须先跑,~1-2 小时,单卡):冻结 SpikingLM + 全新 SNN 融合
#       都是首次组合训练;验证显存、loss 下降、无 NaN、TEXT_MODE 门禁通过。
#       注意:smoke 全局 batch=16,正式=256 —— smoke 只验证工程启动,不证明正式优化动力学;
#       warmup 默认 10000 会触发 trainer 的 schedule>warmup 校验,显式给 250(脚本也会自动钳制)
MAX_STEPS=2000 WARMUP_STEPS=250 TEXT_MODE=frozen GPU_IDS=<1块空闲卡> BATCH_SIZE=16 TARGET_GLOBAL_BATCH=16 \
  CHECKPOINT_PREFIX=turbovla_v2_smoke \
  OFFICIAL_CKPT=$PWD/../resources/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth ./run_train_v2.sh
#   失败(loss 不降/NaN/OOM/TEXT_MODE 门禁拦截)→ 按门禁提示调整 TEXT_MODE 后重跑 4a;通过 → 4b
#   4b) 正式训练(~2 天,4×4090,80k 步,从头训,TEXT_MODE=frozen,全 lr 5e-5 对齐官方)
GPU_IDS=<4块空闲卡> TEXT_MODE=frozen BATCH_SIZE=8 TARGET_GLOBAL_BATCH=256 \
  OFFICIAL_CKPT=$PWD/../resources/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth \
  ./run_train_v2.sh
#   **BATCH_SIZE=8 的原因(2026-09-05 实测)**:单卡 fp32 batch16 训练 ≈21.5GB,本机 7 张 4090
#   均有其他用户(zqp ablation)0.8-4.6GB 共存进程,batch16 只在共存 ≤2.2GB 的卡勉强放下,
#   4 张卡同时满足太脆弱。batch8×4GPU×accum8 = 同一全局 batch 256(无 BN、LayerNorm,
#   优化数学不变;微批内 masked-loss 平均粒度略粗 = 工程偏差,记录在案),显存 ~13GB/卡全适配,
#   代价 ~1.5 天(约 4-4.5 天 vs batch16 ~3 天)。若届时 4 卡真被清空,可改回 BATCH_SIZE=16。
#   预演不出进程:DRY_RUN=1 GPU_IDS=... TEXT_MODE=frozen OFFICIAL_CKPT=... ./run_train_v2.sh
#   产物:output/train_v2/checkpoints/turbovla_v2_snnfusion_*.pth

# ⑤ v2 clean 评测(复用 run_eval.sh,SOURCE_CKPT 指向 v2 ckpt)
#   **WEIGHT_SOURCE=raw:官方发布无 EMA,锚点=raw;主对比必须同口径(raw↔raw)。
#   EMA 可另跑一次(WEIGHT_SOURCE=ema,输出目录自动区分)仅作参考。**
REPO_DIR=/home/dwh/work/vla/libero_vla/resources \
SOURCE_CKPT=$PWD/output/train_v2/checkpoints/turbovla_v2_snnfusion_80000.pth \
WEIGHT_SOURCE=raw \
BERT_PATH=/home/dwh/work/vla/libero_vla/resources/pretrained/SpikingLM \
LIBERO_CHECKOUT=/home/dwh/work/vla/libero_vla/environments/LIBERO-8f1084e-clean \
EVAL_OUTPUT_ROOT=$PWD/output/eval/v2_step_80000_raw_clean_8f1084e \
EVAL_GPU_IDS=<4块空闲卡> \
MODEL_LABEL="SDT-V3 + frozen SpikingLM + SNN fusion (v2)" ./run_eval.sh
#   注意:BERT_PATH 必须指 SpikingLM 权重目录(导出步骤用它改写 model_config 的 text 路径,
#   并会把 text.model_source_path 重写到本地 SpikingLM 源码目录);
#   TEXT_MODE 已仲裁为 frozen,MODEL_LABEL 相应写 frozen。
#   评测入口 evaluate.py 带 torch.load 兼容 shim(torch 2.6 weights_only 默认翻转,
#   LIBERO init-states 裸加载会崩),属运行指纹覆盖的同一份代码,锚点与 v2 共用。
```

## 5. 预检已验证(2026-09-05)

| 项 | 结果 |
|---|---|
| Part A:v2 学生从头构建(无热启动) | ✅ 1421 张量;文本在线 249 张量可训练、无冻结;SNN 融合 486 张量/14.3M 全新 |
| Part B:官方 ckpt 作 KD teacher | ✅ teacher built OK: 216.1M parameters(严格加载零缺漏,2026-09-05) |
| ② 转换器自述探查 | ✅ 单 ckpt(672 张量);text.frozen=**True**(→TEXT_MODE=frozen 定案);vision bf16_autocast;无 EMA/args/global_step;padding_length_by_instruction 本包原生支持 |
| SpikingLM 资产 | ✅ 已拷入 `resources/pretrained/SpikingLM`(419M)+ `resources/third_party/SpikingLM` |
| KD teacher 代码通路 | ✅ 静态核实(model_config 重建 + dinov3 + 256 维 feature 对齐 + teacher_pixel_values) |
| 转换器 + 评测链路 | ✅ 用随机权重假官方 ckpt(672 张量,216M)端到端走通:裸 state dict → 严格加载 → FP32 policy → dry-run |
| 训练环境 | ✅ 修复:numpy 2.2.6 → 1.26.4(从 robotwin-v1 环境复制,备份在 /tmp/venv_numpy226_backup);tf 2.15.1 + tfds 4.9.4 + cupy 13.6 + torch 2.6 全部可导入 |

## 6. 已知风险与偏差清单

### 6.1 协议/超参对齐核查(2026-09-05 二审:逐函数 diff 官方 repo 后更新)
- **已对齐**(标 ✦ 的是本轮逐行 diff/逐默认值核实的):
  全局 batch 256(16×4×accum4)、80k 步、warmup 10k、min_lr_ratio 1.0、
  lr 5e-5/head 5e-5/**vision lr 5e-5(已从 1e-5 改齐官方 dinov3_lr)**/wd 1e-10、
  **✦ 文本编码器训练模式 = frozen(已仲裁 2026-09-05):官方 trainer 默认
  freeze_text_encoder=True(DWH-VLA/TurboVLA/trainer.py:117 set_defaults)、发布 ckpt 记录
  text.frozen=True,两者一致;README "online BERT" 表述与实物不符,以实物为准。
  run_train_v2.sh 自动门禁保持启用(启动时读官方 ckpt 的 text.frozen,与 TEXT_MODE 不符即拒启)**、
  grad clip 1.0、shuffle_buffer 512、
  step_mix 64、text_dropout 0、fusion_droppath 0.1、max_text_len 256、expected_image_size 256、
  fp32、seed 42、✦ 动作损失 = 官方同一 `masked_l1_loss`(closed_loop 包装在默认权重下短路,逐行验证)、
  ✦ `build_scheduler` 逐行 identical、✦ 优化器分组逻辑 identical(仅 dinov3→vision 改名)、
  ✦ EMA 同一 pi05 模块同一 EMA_DECAY、
  ✦ shuffle_steps_within_episode=True(官方 trainer.py:109 与本包 trainer.py:197 set_defaults 均为
  True,两边都不传即一致;2026-09-05 二审更正,一审文档误写 False)、
  ✦ num_workers 4(已对齐)、数据 4 套件 no_noops + libero_all4_no_noops stats、
  **✦ 初始化路线对齐官方:预训练骨干出发、无热启动,从头训 80k(官方无 student_init 概念;
  官方 README 传 `--pretrained_init_ckpt` GroundingDINO 给 enhancer/text_projection 预加载,
  v2 无对应物,SNN 融合全新初始化)**、
  评测 50 trials/chunk 12/open-loop 12/seed 7/LIBERO 8f1084e。
- **实验性偏差(= 实验本身,对齐即取消实验)**:
  ① 视觉 SDT-V3@224 vs DINOv3@256(官方 DINOv3 前向 bf16_autocast,SDT 走 fp32);
  ② 文本 SpikingLM vs BERT(训练模式已对齐);③ 融合 SNN vs ANN enhancer;
  ④ SNN 融合无法加载 GroundingDINO 预加载(全新 LayerScale 初始化);⑤ KD 0.25/0.5(官方无蒸馏)。
- **工程偏差(经评估保持现状,均不触碰学生优化数学/不构成不公平)**:
  ① KD teacher 前向 fp32(官方训练时 DINOv3 bf16_autocast)—— 带 model_config 的 teacher 路径
  **已代码强制**(build_distillation_teacher 显式置 `config.vision/interaction.compute_precision="fp32"`);
  无 model_config 的 fallback 路径依赖启动器传入参数(run_train_v2.sh 传 fp32);
  ② KD teacher 输入 224px(官方训练 256px)—— **这是 shape 约束而非疏漏**:feature 蒸馏在
  trainer.py:1258 严格校验 token 形状(224→14×14=196 token 逐 token MSE,256 会出 256 token 直接崩),
  且 snnve 谱系用同一管线从 DINOv3 架构 teacher 蒸出过可用模型;
  ③ save_steps 5000(官方 1000)—— 仅磁盘 I/O;
  ④ 评测 fp32(官方 README 示例 bf16)—— 锚点与 v2 用同一脚本,内部自洽,fp32 只会更准;
  ⑤ `--no_allow_hf_download`(离线机器,本地路径等价);
  ⑥ ④b 若以 BATCH_SIZE=8×4GPU×accum8 跑(全局 batch 仍 256):微批 8 vs 官方 16 仅改变
     masked-loss 的微批内平均粒度(无 BN,优化数学不变),起因 = 本机共存进程显存约束(见 §4)。

### 6.2 其余风险

1. **从头训风险(决定 2026-09-05)**:冻结 SpikingLM + 全新 SNN 融合在 LIBERO 线是首次组合训练
   (SNN 融合从头训仅在 RoboTwin E4 验证过),vision lr 也从 1e-5 提到官方 5e-5。
   缓解 = 4a 强制 smoke(2k 步单卡):loss 不降/NaN/OOM 任一出现即停,再考虑回退 vision lr。
   **smoke 通过前绝不启动 4b。**
2. **官方 ckpt 权重 key 命名**:转换器/KD teacher 都是严格加载。若 HF 发布的存储 key 与本包模型类不一致
   (概率低,同源代码),预检 Part B / 转换器会打印完整 missing/unexpected 清单,按需加映射即可。
3. 官方发布可能是**每套件一个 ckpt**(README 示例 `checkpoints/libero/libero_object.pth`):
   评测可按套件分别跑(每次改 OFFICIAL_CKPT);KD 用任一套件 ckpt 皆可(同一模型)。
4. 官方模型若以 bf16 存储,转换器原样保存;评测脚本固定 FP32(policy 内部会 float())。

## 7. 文件清单(本次新增)

- `run_train_v2.sh` — v2 训练启动器(DRY_RUN=1 可预演)
- `preflight_v2.py` — Part A 学生从头构建校验(已跑通)/ Part B 官方 teacher 校验(待官方 ckpt)
- `convert_official_ckpt.py` — 官方 ckpt → 本包评测格式(含 key 诊断、`--weight_source` 开关)
- `run_eval_official.sh` — 官方模型 clean 评测(4 套件并行,WEIGHT_SOURCE 可选)
- `V2_FLOW_AND_COMPARISON_CN.md` — 实验流程 + 与官方对比的协议同一性表(逐项标注实现代码行号)
