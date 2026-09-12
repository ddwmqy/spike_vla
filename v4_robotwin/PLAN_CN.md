# v4-robotwin 任务规划 v7:RoboTwin 2.0 clean50 对标 TurboVLA

> **状态:规划稿 v7(2026-09-11)。v6 全部机制条款(§1–§5.6、§6)原样继承,本轮只做四件事:**
> ① **主线改三段式**——依据用户 2026-09-11 指令"用 v3 架构(=A)跑 RoboTwin 试试",
>   A 的 1k 冒烟(A-S)前置去风险,C1∥B 仍为科学结论主体,A 是否 55k 正式视 B 结论拍板;
> ② 新增 **§5.7 方案 A 接入提纲**(移植单元 + 验证门 + 待拍板项)与 **§4-P0A / §4-A-S** 阶段;
> ③ 新增 **§5.8 C1-vs-B 判定规则(预注册)**;
> ④ §7 补**总预算表**;资产事实修正:RoboTwin 2.0 模拟器仓库 = **robotwin-Platform/RoboTwin**
>   (TianxingChen/RoboTwin 是 v1 落地页,主分支仅 README),官方 55k EMA ckpt 已落地
>   `v4_assets/`——进度明细见 `ASSETS.md`。
> v6 历史(机制未变):1k 短跑专用配置跨触发点、warmup 跨第 1000 个 optimizer step、
> 文本投影"重初始化"与"校准"同为须声明的干预;两级精度参考(高精度参考 / 自身训练配置参考,
> 官方默认降为候选)、跨精度动作比较限定在固定输入批、夹爪双口径指标、快照时机与来源、
> 两臂统一 micro-batch+accum 档位、optimizer.step() 空操作澄清。
> 官方参考代码:`/data/260010028/dwh_vla/TurboVLA_official`,commit **b29ab14**(2026-09-02)。

## 0. 一句话目标

在 RoboTwin 2.0 clean50(50 个双臂任务)上回答两个问题:

1. **主问题(成对重训对照)**:C1(ANN BERT 干净重训)与 B(sootspike 文本)同 seed、
   同数据、同全局 batch、同步数、同评测协议各训一次、各评一次,与官方发布 60.2% 对比
   ——检验"文本 spike 化在双臂基准上是否成立"(判定规则预注册于 §5.8,评测开始前不得更改)。
2. **前置去风险(2026-09-11 新增)**:A(=v3 三组件 spike 架构:SDT-V3 视觉 + sootspike
   文本 + Spike2Max 融合)先做 **1k 短训冒烟(A-S)**,把计划自标的最大风险——"spike 组件
   从未在双臂/3 视图/SAPIEN 上验证过"——最先排掉;A 是否投入 55k 正式训练,视 B 结论与
   算力再拍(§4)。

## 1. 官方基准口径(含工程事实)

### 1.1 发布结果与协议

| 项 | 值 |
|---|---|
| 基准 | RoboTwin 2.0,clean50 = 50 任务 clean 设定 |
| 官方数字 | **60.2%** |
| 评测 ckpt | step-55k **EMA**;加载要求某级祖先目录同时含 `config.yaml` + `dataset_statistics.json`(`share_tools.py:83`) |
| 评测规模 | 每任务 **100 trials**(脚本默认 20) |
| 评测模式 | `--mode demo_clean` |

### 1.2 训练配方(clean50.yaml + train.sh)

| 项 | 值 |
|---|---|
| 全局 batch | `GPU数 × per-device × grad_accum`(train.sh:61);官方 4×48×1=192;单卡 48×4 / 32×6 / 24×8 可精确凑 192,前提:§5.4 修正版 trainer,**两臂同一套 micro-batch+accum 档位(§4-P0T)** |
| 步数 / lr / EMA / 训练 seed | 100k(发布点 55k)/ 5e-5 / 0.999 / **42** |
| **精度(三层语义)** | ① **参数 dtype:训练默认 FP32**(ds_config.json 无 bf16 段,ZeRO-2 FP32 主权重);② 模块 autocast:`encode_condition` 对文本+视觉+融合整体开 BF16 autocast(`turbovla/models/turbovla.py:179`),且 **DINOv3 有独立 precision 配置**(`vision_encoder.py:83-102`,默认 bf16_autocast);③ FP32 matmul 策略(`float32_matmul_precision`)。**以上为按当前源码/配置确认的默认组合,尚无 RoboTwin 实测记录**。整模型参数转 BF16 是评测服务行为(`--use_bf16`,默认开;关闭**不改变**内部 autocast) |
| 数据 | StarVLA/RoboTwin-Clean(HF LeRobot/parquet),50 任务 `Clean/<task_name>` |
| 模型 | DINOv3 **ViT-L** × 3 视图(高+双腕)参与训练,BERT frozen,6 层 interaction,ACT 头 horizon 50、14-D 双臂 abs qpos:**12 个关节分量 min-max 归一化 + 2 个夹爪分量 binary**(非 14 维全 min-max) |
| 初始化 | `pretrained_ckpt=groundingdino_swint_ogc.pth`;`load_bert: true` 会把其中普通 BERT 参数加载进文本编码器(§5.2) |
| seed 时序缺陷 | `main()` 先 `build_framework`(train_starvla.py:462)后设 seed;文本构造扰动 RNG → §5.5 快照法 |

### 1.3 评测栈工程事实(含协议参数,全部已核实)

- launcher 只建日志目录(`start_eval.sh:508` 附近),**无聚合器** → §6 解析器。
- server 与 simulator 绑同一 gpu_id(`eval_task.sh:52`);分卡是待开发改造。
- **评测协议默认值**:`eval_task.sh` 评测 seed 默认 **0**(与训练 seed 42 无关联);`deploy_policy.yml`:`instruction_type: unseen`、`action_ensemble: true`、`action_ensemble_horizon: 50`、`adaptive_ensemble_alpha: 0.1`、`normalization_mode: min_max`、`binary_threshold: 0.49`。
- **夹爪经两次二值化**:先按 `binary_threshold=0.49` 归一化域内二值化,temporal ensemble 后
  **再按 `>= 0.5` 二值化一次**(`model2robotwin_interface.py:205`,`binary_action_source` 默认 ensemble)。
- RoboTwin 2.0 = SAPIEN 独立 env;`pip install -e ".[robotwin]"` **不含 FlashAttention-2**,须另装。

## 2. 与现有 v3 资产的差距

| 维度 | LIBERO(v3 现状) | RoboTwin clean50 | 差距性质 |
|---|---|---|---|
| 臂 | 单臂 7-D | 双臂 **14-D**(12 关节 min-max + 2 夹爪 binary) | 动作头/状态适配 |
| 视图 | **2 视图**(trainer.py:351) | **3 视图** | 输入管道适配 |
| 视觉骨干 | SDT-V3 SNN(v3)/ ViT-B(官方 LIBERO) | ViT-L 参与训练 | 权重不通用 |
| 精度 | 三层语义同样适用,取值组合以 FP32 为主 | 三层取值组合不同(§1.2),官方 FP32 参数 + BF16 autocast | SmoothSpike 精度敏感 → §5.3 |
| 模型构建路径 | 本包 trainer 直构 | **starVLA wrapper `_core_config` 硬编码 BERT**(`VLM4A/TurboVLA.py:116`) | §5.1 |
| 数据管道 | LIBERO RLDS(`libero_rlds.py`) | **LeRobot/parquet**(starVLA runtime) | spike 模块输入衔接(分辨率/文本长度/padding)→ §5.7 验证门 |
| 模拟器 | MuJoCo | SAPIEN(独立 env) | 全新评测栈 |
| spike 组件验证 | LIBERO 真卡验证 + 80k 训练 | **从未验证** | 核心风险 |

## 3. 方案定义

| 代号 | 内容 | 说明 |
|---|---|---|
| **C0-official** | 官方 55k EMA ckpt,**官方默认精度**(三层全官方默认) | 环境锚点 + **候选精度配置的 C0 复现口径**;不作 sootspike 数值正确性基准(§5.3) |
| **C0-common**(可选) | 同 ckpt,改用 C1/B 统一选定精度 | 仅当矩阵选中精度 ≠ 官方默认时另跑;评测量另计 |
| **C1** | ANN BERT 干净重训(修正版 trainer,seed 42,全局 192,55k) | B 的单变量训练对照 |
| **B** | 官方架构 + 文本→sootspike(frozen),其余与 C1 逐项相同 | 单变量 = 文本;§5 |
| **A** | **三组件 spike(混合 SNN-ANN)**,模块边界按我方代码:SDT-V3 SNN 视觉 + sootspike 文本(含 **SpikeLinearProjection**,turbovla.py:118)+ Spike2Max 融合(**SpikeFFNResidual 带 LIF**,spiking.py:735);ANN 保留 = ACT 动作头等 | 接入提纲见 §5.7;是否 55k 正式视 B 结论与算力拍板 |
| **A-S** | A 架构 1k 短训冒烟(§5.4 修正版 trainer,seed 42) | 只回答"能否训、数值是否健康、ckpt/EMA 能否落盘";**数字不作结论、不进对照表** |

**主线(v7 三段式):P0E → [P0A 移植 ∥ C0-20 → C0-100(official)] → P0T → A-S 冒烟 → C1 ∥ B 成对训练 → 统一协议评测 → 视 B 结论拍 A 是否 55k 正式。**

## 4. 阶段分解

### P0E 评测环境与 C0 全部资产
1. `turbovla-robotwin` env(python 3.10,`pip install -e ".[robotwin]"`)+ **FlashAttention-2 另装**;RoboTwin 2.0 SAPIEN env(镜像克隆)
2. **C0 资产**:官方 55k ckpt(`config.yaml`+`dataset_statistics.json` 祖先结构)、DINOv3 ViT-L、bert-base-uncased、GroundingDINO、tokenizer/processor 全部就位
3. 钉版本:TurboVLA=b29ab14、RoboTwin=`96c1fea`(robotwin-Platform/RoboTwin,已克隆)、SmoothSpike 源码/权重 sha256 → `ASSETS.md`
4. 最小 sim 冒烟:官方 ckpt × 1 任务 × 5 episodes;`--use_bf16` 开/关各跑通
5. **退出标准**:冒烟通过;资产/版本记录完整。

**进度(2026-09-11)**:RoboTwin 2.0 模拟器仓库已克隆(`robotwin-Platform/RoboTwin` @
`96c1fea`,2026-09-05 提交,752 文件);官方 RoboTwin 55k EMA ckpt(828 MB,
sha256 `d0183df6…f7c034`)+ `config.yaml` + `dataset_statistics.json` 已落地
`v4_assets/TurboVLA_robotwin/`(祖先目录结构满足 `share_tools.py:83` 加载要求)。
待办:DINOv3 ViT-L、StarVLA/RoboTwin-Clean 数据、`turbovla-robotwin` env、
GroundingDINO(C1/B 用)——明细见 `ASSETS.md`。

### P0A 方案 A 移植(v7 新增;代码工作不占卡,与 C0 评测并行)
- 内容 = §5.7 四个移植单元(P1 文本 / P2 视觉 / P3 融合 / P4 动作头)+ 验证门;
  wrapper framework 配置补 `text.encoder_type` / `vision.encoder_type` / `fusion.type`
  三分支(默认 = 官方,官方路径零改动)。
- **退出标准**:逐模块结构加载零缺漏 + 参数哈希断言(§5.2 同款);真卡前向单步通过;
  spikingjelly LIF CUDA kernel 在 `turbovla-robotwin` env 冒烟通过;移植差异清单记
  `ASSETS.md`。
- **进度(2026-09-12)**:前两项**已在本机 H100 MIG 上全过**(门 #0 官方路径逐位一致、
  门 #1 A 臂构建+前向、门 #2 权重哈希、门 #2b yaml→wrapper→predict_action、门 #3 真卡
  fp32/bf16 前向);**第三项不适用**——SDT-V3 上游 backbone 只依赖 timm+einops,不含
  spikingjelly LIF kernel(v3 的 Spike2Max 亦为纯 PyTorch autograd),故无需该 env 冒烟;
  移植差异清单见 `ASSETS.md`。**新增产出**:`experiments/robotwin/ARMS_CN.md`(四臂启动
  手册,算力服务器交接用)。

### C0-20(运行筛查)
- 官方 ckpt × 50 × 20 trials:验证 launcher、解析器、分片并行。
- **退出标准**:解析器按实际完成记录提取 50 任务,episode 总数 = 1000 硬校验通过。

### C0-100 official(协议校准)
- 官方 ckpt × 50 × 100 trials,官方默认精度 = 我方环境锚点(5000 局)。
- **统计口径**:与发布 60.2% 为 5000 vs 5000;报告**实际成功数**与**单个成功率估计的双侧
  95% 正态近似区间**(p≈0.6 时约 ±1.4pp);任务内试验非独立,区间只作参考。**±3pp 仅是
  工程排查阈值,不是统计等价证明**。C1-vs-发布值是单 seed 单差值(混含实现/环境/协议差异),
  不得用于估计 seed 方差;所有单 seed 结论只在该 seed 下成立。

### P0T 训练资产
1. `prepare_data.sh` 下载 StarVLA/RoboTwin-Clean(`HF_ENDPOINT=https://hf-mirror.com`;体量实测回填);sootspike/spikingjelly 进新 env
2. **batch 档 probe(两臂统一)**:`NUM_PROCESSES=1` 下扫 24×8 / 32×6 / 48×4(=192),
   **以"C1 与 B 都能容纳"的档位为准统一选定**——文本按批内最长指令 padding,micro-batch
   布局影响输入形状与数值行为,禁止 C1/B 各用不同档位
3. §5.3 验收阈值定稿进 `ASSETS.md`
4. **退出标准**:50 任务数据齐全(哈希抽查);probe 报告(统一档位)回填。

### A-S 冒烟训练(v7 新增;P0A + P0T 完成后即可,不与 C1/B 抢卡)
- 1k 步,§5.4 修正版 trainer + 专用短跑配置(跨过 save/eval 触发点),seed 42;batch 档位
  按 A 自身显存实测选定(**A 与 C1/B 不构成对照,允许不同档**,实测组合记 `ASSETS.md`)。
- **退出标准**:loss 形态正常、无 NaN、ckpt + EMA 落盘;(可选)1 任务 × 20-trial 短评
  跑通 RoboTwin 评测栈。
- **边界:A-S 数字不作任何结论,不进主表、不与官方 60.2% 并列。**

### C1 ∥ B 成对训练(前置:§5.4 修正版 trainer + 续训策略拍板 + §5.5 快照初始化)
- **C1**:官方配方(`load_bert: true` 保持官方行为)。
- **B**:`load_bert: false` + §5.2/§5.3。
- **mask 版本(拍板 2026-09-12,按推荐维持官方原味)**:C1 用 fork 默认 `legacy`(=
  官方数值),B 与 A 显式用 `corrected`。已知该设定下 mask 版本与文本编码器共变
  (C1↔legacy, B/A↔corrected),**作为声明过的组合变量接受**——C1 的定位是官方配方复现
  锚点而非纯文本对照;若审稿或结论需要,可补 C1-corrected 佐证(预算另计,默认不跑)。
- 墙钟(1 卡,全局 192):55k 步 ≈ 2–4 天/次,串行 ≈ 4–8 天;借 4 卡各 ≈1 天。
  *batch 48(accum=1)只见 1/4 数据,不构成同配方,仅预实验。*
- **退出标准**:两臂 EMA-55k 落盘、曲线无异象;`ASSETS.md` 记录计数验证、续训验证(若启用)、
  快照一致性结果(含视觉骨干)、两臂统一档位/步数/时长。

### 统一协议评测(§5.6)
- C1、B 各 100 trials × 50(20-trial 筛查可选);**C1/B 同精度同协议**;
  与 C0-official 对比注明精度组合差异;如需同精度锚点 → C0-common(预算另计)。
- **退出标准**:summary + 逐任务 json + 分母硬校验;`RESULT_CN.md`(实际成功数 + 区间)。

### A(占位,后续另拟方案文档)

## 5. 方案 B / A 的接入与前置修正

### 5.1 starVLA wrapper 接入
`_core_config` 硬编码 BERT → 补丁:framework 配置加 `text.encoder_type / text.model_source_path`
(默认 = bert,官方路径零改动),按类型分支构造 sootspike(自 v2_code_bundle 移植,
权重缺失即报错);训练/评测同 wrapper,自动继承。

### 5.2 init 覆盖防护与文本投影
B:`load_bert: false`;初始化后对 H2/H3 折入权重做哈希断言。**文本投影**:沿用 → 校准 →
重初始化,由小实验定(默认顺序)。**校准与重初始化同为初始化策略干预,采用任一都须在方案
与结果报告中声明**——校准 = 骨干替换 + 接口校准,重初始化 = 骨干替换 + 投影重新初始化,
两者都与 C1 的"无额外干预"区分;校准/预实验权重**禁止续训冒充全新 55k 对照**——正式 B
从同一初始状态重开。

### 5.3 精度矩阵(两级参考 + 固定输入比较 + 验收阈值前置)
三层定义见 §1.2;`dinov3_precision`(视觉内部 autocast)是独立开关,必须显式纳入变量。

**参考配置(两级,修正 v4 的单一官方参考)**:
- **初始化数值检查的参考 = 明确高精度配置**(如:参数 FP32 + 全模块 autocast 关闭 + matmul
  `highest`——具体组合 P0T 定稿进 `ASSETS.md`);
- **部署检查的参考 = 该短训 ckpt 自己训练时的数值配置**(自洽性:换精度部署偏离多少);
- **官方默认三层组合只作候选配置与 C0 复现口径**——若以它为数值基准,官方配置自身误差恒零,
  反而可能把"修复精度的配置"淘汰。

**检查流程**:
1. **数值检查 A(初始化 B,随机动作头)**:固定一批真实输入(观察/状态/指令),独立进程,
   跨精度格比较**文本特征**(逐元素 max|Δ|)与**动作输出**;随机动作头无性能含义,**禁止 rollout 下结论**。
2. **动作误差只在固定输入批上计算**:两种精度格各自对**同一批固定观察/状态/指令**出动作,
   报本节第 4 项指标(验收阈值由 P0T 定稿,§5.3 第 5 条);**rollout 不算格间动作误差**——一旦某步动作不同,后续观察与机器人
   状态分叉,动作差混入环境反馈,无法归因精度。
3. **部署检查(短训 B)**:2k 步短训 ckpt,(a) 固定输入批动作误差(同上);(b) rollout 只比
   **成功率与轨迹表现**(1 任务 × 20-trial);若比较 ensemble 后动作,**固定输入序列并分别
   重置 ensemble 状态**。
4. **指标(全部在固定输入批上)**:
   - 关节动作 MAE / 最大误差(归一化域);
   - **反归一化后的关节误差**(12 关节分量,min-max 换算);
   - 夹爪**双口径**:原始输出阈值一致率(0.49,transform 侧)+ **最终执行指令一致率**
     (ensemble 后 ≥0.5 二值化侧,`model2robotwin_interface.py:205`);
   - 不使用 argmax。
5. **验收阈值在 P0T 定稿并写入 `ASSETS.md`**(数值检查 max|Δ| 上限、固定输入批动作误差上限),
   不达标即回退精度组合。
6. 2k 步短训只确认"该精度可训练、loss 形态正常";格间结论一律基于同一 ckpt 的比较。
7. "文本排除 autocast"方案必须**同时覆盖训练与评测**。
8. **C1/B 最终精度由矩阵决定且两臂一致**;C0-official 固定官方默认;若 ≠ 官方默认,另跑 C0-common。

### 5.4 训练循环计数修正与续训完整性
- **澄清**:每 micro-batch 的 `optimizer.step()` 本身**不算缺陷**——Accelerate 的 DeepSpeed
  wrapper 中该调用是空操作。已确认的问题:`lr_scheduler.step()` 每 micro-batch 推进
  (train_starvla.py:423,accum=4/6/8 时 warmup 1000 实际 ≈250/167/125 个 optimizer step)、
  resume 补步同语义、eval/save 门禁与 optimizer step 不对齐且评估额外消耗训练 batch。
- **修正 + 验证**:按**实际参数更新次数**验证 counting 关系(optimizer step ↔ scheduler ↔
  EMA ↔ save/eval 门禁)。验证用**专用短跑配置**:临时缩短 save/eval 间隔(官方默认
  save_interval=5000,1k 短跑覆盖不到默认间隔),并**跨过触发点运行若干累积周期**;
  warmup 检查须**跨过第 1000 个 optimizer step**,观察边界前后行为;eval 消耗计入。
- **续训完整性**:官方保存以模型权重为主,resume 不完整恢复 optimizer/EMA/RNG/数据进度。
  二选一(拍板):①实现完整状态保存/恢复并验证;②规定 55k 不得中断,中断作废重跑。
  未拍板前按 ② 执行。
- 两臂同用修正版;验证输出记 `ASSETS.md`。

**§5.4 进度(2026-09-12)**:
- **修正已落地**(`v4_code` `third_party/.../train_starvla.py`):① `lr_scheduler.step()` 加
  `sync_gradients` 门控(调度器在 `accelerator.prepare()` 之外创建,原先每 micro-batch 推进,
  warmup 1000 在 accum=N 时实际只有 1000/N 个 optimizer step,且 resume 补步会累积漂移);
  ② eval/save 门禁加 update-step 门控(`completed_steps` 只在同步步变化,原先同一 step 值会
  重复触发、且在 step 0 误触发,每次 eval 额外消耗一个训练 batch,见 `eval_action_model`)
  ③ EMA 侧 `_update_ema` 原本已有 sync 门控,无需改动。**accum=1(官方 RoboTwin 配置)时
  两处门控均为恒真,官方路径行为不变。**
- **CPU 计数验证已过**(`scripts/gate5_trainer_counting.py`,直接驱动真实 `train()` /
  `_train_step`):accum=1 与 4 下 `completed_steps` = scheduler 步 = EMA 更新 = 20,
  eval/save 各恰好 4/2 次;**变异检验**(临时还原旧代码)确认测试有牙:旧代码下
  accum=4 时 scheduler=80、eval=16。
- **GPU 验证已在本机(dev pod 的 H100 MIG 3g.40gb)完成**,真实数据 + 真实模型:
  - accum=1(5 步)与 accum=4(32 micro-batch → 8 step)均 PASS,scheduler/EMA/门禁计数一致;
  - **DeepSpeed ZeRO-2 路径**(`accelerate launch --config_file deepspeed_zero2.yaml`,
    即服务器 train.sh 的同款启动方式)亦 PASS —— 证明 `sync_gradients` 在 DeepSpeed 下语义一致;
  - 报告存 run 目录 `54_verify_report.json`,详见 `ASSETS.md`。
- 调试中修掉的验证器缺陷(都会被带到服务器,故必须先进本地):① 覆盖参数必须带 `--` 前缀,
  否则 `normalize_dotlist_args` 静默丢弃(当时所有覆盖失效,会按 yaml 的 10 万步/每 5000 存跑,
  验证完全落空);② 单进程裸跑需自建进程组(dataloader 无条件 `dist.get_rank()`);
  ③ 断言边界:只有内部步带完整累积窗口,step 0 不计入门禁期望。

### 5.5 初始化一致性(快照法)
1. **快照来源与时机**:取自**完成 DINOv3/GroundingDINO 预加载之后的 C1 step-0 模型**
   (避免取到预加载前的随机 interaction——那样两组虽一致却共同偏离官方初始化);
   随后用于初始化 B。
2. **范围**:视觉投影、视图/位置嵌入、interaction、动作头;文本投影"沿用"方案下入快照,
   "校准"或"重初始化"方案列为明示例外(§5.2)。
3. **视觉骨干初始权重也校验一致**(两组同源 DINOv3 加载,哈希比对,即使不重复存入快照)。
4. C1/B 分别加载同一快照 → 哈希校验;`build_framework` 仍提前播种;结果记 `ASSETS.md`。

### 5.6 评测协议定义
- **训练 seed = 42;评测 seed = 0**(`eval_task.sh:28` 默认;两臂显式传同一值并记录);
  两者不得混称。
- 固定并记录:任务场景 seed、指令 seed、`instruction_type=unseen`、`action_ensemble=true`、
  `action_ensemble_horizon=50`、`adaptive_ensemble_alpha=0.1`、`normalization_mode=min_max`、
  `binary_threshold=0.49`、夹爪二次二值化(≥0.5)、评测精度组合 → 全进**结果指纹**(§6)。
- C1/B 试验条件逐项一致;**"同 trials 数量"不是配对试验**,统计口径沿用 §C0-100。

### 5.7 方案 A 接入提纲(v7 新增;A 若推进到 55k 正式,须先另拟完整方案并经确认)

**接入点(已核,2026-09-11)**:`third_party/starvla_runtime/starVLA/model/framework/
VLM4A/TurboVLA.py` 的 `_core_config` 把 framework 配置硬编码进
`TextEncoderConfig / VisionEncoderConfig / InteractionConfig` 等 → 补丁:framework 配置加
`text.encoder_type`、`vision.encoder_type`、`fusion.type` 三字段(默认 = 官方值,官方路径
零改动),构建前按类型分支;训练/评测同 wrapper,自动继承。

**四个移植单元**(源码 = `spike-turbovla` + `v2_code_bundle_20260906`):

| # | 单元 | 来源 | 关键适配点 |
|---|---|---|---|
| P1 | 文本 sootspike(含 SpikeLinearProjection) | 复用 §5.1–§5.2 成果 | 与 B 完全同一段代码,不重复实现 |
| P2 | 视觉 SDT-V3(`V3_19.0M_1x4.pth` 已在) | v2_code_bundle | **3 视图**:LIBERO 版按 2 视图训练,3 视图位置嵌入需扩容/重初始化并**声明为干预**;输入分辨率对齐 RoboTwin 相机 |
| P3 | 融合 Spike2Max(SpikeFFNResidual 带 LIF) | spike-turbovla `spiking.py` | spikingjelly LIF CUDA kernel 在新 env 冒烟;GroundingDINO 预加载不适用(同 LIBERO,全新 LayerScale,声明) |
| P4 | 动作头 14-D | 官方已有 | 官方 RoboTwin 头本就是 14-D(12 关节 min-max + 2 夹爪 binary),沿用;horizon 50 |

**验证门(P0A 退出标准)**:逐模块结构加载零缺漏 + 参数哈希断言(§5.2 同款)→ 真卡前向
单步 → 固定输入批数值检查(**§5.3 矩阵扩展覆盖视觉特征与融合输出**,不止文本)→ 1k 短训
(= A-S)。

**待拍板(不阻塞移植)**:
1. A 正式化初始化:**从头**(对齐 C1/B,默认;**A-S 已拍板 = 从头**,
   `clean50_a.yaml` `load_pretrained: false`,2026-09-12)vs 骨干热启动(载 v3 LIBERO 的
   SDT-V3/sootspike 权重——interaction/融合/动作头形状不匹配仍全新,叙事变,须声明);
2. A 精度组合:§5.3 矩阵扩展到三组件后定;
3. A 与 C1 的快照共享范围(§5.5)——仅当 A 推进到 55k 正式才需要。

### 5.8 C1-vs-B 判定规则(预注册,v7 新增;首次评测开始前不得更改)

- 统计口径沿用 §C0-100:报告**实际成功数** + 95% 正态近似区间;两臂各 5000 局,差值参考
  CI ≈ ±1.9pp(p≈0.6 双侧;任务内试验非独立,仅作量级参考)。**单 seed 单差值,混含
  实现/环境/协议差异,所有结论只在该 seed 下成立。**
- **成立** = B − C1 ≥ −2pp(含噪声内):文本 spike 化在双臂基准上不劣于 ANN BERT;
- **不成立** = B − C1 ≤ −4pp(≈2 倍 CI,触发 §3-B 回退顺序:数值行为复查 → 短解冻微调
  (声明干预)→ 回退 SpikingLM);
- **中间带(−4pp < B − C1 < −2pp)** = 不定:是否加测由用户拍板,禁止事后追加倍数直到
  出"想要"的符号;
- 判定用**与 C1 相同权重口径**(EMA-55k,对齐官方 §1.1);raw 列可另报,不参与判定。

## 6. 结果解析与硬校验

- 输入:launcher LOG_DIR 逐任务日志;
- episode 数从**实际完成记录**逐条计数,再与 `ROBOTWIN_TEST_NUM` 比对(配置值不作分母);
- 硬校验:任务数 = 50、每任务实际完成数 = ROBOTWIN_TEST_NUM、分子分母一致,任一不满足即拒绝出 summary;
- 产出:`summary.json` + 逐任务 json + **协议指纹 sidecar**(§5.6 全部字段)至 `v4_robotwin/results/<run_id>/`。

## 7. 硬件与墙钟

- 全局 192 单卡可达(前提 §5.4 修正版 + §4-P0T 统一档位);55k 步 ≈ 2–4 天/次,串行 4–8 天;借 4 卡各 ≈1 天。
- 评测预算:**C0-official 5000 + C1 5000 + B 5000 = 15000;C0-common(若需)+5000 = 上限 20000**;20-trial 筛查另计。同卡绑定,按卡分片;估时以 C0-20 实测回填。

**总预算表(v7 新增;全部为估算,以实测回填,拍 §10-2 算力前先看这张)**:

| 项 | 规模 | 卡时估算(1×H100 折算) |
|---|---|---|
| A-S 冒烟 | 1k 步 + 可选 20-trial | ≈0.5 天 |
| C0-20 + C0-100(official) | 1 000 + 5 000 局 | 待 C0-20 实测回填 |
| C1 55k | 全局 192 | 2–4 天 |
| B 55k | 同上 | 2–4 天 |
| 统一评测 C1/B | 2 × 5 000 局 | 待实测回填 |
| **小计(不含 A 正式)** | | **≈5–9 天训练 + 评测** |
| A-55k(若拍板) | 三 spike 组件,较 C1/B 上浮 | ≈3–6 天 + 评测 5 000 局 |

借 4 卡可把 C1+B 从 4–8 天压到各 ≈1 天;A-S 只需 1 卡半天,可与任何阶段穿插。

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| 精度参考错配(官方默认当数值基准会自证为零) | §5.3 两级参考:高精度参考 + 自身训练配置参考 |
| 跨精度 rollout 动作误差混入环境反馈 | §5.3 动作误差只在固定输入批;rollout 只比成功率/轨迹 |
| 夹爪两次二值化(0.49 → ensemble → 0.5) | §1.3/§5.3 双口径指标 |
| micro-batch 档位不一致引入数值差 | §4-P0T 两臂统一档位(文本按批内最长 padding) |
| 快照取自预加载前(共同偏离官方初始化) | §5.5 快照取自预加载后的 C1 step-0 |
| 精度三层误解 | §1.2 三层定义;`dinov3_precision` 显式纳入(§5.3) |
| 训练循环计数缺陷 | §5.4 修正 + 按实际参数更新次数验证 |
| 续训不完整恢复 | §5.4 二选一拍板,未拍板按"不得中断" |
| C1/B 共享模块初始化漂移 | §5.5 快照法 + 视觉骨干哈希校验 |
| `load_bert` 静默覆盖 sootspike 权重 | §5.2 显式关闭 + 哈希断言 |
| 投影校准/重初始化混入未声明干预 | §5.2 干预声明 + 禁止续训冒充 |
| 评测协议隐式默认漂移 | §5.6 显式定义 + 指纹 sidecar |
| 官方无结果聚合 | §6 解析器 + 实际完成记录比对 |
| HF 下载不通/慢 | `HF_ENDPOINT=https://hf-mirror.com`;备选 ModelScope/手动 |
| SAPIEN 渲染问题 | P0E 冒烟,不拖后 |
| 单卡训练过慢 | P0T probe 定档;两臂必须同步数 |
| spike 组件双臂失效 | C1-vs-B 单变量设计;**A-S 冒烟前置去风险**(§4-P0A/§5.7) |
| A 移植形状适配失败(3 视图位置嵌入 / 14-D 头) | P0A 逐单元验证门;P2 位置嵌入扩容**声明为干预** |
| spikingjelly LIF 在新 env 不可用(无 CUDA kernel) | P0A kernel 冒烟前置;不可用即冻结 P3 排期并上报,不带病开训 |
| RoboTwin LeRobot 数据管道与 spike 模块衔接(分辨率/文本长度/padding) | P0A 固定输入批数值检查覆盖(§5.3 扩展到视觉/融合输出) |
| **数据目录缺 `Clean/` 层**(HF 仓库平铺,注册表按 `Clean/<task>` 解析) | ✅ 2026-09-12 本地定位并修复:软链 `robotwin_data/RoboTwin/Clean`;记入 `ASSETS.md` |
| **`torch.load` 读不了 safetensors;torch≥2.6 默认 `weights_only=True` 拒绝含非张量对象的 ckpt** | ✅ 2026-09-12 已加固:`share_tools.load_checkpoint_file`(safetensors + `weights_only=False`),wrapper/base_framework/trainer_tools 三处统一。**注:发布版 GroundingDINO ckpt 实测为纯张量字典、默认设置可读**,故这是加固而非阻塞(训练脚本产出的含 `args` 变体、以及官方 55k safetensors 才需要它) |
| **官方 init ckpt 的普通 BERT 张量会半覆盖 sootspike**(200 张里 149 张形状匹配) | ✅ 2026-09-12 本地以真实 ckpt 实测:`load_bert=true` → 149/211 脉冲张量被覆盖(max\|Δ\| 2.05);`false` → 211 张逐位未动。**§5.2 的 `load_bert: false` 是必要防护** |
| **覆盖参数无 `--` 前缀被静默丢弃**(`normalize_dotlist_args`) | ✅ 2026-09-12 本地定位;手册 `ARMS_CN.md` §2/§4 显式警示 |

## 9. 命名

- 实验系列短名:**v4-robotwin**
- 全架构名 `{基准}_{视觉}_{文本}_{融合}_{日期}`:
  - **B**:`robotwin_dinov3l_sootspike_annfusion_<启动日>`
  - **A**:`robotwin_sdtv3_sootspike_snnfusion_<启动日>`(三组件 spike/混合 SNN-ANN;文本投影与融合 FFN 按我方实现本身即脉冲模块)
  - C1 记 `ann_retrain_seed42`;C0-official / C0-common 为官方模型复评,不占架构名;
    **A-S 冒烟记 `asmoke_<日>`,不占架构名、不进对照表。**

## 10. 决策点状态(2026-09-12 用户拍板"按推荐的来";仅 §10-2 算力仍开放)

1. **主线确认**:✅ **v7 三段式确认**(A 冒烟前置;C1∥B 为结论主体;A 正式视 B 结论)。
2. **算力**:⏳ **仍待拍板(唯一开放项)**——A-S 不受影响;C1+B 串行 4–8 天(1 卡)vs
   借 4 卡各 ≈1 天;用户排卡时告知卡数即可。
3. **精度决策流**:✅ 按计划默认流(数值检查 A 高精度参考 → 部署检查自身配置参考 →
   定统一精度;官方默认仅作候选与 C0 口径;必要时另跑 C0-common)。
4. **续训策略**:✅ **"55k 不得中断"**(中断作废重跑)。
5. **文本投影**:✅ 沿用 → 校准 → 重初始化(校准/重初始化均须声明为干预)。
6. **评测预算**:✅ 15000(无 C0-common)~ 20000(含)+ A-S 可选 20 局;紧张时 C1/B 降
   50 trials 并记录口径。
7. **下载**:✅ RoboTwin 仓库 @`96c1fea`、官方 55k EMA ckpt、**ViT-L**、
   **RoboTwin-Clean 数据(50 任务 × 50 eps = 2500,3.9G,结构抽查通过)**均已落地;
   GroundingDINO 下载中。P0T 与 env 搭建并行 = ✅(数据已在共享卷)。
8. **mask 版本 harness 变量**(2026-09-12 新增拍板):✅ **C1 = legacy(官方原味),
   B/A = corrected**;共变已声明(见 §4-C1∥B),默认不补 C1-corrected。
