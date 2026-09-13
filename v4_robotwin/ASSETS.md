# v4-robotwin 资产与版本钉(P0E/P0T 进度,2026-09-12)

> **算力(2026-09-12 定)**:算力服务器 **4 张卡** → 走官方配置路径
> (`NUM_PROCESSES=4 PER_DEVICE_BATCH_SIZE=48 GRADIENT_ACCUMULATION_STEPS=1`,全局 192),
> C1、B 各 ≈1 天、串行;评测按卡分片。

> 配套:`PLAN_CN.md` v7。本文件按计划 §4-P0E 第 3 条建立;每项落地即回填,sha256 齐全后本文件即为"钉版本"凭证。

## 已就位

| 资产 | 位置 | 版本 / 指纹 | 备注 |
|---|---|---|---|
| TurboVLA 官方参考代码(含 RoboTwin 训练/评测栈) | `/data/260010028/dwh_vla/TurboVLA_official` | commit **b29ab14**(2026-09-02) | wrapper:`third_party/starvla_runtime/starVLA/model/framework/VLM4A/TurboVLA.py`(`_core_config` 硬编码,§5.1/§5.7 接入点) |
| RoboTwin 2.0 模拟器仓库 | `/data/260010028/dwh_vla/RoboTwin` | **robotwin-Platform/RoboTwin @ `96c1fea`**(2026-09-05,752 文件) | ⚠️ TianxingChen/RoboTwin 是 v1 落地页(主分支仅 README),勿用 |
| 官方 RoboTwin 55k EMA ckpt | `/data/260010028/dwh_vla/v4_assets/TurboVLA_robotwin/checkpoints/robotwin/steps_55000_ema_model.safetensors` | 828 MB;sha256 `d0183df6bafd44507b6c797da5c5ab080ef8446cde4a8127d7280546d9f7c034` | 来源 HF `H-EmbodVis/TurboVLA`(经 hf-mirror);同层 `config.yaml` + `dataset_statistics.json` + `config.json`,祖先目录结构满足 `share_tools.py:83` |
| v3 spike 组件源码(移植源 P1/P2/P3) | `/data/260010028/dwh_vla/spike-turbovla` + `v2_code_bundle_20260906/code` | 独立 git 仓库(remote wxqnl/spike-turbovla) | sootspike / SDT-V3 / Spike2Max(turbovla.py:118、spiking.py:735) |
| SDT-V3 视觉权重(P2) | `v2_code_bundle_20260906/resources/pretrained/V3_19.0M_1x4.pth` | 19.0M;sha256 `72adec1cabce8f9aacb4f9f16deaf37ae28c22e72d815d19e6c6fea0ec9b05a7` | LIBERO 版按 2 视图训练;3 视图位置嵌入适配见 §5.7 |
| sootspike 文本权重(P1) | `v2_code_bundle_20260906/resources/pretrained/SmoothSpike/smoothspike-bert-base-fused/model.safetensors` | fused 版,**T=4**(门 #1 实测);sha256 `e41c28eac27b9dd52b08e876e1d77141200039462d62b2f6a497061433ee18ba` | A 臂 yaml 必须 `text.timesteps=4`,否则加载即拒 |
| v4 移植工作仓库(P0A) | `/data/260010028/dwh_vla/v4_code`(branch `v4-port`) | b29ab14 + 移植提交(`4b40282`);A 臂配置 `clean50_a.yaml`(文本 `attn_implementation: eager`) | 门 #0 官方路径**逐位一致**(672 参数,216,072,981,max\|Δ\|=0.0);门 #1 A 臂三组件 CPU 构建+前向通过([2,50,14] 全有限);门 #2 权重哈希断言通过(文本 211/216 逐位 + cls.* 5 项豁免、视觉 603/603 逐位);**门 #2b** yaml→wrapper 全链路构建 + `predict_action` (1,50,14) 通过(逮住并修掉 eager 配置 bug);`scripts/{parity_probe,gate1_spike_build,gate2_weight_hash,gate2b_wrapper_build,gate3_gpu_check}.py` 可复跑。**余项:门 #3 真卡前向 + 固定输入批数值检查(算力服务器)** |
| bert-base-uncased | `v2_code_bundle_20260906/resources/pretrained/bert-base-uncased/` | — | C1 / 官方路径用 |
| DINOv3 ViT-B(LIBERO 用) | `v2_code_bundle_20260906/resources/pretrained/dinov3-vitb16-pretrain-lvd1689m/` | — | **RoboTwin 不用**(官方 README §130:RoboTwin = ViT-L);其 224px 处理器目录兼作 A 臂 `SDTV3_PROCESSOR_PATH` |
| DINOv3 **ViT-L**(C1/B/官方 ckpt 用) | `/data/260010028/dwh_vla/v4_assets/dinov3-vitl16-pretrain-lvd1689m/` | model.safetensors 1.21 GB,sha256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`;415 tensors 自洽,dinov3_vit 1024/24L/patch16/224 | 来源 ModelScope 官方镜像 `facebook/dinov3-vitl16-pretrain-lvd1689m`(hf-mirror 对 gated 文件 403);config `135ecd23…`,preprocessor `960c41d1…` |
| RoboTwin-Clean 数据(P0T) | `/data/260010028/dwh_vla/v4_assets/robotwin_data/StarVLA_RoboTwin_Clean/` | **50 任务 × 50 eps = 2500 局,3.9 GB**;LeRobot 格式(parquet action/state 均 14-D、fps 15、视频配对抽查通过) | 来源 HF `StarVLA/RoboTwin-Clean`(hf-mirror);**仓库是平铺布局**,训练需经 `robotwin_data/RoboTwin/Clean` 软链;`ROBOTWIN_DATA_ROOT=/data/260010028/dwh_vla/v4_assets/robotwin_data/RoboTwin` |
| GroundingDINO swint_ogc(C1 初始化) | `/data/260010028/dwh_vla/v4_assets/groundingdino/groundingdino_swint_ogc.pth` | 694 MB;sha256 `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`;**纯 `{'model': …}` 940 张量(无 `args` Namespace)**,默认 `weights_only=True` 可读 | 来源 HF 镜像 `ShilongLiu/GroundingDINO`(hf-mirror,快;GitHub releases 只有 ~70 KB/s 已弃)。键结构:`bert.` 200 / `feat_map.` 2 / `transformer.encoder.text_layers.` 72 / `fusion_layers.` 108(**B 臂 §5.2 危害源 = 那 200 个普通 BERT 张量**)|

## 待办

- [x] ~~DINOv3 **ViT-L** 权重~~(✅ 2026-09-11,见上表)
- [x] ~~StarVLA/RoboTwin-Clean 50 任务数据~~(✅ 2026-09-12:50×50=2500 局 / 3.9 GB,结构抽查通过,见上表)
- [x] ~~GroundingDINO swint ogc(C1/B 初始化用)~~(✅ 2026-09-12:694 MB,sha256 见上表;A 臂不用)
- [ ] `turbovla-robotwin` env(python 3.10,`pip install -e ".[robotwin]"` + **FlashAttention-2 另装**)——在算力服务器建
- [x] ~~SmoothSpike 源码/权重 sha256 补齐~~(✅ 2026-09-12:5 个上游运行时文件已哈希,见下)

### §5.4 训练循环计数修正(v4_code,2026-09-12)

`third_party/starvla_runtime/starVLA/training/train_starvla.py` 两处修改(accum=1 时行为不变):
① `lr_scheduler.step()` 加 `if self.accelerator.sync_gradients:` 门控;② eval/save 门禁统一由
`is_update_step = self.accelerator.sync_gradients` 门控。

| 验证 | 工具 | 结果 |
|---|---|---|
| CPU 计数(accum=1 与 4) | `scripts/gate5_trainer_counting.py` | ✅ PASS:scheduler/EMA 步数 = completed_steps = 20;eval/save = 4/2 次;**变异检验**旧代码 accum=4 → scheduler 80、eval 16(测试有牙) |
| **GPU(本机 H100 MIG)真实数据+真实模型** | `scripts/robotwin/verify_54_counting.py` | ✅ PASS ×3:accum=1(5 步)、accum=4(32 micro→8 step)、**accum=4 + DeepSpeed ZeRO-2**(`accelerate launch --config_file deepspeed_zero2.yaml`,服务器同款启动);scheduler 步 = completed_steps、warmup 峰值在 warmup-1(记录口径,见脚本注释)、门禁触发数与落盘 ckpt 对账一致。报告存各 run 目录 `54_verify_report.json` |
| 待算力服务器 | 同上 | 可跑 1100 步跨 warmup 1000 的全尺寸复验(非阻塞;机制已在真实数据上验证) |

### A 臂冒烟(本地,dev pod H100 MIG;非计划 A-S,数字不作结论)

**流水线健康证据**:`clean50_a.yaml` 全链路(真实 clean50 数据 + 三 spike 组件)1 000 步实测:

| 项 | 值 |
|---|---|
| 规模 | 1 000 步,batch 4 × 1 进程(非官方全局 192;故**任何数字都不具比较含义**) |
| 墙钟 | 8 分 45 秒(≈0.5 s/it) |
| loss | 0.352(step 25) → 0.104(step 1000);前 1/4 均值 0.233 → 后 1/4 均值 0.117(**腰斩**) |
| 数值 | 40 个记录点**全部有限**,min 0.069 / max 0.352,**无 NaN** |
| 产物 | ckpt + EMA 各两次(step 500、1000),`summary.jsonl` 记录一致 |
| 日志 | `v4_code/results/Checkpoints/asmoke_log_20260912.txt`(run 目录 `asmoke_localdebug2_20260912/`) |

→ A-S 的退出标准(loss 形态正常、无 NaN、ckpt+EMA 落盘)在本地规模下**已满足**;服务器版
A-S 的价值收敛为"确认官方 4 卡 × batch 48 档位",不再是去风险必需。命名仍按
§9 保留 `asmoke_<日>` 给服务器正式冒烟。

### §5.3 数值检查 A 实测(本地,dev pod;固定输入批 = 2 个真实 clean50 样本,eval 模式,同种子)

工具:`v4_code/scripts/precision_matrix.py`(每格独立进程);报告
`v4_code/results/PrecisionMatrix/{b,c1}_20260912/report.json`。表内为相对高精度参考
(ref = matmul highest + interaction fp32 + vision fp32 + sdpa)的 **max|Δ| 占参考张量 RMS 百分比**。

| 格(interaction / vision / attn) | B 文本投影 | B 视觉 | B 动作max | B 关节MAE(raw) | C1 文本投影 | C1 视觉 | C1 动作max | C1 关节MAE(raw) |
|---|---|---|---|---|---|---|---|---|
| **官方默认**(bf16_autocast / bf16_autocast / sdpa) | **253%** | 61.9% | 0.0510 | 0.0607 | 5.7% | 71.8% | 0.0021 | 0.0028 |
| 官方默认 + **flash**(真实训练配置) | 253% | 55.5% | 0.0510 | 0.0608 | 5.7% | 60.0% | 0.0021 | 0.0025 |
| **fp32 / bf16_autocast / sdpa** | **0%** | 62.1% | **0.0024** | 0.0029 | 0% | 71.7% | 0.0023 | 0.0027 |
| fp32 / fp32 / sdpa,matmul=high(TF32) | 158% | 3.2% | 0.0454 | 0.0621 | 0.23% | 3.4% | 0.0004 | 0.0004 |
| 夹爪双口径(0.49 / 0.5 一致率) | **1.0000 全部格、两臂** | | | | | | | |

**发现**:
1. **精度敏感度高度不对称**:同口径下 spike 文本路径被扰动到 **253%RMS**,而 ANN(BERT)只有
   **5.7%** —— 即"统一精度"决策的风险几乎全部由 B 臂承担;机制上符合 spike 阈值/替代梯度
   对舍入的放大,但这是必须记录在案的 B 臂固有劣势。
2. **三层精度可分离**(新格 `fp32/bf16_autocast` 证明):文本误差 ← interaction autocast
   (关掉即 0%);视觉误差 ← vision 精度。**注意:interaction autocast 开启时 `vision.compute_precision`
   被外层 autocast 掩盖**(`ann_bf16_vis_fp32` 与 `vis_bf16` 逐位相同)——计划 §1.2/§5.3 把它
   描述为"独立开关",在**官方默认组合下不成立**,只有 interaction 关掉 autocast 时它才可观测。
3. **B 臂动作误差主要由文本路径主导**:官方组合 0.0510 → 改 `fp32/bf16_autocast` 后 0.0024
   (**21× 更小**,而视觉仍在 bf16)。
4. **注意力后端影响与精度同量级**(flash vs sdpa:视觉 55.5% vs 61.9%)→ 阈值必须按**实际训练
   用的后端**定,不能跨后端搬运。
5. **flash-attn 不支持 fp32** → "全模块关 autocast 的 FP32 参考"在官方后端下**不可达**;
   故矩阵在 sdpa 上跑 + 单列 flash 量化后端影响。
6. `float32_matmul_precision` 环境默认 = **highest**(官方路径无 TF32 污染);若被设为
   high/medium,**即使全 fp32**,B 文本投影也会偏 158% → **不得改动该全局设置**。
7. **夹爪二值决策对精度完全鲁棒**(所有格、两臂 100% 一致)。

**精度组合决议(2026-09-12 用户拍板:方案 A)**:**C1/B 两臂均维持官方默认三层组合**
(参数 FP32 + `interaction.compute_precision=bf16_autocast` + `vision.compute_precision=bf16_autocast`
+ `float32_matmul_precision` 保持环境默认 `highest`;**不得改动该全局设置**)。理由:C1 的定位
是官方配方复现锚点,改精度会同时破坏锚点定位与单变量设计;B 的精度脆弱性(253%RMS)作为
**结果**报告而非抹平;若 B 结论不佳,§5.8 回退链已覆盖"精度是元凶"分支。**已声明的代价**:
B 在高敏感数值环境下训练(A/B 方案取舍见 §5.3 进度段)。

**验收阈值(暂定,2k 短训后复核)**:特征类 max|Δ| ≤10%RMS 记"等价"、10–100% 记"可观测偏移
(须声明)"、>100% 记"劣化";动作类固定输入批关节 MAE(raw 单位)≤0.01、夹爪两口径一致率 =100%。
按此口径官方默认组合对 B 的文本路径判"劣化"(253%)——**已接受并记录**。**注:以上均在随机动作头
上测得,只保证格间相对比较有效,绝对口径待 2k 部署检查复核。**

### A 臂脉冲注意力:softmax 变种(2026-09-12 用户指定,**声明干预**)

按用户指示,脉冲交叉注意力不再限定 SDSA 重合计数形式,改用**带 softmax 的变种**(即用户
此前提的 "shiftmax" 思路)。实现为**可切换模式**而非替换:

| 项 | 内容 |
|---|---|
| 开关 | `interaction.cross_attention_softmax: "none" \| "softmax"`;**代码默认 = `none`**(移植来的 SDSA 行为不变),A 臂配置选 `softmax` |
| softmax 语义 | 重合计数 × 1/√head_dim → **减最大值**(max-shift/"shiftmax")→ 仅在**有效源**上归一化(padding 不占注意力质量;全 padding 行输出精确 0 而非 NaN) |
| 保持项 | Q/K/V 仍为**二值脉冲**;层归一化/膜电位/残差/时域读出不变 |
| context 阈值 | 权重成为分布后计数缩放不再适用,改用普通膜阈值;初始化 **0.375**,由实测选定(与 `none` 模式同发放量级:context_v 0.036/0.034/0.060、out_v RMS 0.107,对 0.033/0.036/0.057、0.105);仍可学习 |
| 验证 | `v4_code/scripts/check_softmax_attention.py`(归一化/padding 排除/全 padding 行/单源/max-shift 不变性/前反向)+ 门 #2b/#2 回归 + A 臂冒烟重跑 |

**影响域**:仅 A 臂(B/C1 用 ANN 交互,不受影响 → C1-vs-B 对照不受影响)。属**声明干预**,
A 的既往前验证(gates、旧冒烟)已随此变更重跑。v3 线(独立仓库,已验证的 LIBERO 结果)未改动。

### 本地调试发现与修复(2026-09-12,dev pod;详见 `v4_code/experiments/robotwin/ARMS_CN.md`)

| 发现 | 影响 | 处置 |
|---|---|---|
| HF `StarVLA/RoboTwin-Clean` 是**平铺**任务目录,而注册表按 `Clean/<task_name>` 解析 | 训练直接找不到数据 | 已建软链 `v4_assets/robotwin_data/RoboTwin/Clean -> ../StarVLA_RoboTwin_Clean`;`ROBOTWIN_DATA_ROOT` 指向 `.../robotwin_data/RoboTwin` |
| `torch.load` 读不了 safetensors;且 torch≥2.6 默认 `weights_only=True` 会拒绝含非张量对象的 ckpt | 官方 55k ckpt(safetensors)在 wrapper 里无法作为初始化源;**发布版** GroundingDINO ckpt 实测是纯张量字典、默认设置可读(训练脚本产出的变体含 `args` 则会失败),属加固而非阻塞 | 已修 `share_tools.load_checkpoint_file`(safetensors + `weights_only=False`),wrapper/base_framework/trainer_tools 三处统一;本地以合成 init(含 Namespace 的 `.pth`)与真实 GroundingDINO 文件分别验证通过 |
| `normalize_dotlist_args` **静默丢弃无 `--` 前缀的覆盖参数** | 覆盖失效(如 `MAX_TRAIN_STEPS` 类覆盖会退回 yaml 的 10 万步) | 手册 `ARMS_CN.md` 显式警示;验证脚本已改为 `--` 形式 |
| 单进程裸跑训练脚本时 dataloader 的 `dist.get_rank()` 报进程组未初始化 | 只能经 `accelerate launch` 启动 | 验证器自带单进程进程组初始化;正式训练走 `train.sh` |
| pod 的 v3 conda env 内 `wandb` import 失败(protobuf 版本错配) | 本地调试受阻(不影响服务器新 env) | 本地用桩 `v4_assets/debug_stubs/wandb.py`(经 `PYTHONPATH` 前置,不入库);不动 v3 环境 |
| cv2 缺 `libGL.so.1`(pod 容器) | 本地调试受阻 | `apt-get install -y libgl1`(容器重启需重装;服务器应装 headless 版 opencv) |

**§5.2 防护验证**(`v4_code/scripts/check_b_init_hash.py`),两种 init 源都跑过:
- 合成 init(393 张量,含 `args` Namespace):`load_bert=false` → 载入 182,211 张脉冲文本张量逐位未动(PASS);`load_bert=true` → 393,211 张**全被覆盖**。
- **真实 GroundingDINO ckpt**:`load_bert=false` → 载入 **182**(2 投影 + 72 文本层 + 108 融合层),211 张逐位未动(PASS);`load_bert=true` → 载入 **331**,其中 **149 张脉冲文本张量被覆盖,max|Δ| 达 2.05**(半覆盖式静默污染)。
→ 结论:`load_bert: false` 是 B 臂的**必要**防护,不是形式条款。

### third_party 上游源码指纹(patch 后;`scripts/setup_third_party.py` 重建产物)

| 文件 | sha256 |
|---|---|
| `third_party/SmoothSpike/spikingbert_rot_inf.py` | `61bd30f20aa51cce0145cfc73479f7901e44dc20532d3f1ec6c440cd2ef3c2ba` |
| `third_party/SmoothSpike/hadamard_utils.py` | `8dfa8985822d03c113f05fbcbe5cb03f29f4e19351f2d27e8cf940c4e1e2bf00` |
| `third_party/SmoothSpike/utils.py` | `c30627f3ce93182ea07d8ee2a0795fcb224311a0f3456d8263d16578226e0287` |
| `third_party/SmoothSpike/convertor.py` | `15d94e0d4a7a964c096138ae71b58f10f1f3a8be267aa989e96ab38a1fef9ff4` |
| `third_party/Spike-Driven-Transformer-V3/.../Model_Base/models.py` | `6b60714b62a6f9ab38a84c53cbc624307668a2e84e80c334552d4359647a0224` |

(A/B 臂训练前应在算力服务器复算比对——沿袭 `spike-turbovla` 的许可处理:上游文件不入库,靠此表钉版本。)
- [ ] LIBERO checkout 8f1084e 已在(仅 v3 线复现用,与本基准无关)

## 网络事实(dev pod,2026-09-11 实测)

- hf-mirror.com ✅ 200;github.com ✅ 200;huggingface.co ❌ 不通(一律走 hf-mirror)
- modelscope.cn ✅ 200(gated 仓库的官方镜像源,hf-mirror 403 时备用)
- 下载在 dev pod 执行、写 /data 共享卷,算力服务器直接可见
