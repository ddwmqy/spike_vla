# v4-robotwin 资产与版本钉(P0E/P0T 进度,2026-09-12)

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
| RoboTwin-Clean 数据(P0T) | `/data/260010028/dwh_vla/v4_assets/robotwin_data/StarVLA_RoboTwin_Clean/` | **50 任务 × 50 eps = 2500 局,3.9 GB**;LeRobot 格式(parquet action/state 均 14-D、fps 15、视频配对抽查通过) | 来源 HF `StarVLA/RoboTwin-Clean`(hf-mirror);训练 yaml `ROBOTWIN_DATA_ROOT=/data/260010028/dwh_vla/v4_assets/robotwin_data` |

## 待办

- [x] ~~DINOv3 **ViT-L** 权重~~(✅ 2026-09-11,见上表)
- [x] ~~StarVLA/RoboTwin-Clean 50 任务数据~~(✅ 2026-09-12:50×50=2500 局 / 3.9 GB,结构抽查通过,见上表)
- [ ] GroundingDINO swint ogc(C1/B 初始化用;**A 不需要**)
- [ ] `turbovla-robotwin` env(python 3.10,`pip install -e ".[robotwin]"` + **FlashAttention-2 另装**)——在算力服务器建
- [x] ~~SmoothSpike 源码/权重 sha256 补齐~~(✅ 2026-09-12:5 个上游运行时文件已哈希,见下)

### §5.4 训练循环计数修正(v4_code,2026-09-12)

`third_party/starvla_runtime/starVLA/training/train_starvla.py` 两处修改(accum=1 时行为不变):
① `lr_scheduler.step()` 加 `if self.accelerator.sync_gradients:` 门控;② eval/save 门禁统一由
`is_update_step = self.accelerator.sync_gradients` 门控。

| 验证 | 工具 | 结果 |
|---|---|---|
| CPU 计数(accum=1 与 4) | `scripts/gate5_trainer_counting.py` | ✅ PASS:scheduler/EMA 步数 = completed_steps = 20;eval/save = 4/2 次;**变异检验**旧代码 accum=4 → scheduler 80、eval 16(测试有牙) |
| GPU 短跑(官方配方,跨 warmup 1000 边界) | `scripts/robotwin/verify_54_counting.py` | ⏳ 待算力服务器执行;报告落 run 目录 `54_verify_report.json` |

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
