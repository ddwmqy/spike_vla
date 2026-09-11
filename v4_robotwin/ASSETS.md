# v4-robotwin 资产与版本钉(P0E 进度,2026-09-11)

> 配套:`PLAN_CN.md` v7。本文件按计划 §4-P0E 第 3 条建立;每项落地即回填,sha256 齐全后本文件即为"钉版本"凭证。

## 已就位

| 资产 | 位置 | 版本 / 指纹 | 备注 |
|---|---|---|---|
| TurboVLA 官方参考代码(含 RoboTwin 训练/评测栈) | `/data/260010028/dwh_vla/TurboVLA_official` | commit **b29ab14**(2026-09-02) | wrapper:`third_party/starvla_runtime/starVLA/model/framework/VLM4A/TurboVLA.py`(`_core_config` 硬编码,§5.1/§5.7 接入点) |
| RoboTwin 2.0 模拟器仓库 | `/data/260010028/dwh_vla/RoboTwin` | **robotwin-Platform/RoboTwin @ `96c1fea`**(2026-09-05,752 文件) | ⚠️ TianxingChen/RoboTwin 是 v1 落地页(主分支仅 README),勿用 |
| 官方 RoboTwin 55k EMA ckpt | `/data/260010028/dwh_vla/v4_assets/TurboVLA_robotwin/checkpoints/robotwin/steps_55000_ema_model.safetensors` | 828 MB;sha256 `d0183df6bafd44507b6c797da5c5ab080ef8446cde4a8127d7280546d9f7c034` | 来源 HF `H-EmbodVis/TurboVLA`(经 hf-mirror);同层 `config.yaml` + `dataset_statistics.json` + `config.json`,祖先目录结构满足 `share_tools.py:83` |
| v3 spike 组件源码(移植源 P1/P2/P3) | `/data/260010028/dwh_vla/spike-turbovla` + `v2_code_bundle_20260906/code` | 独立 git 仓库(remote wxqnl/spike-turbovla) | sootspike / SDT-V3 / Spike2Max(turbovla.py:118、spiking.py:735) |
| SDT-V3 视觉权重(P2) | `v2_code_bundle_20260906/resources/pretrained/V3_19.0M_1x4.pth` | 19.0M | LIBERO 版按 2 视图训练;3 视图位置嵌入适配见 §5.7 |
| sootspike 文本权重(P1) | `v2_code_bundle_20260906/resources/pretrained/SmoothSpike/smoothspike-bert-base-fused/model.safetensors` | fused 版 | sha256 待补(§4-P0E 第 3 条) |
| bert-base-uncased | `v2_code_bundle_20260906/resources/pretrained/bert-base-uncased/` | — | C1 / 官方路径用 |
| DINOv3 ViT-B(LIBERO 用) | `v2_code_bundle_20260906/resources/pretrained/dinov3-vitb16-pretrain-lvd1689m/` | — | **RoboTwin 不用**(官方 README §130:RoboTwin = ViT-L) |

## 待办

- [ ] DINOv3 **ViT-L** 权重(hf-mirror 直下;`facebook/dinov3-vitl16-pretrain-lvd1689m`)
- [ ] StarVLA/RoboTwin-Clean 50 任务数据(`scripts/robotwin/prepare_data.sh`,`HF_ENDPOINT=https://hf-mirror.com`;**体量实测回填**)
- [ ] GroundingDINO swint ogc(C1/B 初始化用;**A 不需要**)
- [ ] `turbovla-robotwin` env(python 3.10,`pip install -e ".[robotwin]"` + **FlashAttention-2 另装**)——在算力服务器建
- [ ] SmoothSpike 源码/权重 sha256 补齐
- [ ] LIBERO checkout 8f1084e 已在(仅 v3 线复现用,与本基准无关)

## 网络事实(dev pod,2026-09-11 实测)

- hf-mirror.com ✅ 200;github.com ✅ 200;huggingface.co ❌ 不通(一律走 hf-mirror)
- 下载在 dev pod 执行、写 /data 共享卷,算力服务器直接可见
