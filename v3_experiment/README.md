# v3-sootspike 实验归档(LIBERO,2026-09-10 快照)

> **本目录是快照副本,不是 canonical 位置**。原件路径见下方"文件地图";更新请改原件,
> 需要时重新复制。整理日期:2026-09-10(commit 5b0d96f 之后)。

## 实验是什么

**v3** = v2 换文本骨干:SpikingLM → **sootspike**(CayleyZ/SmoothSpike 的 fused 版)。
全架构名 `libero_sdtv3_sootspike_snnfusion_20260907`(SDT-V3 SNN 视觉 + sootspike 文本 +
Spike2Max SNN 融合 + ANN ACT 头),自训 80k 步(batch 56 单卡 H100,13h08m)。

## 核心结果(同协议:fp32 · seed 7 · 50 trials · chunk 12 · open-loop 12 · LIBERO 8f1084e)

| 套件 | v3 raw(主表) | v3 EMA(D1 补充) | 官方发布包复评(=raw 口径) |
|---|---|---|---|
| libero_spatial | 78.2% | **95.6%** | 93.6% |
| libero_object | **99.2%** | 98.0% | 98.8% |
| libero_goal | 80.8% | **95.2%** | 94.4% |
| libero_10 | 67.6% | **80.0%** | 84.0% |
| **合计** | **81.5%**(1629/2000) | **92.2%**(1844/2000) | 92.7%(1854/2000) |

关键读法:

- EMA 增益 **+10.8pp**,集中在 raw 落后的套件;raw 已饱和的 object 无增益(-1.2,噪声内);
- goal task5:raw 2% → EMA 98%(权重选择高度敏感);libero_10-t9 EMA 仍 24%(唯一残留崩点);
- **锚点数字 92.7% 是我方复评值**(官方发布权重 × 我方协议),官方论文唯一公开数字是 97.7%
  (其自有协议,差 5pp = 协议差异:per-suite 步数上限 220/280/300/520、no_noops 统计);
- 上游时间线定案:发布(07-30/31)= raw 口径;官方 09-02 自己转向 EMA-only 评测,
  同提交把配方全局 batch 256→128(早于我们转向 EMA 五周)。

## 文件地图(快照 → 原件)

| 快照 | 原件(canonical) |
|---|---|
| `docs/V3_SOOTSPIKE_CN.md` | `v2_code_bundle_20260906/package/V3_SOOTSPIKE_CN.md`(全景入口,先读它) |
| `docs/V3_EVAL_VS_OFFICIAL_ANCHOR_CN.md` | 同目录(逐任务对比 + 对齐剔除 + EMA 表) |
| `docs/V3_OPTIMIZATION_PLAN_CN.md` | 同目录(优化计划:A1/A2/B、解释边界、候选机制假设) |
| `docs/V2_*.md`、`docs/CLUSTER_SETUP_CN.md` | 同目录(v2 上下文/集群操作) |
| `scripts/eval_v3.sh` | `/data/260010028/dwh_vla/eval_v3.sh`(评测统一入口) |
| `scripts/launch_v3.sh` | `/data/260010028/dwh_vla/launch_v3.sh`(平台训练启动器) |
| `scripts/run_eval.sh` / `run_eval_official.sh` / `run_batch_probe.sh` | `v2_code_bundle_20260906/package/` |
| `scripts/run_train_v2.sh` | 同上(v3 复用:`TEXT_ENCODER_TYPE=sootspike`) |
| `results/summary_raw.json` | `package/output/eval/v3_sootspike_step_80000_raw_clean_8f1084e/` |
| `results/summary_ema.json` | `package/output/eval/v3_sootspike_step_80000_ema_clean_8f1084e/`(D1) |
| `results/anchor_official/` | `/data/260010028/dwh_vla/anchor_export_20260909/`(官方逐任务 json,amax 导出) |

**未入快照**(体积/敏感原因):checkpoint 权重、评测视频、训练/评测日志、conda 环境、
上游源码仓库。逐任务 EMA/官方 json 的全部数字已写进 `docs/V3_EVAL_VS_OFFICIAL_ANCHOR_CN.md`。

## 复现

```bash
# 评测(算力服务器;/data 共享卷已含环境与权重)
WEIGHT_SOURCE=raw bash scripts/eval_v3.sh                             # 主表口径
WEIGHT_SOURCE=ema EVAL_SUITES=libero_spatial,libero_goal bash scripts/eval_v3.sh   # D1 用法
# 训练
bash scripts/launch_v3.sh                                             # 单 H100,batch 56
```

## 代码与环境

- **模型/训练/评测源码**:`/data/260010028/dwh_vla/spike-turbovla`(独立 git 仓库,
  remote = github.com/wxqnl/spike-turbovla;上游 = H-EmbodVis/TurboVLA,pi05.py EMA
  注入与上游逐字一致)。
- **环境**:conda env `turbovla`(`/data/260010028/dwh_vla/miniconda3/envs/turbovla`,
  editable 安装 spike-turbovla;robosuite EGL 补丁已打在 site-packages)。
- **数据**:LIBERO checkout 固定 commit 8f1084e(`/data/260010028/dwh_vla/LIBERO_eval`)。

## 状态与下一步(2026-09-10)

- 待决策:**A1 重训 batch**(128 = 官方现行配方 vs 256 = 原计划)与是否并入
  text_projection 校准(官方对 spike 文本有 `calibrate_text_projection.py`,v3 训练跳过了);
  **EMA 部署口径**(官方 09-02 已转向 EMA-only,预注册 EMA 口径有官方先例)。
- v4-robotwin 为论文主线(`v4_robotwin/PLAN_CN.md` v6,§10 待确认),LIBERO 线按
  "A1 一次重训后收手"标准执行。
