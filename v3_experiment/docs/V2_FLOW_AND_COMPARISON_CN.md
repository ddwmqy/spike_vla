# v2 实验流程与官方模型对比说明

> 2026-09-05 定稿。配套文档:`V2_PLAN_CN.md`(完整计划 + §6.1 超参对齐审计全文)。
> 行号引用均以本包当前代码为准,已逐一核实。
> 2026-09-07 后续实验:文本骨干换 sootspike(SmoothSpike BERT),架构与启动见 `V3_SOOTSPIKE_CN.md`。

## 0. 一句话结论

**v2 评测结果与官方锚点可直接对比**(2026-09-05 三审修复后):两者在同一台机器、同一份 LIBERO
(commit 8f1084e,脚本硬校验)、同 seed 7、同 4×10×50=2000 episodes、同 chunk 12 / open-loop 12、
同 FP32、同一份 `evaluate.py` 代码下测出;对比对象是我们实测的官方锚点,不是 README 论文数字。
前提条件(② 步 2026-09-05 已完成):发布物结构 = 单 ckpt
(`checkpoints/libero/turbovla_libero.pth`,672 张量严格加载零缺漏,216.1M);TEXT_MODE 已仲裁 =
**frozen**(官方 ckpt 记录 text.frozen=True,与 trainer 默认一致,README "online" 表述与实物不符);
权重口径已定 = **raw↔raw**(官方发布无 EMA,锚点 = raw,v2 主对比用 WEIGHT_SOURCE=raw)。

**比较边界(不可直接比)**:官方 README/论文数字(97.7%、BF16、官方发布环境);历史 93.95%
(旧环境旧管线);历史 55.95%(**旧脚本用 osmesa 渲染,当前管线是 EGL,协议不同**——如需把它
当参考,须在当前 EGL 管线重跑)。

## 1. 实验流程

包根 = `/home/dwh/work/vla/libero_vla/libero_sdtv3_bert_snnfusion_20260831/`(下称 `$PKG`)。

| 步 | 内容 | 脚本/代码 | 产物 |
|---|---|---|---|
| ① | 官方发布获取【已完成 2026-09-05】 | 本机 hf-mirror 直下:`HF_ENDPOINT=https://hf-mirror.com` + `snapshot_download('H-EmbodVis/TurboVLA')`(1.7G) | `resources/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth` |
| ② | 预检 Part B + 转换器自述【已完成,结论见 §0】 | `preflight_v2.py $CKPT` ✅ teacher 216.1M 严格加载;`convert_official_ckpt.py --src $CKPT --dst /tmp/probe.pth` ✅(转换器已补本地路径覆盖,与评测端 policy 同构) | TEXT_MODE=frozen 定案;raw↔raw 定案 |
| ③ | 官方锚点评测(4 套件并行)【进行中,GPU 0,1,3,5】 | `OFFICIAL_CKPT=$CKPT EVAL_GPU_IDS=0,1,3,5 ./run_eval_official.sh`(单 ckpt 模式;转换导出 raw=model_state_dict) | `output/eval/official_clean_8f1084e/results/summary.json` |
| ④a | **强制 smoke**(2k 步单卡):冻结 SpikingLM + 全新 SNN 融合 + vision lr 5e-5 首次组合;warmup 超 max_steps 会自动钳制 | `MAX_STEPS=2000 WARMUP_STEPS=250 TEXT_MODE=frozen GPU_IDS=<1卡> BATCH_SIZE=16 TARGET_GLOBAL_BATCH=16 CHECKPOINT_PREFIX=turbovla_v2_smoke OFFICIAL_CKPT=$CKPT ./run_train_v2.sh`(TEXT_MODE 与 ckpt 记录不符会被门禁拦截) | loss 曲线 / NaN / OOM 检查;**smoke 全局 batch(16)≠ 正式(256),只验证工程启动,不证明优化动力学一致** |
| ④b | v2 正式训练(80k,从头训,KD 0.25/0.5) | `GPU_IDS=<4卡> TEXT_MODE=frozen OFFICIAL_CKPT=$CKPT ./run_train_v2.sh`(smoke 通过才允许) | `output/train_v2/checkpoints/turbovla_v2_snnfusion_80000.pth` |
| ⑤ | v2 评测(**主对比 WEIGHT_SOURCE=raw**,EMA 可另跑作参考) | `REPO_DIR=... SOURCE_CKPT=<v2 ckpt> WEIGHT_SOURCE=raw BERT_PATH=$PKG/../resources/pretrained/SpikingLM LIBERO_CHECKOUT=<clean 8f1084e> EVAL_GPU_IDS=<4卡> MODEL_LABEL="SDT-V3 + frozen SpikingLM + SNN fusion (v2)" ./run_eval.sh` | `output/eval/v2_step_80000_raw_clean_8f1084e/results/summary.json` |
| ⑥ | 对比:v2 数字 vs ③ 锚点(±2pp 解读,raw↔raw) | 两个 summary.json 并排 | 结论 |

## 2. 模型与训练配方(与官方的关系)

- v2 模型:SDT-V3 19M(SNN 视觉,224px)+ SpikingLM(SNN 文本,**在线微调 @ 5e-5**,
  对齐官方 online-BERT 策略)+ 6 层 SNN 融合(Spike2Max exact,learned readout)+ ANN ACT 头。
- 初始化(2026-09-05 用户拍板):**从头训,对齐官方**。官方 = HF 预训练骨干 +
  GroundingDINO 预加载 interaction,从头 80k(`DWH-VLA/TurboVLA/README.md:183`、
  `.../turbovla/training/trainer.py:479-501`);v2 = SDT-V3 分类权重 + SpikingLM 预训练权重,
  其余全新。已删除 `--student_init_checkpoint`(`run_train_v2.sh`)。
- KD:teacher = 官方 ckpt,feature 0.25 / action 0.5(`run_train_v2.sh:152-153`;
  teacher 构建:`code/turbovla/training/trainer.py:356`)。
- 文本模式选择逻辑:`run_train_v2.sh:106-108`(`TEXT_MODE=online` 默认)。

## 3. 协议同一性表(逐项给出实现代码)

| 协议项 | 实现代码 | 两边是否一致 |
|---|---|---|
| LIBERO commit 8f1084e + 干净工作区 | `run_eval_official.sh:60-71`、`run_eval.sh:60-71`(git rev-parse 硬校验) | ✅ 同一 gate |
| 评测参数:50 trials / chunk 12 / open-loop 12 / seed 7 / FP32 | `run_eval_official.sh:167-171` ≡ `run_eval.sh` common_args | ✅ 逐行相同 |
| 缓存防串用:导出/结果按指纹(源 ckpt 路径+大小+mtime + 权重源 + 协议常量)复用 | `run_eval_official.sh:96-99,144-159,190-207`;`run_eval.sh` `source_fp` + `.fp` sidecar | ✅ 切换模型/权重源不会汇总旧 JSON |
| 2000 episodes(4 套件 × 10 任务 × 50) | 汇总脚本 `run_eval_official.sh` / `run_eval.sh` 尾部 | ✅ |
| 模型加载路径 | `evaluate.py` → `code/turbovla/evaluation/policy.py:268`(`model_state_dict` 优先) | ✅ 官方模型经转换器进同一 policy |
| 评测精度 | 两脚本 `--precision fp32`(官方 README 示例 bf16,仅示例;锚点与 v2 同 fp32 内部自洽) | ✅ |
| 渲染/环境 | 两脚本同 export(MuJoCo EGL 等) | ✅ |
| 训练超参对齐(80k / batch 256 / warmup 10k / 全 lr 5e-5 / wd 1e-10 / clip 1.0 / seed 42 / fp32) | `run_train_v2.sh:166-180`;逐函数 diff 见 `V2_PLAN_CN.md` §6.1 | ✅ |
| 动作损失 = 官方 `masked_l1_loss` | `code/turbovla/training/trainer.py:432`(官方同函数)、`closed_loop_action_loss` 默认权重短路 `:471` | ✅ |
| LR 调度 / 优化器分组 / EMA | `code/turbovla/training/trainer.py:534,559`;EMA `code/turbovla/training/pi05.py:13`(`EMA_DECAY=0.999`) | ✅ |
| shuffle_steps_within_episode | 官方 `trainer.py:109` 与本包 `trainer.py:197` set_defaults 均为 **True**,两边都不传即一致 | ✅(2026-09-05 二审更正) |
| 文本训练模式 | 开关 `code/turbovla/training/trainer.py:206-207`;官方 ckpt 实测 `text.frozen=True`(与官方 trainer 默认 freeze 一致,README "online" 表述与实物不符)→ **TEXT_MODE=frozen 定案**;run_train_v2.sh 门禁保持启用(不符即拒启) | ✅ 已仲裁 |
| torch 2.6 torch.load 兼容 | `evaluate.py` 入口 shim(恢复 weights_only=False 默认;LIBERO init-states 裸加载在 torch 2.6 下会崩,`benchmark/__init__.py:164`) | ✅ 同一份被指纹覆盖的入口,锚点与 v2 共用 |
| KD teacher 前向 fp32 | 带 model_config 的 teacher 路径:`build_distillation_teacher` 显式置 `config.vision/interaction.compute_precision="fp32"`(**代码强制**;ckpt 记录可能是 bf16_autocast);无 model_config 的 fallback 路径依赖启动器传入参数(run_train_v2.sh 传 fp32) | ✅ 本启动路径已保证 |

## 4. 权重口径(唯一残留自由度,已处理)

- 官方自家评测用 **raw** 权重(归档 repo `turbovla/evaluation/policy.py:252` 优先
  `model_state_dict`,无 EMA);v2 评测导出 **EMA**(`run_eval.sh:90` 取
  `ema_model_state_dict`)。
- 处理:官方发布实测**无 EMA**(顶层仅 `model_state_dict` + `model_config`)→ **锚点 = raw 定案**
  (`run_eval_official.sh` WEIGHT_SOURCE 默认 model;对无 EMA 的发布请求 ema 会被转换器拒绝)。
  **2026-09-10 上游时间线复核加固**:官方仓库(H-EmbodVis/TurboVLA)07-28 初版 eval 代码
  即加载 `model_state_dict`,发布(07-30/31)时点未变——发布包 = **raw 口径**自洽;
  EMA-only(`policy.py` 明文拒用 raw)是 **09-02 提交 ced2b0c** 才改的,同提交把 released
  recipe 全局 batch 256→128。即:官方作者 09-02 起自己也转向 EMA 评测(早于我们 5 周)。
  v2 侧 `run_eval.sh` 新增 `WEIGHT_SOURCE=raw|ema` 开关(默认 ema;导出键、输出目录命名、
  summary 权重源均随之切换;ema 请求遇无 EMA 的 ckpt 干净报错,不静默回退;单元测试通过)。
- **主对比规则:raw↔raw(锚点 raw + `run_eval.sh WEIGHT_SOURCE=raw`)。**
  v2 EMA 可另跑一次(输出目录自动区分)仅作参考,不进入主对比。

## 5. 已知不可对齐项(= 实验本身,详见 `V2_PLAN_CN.md` §6.1)

1. 视觉 SDT-V3@224 vs DINOv3@256;2. 文本 SpikingLM vs BERT(训练模式已对齐);
3. SNN 融合 vs ANN enhancer;4. SNN 融合无法加载 GroundingDINO 预加载(全新 LayerScale);
5. KD 0.25/0.5(官方无蒸馏)。
工程性偏差(不触碰学生优化数学):KD teacher 前向 fp32;KD teacher 输入 224px
(**shape 约束非疏漏**:`code/turbovla/training/trainer.py:1258` 严格校验 token 形状,
224→14×14=196 token 逐 token MSE,喂 256 会直接崩;原因注释在
`code/turbovla/data/libero_rlds.py:172-177`);save_steps 5000;评测 fp32。

## 6. 历史成绩归因(2026-09-05 二审更正)

- **55.95%**:确证的真 SpikingLM 架构成绩(ckpt model_config.text=spikinglm)。**但注意**:
  旧评测脚本用 `MUJOCO_GL=osmesa`(旧 `run_eval.sh:18`),当前管线是 EGL —— renderer 不同,
  与当前管线结果**不构成完全同协议**;如需作为参考,应在当前 EGL 管线重跑。
- **93.95%**:归因更正 = **BERT 架构**。硬证据:旧评测日志明确 "Some weights of **BertModel**
  were not initialized ... pooler.dense.*";旧训练脚本未传 `--text_encoder_type spikinglm`;
  SpikingLM 权重目录本身是 BERT 命名格式(`model_type: bert`,无 pooler)。即 BERT 架构 +
  从 SpikingLM 目录加载的 BERT 兼容权重 + pooler 全新初始化。
- **不作严格因果结论**:55.95% vs 17.4%(或 93.95%)之间混有训练谱系、KD teacher、fusion 配置、
  评测代码版本差异,"文本编码器是决定性变量"这类单变量归因不成立。

## 7. 统计解读

2000 episodes 的二项成功率 95% CI ≈ ±2pp;**v2 与锚点差距 >4–5pp 才有统计意义**。
锚点与官方 README 数字若差距大,只说明协议差异,不影响 v2 vs 锚点的公平性
(两者永远同协议、同管线、同机器)。

## 8. 当前状态(2026-09-05 执行中)

- **① 完成**:hf-mirror 直下官方发布(1.7G,10 文件)→ `resources/pretrained/TurboVLA`;
  LIBERO 为单 ckpt(`checkpoints/libero/turbovla_libero.pth`)。
- **② 完成**:Part B teacher 严格加载 OK(216.1M);转换器自述 text.frozen=True(→TEXT_MODE=frozen
  定案)、vision bf16_autocast(证实 teacher 强制 fp32 必要)、无 EMA(→raw↔raw 定案)、
  无 args/global_step;`padding_length_by_instruction` 本包原生支持;转换器补本地路径覆盖
  (官方存 hub 名 "bert-base-uncased"/"facebook/dinov3-...",离线构建需指本地,与评测端 policy 行为同构)。
- **③ 进行中**:官方锚点评测 GPU 0,1,3,5(4 套件并行)。执行中修复:torch 2.6
  `torch.load` 默认 `weights_only=True` 使 LIBERO init-states 裸加载崩
  (`benchmark/__init__.py:164`)→ `evaluate.py` 入口 shim 恢复旧默认(不脏 LIBERO checkout,
  属指纹覆盖代码,两边共用)。
- **run_eval.sh 新增 WEIGHT_SOURCE**(raw|ema,默认 ema):导出键/输出目录/summary 权重源联动;
  ema 遇无 EMA ckpt 干净报错;单元测试通过(ema/raw/raw-only/ema-对-raw-only 四例)。
- TEXT_MODE=frozen 已同步进 V2_PLAN_CN.md(§1/§2/§4/§6.1)与本文档(§0/§1/§3/§4)。
- preflight Part A(从头构建)已通过:1421 张量、SNN 融合 486 张量/14.3M 全新。
- 待办:③ 出锚点 → ④a smoke(TEXT_MODE=frozen)→ ④b 正式 → ⑤ v2 评测(主对比 raw)→ ⑥ 对比。
