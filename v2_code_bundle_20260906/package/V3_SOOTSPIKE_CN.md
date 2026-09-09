# v3 实验架构:sootspike 文本骨干(单卡 H100 80G)

> **正式命名**:短名 `v3-sootspike`(版本号 + 本次唯一改动模块);全架构名
> `libero_sdtv3_sootspike_snnfusion_20260907`,沿用 v2 的
> `libero_sdtv3_bert_snnfusion_20260831` 约定——四槽位 `libero_{视觉}_{文本}_{融合}_{日期}`,
> 与上一版逐槽位对比即得架构差异(本实验只有 text 槽位 bert→sootspike)。
> 评测时 `MODEL_LABEL` 一律用全架构名,保证 summary.json 自描述。

> 2026-09-07。v2 的后续实验:**文本骨干从 SpikingLM 换成 sootspike**,其余全部继承 v2。
> 执行入口:`/data/260010028/dwh_vla/launch_v3.sh`。行号/路径以本包当前代码为准。
> 配套文档:`V2_PLAN_CN.md`(v2 完整计划)、`V2_FLOW_AND_COMPARISON_CN.md`(v2 流程与对比口径)。

## 0. 当前状态一句话

**代码接入与资产全部就绪,真卡端到端验证已通过**(2026-09-07,详见 §5);剩下的是
算力服务器上的两步:**batch probe → 正式训练**(§6)。本实验**只用 1 张卡(GPU 0)**,
另外三张另有用途(2026-09-07 拍板,§4)。

## 1. sootspike 是什么

- 上游:`CayleyZ/SmoothSpike`——**SpikingBERT 骨干 + 可学习正交 SmoothSpike 变换**的脉冲驱动 BERT。
- 我们跑的是 **fused 推理版**(`third_party/SmoothSpike/spikingbert_rot_inf.py`):
  逐层 H2/H3 旋转已折入线性权重,全局 `bert.H1` 保留。config `T=4`。
- 权重管线:ModelScope `kailai1104/SmoothSpike` 未融合 ckpt →
  `third_party/SmoothSpike/convertor.py` 本地融合 → `resources/pretrained/SmoothSpike/smoothspike-bert-base-fused/`。
- 集成补丁来自参考仓库 `wxqnl/spike-turbovla`(已克隆到 `/data/260010028/dwh_vla/spike-turbovla`):
  相对导入修复、可选 fast-Hadamard(`patches/integration.patch`),其 `turbovla/text/smoothspike.py`
  是我们加载器的 API 参照。
- **与 SpikingLM 的关键差异**:SmoothSpike 跨前向**无脉冲状态残留**,`reset_sootspike_state`
  是 no-op(已实测二次前向逐位一致);SpikingLM 则需要每步 reset。

## 2. 资产位置(全部在 /data 持久卷,容器重建不丢)

| 资产 | 路径 |
|---|---|
| fused 权重 + tokenizer | `v2_code_bundle_20260906/resources/pretrained/SmoothSpike/smoothspike-bert-base-fused/` |
| SmoothSpike 源码(已打补丁)+ convertor.py | `v2_code_bundle_20260906/resources/third_party/SmoothSpike/` |
| 未融合原始下载(dl/) | `v2_code_bundle_20260906/resources/pretrained/SmoothSpike/dl/checkpoints/smoothspike-bert-base/` |
| 参考仓库(只读参照) | `/data/260010028/dwh_vla/spike-turbovla/` |
| 启动器 | `/data/260010028/dwh_vla/launch_v3.sh` |
| batch 探测 | `v2_code_bundle_20260906/package/run_batch_probe.sh` |

## 3. 代码接入点(v2 → v3 的 diff 面)

| 文件 | 改动 |
|---|---|
| `code/turbovla/training/trainer.py` | `--text_encoder_type` choices 增加 `sootspike`;新增 `--text_source_path`;分支接线 |
| `code/turbovla/models/configuration.py` | encoder_type 白名单放行 `sootspike` |
| `code/turbovla/models/turbovla.py` | `model_source_path` 接线:`text_source_path` 优先,向后兼容 `spikinglm_source_path` |
| `code/turbovla/models/text_encoder.py` | `sootspike` 构建分支(`load_sootspike_backbone`)+ reset 钩子分发 |
| `code/turbovla/text/sootspike.py` | **新加载器**,契约镜像 `spikinglm.py`:返回带 `config.hidden_size`、BertModel 风格 forward(`input_ids`/`attention_mask`/`position_ids` → `last_hidden_state [B,L,D]`)的 nn.Module;权重/文件缺失时清晰报错 |
| `run_train_v2.sh` | 参数化:`TEXT_ENCODER_TYPE=spikinglm\|sootspike`;**默认行为与 v2 逐字节一致**;`sootspike` 强制要求 `TEXT_WEIGHTS_PATH`(权重+tokenizer)与 `TEXT_SOURCE_PATH`(模型代码目录) |

门禁不变:`TEXT_MODE` 与官方 ckpt 自述(`text.frozen=True`)不符会拒绝启动(`FORCE_TEXT_MODE=1` 可越过,不建议)。

## 4. 单卡决策与 batch 语义(2026-09-07 拍板)

- **本实验只用 1 张卡(GPU 0),另外三张另有打算**。
- 后果:单卡 fp32 填满 80G,micro-batch 大约只能到 ~56-64,**全局 batch ≈ 64**——
  4 卡时代"占满显存 ≈ 全局 256"的等价关系不存在了。
- `launch_v3.sh` 默认:`BATCH_SIZE=64` = `TARGET_GLOBAL_BATCH`(accum=1,显存填满优先);
  精确上限以 `run_batch_probe.sh` 实测为准(默认扫 48/56/64/72,输出峰值显存 + s/it)。
- **保配方备选**(全局 256,accum=4,墙钟时间约 4×):
  在 launch 前改 `export TARGET_GLOBAL_BATCH=256`。

## 5. 验证记录(2026-09-07)

| 项 | 结果 |
|---|---|
| 融合(convertor.py) | Removed 24 keys / Kept bert.H1 True / 逐层 H2/H3 残留 0(与上游 README 预期一致) |
| 加载器严格加载 | hidden 768、216 张量、0 unexpected;missing 仅 tied aliases(`cls.predictions.*`,无害) |
| 完整 `TurboVLATextEncoder` 构建 | 110.1M 参数,cuda:0 正常 |
| 真卡前向(MIG H100 40G) | `[2,13,768]` fp32,finite=True,mean=-0.017/std=0.323 |
| reset 契约 | no-op;二次前向 `torch.equal=True`(无状态残留) |
| **训练闭环 smoke**(2026-09-07,MIG 40G,batch 8,30 步) | **一次通过**:loss=1.047(avg 1.115),act=0.362 / feat_kd=2.532 / act_kd=0.238,无 NaN,gacc=0.525;~2.5-3 it/s;ckpt 正常落盘(`train_v3_smoke/checkpoints/turbovla_v3_smoke_30.pth`) |

注意:**CPU 无法前向**(spikingjelly LIF 依赖 CUDA kernel),所以一切前向验证必须在真卡做;
本 dev pod 只有 MIG 40G 切片,能做功能验证,80G 显存级别的 batch 探测必须在算力服务器跑。

## 5.1 正式训练结果(2026-09-07 14:13 → 09-08 03:22,13h08m,单卡 H100 80G,batch 56)

- 80,000/80,000 步完成,1.69 it/s(与 probe 预估一致);无 NaN、无跳变,40k 后平台期。
- 末态:loss 0.132 / act 0.077 / feat_kd 0.227 / act_kd 0.016 / **gacc 0.966**;action horizon MAE 0.073-0.081(12 档一致);spike 诊断 firing rate 0.215(range 0.027-0.455,103 LIF 节点)。
- ckpt:`output/train_v3/checkpoints/turbovla_v3_sootspike_{70000,75000,80000}.pth`。

## 5.2 评测侧接入(2026-09-08 补丁)

- **评测链路 python 零改动**:ckpt 内 `model_config.text` 已存正确 sootspike 配置(权重/源码路径在共享盘),`policy.py` 直接信任存储值。
- `run_eval.sh` 补丁:① 导出步新增 `TEXT_SOURCE_PATH` 重写(仅 sootspike 生效,v2 指纹不变);② GPU 门禁放宽为 **1 卡(串行 4 suite)或 4 卡(并行)**;③ 新增 `EVAL_SUITES` 子集选择;④ 汇总脚本按实际 suite 集合校验 episode 总数。
- dev pod 依赖修复:`apt install libgl1 libglib2.0-0 libegl1 libglx-mesa0`(容器重建后需重装);LIBERO checkout=`/data/260010028/dwh_vla/LIBERO_eval`(commit 8f1084e,干净)。
- dev pod(MIG 40G)实测吞吐:**~45 s/episode → 单 suite ≈ 6.3h,4 suite 串行 ≈ 25h**;算力服务器 4 卡并行 ≈ 6h 全套。
- **统一评测入口:`/data/260010028/dwh_vla/eval_v3.sh`**(算力服务器/任意 pod 通用):自动装 libGL(新容器缺)、校验/自克隆 LIBERO checkout(共享盘 `/data/260010028/dwh_vla/LIBERO_eval`,commit 8f1084e)、默认 `EVAL_GPU_IDS=0,1,2,3` 四卡并行,可 `EVAL_GPU_IDS=0` 单卡串行、`EVAL_SUITES=` 选子集;已完成 suite 指纹缓存自动跳过。
- dev pod 首跑残段(2026-09-08,随会话断开中断):libero_spatial 6.7/10 任务 **61.9%**(task1 0/50 异常,task2/5 100%),suite 结果 json 未落盘,重跑从零计。
- **4 卡并行首跑事故与修复(2026-09-09)**:robosuite `renderers/context/egl_context.py` 在
  `MUJOCO_EGL_DEVICE_ID` 未设时把 `CUDA_VISIBLE_DEVICES` 的裸数字**直接当 EGL 设备下标**,
  而 NVIDIA EGL 枚举本身已被 CUDA_VISIBLE_DEVICES 过滤成 1 个设备 → 锁非 0 卡的进程全部
  启动即崩(`must be an integer between 0 and 0 (inclusive), got 1`)。只有 GPU 0 的
  libero_spatial 幸存并完整跑完:**78.2%(391/500),4h05m**;10 任务全在 42%–98%,无 0% 任务
  (dev pod 残段的"task1 全错"确认是中断统计假象)。
  修复:① robosuite egl_context.py 本地补丁(裸数字按可见列表翻译成局部下标;映射单测 6 例
  全过 + dev pod 真机 EGL 初始化不回归);② run_eval.sh suite 失败时不再写指纹 sidecar。
  两处均不进 code_fp/source_fp,spatial 缓存仍有效,重启只补跑 3 suite。

## 6. 启动命令(算力服务器,换算力后)

```bash
# 0) 先探最大可用 batch(约几分钟/档,输出峰值显存 + s/it)
cd /data/260010028/dwh_vla/v2_code_bundle_20260906/package && bash run_batch_probe.sh

# 1) 按探得结果改 launch_v3.sh 里的 BATCH_SIZE,然后一行启动
bash /data/260010028/dwh_vla/launch_v3.sh
#    启动器自带:资产存在性预检、CUDA 预检、日志 tee 到 output/launch_v3_<时间戳>.log
```

其余可能用到的:

```bash
tail -f /data/260010028/dwh_vla/v2_code_bundle_20260906/package/output/launch_v3_*.log  # 看训练日志
nvidia-smi                                                                              # 显存/利用率
bash /data/260010028/dwh_vla/cc.sh --resume                                             # 恢复 Claude 历史对话
```

## 7. 与 v2 的关系(对比口径)

- 除文本骨干(sootspike 替代 SpikingLM)外全部继承 v2:SDT-V3 19M SNN 视觉、6 层 Spike2Max
  SNN 融合(learned readout)、ANN ACT 头、KD teacher = 官方 TurboVLA LIBERO ckpt
  (feature 0.25 / action 0.5)、从头训、`TEXT_MODE=frozen`。
- **注意 batch 语义差异**(§4):v3 默认全局 ~64 vs v2 全局 256,做横向对比时要么用
  accum=4 保配方档,要么把 batch 差异记入结论边界。
- 对比锚点不变:v2 实测锚点(同机同 seed 同协议,raw↔raw)与官方 LIBERO ckpt 锚点,
  ±2pp 解读;不与官方 README 论文数字直接比。
