# 消融四臂启动手册（已验证）

> 状态（2026-09-15）：**启动链路已在 pod 上端到端验证通过**（见 §4）。
> 脚本：`scripts/run_train_v2_ablation.sh`（= 你的 `run_train_v2.sh`，仅 4 处改动）。
> 配置：KD 全开（教师=官方 LIBERO ckpt，feat 0.25 / act 0.5）、ACT 固定 `ann`、文本冻结、80k 步、fp32。

## 1. 与 `run_train_v2.sh` 的差异（只有这 4 处）

| # | 改动 | 原因 |
|---|---|---|
| 1 | 新增 `VISION_TYPE` / `DOWNSTREAM_TYPE` 两个轴 + `VISION_{MODEL,PRETRAINED,SOURCE}_PATH` 可覆盖 | 原来视觉写死 `sdtv3_19m`、融合写死 `spiking` |
| 2 | 文本分支新增 `bert`（`case` 里原来只有 `spikinglm`/`sootspike`） | **模型侧本就支持 BERT**（`trainer.py:55` 的 choices、`configuration.py:130` 只对脉冲文本要 source path），是启动脚本缺分支；不补会直接 `die` |
| 3 | `--vision_encoder_type` / `--vision_model_path` / `--vision_pretrained_checkpoint` / `--vision_model_source_path` 改为变量 | 同 1 |
| 4 | `--downstream_type` 改为变量；source-path 改为"空值不传参" | 同 1；BERT 不需要 source path，原写法会多出一个空参数 |

其余（KD 教师/权重、优化器、数据、步数、精度、`PYTHONPATH` 顺序、resolved-package 校验）**一字未改**。
`CODE_DIR` 现在可覆盖，便于异地干跑。

**已按你们的实际部署放好**（与 `launch_v3.sh` 同风格）：

| 文件 | 位置 |
|---|---|
| `run_train_v2_ablation.sh` | `$PKG`（与 `run_train_v2.sh` 同级，`PKG=/data/260010028/dwh_vla/v2_code_bundle_20260906/package`）|
| `launch_ablation.sh` | `/data/260010028/dwh_vla/`（与 `launch_v3.sh` 同级）|

## 2. 启动：四卡并行（推荐，13 小时四臂全出）

**训练配方对齐 A4**（v3 sootspike 80k）：**1 卡 × batch 56 = 全局 56（accum=1）**、
`TEXT_MODE=frozen`、fp32、80k 步、KD 教师=官方 ckpt。A4 实测 1.69 it/s ≈ 13h08m/臂。

```bash
# 平台任务命令填这一行(前台跑,勿 nohup —— 容器随主进程退出被回收)
bash /data/260010028/dwh_vla/launch_ablation_4gpu.sh
```

四臂各占一张卡并行：`a0p→GPU0, a1→GPU1, a2→GPU2, a3→GPU3`，**每臂都是"1 卡 × batch 56"**，
与 A4 逐项一致 → 四臂 + A4 完全同配方，**无需任何口径说明**。

**为什么不是"4 卡 DDP 单臂"**：那样每卡 micro-batch 只剩 14（全局才等于 56），而 SDT-V3 主干用
BatchNorm，micro-batch 一变 BN 统计量就变 → 与 A4 不可比。四臂并行则两全：总墙钟同为 ~13 h，
且每臂配方不变。

**batch 56 已实测可行**：最重的 A0'（DINOv3-B，85.66M）在 **40G 的 MIG 切片**上跑通 3 步无 OOM
→ 80G 卡上四臂用 56 稳。因此默认**跳过 probe**；若仍想探（或换机器复验）：

```bash
PROBE=1 bash /data/260010028/dwh_vla/launch_ablation_4gpu.sh    # 先用 GPU0 探 a0p
```

其它模式：

```bash
DRY=1 bash /data/260010028/dwh_vla/launch_ablation_4gpu.sh          # 只打印四臂命令
ARMS="a1,a3" GPUS="0,1" bash ...                                    # 只用两张卡
MAX_STEPS=2000 bash ...                                             # 短跑自检
bash /data/260010028/dwh_vla/launch_ablation.sh a1                  # 单臂(旧路径仍可用)
bash /data/260010028/dwh_vla/launch_ablation.sh dry:a1              # 单臂只看命令
```

**★ 别用 `cmd | grep -q` 做判断（2026-09-17 实测踩到，评测任务启动 45 秒即 exit 1）**：
`grep -q` 命中即退出 → 上游若还在写就吃 SIGPIPE → 在 `set -o pipefail` 下**整条管道判失败** → 明明
ckpt 存在也报"缺"，且**时通时不通**（取决于时序）。同理 `... | tail -1` / `head -n` 也会 SIGPIPE 上游，
出现在脚本末尾汇总时会**把已成功的任务标成失败**。正确写法：先捕获再判断
（`out=$(cmd 2>&1) || true; case "$out" in ...;; esac`），或给管道加 `|| true`。

**★ TMPDIR 必须短（2026-09-16 实测踩到，四臂全崩）**：Python multiprocessing 的 AF_UNIX
socket 路径上限 **107 字符**。最初把每臂 TMPDIR 设成 `$PKG/output/ablation/tmp_<arm>`
（79 字符）→ `+ socket 名 ≈ 109` → 在 `resource_sharer.DupFd` 上崩
`OSError: AF_UNIX path too long`，四条臂在第 0 步全灭。现在固定用 `/tmp/abl_<arm>`（11–12 字符），
并在启动前断言长度 < 60。**任何放共享卷长路径下的 TMPDIR 都有这个风险**。

并行安全性：每臂独立 `TMPDIR`（`/tmp/abl_<arm>`）+ 独立日志（`$PKG/output/ablation/parallel_<ts>/`）；
`torchrun --standalone` 用自动分配的空闲端口，不会撞端口；启动错开 45 s；数据集仅 9.6 GB
（节点 page cache 可完全容纳），四臂并发读盘不是瓶颈；主进程 `wait` 住直到四臂全部结束。
`RESUME_MODE=all`：任务被抢占后重启会从最近 ckpt 续训。

> **A4（SDT-V3+sootspike+spiking，92.2%）不要重跑** —— 现有 ckpt/结果直接复用。

## 3. 启动后 5 分钟内核对

```bash
tail -f $PKG/output/ablation_<arm>_*.log
# 期望:
#   [INFO] arm=... vision=... text=... fusion=ann batch=56
#   [INFO] arm: vision=... | text=... | fusion=ann | action=ann(ACT, 固定)
#   [INFO] official ckpt self-report: text.frozen=True; TEXT_MODE=frozen
#   首次 spike 诊断行:rate/rate_min/rate_max/qkv/attn_entropy/lif_nodes,
#     lif_nodes 期望 ≈ 103(A4 实测)
```
**若 `[INFO] arm` 行里的三轴与臂不符 → 立刻停**（说明环境变量没传进去）。

## 4. 本地验证记录（2026-09-15，pod）

### 4.1 构建与命令行（两条独立链路互相印证）

| 校验 | 脚本 | 结果 |
|---|---|---|
| 真实 trainer 路径构建 5 个臂 + 教师 | `arm_build_check.py` | **PASS 5/5**，报告 `../arm_build_report.json` |
| dry-run 命令行 → 真实解析器 → 构建 | `check_arm_cli.py` | **PASS 4/4**（参数总量与上表逐位一致）|
| 与原脚本等价性 | `diff` argv | A4 配置下与 `run_train_v2.sh` **逐字节一致** |

### 4.2 四臂训练闭环冒烟（MIG 40G，batch 8，30 步）

四臂都是**新组合**（A4 是 spiking 融合，ANN 融合臂从未训过），故逐臂实跑：

| 臂 | 步数 | loss | feat_kd | act_kd | 结果 |
|---|---|---|---|---|---|
| A0' | 30/30 | 0.775 | 1.73 | 0.24 | ✅ ckpt 2.14 GB |
| A1 | 30/30 | 0.924 | 1.99 | 0.24 | ✅ ckpt 1.06 GB |
| A2 | 30/30 | 0.763 | 1.66 | 0.24 | ✅ ckpt 2.14 GB |
| A3 | 30/30 | 0.917 | 1.98 | 0.23 | ✅ ckpt 1.06 GB |

- 全部无 NaN；**KD 生效**；`spk_reg=0`（ann 融合下预期为 0）；**EMA 权重随 ckpt 落盘**
  （`ema_decay=0.999` + `ema_model_state_dict`，与 raw 同为 1065 张量）→ 主口径 EMA 可用。
- ckpt 记录 config 正确（实测 A1）：`vision=sdtv3_19m@224`、`text=bert`（`model_source=None`）、
  `text.frozen=True`、`spike.downstream_type=ann`、`action.decoder_type=ann/horizon=12`。
- 训练速度排序与推理延迟排序一致（DINOv3 臂 ~6.7 it/s vs SDT-V3 臂 ~5.3 it/s）—— 脉冲算子在
  GPU 上更慢，训练侧同样成立。
- 冒烟 ckpt 留在 `$PKG/output/ablation_smoke/checkpoints/`（勿与正式 run 混淆）。

### 4.3 batch 56 显存实测（MIG 40G 上跑最重的 A0'）

| 项 | 值 |
|---|---|
| 臂 / 配置 | A0'（DINOv3-B 85.66M，最重）× batch **56** × accum 1，fp32 |
| 卡 | H100 MIG **3g.40gb**（只有 40G，比正式卡小一半） |
| 结果 | 3/3 步完成、**无 OOM**、ckpt 落盘 |

→ 80G 卡上四臂用 batch 56 稳妥，probe 可跳过。

参数（实测，fp32）：

| 臂 | 视觉 | 文本 | 融合 | 总参 | 可训练 | 视觉 | 文本\* | 融合 | ACT |
|---|---|---|---|---|---|---|---|---|---|
| A0' | DINOv3-B | BERT | ANN | 216.073M | 106.590M | 85.660M | 109.679M | 14.213M | 5.272M |
| A1 | SDT-V3 | BERT | ANN | 148.516M | 39.034M | 18.627M | 109.679M | 14.213M | 5.272M |
| A2 | DINOv3-B | sootspike | ANN | 216.082M | 106.590M | 85.660M | 109.688M | 14.213M | 5.272M |
| A3 | SDT-V3 | sootspike | ANN | 148.525M | 39.034M | 18.627M | 109.688M | 14.213M | 5.272M |
| A4 | SDT-V3 | sootspike | spiking | 148.631M | 39.139M | 18.627M | 109.689M | 14.316M | 5.272M |
| 教师 | DINOv3-B | BERT | ANN | 216.073M | — | — | — | — | — |

\* 文本编码器整体冻结，可训练只有 0.197M 的 `text_projection`。

关键结论：
- **ACT 结构签名 `add6bc7a001db16d` 五臂全同** → "ACT 不换"有硬证据；
- A0'–A3 融合模块均为 `VisionLanguageInteraction`（官方 ANN），A4 才是 `SpikeVisionLanguageInteraction` → 前端效应隔离干净；
- 视觉参数量 DINOv3-B 85.66M vs SDT-V3 18.63M（**4.6×**），文本两者几乎同参（同构替换）→ 参数量差异几乎全部来自视觉。

## 5. 口径说明（写论文时必须声明）

1. **输入分辨率 224**：官方 LIBERO ckpt 记的是 `image_size=256`，本消融统一用 224（SDT-V3 的原生尺寸；否则脉冲视觉臂会在非原生分辨率下被不公平削弱）。代价是 **A0' 不等于官方 92.7% 那条锚点**——A0' 是"同 pipeline 的 DINOv3@224"，这正是它存在的意义：用同 harness 的自训对照臂，而不是跨设置比。
2. **所有臂都开了 KD**（教师=官方 ckpt）：A0' 与教师同构（216.073M 一致），它回答的是"我们的 harness + KD 能到什么水平"，不是"无 KD 的纯 ANN 上限"。
3. 文本冻结策略全臂一致（`frozen`），与学生网络规模无关。

## 6. 顺序建议

1. **A1**（SDT-V3+BERT+ANN）—— 离已有 A4 最近，先跑通流程
2. **A0'**（DINOv3+BERT+ANN）—— 对照臂
3. **A2 / A3**
4. 训完 → 统一评测（复用 A0/A4 + `libero_object` 漂移核对，见 `../ABLATION_DESIGN_CN.md` §3）
5. 四维测量（参数已完成；内存/能耗/延迟见 `measure_arm.py`）
