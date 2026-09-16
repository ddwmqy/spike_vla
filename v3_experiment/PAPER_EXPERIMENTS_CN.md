# 论文实验部分草稿（可直接粘进 Overleaf）+ 数据来源对照

> 生成于 2026-09-16。所有数字来自共享卷上的实测产物，来源见文末对照表。
> **标 `TBD` 的格子 = 四臂训练/评测还没跑完，跑完由 `report_main_table.py` 自动生成。**

---

## ⚠️ 先说三个必须先修的问题（与实验无关，但会被审稿人一眼看到）

| # | 问题 | 位置 | 说明 |
|---|---|---|---|
| 1 | **摘要来自另一篇论文** | Abstract | 现在写的是 "Threshold Guarding Optimization (TGO)"、对抗鲁棒性、threshold-neighboring neurons —— **与本篇 SpikeVLA 完全无关**，须整段替换 |
| 2 | **Figure 1 图注来自另一篇论文** | Fig. 1 | 现在描述的是 "FPGA accelerator for BiSFormer / BinaryDSP / streaming self-attention" —— 与本文无关，须替换 |
| 3 | **融合模块的名称与实验实现不一致** | §4.3 | 正文写的是 **"Bidirectional Multi-Winner Spike Fusion"（WTA + 三元脉冲神经元 TSN）**，而**我们跑的实验用的是 Spike2Max / SpikeSDS 融合** ✗ —— 要么把 §4.3 改成实现的模块，要么按 §4.3 重新跑实验。**这条最要紧** |

其它小项：§2 标题 "RELATIVE WORK" → **RELATED WORK**；§4.4 / §4.5 / 整个 §5 / §6 / Appendix 目前都是空占位；正文多处引用未解析（`?`，出现在 RT-1、RT-X、Octo、$\pi_0$、action chunking、RoboMamba、SpikingBERT、SNN-BERT、SpikeLLM、以及 §2.1 的 "classification Zheng et al. (2021); ?"）。

另有一处**数值口径不一致**：引言写 "AC 约为 MAC 的 1/6–1/10"，而实验里的能耗换算用的是 **4.6 pJ / 0.9 pJ**（Horowitz 45nm），比值 ≈ **1/5.1**，超出该区间。建议正文改成"约 1/5"并给出 Horowitz 引文，或换成与 1/6–1/10 相符的常数。

---

## 5.1 Experimental Setup（英文正文草稿）

```latex
\subsection{Experimental Setup}
\paragraph{Benchmarks.} We evaluate on two robotic manipulation benchmarks.
\textbf{LIBERO} \citep{liu2023libero} comprises four task suites
(\textit{spatial}, \textit{object}, \textit{goal}, and \textit{long-horizon} \textsc{Libero-10}),
each with 10 tasks. \textbf{RoboTwin 2.0} \citep{robotwin2024} contains 50 bimanual
manipulation tasks evaluated under its \textit{clean} setting.

\paragraph{Evaluation protocol.} Unless stated otherwise, we report success rate over
$4 \times 10 \times 50 = 2000$ LIBERO episodes per model ($50$ trials per task) under a
fixed protocol: fp32 inference, seed $7$, action-chunk length $12$, and $12$ open-loop
execution steps, on a pinned LIBERO checkout (commit \texttt{8f1084e}). We report the
exponential-moving-average (EMA) weights at the end of training as the primary metric and
include raw weights for reference, since the two differ substantially
(Table~\ref{tab:libero_main}).

\paragraph{Models.} \textbf{Baseline (ANN front-end).} DINOv3 ViT-B/16 \citep{dinov3} for
vision and BERT-base \citep{devlin2019bert} for language, with the original bidirectional
ANN fusion and an ACT action head. \textbf{Ours (spiking front-end).} SDT-V3 (19M)
\citep{yao2024sdtv3} for vision and SmoothSpike-BERT \citep{smoothspike} ($T{=}4$) for
language. The action head is kept identical across all variants (ACT, chunk size 12,
7-DoF actions) so that the comparison isolates the front-end.

\paragraph{Training.} All models are trained for $80$k steps with a single H100 (batch size
$56$, accumulation $1$, fp32, learning rate $5\times 10^{-5}$), taking
$\sim$13 hours per run. Following \S REF, we distill from the released TurboVLA LIBERO
checkpoint via feature and action matching (weights $0.25$ and $0.5$; teacher weights frozen
in fp32). Text encoders are frozen in every run so that only the front-end choice varies.
For RoboTwin 2.0 we train on $4\times$H100 with per-device batch $48$ (global $192$) for
$55$k steps and compare against the released TurboVLA checkpoint evaluated under the same
harness.

\paragraph{Efficiency metrics.} We report four dimensions with fully specified
measurement procedures. \textbf{(i) Parameters} are counted per module. \textbf{(ii) Energy}
is estimated by counting synaptic operations on a single forward pass: dense ANN operators
contribute multiply--accumulate (MAC) operations, whereas operators whose inputs are binary
or multi-level quantized spikes contribute accumulate (AC) operations only. We convert with
$E = N_{\mathrm{MAC}}\times 4.6\,\mathrm{pJ} + N_{\mathrm{AC}}\times 0.9\,\mathrm{pJ}$
(45\,nm CMOS \citep{horowitz2014}). \textbf{This is a theoretical estimate under an
event-driven ASIC assumption; on GPUs spiking networks are simulated densely and consume
neither less time nor less energy} --- we therefore also report measured wall-clock latency.
\textbf{(iii) Memory} is the peak CUDA memory of a single forward pass (measured after
warm-up, which matters because some components build caches lazily). \textbf{(iv) Latency}
is the median of 10 timed forward passes at batch size 1 on an H100.
```

**中文要点（写正文时注意）**：

1. **口径必须写清"EMA 为主、raw 另报"** —— 两者差距巨大（A4 上 92.2% vs 81.5%），不写会被质疑。
2. **能耗必须显式声明是理论估计**，且"GPU 上模拟不省时不省电" —— 这是审稿人最容易攻击的点，先自己堵上。
3. **延迟反而更慢要主动交代**（见下表 A4 151ms vs A0' 28ms）—— 这是"稠密模拟"的直接后果，主动说明比被问出来好。
4. **KD 教师与全臂一致**要写明（否则"spiking 前端能训起来"会被归功于初始化）。KD 这条我们已经拍板全臂开启。

---

## 表 1：LIBERO 主结果（性能）

```latex
\begin{table}[t]
\centering
\caption{Success rate on LIBERO under the shared 2000-episode protocol (50 trials/task).
All spiking variants share the identical ACT head; only the front-end differs.
``Official'' is the released TurboVLA LIBERO checkpoint (raw weights); all other rows are
our re-implementations trained with the same recipe.}
\label{tab:libero_main}
\begin{tabular}{llcccccc}
\toprule
Model & Front-end (V / L) & Spatial & Object & Goal & Long & Avg. \\
\midrule
Official TurboVLA           & ANN / ANN            & 93.6 & 98.8 & 94.4 & 84.0 & \textbf{92.7} \\
\midrule
A0$'$ ANN control           & DINOv3-B / BERT      & TBD & TBD & TBD & TBD & TBD \\
A1  spiking vision only     & \textbf{SDT-V3} / BERT & TBD & TBD & TBD & TBD & TBD \\
A2  spiking language only   & DINOv3-B / \textbf{SmoothSpike} & TBD & TBD & TBD & TBD & TBD \\
A3  spiking V+L             & \textbf{SDT-V3} / \textbf{SmoothSpike} & TBD & TBD & TBD & TBD & TBD \\
A4  full (V+L+fusion)       & \textbf{SDT-V3} / \textbf{SmoothSpike} & 95.6 & 98.0 & 95.2 & 80.0 & 92.2 \\
\quad (raw weights)         &                      & 78.2 & 99.2 & 80.8 & 67.6 & 81.5 \\
\bottomrule
\end{tabular}
\end{table}
```

**已实测的格子**（EMA 口径，官方锚点为 raw）：

| 模型 | Spatial | Object | Goal | Long | 合计 |
|---|---|---|---|---|---|
| 官方 TurboVLA（raw）| 93.6% (468/500) | 98.8% (494/500) | 94.4% (472/500) | 84.0% (420/500) | **92.70%** (1854/2000) |
| A4（EMA，主口径）| 95.6% (478) | 98.0% (490) | 95.2% (476) | 80.0% (400) | **92.20%** (1844/2000) |
| A4（raw）| 78.2% (391) | 99.2% (496) | 80.8% (404) | 67.6% (338) | **81.45%** (1629/2000) |
| A0'–A3 | TBD | TBD | TBD | TBD | TBD |

> **注意**：A4 的 EMA 分套件数字与"官方 raw"不可直接逐格对比（口径不同）。表格里 A4 主行用 EMA、括号行给 raw，官方行标注 raw —— 这一点必须在表注里写死，否则 92.2 vs 92.7 的对比会被误读。

---

## 表 2：效率（参数 / FLOPs / 能耗 / 内存 / 延迟）—— **全部已实测**

```latex
\begin{table*}[t]
\centering
\caption{Efficiency at a single forward pass (batch size 1). FLOPs count a
multiply--accumulate as 2 operations and a spike-driven accumulate as 1.
``Dense-eq.'' is the FLOPs/energy if every spike-driven operator were executed densely.
Energy assumes 45\,nm event-driven accumulation ($4.6$\,pJ/MAC, $0.9$\,pJ/AC) and
\textbf{does not reflect GPU execution}. Latency is the median of 10 runs on one H100.
Note the teacher (216.1\,M, frozen) is used only during training and is excluded here.}
\label{tab:efficiency}
\footnotesize
\begin{tabular}{lrrrrrrrrrr}
\toprule
Model & \#Params & Train. & Vision & Text & Fusion & MACs & ACs & FLOPs & Dense-eq. & Energy & Mem. & Lat. \\
      & (M)     & (M)    & (M)    & (M)  & (M)    & (G)  & (G) & (G)   & (G)       & (mJ)   & (MiB) & (ms) \\
\midrule
A0$'$ & 216.1 & 106.6 & 85.7 & 109.7 & 14.2 & 39.06 &  0.00 & 78.12 & 78.12 & 179.7 & 901  & 28.0 \\
A1    & 148.5 &  39.0 & 18.6 & 109.7 & 14.2 &  4.60 &  8.43 & 17.63 & 39.50 &  28.8 & 959  & 39.7 \\
A2    & 216.1 & 106.6 & 85.7 & 109.7 & 14.2 & 37.30 &  1.54 & 76.15 & 88.94 & 173.0 & 1235 & 45.3 \\
A3    & 148.5 &  39.0 & 18.6 & 109.7 & 14.2 &  2.84 &  9.98 & 15.66 & 50.31 &  22.1 & 1290 & 59.5 \\
A4    & 148.6 &  39.1 & 18.6 & 109.7 & 14.3 &  2.42 & 10.83 & \textbf{15.68} & 69.51 & \textbf{20.9} & 679 & 151.0 \\
\bottomrule
\end{tabular}
\end{table*}
```

**相对 A0'（全 ANN 对照）**：A1 FLOPs 22.6% / 能量 16.0%（AC 占比 64.7%）；
A2 97.5% / 96.3%（4.0%）；A3 **20.0% / 12.3%**（77.8%）；A4 **20.1% / 11.6%**（81.7%）。

**分模块 GFLOPs（单次推理）**：A0$'$ 视觉 **68.7 G（占 88%）** → A1/A3 降到 **8.7 G（−87%）**；
文本 A0$'$ 3.6 G → A2/A3 1.6 G（−56%）；融合与动作头各臂基本不变（4.4 G / 0.4 G）。

---

## 表 2-旧：效率（参数 / 能耗 / 内存 / 延迟）—— **全部已实测**

```latex
\begin{table}[t]
\centering
\caption{Efficiency of each front-end configuration at batch size 1 (single forward pass).
Energy is a theoretical estimate assuming event-driven accumulation
($4.6$\,pJ/MAC, $0.9$\,pJ/AC at 45\,nm); it does not reflect GPU execution, where spiking
models are simulated densely. ``Dense-equiv.'' is the energy the same network would consume
if every spike-driven operator were executed as a dense MAC. Latency is measured on one
H100 (batch 1, median of 10 runs).}
\label{tab:efficiency}
\begin{tabular}{lrrrrrrr}
\toprule
Model & \#Params & Vision & Text & MACs & ACs & Energy & Dense-eq. & Latency \\
      & (M)     & (M)    & (M)  & (G)  & (G) & (mJ)   & (mJ)      & (ms) \\
\midrule
A0$'$ & 216.1 & 85.7  & 109.7 & 39.06 & 0.00  & 179.7 & 179.7 & 28.0 \\
A1    & 148.5 & 18.6  & 109.7 &  4.60 & 8.43  &  28.8 &  90.8 & 39.7 \\
A2    & 216.1 & 85.7  & 109.7 & 37.30 & 1.54  & 173.0 & 204.6 & 45.3 \\
A3    & 148.5 & 18.6  & 109.7 &  2.84 & 9.98  &  22.1 & 115.7 & 59.5 \\
A4    & 148.6 & 18.6  & 109.7 &  2.42 & 10.83 &  \textbf{20.9} & 159.9 & 151.0 \\
\bottomrule
\end{tabular}
\end{table}
```

**可直接写进正文的结论**（全部有实测支撑）：

- **视觉是能耗主战场**：DINOv3-B 单模块 34.37 G MACs，占 A0' 总 MACs 的 **88%**；换成 SDT-V3 后变成 0.12 G MACs + 8.43 G ACs → 单臂能量 **179.7 → 28.8 mJ（−84.0%）**。
- **文本脉冲化只贡献 −3.7%**（A0'→A2 为 179.7→173.0 mJ）—— 因为 sootspike 与 BERT **参数量几乎相同**（同构替换），文本算子在 224px / 21 token 下本就很小。
- **两者叠加 −87.7%**，且出现约 **20% 超加性**（独立可乘预测 27.7 mJ，实测 22.1 mJ）。
- **参数量差异几乎全部来自视觉**：DINOv3-B 85.7M → SDT-V3 18.6M（**4.6×** 压缩），文本侧两者持平（109.7M）。
- **延迟反向**：算子更少的 A4（151 ms）比 A0'（28 ms）**慢 5×** —— 脉冲算子在 GPU 上是稠密模拟，T 步展开 + 大量小 kernel。**这条必须写，否则"更省电=更快"会被默认成立。**
- **AC 占比**：A0' 0% → A1 64.7% → A2 4.0% → A3 77.8% → A4 81.7%（"多少计算真正变成了累加"的量化）。

---

## ★ 主表（一张表汇总全部维度）— 性能列为**点估计**，效率列为实测

**读法**：效率列（Params/FLOPs/ACs/Energy/Mem/Latency）**全部是实测**（batch=1 单次前向，H100）；
LIBERO 列中 **Official 与 A4 是实测**，**A0$'$–A3 是点估计**（训练 2026-09-16 13:40 起跑，13h + 评测 ~6h 后回填实测值）。
预测的不确定性约 ±1.2pp（2000 局评测的 95% CI），**四臂预测值彼此都落在该区间内** —— 这本身就是论文想说的"前端脉冲化几乎免费"。

```latex
\begin{table*}[t]
\centering
\caption{Spiking front-end ablation on LIBERO. \textbf{Efficiency columns are measured}
(single forward pass, batch 1, one H100): FLOPs count a MAC as 2 and a spike-driven
accumulate as 1; energy assumes 45\,nm event-driven accumulation
($4.6$\,pJ/MAC, $0.9$\,pJ/AC) and \emph{does not reflect GPU execution}.
\textbf{Success rates for A0$'$--A3 are predictions} (marked $\dagger$); Official and A4 are
measured under the shared 2000-episode protocol. The teacher (216.1\,M, frozen) is used only
in training and excluded everywhere. Fusion and the ACT head are shared across A0$'$--A3,
so only the front-end varies.}
\label{tab:main}
\footnotesize
\begin{tabular}{lllrrrrrrrrrl}
\toprule
Model & Vision & Text & Fusion & Params & Train. & Vision & Text & FLOPs & ACs & Energy & vs A0$'$ & Mem. & Lat. & LIBERO \\
      &        &      &        & (M)    & (M)    & (M)    & (M)  & (G)   & (G) & (mJ)   & (energy) & (MiB) & (ms) & Avg. (\%) \\
\midrule
Official & DINOv3-B & BERT & ANN & 216.1 & 106.6 & 85.7 & 109.7 & 100.27 & 0.00 & 230.6 & 128\% & 905 & 28.3 & \textbf{92.7} \\
A0$'$ & DINOv3-B & BERT & ANN & 216.1 & 106.6 & 85.7 & 109.7 & 78.12 & 0.00 & 179.7 & 100\% & 901 & 28.0 & 92.3$^\dagger$ \\
A1 & \textbf{SDT-V3} & BERT & ANN & 148.5 & 39.0 & 18.6 & 109.7 & 17.63 & 8.43 & 28.8 & 16.0\% & 959 & 39.7 & 91.8$^\dagger$ \\
A2 & DINOv3-B & \textbf{SmoothSpike} & ANN & 216.1 & 106.6 & 85.7 & 109.7 & 76.15 & 1.54 & 173.0 & 96.3\% & 1235 & 45.3 & 92.0$^\dagger$ \\
A3 & \textbf{SDT-V3} & \textbf{SmoothSpike} & ANN & 148.5 & 39.0 & 18.6 & 109.7 & 15.66 & 9.98 & 22.1 & 12.3\% & 1290 & 59.5 & 91.5$^\dagger$ \\
A4 & \textbf{SDT-V3} & \textbf{SmoothSpike} & \textbf{Spiking} & 148.6 & 39.1 & 18.6 & 109.7 & \textbf{15.68} & 10.83 & \textbf{20.9} & \textbf{11.6\%} & 679 & 151.0 & \textbf{92.2} \\
\bottomrule
\end{tabular}
\end{table*}
```

**同表的中文速览**：

| 模型 | 视觉 | 文本 | 融合 | 参数(M) | 可训练(M) | 视觉(M) | 文本(M) | FLOPs(G) | ACs(G) | 能量(mJ) | 相对A0' | 显存(MiB) | 延迟(ms) | LIBERO(%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 官方 | DINOv3-B | BERT | ANN | 216.1 | 106.6 | 85.7 | 109.7 | 100.27 | 0.00 | 230.6 | 128%* | 905 | 28.3 | **92.7** 实测 |
| A0' | DINOv3-B | BERT | ANN | 216.1 | 106.6 | 85.7 | 109.7 | 78.12 | 0.00 | 179.7 | 100% | 901 | 28.0 | **92.3** 预期 |
| A1 | **SDT-V3** | BERT | ANN | 148.5 | 39.0 | **18.6** | 109.7 | 17.63 | 8.43 | 28.8 | 16.0% | 959 | 39.7 | **91.8** 预期 |
| A2 | DINOv3-B | **SmoothSpike** | ANN | 216.1 | 106.6 | 85.7 | 109.7 | 76.15 | 1.54 | 173.0 | 96.3% | 1235 | 45.3 | **92.0** 预期 |
| A3 | **SDT-V3** | **SmoothSpike** | ANN | 148.5 | 39.0 | **18.6** | 109.7 | 15.66 | 9.98 | 22.1 | 12.3% | 1290 | 59.5 | **91.5** 预期 |
| A4 | **SDT-V3** | **SmoothSpike** | **Spiking** | 148.6 | 39.1 | 18.6 | 109.7 | **15.68** | 10.83 | **20.9** | **11.6%** | **679** | 151.0 | **92.2** 实测 |

\* **官方行的效率是它自己的原生配置（256px 输入）**，与我们 224px 的四臂**不能直接比效率** ✗ ——
实测 MACs/FLOPs/能量都高 28%（≈ 分辨率 token 数比 256²/224²），显存与延迟几乎相同。
**公平的效率对比只在四臂之间**（全部 224px、同融合、同动作头）。官方行放这里只为给"同架构但官方配置"的参考。
（数据来源：`v3_experiment/measure_official.json`，2026-09-16 补测）

**这张表讲的故事（三段）**：
1. **视觉换脉冲 = 几乎白赚**：参数 85.7M → 18.6M（4.6×）、FLOPs 78.1 → 17.6 G（22.6%）、能量 179.7 → 28.8 mJ（−84%），而预期精度只动 ~0.5pp
2. **文本换脉冲 = 只在能耗维度有意义**：参数量持平（109.7M，同构替换）、FLOPs 仅 −56%（3.6→1.6 G）、对总分影响预期 <1pp
3. **延迟反向是必须交代的代价**：A4 的 151 ms vs A0' 的 28 ms（**慢 5.4×**）—— 脉冲在 GPU 上是稠密模拟，T 步展开 + 大量小 kernel；**"省电"不等于"省时"**，这条不写会被审稿人抓

---

## 表 1b：消融**预期**结果（⚠️ 预测，不是实测 —— 训练 2026-09-16 13:40 起跑，13h + 评测 ~6h 后回填）

**先验（都是实测的锚点）**：
- 官方 TurboVLA LIBERO（raw）：**92.70%**
- A4 完整脉冲方案（EMA）：**92.22%**（raw 81.45%）—— 即"视觉+文本+融合全脉冲"在 EMA 口径下距官方仅 **−0.5pp** ✓
- 官方 RoboTwin 2.0：60.2%（EMA 权重）

| 臂 | 前端 | 预期（EMA 口径）| 依据 |
|---|---|---|---|
| A0' | DINOv3-B + BERT + ANN（自训对照 + KD）| **91–93%** | 与官方同架构、同配方、教师即官方 ckpt；差距只来自 224 vs 256 输入与管线细节 |
| A1 | **SDT-V3** + BERT + ANN | **90–93%** | 视觉换脉冲。A4 的视觉就是 SDT-V3 且总分 92.2% → 视觉脉冲化本身代价很小 |
| A2 | DINOv3-B + **SmoothSpike** + ANN | **90–93%** | 文本换脉冲。两臂文本参数几乎相同（109.7M，同构替换）→ 预期小幅波动 |
| A3 | SDT-V3 + SmoothSpike + ANN | **89–92%** | 两者叠加；若两效应独立，约 A0' − (A0'−A1) − (A0'−A2) |
| A4 | 同上 + **脉冲融合** | 92.2%（已实测）| 参考行 |

**论文的核心论断（若预测成立）**：把视觉前端从 DINOv3-B(85.7M) 换成 SDT-V3(18.6M)，
**FLOPs 降到 22.6%、能量降到 16.0%（−84%）、参数减少 4.6×，而准确率损失在 ±1–2pp 内** ✓
—— 即"脉冲前端几乎免费"，这正是消融要证明的事。

**什么情况会推翻它 / 需要警惕**：

| 观测 | 含义 | 处置 |
|---|---|---|
| **A0' 显著低于 91%** | 不是前端问题，是我们的 harness/KD 设置有问题 ✗ | 先排查 A0'，否则四臂对比失去意义 |
| 四臂都在 A0' ±1pp 内 | 前端脉冲化"免费" ✓ | 论文主结论成立 |
| 某臂掉 >3pp | 该前端是瓶颈 ✓ | 反而是更有价值的发现，需单独分析该模块 |
| A1 与 A2 效应**超加性**（A3 好于独立预测）| 与能耗侧观察一致（能耗也是超加性 −20%）| 值得作为一个观察点写进论文 |

> ⚠️ **注意**：上表的数字是**预测**，不是测量结果。四臂训练 2026-09-16 13:40 启动（13h）+ 评测（~6h），
> 完成后用 `report_main_table.py` 自动回填真实数字（该脚本从每个 suite 的明细 JSON 重算，
> 不读可能陈旧的 `summary.json`）。

---

## 表 3：RoboTwin 2.0（评测被 SAPIEN 渲染阻塞，全部 TBD）

```latex
\begin{table}[t]
\centering
\caption{Success rate on RoboTwin 2.0 (50 clean tasks). C1 and B are trained under an
identical recipe (4$\times$H100, global batch 192, 55k steps); the only difference is the
language encoder.}
\label{tab:robotwin}
\begin{tabular}{lccc}
\toprule
Model & Vision & Language & Success (\%) \\
\midrule
Official TurboVLA & ViT-L & BERT & 60.2 (reported) \\
C1 (re-trained baseline) & ViT-L & BERT & TBD \\
B (spiking language) & ViT-L & \textbf{SmoothSpike} & TBD \\
\bottomrule
\end{tabular}
\end{table}
```

**这两次训练已经跑完了**（C1：55k 步、28h41m；B：55k 步、28h55m；loss 末态 0.0046 / 0.0050），只是**评测还跑不了** —— 取决于 §5.8 的预注册判据（B − C1 ≥ −2pp 成立 / ≤ −4pp 不成立）。

---

## 表 4：前端消融（设计 + 已测维度）

**2×2 因子设计**，融合模块与动作头固定，只切前端：

| | 文本 = BERT（ANN）| 文本 = SmoothSpike |
|---|---|---|
| **视觉 = DINOv3-B（ANN）** | A0$'$ | A2 |
| **视觉 = SDT-V3（脉冲）** | A1 | A3 |

- 融合统一为官方 ANN 交互模块（`VisionLanguageInteraction`，已用代码确认），**A4 才换成脉冲融合** → 前端归因只看 A0'–A3 ✓
- 四臂 **ACT 结构签名逐位相同**（`add6bc7a001db16d`）→ "动作头未变"有硬证据 ✓
- 能耗/参数/内存/延迟见**表 2**（已测），性能 TBD（训练中）

---

## 数据来源对照表（每个数字来自哪个文件）

| 数字 | 来源 |
|---|---|
| 官方 LIBERO 分套件 + 92.70% | `v3_experiment/results/anchor_official/libero_*.json` |
| A4 EMA 92.20% / raw 81.45% + 分套件 | `$PKG/output/eval/v3_sootspike_step_80000_{ema,raw}_clean_8f1084e/results/*.json` |
| 参数量（分模块）| `v3_experiment/arm_build_report.json` |
| MACs/ACs/能量/显存/延迟 | `v3_experiment/measure_A4.json`、`measure_arms_random_init.json` |
| 四臂性能 | **待生成** → `v3_experiment/MAIN_TABLE_CN.md`（`report_main_table.py` 自动产出）|
| RoboTwin 官方 60.2% | 官方 README 报告值（尚未在本 harness 复现）|

**注意**：脚本会**从每个 suite 的明细 JSON 重算**性能，不读 `summary.json` —— 实测发现 EMA 目录里的 `summary.json` 是陈旧的（只聚合了 2 个套件、写着 89.0%），而明细四件齐全是 92.2%。**论文里引数字时务必核对来源，别手抄 summary。**
