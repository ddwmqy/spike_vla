# v2 正式训练(④b)最小包 — 算力服务器部署说明

打包 2026-09-06。**原则:能下载的都在服务器重新下载,包里只放下载不到的。**
全量 12G 打包(含数据)仍在源机 `v2_train_bundle_20260906.tar.zst` 备用。

## 1. 包内容(966M)

```
v2_code_bundle_20260906/
├── package/                  # 代码 + 启动脚本 + 文档(~1M)
│   ├── run_train_v2.sh       # ④b 启动器(TEXT_MODE 门禁、warmup 自动钳制)
│   ├── preflight_v2.py / convert_official_ckpt.py / evaluate.py / run_eval*.sh
│   ├── code/turbovla/ + third_party/vla_adapter/
│   ├── V2_PLAN_CN.md / V2_FLOW_AND_COMPARISON_CN.md   # 先读 V2_PLAN §4/§6.1
│   └── CLUSTER_SETUP_CN.md(本文件)
├── resources/                # 只含"下载不到/受限"的资产(966M)
│   ├── pretrained/
│   │   ├── dinov3-vitb16-pretrain-lvd1689m/   (327M;HF 上是 gated 模型,故随包)
│   │   ├── SpikingLM/                          (419M;自研预训练,不公开)
│   │   └── V3_19.0M_1x4.pth                    (219M;SDT-V3 分类 ckpt,公开渠道不确定,随包)
│   ├── third_party/{SpikingLM, Spike-Driven-Transformer-V3/.../models.py}
│   └── experiments/libero/configs/{libero_all4_stats.json, online_text_layout.json}
└── environment/{pip-freeze.txt, requirements-install.txt, pyvenv.cfg}
```

## 2. 服务器端需要下载的三样(全部公开)

```bash
pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com        # 服务器可直连 HF 则去掉
B=/path/to/v2_code_bundle_20260906

# ① RLDS 训练数据(9.8G;已验证与源机数据同源:shard 字节数/数量逐一致)
hf download openvla/modified_libero_rlds --repo-type dataset \
  --local-dir $B/resources/data/libero

# ② 官方 TurboVLA 发布(取其中的 checkpoints/libero/turbovla_libero.pth)
hf download H-EmbodVis/TurboVLA --local-dir $B/resources/pretrained/TurboVLA

# ③ BERT(KD teacher 的文本编码器)
hf download bert-base-uncased --local-dir $B/resources/pretrained/bert-base-uncased
```

**下载后校验**(数据转错版本会静默改变训练结果,务必跑):

```bash
cd $B/resources/data/libero
for s in libero_spatial:16 libero_goal:16 libero_object:32 libero_10:32; do
  suite=${s%%:*}; n=${s##*:}; cnt=$(ls $suite/1.0.0/ | grep -c 'tfrecord-'); \
  echo "$suite shards=$cnt (期望 $n)"; done
# 另抽一个 shard 比字节数:libero_spatial 00000-of-00016 = 118202941 字节
```

## 3. 环境搭建

```bash
conda create -n turbovla python=3.10 -y && conda activate turbovla
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r $B/environment/requirements-install.txt   # 已剔除 torch/torchvision
python -c "import torch,numpy,tensorflow,tensorflow_datasets,cupy,transformers,timm; \
  print(torch.__version__, numpy.__version__)"
# 期望:2.6.0+cu124 1.26.4
```

- **numpy 必须 <2(1.26.4)**:tf 2.15.1 与 numpy 2.x 不兼容,requirements 已钉死。
- 驱动需支持 CUDA 12.4(torch 2.6.0+cu124)。
- 训练不需要 MuJoCo/LIBERO 仿真环境(那是评测 ⑤ 的事)。

## 4. 预检 + 启动

```bash
cd $B/package
python preflight_v2.py $B/resources/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth
# 期望:Part A "student built OK: 1421 state tensors"
#       Part B "teacher built OK: 216.1M parameters, frozen=True"

GPU_IDS=0,1,2,3 TEXT_MODE=frozen BATCH_SIZE=16 TARGET_GLOBAL_BATCH=256 \
  REPO_DIR=$B/resources PYTHON_BIN=$(which python) TORCHRUN_BIN=$(which torchrun) \
  OFFICIAL_CKPT=$B/resources/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth \
  nohup bash run_train_v2.sh > train_4b.log 2>&1 &
```

- **TEXT_MODE=frozen 是定案,别改**(官方 ckpt 实测 text.frozen=True;传错会被门禁拒启)。
- **BATCH_SIZE**:4 卡全空(单卡空闲 ≥22GB)用 16(官方档位,fp32 batch16 ≈ 21.5GB/卡);
  有 ≥2GB 共存进程改 8(全局 batch 仍 256,优化数学不变,慢 ~1.5 天)。
- 80k 步,4×4090 基准 batch16 ≈ 3 天;更快卡按比例。
- 产物:`package/output/train_v2/checkpoints/turbovla_v2_snnfusion_*.pth` + `logs/*.log`。
  跑完拷回 80000 档(建议连 50000/60000/70000 中间档一起)做 ⑤ 评测。

## 5. 跑完后

把 `output/train_v2/` 拷回源机(或通知我),⑤ clean 评测(WEIGHT_SOURCE=raw 主对比)在源机跑,
与 ③ 官方锚点(进行中)做 ⑥ 对比。
