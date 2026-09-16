#!/usr/bin/env python
"""补测**官方 LIBERO ckpt**的效率维度:参数/MACs/ACs/FLOPs/能量/显存/延迟。

为什么单独测:官方 ckpt 是 256px 输入(我们的四臂是 224px),视觉 token 数 256 vs 196,
FLOPs 差约 31% —— 直接从 A0' 抄数字是错的 ✗。

复用 measure_arm.py 的算子计数器(按输入张量判定 MAC/AC),按官方 ckpt 自带的
model_config 重建架构(经 build_distillation_teacher,它会把 vision 强制 fp32、
text 强制 bert/frozen —— 与官方发布一致)。

用法(需 GPU):
  PYTHONPATH=<code>:<resources> python measure_official.py --iters 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_build_check import ARMS, build_argv, assert_imported_code  # noqa: E402
from measure_arm import OpAccounting, _install_tracking, rollup, PJ_PER_MAC, PJ_PER_AC  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default="/data/260010028/dwh_vla/v3_experiment/measure_official.json")
    args_cli = ap.parse_args()

    assert_imported_code()
    from turbovla.training.trainer import build_distillation_teacher, build_model_architecture
    from turbovla.training.train_mixed import parse_args_with_mixed_suite_stats as parse_args

    argv = build_argv(ARMS["A3"], text_mode="frozen", kd=True)   # 仅为凑齐 args(教师配置来自 ckpt)
    saved, sys.argv = sys.argv, ["measure_official"] + argv
    try:
        args = parse_args()
    finally:
        sys.argv = saved
    args.allow_hf_download = False

    device = torch.device("cuda")
    model = build_distillation_teacher(args).to(device).eval()
    cfg = model.config
    size = int(cfg.vision.image_size)
    views = int(cfg.vision.num_views)
    print(f"[官方 ckpt 架构] image_size={size} num_views={views} "
          f"| vision={cfg.vision.encoder_type} text={cfg.text.encoder_type} "
          f"downstream={cfg.spike.downstream_type}")

    params_total = sum(p.numel() for p in model.parameters())
    per_module_params = {n: sum(p.numel() for p in c.parameters())
                         for n, c in model.named_children()}

    images = torch.rand(1, views, 3, size, size, device=device)
    state = torch.rand(1, 8, device=device)
    instr = ["put the black bowl on the plate"]

    with torch.no_grad():                      # warmup(msign 等惰性缓存)
        model(instr, images, state, reset_spiking=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    stack = _install_tracking(model)
    acct = OpAccounting(stack)
    with torch.no_grad(), acct:
        out = model(instr, images, state, reset_spiking=True)
    peak_alloc = torch.cuda.max_memory_allocated() / 2**20
    peak_res = torch.cuda.max_memory_reserved() / 2**20

    rolled = rollup(acct.per_module)
    macs = sum(v["macs"] for v in rolled.values())
    acs = sum(v["acs"] for v in rolled.values())
    mdense = sum(v["macs_dense_equiv"] for v in rolled.values())
    flops = 2 * macs + acs
    energy = (macs * PJ_PER_MAC + acs * PJ_PER_AC) / 1e9

    times = []
    for _ in range(args_cli.iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            model(instr, images, state, reset_spiking=True)
        torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1e3)

    rep = {
        "model": "official TurboVLA LIBERO ckpt",
        "image_size": size, "num_views": views,
        "params_total_M": params_total / 1e6,
        "params_per_module_M": {k: v / 1e6 for k, v in per_module_params.items()},
        "macs_G": macs / 1e9, "acs_G": acs / 1e9,
        "flops_G": flops / 1e9, "dense_equiv_G": 2 * mdense / 1e9,
        "energy_mJ": energy,
        "peak_alloc_MiB": round(peak_alloc, 1), "peak_reserved_MiB": round(peak_res, 1),
        "latency_median_ms": round(statistics.median(times), 2),
        "per_module": rolled,
        "action_shape": list(out.shape),
    }
    Path(args_cli.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "per_module"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
