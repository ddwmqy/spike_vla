#!/usr/bin/env python
"""训练**显存峰值**实测:学生 + KD 教师 + AdamW 状态 + EMA 副本 + 反向 + 优化器步。

为什么要单独测:推理峰值(measure_arm.py)远小于训练峰值;而目标机器若只有 32GB
(如 RTX 5090),必须知道 batch 56 是否装得下,而不是靠 40GB/80GB 上的经验外推。

用法(需 GPU):
  python train_mem_probe.py --arm A0p --batch 56 --steps 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_build_check import ARMS, assert_imported_code, build_argv  # noqa: E402


def probe(arm: str, batch: int, steps: int, image_size: int = 224, views: int = 2) -> dict:
    from turbovla.training.trainer import build_distillation_teacher, build_model_architecture
    from turbovla.training.train_mixed import parse_args_with_mixed_suite_stats as parse_args

    argv = build_argv(ARMS[arm], text_mode="frozen", kd=True)
    saved, sys.argv = sys.argv, ["probe"] + argv
    try:
        args = parse_args()
    finally:
        sys.argv = saved
    args.allow_hf_download = False
    if ARMS[arm]["vision"] == "dinov3":
        args.vision_pretrained_checkpoint = None
        args.vision_model_source_path = None

    device = torch.device("cuda")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    student = build_model_architecture(args).to(device).train()
    teacher = build_distillation_teacher(args).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=1e-10)
    # 训练器还维护一份 EMA 权重(全模型副本)
    ema = {k: v.detach().clone() for k, v in student.state_dict().items()}

    n_train = sum(p.numel() for p in trainable)
    static_gb = (
        sum(p.numel() * p.element_size() for p in student.parameters())   # 学生权重
        + sum(p.numel() * 4 for p in trainable) * 3                       # 梯度 + AdamW (m,v)
        + sum(v.numel() * v.element_size() for v in ema.values())         # EMA
        + sum(p.numel() * p.element_size() for p in teacher.parameters()) # 教师
    ) / 1e9
    after_build = torch.cuda.max_memory_allocated() / 1e9

    images = torch.rand(batch, views, 3, image_size, image_size, device=device)
    state = torch.rand(batch, 8, device=device)
    instr = ["put the black bowl on the plate"] * batch

    for i in range(steps):
        # 与 trainer.py:1265 的真实 KD 路径一致:双方都取视觉 token 做 MSE
        with torch.no_grad():
            t_actions, t_tokens = teacher(
                instr, {"pixel_values": images}, state, return_visual_tokens=True
            )
        pred, s_tokens = student(instr, images, state, return_visual_tokens=True)
        if s_tokens.shape != t_tokens.shape:
            raise ValueError(f"student/teacher token mismatch: {s_tokens.shape} vs {t_tokens.shape}")
        loss = (
            0.5 * (pred.float() - t_actions.float()).pow(2).mean()
            + 0.25 * torch.nn.functional.mse_loss(s_tokens.float(), t_tokens.float())
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        # EMA 更新(与训练器同形式的就地 lerp,不额外占显存)
        with torch.no_grad():
            for k, v in student.state_dict().items():
                if k in ema and ema[k].dtype.is_floating_point:
                    ema[k].lerp_(v.detach(), 0.001)

    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return {
        "arm": arm, "batch": batch, "steps": steps, "gpu_total_GB": round(total, 1),
        "trainable_params_M": n_train / 1e6,
        "static_estimate_GB": round(static_gb, 2),
        "after_build_GB": round(after_build, 2),
        "peak_alloc_GB": round(peak_alloc, 2),
        "peak_reserved_GB": round(peak_reserved, 2),
        "fits_32GB": bool(peak_reserved < 32 * 0.95),
        "fits_40GB": bool(peak_reserved < 40 * 0.95),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A0p")
    ap.add_argument("--batch", default="56", help="逗号分隔可测多档,如 56,32")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--views", type=int, default=2, help="LIBERO 臂固定 2")
    args = ap.parse_args()
    assert_imported_code()
    if not torch.cuda.is_available():
        raise SystemExit("[FATAL] 需要 GPU")
    print(f"[env] {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    for b in [x.strip() for x in str(args.batch).split(",")]:
        r = probe(args.arm, int(b), args.steps, views=args.views)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        del_v = "✅ 32GB 可跑" if r["fits_32GB"] else ("⚠️ 32GB 不行,40GB 可跑" if r["fits_40GB"] else "❌ 40GB 也不够")
        print(f"→ {del_v}（峰值保留 {r['peak_reserved_GB']} GB / 卡总量 {r['gpu_total_GB']} GB）")
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
