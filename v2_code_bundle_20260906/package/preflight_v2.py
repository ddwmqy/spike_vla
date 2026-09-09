#!/usr/bin/env python3
"""CPU pre-flight for the v2 run (no training, no data).

Part A (always): build the v2 student from scratch (SDT-V3 + online SpikingLM +
SNN fusion + learned readout + ANN action), mirroring the official recipe of
pretrained backbones trained from scratch. Verifies SpikingLM loads, the text
encoder is trainable (online), the SDT-V3 classification weights load into the
vision encoder, and reports the freshly initialized SNN fusion parameter count.
(No student warm-start: decided 2026-09-05 to match the official from-scratch
initialization.)

Part B (optional): pass the official checkpoint path as argv[1] to also build the
KD teacher exactly as build_distillation_teacher would. A strict-load failure
here prints the missing/unexpected keys needed to write the converter.

Usage:
    python preflight_v2.py
    python preflight_v2.py /path/to/official/libero.pth
"""

from __future__ import annotations

import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.environ.get("REPO_DIR", "/home/dwh/work/vla/libero_vla/resources")

sys.path.insert(0, os.path.join(PACKAGE_ROOT, "code"))
sys.path.insert(0, REPO_DIR)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from turbovla.training import trainer  # noqa: E402  (import before train_mixed keeps the original parser)


def build_v2_args():
    argv_backup = sys.argv
    sys.argv = [
        "preflight",
        "--dataset_dir", f"{REPO_DIR}/data/libero/libero_10_no_noops/1.0.0",
        "--text_layout_path", f"{REPO_DIR}/experiments/libero/configs/online_text_layout.json",
        "--dinov3_path", f"{REPO_DIR}/pretrained/dinov3-vitb16-pretrain-lvd1689m",
        "--vision_encoder_type", "sdtv3_19m",
        "--vision_model_path", f"{REPO_DIR}/pretrained/V3_19.0M_1x4.pth",
        "--vision_pretrained_checkpoint", f"{REPO_DIR}/pretrained/V3_19.0M_1x4.pth",
        "--vision_model_source_path",
        f"{REPO_DIR}/third_party/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Base/models.py",
        "--vision_image_size", "224",
        "--vision_output_grid_size", "14",
        "--bert_path", f"{REPO_DIR}/pretrained/SpikingLM",
        "--text_encoder_type", "spikinglm",
        "--spikinglm_source_path", f"{REPO_DIR}/third_party/SpikingLM",
        "--spikinglm_time_steps", "4",
        "--train_text_encoder",
        "--downstream_type", "spiking",
        "--temporal_readout", "learned",
        "--action_head_type", "ann",
        "--precision", "fp32",
        "--no_allow_hf_download",
    ]
    try:
        args = trainer.parse_args()
    finally:
        sys.argv = argv_backup
    # mixed-suite pre-parser arguments (not part of trainer.parse_args)
    args.dataset_dirs = ",".join(
        f"{REPO_DIR}/data/libero/{suite}_no_noops/1.0.0"
        for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial")
    )
    args.stats_path = f"{REPO_DIR}/experiments/libero/configs/libero_all4_stats.json"
    args.stats_key = "libero_all4_no_noops"
    args.dinov3_precision = "fp32"
    args.normalize_binary_gripper = "auto"
    return args


def part_a() -> None:
    print("=== PART A: build v2 student from scratch (no student warm-start) ===", flush=True)
    args = build_v2_args()
    model = trainer.build_model_architecture(args)
    named = dict(model.named_parameters())
    fusion = {name for name in named if name.startswith("vision_language_interaction")}
    text_trainable = [
        name for name in named if name.startswith("text_encoder") and named[name].requires_grad
    ]
    if not text_trainable:
        raise SystemExit("FAIL: online text mode requested but no text_encoder parameter is trainable")
    frozen_text = [name for name in named if name.startswith("text_encoder") and not named[name].requires_grad]
    if frozen_text:
        raise SystemExit(f"FAIL: {len(frozen_text)} text_encoder parameters are frozen in online mode")
    n_state = sum(1 for _ in model.state_dict())
    n_fusion = sum(named[name].numel() for name in fusion)
    print(f"student built OK: {n_state} state tensors")
    print(f"text encoder online: {len(text_trainable)} trainable tensors, none frozen")
    print(f"SNN fusion freshly initialized: {len(fusion)} tensors / {n_fusion} params")
    del model


def part_b(official_ckpt: str) -> None:
    print("=== PART B: build KD teacher from the official checkpoint ===", flush=True)
    args = build_v2_args()
    args.teacher_checkpoint = official_ckpt
    args.teacher_weight_source = os.environ.get("TEACHER_WEIGHT_SOURCE", "model")
    args.teacher_bert_path = f"{REPO_DIR}/pretrained/bert-base-uncased"
    args.distill_feature_weight = 0.25
    args.distill_action_weight = 0.5
    try:
        teacher = trainer.build_distillation_teacher(args)
    except FileNotFoundError as error:
        raise SystemExit(f"FAIL: {error}")
    except RuntimeError as error:
        message = str(error)
        if "not strictly compatible" not in message:
            raise
        print("STRICT-LOAD MISMATCH (converter needed). Detail:")
        print(message)
        raise SystemExit(1)
    n_params = sum(parameter.numel() for parameter in teacher.parameters())
    print(f"teacher built OK: {n_params / 1e6:.1f}M parameters, frozen={all(not p.requires_grad for p in teacher.parameters())}")


def main() -> None:
    part_a()
    if len(sys.argv) > 1:
        part_b(sys.argv[1])
    else:
        print("PART B skipped: pass the official checkpoint path once downloaded.")


if __name__ == "__main__":
    main()
