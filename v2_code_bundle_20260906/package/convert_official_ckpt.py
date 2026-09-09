#!/usr/bin/env python3
"""Convert an official TurboVLA LIBERO checkpoint into this package's eval format.

The official release (H-EmbodVis/TurboVLA) stores checkpoints in its own layout.
Our evaluate.py expects {model_state_dict, model_config} where model_config
describes the official architecture (DINOv3 ViT-B + BERT + ANN fusion).

The converter:
  1. loads the official file (torch .pth or .safetensors),
  2. extracts the model state dict (model_state_dict / model / state_dict /
     ema_model_state_dict, or a raw tensor mapping),
  3. builds the official model from the config below and strict-loads the state;
     any missing/unexpected key aborts with a diagnostic listing,
  4. writes the export for evaluate.py.

Usage:
    python convert_official_ckpt.py --src <official.pth> --dst <export.pth>
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.environ.get("REPO_DIR", "/home/dwh/work/vla/libero_vla/resources")

sys.path.insert(0, os.path.join(PACKAGE_ROOT, "code"))
sys.path.insert(0, REPO_DIR)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def official_model_config() -> dict:
    """Official LIBERO architecture, mirroring policy._make_model_args defaults."""
    return {
        "name": "TurboVLA",
        "text": {
            "encoder_type": "bert",
            "model_name_or_path": f"{REPO_DIR}/pretrained/bert-base-uncased",
            "model_source_path": None,
            "time_steps": 4,
            "return_temporal_hidden": False,
            "max_length": 256,
            "padding_length": 21,
            "sub_sentence_present": True,
            "frozen": True,
            "force_eval_when_frozen": True,
            "zero_padded_tokens": False,
            "local_files_only": True,
            "attention_implementation": None,
        },
        "vision": {
            "encoder_type": "dinov3",
            "model_name_or_path": f"{REPO_DIR}/pretrained/dinov3-vitb16-pretrain-lvd1689m",
            "image_size": 256,
            "num_views": 2,
            "position_embedding": "view",
            "encode_views_separately": True,
            "frozen": True,
            "local_files_only": True,
            "attention_implementation": None,
            "compute_precision": "fp32",
        },
        "interaction": {
            "hidden_dim": 256,
            "nheads": 8,
            "num_layers": 6,
            "dim_feedforward": 2048,
            "enhancer_inner_dim": 1024,
            "text_dropout": 0.0,
            "fusion_dropout": 0.0,
            "fusion_droppath": 0.1,
            "padding_strategy": "key_padding_mask",
            "residual_style": "normalized",
            "attention_backend": "manual",
            "compute_precision": "fp32",
        },
        "action": {
            "action_dim": 7,
            "state_dim": 8,
            "horizon": 12,
            "num_state_tokens": 2,
            "num_layers": 3,
            "mlp_hidden_dim": 512,
            "state_hidden_dim": 256,
            "dropout": 0.1,
            "decoder_type": "ann",
        },
        "spike": {
            "downstream_type": "ann",
            "time_steps": 4,
            "tau": 2.0,
            "threshold": 1.0,
            "surrogate_alpha": 4.0,
            "detach_reset": True,
            "backend": "cupy",
            "reset_policy": "per_forward",
        },
    }


def load_state(src: str, weight_source: str = "auto") -> tuple[dict, dict, str]:
    """Return (state_dict, checkpoint_metadata, chosen_source) from a .pth or .safetensors file.

    weight_source: "auto" prefers model_state_dict (the official evaluate.py
    convention, turbovla/evaluation/policy.py), "ema" prefers
    ema_model_state_dict when the release ships it.
    """
    if src.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(src), {}, "raw"
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"unsupported checkpoint payload type: {type(checkpoint)}")
    preference = ("model_state_dict", "model", "state_dict")
    if weight_source == "ema":
        # No silent fallback: requesting EMA and silently evaluating raw would
        # be a hidden weight-source mismatch in the comparison.
        if not (isinstance(checkpoint.get("ema_model_state_dict"), dict) and checkpoint["ema_model_state_dict"]):
            raise SystemExit(
                "--weight_source ema was requested but the checkpoint has no "
                "ema_model_state_dict; refusing to fall back to raw weights "
                "(use --weight_source model explicitly)"
            )
        preference = ("ema_model_state_dict",)
    for key in preference:
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            return value, checkpoint, key
    if all(hasattr(value, "shape") for value in checkpoint.values()):
        return checkpoint, {}, "raw"
    raise SystemExit(
        "could not locate a state dict in the checkpoint; top-level keys: "
        + ", ".join(checkpoint.keys())
    )


def main() -> None:
    global torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="official TurboVLA LIBERO checkpoint")
    parser.add_argument("--dst", required=True, help="output path in this package's eval format")
    parser.add_argument("--weight_source", default="model", choices=["model", "ema"])
    args = parser.parse_args()

    state, checkpoint, chosen_source = load_state(args.src, args.weight_source)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    print(f"loaded {len(state)} tensors from {args.src} (weight source: {chosen_source})")
    if isinstance(checkpoint, dict):
        stored_config = checkpoint.get("model_config")
        if isinstance(stored_config, dict):
            text_cfg = stored_config.get("text") or {}
            vision_cfg = stored_config.get("vision") or {}
            print("ckpt self-report: text.frozen =", text_cfg.get("frozen", "n/a"),
                  "| text.encoder_type =", text_cfg.get("encoder_type", "n/a"))
            print("ckpt self-report: vision.compute_precision =", vision_cfg.get("compute_precision", "n/a"))
        for meta_key in ("args", "training_args", "train_args"):
            meta = checkpoint.get(meta_key)
            if isinstance(meta, dict):
                for arg_key in ("dataset_dirs", "freeze_text_encoder", "shuffle_steps_within_episode", "max_steps"):
                    if arg_key in meta:
                        print(f"ckpt {meta_key}.{arg_key} = {meta[arg_key]}")

    from turbovla.models.configuration import TurboVLAConfig
    from turbovla.models.turbovla import build_turbovla

    # Prefer the checkpoint's own model_config: the strict load then verifies
    # keys/shapes against the architecture the release was actually trained
    # with, and it matches what evaluation does (TurboVLAPolicy rebuilds from
    # model_config too, overriding paths/precision from CLI). The hardcoded
    # config below is only a fallback for pathless/raw releases.
    stored_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    if isinstance(stored_config, dict) and stored_config:
        config = TurboVLAConfig.from_mapping(stored_config)
        print("model config source: checkpoint model_config (paths/precision are overridden by the evaluator)")
    else:
        config = TurboVLAConfig.from_mapping(official_model_config())
        print("model config source: hardcoded official_model_config (checkpoint carried no usable model_config)")
    # The release records training-machine resource names ("bert-base-uncased",
    # "facebook/dinov3-...") that would trigger a hub download. Point them at
    # the local mirrors for construction — the same path override the evaluator
    # applies (policy.py overrides text/vision paths from CLI before rebuild).
    # Architecture fields verified by the strict load stay from the checkpoint.
    if getattr(config.text, "encoder_type", "bert") == "bert":
        config.text.model_name_or_path = f"{REPO_DIR}/pretrained/bert-base-uncased"
        config.text.local_files_only = True
    if getattr(config.vision, "encoder_type", "dinov3") == "dinov3":
        config.vision.model_name_or_path = f"{REPO_DIR}/pretrained/dinov3-vitb16-pretrain-lvd1689m"
        config.vision.local_files_only = True
    print("local construction overrides: text ->", config.text.model_name_or_path,
          "| vision ->", config.vision.model_name_or_path)
    model = build_turbovla(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("STRICT-LOAD MISMATCH — converter needs a key mapping before this")
        print("checkpoint can be used. missing:", len(missing))
        for key in missing[:30]:
            print("  missing:", key)
        print("unexpected:", len(unexpected))
        for key in unexpected[:30]:
            print("  unexpected:", key)
        raise SystemExit(2)

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    payload = {
        "model_state_dict": state,
        "selected_weight_source": chosen_source,
        "model_config": config.to_dict(),
    }
    for key in ("global_step", "loss", "ema_decay"):
        if key in checkpoint:
            payload[key] = checkpoint[key]
    temporary = args.dst + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, args.dst)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"OK: {total / 1e6:.1f}M parameters, strict load clean -> {args.dst}")


if __name__ == "__main__":
    main()
