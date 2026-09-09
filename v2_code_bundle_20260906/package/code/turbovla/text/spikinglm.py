"""Adapter for the official hamings1/SpikingLM BERT backbone."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig


def _resolve_source_root(source_path: str) -> Path:
    path = Path(source_path).expanduser().resolve()
    if path.is_file():
        if path.name != "modeling_spiking_bert.py":
            raise ValueError(f"SpikingLM source file must be modeling_spiking_bert.py, got: {path}")
        path = path.parent.parent
    module_file = path / "spiking_bert" / "modeling_spiking_bert.py"
    if not module_file.is_file():
        raise FileNotFoundError(
            f"official SpikingLM source not found: {module_file}. "
            "Clone https://github.com/hamings1/SpikingLM into the configured source directory."
        )
    return path


def _import_official_model(source_path: str):
    source_root = _resolve_source_root(source_path)
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        module = importlib.import_module("spiking_bert.modeling_spiking_bert")
    except Exception as error:
        raise RuntimeError(
            "failed to import the official SpikingLM implementation; install its "
            "transformers/spikingjelly/CuPy dependencies first"
        ) from error
    loaded_file = Path(module.__file__).resolve()
    expected_file = source_root / "spiking_bert" / "modeling_spiking_bert.py"
    if loaded_file != expected_file.resolve():
        raise RuntimeError(f"loaded SpikingLM from {loaded_file}, expected {expected_file}")
    return module.BertModel


def load_spikinglm_backbone(
    checkpoint_path: str,
    source_path: str,
    time_steps: int,
    local_files_only: bool,
):
    """Build BertModel and strictly load the released MLM backbone weights."""
    model_cls = _import_official_model(source_path)
    config = AutoConfig.from_pretrained(checkpoint_path, local_files_only=local_files_only)
    config.T = int(time_steps)
    config._attn_implementation = "eager"
    model = model_cls(config, add_pooling_layer=False)

    weight_path = Path(checkpoint_path) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"SpikingLM checkpoint is missing model.safetensors: {weight_path}")
    released_state = load_file(str(weight_path), device="cpu")
    expected_state = model.state_dict()
    backbone_state = {}
    unexpected_backbone = []
    for key, value in released_state.items():
        candidate = key[5:] if key.startswith("bert.") else key
        if candidate in expected_state:
            if expected_state[candidate].shape != value.shape:
                raise RuntimeError(
                    f"SpikingLM shape mismatch for {candidate}: "
                    f"expected {tuple(expected_state[candidate].shape)}, got {tuple(value.shape)}"
                )
            backbone_state[candidate] = value
        elif key.startswith("bert."):
            unexpected_backbone.append(key)
    missing = sorted(set(expected_state) - set(backbone_state))
    if missing or unexpected_backbone:
        raise RuntimeError(
            "SpikingLM backbone checkpoint is not strictly compatible: "
            f"missing={missing[:20]}, unexpected={unexpected_backbone[:20]}"
        )
    model.load_state_dict(backbone_state, strict=True)
    model.config._name_or_path = os.fspath(checkpoint_path)
    return model


def reset_spiking_state(module: torch.nn.Module) -> None:
    try:
        from spikingjelly.activation_based import functional
    except ImportError as error:
        raise RuntimeError("SpikingLM requires spikingjelly") from error
    functional.reset_net(module)
