"""Adapter for the sootspike text backbone (v3 experiment).

"sootspike" = SmoothSpike BERT: the spike-driven BERT of
``CayleyZ/SmoothSpike`` (SpikingBERT backbone + learnable orthogonal
SmoothSpike transforms). We run the *fused* inference model
(``spikingbert_rot_inf.py``): per-layer H2/H3 rotations folded into the
linear weights, global ``bert.H1`` kept. Integration patches (relative
imports, optional fast-Hadamard) come from the reference repo
``wxqnl/spike-turbovla`` (``third_party/patches/integration.patch``).

Weights layout expected at ``checkpoint_path`` (see V3 doc):
- ``config.json`` (BertConfig incl. ``T``), ``model.safetensors`` (fused)
- tokenizer files (``tokenizer_config.json``/``vocab.txt``) - the shared
  text encoder builds its ``AutoTokenizer`` from this same directory
- generated from the ModelScope unfused release via
  ``third_party/SmoothSpike/convertor.py`` (H2/H3 fusion, "Removed 24 keys")

Contract (mirrors turbovla/text/spikinglm.py, consumed by
turbovla/models/text_encoder.py): returns an ``nn.Module`` with
``config.hidden_size`` and a BertModel-style forward accepting
``input_ids``/``attention_mask``/``position_ids`` and returning
``.last_hidden_state`` as ``[B, L, D]``. Unlike SpikingLM, SmoothSpike keeps
no cross-forward spiking state, so ``reset_sootspike_state`` is a no-op.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
from transformers import BertConfig

_TOKENIZER_MARKERS = ("tokenizer_config.json", "vocab.txt", "tokenizer.json")
_WEIGHT_MARKERS = ("model.safetensors", "pytorch_model.bin", "model.bin")


def _import_runtime(source_path: str):
    """Import the patched SmoothSpike runtime as the package
    ``third_party.SmoothSpike.spikingbert_rot_inf`` (relative imports inside
    the package require package context; REPO_DIR carries ``third_party/``)."""
    source_root = Path(source_path).expanduser().resolve()
    module_file = source_root / "spikingbert_rot_inf.py"
    if not module_file.is_file():
        raise FileNotFoundError(
            f"sootspike runtime not found: {module_file}. "
            "Provision third_party/SmoothSpike per the V3 doc (upstream CayleyZ/SmoothSpike "
            "+ integration.patch from wxqnl/spike-turbovla)."
        )
    package_root = source_root.parent.parent  # .../resources (holds third_party/)
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    try:
        module = importlib.import_module("third_party.SmoothSpike.spikingbert_rot_inf")
    except Exception as error:
        raise RuntimeError(
            "failed to import the SmoothSpike runtime; check spikingjelly and the "
            "integration patch (see V3 doc)"
        ) from error
    loaded_file = Path(module.__file__).resolve()
    if loaded_file != module_file.resolve():
        raise RuntimeError(
            f"imported SmoothSpike runtime from {loaded_file}, expected {module_file}"
        )
    return module.BertForMaskedLM


def load_sootspike_backbone(
    checkpoint_path: str,
    source_path: str,
    time_steps: int,
    local_files_only: bool,
):
    """Build the fused SmoothSpike BERT and strictly load its released weights."""
    model_dir = Path(checkpoint_path).expanduser().resolve()
    config_path = model_dir / "config.json"
    weight_path = model_dir / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"sootspike config not found: {config_path}")
    if not weight_path.is_file():
        raise FileNotFoundError(f"sootspike fused checkpoint not found: {weight_path}")
    if not any((model_dir / name).is_file() for name in _TOKENIZER_MARKERS):
        raise FileNotFoundError(
            f"sootspike weights directory has no tokenizer files ({', '.join(_TOKENIZER_MARKERS)}): "
            f"{model_dir}. Copy the matching tokenizer files into it (see V3 doc)."
        )
    model_cls = _import_runtime(source_path)

    config = BertConfig.from_pretrained(str(model_dir), local_files_only=local_files_only)
    published_timesteps = int(getattr(config, "T", time_steps))
    if published_timesteps != int(time_steps):
        raise ValueError(
            f"sootspike checkpoint was trained with T={published_timesteps}, "
            f"but text time_steps={time_steps}"
        )
    config.T = int(time_steps)
    config._attn_implementation = "eager"

    from safetensors.torch import load_file

    full_model = model_cls(config)
    state = load_file(str(weight_path), device="cpu")
    if any(".H2" in key or ".H3" in key for key in state):
        raise ValueError(
            "sootspike runtime expects the fused checkpoint; the supplied file still "
            "contains per-layer H2/H3 tensors - run third_party/SmoothSpike/convertor.py first"
        )

    missing_backbone = sorted(
        key for key in full_model.bert.state_dict() if f"bert.{key}" not in state
    )
    if missing_backbone:
        raise RuntimeError(
            f"sootspike checkpoint is missing backbone tensors: {missing_backbone[:20]}"
        )

    missing, unexpected = full_model.load_state_dict(state, strict=False)
    allowed_tied_aliases = {
        "cls.predictions.decoder.weight",
        "cls.predictions.decoder.bias",
    }
    disallowed_missing = sorted(set(missing) - allowed_tied_aliases)
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "sootspike checkpoint is not strictly compatible: "
            f"missing={disallowed_missing[:20]}, unexpected={unexpected[:20]}"
        )
    full_model.tie_weights()

    backbone = full_model.bert
    backbone.load_report = {
        "checkpoint_tensors": len(state),
        "missing_tied_aliases": sorted(set(missing) & allowed_tied_aliases),
        "unexpected": list(unexpected),
    }
    return backbone


def reset_sootspike_state(module: torch.nn.Module) -> None:
    """No-op: the fused SmoothSpike runtime keeps no cross-forward state
    (spikingjelly reset is only needed for stateful neurons; the reference
    wxqnl/spike-turbovla integration calls no reset around SmoothSpike)."""
    del module
