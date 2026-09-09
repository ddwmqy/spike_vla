import argparse
import glob
import json
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.libero_rlds import (
    LiberoRLDSDataset,
    vla_collate_fn,
)
from ..models.turbovla import (
    build_turbovla,
)
from ..models.components.spiking import (
    clear_spike_activity_,
    restore_spiking_state_,
    set_hardware_qat_,
    set_spike_diagnostics_,
    snapshot_spiking_state,
    spike_activity_regularization,
    spike_diagnostics_vector,
    spike_membrane_statistics,
)


class DummyArgs:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the complete TurboVLA model with online BERT and DINOv3."
    )

    parser.add_argument("--dataset_dir", type=str, default="./data/libero/libero_10_no_noops/1.0.0")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dinov3_path", type=str, required=True)
    parser.add_argument("--vision_encoder_type", type=str, default="dinov3", choices=["dinov3", "sdtv3_19m"])
    parser.add_argument("--vision_model_path", type=str, default=None)
    parser.add_argument("--vision_pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--vision_model_source_path", type=str, default=None)
    parser.add_argument("--vision_image_size", type=int, default=None)
    parser.add_argument("--vision_output_grid_size", type=int, default=14)
    parser.add_argument("--bert_path", type=str, required=True)
    parser.add_argument("--text_encoder_type", type=str, default="bert", choices=["bert", "spikinglm", "sootspike"])
    parser.add_argument("--spikinglm_source_path", type=str, default=None)
    parser.add_argument("--spikinglm_time_steps", type=int, default=4)
    # v3: source directory of the sootspike model implementation (see turbovla/text/sootspike.py)
    parser.add_argument("--text_source_path", type=str, default=None)
    parser.add_argument("--downstream_type", type=str, default="ann", choices=["ann", "spiking"])
    parser.add_argument("--downstream_time_steps", type=int, default=4)
    parser.add_argument("--downstream_lif_tau", type=float, default=2.0)
    parser.add_argument("--downstream_lif_backend", type=str, default="cupy", choices=["torch", "cupy"])
    parser.add_argument("--spike_lif_threshold", type=float, default=1.0)
    parser.add_argument("--spike_surrogate_alpha", type=float, default=4.0)
    parser.add_argument("--spike_detach_reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spike_reset_policy", type=str, default="per_forward", choices=["per_forward", "manual"])
    parser.add_argument("--spike_attention_normalization", type=str, default="pow2_nearest", choices=["pow2_nearest", "pow2_ceil", "exact", "none"])
    parser.add_argument("--spike_attention_max_exponent", type=int, default=12)
    parser.add_argument("--spike_attention_reference_offset", type=int, default=1)
    parser.add_argument("--spike_cross_score_shift", type=int, default=0)
    parser.add_argument("--spike_self_score_shift", type=int, default=0)
    parser.add_argument("--spike_action_score_shift", type=int, default=0)
    parser.add_argument("--spike_quantize_qkv_scales", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--spike_quantize_attention_scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--spike_shared_cross_affinity", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--spike_enable_visual_self_attention", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--spike_enable_visual_ffn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spike_collect_diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--spike_firing_rate_weight", type=float, default=0.0)
    parser.add_argument("--spike_target_firing_rate", type=float, default=0.2)
    parser.add_argument("--spike_diagnostics_freq", type=int, default=100)
    parser.add_argument(
        "--spike_eval_freq",
        type=int,
        default=5000,
        help="Run a no-grad SNN health probe every N optimizer steps; 0 disables it.",
    )
    parser.add_argument("--spike_qat_start_step", type=int, default=-1)
    parser.add_argument("--spike_cross_attention_dim", type=int, default=512)
    parser.add_argument("--spike_cross_attention_heads", type=int, default=8)
    parser.add_argument("--spike_cross_layer_scale_init", type=float, default=1.0e-2)
    parser.add_argument("--spike_self_layer_scale_init", type=float, default=1.0e-1)
    parser.add_argument("--spike_ffn_layer_scale_init", type=float, default=1.0e-1)
    parser.add_argument("--temporal_readout", type=str, default="mean", choices=["mean", "last", "learned"])
    parser.add_argument(
        "--teacher_bert_path",
        type=str,
        default=None,
        help="ANN BERT path used only by the distillation teacher; defaults to --bert_path.",
    )
    parser.add_argument(
        "--pretrained_init_ckpt",
        type=str,
        default=None,
    )
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--checkpoint_prefix", type=str, default="turbovla_step")
    parser.add_argument("--resume_mode", type=str, default="none", choices=["none", "model", "all"])

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--dinov3_lr", type=float, default=5e-5)
    parser.add_argument("--head_weight_decay", type=float, default=1e-10)
    parser.add_argument("--dinov3_weight_decay", type=float, default=1e-10)
    parser.add_argument("--vision_encoder_lr", type=float, default=None)
    parser.add_argument("--vision_encoder_weight_decay", type=float, default=None)
    parser.add_argument("--vision_precision", type=str, default=None, choices=["fp32", "bf16", "bf16_autocast"])
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--teacher_weight_source", type=str, default="model", choices=["model", "ema"])
    parser.add_argument("--student_init_checkpoint", type=str, default=None)
    parser.add_argument("--student_init_weight_source", type=str, default="ema", choices=["model", "ema"])
    parser.add_argument("--distill_feature_weight", type=float, default=0.0)
    parser.add_argument("--distill_action_weight", type=float, default=0.0)
    parser.add_argument("--horizon_first_weight", type=float, default=1.0)
    parser.add_argument("--horizon_last_weight", type=float, default=1.0)
    parser.add_argument("--gripper_loss_weight", type=float, default=0.0)
    parser.add_argument("--gripper_sign_temperature", type=float, default=0.25)
    parser.add_argument("--action_delta_weight", type=float, default=0.0)
    parser.add_argument("--action_diagnostics_freq", type=int, default=100)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16_amp"])
    parser.add_argument("--max_steps", type=int, default=80000)
    parser.add_argument(
        "--lr_schedule_steps",
        type=int,
        default=None,
        help="Cosine LR horizon. Defaults to max_steps; set separately for multi-fidelity searches.",
    )
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--min_lr_ratio", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--log_freq", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.set_defaults(save_final=True)
    parser.add_argument("--save_final", dest="save_final", action="store_true")
    parser.add_argument("--no_save_final", dest="save_final", action="store_false")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle_buffer", type=int, default=512)
    parser.add_argument("--step_mix_buffer_size", type=int, default=64)
    parser.add_argument("--expected_image_size", type=int, default=256)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--max_text_len", type=int, default=256)
    parser.add_argument(
        "--text_padding_length",
        type=int,
        default=21,
        help="Fixed token length. LIBERO uses 21 to match the released training features.",
    )
    parser.add_argument(
        "--text_layout_path",
        type=str,
        default="experiments/libero/configs/online_text_layout.json",
        help="Online tokenizer layout metadata for exact released-checkpoint compatibility.",
    )
    parser.add_argument("--vla_feature_enhancer_layers", type=int, default=6)
    parser.add_argument("--enhancer_inner_dim", type=int, default=1024)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--chunk_size", type=int, default=12)
    parser.add_argument("--state_dim", type=int, default=8)
    parser.add_argument("--num_state_tokens", type=int, default=2)
    parser.add_argument("--act_num_layers", type=int, default=3)
    parser.add_argument("--act_mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--act_state_hidden_dim", type=int, default=256)
    parser.add_argument("--act_dropout", type=float, default=0.1)
    parser.add_argument(
        "--action_head_type",
        type=str,
        default="auto",
        choices=["auto", "ann", "spiking"],
        help="Action decoder type; ann with spiking downstream averages Fusion membranes over T.",
    )
    parser.add_argument("--text_dropout", type=float, default=0.0)
    parser.add_argument("--fusion_dropout", type=float, default=0.0)
    parser.add_argument("--fusion_droppath", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.set_defaults(allow_hf_download=False)
    parser.add_argument("--allow_hf_download", dest="allow_hf_download", action="store_true")
    parser.add_argument("--no_allow_hf_download", dest="allow_hf_download", action="store_false")

    parser.set_defaults(shuffle_steps_within_episode=True)
    parser.add_argument("--shuffle_steps_within_episode", dest="shuffle_steps_within_episode", action="store_true")
    parser.add_argument("--no_shuffle_steps_within_episode", dest="shuffle_steps_within_episode", action="store_false")

    parser.set_defaults(freeze_backbones=False)
    parser.add_argument("--freeze_backbones", dest="freeze_backbones", action="store_true")
    parser.add_argument("--no_freeze_backbones", dest="freeze_backbones", action="store_false")

    parser.set_defaults(freeze_text_encoder=True)
    parser.add_argument("--freeze_text_encoder", dest="freeze_text_encoder", action="store_true")
    parser.add_argument("--train_text_encoder", dest="freeze_text_encoder", action="store_false")

    parser.set_defaults(require_feature_enhancer_preload=True)
    parser.add_argument(
        "--require_feature_enhancer_preload",
        dest="require_feature_enhancer_preload",
        action="store_true",
    )
    parser.add_argument(
        "--no_require_feature_enhancer_preload",
        dest="require_feature_enhancer_preload",
        action="store_false",
    )

    parser.set_defaults(load_text_projection_from_init=True)
    parser.add_argument(
        "--load_text_projection_from_init",
        dest="load_text_projection_from_init",
        action="store_true",
    )
    parser.add_argument(
        "--no_load_text_projection_from_init",
        dest="load_text_projection_from_init",
        action="store_false",
    )

    parser.set_defaults(require_text_proj_preload=True)
    parser.add_argument("--require_text_proj_preload", dest="require_text_proj_preload", action="store_true")
    parser.add_argument("--no_require_text_proj_preload", dest="require_text_proj_preload", action="store_false")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        use_cuda = torch.cuda.is_available()
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend)
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        is_distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_distributed = False
    return is_distributed, rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_model_architecture(args):
    text_layout = {}
    if args.text_layout_path:
        with open(args.text_layout_path, "r", encoding="utf-8") as handle:
            text_layout = json.load(handle)
        configured_length = int(text_layout["output_padding_length"])
        if configured_length != args.text_padding_length:
            raise ValueError(
                f"text layout output length {configured_length} does not match "
                f"--text_padding_length={args.text_padding_length}"
            )
    model_args = DummyArgs()
    model_args.dinov3_path = args.dinov3_path
    model_args.vision_encoder_type = args.vision_encoder_type
    model_args.vision_model_path = args.vision_model_path or args.dinov3_path
    model_args.vision_pretrained_checkpoint = args.vision_pretrained_checkpoint
    model_args.vision_model_source_path = args.vision_model_source_path
    model_args.vision_output_grid_size = args.vision_output_grid_size
    model_args.bert_path = args.bert_path
    model_args.text_encoder_type = args.text_encoder_type
    model_args.spikinglm_source_path = args.spikinglm_source_path
    model_args.spikinglm_time_steps = args.spikinglm_time_steps
    model_args.text_source_path = args.text_source_path
    model_args.downstream_type = args.downstream_type
    model_args.downstream_time_steps = args.downstream_time_steps
    model_args.downstream_lif_tau = args.downstream_lif_tau
    model_args.downstream_lif_backend = args.downstream_lif_backend
    model_args.spike_lif_threshold = args.spike_lif_threshold
    model_args.spike_surrogate_alpha = args.spike_surrogate_alpha
    model_args.spike_detach_reset = args.spike_detach_reset
    model_args.spike_reset_policy = args.spike_reset_policy
    model_args.spike_attention_normalization = args.spike_attention_normalization
    model_args.spike_attention_max_exponent = args.spike_attention_max_exponent
    model_args.spike_attention_reference_offset = args.spike_attention_reference_offset
    model_args.spike_cross_score_shift = args.spike_cross_score_shift
    model_args.spike_self_score_shift = args.spike_self_score_shift
    model_args.spike_action_score_shift = args.spike_action_score_shift
    model_args.spike_quantize_qkv_scales = args.spike_quantize_qkv_scales
    model_args.spike_quantize_attention_scale = args.spike_quantize_attention_scale
    model_args.spike_shared_cross_affinity = args.spike_shared_cross_affinity
    model_args.spike_enable_visual_self_attention = args.spike_enable_visual_self_attention
    model_args.spike_enable_visual_ffn = args.spike_enable_visual_ffn
    model_args.spike_collect_diagnostics = args.spike_collect_diagnostics or args.spike_firing_rate_weight > 0.0
    model_args.spike_cross_attention_dim = args.spike_cross_attention_dim
    model_args.spike_cross_attention_heads = args.spike_cross_attention_heads
    model_args.spike_cross_layer_scale_init = args.spike_cross_layer_scale_init
    model_args.spike_self_layer_scale_init = args.spike_self_layer_scale_init
    model_args.spike_ffn_layer_scale_init = args.spike_ffn_layer_scale_init
    model_args.temporal_readout = args.temporal_readout
    model_args.hidden_dim = args.hidden_dim
    model_args.nheads = args.nheads
    model_args.dim_feedforward = args.dim_feedforward
    model_args.max_text_len = args.max_text_len
    model_args.text_padding_length = args.text_padding_length
    model_args.text_padding_length_by_instruction = text_layout.get("padding_length_by_instruction", {})
    model_args.vla_feature_enhancer_layers = args.vla_feature_enhancer_layers
    model_args.enhancer_inner_dim = args.enhancer_inner_dim
    model_args.text_dropout = args.text_dropout
    model_args.fusion_dropout = args.fusion_dropout
    model_args.fusion_droppath = args.fusion_droppath
    model_args.action_dim = args.action_dim
    model_args.chunk_size = args.chunk_size
    model_args.state_dim = args.state_dim
    model_args.num_state_tokens = args.num_state_tokens
    model_args.act_num_layers = args.act_num_layers
    model_args.act_mlp_hidden_dim = args.act_mlp_hidden_dim
    model_args.act_state_hidden_dim = args.act_state_hidden_dim
    model_args.act_dropout = args.act_dropout
    model_args.action_head_type = args.action_head_type
    model_args.local_files_only = not args.allow_hf_download
    model_args.freeze_vision_encoder = args.freeze_backbones
    model_args.freeze_text_encoder = args.freeze_text_encoder
    model_args.dinov3_precision = args.vision_precision or getattr(args, "dinov3_precision", "bf16_autocast")
    model_args.num_views = 2
    model_args.image_size = args.vision_image_size or args.expected_image_size
    model_args.position_embedding = "view"
    model_args.encode_views_separately = True
    model_args.padding_strategy = "key_padding_mask"
    return build_turbovla(model_args)


def build_distillation_teacher(args):
    if args.teacher_checkpoint is None:
        raise ValueError("--teacher_checkpoint is required when a distillation weight is non-zero")
    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    if model_config is not None:
        from ..models.configuration import TurboVLAConfig

        config = TurboVLAConfig.from_mapping(model_config)
        if config.vision.encoder_type == "dinov3":
            config.vision.model_name_or_path = args.dinov3_path
        elif config.vision.encoder_type == "sdtv3_19m":
            config.vision.model_name_or_path = args.vision_model_path
            config.vision.pretrained_checkpoint = args.vision_pretrained_checkpoint
            config.vision.model_source_path = args.vision_model_source_path
        else:
            raise ValueError(f"unsupported teacher vision encoder: {config.vision.encoder_type}")
        config.vision.frozen = True
        config.vision.local_files_only = not args.allow_hf_download
        # The checkpoint's stored config may record bf16_autocast (the official
        # training default). Force fp32 so the KD targets are deterministic and
        # precision-independent of whatever the release was trained with.
        config.vision.compute_precision = "fp32"
        config.interaction.compute_precision = "fp32"
        config.spike.downstream_type = "ann"
        config.action.decoder_type = "ann"
        config.text.encoder_type = "bert"
        config.text.model_name_or_path = args.teacher_bert_path or args.bert_path
        config.text.model_source_path = None
        config.text.return_temporal_hidden = False
        config.text.frozen = True
        config.text.local_files_only = not args.allow_hf_download
        teacher = build_turbovla(config)
    else:
        teacher_args = DummyArgs()
        for key, value in vars(args).items():
            setattr(teacher_args, key, value)
        teacher_args.text_encoder_type = "bert"
        teacher_args.bert_path = args.teacher_bert_path or args.bert_path
        teacher_args.spikinglm_source_path = None
        teacher_args.downstream_type = "ann"
        teacher_args.action_head_type = "ann"
        teacher_args.vision_encoder_type = "dinov3"
        teacher_args.vision_model_path = args.dinov3_path
        teacher_args.vision_pretrained_checkpoint = None
        teacher_args.freeze_backbones = True
        teacher_args.freeze_vision_encoder = True
        teacher = build_model_architecture(teacher_args)
    state = _extract_model_weights(checkpoint, args.teacher_weight_source)
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"teacher checkpoint is not strictly compatible: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def freeze_backbones(model):
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder.backbone"):
            param.requires_grad = False


def get_latest_checkpoint(ckpt_dir, prefix):
    ckpts = glob.glob(os.path.join(ckpt_dir, f"{prefix}_*.pth"))
    if not ckpts:
        return None

    def extract_step(path):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return -1

    return max(ckpts, key=extract_step)


def masked_l1_loss(pred, target, mask):
    l1 = torch.abs(pred - target)
    mask = mask.unsqueeze(-1).float()
    l1 = l1 * mask
    denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    return l1.sum() / denom


def masked_mse_loss(pred, target, mask):
    squared = (pred - target).square()
    mask = mask.unsqueeze(-1).float()
    squared = squared * mask
    denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    return squared.sum() / denom


def closed_loop_action_loss(
    pred,
    target,
    mask,
    horizon_first_weight=1.0,
    horizon_last_weight=1.0,
    gripper_loss_weight=0.0,
    gripper_sign_temperature=0.25,
    action_delta_weight=0.0,
):
    """Action objective emphasizing executable early actions and gripper sign."""
    if pred.shape[-1] < 7:
        raise ValueError("closed-loop action loss expects at least 7 action dimensions")
    # Exact TurboVLA-compatible objective. Keep the optional closed-loop terms
    # available for ablations, but the fair-comparison launcher selects this
    # branch and therefore trains on all 12 x 7 masked values identically to
    # the original masked L1 implementation.
    if (
        float(horizon_first_weight) == 1.0
        and float(horizon_last_weight) == 1.0
        and float(gripper_loss_weight) == 0.0
        and float(action_delta_weight) == 0.0
    ):
        return masked_l1_loss(pred, target, mask)
    horizon = pred.shape[1]
    first_weight = float(horizon_first_weight)
    last_weight = float(horizon_last_weight)
    if first_weight <= 0.0 or last_weight <= 0.0:
        raise ValueError("horizon endpoint weights must be positive")
    time_weights = torch.linspace(
        first_weight,
        last_weight,
        steps=horizon,
        device=pred.device,
        dtype=pred.dtype,
    )[None, :]
    valid_weights = mask.to(dtype=pred.dtype) * time_weights

    arm_error = (pred[..., :6] - target[..., :6]).abs()
    arm_denom = (valid_weights.sum() * 6).clamp_min(1.0)
    arm_loss = (arm_error * valid_weights[..., None]).sum() / arm_denom

    total = arm_loss
    if gripper_loss_weight > 0:
        temperature = float(gripper_sign_temperature)
        if temperature <= 0.0:
            raise ValueError("gripper_sign_temperature must be positive")
        # A smooth sign objective on the bounded action avoids the hard clamp
        # used by probability BCE. It keeps a useful gradient when the model
        # confidently predicts the wrong open/close sign near +/-1.
        target_sign = torch.where(target[..., 6] >= 0, 1.0, -1.0).to(pred.dtype)
        gripper_sign = F.softplus(-target_sign * pred[..., 6] / temperature) * temperature
        gripper_loss = (gripper_sign * valid_weights).sum() / valid_weights.sum().clamp_min(1.0)
        total = total + float(gripper_loss_weight) * gripper_loss

    if action_delta_weight > 0 and horizon > 1:
        # Smooth only the continuous arm channels. Penalizing the gripper
        # delta would discourage the discrete open/close transition that is
        # essential to LIBERO success.
        pred_delta = pred[:, 1:, :6] - pred[:, :-1, :6]
        target_delta = target[:, 1:, :6] - target[:, :-1, :6]
        delta_mask = mask[:, 1:] * mask[:, :-1]
        delta_loss = masked_l1_loss(pred_delta, target_delta, delta_mask)
        total = total + float(action_delta_weight) * delta_loss
    return total


@torch.no_grad()
def action_diagnostics(pred, target, mask):
    """Return additive numerators/denominators for distributed action metrics."""
    valid = mask.to(dtype=torch.float32)
    error = (pred.float() - target.float()).abs()
    valid_count = valid.sum().clamp_min(1.0)
    translation = (error[..., :3] * valid[..., None]).sum() / (valid_count * 3.0)
    rotation = (error[..., 3:6] * valid[..., None]).sum() / (valid_count * 3.0)
    gripper = (error[..., 6] * valid).sum() / valid_count
    pred_sign = pred[..., 6] >= 0
    target_sign = target[..., 6] >= 0
    gripper_accuracy = ((pred_sign == target_sign).float() * valid).sum() / valid_count
    horizon_den = valid.sum(dim=0).clamp_min(1.0) * pred.shape[-1]
    horizon_mae = (error * valid[..., None]).sum(dim=(0, 2)) / horizon_den
    return torch.cat(
        [translation[None], rotation[None], gripper[None], gripper_accuracy[None], horizon_mae]
    )


def build_scheduler(optimizer, max_steps, warmup_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            warmup_scale = float(step + 1) / float(max(1, warmup_steps))
            return 0.1 + 0.9 * warmup_scale

        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _use_weight_decay(name, param):
    if param.ndim <= 1:
        return False
    lowered = name.lower()
    if lowered.endswith(".bias"):
        return False
    if "norm" in lowered or "layernorm" in lowered:
        return False
    return True


def build_param_group_optimizer(model, args):
    head_lr = args.lr if args.head_lr is None else args.head_lr
    head_wd = args.weight_decay if args.head_weight_decay is None else args.head_weight_decay
    vision_lr = args.dinov3_lr if args.vision_encoder_lr is None else args.vision_encoder_lr
    vision_wd = (
        args.dinov3_weight_decay
        if args.vision_encoder_weight_decay is None
        else args.vision_encoder_weight_decay
    )
    grouped = {
        ("vision_decay", vision_lr, vision_wd): [],
        ("vision_no_decay", vision_lr, 0.0): [],
        ("head_decay", head_lr, head_wd): [],
        ("head_no_decay", head_lr, 0.0): [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_vision = name.startswith("vision_encoder.backbone")
        decay = _use_weight_decay(name, param)
        if is_vision and decay:
            key = ("vision_decay", vision_lr, vision_wd)
        elif is_vision:
            key = ("vision_no_decay", vision_lr, 0.0)
        elif decay:
            key = ("head_decay", head_lr, head_wd)
        else:
            key = ("head_no_decay", head_lr, 0.0)
        grouped[key].append(param)

    param_groups = []
    summary = []
    for (group_name, lr, weight_decay), params in grouped.items():
        if not params:
            continue
        count = sum(p.numel() for p in params)
        param_groups.append({"params": params, "lr": lr, "weight_decay": weight_decay, "name": group_name})
        summary.append({"name": group_name, "lr": lr, "weight_decay": weight_decay, "params": count})
    return AdamW(param_groups), summary


def reduce_mean(value, device, is_distributed, world_size):
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    if is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor.item()


SPIKE_EVAL_METRIC_NAMES = (
    "lif_rate_mean",
    "lif_rate_min",
    "lif_rate_max",
    "qkv_firing_rate_mean",
    "attention_entropy_mean",
    "attention_top1_mass_mean",
    "active_lif_nodes",
    "membrane_abs_mean",
    "membrane_abs_max",
    "membrane_active_node_fraction",
    "output_finite_fraction",
    "output_abs_mean",
    "output_abs_max",
    "output_std",
)


@torch.no_grad()
def run_spike_health_probe(
    model,
    model_to_load,
    instructions,
    samples,
    states,
    diagnostics_enabled,
):
    """Run one non-training SNN probe and restore the caller state."""

    membrane_snapshot = snapshot_spiking_state(model_to_load)
    training_modes = [(child, child.training) for child in model.modules()]
    set_spike_diagnostics_(model_to_load, True)
    model.eval()
    try:
        # The explicit reset makes the probe reproducible even in manual mode.
        # The snapshot below restores the live episode state afterwards.
        prediction = model(
            instructions,
            samples,
            states,
            reset_spiking=True,
        )
        prediction = prediction.float()
        finite = torch.isfinite(prediction).float()
        output_stats = torch.stack((
            finite.mean(),
            prediction.abs().mean(),
            prediction.abs().max(),
            prediction.std(unbiased=False),
        ))
        return torch.cat((
            spike_diagnostics_vector(model_to_load),
            spike_membrane_statistics(model_to_load),
            output_stats,
        ))
    finally:
        clear_spike_activity_(model_to_load)
        restore_spiking_state_(model_to_load, membrane_snapshot)
        set_spike_diagnostics_(model_to_load, diagnostics_enabled)
        for child, training in training_modes:
            child.training = training


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["model_state_dict", "model", "state_dict"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("unsupported checkpoint format")


def _extract_model_weights(ckpt_obj, source):
    if source not in {"model", "ema"}:
        raise ValueError(f"unsupported checkpoint weight source: {source}")
    if not isinstance(ckpt_obj, dict):
        if source == "ema":
            raise KeyError("EMA weights requested from a checkpoint without named states")
        return _extract_state_dict(ckpt_obj)
    key = "ema_model_state_dict" if source == "ema" else "model_state_dict"
    state = ckpt_obj.get(key)
    if not isinstance(state, dict) or not state:
        raise KeyError(f"{key!r} is missing or empty in checkpoint")
    return state


def load_student_components_from_checkpoint(model, ckpt_path, weight_source):
    """Warm-start only components shared by ANN- and SNN-fusion students."""

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"student init checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    source_state = _extract_model_weights(checkpoint, weight_source)
    source_state = {(key[7:] if key.startswith("module.") else key): value for key, value in source_state.items()}
    target_state = model.state_dict()
    mapped = {}

    for key, tensor in source_state.items():
        if not key.startswith(("vision_encoder.", "action_head.")):
            continue
        if key in target_state and target_state[key].shape == tensor.shape:
            mapped[key] = tensor

    aliases = {
        "text_encoder.text_projection.weight": "text_encoder.text_projection.linear.weight",
        "text_encoder.text_projection.bias": "text_encoder.text_projection.linear.bias",
        "vision_projection.mlp.0.weight": "vision_projection.fc1.weight",
        "vision_projection.mlp.0.bias": "vision_projection.fc1.bias",
        "vision_projection.mlp.3.weight": "vision_projection.fc2.weight",
        "vision_projection.mlp.3.bias": "vision_projection.fc2.bias",
        "vision_projection.skip.weight": "vision_projection.skip.weight",
    }
    for source_key, target_key in aliases.items():
        tensor = source_state.get(source_key)
        if tensor is not None and target_key in target_state and target_state[target_key].shape == tensor.shape:
            mapped[target_key] = tensor

    for suffix in ("weight", "bias"):
        source_key = f"vision_projection.output_norm.{suffix}"
        tensor = source_state.get(source_key)
        if tensor is None:
            continue
        for norm_name in ("fc2_norm", "skip_norm"):
            target_key = f"vision_projection.{norm_name}.{suffix}"
            if target_key in target_state and target_state[target_key].shape == tensor.shape:
                mapped[target_key] = tensor

    if not mapped:
        raise RuntimeError("student component warm-start mapped no parameters")
    target_state.update(mapped)
    model.load_state_dict(target_state, strict=True)
    return {
        "mapped": len(mapped),
        "vision_encoder": sum(key.startswith("vision_encoder.") for key in mapped),
        "action_head": sum(key.startswith("action_head.") for key in mapped),
        "projection": sum("projection" in key and not key.startswith("action_head.") for key in mapped),
    }


def load_interaction_from_init_checkpoint(model, ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    source_state = _extract_state_dict(ckpt)
    source_state = {(k[7:] if k.startswith("module.") else k): v for k, v in source_state.items()}

    target_state = model.state_dict()
    mapped = {}
    skipped_missing = 0
    skipped_shape = 0
    missing_keys = []
    shape_mismatch_keys = []

    prefix_pairs = [
        ("transformer.encoder.fusion_layers.", "vision_language_interaction.fusion_layers."),
        ("transformer.encoder.text_layers.", "vision_language_interaction.text_layers."),
    ]

    for src_key, tensor in source_state.items():
        tgt_key = None
        for src_prefix, tgt_prefix in prefix_pairs:
            if src_key.startswith(src_prefix):
                tgt_key = tgt_prefix + src_key[len(src_prefix):]
                break
        if tgt_key is None:
            continue
        if tgt_key not in target_state:
            skipped_missing += 1
            missing_keys.append((src_key, tgt_key))
            continue
        if target_state[tgt_key].shape != tensor.shape:
            skipped_shape += 1
            shape_mismatch_keys.append((src_key, tgt_key, tuple(tensor.shape), tuple(target_state[tgt_key].shape)))
            continue
        mapped[tgt_key] = tensor

    if not mapped:
        raise RuntimeError("no feature-enhancer parameters were mapped from checkpoint")

    target_state.update(mapped)
    missing, unexpected = model.load_state_dict(target_state, strict=False)
    return {
        "mapped": len(mapped),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "load_missing_after_update": len(missing),
        "load_unexpected_after_update": len(unexpected),
        "missing_keys": missing_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
    }


def load_text_projection_from_init_checkpoint(model, ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    source_state = _extract_state_dict(ckpt)
    source_state = {(k[7:] if k.startswith("module.") else k): v for k, v in source_state.items()}

    target_state = model.state_dict()
    mapped = {}
    skipped_missing = 0
    skipped_shape = 0
    missing_keys = []
    shape_mismatch_keys = []

    for src_key, tensor in source_state.items():
        tgt_key = None
        if src_key.startswith("feat_map."):
            tgt_key = "text_encoder.text_projection." + src_key[len("feat_map."):]
        else:
            continue

        if tgt_key not in target_state:
            skipped_missing += 1
            missing_keys.append((src_key, tgt_key))
            continue
        if target_state[tgt_key].shape != tensor.shape:
            skipped_shape += 1
            shape_mismatch_keys.append((src_key, tgt_key, tuple(tensor.shape), tuple(target_state[tgt_key].shape)))
            continue
        mapped[tgt_key] = tensor

    if not mapped:
        raise RuntimeError("no text projection parameters were mapped from checkpoint")

    target_state.update(mapped)
    missing, unexpected = model.load_state_dict(target_state, strict=False)
    return {
        "mapped": len(mapped),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "load_missing_after_update": len(missing),
        "load_unexpected_after_update": len(unexpected),
        "missing_keys": missing_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
    }


def move_samples_to_device(samples, device):
    if isinstance(samples, dict):
        return {k: v.to(device, non_blocking=True) for k, v in samples.items()}
    return samples.to(device, non_blocking=True)


def train_model():
    args = parse_args()
    if args.head_lr is None:
        args.head_lr = args.lr
    if args.head_weight_decay is None:
        args.head_weight_decay = args.weight_decay
    if args.vision_encoder_lr is None:
        args.vision_encoder_lr = args.dinov3_lr
    if args.vision_encoder_weight_decay is None:
        args.vision_encoder_weight_decay = args.dinov3_weight_decay
    if args.vision_image_size is None:
        args.vision_image_size = 224 if args.vision_encoder_type == "sdtv3_19m" else args.expected_image_size
    if args.distill_feature_weight < 0 or args.distill_action_weight < 0:
        raise ValueError("distillation weights must be non-negative")
    if args.horizon_first_weight <= 0.0 or args.horizon_last_weight <= 0.0:
        raise ValueError("--horizon_first_weight and --horizon_last_weight must be positive")
    if args.gripper_loss_weight < 0 or args.action_delta_weight < 0:
        raise ValueError("gripper and action-delta loss weights must be non-negative")
    if args.gripper_sign_temperature <= 0 or args.action_diagnostics_freq < 1:
        raise ValueError("gripper sign temperature and diagnostics frequency must be positive")
    if args.spike_firing_rate_weight < 0:
        raise ValueError("--spike_firing_rate_weight must be non-negative")
    if not 0.0 <= args.spike_target_firing_rate <= 1.0:
        raise ValueError("--spike_target_firing_rate must be in [0, 1]")
    if args.spike_diagnostics_freq < 1:
        raise ValueError("--spike_diagnostics_freq must be positive")
    if args.spike_eval_freq < 0:
        raise ValueError("--spike_eval_freq must be non-negative; 0 disables evaluation")
    if args.spike_qat_start_step < -1:
        raise ValueError("--spike_qat_start_step must be -1 or non-negative")
    if args.spike_firing_rate_weight > 0.0 and args.downstream_type != "spiking":
        raise ValueError("spike firing-rate regularization requires --downstream_type spiking")
    distillation_enabled = args.distill_feature_weight > 0 or args.distill_action_weight > 0
    if args.precision == "bf16_amp":
        if not torch.cuda.is_available():
            raise ValueError("bf16_amp precision requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bf16_amp requested but current CUDA device does not report bf16 support")
    torch.set_float32_matmul_precision("high")

    if args.lr_schedule_steps is None:
        args.lr_schedule_steps = args.max_steps
    if args.lr_schedule_steps <= args.warmup_steps:
        raise ValueError(
            f"lr_schedule_steps={args.lr_schedule_steps} must be greater than warmup_steps={args.warmup_steps}"
        )
    set_seed(args.seed)

    is_distributed, rank, world_size, local_rank, device = setup_distributed()

    try:
        model = build_model_architecture(args)
        student_init_report = None
        if args.student_init_checkpoint is not None:
            student_init_report = load_student_components_from_checkpoint(
                model,
                args.student_init_checkpoint,
                args.student_init_weight_source,
            )
        diagnostics_enabled = args.downstream_type == "spiking" and (
            args.spike_collect_diagnostics or args.spike_firing_rate_weight > 0.0
        )
        set_spike_diagnostics_(model, diagnostics_enabled)
        teacher = build_distillation_teacher(args) if distillation_enabled else None

        need_any_init_preload = (
            args.require_feature_enhancer_preload
            or args.require_text_proj_preload
            or args.load_text_projection_from_init
        )
        if args.pretrained_init_ckpt is None and need_any_init_preload:
            raise ValueError(
                "`--pretrained_init_ckpt` is required for requested preload options "
                "(feature-enhancer and/or text_proj)"
            )

        if args.pretrained_init_ckpt is not None:
            if args.require_feature_enhancer_preload:
                fe_report = load_interaction_from_init_checkpoint(model, args.pretrained_init_ckpt)
                if rank == 0:
                    print("vision-language interaction preload from init checkpoint:")
                    print(f"  ckpt: {args.pretrained_init_ckpt}")
                    print(f"  mapped={fe_report['mapped']}")
                    print(f"  skipped_missing={fe_report['skipped_missing']}")
                    print(f"  skipped_shape={fe_report['skipped_shape']}")
                    print("missing_keys:", fe_report["missing_keys"])
                    print("shape_mismatch_keys:", fe_report["shape_mismatch_keys"])

            if args.load_text_projection_from_init:
                text_proj_report = load_text_projection_from_init_checkpoint(model, args.pretrained_init_ckpt)
                if rank == 0:
                    print("text projection preload from init checkpoint:")
                    print(f"  ckpt: {args.pretrained_init_ckpt}")
                    print(f"  mapped={text_proj_report['mapped']}")
                    print(f"  skipped_missing={text_proj_report['skipped_missing']}")
                    print(f"  skipped_shape={text_proj_report['skipped_shape']}")
                    print("missing_keys:", text_proj_report["missing_keys"])
                    print("shape_mismatch_keys:", text_proj_report["shape_mismatch_keys"])
            elif args.require_text_proj_preload:
                raise ValueError(
                    "text projection preload is required, but `--no_load_text_projection_from_init` was set"
                )

        if args.freeze_backbones:
            freeze_backbones(model)

        if rank == 0:
            trainable = [n for n, p in model.named_parameters() if p.requires_grad]
            frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
            print(f"device={device}, distributed={is_distributed}, world_size={world_size}")
            print(f"vision_encoder_type={args.vision_encoder_type}")
            print(f"vision_model_path={args.vision_model_path or args.dinov3_path}")
            print(f"vision_pretrained_checkpoint={args.vision_pretrained_checkpoint}")
            print(f"teacher_checkpoint={args.teacher_checkpoint}")
            print(f"teacher_weight_source={args.teacher_weight_source}")
            print(
                f"distill_feature_weight={args.distill_feature_weight}, "
                f"distill_action_weight={args.distill_action_weight}"
            )
            print(
                f"action_objective: horizon_first_weight={args.horizon_first_weight}, "
                f"horizon_last_weight={args.horizon_last_weight}, "
                f"gripper_loss_weight={args.gripper_loss_weight}, "
                f"gripper_sign_temperature={args.gripper_sign_temperature}, "
                f"action_delta_weight={args.action_delta_weight}"
            )
            print(f"text_encoder_type={args.text_encoder_type}, text_model_path={args.bert_path}")
            if args.text_encoder_type == "spikinglm":
                print(
                    f"spikinglm_source_path={args.spikinglm_source_path}, "
                    f"spikinglm_time_steps={args.spikinglm_time_steps}"
                )
            elif args.text_encoder_type == "sootspike":
                print(
                    f"text_source_path={args.text_source_path}, "
                    f"text_time_steps={args.spikinglm_time_steps}"
                )
            print(
                f"downstream_type={args.downstream_type}, T={args.downstream_time_steps}, "
                f"lif_tau={args.downstream_lif_tau}, lif_backend={args.downstream_lif_backend}, "
                f"threshold={args.spike_lif_threshold}, surrogate_alpha={args.spike_surrogate_alpha}, "
                f"reset_policy={args.spike_reset_policy}"
            )
            if args.downstream_type == "spiking":
                print(
                    f"spike_attention: normalization={args.spike_attention_normalization}, "
                    f"max_exponent={args.spike_attention_max_exponent}, "
                    f"reference_offset={args.spike_attention_reference_offset}, "
                    f"score_shift(cross/self/action)={args.spike_cross_score_shift}/"
                    f"{args.spike_self_score_shift}/{args.spike_action_score_shift}"
                )
                print(
                    f"spike_monitoring: enabled={diagnostics_enabled}, "
                    f"rate_weight={args.spike_firing_rate_weight}, "
                    f"target_rate={args.spike_target_firing_rate}, "
                    f"eval_freq={args.spike_eval_freq}, "
                    f"qat_start={args.spike_qat_start_step}"
                )
            print(f"action_head_type={args.action_head_type}")
            print(f"temporal_readout={args.temporal_readout}")
            print(
                f"student_init_checkpoint={args.student_init_checkpoint}, "
                f"weight_source={args.student_init_weight_source}, report={student_init_report}"
            )
            if args.downstream_type == "spiking":
                print(
                    "spike_structure: "
                    f"shared_biattention={args.spike_cross_attention_dim}d/"
                    f"{args.spike_cross_attention_heads}h, "
                    f"fusion_layers={args.vla_feature_enhancer_layers}, "
                    f"act_layers={args.act_num_layers}, "
                    f"layerscale(cross/self/ffn)="
                    f"{args.spike_cross_layer_scale_init}/"
                    f"{args.spike_self_layer_scale_init}/"
                    f"{args.spike_ffn_layer_scale_init}"
                )
            print(
                f"online_text_encoder=True, freeze_text_encoder={args.freeze_text_encoder}, "
                f"text_padding_length={args.text_padding_length}, text_layout_path={args.text_layout_path}"
            )
            print(f"pretrained_init_ckpt={args.pretrained_init_ckpt}")
            print(f"load_text_projection_from_init={args.load_text_projection_from_init}")
            print(f"max_steps={args.max_steps}, lr_schedule_steps={args.lr_schedule_steps}")
            print(
                f"precision={args.precision}, head_lr={args.head_lr}, vision_lr={args.vision_encoder_lr}, "
                f"head_wd={args.head_weight_decay}, vision_wd={args.vision_encoder_weight_decay}"
            )
            print(f"warmup_steps={args.warmup_steps}, min_lr_ratio={args.min_lr_ratio}")
            print(
                f"shuffle: episode_buffer={args.shuffle_buffer}, "
                f"within_episode={args.shuffle_steps_within_episode}"
            )
            print(f"trainable params: {len(trainable)}")
            print(f"frozen params: {len(frozen)}")
            print("first 30 trainable:", trainable[:30])

        if rank == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        if is_distributed:
            dist.barrier()

        global_step = 0
        ckpt = None
        latest_ckpt_path = get_latest_checkpoint(args.checkpoint_dir, args.checkpoint_prefix)
        if latest_ckpt_path is not None and args.resume_mode != "none":
            if rank == 0:
                print(f"resume from {latest_ckpt_path} with mode={args.resume_mode}")
            ckpt = torch.load(latest_ckpt_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            legacy_reverse_missing = any(
                "text_query" in key or "visual_key" in key
                for key in missing
            )
            if legacy_reverse_missing:
                interaction = getattr(model, "vision_language_interaction", None)
                upgrade = getattr(interaction, "upgrade_legacy_interaction_", None)
                if upgrade is not None:
                    upgrade()
                    if rank == 0:
                        print("initialized new reverse cross-attention projections from shared checkpoint weights")
            if rank == 0:
                print("resume missing keys:", len(missing))
                print("resume unexpected keys:", len(unexpected))
            if args.resume_mode == "all":
                global_step = int(ckpt.get("global_step", 0))
            else:
                global_step = 0
                if rank == 0:
                    print("optimizer/scheduler reset due to resume_mode=model")
        elif rank == 0:
            print("no checkpoint resumed, train from current initialization")

        model.to(device)
        qat_started = bool(
            args.downstream_type == "spiking"
            and args.spike_qat_start_step >= 0
            and global_step >= args.spike_qat_start_step
        )
        if qat_started:
            set_hardware_qat_(model, enabled=True)
            if rank == 0:
                print(f"spike QAT enabled at global_step={global_step}")
        if teacher is not None:
            teacher.to(device)
            teacher.eval()
        if is_distributed:
            if device.type == "cuda":
                model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            else:
                model = DDP(model, find_unused_parameters=True)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer, optimizer_summary = build_param_group_optimizer(unwrap_model(model), args)
        if rank == 0:
            print("optimizer param groups:", optimizer_summary)
        scheduler = build_scheduler(
            optimizer,
            max_steps=args.lr_schedule_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

        if ckpt is not None and args.resume_mode == "all":
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        model_to_load = unwrap_model(model)

        dataset = LiberoRLDSDataset(
            dataset_dir=args.dataset_dir,
            LOCAL_DINOV3_PATH=args.dinov3_path,
            rank=rank,
            world_size=world_size,
            chunk_size=args.chunk_size,
            split=args.dataset_split,
            shuffle_buffer=args.shuffle_buffer,
            shuffle_steps_within_episode=args.shuffle_steps_within_episode,
            step_mix_buffer_size=args.step_mix_buffer_size,
            seed=args.seed,
            local_files_only=not args.allow_hf_download,
            expected_image_size=args.expected_image_size,
            vision_encoder_type=args.vision_encoder_type,
            vision_image_size=args.vision_image_size,
            enable_dinov3_teacher=distillation_enabled,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=None,
            collate_fn=vla_collate_fn,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        data_iter = iter(dataloader)

        model.train()
        if args.freeze_backbones:
            model_to_load.dinov3.eval()

        if rank == 0:
            per_gpu_effective_bs = args.batch_size * args.grad_accum_steps
            global_effective_bs = per_gpu_effective_bs * world_size
            print(f"batch_size(per_gpu)={args.batch_size}, grad_accum_steps={args.grad_accum_steps}")
            print(f"effective_batch_size(per_gpu)={per_gpu_effective_bs}")
            print(f"effective_batch_size(global)={global_effective_bs}")
            pbar = tqdm(total=args.max_steps, initial=global_step, desc="training")
        else:
            pbar = None

        loss_window = []
        action_loss_window = []
        feature_distill_window = []
        action_distill_window = []
        spike_rate_regularization_window = []
        spike_diagnostics_window = []
        action_diagnostics_window = []
        spike_eval_path = os.path.join(
            args.checkpoint_dir,
            f"{args.checkpoint_prefix}_spike_eval.jsonl",
        )
        while global_step < args.max_steps:
            if (
                args.downstream_type == "spiking"
                and args.spike_qat_start_step >= 0
                and not qat_started
                and global_step >= args.spike_qat_start_step
            ):
                set_hardware_qat_(model, enabled=True)
                qat_started = True
                if rank == 0:
                    print(f"spike QAT enabled at global_step={global_step}")
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            action_loss_accum = 0.0
            feature_distill_accum = 0.0
            action_distill_accum = 0.0
            spike_rate_regularization_accum = 0.0
            spike_diagnostics_accum = None
            diagnostics_accum = None

            for _ in range(args.grad_accum_steps):
                try:
                    samples, instructions, states, gt_actions, action_chunk_masks = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    samples, instructions, states, gt_actions, action_chunk_masks = next(data_iter)

                samples = move_samples_to_device(samples, device)
                states = states.to(device, non_blocking=True)
                gt_actions = gt_actions.to(device, non_blocking=True)
                action_chunk_masks = action_chunk_masks.to(device, non_blocking=True)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=args.precision == "bf16_amp",
                ):
                    if distillation_enabled:
                        pred_actions, student_visual_tokens = model(
                            instructions, samples, states, return_visual_tokens=True
                        )
                    else:
                        pred_actions = model(instructions, samples, states)
                    if pred_actions.shape != gt_actions.shape:
                        raise ValueError(
                            f"pred_actions.shape={pred_actions.shape}, gt_actions.shape={gt_actions.shape} mismatch"
                        )

                    action_loss = closed_loop_action_loss(
                        pred_actions,
                        gt_actions,
                        action_chunk_masks,
                        horizon_first_weight=args.horizon_first_weight,
                        horizon_last_weight=args.horizon_last_weight,
                        gripper_loss_weight=args.gripper_loss_weight,
                        gripper_sign_temperature=args.gripper_sign_temperature,
                        action_delta_weight=args.action_delta_weight,
                    )
                    spike_rate_regularization = pred_actions.new_zeros(())
                    spike_batch_diagnostics = None
                    if diagnostics_enabled:
                        if args.spike_firing_rate_weight > 0.0:
                            spike_rate_regularization = spike_activity_regularization(
                                model_to_load,
                                args.spike_target_firing_rate,
                            )
                        spike_batch_diagnostics = spike_diagnostics_vector(model_to_load)
                    diagnostics = action_diagnostics(
                        pred_actions, gt_actions, action_chunk_masks
                    )
                    feature_distill_loss = pred_actions.new_zeros(())
                    action_distill_loss = pred_actions.new_zeros(())
                    if teacher is not None:
                        teacher_samples = {"pixel_values": samples["teacher_pixel_values"]}
                        with torch.no_grad():
                            teacher_actions, teacher_visual_tokens = teacher(
                                instructions,
                                teacher_samples,
                                states,
                                return_visual_tokens=True,
                            )
                        if student_visual_tokens.shape != teacher_visual_tokens.shape:
                            raise ValueError(
                                "student/teacher visual token mismatch: "
                                f"{student_visual_tokens.shape} vs {teacher_visual_tokens.shape}"
                            )
                        feature_distill_loss = F.mse_loss(
                            student_visual_tokens.float(), teacher_visual_tokens.float()
                        )
                        action_distill_loss = masked_mse_loss(
                            pred_actions.float(), teacher_actions.float(), action_chunk_masks
                        )
                    loss = (
                        action_loss
                        + args.distill_feature_weight * feature_distill_loss
                        + args.distill_action_weight * action_distill_loss
                        + args.spike_firing_rate_weight * spike_rate_regularization
                    )
                (loss / args.grad_accum_steps).backward()
                clear_spike_activity_(model_to_load)
                loss_accum += loss.detach().item()
                spike_rate_regularization_accum += spike_rate_regularization.detach().item()
                if spike_batch_diagnostics is not None:
                    spike_diagnostics_accum = (
                        spike_batch_diagnostics.detach()
                        if spike_diagnostics_accum is None
                        else spike_diagnostics_accum + spike_batch_diagnostics.detach()
                    )
                action_loss_accum += action_loss.detach().item()
                feature_distill_accum += feature_distill_loss.detach().item()
                action_distill_accum += action_distill_loss.detach().item()
                diagnostics_accum = (
                    diagnostics.detach()
                    if diagnostics_accum is None
                    else diagnostics_accum + diagnostics.detach()
                )

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            local_loss = loss_accum / args.grad_accum_steps
            global_loss = reduce_mean(local_loss, device, is_distributed, world_size)
            global_action_loss = reduce_mean(
                action_loss_accum / args.grad_accum_steps, device, is_distributed, world_size
            )
            global_feature_distill = reduce_mean(
                feature_distill_accum / args.grad_accum_steps, device, is_distributed, world_size
            )
            global_action_distill = reduce_mean(
                action_distill_accum / args.grad_accum_steps, device, is_distributed, world_size
            )
            global_spike_rate_regularization = reduce_mean(
                spike_rate_regularization_accum / args.grad_accum_steps,
                device,
                is_distributed,
                world_size,
            )
            global_diagnostics = diagnostics_accum / args.grad_accum_steps
            global_spike_diagnostics = None
            if spike_diagnostics_accum is not None:
                global_spike_diagnostics = spike_diagnostics_accum / args.grad_accum_steps
                if is_distributed:
                    dist.all_reduce(global_spike_diagnostics, op=dist.ReduceOp.SUM)
                    global_spike_diagnostics /= world_size
            if is_distributed:
                dist.all_reduce(global_diagnostics, op=dist.ReduceOp.SUM)
                global_diagnostics /= world_size

            loss_window.append(global_loss)
            if len(loss_window) > args.log_freq:
                loss_window.pop(0)
            action_loss_window.append(global_action_loss)
            feature_distill_window.append(global_feature_distill)
            action_distill_window.append(global_action_distill)
            spike_rate_regularization_window.append(global_spike_rate_regularization)
            if global_spike_diagnostics is not None:
                spike_diagnostics_window.append(global_spike_diagnostics.cpu())
            action_diagnostics_window.append(global_diagnostics.cpu())
            for window in (action_loss_window, feature_distill_window, action_distill_window, spike_rate_regularization_window):
                if len(window) > args.log_freq:
                    window.pop(0)
            if len(spike_diagnostics_window) > args.log_freq:
                spike_diagnostics_window.pop(0)
            if len(action_diagnostics_window) > args.log_freq:
                action_diagnostics_window.pop(0)

            global_step += 1

            if (
                args.downstream_type == "spiking"
                and args.spike_eval_freq > 0
                and global_step % args.spike_eval_freq == 0
            ):
                spike_eval_vector = run_spike_health_probe(
                    model,
                    model_to_load,
                    instructions,
                    samples,
                    states,
                    diagnostics_enabled,
                )
                if is_distributed:
                    dist.all_reduce(spike_eval_vector, op=dist.ReduceOp.SUM)
                    spike_eval_vector /= world_size
                if rank == 0:
                    spike_eval_record = {
                        "step": global_step,
                        **dict(zip(
                            SPIKE_EVAL_METRIC_NAMES,
                            spike_eval_vector.detach().cpu().tolist(),
                        )),
                    }
                    with open(spike_eval_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(spike_eval_record, sort_keys=True) + "\n")
                    spike_eval_text = ", ".join(
                        f"{name}={spike_eval_record[name]:.4f}"
                        for name in SPIKE_EVAL_METRIC_NAMES
                    )
                    print(f"spike_eval step={global_step}: {spike_eval_text}")

            if rank == 0:
                avg_window_loss = sum(loss_window) / len(loss_window)
                pbar.update(1)
                if global_step % args.log_freq == 0:
                    mean_diagnostics = torch.stack(action_diagnostics_window).mean(0)
                    pbar.set_postfix(
                        {
                            "loss": f"{global_loss:.5f}",
                            "avg": f"{avg_window_loss:.5f}",
                            "act": f"{sum(action_loss_window) / len(action_loss_window):.5f}",
                            "feat_kd": f"{sum(feature_distill_window) / len(feature_distill_window):.5f}",
                            "act_kd": f"{sum(action_distill_window) / len(action_distill_window):.5f}",
                            "spk_reg": f"{sum(spike_rate_regularization_window) / len(spike_rate_regularization_window):.5f}",
                            "trans": f"{mean_diagnostics[0].item():.4f}",
                            "rot": f"{mean_diagnostics[1].item():.4f}",
                            "grip": f"{mean_diagnostics[2].item():.4f}",
                            "gacc": f"{mean_diagnostics[3].item():.3f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        }
                    )
                if global_step % args.action_diagnostics_freq == 0:
                    mean_diagnostics = torch.stack(action_diagnostics_window).mean(0)
                    horizon_values = ", ".join(
                        f"{value:.4f}" for value in mean_diagnostics[4:].tolist()
                    )
                    print(f"action_horizon_mae step={global_step}: [{horizon_values}]")
                if (
                    diagnostics_enabled
                    and global_step % args.spike_diagnostics_freq == 0
                    and spike_diagnostics_window
                ):
                    mean_spike = torch.stack(spike_diagnostics_window).mean(0)
                    print(
                        "spike_diagnostics "
                        f"step={global_step}: "
                        f"rate={mean_spike[0].item():.4f}, "
                        f"rate_min={mean_spike[1].item():.4f}, "
                        f"rate_max={mean_spike[2].item():.4f}, "
                        f"qkv={mean_spike[3].item():.4f}, "
                        f"attn_entropy={mean_spike[4].item():.4f}, "
                        f"attn_top1={mean_spike[5].item():.4f}, "
                        f"lif_nodes={mean_spike[6].item():.0f}"
                    )

            should_save = global_step % args.save_steps == 0 or (
                args.save_final and global_step == args.max_steps
            )
            if rank == 0 and should_save:
                save_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_prefix}_{global_step}.pth")
                save_model = unwrap_model(model)
                # QAT toggles runtime flags rather than parameters; mirror the
                # active mode into the serialized config so inference rebuilds
                # the same numerical path.
                if qat_started and save_model.spiking_downstream:
                    save_model.config.spike.quantize_qkv_scales = True
                    save_model.config.spike.quantize_attention_scale = True
                torch.save(
                    {
                        "global_step": global_step,
                        "model_state_dict": save_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": global_loss,
                        "args": vars(args),
                        "distillation": {
                            "enabled": distillation_enabled,
                            "teacher_checkpoint": args.teacher_checkpoint,
                            "teacher_bert_path": args.teacher_bert_path or args.bert_path,
                            "feature_weight": args.distill_feature_weight,
                            "action_weight": args.distill_action_weight,
                        },
                        "model_config": save_model.config.to_dict(),
                    },
                    save_path,
                )
                print(f"saved: {save_path}")

        if pbar is not None:
            pbar.close()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    train_model()
