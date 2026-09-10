#!/usr/bin/env bash
# v2: SDT-V3 (SNN vision) + SpikingLM (SNN text, online by default mirroring the official
# recipe; TEXT_MODE=frozen for the legacy snnve recipe) + SNN fusion (Spike2Max, learned readout) + ANN action
# v3 addition (2026-09-07): TEXT_ENCODER_TYPE=spikinglm|sootspike selects the text backbone.
# Defaults keep v2 behavior byte-for-byte; sootspike requires TEXT_WEIGHTS_PATH/TEXT_SOURCE_PATH.
# KD teacher = OFFICIAL TurboVLA LIBERO checkpoint (DINOv3 ViT-B + BERT + ANN), feature 0.25 / action 0.5.
#
# Initialization mirrors the official recipe as closely as the SNN modules allow:
# official = HF-pretrained DINOv3/BERT + GroundingDINO-preloaded interaction, 80k from
# scratch. v2 = SDT-V3 classification ckpt + SpikingLM pretrain ckpt, everything else
# fresh; the SNN fusion stack cannot take the GroundingDINO preload (different
# architecture), so it initializes fresh with LayerScale. NO student warm-start
# (decided 2026-09-05: from-scratch for a clean comparison against official).
#
# DO NOT LAUNCH until the official checkpoint exists (see OFFICIAL_CKPT below).
set -Eeuo pipefail

EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$EXPERIMENT_ROOT/code"
REPO_DIR="${REPO_DIR:-/home/dwh/work/vla/libero_vla/resources}"
PYTHON_BIN="${PYTHON_BIN:-/home/dwh/venvs/turbovla-libero/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/dwh/venvs/turbovla-libero/bin/torchrun}"

# --- v2-specific inputs ------------------------------------------------------
# Official release: hf download H-EmbodVis/TurboVLA --local-dir <anywhere>; point
# OFFICIAL_CKPT at the LIBERO checkpoint inside (e.g. pretrained/TurboVLA/checkpoints/libero/....pth).
OFFICIAL_CKPT="${OFFICIAL_CKPT:-}"
TEACHER_WEIGHT_SOURCE="${TEACHER_WEIGHT_SOURCE:-model}"
DISTILL_FEATURE_WEIGHT="${DISTILL_FEATURE_WEIGHT:-0.25}"
DISTILL_ACTION_WEIGHT="${DISTILL_ACTION_WEIGHT:-0.5}"
# online (default) mirrors the official recipe: the pretrained text encoder is
# fine-tuned on LIBERO at the base lr. frozen = legacy snnve recipe (kept as fallback
# if the online smoke test shows instability).
TEXT_MODE="${TEXT_MODE:-online}"
# -----------------------------------------------------------------------------

GPU_IDS="${GPU_IDS:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-80000}"
WARMUP_STEPS="${WARMUP_STEPS:-10000}"
# trainer requires lr_schedule_steps > warmup_steps (trainer.py:907); a short
# smoke run would otherwise abort at startup. Clamp instead of failing.
if (( WARMUP_STEPS >= MAX_STEPS )); then
  WARMUP_STEPS=$((MAX_STEPS / 8))
  echo "[WARN] WARMUP_STEPS >= MAX_STEPS; clamped warmup to ${WARMUP_STEPS} for this run"
fi
SAVE_STEPS="${SAVE_STEPS:-5000}"
RESUME_MODE="${RESUME_MODE:-none}"
DRY_RUN="${DRY_RUN:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$EXPERIMENT_ROOT/output/train_v2}"
CHECKPOINT_DIR="$OUTPUT_ROOT/checkpoints"
LOG_DIR="$OUTPUT_ROOT/logs"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-turbovla_v2_snnfusion}"

DATA_ROOT="$REPO_DIR/data/libero"
DINOV3_PATH="$REPO_DIR/pretrained/dinov3-vitb16-pretrain-lvd1689m"
BERT_PATH="$REPO_DIR/pretrained/bert-base-uncased"
SPIKINGLM_WEIGHTS="$REPO_DIR/pretrained/SpikingLM"
SPIKINGLM_SOURCE="$REPO_DIR/third_party/SpikingLM"
SDTV3_CHECKPOINT="$REPO_DIR/pretrained/V3_19.0M_1x4.pth"
SDTV3_SOURCE="$REPO_DIR/third_party/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Base/models.py"
STATS_PATH="$REPO_DIR/experiments/libero/configs/libero_all4_stats.json"
TEXT_LAYOUT_PATH="$REPO_DIR/experiments/libero/configs/online_text_layout.json"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

# Text encoder selection (v3): spikinglm (v2 default) or sootspike (v3, weights pending
# as of 2026-09-07). TEXT_WEIGHTS_PATH = weights + tokenizer dir; TEXT_SOURCE_PATH =
# model implementation checkout. Defaults below reproduce v2 behavior exactly.
TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-spikinglm}"
TEXT_WEIGHTS_PATH="${TEXT_WEIGHTS_PATH:-}"
TEXT_SOURCE_PATH="${TEXT_SOURCE_PATH:-}"
TEXT_TIME_STEPS="${TEXT_TIME_STEPS:-4}"
case "$TEXT_ENCODER_TYPE" in
  spikinglm)
    TEXT_WEIGHTS_PATH="${TEXT_WEIGHTS_PATH:-$SPIKINGLM_WEIGHTS}"
    TEXT_SOURCE_PATH="${TEXT_SOURCE_PATH:-$SPIKINGLM_SOURCE}"
    TEXT_SOURCE_ARG=(--spikinglm_source_path)
    ;;
  sootspike)
    [[ -n "$TEXT_WEIGHTS_PATH" ]] || die "TEXT_ENCODER_TYPE=sootspike requires TEXT_WEIGHTS_PATH (weights + tokenizer dir)"
    [[ -n "$TEXT_SOURCE_PATH" ]] || die "TEXT_ENCODER_TYPE=sootspike requires TEXT_SOURCE_PATH (sootspike model code dir)"
    TEXT_SOURCE_ARG=(--text_source_path)
    ;;
  *) die "TEXT_ENCODER_TYPE must be 'spikinglm' or 'sootspike', got: $TEXT_ENCODER_TYPE" ;;
esac

[[ -n "$GPU_IDS" ]] || die "Set GPU_IDS to free physical GPUs, for example GPU_IDS=0,1,2,3"
[[ -n "$OFFICIAL_CKPT" ]] || die "Set OFFICIAL_CKPT to the official TurboVLA LIBERO checkpoint (download H-EmbodVis/TurboVLA first)"
[[ -x "$PYTHON_BIN" ]] || die "Python not executable: $PYTHON_BIN"
[[ -x "$TORCHRUN_BIN" ]] || die "torchrun not executable: $TORCHRUN_BIN"
[[ -f "$OFFICIAL_CKPT" ]] || die "Official checkpoint not found: $OFFICIAL_CKPT"
if [[ "$TEXT_ENCODER_TYPE" == "spikinglm" ]]; then
  [[ -d "$SPIKINGLM_WEIGHTS" ]] || die "SpikingLM weights directory not found: $SPIKINGLM_WEIGHTS"
  [[ -d "$SPIKINGLM_SOURCE" ]] || die "SpikingLM source directory not found: $SPIKINGLM_SOURCE"
else
  [[ -d "$TEXT_WEIGHTS_PATH" ]] || die "TEXT_WEIGHTS_PATH directory not found: $TEXT_WEIGHTS_PATH"
  [[ -d "$TEXT_SOURCE_PATH" ]] || die "TEXT_SOURCE_PATH directory not found: $TEXT_SOURCE_PATH"
fi
for path in "$SDTV3_CHECKPOINT" "$SDTV3_SOURCE" "$STATS_PATH" "$TEXT_LAYOUT_PATH"; do
  [[ -f "$path" ]] || die "Required file not found: $path"
done
for path in "$BERT_PATH" "$DINOV3_PATH" "$CODE_DIR/turbovla"; do
  [[ -d "$path" ]] || die "Required directory not found: $path"
done

# TEXT_MODE gate: the official trainer defaults to frozen text while its prose
# says "online BERT" — the released checkpoint's recorded model_config is the
# arbiter. Refuse to start with a mismatched mode unless FORCE_TEXT_MODE=1.
FORCE_TEXT_MODE="${FORCE_TEXT_MODE:-0}"
OFFICIAL_TEXT_FROZEN="$("$PYTHON_BIN" - "$OFFICIAL_CKPT" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
if isinstance(config, dict):
    text_cfg = config.get("text") or {}
    frozen = text_cfg.get("frozen", None)
    if frozen is None:
        frozen = text_cfg.get("freeze_text_encoder", None)
    print(frozen if frozen is not None else "unknown")
else:
    print("unknown")
PY
)" || die "Failed to inspect the official checkpoint's text config"
case "$OFFICIAL_TEXT_FROZEN" in
  True|False)
    echo "[INFO] official ckpt self-report: text.frozen=$OFFICIAL_TEXT_FROZEN; TEXT_MODE=$TEXT_MODE"
    if [[ "$FORCE_TEXT_MODE" != "1" ]]; then
      if [[ "$OFFICIAL_TEXT_FROZEN" == "True" && "$TEXT_MODE" == "online" ]] ||
         [[ "$OFFICIAL_TEXT_FROZEN" == "False" && "$TEXT_MODE" == "frozen" ]]; then
        die "TEXT_MODE=$TEXT_MODE contradicts the official checkpoint (text.frozen=$OFFICIAL_TEXT_FROZEN). Use TEXT_MODE=$([ "$OFFICIAL_TEXT_FROZEN" == "True" ] && echo frozen || echo online) or set FORCE_TEXT_MODE=1 to override."
      fi
    fi
    ;;
  *)
    echo "[WARN] official checkpoint does not record text.frozen; TEXT_MODE=$TEXT_MODE is NOT verified against official"
    ;;
esac

GPU_IDS="${GPU_IDS//[[:space:]]/}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
(( NUM_GPUS >= 1 )) || die "GPU_IDS cannot be empty"
for gpu_id in "${GPU_ARRAY[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ ]] || die "Invalid GPU ID: $gpu_id"
done
MICRO_GLOBAL=$((BATCH_SIZE * NUM_GPUS))
(( TARGET_GLOBAL_BATCH % MICRO_GLOBAL == 0 )) || \
  die "TARGET_GLOBAL_BATCH must be divisible by BATCH_SIZE x NUM_GPUS ($MICRO_GLOBAL)"
GRAD_ACCUM_STEPS=$((TARGET_GLOBAL_BATCH / MICRO_GLOBAL))

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$EXPERIMENT_ROOT/tmp"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$CODE_DIR:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TMPDIR="$EXPERIMENT_ROOT/tmp"
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT

DATASET_DIRS="$DATA_ROOT/libero_10_no_noops/1.0.0,$DATA_ROOT/libero_goal_no_noops/1.0.0,$DATA_ROOT/libero_object_no_noops/1.0.0,$DATA_ROOT/libero_spatial_no_noops/1.0.0"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_v2_${NUM_GPUS}gpu_${RUN_ID}.log"

case "$TEXT_MODE" in
  online) TEXT_ARGS=(--train_text_encoder) ;;
  frozen) TEXT_ARGS=(--freeze_text_encoder) ;;
  *) die "TEXT_MODE must be 'online' or 'frozen', got: $TEXT_MODE" ;;
esac

COMMAND=(
  "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS"
  --module turbovla.training.train_mixed
  --dataset_dir "$DATA_ROOT/libero_10_no_noops/1.0.0"
  --dataset_dirs "$DATASET_DIRS"
  --stats_path "$STATS_PATH"
  --stats_key libero_all4_no_noops
  --text_layout_path "$TEXT_LAYOUT_PATH"
  --dinov3_path "$DINOV3_PATH"
  --vision_encoder_type sdtv3_19m
  --vision_model_path "$SDTV3_CHECKPOINT"
  --vision_pretrained_checkpoint "$SDTV3_CHECKPOINT"
  --vision_model_source_path "$SDTV3_SOURCE"
  --vision_image_size 224
  --vision_output_grid_size 14
  --bert_path "$TEXT_WEIGHTS_PATH"
  --text_encoder_type "$TEXT_ENCODER_TYPE"
  "${TEXT_SOURCE_ARG[@]}" "$TEXT_SOURCE_PATH"
  --spikinglm_time_steps "$TEXT_TIME_STEPS"
  ${TEXT_ARGS[@]}
  --downstream_type spiking
  --downstream_time_steps 4
  --downstream_lif_tau 2.0
  --downstream_lif_backend cupy
  --spike_reset_policy per_forward
  --spike_attention_normalization exact
  --spike_cross_attention_dim 512
  --spike_cross_attention_heads 8
  --spike_cross_layer_scale_init 0.01
  --spike_self_layer_scale_init 0.1
  --spike_ffn_layer_scale_init 0.1
  --spike_collect_diagnostics
  --spike_diagnostics_freq 500
  --spike_eval_freq 5000
  --spike_firing_rate_weight 0.0
  --temporal_readout learned
  --action_head_type ann
  --teacher_checkpoint "$OFFICIAL_CKPT"
  --teacher_weight_source "$TEACHER_WEIGHT_SOURCE"
  --teacher_bert_path "$BERT_PATH"
  --distill_feature_weight "$DISTILL_FEATURE_WEIGHT"
  --distill_action_weight "$DISTILL_ACTION_WEIGHT"
  --no_require_feature_enhancer_preload
  --no_load_text_projection_from_init
  --no_require_text_proj_preload
  --checkpoint_dir "$CHECKPOINT_DIR"
  --checkpoint_prefix "$CHECKPOINT_PREFIX"
  --resume_mode "$RESUME_MODE"
  --batch_size "$BATCH_SIZE"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --precision fp32
  --dinov3_precision fp32
  --vision_precision fp32
  --no_freeze_backbones
  --lr 5e-5
  --head_lr 5e-5
  --dinov3_lr 5e-5
  --vision_encoder_lr 5e-5
  --weight_decay 1e-10
  --head_weight_decay 1e-10
  --dinov3_weight_decay 1e-10
  --vision_encoder_weight_decay 1e-10
  --max_steps "$MAX_STEPS"
  --lr_schedule_steps "$MAX_STEPS"
  --warmup_steps "$WARMUP_STEPS"
  --min_lr_ratio 1.0
  --save_steps "$SAVE_STEPS"
  --num_workers "$NUM_WORKERS"
  --seed 42
  --no_allow_hf_download
)

echo "[INFO] model: SDT-V3 + $TEXT_ENCODER_TYPE (text: $TEXT_MODE) + SNN fusion (exact Spike2Max) + learned-T readout + ANN action"
echo "[INFO] KD teacher: $OFFICIAL_CKPT (DINOv3 ViT-B + BERT, forced ANN; feature $DISTILL_FEATURE_WEIGHT / action $DISTILL_ACTION_WEIGHT)"
echo "[INFO] init: from scratch (SDT-V3 classification ckpt + $TEXT_ENCODER_TYPE pretrain; no student warm-start)"
echo "[INFO] GPUs: $GPU_IDS; effective batch: $TARGET_GLOBAL_BATCH; accumulation: $GRAD_ACCUM_STEPS"
echo "[INFO] checkpoints: $CHECKPOINT_DIR"
echo "[INFO] log: $LOG_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$EXPERIMENT_ROOT"
RESOLVED_PACKAGE="$($PYTHON_BIN -c 'import turbovla; print(turbovla.__file__)')"
[[ "$RESOLVED_PACKAGE" == "$CODE_DIR"/* ]] || \
  die "Python resolved the wrong turbovla package: $RESOLVED_PACKAGE"
"${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
