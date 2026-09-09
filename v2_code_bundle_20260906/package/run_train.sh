#!/usr/bin/env bash
set -Eeuo pipefail

EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/data/users/wangs/VLA/TurboVLA}"
CODE_DIR="$EXPERIMENT_ROOT/code"
PYTHON_BIN="${PYTHON_BIN:-/data/users/wangs/miniconda3/envs/turbovla-libero/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/data/users/wangs/miniconda3/envs/turbovla-libero/bin/torchrun}"

GPU_IDS="${GPU_IDS:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_STEPS="${MAX_STEPS:-80000}"
WARMUP_STEPS="${WARMUP_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
RESUME_MODE="${RESUME_MODE:-none}"
DRY_RUN="${DRY_RUN:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$EXPERIMENT_ROOT/output/train}"
CHECKPOINT_DIR="$OUTPUT_ROOT/checkpoints"
LOG_DIR="$OUTPUT_ROOT/logs"
CHECKPOINT_PREFIX="turbovla_sdtv3_bert_snnfusion_learned"

DATA_ROOT="$REPO_DIR/data/libero"
DINOV3_PATH="$REPO_DIR/pretrained/dinov3-vitb16-pretrain-lvd1689m"
BERT_PATH="$REPO_DIR/pretrained/bert-base-uncased"
SDTV3_CHECKPOINT="$REPO_DIR/pretrained/V3_19.0M_1x4.pth"
SDTV3_SOURCE="$REPO_DIR/third_party/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Base/models.py"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-$REPO_DIR/outputs/checkpoints/libero_sdtv3_19m_2gpu/turbovla_sdtv3_19m_80000.pth}"
STATS_PATH="$REPO_DIR/experiments/libero/configs/libero_all4_stats.json"
TEXT_LAYOUT_PATH="$REPO_DIR/experiments/libero/configs/online_text_layout.json"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

[[ -n "$GPU_IDS" ]] || die "Set GPU_IDS to free physical GPUs, for example GPU_IDS=0,1,2,3"
[[ -x "$PYTHON_BIN" ]] || die "Python not executable: $PYTHON_BIN"
[[ -x "$TORCHRUN_BIN" ]] || die "torchrun not executable: $TORCHRUN_BIN"
for path in "$SDTV3_CHECKPOINT" "$SDTV3_SOURCE" "$TEACHER_CHECKPOINT" "$STATS_PATH" "$TEXT_LAYOUT_PATH"; do
  [[ -f "$path" ]] || die "Required file not found: $path"
done
for path in "$BERT_PATH" "$DINOV3_PATH" "$CODE_DIR/turbovla"; do
  [[ -d "$path" ]] || die "Required directory not found: $path"
done

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
LOG_FILE="$LOG_DIR/train_${NUM_GPUS}gpu_${RUN_ID}.log"

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
  --bert_path "$BERT_PATH"
  --text_encoder_type bert
  --freeze_text_encoder
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
  --student_init_checkpoint "$TEACHER_CHECKPOINT"
  --student_init_weight_source ema
  --teacher_checkpoint "$TEACHER_CHECKPOINT"
  --teacher_weight_source ema
  --teacher_bert_path "$BERT_PATH"
  --distill_feature_weight 0.1
  --distill_action_weight 0.1
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
  --dinov3_lr 1e-5
  --vision_encoder_lr 1e-5
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

echo "[INFO] model: SDT-V3 + frozen BERT + SNN fusion (exact Spike2Max) + learned-T readout + ANN action"
echo "[INFO] GPUs: $GPU_IDS; effective batch: $TARGET_GLOBAL_BATCH; accumulation: $GRAD_ACCUM_STEPS"
echo "[INFO] teacher/init: $TEACHER_CHECKPOINT (EMA)"
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
