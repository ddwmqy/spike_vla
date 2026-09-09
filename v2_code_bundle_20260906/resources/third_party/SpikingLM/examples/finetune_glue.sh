#!/usr/bin/env bash
set -euo pipefail

python scripts/finetune_glue.py \
  --model_name_or_path bert-base-uncased \
  --pretrained_checkpoint ./outputs/snn_full_110m \
  --task_name sst2 \
  --output_dir ./outputs/glue-sst2 \
  --T 4 \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 3
