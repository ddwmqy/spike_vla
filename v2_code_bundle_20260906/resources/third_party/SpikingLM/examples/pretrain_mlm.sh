#!/usr/bin/env bash
set -euo pipefail

python -m torch.distributed.run \
  --nproc_per_node 1 \
  --nnodes 1 \
  scripts/pretrain_mlm.py \
  --model_name_or_path bert-base-uncased \
  --tokenized_dataset_path ./data/128_tokenized_data \
  --output_dir ./outputs/snn_full_110m \
  --T 4 \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --learning_rate 2e-4 \
  --max_train_steps 500000 \
  --num_warmup_steps 5000 \
  --checkpointing_steps 50000
