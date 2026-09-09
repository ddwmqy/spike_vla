# SpikingLM: Towards Fully Spiking Language Model

Official code release for **SpikingLM: Towards Fully Spiking Language Model**, accepted by **ICML 2026**.

SpikingLM is a fully spiking BERT-style language model designed to bridge energy-efficient spiking neural networks and modern language modeling. It addresses two core obstacles in spiking language models: gradient degradation from dead neurons and weak token selectivity after removing Softmax.

## Highlights

- **Fully spiking language model:** removes floating-point Softmax/GELU-style attention dependencies and uses spiking computation in the BERT backbone.
- **Distribution-aware Scaling:** learnable Q/K/V scaling factors rescale linear outputs into an activation-friendly range. In this release, `q_lam`, `k_lam`, and `v_lam` are initialized to `7` and remain trainable.
- **No extra inference overhead:** Distribution-aware Scaling can be fused into the preceding linear projection weights at inference.
- **Spike2Max attention:** restores winner-takes-all token competition through max-subtraction and base-2 exponentiation.
- **ICML main result:** SpikingLM achieves **77.1 average GLUE score** with **5.79 mJ** estimated energy at `T=4`.

## Repository Layout

```text
spiking_bert/modeling_spiking_bert.py  # SpikingLM model implementation
spiking_bert/__init__.py               # model exports
scripts/pretrain_mlm.py                # masked-language-model pre-training
scripts/finetune_glue.py               # GLUE fine-tuning and checkpoint loading
configs/bert_base_t4.json              # BERT-base T=4 configuration
examples/pretrain_mlm.sh               # pre-training command template
examples/finetune_glue.sh              # fine-tuning command template
results/glue_logs/                     # ICML GLUE logs and best-result summary
```

## Installation

```bash
git clone https://github.com/hamings1/SpikingLM.git
cd SpikingLM
pip install -r requirements.txt
```

`spikingjelly` requires a CUDA/CuPy setup compatible with your local PyTorch and GPU runtime.

## Model Details

The final ICML model is implemented in `spiking_bert/modeling_spiking_bert.py`.

Key attention components:

```python
self.q_lam = nn.Parameter(torch.ones(self.all_head_size) * 7)
self.k_lam = nn.Parameter(torch.ones(self.all_head_size) * 7)
self.v_lam = nn.Parameter(torch.ones(self.all_head_size) * 7)
self.learnmax = FP16OptimizedExp2Softmax(T=config.T)
attention_probs = self.learnmax(attention_scores)
```

For a Q/K/V projection `y = Linear(x)`, Distribution-aware Scaling applies `lambda * y`. At inference, this can be folded into the preceding linear layer:

```python
W_fused = lambda[:, None] * W
b_fused = lambda * b
```

This gives the same forward output while removing the explicit scaling operation.

## Pre-training

The ICML setting pre-trains a BERT-base backbone with `T=4`, sequence length `128`, learning rate `2e-4`, and `5,000` warmup steps on a mixture of English corpora: STORIES, BookCorpus, CC-News, OpenWebText, and English Wikipedia.

The script supports either dataset names or a pre-tokenized `datasets.save_to_disk()` directory. A minimal command template is provided:

```bash
bash examples/pretrain_mlm.sh
```

Example:

```bash
python -m torch.distributed.run \
  --nproc_per_node 8 \
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
  --max_train_steps 800000 \
  --num_warmup_steps 5000 \
  --checkpointing_steps 50000
```

## GLUE Fine-tuning

Fine-tuning updates all model parameters end-to-end with a task-specific classifier head.

```bash
bash examples/finetune_glue.sh
```

Example:

```bash
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
```

The paper uses batch size `32`, max sequence length `128`, and task-specific learning rates/epochs for GLUE.

## Results

The best ICML GLUE logs are included in `results/glue_logs/`.

| Model | Time | MNLI m/mm | QQP F1 | QNLI | SST-2 | CoLA | STS-B | MRPC F1 | RTE | Avg. | Energy mJ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SpikingLM | 4 | 75.8/76.0 | 84.7 | 84.6 | 90.1 | 50.4 | 82.4 | 87.9 | 61.7 | **77.1** | **5.79** |

Per-task logs, run IDs, selected epochs, and the LaTeX table row are available under:

```text
results/glue_logs/
```

## Checkpoints

Large model weights are intentionally not tracked by git. The released weights are available on Hugging Face:

```text
https://huggingface.co/hamingsi/SpikingLM
```

Expected checkpoint layout:

```text
checkpoint_dir/
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  vocab.txt
```

## Load a Checkpoint

```python
from safetensors.torch import load_file
from transformers import AutoConfig
from spiking_bert import BertForSequenceClassification

checkpoint_dir = "path/to/checkpoint_dir"

config = AutoConfig.from_pretrained(checkpoint_dir)
config.T = 4
config._attn_implementation = "eager"
config.num_labels = 2

model = BertForSequenceClassification(config)
state = load_file(f"{checkpoint_dir}/model.safetensors")
model_state = model.state_dict()

for key, value in state.items():
    if key in model_state and value.shape == model_state[key].shape:
        model_state[key] = value

model.load_state_dict(model_state)
```

## Verification

This release was smoke-tested on an NVIDIA H100 MIG environment:

- model instantiation: `109,541,994` parameters
- `q_lam/k_lam/v_lam` initialization: `7.0`
- attention normalization: `FP16OptimizedExp2Softmax`
- masked-language-model pre-training: `2` steps completed and saved checkpoint
- GLUE-style fine-tuning: `2` steps completed and saved checkpoint

## Citation

```bibtex
@inproceedings{liang2026spikinglm,
  title     = {SpikingLM: Towards Fully Spiking Language Model},
  author    = {Liang, Yu and Zhou, Zijian and Wei, Wenjie and Wang, Shuai and Cao, Honglin and Belatreche, Ammar and Yang, Yu and Zhang, Malu and Yang, Yang and Li, Haizhou},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  address   = {Seoul, South Korea}
}
```
