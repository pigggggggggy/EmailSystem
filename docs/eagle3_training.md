# EAGLE3 Draft Training

This pipeline trains a new EAGLE3 draft model against the merged email-classification target model. It uses the official AngelSlim online trainer: the target model remains frozen and supplies hidden states/logits while the one-layer EAGLE3 draft is optimized.

## 1. Prepare conversation data

```bash
python training/prepare_eagle3_data.py
```

Defaults:

```text
train input: data/finetune/classification_lora/train.jsonl
validation input: data/finetune/classification_lora/validation.jsonl
train output: data/finetune/eagle3_classification/train.jsonl (5,000 rows)
validation output: data/finetune/eagle3_classification/validation.jsonl (500 rows)
```

The first dataset focuses on short classification JSON outputs. Add target-generated summary and reply conversations later if the main goal is accelerating long-form agent tasks.

## 2. Install AngelSlim in a separate environment

AngelSlim's current speculative-training dependencies may upgrade PyTorch and Transformers. Use a separate environment instead of the vLLM serving environment.

```bash
conda create -n eagle3 python=3.10 -y
conda activate eagle3
scripts/setup_angelslim_eagle3.sh
```

The setup script pins the inspected AngelSlim commit and installs the `speculative` extra under `third_party/AngelSlim`.

It also applies compatibility fixes for Qwen3 LLM dataset registration and uses PyTorch SDPA for the target model, so `flash-attn` is not required. If FlashAttention2 is installed and compatible, opt in with `TARGET_ATTN_IMPLEMENTATION=flash_attention_2`.

## 3. Train on four GPUs

```bash
conda activate eagle3
scripts/train_eagle3_online.sh
```

Defaults:

```text
target: models/Qwen3-4B-email-classifier-ckpt1563
GPUs: 0,1,2,3
DeepSpeed: ZeRO-3 with CPU optimizer offload
context length: 2048
epochs: 3
effective batch: 4 GPUs x batch 1 x accumulation 4 = 16
checkpoints: every epoch, keep 2
output: outputs/eagle3/qwen3_4b_email_classifier
```

Override settings through environment variables:

```bash
GPU_IDS=1,2,3 EPOCHS=1 MODEL_MAX_LENGTH=1024 scripts/train_eagle3_online.sh
```

## 4. Detached training

```bash
scripts/start_eagle3_training_detached.sh
tail -f outputs/logs/qwen3_4b_email_eagle3.log
scripts/kill_eagle3_training.sh qwen3_4b_email_eagle3
```

The existing `models/Qwen3-4B_eagle3` checkpoint is not overwritten. AngelSlim initializes a fresh draft from `training/configs/qwen3_4b_email_eagle3.json` because its online trainer accepts a draft config, not a warm-start checkpoint path.
