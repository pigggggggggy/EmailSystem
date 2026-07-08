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
DeepSpeed: ZeRO-3 with PyTorch AdamW state offload (no CUDA extension build)
context length: 512 (fits 16 GB GPUs; override when memory permits)
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

## 5. Use the trained draft with vLLM

The draft must be paired with the exact target model used during training:

```bash
python scripts/run_independent_eval.py \
  --backend vllm \
  --model-path models/Qwen3-4B-email-classifier-ckpt1563 \
  --eagle3-model-path outputs/eagle3/qwen3_4b_email_classifier/checkpoint-939 \
  --speculative-tokens 3 \
  --quality-limit 20 \
  --speed-limit 10 \
  --run-dir outputs/runs/qwen3_4b_eagle3_smoke
```

Omit `--eagle3-model-path` to run the target-only baseline. EAGLE3 changes decoding speed, not the target model classification behavior.
The vLLM client tokenizes and truncates overlong email prompts automatically while reserving the requested output-token budget.


Run the FastAPI Agent on the last GPU:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/run_api.py \
  --backend vllm \
  --model-path models/Qwen3-4B-email-classifier-ckpt1563 \
  --eagle3-model-path outputs/eagle3/qwen3_4b_email_classifier/checkpoint-939 \
  --speculative-tokens 3 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9
```

Run the LangGraph Agent on recent Gmail messages over IMAP:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/run_imap_agent.py \
  --backend vllm \
  --model-path models/Qwen3-4B-email-classifier-ckpt1563 \
  --eagle3-model-path outputs/eagle3/qwen3_4b_email_classifier/checkpoint-939 \
  --speculative-tokens 3 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9 \
  --user "$EMAILSYSTEM_IMAP_USER" \
  --password-env EMAILSYSTEM_IMAP_PASSWORD \
  --limit 10 \
  --output outputs/predictions/qwen_eagle3_imap_predictions.jsonl
```

## 6. Train a mixed-task draft from parent-model outputs

The target model can generate all supervision automatically. Human-written summary, action-item, and reply labels are not required. The generator uses the same four prompts as the Agent, validates outputs, keeps each task balanced, and resumes by conversation ID after interruption.

Generate 2,500 accepted conversations per task for training and 250 per task for validation:

```bash
conda activate vllm
CUDA_VISIBLE_DEVICES=3 python training/generate_eagle3_distillation_data.py
```

The default output is `data/finetune/eagle3_mixed`. Invalid, empty, and length-truncated teacher responses are written to `*_rejected.jsonl`. A 2x deterministic candidate pool is used to replace rejected responses until each task reaches its target count.

Run generation detached from the terminal:

```bash
conda activate vllm
GPU_ID=3 scripts/start_eagle3_distillation_detached.sh
tail -f outputs/logs/qwen3_4b_email_eagle3_distillation.log
scripts/kill_eagle3_distillation.sh
```

Rerunning the same generation command resumes existing output. If generation arguments or the prompt version change, use a new `--output-dir`; mixing incompatible runs is rejected.

Train a fresh mixed-task EAGLE3 draft:

```bash
conda activate eagle3
scripts/start_eagle3_mixed_training_detached.sh
tail -f outputs/logs/qwen3_4b_email_eagle3_mixed.log
```

The mixed checkpoint is written under `outputs/eagle3/qwen3_4b_email_mixed`. It remains paired with `models/Qwen3-4B-email-classifier-ckpt1563` and does not replace the classification-only draft.

## 7. Train a draft for a finetuned target model

EAGLE3 drafts must be paired with the exact target model used during draft training. If the target model changes after LoRA merging or other finetuning, train a new draft instead of reusing an old draft.

The default finetuned target is:

```text
models/Qwen3-4B-email-multiclass-v2
```

Generate mixed-task distillation data from that target:

```bash
conda activate vllm
GPU_ID=3 scripts/start_eagle3_finetuned_distillation_detached.sh
tail -f outputs/logs/qwen3_4b_email_multiclass_eagle3_distillation.log
```

Train a new EAGLE3 draft from the generated data:

```bash
conda activate eagle3
scripts/start_eagle3_finetuned_training_detached.sh
tail -f outputs/logs/qwen3_4b_email_multiclass_eagle3.log
```

Defaults:

```text
target: models/Qwen3-4B-email-multiclass-v2
distillation data: data/finetune/eagle3_multiclass_v2_mixed
draft output: outputs/eagle3/qwen3_4b_email_multiclass_v2
GPUs: 0,1,2,3
context length: 512
```

Override a different merged target:

```bash
TARGET_MODEL=/path/to/merged-finetuned-qwen \
DISTILL_OUTPUT_DIR=data/finetune/eagle3_my_target_mixed \
RUN_NAME=my_target_eagle3_distillation \
scripts/start_eagle3_finetuned_distillation_detached.sh

TARGET_MODEL=/path/to/merged-finetuned-qwen \
TRAIN_DATA=data/finetune/eagle3_my_target_mixed/train.jsonl \
EVAL_DATA=data/finetune/eagle3_my_target_mixed/validation.jsonl \
OUTPUT_DIR=outputs/eagle3/my_target_eagle3 \
RUN_NAME=my_target_eagle3 \
scripts/start_eagle3_finetuned_training_detached.sh
```

Use the trained draft with the same target:

```bash
python scripts/run_independent_eval.py \
  --backend vllm \
  --model-path models/Qwen3-4B-email-multiclass-v2 \
  --eagle3-model-path outputs/eagle3/qwen3_4b_email_multiclass_v2/checkpoint-<step> \
  --speculative-tokens 3 \
  --quality-mode multiclass \
  --input data/finetune/multiclass_consensus_v3_maildir/validation.jsonl \
  --quality-limit 100 \
  --speed-limit 50 \
  --run-dir outputs/runs/qwen3_4b_multiclass_v2_eagle3_smoke
```

