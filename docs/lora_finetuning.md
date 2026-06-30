# LoRA Fine-tuning For Email Classification

This project fine-tunes Qwen3-4B as a classification specialist first. The default training data merges the processed spam and phishing benchmarks, both projected into binary spam/ham labels, so the supervised target maps:

- `spam` -> `{"category":"spam","priority":"normal","confidence":0.95}`
- `ham` -> `{"category":"other","priority":"normal","confidence":0.90}`

The output remains compatible with the existing `ClassifyEmailSkill` and LangGraph workflow.

## 1. Build the benchmark splits

```bash
python scripts/build_spam_dataset.py
```

This writes deterministic train/validation/test splits under `data/processed/spam_benchmark/`. Keep `test.jsonl` only for final evaluation.

## 2. Prepare chat-format LoRA data

```bash
python training/prepare_lora_classification_data.py \
  --output-dir data/finetune/classification_lora \
  --max-body-chars 6000
```

The generated files are:

```text
data/finetune/classification_lora/train.jsonl
data/finetune/classification_lora/validation.jsonl
data/finetune/classification_lora/manifest.json
```

Each row contains `messages` in the same chat format used by the runtime classification prompt. By default, `training/prepare_lora_classification_data.py` merges `data/processed/spam_benchmark` and `data/processed/phishing_benchmark`; pass one or more `--input-dir` values to override that set.

## 3. Install training dependencies

```bash
pip install -e '.[finetune]'
```

For QLoRA 4-bit training, make sure `bitsandbytes` supports the local CUDA environment.

## 4. Train the adapter

```bash
python training/train_lora_classification.py \
  --model-path models/Qwen3-4B \
  --train-file data/finetune/classification_lora/train.jsonl \
  --validation-file data/finetune/classification_lora/validation.jsonl \
  --output-dir outputs/lora/qwen3_4b_classification_lora \
  --max-train-samples 10000 \
  --max-validation-samples 2000 \
  --load-in-4bit \
  --bf16
```

By default the script shuffles and caps training at 10,000 rows and validation at 2,000 rows. Set either cap to `0` to use the full split. Start with one epoch. If validation loss keeps improving and spam recall is still weak, try two epochs or `--lora-r 32 --lora-alpha 64`.

## 5. Merge for simple vLLM evaluation

```bash
python training/merge_lora.py \
  --base-model models/Qwen3-4B \
  --adapter outputs/lora/qwen3_4b_classification_lora \
  --output-dir models/Qwen3-4B-email-classifier
```

Then evaluate the merged model with the existing independent benchmark:

```bash
python scripts/run_independent_eval.py \
  --backend vllm \
  --model-path models/Qwen3-4B-email-classifier \
  --quality-limit 1000 \
  --speed-limit 100 \
  --run-dir outputs/runs/qwen3_4b_lora_eval
```

Compare against the previous baseline using `accuracy`, `macro_f1`, `spam_recall`, `accepted_coverage`, and per-task speed. The LoRA should mainly affect classification quality; summary/action/reply speed may change only slightly after merge.

## Notes

Do not train on `test.jsonl`. Use the validation split for prompt, epoch, and threshold decisions. To train the full seven-class classifier later, add human-labeled examples for `invoice`, `support`, `meeting`, `sales`, and `personal`; the spam dataset alone cannot teach those classes.
