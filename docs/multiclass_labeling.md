# Multiclass Consensus Labeling

This pipeline converts the existing binary spam/phishing email corpora into seven-category silver-label data. Two OpenAI-compatible models label every selected email independently; a row is accepted only when both return the same valid category.

Categories: `invoice`, `support`, `meeting`, `sales`, `spam`, `personal`, and `other`.

## 1. Configure credentials

Never put the API key in a command, source file, or Git-tracked environment file. Export a newly issued key in the shell:

```bash
export EMAILSYSTEM_LABEL_API_URL="http://192.168.1.79:21030/v1/chat/completions"
export EMAILSYSTEM_LABEL_API_KEY="<new-api-key>"
```

## 2. Run a small validation sample

```bash
python training/label_multiclass_consensus.py \
  --train-limit 20 \
  --validation-limit 10 \
  --output-dir data/finetune/multiclass_consensus_smoke
```

The default teacher models are `gemma4-26b` and `qwen3.6-27b`. Override them by passing `--model` exactly twice.
The request asks the server for a JSON object, disables model thinking through `chat_template_kwargs`, and allows up to 256 output tokens. Empty or malformed responses are retried and recorded with a bounded raw-output preview in `*_errors.jsonl`.

## 3. Generate the full silver-label dataset

Run in the foreground:

```bash
python training/label_multiclass_consensus.py \
  --train-limit 10000 \
  --validation-limit 1000 \
  --workers 8 \
  --output-dir data/finetune/multiclass_consensus
```

Or detach it from the terminal:

```bash
LABEL_ARGS="--train-limit 10000 --validation-limit 1000 --workers 8" \
  scripts/start_multiclass_labeling_detached.sh

tail -f outputs/logs/multiclass_consensus_labeling.log
scripts/kill_multiclass_labeling.sh
```

Rerunning with the same arguments resumes accepted and disagreement decisions. API errors are retried on the next run. Changing models, limits, sources, taxonomy, or prompt version requires a new output directory.

Outputs:

```text
data/finetune/multiclass_consensus/train.jsonl
data/finetune/multiclass_consensus/validation.jsonl
data/finetune/multiclass_consensus/*_annotations.jsonl
data/finetune/multiclass_consensus/*_disagreements.jsonl
data/finetune/multiclass_consensus/*_errors.jsonl
data/finetune/multiclass_consensus/manifest.json
```

`train.jsonl` and `validation.jsonl` contain runtime-compatible chat messages and original email fields. Disagreements never enter training data.

## 4. Train a multiclass LoRA adapter

```bash
conda activate vllm
python training/train_lora_classification.py \
  --model-path models/Qwen3-4B \
  --train-file data/finetune/multiclass_consensus/train.jsonl \
  --validation-file data/finetune/multiclass_consensus/validation.jsonl \
  --output-dir outputs/lora/qwen3_4b_multiclass_lora \
  --max-train-samples 10000 \
  --max-validation-samples 1000 \
  --balance-category-labels \
  --load-in-4bit \
  --bf16
```

Category balancing deterministically oversamples rare accepted classes. Inspect `manifest.json` first; if a class has very few accepted examples, collect or manually label more examples instead of relying only on oversampling.
The training script removes the source email `labels` metadata column after category-balanced sampling because `labels` is reserved by TRL for token-level targets. `category_label` and chat `messages` remain intact.

## 5. Evaluate all seven categories

```bash
python scripts/run_independent_eval.py \
  --backend vllm \
  --model-path <merged-multiclass-model> \
  --input data/finetune/multiclass_consensus/validation.jsonl \
  --quality-mode multiclass \
  --quality-limit 1000 \
  --skip-speed \
  --run-dir outputs/runs/qwen3_4b_multiclass_quality
```

The report includes seven-class accuracy, Macro-F1, per-class precision/recall/F1, and a multiclass confusion matrix. These are silver-label agreement metrics, not a substitute for a separately held-out human-reviewed test set.
