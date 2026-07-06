#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for Qwen3 email classification.")
    parser.add_argument("--model-path", default="models/Qwen3-4B")
    parser.add_argument("--train-file", default="data/finetune/classification_lora/train.jsonl")
    parser.add_argument("--validation-file", default="data/finetune/classification_lora/validation.jsonl")
    parser.add_argument("--output-dir", default="outputs/lora/qwen3_4b_classification_lora")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true", help="Use bf16 training when the GPU supports it.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable QLoRA 4-bit loading with bitsandbytes.")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--max-train-samples", type=int, default=5000, help="Shuffle and keep this many train rows per epoch; 0 disables the cap.")
    parser.add_argument("--max-validation-samples", type=int, default=1000, help="Shuffle and keep this many validation rows; 0 disables the cap.")
    parser.add_argument(
        "--balance-category-labels",
        action="store_true",
        help="Soft-balance training categories with deterministic oversampling for rare classes.",
    )
    parser.add_argument(
        "--balance-validation-category-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the same soft category balancing to the validation sample.",
    )
    parser.add_argument(
        "--category-balance-max-ratio",
        type=float,
        default=3.0,
        help="Maximum target count ratio between the largest and smallest category after balancing.",
    )
    parser.add_argument(
        "--validation-category-balance-max-ratio",
        type=float,
        default=10.0,
        help="Maximum validation category ratio; validation never oversamples rows.",
    )
    parser.add_argument(
        "--resample-train-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When max train samples is enabled and epochs > 1, draw a different deterministic sample for each epoch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _assert_input(args.train_file)
    _assert_input(args.validation_file)

    from datasets import concatenate_datasets, load_dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    import torch
    from trl import SFTTrainer
    try:
        from trl import SFTConfig
    except ImportError:
        SFTConfig = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=quantization_config,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.validation_file},
    )
    dataset["train"] = _prepare_train_dataset(dataset["train"], args, concatenate_datasets)
    dataset["validation"] = _limit_dataset(
        dataset["validation"],
        args.max_validation_samples,
        seed=args.seed + 1,
        balance_category_labels=args.balance_validation_category_labels,
        category_balance_max_ratio=args.validation_category_balance_max_ratio,
        allow_oversampling=False,
    )
    dataset["train"] = _drop_reserved_labels_column(dataset["train"])
    dataset["validation"] = _drop_reserved_labels_column(dataset["validation"])
    _configure_logical_epoch_strategies(args)
    train_size = len(dataset["train"])
    validation_size = len(dataset["validation"])
    train_category_counts = _category_counts(dataset["train"])
    validation_category_counts = _category_counts(dataset["validation"])
    print(
        f"Using train={train_size} validation={validation_size} "
        f"max_train_samples={args.max_train_samples} max_validation_samples={args.max_validation_samples} "
        f"logical_epochs={args._logical_epochs} effective_epochs={args._effective_epochs} ",
        f"train_categories={train_category_counts} ",
        f"validation_categories={validation_category_counts}",
        flush=True,
    )

    def formatting_func(example: dict) -> str:
        return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

    trainer_signature = inspect.signature(SFTTrainer.__init__).parameters
    if SFTConfig is not None and "processing_class" in trainer_signature:
        training_args = _build_sft_config(SFTConfig, args)
        trainer_values = {
            "model": model,
            "processing_class": tokenizer,
            "args": training_args,
            "train_dataset": dataset["train"],
            "eval_dataset": dataset["validation"],
            "formatting_func": formatting_func,
        }
    else:
        training_args = _build_training_arguments(TrainingArguments, args)
        trainer_values = {
            "model": model,
            "tokenizer": tokenizer,
            "args": training_args,
            "train_dataset": dataset["train"],
            "eval_dataset": dataset["validation"],
            "formatting_func": formatting_func,
            "max_seq_length": args.max_seq_length,
            "packing": False,
        }
    trainer = SFTTrainer(**_filter_kwargs(SFTTrainer.__init__, trainer_values))
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    _write_run_config(args)


def _category_counts(split) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index in range(len(split)):
        row = split[index]
        category = row.get("category_label") or (row.get("labels") or {}).get("category")
        if category:
            counts[str(category)] = counts.get(str(category), 0) + 1
    return dict(sorted(counts.items()))


def _drop_reserved_labels_column(split):
    if "labels" not in split.column_names:
        return split
    return split.remove_columns(["labels"])


def _prepare_train_dataset(split, args: argparse.Namespace, concatenate_datasets_fn):
    args._logical_epochs = _logical_epoch_count(args.epochs)
    args._effective_epochs = args.epochs
    args._steps_per_logical_epoch = None
    if (
        not args.resample_train_each_epoch
        or args.max_train_samples <= 0
        or args._logical_epochs <= 1
        or len(split) <= args.max_train_samples
    ):
        return _limit_dataset(
            split,
            args.max_train_samples,
            seed=args.seed,
            balance_category_labels=getattr(args, "balance_category_labels", False),
            category_balance_max_ratio=getattr(args, "category_balance_max_ratio", 3.0),
        )

    pieces = [
        _limit_dataset(
            split,
            args.max_train_samples,
            seed=args.seed + epoch_index,
            balance_category_labels=getattr(args, "balance_category_labels", False),
            category_balance_max_ratio=getattr(args, "category_balance_max_ratio", 3.0),
        )
        for epoch_index in range(args._logical_epochs)
    ]
    args._effective_epochs = 1.0
    args._steps_per_logical_epoch = _optimizer_steps_per_epoch(args.max_train_samples, args)
    return concatenate_datasets_fn(pieces)


def _logical_epoch_count(epochs: float) -> int:
    if epochs <= 1:
        return 1
    if not float(epochs).is_integer():
        raise ValueError("--resample-train-each-epoch requires integer --epochs when epochs > 1")
    return int(epochs)


def _optimizer_steps_per_epoch(samples: int, args: argparse.Namespace) -> int:
    effective_batch = max(1, args.per_device_train_batch_size * args.gradient_accumulation_steps)
    return max(1, math.ceil(samples / effective_batch))


def _configure_logical_epoch_strategies(args: argparse.Namespace) -> None:
    args._effective_eval_strategy = args.eval_strategy
    args._effective_save_strategy = args.save_strategy
    args._effective_eval_steps = args.eval_steps
    args._effective_save_steps = args.save_steps
    if args._steps_per_logical_epoch is None:
        return
    if args.eval_strategy == "epoch":
        args._effective_eval_strategy = "steps"
        args._effective_eval_steps = args._steps_per_logical_epoch
    if args.save_strategy == "epoch":
        args._effective_save_strategy = "steps"
        args._effective_save_steps = args._steps_per_logical_epoch


def _limit_dataset(
    split,
    max_samples: int,
    *,
    seed: int,
    balance_category_labels: bool = False,
    category_balance_max_ratio: float = 3.0,
    allow_oversampling: bool = True,
):
    if balance_category_labels:
        sample_count = max_samples if max_samples > 0 else len(split)
        return _balanced_category_dataset(
            split,
            sample_count,
            seed=seed,
            max_ratio=category_balance_max_ratio,
            allow_oversampling=allow_oversampling,
        )
    if max_samples <= 0 or len(split) <= max_samples:
        return split
    return split.shuffle(seed=seed).select(range(max_samples))


def _balanced_category_dataset(
    split,
    sample_count: int,
    *,
    seed: int,
    max_ratio: float = 3.0,
    allow_oversampling: bool = True,
):
    if max_ratio < 1:
        raise ValueError("--category-balance-max-ratio must be at least 1")
    groups: dict[str, list[int]] = {}
    for index in range(len(split)):
        row = split[index]
        category = row.get("category_label") or (row.get("labels") or {}).get("category")
        if not category:
            raise ValueError("--balance-category-labels requires category_label or labels.category on every row")
        groups.setdefault(str(category), []).append(index)
    if not groups or sample_count <= 0:
        return split.select([])

    rng = random.Random(seed)
    labels = sorted(groups)
    for indexes in groups.values():
        rng.shuffle(indexes)
    allocations = _soft_balanced_allocations(
        {label: len(indexes) for label, indexes in groups.items()},
        sample_count,
        max_ratio=max_ratio,
        allow_oversampling=allow_oversampling,
    )
    selected = []
    for label in labels:
        indexes = groups[label]
        selected.extend(indexes[position % len(indexes)] for position in range(allocations[label]))
    rng.shuffle(selected)
    return split.select(selected)


def _soft_balanced_allocations(
    sizes: dict[str, int],
    sample_count: int,
    *,
    max_ratio: float,
    allow_oversampling: bool,
) -> dict[str, int]:
    """Allocate by sqrt frequency while keeping the largest class within max_ratio of the smallest."""
    labels = sorted(sizes)
    if not labels or sample_count <= 0:
        return {label: 0 for label in labels}
    weights = {label: math.sqrt(sizes[label]) for label in labels}
    floor_weight = max(weights.values()) / max_ratio
    weights = {label: max(weight, floor_weight) for label, weight in weights.items()}
    if allow_oversampling:
        capacities = {label: sample_count for label in labels}
    else:
        smallest = min(sizes.values())
        capacities = {label: min(sizes[label], math.floor(smallest * max_ratio)) for label in labels}
        sample_count = min(sample_count, sum(capacities.values()))

    allocations = {label: 0 for label in labels}
    for _ in range(sample_count):
        eligible = [label for label in labels if allocations[label] < capacities[label]]
        if not eligible:
            break
        label = min(eligible, key=lambda item: ((allocations[item] + 1) / weights[item], item))
        allocations[label] += 1

    if allow_oversampling:
        while True:
            smallest_label = min(labels, key=lambda item: (allocations[item], item))
            largest_label = max(labels, key=lambda item: (allocations[item], item))
            if allocations[largest_label] <= allocations[smallest_label] * max_ratio:
                break
            allocations[largest_label] -= 1
            allocations[smallest_label] += 1
    return allocations


def _build_training_arguments(training_arguments_cls, args: argparse.Namespace):
    values = _base_training_arg_values(args)
    values[_eval_strategy_argument(training_arguments_cls)] = getattr(args, "_effective_eval_strategy", args.eval_strategy)
    return training_arguments_cls(**_filter_kwargs(training_arguments_cls.__init__, values))


def _build_sft_config(sft_config_cls, args: argparse.Namespace):
    values = _base_training_arg_values(args)
    values[_eval_strategy_argument(sft_config_cls)] = getattr(args, "_effective_eval_strategy", args.eval_strategy)
    parameters = inspect.signature(sft_config_cls.__init__).parameters
    if "max_length" in parameters:
        values["max_length"] = args.max_seq_length
    elif "max_seq_length" in parameters:
        values["max_seq_length"] = args.max_seq_length
    if "packing" in parameters:
        values["packing"] = False
    return sft_config_cls(**_filter_kwargs(sft_config_cls.__init__, values))


def _base_training_arg_values(args: argparse.Namespace) -> dict:
    return {
        "output_dir": args.output_dir,
        "num_train_epochs": getattr(args, "_effective_epochs", args.epochs),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "eval_steps": getattr(args, "_effective_eval_steps", args.eval_steps),
        "save_strategy": getattr(args, "_effective_save_strategy", args.save_strategy),
        "save_steps": getattr(args, "_effective_save_steps", args.save_steps),
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "report_to": "none",
        "remove_unused_columns": True,
        "seed": args.seed,
    }


def _eval_strategy_argument(arguments_cls) -> str:
    parameters = inspect.signature(arguments_cls.__init__).parameters
    if "eval_strategy" in parameters:
        return "eval_strategy"
    if "evaluation_strategy" in parameters:
        return "evaluation_strategy"
    raise RuntimeError(f"{arguments_cls.__name__} does not expose an eval strategy parameter")


def _filter_kwargs(fn, values: dict) -> dict:
    parameters = inspect.signature(fn).parameters
    return {key: value for key, value in values.items() if key in parameters}


def _assert_input(path: str) -> None:
    if not Path(path).exists():
        raise SystemExit(f"Input file not found: {path}. Run training/prepare_lora_classification_data.py first.")


def _write_run_config(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
