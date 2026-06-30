#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
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
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true", help="Use bf16 training when the GPU supports it.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable QLoRA 4-bit loading with bitsandbytes.")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260630)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _assert_input(args.train_file)
    _assert_input(args.validation_file)

    from datasets import load_dataset
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


def _build_training_arguments(training_arguments_cls, args: argparse.Namespace):
    values = _base_training_arg_values(args)
    values[_training_strategy_argument(training_arguments_cls)] = "steps"
    return training_arguments_cls(**_filter_kwargs(training_arguments_cls.__init__, values))


def _build_sft_config(sft_config_cls, args: argparse.Namespace):
    values = _base_training_arg_values(args)
    values[_training_strategy_argument(sft_config_cls)] = "steps"
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
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "report_to": "none",
        "remove_unused_columns": True,
        "seed": args.seed,
    }


def _training_strategy_argument(arguments_cls) -> str:
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
