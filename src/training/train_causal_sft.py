#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

import torch
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset(path: Path, limit: int | None = None) -> Dataset:
    return Dataset.from_list(load_jsonl(path, limit=limit))


def pick_compute_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA SFT for NL->Scratch pseudocode with Qwen.")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--max-seq-length", type=int, default=768)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument(
        "--quantization",
        choices=["4bit", "none"],
        default="4bit",
        help="Use bitsandbytes 4-bit QLoRA ('4bit') or full-precision LoRA ('none'). "
             "Use 'none' on GPUs with ample VRAM (e.g. GH200 96GB).",
    )
    ap.add_argument("--save-limit", type=int, default=2)
    ap.add_argument("--resume-from-checkpoint", default=None)
    ap.add_argument("--disable-gradient-checkpointing", action="store_true")
    ap.add_argument(
        "--report-to",
        nargs="*",
        default=[],
        help="HF reporting integrations, e.g. --report-to wandb",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    distributed_state = PartialState()
    compute_dtype = pick_compute_dtype(args.dtype)
    if torch.cuda.is_available():
        device_map = {"": distributed_state.process_index}
    else:
        device_map = None

    if args.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=device_map,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
        model.config.use_cache = False
        if not args.disable_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)

    train_ds = build_dataset(args.train, limit=args.train_limit)
    val_ds = build_dataset(args.val, limit=args.val_limit)

    desired_config = {
        "output_dir": str(args.out),
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_length,
        "per_device_train_batch_size": args.bsz,
        "per_device_eval_batch_size": args.bsz,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "bf16": args.dtype == "bf16",
        "fp16": args.dtype == "fp16",
        "gradient_checkpointing": not args.disable_gradient_checkpointing,
        "optim": "paged_adamw_8bit" if args.quantization == "4bit" else "adamw_torch",
        "seed": args.seed,
        "report_to": args.report_to,
        "save_total_limit": args.save_limit,
        "load_best_model_at_end": False,
        "ddp_find_unused_parameters": False,
        "save_safetensors": True,
    }
    supported = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    training_args = SFTConfig(**{k: v for k, v in desired_config.items() if k in supported})

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "processing_class": tokenizer,
    }
    trainer_supported = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    if "processing_class" not in trainer_supported:
        trainer_kwargs.pop("processing_class", None)
        if "tokenizer" in trainer_supported:
            trainer_kwargs["tokenizer"] = tokenizer
    if "max_seq_length" in trainer_supported:
        trainer_kwargs["max_seq_length"] = args.max_seq_length
    if "dataset_text_field" in trainer_supported:
        trainer_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(
        json.dumps(
            {
                "saved_to": str(args.out),
                "world_size": distributed_state.num_processes,
                "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
                "device_map": device_map,
            }
        )
    )


if __name__ == "__main__":
    main()
