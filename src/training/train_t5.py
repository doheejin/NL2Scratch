"""
Minimal seq2seq training loop for NL -> scratchblocks pseudocode.

Uses Hugging Face Transformers + Datasets. Designed to consume the
data/train.jsonl, data/val.jsonl, data/test.jsonl files produced by
generate_training_data.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import inspect
from datasets import Dataset
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def load_jsonl(path: Path, limit: int | None = None):
    items = []
    for i, line in enumerate(path.open()):
        if not line.strip():
            continue
        if limit is not None and i >= limit:
            break
        items.append(json.loads(line))
    return items


def build_dataset(path: Path, tokenizer, max_source: int, max_target: int, limit: int | None = None):
    data = load_jsonl(path, limit=limit)
    # Join pseudocode lines with \n to preserve block boundaries.
    for d in data:
        d["target_text"] = "\n".join(d["pseudocode"])
    ds = Dataset.from_list(data)

    def tokenize(batch):
        src = tokenizer(
            batch["nl"],
            max_length=max_source,
            truncation=True,
            padding=False,
        )
        tgt = tokenizer(
            batch["target_text"],
            max_length=max_target,
            truncation=True,
            padding=False,
        )
        batch_out = {**src, "labels": tgt["input_ids"]}
        return batch_out

    return ds.map(tokenize, batched=True, remove_columns=ds.column_names)


class SampleGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, raw_examples, max_source: int, max_target: int):
        self.tokenizer = tokenizer
        self.raw_examples = raw_examples
        self.max_source = max_source
        self.max_target = max_target

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not self.raw_examples or model is None:
            return control

        was_training = model.training
        model.eval()

        print(f"\n=== Sample Generations @ step {state.global_step} epoch {state.epoch} ===")
        for idx, example in enumerate(self.raw_examples, start=1):
            encoded = self.tokenizer(
                example["nl"],
                max_length=self.max_source,
                truncation=True,
                padding=False,
                return_tensors="pt",
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=self.max_target,
                )
            pred = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            gold = "\n".join(example["pseudocode"])
            print(f"[sample {idx}] key={example.get('key', '')}")
            print(f"NL: {example['nl']}")
            print("PRED:")
            print(pred)
            print("GOLD:")
            print(gold)
            print("---")

        if was_training:
            model.train()
        return control


def main():
    ap = argparse.ArgumentParser(description="Train NL->Scratchblocks model.")
    ap.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--val", type=Path, default=Path("data/val.jsonl"))
    ap.add_argument("--train-limit", type=int, default=None, help="Optional cap on number of train examples (to reduce memory).")
    ap.add_argument("--val-limit", type=int, default=None, help="Optional cap on number of val examples (to reduce memory).")
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/nl2scratch"))
    ap.add_argument("--max-source", type=int, default=256)
    ap.add_argument("--max-target", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", help="Enable fp16 if GPU supports it.")
    ap.add_argument("--eval-steps", type=int, default=None, help="Run evaluation every N steps (overrides epoch eval).")
    ap.add_argument("--save-steps", type=int, default=None, help="Save checkpoint every N steps (overrides epoch save).")
    ap.add_argument("--sample-predictions", type=int, default=0, help="Print N validation generations at each evaluation boundary.")
    ap.add_argument("--report-to", nargs="*", default=[], help="HF reporting integrations, e.g. --report-to wandb")
    ap.add_argument("--run-name", default=None, help="Optional run name for logging integrations.")
    args = ap.parse_args()

    tokenizer = T5TokenizerFast.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model)

    train_ds = build_dataset(args.train, tokenizer, args.max_source, args.max_target, limit=args.train_limit)
    val_ds = build_dataset(args.val, tokenizer, args.max_source, args.max_target, limit=args.val_limit)
    val_raw = load_jsonl(args.val, limit=args.val_limit)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Build kwargs dynamically for backwards compatibility with older transformers.
    desired_kwargs = {
        "output_dir": str(args.out),
        "per_device_train_batch_size": args.bsz,
        "per_device_eval_batch_size": args.bsz,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "logging_steps": 50,
        "seed": args.seed,
        "fp16": args.fp16,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "report_to": args.report_to,
        "run_name": args.run_name,
        # Nice-to-have if supported:
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "predict_with_generate": True,
        "generation_max_length": args.max_target,
        "load_best_model_at_end": True,
    }
    # HF changed arg name from evaluation_strategy -> eval_strategy in newer versions.
    supported = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    eval_key = "evaluation_strategy" if "evaluation_strategy" in supported else "eval_strategy"
    if eval_key not in supported:
        eval_key = None
    if args.eval_steps and eval_key:
        desired_kwargs[eval_key] = "steps"
        desired_kwargs["eval_steps"] = args.eval_steps
    if args.save_steps:
        desired_kwargs["save_strategy"] = "steps"
        desired_kwargs["save_steps"] = args.save_steps
    filtered_kwargs = {k: v for k, v in desired_kwargs.items() if k in supported}
    # If load_best_model_at_end is present, make sure we also provide matching
    # eval/save strategies; otherwise drop it to avoid version mismatch errors.
    if "load_best_model_at_end" in supported:
        if "evaluation_strategy" in supported and "save_strategy" in supported:
            filtered_kwargs.setdefault("evaluation_strategy", "epoch")
            filtered_kwargs.setdefault("save_strategy", "epoch")
        else:
            filtered_kwargs.pop("load_best_model_at_end", None)

    training_args = TrainingArguments(**filtered_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
    }
    trainer_supported = set(inspect.signature(Trainer.__init__).parameters.keys())
    if "tokenizer" in trainer_supported:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_supported:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    if args.sample_predictions > 0:
        trainer.add_callback(
            SampleGenerationCallback(
                tokenizer=tokenizer,
                raw_examples=val_raw[: args.sample_predictions],
                max_source=args.max_source,
                max_target=args.max_target,
            )
        )

    trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"Training complete. Saved to {args.out}")


if __name__ == "__main__":
    main()
