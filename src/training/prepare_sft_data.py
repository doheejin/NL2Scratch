#!/usr/bin/env python3
"""Build SFT-format JSONL (prompt + target + concatenated 'text' field) for the
NL2Scratch causal-LM training, defaulting to the public HF dataset.

Usage (default — load from HuggingFace):
    python3 src/training/prepare_sft_data.py --output-dir runs/sft_data/<tag>

Override with local JSONL files (must each carry {key, nl, pseudocode}):
    python3 src/training/prepare_sft_data.py \\
      --train data/train.jsonl --val data/val.jsonl --test data/test.jsonl \\
      --output-dir runs/sft_data/<tag>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You convert natural-language Scratch instructions into valid Scratch pseudocode. "
    "Output only pseudocode. Use one block per line. Use exactly 4 leading spaces for nested "
    "blocks inside control structures. Preserve names, messages, numbers, and ordering."
)


def load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_prompt(nl: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{nl.strip()}\n"
        "<|assistant|>\n"
    )


def convert_record(record: dict) -> dict:
    target = "\n".join(record["pseudocode"])
    prompt = build_prompt(record["nl"])
    return {
        "key": record["key"],
        "nl": record["nl"],
        "pseudocode": record["pseudocode"],
        "prompt": prompt,
        "target": target,
        "completion": target,
        "text": prompt + target,
    }


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def iter_split(split_name: str, local_path: Path | None, hf_dataset: str | None):
    """Yield raw records for one split — local JSONL if provided, else HF Hub."""
    if local_path is not None:
        yield from load_jsonl(local_path)
        return
    from datasets import load_dataset  # imported lazily to avoid the dep when unused
    hf_split = {"train": "train", "val": "validation", "test": "test"}[split_name]
    for row in load_dataset(hf_dataset, split=hf_split):
        yield row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--hf-dataset", default="Heejindo/nl2scratch", help="HF dataset id (used when --train/--val/--test are omitted).")
    ap.add_argument("--train", type=Path, default=None, help="Override: local train.jsonl.")
    ap.add_argument("--val", type=Path, default=None, help="Override: local val.jsonl.")
    ap.add_argument("--test", type=Path, default=None, help="Override: local test.jsonl.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split_name, local_path in (("train", args.train), ("val", args.val), ("test", args.test)):
        rows = [convert_record(r) for r in iter_split(split_name, local_path, args.hf_dataset)]
        write_jsonl(rows, args.output_dir / f"{split_name}.jsonl")
        counts[split_name] = len(rows)

    print(json.dumps({"output_dir": str(args.output_dir), "source": args.hf_dataset if args.train is None else "local", "counts": counts}))


if __name__ == "__main__":
    main()
