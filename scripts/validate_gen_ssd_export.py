"""Validate matched Gen-SSD exports exactly as 4K response-only training data."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from reasoning.data import digest, prompt_id
from reasoning.gen_ssd import serialized_training_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--cutoff", type=int, default=4096)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    guided_path = args.dataset_dir / "guided.json"
    existing_path = args.dataset_dir / "existing.json"
    guided = load_dataset("json", data_files=str(guided_path), split="train")
    existing = load_dataset("json", data_files=str(existing_path), split="train")
    if len(guided) != args.expected or len(existing) != args.expected:
        raise ValueError((len(guided), len(existing), args.expected))

    guided_rows = list(guided)
    existing_rows = list(existing)
    guided_ids = [prompt_id(row["instruction"]) for row in guided_rows]
    existing_ids = [prompt_id(row["instruction"]) for row in existing_rows]
    if guided_ids != existing_ids or len(set(guided_ids)) != args.expected:
        raise ValueError("guided and existing arms must have identical ordered unique question IDs")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    guided_lengths = [serialized_training_tokens(tokenizer, x["instruction"], x["output"]) for x in guided_rows]
    existing_lengths = [serialized_training_tokens(tokenizer, x["instruction"], x["output"]) for x in existing_rows]
    if max(guided_lengths + existing_lengths) > args.cutoff:
        raise ValueError("serialized prompt + solution + chat template exceeds cutoff")

    receipt = {
        "schema": "reasoning-sft-gen-ssd-export-validation-v1",
        "rows_per_arm": args.expected,
        "identical_ordered_question_ids": True,
        "unique_question_ids": len(set(guided_ids)),
        "training_cutoff": args.cutoff,
        "max_guided_training_tokens": max(guided_lengths),
        "max_existing_training_tokens": max(existing_lengths),
        "guided_sha256": digest(guided_path),
        "existing_sha256": digest(existing_path),
        "loader": "datasets.load_dataset(json)",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
