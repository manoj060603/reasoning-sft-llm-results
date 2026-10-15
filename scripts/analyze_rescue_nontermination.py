"""Classify rescue truncations and optionally audit serialized training targets."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning.data import boxed, digest, write_json
from reasoning.features import repetition, token_ids


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def classify(rows: list[dict], threshold: float) -> dict:
    counts = Counter()
    records = []
    for row in rows:
        for index, (response, answer, correct, finish, tokens) in enumerate(zip(
            row["generated_responses"], row["generated_answers"], row["answers_correctness"],
            row["finish_reasons"], row["output_tokens"]
        )):
            if finish != "length":
                continue
            rep = repetition(response)
            has_answer = bool(str(answer).strip())
            if has_answer:
                category = "answer_emitted_then_length_limit"
            elif rep > threshold:
                category = "repetitive_loop_without_answer"
            else:
                category = "nonrepetitive_unfinished_reasoning"
            counts[category] += 1
            records.append({
                "category": category,
                "problem_id": row["problem_id"],
                "sample_index": index,
                "output_tokens": tokens,
                "repeated_8gram_fraction": rep,
                "extracted_answer": answer,
                "correct": bool(correct),
                "question": row["question"],
                "response_head": response[:600],
                "response_tail": response[-600:],
            })
    sample_rows = {}
    for category in counts:
        subset = [row for row in records if row["category"] == category]
        if category == "repetitive_loop_without_answer":
            subset.sort(key=lambda row: (-row["repeated_8gram_fraction"], row["problem_id"], row["sample_index"]))
        else:
            subset.sort(key=lambda row: (row["problem_id"], row["sample_index"]))
        sample_rows[category] = subset[:5]
    return {
        "length_limited_samples": len(records),
        "exclusive_categories": dict(counts),
        "correct_length_limited_samples": sum(row["correct"] for row in records),
        "representative_samples": sample_rows,
    }


def training_audit(data_path: Path, model: str, serialization_sample_size: int) -> dict:
    from transformers import AutoTokenizer

    rows = json.loads(data_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(model)
    response_lengths = []
    repetitions = []
    counts = Counter()
    for row in rows:
        output = row["output"]
        rep = repetition(output)
        repetitions.append(rep)
        counts["complete_box"] += boxed(output) is not None
        counts["think_open"] += "<think>" in output
        counts["think_close"] += "</think>" in output
        counts["repetition_above_0.10"] += rep > .10
        counts["repetition_above_0.35"] += rep > .35
    sample_size = min(serialization_sample_size, len(rows))
    sample_indices = sorted({round(i * (len(rows) - 1) / max(1, sample_size - 1))
                             for i in range(sample_size)})
    for index in sample_indices:
        prompt, full = token_ids(tokenizer, rows[index])
        if full[:len(prompt)] != prompt:
            raise RuntimeError("Prompt is not a prefix of the serialized training example")
        response_lengths.append(len(full) - len(prompt))
        counts["sampled_prompt_prefix_ok"] += 1
        response_ids = full[len(prompt):]
        counts["sampled_assistant_end_token_supervised"] += tokenizer.eos_token_id in response_ids
        counts["sampled_assistant_end_is_penultimate_before_newline"] += bool(
            len(full) >= 2 and full[-2] == tokenizer.eos_token_id
        )
    return {
        "path": str(data_path),
        "sha256": digest(data_path),
        "rows": len(rows),
        "serialization_sample_rows": len(sample_indices),
        "counts": dict(counts),
        "response_tokens": {
            "min": min(response_lengths),
            "median": statistics.median(response_lengths),
            "mean": statistics.mean(response_lengths),
            "p90": percentile(response_lengths, .90),
            "p99": percentile(response_lengths, .99),
            "max": max(response_lengths),
        },
        "repeated_8gram_fraction": {
            "median": statistics.median(repetitions),
            "mean": statistics.mean(repetitions),
            "p90": percentile(repetitions, .90),
        },
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "loss_mask": "train_adapter.py labels every serialized assistant token; sampled rows check that <|im_end|> is supervised even when the chat template appends a newline after it",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetition-threshold", type=float, default=.35)
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--serialization-sample-size", type=int, default=256)
    args = parser.parse_args()
    base = json.loads(args.base.read_text())
    candidate = json.loads(args.candidate.read_text())
    if [row["problem_id"] for row in base] != [row["problem_id"] for row in candidate]:
        raise RuntimeError("Base and candidate question ordering differs")
    result = {
        "schema": "reasoning-sft-rescue-nontermination-audit-v1",
        "repetition_definition": "fraction of repeated whitespace-token 8-grams",
        "repetition_threshold": args.repetition_threshold,
        "base": classify(base, args.repetition_threshold),
        "candidate": classify(candidate, args.repetition_threshold),
        "source_sha256": {"base": digest(args.base), "candidate": digest(args.candidate)},
    }
    if args.training_data:
        if not args.model:
            parser.error("--model is required with --training-data")
        result["training_targets"] = training_audit(
            args.training_data, args.model, args.serialization_sample_size
        )
    write_json(args.output, result)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest(args.output)}  {args.output.name}\n"
    )
    print(json.dumps({
        "base": result["base"]["exclusive_categories"],
        "candidate": result["candidate"]["exclusive_categories"],
        "training_targets": result.get("training_targets"),
    }, indent=2))


if __name__ == "__main__":
    main()
