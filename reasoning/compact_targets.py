"""Build a 15K benchmark-balanced set from complete visible solutions.

AceReason responses commonly contain a long ``<think>`` trace followed by a
shorter, polished derivation.  This module removes the hidden trace rather than
truncating it, retains only complete boxed solutions, and balances broad math
families without inspecting or matching public benchmark questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .data import SYSTEM, alpaca, boxed, digest, prompt_id, read_rows, write_json
from .features import token_ids


DOMAIN_WEIGHTS = {
    "geometry": 0.22,
    "number_theory": 0.18,
    "algebra": 0.20,
    "combinatorics": 0.15,
    "calculus": 0.10,
    "probability": 0.08,
    "other_math": 0.07,
}

_DOMAIN_PATTERNS = {
    "geometry": re.compile(
        r"(?i)triangle|circle|angle|polygon|quadrilateral|geometry|coordinate plane|"
        r"perpendicular|parallel|tangent|circum|inradius|area of|volume of|sphere|cone"
    ),
    "number_theory": re.compile(
        r"(?i)prime|divisib|integer|modulo|remainder|congruen|diophantine|gcd|lcm|"
        r"last digit|digits?|factorization|perfect square"
    ),
    "combinatorics": re.compile(
        r"(?i)combin|permutation|arrang|coloring|pigeonhole|graph|subset|ways to|"
        r"how many|choose|binomial|recurrence"
    ),
    "calculus": re.compile(
        r"(?i)derivative|integral|limit|continuous|differenti|series|sequence|matrix|"
        r"eigen|vector|linear transformation|differential equation"
    ),
    "probability": re.compile(
        r"(?i)probability|random|expected value|expectation|variance|distribution|"
        r"independent|conditional probability"
    ),
    "algebra": re.compile(
        r"(?i)polynomial|equation|inequal|function|logarithm|exponential|complex number|"
        r"roots?|coefficient|quadratic|system of equations|minimum|maximum"
    ),
}


def visible_solution(output: str) -> str:
    """Return the complete polished solution after the final thinking block."""
    if not isinstance(output, str):
        return ""
    end = output.lower().rfind("</think>")
    text = output[end + len("</think>"):] if end >= 0 else output
    text = text.strip()
    # Reject malformed nested traces instead of teaching control tokens.
    if "<think>" in text.lower() or "</think>" in text.lower():
        return ""
    return text


def domain(instruction: str) -> str:
    scores = {name: len(rx.findall(instruction)) for name, rx in _DOMAIN_PATTERNS.items()}
    best = max(scores, key=lambda name: (scores[name], name))
    return best if scores[best] else "other_math"


def repeated_ngram_fraction(text: str, n: int = 8) -> float:
    words = re.findall(r"\S+", text.lower())
    grams = [tuple(words[i:i+n]) for i in range(max(0, len(words) - n + 1))]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def quotas(size: int, capacities: dict[str, int]) -> dict[str, int]:
    if size < 1 or sum(capacities.values()) < size:
        raise ValueError("insufficient eligible examples")
    out = {k: min(capacities.get(k, 0), int(math.floor(size * w)))
           for k, w in DOMAIN_WEIGHTS.items()}
    while sum(out.values()) < size:
        eligible = [k for k in DOMAIN_WEIGHTS if out[k] < capacities.get(k, 0)]
        if not eligible:
            raise ValueError("cannot fill domain quotas")
        k = max(eligible, key=lambda x: (size * DOMAIN_WEIGHTS[x] - out[x], x))
        out[k] += 1
    return out


def _stable_jitter(seed: int, row_id: str) -> float:
    raw = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()[:8]
    return int.from_bytes(raw, "big") / 2**64


def build(rows, raw_rows, tokenizer, size=15000, cutoff=4096, seed=42):
    metadata = {prompt_id(x["instruction"]): x.get("extra_info", {}) for x in raw_rows}
    candidates = defaultdict(list)
    rejection = Counter()
    for row in rows:
        rid = row["id"]
        info = metadata.get(rid, {})
        if info.get("source") not in {"OpenMathReasoning", "NuminaMath-CoT"}:
            rejection["non_math_source"] += 1
            continue
        solution = visible_solution(row["output"])
        if not solution or boxed(solution) is None:
            rejection["incomplete_visible_solution"] += 1
            continue
        rep = repeated_ngram_fraction(solution)
        if rep > 0.08:
            rejection["repetition"] += 1
            continue
        compact = dict(row, output=solution)
        prompt_tokens, full_tokens = token_ids(tokenizer, {
            "system": SYSTEM, "instruction": row["instruction"], "output": solution,
        })
        response_tokens = len(full_tokens) - len(prompt_tokens)
        if not 96 <= response_tokens <= 2048 or len(full_tokens) > cutoff:
            rejection["token_bounds"] += 1
            continue
        family = domain(row["instruction"])
        # Prefer substantial derivations near 512 tokens, while preserving a
        # deterministic amount of diversity inside every family.
        length_score = -abs(math.log2(max(response_tokens, 1)) - 9.0)
        score = length_score - 2.0 * rep + 0.10 * _stable_jitter(seed, rid)
        candidates[family].append((score, compact, response_tokens, info.get("source")))
    caps = {k: len(candidates[k]) for k in DOMAIN_WEIGHTS}
    allocation = quotas(size, caps)
    chosen = []
    for family in DOMAIN_WEIGHTS:
        ranked = sorted(candidates[family], key=lambda x: (-x[0], x[1]["id"]))
        chosen.extend((family, *x[1:]) for x in ranked[:allocation[family]])
    random.Random(seed).shuffle(chosen)
    records = [x[1] for x in chosen]
    manifest = {
        "schema": "reasoning-sft-compact-balanced-15k-v1",
        "size": len(records), "seed": seed, "cutoff": cutoff,
        "method": "post-think complete-solution supervision with fixed domain quotas",
        "domain_weights": DOMAIN_WEIGHTS, "domain_counts": Counter(x[0] for x in chosen),
        "source_counts": Counter(x[3] for x in chosen),
        "response_tokens": {
            "total": sum(x[2] for x in chosen),
            "mean": sum(x[2] for x in chosen) / len(chosen),
            "max": max(x[2] for x in chosen),
        },
        "rejections": rejection,
        "selected_ids": [x[1]["id"] for x in chosen],
    }
    return records, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, default=15000)
    p.add_argument("--cutoff", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    records, manifest = build(
        read_rows(args.pool), read_rows(args.raw), tok,
        size=args.size, cutoff=args.cutoff, seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "compact_balanced.json", alpaca(records))
    write_json(args.output / "dataset_info.json", {
        "compact_balanced": {"file_name": "compact_balanced.json"}
    })
    manifest.update({
        "pool_sha256": digest(args.pool), "raw_sha256": digest(args.raw),
        "dataset_sha256": digest(args.output / "compact_balanced.json"),
    })
    write_json(args.output / "selection_manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "selected_ids"}, indent=2))


if __name__ == "__main__":
    main()

