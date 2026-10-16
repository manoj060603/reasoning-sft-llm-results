"""Benchmark-blind mathematical skill labeling and coverage selection.

The skill taxonomy and selection objective are frozen before the public suite.
Labels are generated from training prompts only; responses are never shown to
the labeler.  The JSONL ledger makes the expensive teacher pass resumable.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
from collections import Counter
from itertools import combinations
from pathlib import Path

from .data import SYSTEM, alpaca, digest, read_rows, write_json
from .features import token_ids
from .select import eligible

DOMAINS = ("algebra", "geometry", "number_theory", "combinatorics", "probability",
           "calculus", "linear_algebra", "statistics", "logic", "other")
SKILLS = (
    "equations", "inequalities", "polynomials", "factoring", "functions", "sequences",
    "complex_numbers", "trigonometry", "coordinate_geometry", "euclidean_geometry",
    "circle_geometry", "solid_geometry", "area_volume", "similarity", "congruence",
    "modular_arithmetic", "divisibility", "primes", "diophantine_equations",
    "combinatorial_counting", "permutations_combinations", "pigeonhole", "invariants",
    "graph_theory", "recurrences", "generating_functions", "probability_counting",
    "conditional_probability", "expectation", "distributions", "derivatives",
    "integrals", "limits", "optimization", "matrices", "vectors", "eigenvalues",
    "induction", "contradiction", "case_analysis", "symmetry", "construction",
    "bounding", "estimation", "symbolic_manipulation", "word_problem_translation",
)
REPRESENTATIONS = ("symbolic", "word_problem", "diagram", "table", "algorithmic", "mixed")
LANGUAGES = ("english", "chinese", "other")
ANSWER_FORMS = ("integer", "real", "expression", "proof", "multiple_choice", "other")

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "domain": {"type": "string", "enum": list(DOMAINS)},
        "skills": {"type": "array", "items": {"type": "string", "enum": list(SKILLS)},
                   "minItems": 2, "maxItems": 5},
        "composition_depth": {"type": "integer", "minimum": 1, "maximum": 4},
        "representation": {"type": "string", "enum": list(REPRESENTATIONS)},
        "language": {"type": "string", "enum": list(LANGUAGES)},
        "answer_form": {"type": "string", "enum": list(ANSWER_FORMS)},
    },
    "required": ["domain", "skills", "composition_depth", "representation",
                 "language", "answer_form"],
}

INSTRUCTIONS = (
    "Classify the mathematical skills needed to solve this training problem. "
    "Return one JSON object matching the provided schema. Choose two to five "
    "specific skills from the allowed vocabulary. Do not solve the problem."
)


def validate_label(label: dict) -> dict:
    if not isinstance(label, dict) or set(label) != set(SCHEMA["required"]):
        raise ValueError("invalid label fields")
    for key, allowed in (("domain", DOMAINS), ("representation", REPRESENTATIONS),
                         ("language", LANGUAGES), ("answer_form", ANSWER_FORMS)):
        if label[key] not in allowed:
            raise ValueError(f"invalid {key}")
    skills = label["skills"]
    # Saved labels can contain one unique skill after xgrammar's repeated
    # 2-item output is deduplicated.  Do not invent a second skill on resume.
    if not isinstance(skills, list) or not 1 <= len(skills) <= 5:
        raise ValueError("invalid skill list")
    if not set(skills) <= set(SKILLS):
        raise ValueError("unknown skill")
    # xgrammar does not implement JSON Schema uniqueItems.  Canonicalize a
    # repeated label instead of fabricating an additional mathematical skill.
    skills = sorted(set(skills))
    depth = label["composition_depth"]
    if type(depth) is not int or not 1 <= depth <= 4:
        raise ValueError("invalid composition depth")
    return {**label, "skills": skills}


def label_items(label: dict, cluster: int) -> tuple[str, ...]:
    skills = label["skills"]
    items = ["d:" + label["domain"], "r:" + label["representation"],
             "l:" + label["language"], "a:" + label["answer_form"],
             "c:" + str(cluster), "depth:" + str(label["composition_depth"])]
    items += ["s:" + s for s in skills]
    items += ["p:" + a + "+" + b for a, b in combinations(skills, 2)]
    return tuple(items)


def load_labels(path: Path, expected_ids: list[str], allow_partial: bool = False) -> list[dict]:
    labels = []
    with path.open() if path.exists() else open(os.devnull) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if len(labels) >= len(expected_ids) or row.get("id") != expected_ids[len(labels)]:
                raise ValueError("skill ledger is out of source order or has duplicate IDs")
            labels.append({"id": row["id"], **validate_label(row["label"])})
    if not allow_partial and len(labels) != len(expected_ids):
        raise ValueError(f"incomplete skill ledger: {len(labels)}/{len(expected_ids)}")
    return labels


def label_pool(pool: Path, teacher: str, output: Path, batch_size: int,
               max_model_len: int, gpu_memory_utilization: float,
               limit: int | None = None) -> None:
    from vllm import LLM, SamplingParams
    from vllm.config import StructuredOutputsConfig
    from vllm.sampling_params import StructuredOutputsParams

    rows = read_rows(pool)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("source IDs are not unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = output.with_suffix(output.suffix + ".meta.json")
    source_hash = digest(pool)
    teacher_path = Path(teacher)
    frozen = {"pool_sha256": source_hash, "teacher": str(teacher_path.resolve()),
              "taxonomy": list(SKILLS), "schema": SCHEMA, "max_output_tokens": 128,
              "instruction_prefix_chars": 2048}
    if metadata.exists():
        if json.loads(metadata.read_text()) != frozen:
            raise ValueError("skill ledger has different source, teacher, or taxonomy")
    else:
        write_json(metadata, frozen)
    complete = load_labels(output, ids, allow_partial=True)
    stop = min(len(rows), limit) if limit is not None else len(rows)
    if len(complete) >= stop:
        return
    # This reasoning model can spend hundreds of constrained tokens on legal
    # JSON whitespace.  The installed xgrammar backend supports compact JSON.
    llm = LLM(model=teacher, tokenizer=teacher, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=False,
              structured_outputs_config=StructuredOutputsConfig(
                  backend="xgrammar", disable_any_whitespace=True))
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0, max_tokens=128,
                            structured_outputs=StructuredOutputsParams(json=SCHEMA))
    retry_params = SamplingParams(temperature=0, max_tokens=512,
                                  structured_outputs=StructuredOutputsParams(json=SCHEMA))
    for start in range(len(complete), stop, batch_size):
        batch = rows[start:min(start + batch_size, stop)]
        prompts = [tokenizer.apply_chat_template([
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": r["instruction"][:2048]},
        ], tokenize=False, add_generation_prompt=True) for r in batch]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        if len(outputs) != len(batch):
            raise RuntimeError("teacher returned incomplete skill-label batch")
        checked = []
        retry_indices = []
        for index, (row, generated) in enumerate(zip(batch, outputs)):
            if len(generated.outputs) != 1:
                raise RuntimeError("teacher returned unexpected label count")
            try:
                label = validate_label(json.loads(generated.outputs[0].text))
                checked.append({"id": row["id"], "label": label})
            except (json.JSONDecodeError, ValueError):
                checked.append(None)
                retry_indices.append(index)
        if retry_indices:
            retry_outputs = llm.generate([prompts[i] for i in retry_indices],
                                         retry_params, use_tqdm=False)
            if len(retry_outputs) != len(retry_indices):
                raise RuntimeError("teacher returned incomplete skill-label retries")
            for index, generated in zip(retry_indices, retry_outputs):
                if len(generated.outputs) != 1:
                    raise RuntimeError("teacher returned unexpected retry count")
                try:
                    label = validate_label(json.loads(generated.outputs[0].text))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise RuntimeError(
                        f"invalid skill label after 512-token retry for {batch[index]['id']}; "
                        f"finish_reason={generated.outputs[0].finish_reason}"
                    ) from exc
                checked[index] = {"id": batch[index]["id"], "label": label}
            print(f"retried {len(retry_indices)} truncated or invalid labels at "
                  f"batch offset {start}", flush=True)
        with output.open("a", encoding="utf-8") as stream:
            for item in checked:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"skill labels: {start + len(batch)}/{len(rows)}", flush=True)


def coverage_select(rows: list[dict], features: list[dict], labels: list[dict],
                    size: int, cutoff: int) -> tuple[list[dict], dict]:
    if size < 1:
        raise ValueError("size must be positive")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or {f["id"] for f in features} != set(ids):
        raise ValueError("pool/features ID mismatch")
    if [x["id"] for x in labels] != ids:
        raise ValueError("skill-label IDs must match pool order exactly")
    fmap = {f["id"]: f for f in features}
    candidates = []
    for r, l in zip(rows, labels):
        f = fmap[r["id"]]
        if eligible(f, cutoff):
            candidates.append((r, label_items(l, f["cluster"]), f))
    if len(candidates) < size:
        raise ValueError(f"only {len(candidates)} cutoff-eligible rows for size {size}")
    frequency = Counter(x for _, items, _ in candidates for x in items)
    weights = {x: 1 / math.sqrt(n) for x, n in frequency.items()}
    coverage = Counter()
    selected = []
    marginal_receipt = []
    heap = []

    def gain(i):
        _, items, f = candidates[i]
        return (sum(weights[x] * (math.sqrt(coverage[x] + 1) - math.sqrt(coverage[x]))
                    for x in items) + .1 * (1 - f["repetition"]))

    for i, (row, _, _) in enumerate(candidates):
        heapq.heappush(heap, (-gain(i), row["id"], i, 0))
    while len(selected) < size:
        if not heap:
            raise RuntimeError("coverage heap exhausted")
        neg, _, i, version = heapq.heappop(heap)
        if version != len(selected):
            heapq.heappush(heap, (-gain(i), candidates[i][0]["id"], i, len(selected)))
            continue
        row, items, _ = candidates[i]
        selected.append(row)
        marginal_receipt.append({"id": row["id"], "marginal_gain": -neg})
        coverage.update(items)
    receipt = {"schema": "skill-graph-coverage-v1", "eligible": len(candidates),
               "selected_ids": [r["id"] for r in selected],
               "skill_frequency": dict(sorted(frequency.items())),
               "selected_coverage": dict(sorted(coverage.items())),
               "marginal_gains": marginal_receipt,
               "objective": "sum inverse-sqrt-frequency weighted sqrt item coverage + 0.1*(1-repetition)",
               "cutoff": cutoff}
    return selected, receipt


def export(pool: Path, features_path: Path, labels_path: Path, output: Path,
           tokenizer_path: str, size: int, cutoff: int) -> None:
    from transformers import AutoTokenizer

    rows = read_rows(pool)
    labels = load_labels(labels_path, [r["id"] for r in rows])
    features = read_rows(features_path)
    selected, receipt = coverage_select(rows, features, labels, size, cutoff)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    exact = {}
    for row in selected:
        _, full = token_ids(tokenizer, row)
        if len(full) > cutoff:
            raise ValueError(f"exact chat-template bound failed: {row['id']}: {len(full)}")
        exact[row["id"]] = len(full)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "advanced.json", alpaca(selected))
    write_json(output / "dataset_info.json", {"advanced": {"file_name": "advanced.json"}})
    receipt.update({"pool_sha256": digest(pool), "features_sha256": digest(features_path),
                    "labels_sha256": digest(labels_path), "exact_chat_template_tokens": exact,
                    "max_exact_tokens": max(exact.values()),
                    "data_sha256": digest(output / "advanced.json")})
    write_json(output / "selection_manifest.json", receipt)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("label")
    a.add_argument("--pool", type=Path, required=True)
    a.add_argument("--teacher", required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--batch-size", type=int, default=64)
    a.add_argument("--max-model-len", type=int, default=4096)
    a.add_argument("--gpu-memory-utilization", type=float, default=.8)
    a.add_argument("--limit", type=int)
    b = sub.add_parser("export")
    b.add_argument("--pool", type=Path, required=True)
    b.add_argument("--features", type=Path, required=True)
    b.add_argument("--labels", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--size", type=int, default=15000)
    b.add_argument("--cutoff", type=int, default=8192)
    x = p.parse_args()
    if x.command == "label":
        label_pool(x.pool, x.teacher, x.output, x.batch_size,
                   x.max_model_len, x.gpu_memory_utilization, x.limit)
    else:
        export(x.pool, x.features, x.labels, x.output, x.tokenizer, x.size, x.cutoff)


if __name__ == "__main__":
    main()
