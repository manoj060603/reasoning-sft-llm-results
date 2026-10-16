"""Independent teacher-answer checks for selected rare/composite training prompts.

These checks audit alternative paths.  They never use public benchmark data or
replace a training target.  Answer agreement is not a proof of intermediate steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import SYSTEM, boxed, digest, read_rows, write_json
from .features import token_ids
from .skill_graph import load_labels


def token_divergence(a: str, b: str) -> float:
    a_set, b_set = set(a.casefold().split()), set(b.casefold().split())
    return 1 - len(a_set & b_set) / max(1, len(a_set | b_set))


def answer_match(candidate: str, reference: str, project_root: Path) -> bool:
    pred, gold = boxed(candidate), boxed(reference)
    if pred is None or gold is None:
        return False
    sys.path.insert(0, str(project_root / "eval"))
    try:
        from utils.grader import check_is_correct
        return bool(check_is_correct(pred, gold))
    finally:
        sys.path.pop(0)


def choose_audit_ids(selected: list[dict], labels: list[dict], count: int) -> list[str]:
    label_map = {x["id"]: x for x in labels}
    ranked = sorted(selected, key=lambda r: (-label_map[r["id"]]["composition_depth"],
                                           -len(label_map[r["id"]]["skills"]), r["id"]))
    return [r["id"] for r in ranked[:count]]


def audit(pool: Path, selection: Path, labels_path: Path, teacher: str,
          base_tokenizer: str, output: Path, count: int, batch_size: int,
          max_new_tokens: int, cutoff: int) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = read_rows(pool)
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("pool has duplicate IDs")
    labels = load_labels(labels_path, [r["id"] for r in rows])
    manifest = json.loads(selection.read_text())
    selected = [by_id[x] for x in manifest["selected_ids"]]
    if count < 1 or count > len(selected):
        raise ValueError("invalid audit count")
    ordered_ids = choose_audit_ids(selected, labels, count)
    existing = read_rows(output) if output.exists() else []
    if [x["id"] for x in existing] != ordered_ids[:len(existing)]:
        raise ValueError("audit receipt is out of order")
    if len(existing) == count:
        return
    base_tok = AutoTokenizer.from_pretrained(base_tokenizer, trust_remote_code=False)
    llm = LLM(model=teacher, tokenizer=teacher, max_model_len=8192,
              gpu_memory_utilization=.8, trust_remote_code=False)
    teacher_tok = llm.get_tokenizer()
    params = SamplingParams(temperature=.6, top_p=.95, max_tokens=max_new_tokens, seed=42)
    receipts = existing
    for start in range(len(existing), count, batch_size):
        batch_ids = ordered_ids[start:min(count, start + batch_size)]
        prompts = [teacher_tok.apply_chat_template([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": by_id[x]["instruction"]},
        ], tokenize=False, add_generation_prompt=True) for x in batch_ids]
        generated = llm.generate(prompts, params, use_tqdm=False)
        if len(generated) != len(batch_ids):
            raise RuntimeError("incomplete alternative-generation batch")
        for row_id, item in zip(batch_ids, generated):
            if len(item.outputs) != 1:
                raise RuntimeError("unexpected alternative count")
            row = by_id[row_id]
            alt = item.outputs[0].text
            try:
                _, full = token_ids(base_tok, {**row, "output": alt})
                token_count = len(full)
            except ValueError:
                token_count = None
            verified = answer_match(alt, row["output"], Path(__file__).resolve().parents[1])
            divergence = token_divergence(alt, row["output"])
            receipts.append({"id": row_id, "original_box": boxed(row["output"]),
                             "alternative_box": boxed(alt), "answer_match": verified,
                             "divergence": divergence, "exact_chat_tokens": token_count,
                             "within_cutoff": token_count is not None and token_count <= cutoff,
                             "finish_reason": item.outputs[0].finish_reason,
                             "alternative_sha256": __import__("hashlib").sha256(alt.encode()).hexdigest()})
        write_json(output, receipts)
        print(f"alternative checks: {len(receipts)}/{count}", flush=True)
    write_json(output.with_suffix(".meta.json"), {
        "pool_sha256": digest(pool), "selection_sha256": digest(selection),
        "labels_sha256": digest(labels_path), "teacher": teacher,
        "requested_count": count, "answer_matches": sum(x["answer_match"] for x in receipts),
        "complete_8k_matches": sum(x["answer_match"] and x["within_cutoff"] for x in receipts),
        "note": "Final-answer agreement does not verify intermediate reasoning.",
    })


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--base-tokenizer", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--cutoff", type=int, default=8192)
    x = p.parse_args()
    audit(x.pool, x.selection, x.labels, x.teacher, x.base_tokenizer,
          x.output, x.count, x.batch_size, x.max_new_tokens, x.cutoff)


if __name__ == "__main__":
    main()
