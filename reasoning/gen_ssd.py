"""Generate student-selected teacher traces, with resumable per-question receipts.

This is a compute-scaled Gen-SSD adaptation: the teacher samples continuations
chunk by chunk and the base student retains the continuation with lowest
teacher-forced perplexity.  Only trajectories whose final answer matches the
sealed reference are exported for training.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from .data import SYSTEM, boxed, digest, prompt_id, read_rows, write_json
from .features import token_ids


@dataclass(frozen=True)
class Candidate:
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str | None


def candidate_count(step: int, first: int, later: int) -> int:
    if step < 0 or first < 1 or later < 1:
        raise ValueError("invalid chunk step or candidate schedule")
    return first if step == 0 else later


def empty_stop_is_terminal(outputs) -> bool:
    """vLLM can return only zero-token stop candidates after a completed prefix."""
    return bool(outputs) and all(not x.text and x.finish_reason == "stop" for x in outputs)


def serialized_training_tokens(tokenizer, instruction: str, output: str) -> int:
    """Count the exact chat-template sequence consumed by response-only SFT."""
    _, full = token_ids(tokenizer, {
        "system": SYSTEM, "instruction": instruction, "output": output,
    })
    return len(full)


def masked_chunk_nll(logits, input_ids, attention_mask, prefix_lengths):
    """Return mean next-token NLL over each candidate continuation."""
    import torch

    shift_logits = logits[:, :-1].float()
    shift_labels = input_ids[:, 1:]
    valid = attention_mask[:, 1:].bool()
    positions = torch.arange(1, input_ids.shape[1], device=input_ids.device)[None, :]
    valid &= positions >= prefix_lengths[:, None]
    token_loss = torch.nn.functional.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels, reduction="none"
    )
    counts = valid.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("empty candidate continuation")
    return ((token_loss * valid).sum(dim=1) / counts).tolist()


def _answer_is_correct(text: str, reference: str, root: Path) -> bool:
    pred = boxed(text)
    if pred is None:
        return False
    sys.path.insert(0, str(root / "eval"))
    try:
        from utils.grader import check_is_correct
        return bool(check_is_correct(pred, reference))
    finally:
        if sys.path and sys.path[0] == str(root / "eval"):
            sys.path.pop(0)


def _score_candidates(model, tokenizer, contexts, continuations, batch_size):
    import torch

    scores = []
    for start in range(0, len(contexts), batch_size):
        ctx = contexts[start:start + batch_size]
        cont = continuations[start:start + batch_size]
        prefix_lengths = [len(tokenizer.encode(x, add_special_tokens=False)) for x in ctx]
        encoded = tokenizer(
            [x + y for x, y in zip(ctx, cont)], return_tensors="pt", padding=True,
            add_special_tokens=False,
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        prefix = torch.tensor(prefix_lengths, device=model.device)
        with torch.inference_mode():
            logits = model(**encoded).logits
            scores.extend(masked_chunk_nll(
                logits, encoded["input_ids"], encoded["attention_mask"], prefix
            ))
        del logits, encoded
        torch.cuda.empty_cache()
    return scores


def _generate_attempt(llm, scorer, scorer_tokenizer, teacher_tokenizer, rows,
                      chunk_tokens, max_new_tokens, first_candidates,
                      later_candidates, score_batch_size, seed):
    from vllm import SamplingParams

    prompts = [teacher_tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": r["instruction"]},
    ], tokenize=False, add_generation_prompt=True) for r in rows]
    prefixes = ["" for _ in rows]
    traces = [[] for _ in rows]
    done = [False for _ in rows]
    max_steps = math.ceil(max_new_tokens / chunk_tokens)
    for step in range(max_steps):
        active = [i for i, flag in enumerate(done) if not flag]
        if not active:
            break
        k = candidate_count(step, first_candidates, later_candidates)
        remaining = max_new_tokens - step * chunk_tokens
        params = [SamplingParams(
            n=k, temperature=.6, top_p=.95, top_k=30,
            max_tokens=min(chunk_tokens, remaining),
            seed=(seed + int(prompt_id(rows[i]["instruction"])[:8], 16) + step * 1009) % (2**31),
        ) for i in active]
        requests = [prompts[i] + prefixes[i] for i in active]
        generated = llm.generate(requests, params)
        flat_contexts, flat_texts, owners, candidates = [], [], [], []
        terminal_owners = set()
        for owner, result in zip(active, generated):
            if len(result.outputs) != k:
                raise RuntimeError("teacher returned an incomplete candidate set")
            if empty_stop_is_terminal(result.outputs):
                done[owner] = True
                terminal_owners.add(owner)
                traces[owner].append({
                    "step": step, "candidate_count": k, "selected_nll": None,
                    "candidate_nlls": [], "selected_tokens": 0,
                    "finish_reason": "stop",
                })
                continue
            for out in result.outputs:
                if not out.text:
                    continue
                flat_contexts.append(prompts[owner] + prefixes[owner])
                flat_texts.append(out.text)
                owners.append(owner)
                candidates.append(Candidate(out.text, tuple(out.token_ids), out.finish_reason))
        scoring_owners = set(active) - terminal_owners
        if set(owners) != scoring_owners:
            raise RuntimeError("teacher returned no continuation for an active problem")
        nlls = _score_candidates(
            scorer, scorer_tokenizer, flat_contexts, flat_texts, score_batch_size
        )
        by_owner = {i: [] for i in scoring_owners}
        for owner, cand, nll in zip(owners, candidates, nlls):
            by_owner[owner].append((float(nll), cand))
        for owner in scoring_owners:
            nll, chosen = min(by_owner[owner], key=lambda x: x[0])
            prefixes[owner] += chosen.text
            traces[owner].append({
                "step": step, "candidate_count": len(by_owner[owner]),
                "selected_nll": nll,
                "candidate_nlls": [x[0] for x in sorted(by_owner[owner], key=lambda x: x[0])],
                "selected_tokens": len(chosen.token_ids),
                "finish_reason": chosen.finish_reason,
            })
            if chosen.finish_reason == "stop" or sum(x["selected_tokens"] for x in traces[owner]) >= max_new_tokens:
                done[owner] = True
    return list(zip(prefixes, traces))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target-size", type=int, default=2000)
    p.add_argument("--max-source", type=int, default=4000)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--score-batch-size", type=int, default=2)
    p.add_argument("--chunk-tokens", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--first-candidates", type=int, default=4)
    p.add_argument("--later-candidates", type=int, default=2)
    p.add_argument("--training-cutoff", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-memory-utilization", type=float, default=.64)
    args = p.parse_args()
    if args.attempts < 1 or args.batch_size < 1 or args.score_batch_size < 1:
        p.error("attempt and batch sizes must be positive")
    if args.chunk_tokens < 1 or args.max_new_tokens < args.chunk_tokens or args.training_cutoff < 1:
        p.error("invalid chunk or generation length")

    root = Path(__file__).resolve().parents[1]
    rows = sorted(read_rows(args.input), key=lambda x: prompt_id(x["instruction"]))[:args.max_source]
    if len(rows) < args.target_size or len({prompt_id(x["instruction"]) for x in rows}) != len(rows):
        raise ValueError("source must contain enough unique questions")
    refs = {prompt_id(x["instruction"]): boxed(x["output"]) for x in rows}
    if any(v is None for v in refs.values()):
        raise ValueError("every sealed reference must contain a balanced boxed answer")
    manifest = {
        "schema": "reasoning-sft-gen-ssd-2k-v1",
        "method": "generation-time chunk selection by lowest base-student PPL",
        "paper": "https://arxiv.org/abs/2604.02819",
        "adaptation": "4/2 candidates, 1024-token chunks, 4096-token cap; paper default is 16/8/4, 4096-token chunks, 16384-token cap",
        "input": str(args.input), "input_sha256": digest(args.input),
        "teacher": args.teacher, "student": args.student,
        "target_size": args.target_size, "max_source": args.max_source,
        "attempts": args.attempts, "batch_size": args.batch_size,
        "score_batch_size": args.score_batch_size,
        "chunk_tokens": args.chunk_tokens, "max_new_tokens": args.max_new_tokens,
        "first_candidates": args.first_candidates, "later_candidates": args.later_candidates,
        "training_cutoff": args.training_cutoff,
        "seed": args.seed,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "generation_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("generation settings changed; use a fresh output directory")
    write_json(manifest_path, manifest)
    records_dir = args.output / "records"
    records_dir.mkdir(exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM
    import torch

    scorer_tokenizer = AutoTokenizer.from_pretrained(args.student, use_fast=True)
    if scorer_tokenizer.pad_token_id is None:
        scorer_tokenizer.pad_token_id = scorer_tokenizer.eos_token_id
    scorer = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).eval()
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, use_fast=True)
    llm = LLM(
        model=args.teacher, tokenizer=args.teacher, dtype="bfloat16",
        max_model_len=8192, seed=args.seed, trust_remote_code=False,
        generation_config="vllm", max_num_seqs=args.batch_size * args.first_candidates,
        gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True,
    )

    for attempt in range(args.attempts):
        pending = []
        for row in rows:
            pid = prompt_id(row["instruction"])
            dest = records_dir / f"{pid}.json"
            if dest.exists():
                saved = json.loads(dest.read_text())
                if saved.get("answer_correct") or saved.get("correct"):
                    guided_tokens = serialized_training_tokens(
                        scorer_tokenizer, row["instruction"], saved.get("output", "")
                    )
                    existing_tokens = serialized_training_tokens(
                        scorer_tokenizer, row["instruction"], row["output"]
                    )
                    fits = max(guided_tokens, existing_tokens) <= args.training_cutoff
                    saved.update({
                        "answer_correct": True,
                        "guided_training_tokens": guided_tokens,
                        "existing_training_tokens": existing_tokens,
                        "fits_training_cutoff": fits,
                        "correct": fits,
                    })
                    write_json(dest, saved)
                    if fits:
                        continue
            pending.append(row)
        if not pending:
            break
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            outputs = _generate_attempt(
                llm, scorer, scorer_tokenizer, teacher_tokenizer, batch,
                args.chunk_tokens, args.max_new_tokens, args.first_candidates,
                args.later_candidates, args.score_batch_size,
                args.seed + attempt * 1_000_003,
            )
            for row, (text, trace) in zip(batch, outputs):
                pid = prompt_id(row["instruction"])
                ref = refs[pid]
                answer_correct = _answer_is_correct(text, ref, root)
                guided_tokens = serialized_training_tokens(
                    scorer_tokenizer, row["instruction"], text
                )
                existing_tokens = serialized_training_tokens(
                    scorer_tokenizer, row["instruction"], row["output"]
                )
                fits = max(guided_tokens, existing_tokens) <= args.training_cutoff
                correct = answer_correct and fits
                write_json(records_dir / f"{pid}.json", {
                    "problem_id": pid, "attempt": attempt + 1,
                    "instruction": row["instruction"], "reference": ref,
                    "output": text, "answer_correct": answer_correct,
                    "guided_training_tokens": guided_tokens,
                    "existing_training_tokens": existing_tokens,
                    "fits_training_cutoff": fits,
                    "correct": correct, "trace": trace,
                    "output_tokens": sum(x["selected_tokens"] for x in trace),
                })
            print(f"attempt={attempt+1} processed={min(start+len(batch),len(pending))}/{len(pending)}", flush=True)
            gc.collect()
            verified = sum(
                json.loads(x.read_text()).get("correct", False)
                for x in records_dir.glob("*.json")
            )
            if verified >= args.target_size:
                break
        if verified >= args.target_size:
            break

    guided, existing, receipts = [], [], []
    for row in rows:
        pid = prompt_id(row["instruction"])
        dest = records_dir / f"{pid}.json"
        if not dest.exists():
            continue
        rec = json.loads(dest.read_text())
        receipts.append(rec)
        if not rec.get("correct"):
            continue
        if len(guided) < args.target_size:
            guided.append({"instruction": row["instruction"], "input": "", "output": rec["output"], "system": SYSTEM})
            existing.append({"instruction": row["instruction"], "input": "", "output": row["output"], "system": SYSTEM})
    if len(guided) != args.target_size:
        write_json(args.output / "generation_summary.json", {
            "requested": args.target_size, "verified": len(guided),
            "failed": args.target_size-len(guided),
            "complete": False,
        })
        raise RuntimeError(f"verified only {len(guided)}/{args.target_size} teacher trajectories")
    write_json(args.output / "guided.json", guided)
    write_json(args.output / "existing.json", existing)
    write_json(args.output / "dataset_info.json", {
        "gen_ssd_guided": {"file_name": "guided.json"},
        "gen_ssd_existing": {"file_name": "existing.json"},
    })
    write_json(args.output / "generation_summary.json", {
        "requested": args.target_size, "verified": args.target_size,
        "failed": 0, "complete": True, "source_considered": len(rows),
        "guided_sha256": digest(args.output / "guided.json"),
        "existing_sha256": digest(args.output / "existing.json"),
        "training_cutoff": args.training_cutoff,
        "max_guided_training_tokens": max(x["guided_training_tokens"] for x in receipts if x.get("correct")),
        "max_existing_training_tokens": max(x["existing_training_tokens"] for x in receipts if x.get("correct")),
        "mean_output_tokens": sum(x["output_tokens"] for x in receipts) / len(receipts),
        "attempt_histogram": {str(i): sum(x["attempt"] == i for x in receipts) for i in range(1, args.attempts+1)},
    })


if __name__ == "__main__":
    main()
