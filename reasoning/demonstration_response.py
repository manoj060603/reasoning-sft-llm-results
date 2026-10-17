"""Training-only demonstration-response preparation; original targets never change.

Plan and export are CPU-only. Score is the full resumable B200 preparation pass,
not a pilot. Public benchmarks are not inputs to any selection operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from .data import SYSTEM, alpaca, boxed, digest, read_rows, write_json
from .features import repetition, token_ids
from .select import allocate, eligible
from .skill_graph import label_items, load_labels

SCHEMA = "demonstration-response-v1"
CATEGORIES = {"retention": 3000, "manageable": 6000, "missed": 6000}
POOL_SHA = "c115f4331f485458616b8e29a3715422b5d4479396cf50c287183014820eabc9"
FEATURES_SHA = "039cd63692d528dcce2caf2be2bfed85be917a66298a2532933ef1107f9db979"
LABELS_SHA = "af60cf26d836eb733aace1267cc1139bcc66891a5968c198b7b50ff2e2ec0a4f"


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def stable_seed(seed, row_id, condition, sample):
    return int(text_hash(f"{seed}:{row_id}:{condition}:{sample}")[:8], 16)


def proportional(capacities, size):
    total = sum(capacities.values())
    if not 0 <= size <= total:
        raise ValueError("insufficient proportional capacity")
    if not size:
        return {k: 0 for k in capacities}
    q = {k: size * v // total for k, v in capacities.items()}
    order = sorted(capacities, key=lambda k: (-(size * capacities[k] % total), str(k)))
    for k in order[:size - sum(q.values())]:
        q[k] += 1
    return q


def prefix_leak_reason(prefix, gold):
    if re.search(r"\\boxed|final\s*answer|最终答案|最后答案", prefix, re.I):
        return "explicit_answer"
    answer = re.sub(r"\s+", "", gold).casefold()
    normalized = (prefix.casefold() if re.fullmatch(r"[A-Za-z0-9.+/-]+", answer)
                  else re.sub(r"\s+", "", prefix).casefold())
    if answer and re.search(r"(?<![\w.])" + re.escape(answer) + r"(?![\w.])", normalized):
        return "answer_surface_overlap"
    return None


def make_prefix(tokenizer, response):
    """Conservative answer withholding; an absent leak is not a proof certificate."""
    gold = boxed(response)
    if gold is None:
        return "", "missing_box"
    ids = tokenizer.encode(response, add_special_tokens=False)
    limit = min(len(ids) // 4, 768)
    text = tokenizer.decode(ids[:limit], skip_special_tokens=False)
    end = text.rfind("\n\n")
    if end <= 0:
        return "", "no_complete_paragraph"
    prefix = text[:end].rstrip()
    reason = prefix_leak_reason(prefix, gold)
    if reason:
        return "", reason
    if not prefix:
        return "", "empty_prefix"
    return prefix, None


def continuation_ids(tokenizer, row, prefix):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["instruction"]}]
    normal = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not prefix:
        return normal
    result = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": prefix}],
                                            tokenize=True, add_generation_prompt=False,
                                            continue_final_message=True)
    if result[:len(normal)] != normal or len(result) <= len(normal):
        raise ValueError("assistant continuation does not preserve the normal prompt")
    if tokenizer.eos_token_id in result[len(normal):]:
        raise ValueError("continuation prefix includes assistant EOS")
    if tokenizer.decode(result[len(normal):], skip_special_tokens=False) != prefix:
        raise ValueError("continuation prefix token round trip differs")
    return result


def candidate_order(rows, features, count, seed):
    """Freeze cluster/quartile reserve orders before any model output is read."""
    fmap = {f["id"]: f for f in features}
    if len(fmap) != len(features) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate source identity")
    if set(fmap) != {r["id"] for r in rows}:
        raise ValueError("feature coverage mismatch")
    groups = defaultdict(list)
    for row in rows:
        if eligible(fmap[row["id"]], 8192):
            groups[fmap[row["id"]]["cluster"]].append(row["id"])
    quotas = allocate({k: len(v) for k, v in groups.items()}, count)
    result = []
    for cluster in sorted(groups, key=str):
        ids = sorted(groups[cluster], key=lambda x: (fmap[x]["response_tokens"], x))
        quartiles = defaultdict(list)
        for i, row_id in enumerate(ids):
            quartiles[min(3, i * 4 // len(ids))].append(row_id)
        qs = proportional({k: len(v) for k, v in quartiles.items()}, quotas[cluster])
        for quartile in sorted(quartiles):
            ordered = sorted(quartiles[quartile], key=lambda x: (text_hash(f"{seed}:{x}"), x))
            result.append({"cluster": cluster, "quartile": quartile,
                           "count": qs[quartile], "reserve_ids": ordered})
    return result


def load_source(pool, features, labels):
    for path, expected in [(pool, POOL_SHA), (features, FEATURES_SHA), (labels, LABELS_SHA)]:
        if digest(path) != expected:
            raise ValueError(f"frozen source hash mismatch: {path}")
    rows, fs = read_rows(pool), read_rows(features)
    if len(rows) != 83037 or len(fs) != len(rows):
        raise ValueError("wrong approved pool size")
    ls = load_labels(Path(labels), [r["id"] for r in rows])
    return rows, fs, ls


def make_plan(args):
    from transformers import AutoTokenizer
    rows, fs, labels = load_source(args.pool, args.features, args.labels)
    by_id, fmap = {r["id"]: r for r in rows}, {f["id"]: f for f in fs}
    label_map = {r["id"]: r for r in labels}
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    cache, rejected = {}, []
    for feature in fs:
        if eligible(feature, 8192):
            row_id = feature["id"]
            row = by_id[row_id]
            if label_map[row_id]["answer_form"] in ("proof", "other"):
                rejected.append({"id": row_id, "reason": "unsupported_reference_form"})
                continue
            prompt, full = token_ids(tokenizer, row)
            if len(full) > 8192:
                rejected.append({"id": row_id, "reason": "exact_length"})
                continue
            if tokenizer.eos_token_id not in full[len(prompt):]:
                raise ValueError("original full response lacks an EOS token")
            prefix, reason = make_prefix(tokenizer, row["output"])
            if prefix:
                continuation_ids(tokenizer, row, prefix)
            cache[row_id] = {"id": row_id, "cluster": fmap[row_id]["cluster"], "prefix": prefix,
                             "prefix_reason": reason, "exact_tokens": len(full),
                             "prefix_tokens": len(tokenizer.encode(prefix, add_special_tokens=False)),
                             "response_sha256": text_hash(row["output"]),
                             "prompt_sha256": text_hash(row["instruction"]),
                             "prefix_sha256": text_hash(prefix)}
    if len(cache) < 30000:
        raise ValueError(f"only {len(cache)} exact eligible supported examples; need 30K")
    selected = []
    valid_rows = [r for r in rows if r["id"] in cache]
    valid_features = [f for f in fs if f["id"] in cache]
    for group in candidate_order(valid_rows, valid_features, 30000, 42):
        for row_id in group["reserve_ids"][:group["count"]]:
            selected.append({**cache[row_id], "quartile": group["quartile"]})
    if len(selected) != 30000 or len({r["id"] for r in selected}) != 30000:
        raise ValueError("wrong candidate pool")
    plan = {"schema": SCHEMA, "model": args.model, "seed": 42,
            "pool_sha256": POOL_SHA, "features_sha256": FEATURES_SHA,
            "labels_sha256": LABELS_SHA, "module_sha256": digest(__file__),
            "model_manifest_sha256": digest(args.model_manifest),
            "grader_sha256": {name: digest(Path("eval/utils") / name)
                              for name in ("grader.py", "parser.py", "math_normalization.py")},
            "size": 30000, "n_per_condition": 2, "max_new_tokens": 2048,
            "temperature": .6, "top_p": .95, "cutoff": 8192,
            "categories": CATEGORIES, "candidates": selected,
            "rejected": rejected, "prefix_usable": sum(bool(r["prefix"]) for r in selected)}
    write_json(args.plan, plan)
    print(json.dumps({k: v for k, v in plan.items() if k not in ("candidates", "rejected")}))
    if plan["prefix_usable"] < 12000:
        raise ValueError("fewer than 12K usable prefixes: acquisition categories cannot be filled")


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan["schema"] != SCHEMA or plan["module_sha256"] != digest(__file__):
        raise ValueError("plan was made with different preparation code")
    if plan["categories"] != CATEGORIES or len(plan["candidates"]) != plan["size"]:
        raise ValueError("invalid frozen plan")
    for name, expected in plan["grader_sha256"].items():
        if digest(Path("eval/utils") / name) != expected:
            raise ValueError("grader changed after preparation was frozen")
    if digest("artifacts/model.json") != plan["model_manifest_sha256"]:
        raise ValueError("base model manifest changed")
    return plan


def request_key(row_id, condition, sample):
    return f"{row_id}:{condition}:{sample}"


def read_ledger(path, requests):
    allowed = {x["key"]: x for x in requests}
    if len(allowed) != len(requests):
        raise ValueError("duplicate generation request")
    completed = {}
    if Path(path).exists():
        with Path(path).open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except ValueError as e:
                    raise ValueError(f"torn generation ledger at line {number}; preserve before repair") from e
                key = row.get("key")
                if key not in allowed or key in completed:
                    raise ValueError("foreign or duplicated generation ledger record")
                request = allowed[key]
                if row["seed"] != request["seed"] or row["prefix_sha256"] != text_hash(request["prefix"]):
                    raise ValueError("generation ledger request differs")
                if row["text_sha256"] != text_hash(row["text"]):
                    raise ValueError("generation ledger text checksum mismatch")
                completed[key] = row
    return completed


def grade_output(text, reference):
    """Only supported math strings; the original target is never sent to generation."""
    from .skill_graph_verify import answer_match
    if boxed(text) is None or boxed(reference) is None:
        return False, False
    try:
        return True, answer_match(text, reference, Path(__file__).resolve().parents[1])
    except (ValueError, TypeError, RuntimeError, TimeoutError):
        return False, False


def run_requests(args, plan, requests, by_id, ledger):
    from vllm import LLM, SamplingParams
    completed = read_ledger(ledger, requests)
    pending = [r for r in requests if r["key"] not in completed]
    if not pending:
        return completed
    llm = LLM(model=plan["model"], tokenizer=plan["model"], trust_remote_code=False,
              dtype="bfloat16", generation_config="vllm",
              max_model_len=12288, gpu_memory_utilization=.85, max_num_seqs=128)
    tokenizer = llm.get_tokenizer()
    Path(ledger).parent.mkdir(parents=True, exist_ok=True)
    with Path(ledger).open("a", encoding="utf-8") as stream:
        for start in range(0, len(pending), args.batch_size):
            started = time.monotonic()
            batch = pending[start:start + args.batch_size]
            prompts = [{"prompt_token_ids": continuation_ids(tokenizer, by_id[r["id"]], r["prefix"])}
                       for r in batch]
            params = [SamplingParams(temperature=.6, top_p=.95, top_k=-1,
                                     max_tokens=2048, seed=r["seed"], n=1) for r in batch]
            outputs = llm.generate(prompts, params, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError("incomplete generation batch")
            for request, output in zip(batch, outputs):
                if len(output.outputs) != 1:
                    raise RuntimeError("wrong number of completion samples")
                result = output.outputs[0]
                if result.finish_reason not in ("stop", "length"):
                    raise RuntimeError(f"invalid completion finish reason={result.finish_reason!r}; "
                                       f"text_chars={len(result.text)}, tokens={len(result.token_ids)}")
                # EOS or whitespace-only terminal responses are valid model outcomes.
                # Preserve them as unsupported/censored, never as wrong answers.
                empty_completion = not result.text.strip()
                # Reassemble the assistant text only for extraction, while the prefix
                # is mechanically answer-withheld and never contains the gold box.
                supported, correct = ((False, False) if empty_completion else
                                      grade_output(request["prefix"] + result.text,
                                                   by_id[request["id"]]["output"]))
                record = {k: request[k] for k in ("key", "id", "condition", "sample", "seed")}
                record.update(prefix_sha256=text_hash(request["prefix"]), text=result.text,
                              text_sha256=text_hash(result.text), finish_reason=result.finish_reason,
                              generated_tokens=len(result.token_ids), supported=supported, correct=correct,
                              empty_completion=empty_completion,
                              extracted_answer=boxed(request["prefix"] + result.text),
                              repetition=repetition(result.text))
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                completed[request["key"]] = record
            stream.flush()
            seconds = time.monotonic() - started
            tokens = sum(len(x.outputs[0].token_ids) for x in outputs)
            print(f"{Path(ledger).name}: {len(completed)}/{len(requests)}; "
                  f"batch_seconds={seconds:.2f}; output_tokens={tokens}; "
                  f"end_to_end_output_tokens_per_second={tokens/max(seconds, .001):.2f}", flush=True)
    return completed


def standard_requests(plan):
    requests = []
    for row in plan["candidates"]:
        for condition in ("normal", "assisted"):
            if condition == "assisted" and not row["prefix"]:
                continue
            for sample in range(2):
                requests.append({"key": request_key(row["id"], condition, sample), "id": row["id"],
                                 "condition": condition, "sample": sample,
                                 "seed": stable_seed(42, row["id"], condition, sample),
                                 "prefix": row["prefix"] if condition == "assisted" else ""})
    return requests


def categorize(candidate, records):
    def get(condition):
        return [records[request_key(candidate["id"], condition, i)] for i in range(2)]
    normal = get("normal")
    # Unsupported extraction and length exhaustion are censored, never "wrong".
    if any(x["finish_reason"] != "stop" or not x["supported"] for x in normal):
        return None, 0.0
    c0 = sum(x["correct"] for x in normal)
    if c0 == 2:
        return "retention", 0.0
    if not candidate["prefix"]:
        return None, 0.0
    assisted = get("assisted")
    if any(x["finish_reason"] != "stop" or not x["supported"] for x in assisted):
        return None, 0.0
    ch = sum(x["correct"] for x in assisted)
    gain = (ch - c0) / 2
    if c0 == 1 and ch == 2:
        return "manageable", gain
    if c0 == 0 and ch >= 1:
        return "missed", gain
    return None, gain


def select_categories(candidates, records, labels, counts):
    label_map = {x["id"]: x for x in labels}
    grouped = defaultdict(list)
    for item in candidates:
        category, gain = categorize(item, records)
        if category:
            grouped[category].append({**item, "category": category, "gain": gain})
    capacities = {name: len(grouped[name]) for name in counts}
    if any(capacities[name] < n for name, n in counts.items()):
        raise ValueError(f"category feasibility failure; requirements={counts}, capacities={capacities}")
    selected, quota_receipt = {}, {}
    for name, count in counts.items():
        clusters = defaultdict(list)
        for item in grouped[name]:
            clusters[item["cluster"]].append(item)
        quotas = allocate({k: len(v) for k, v in clusters.items()}, count)
        seen, chosen = Counter(), []
        for cluster in sorted(clusters, key=str):
            remaining = {x["id"]: x for x in clusters[cluster]}
            for _ in range(quotas[cluster]):
                def utility(row_id):
                    row = remaining[row_id]
                    items = label_items(label_map[row_id], row["cluster"])
                    novelty = sum(math.sqrt(seen[x] + 1) - math.sqrt(seen[x]) for x in items)
                    return (-row["gain"], -novelty, row_id)
                key = min(remaining, key=utility)
                row = remaining.pop(key)
                seen.update(label_items(label_map[key], row["cluster"]))
                chosen.append(row)
        selected[name] = chosen
        quota_receipt[name] = quotas
    return selected, {"capacities": capacities, "quotas": quota_receipt}


def curriculum(selected, seed=42):
    rng = random.Random(seed)
    anchors = list(selected["retention"])
    if len(anchors) % 2:
        raise ValueError("retention count must split evenly")
    # Distribute every two members of each cluster across halves, then balance.
    anchors.sort(key=lambda x: (str(x["cluster"]), text_hash(f"{seed}:{x['id']}")))
    halves = [anchors[::2], anchors[1::2]]
    result = []
    for name, retention in zip(("manageable", "missed"), halves):
        acquisition = list(selected[name]); rng.shuffle(acquisition); rng.shuffle(retention)
        block, a = [], 0
        for i, row in enumerate(acquisition):
            block.append(row)
            target = (i + 1) * len(retention) // len(acquisition)
            while a < target:
                block.append(retention[a]); a += 1
        result.extend(block)
    if len({x["id"] for x in result}) != len(result):
        raise ValueError("curriculum contains duplicate IDs")
    return result


def export(args):
    plan = load_plan(args.plan)
    rows, _, labels = load_source(args.pool, args.features, args.labels)
    by_id = {r["id"]: r for r in rows}
    records = read_ledger(args.ledger, standard_requests(plan))
    if len(records) != len(standard_requests(plan)):
        raise ValueError("incomplete generation ledger")
    selected, receipt = select_categories(plan["candidates"], records, labels, CATEGORIES)
    ordered = curriculum(selected)
    data = [by_id[x["id"]] for x in ordered]
    for item, row in zip(ordered, data):
        if text_hash(row["output"]) != item["response_sha256"]:
            raise ValueError("original target changed")
    out = Path(args.output)
    write_json(out / "advanced.json", alpaca(data))
    write_json(out / "dataset_info.json", {"advanced": {"file_name": "advanced.json"}})
    receipt.update(schema=SCHEMA, plan_sha256=digest(args.plan), ledger_sha256=digest(args.ledger),
                   selected_ids=[x["id"] for x in ordered], selected=ordered,
                   data_sha256=digest(out / "advanced.json"),
                   stage_boundary_examples=7500, original_responses_unchanged=True)
    write_json(out / "selection_manifest.json", receipt)
    print(json.dumps({"selected": len(data), "capacities": receipt["capacities"]}), flush=True)


def sign_test(differences):
    wins = sum(x > 0 for x in differences)
    losses = sum(x < 0 for x in differences)
    n = wins + losses
    p = sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n if n else 1.0
    return {"wins": wins, "losses": losses, "ties": len(differences) - n,
            "one_sided_p": p, "passed": p < .025 and wins > losses}


def audit_requests(plan, selection, label_map, by_id):
    """Fresh outcomes for all three conditions, independent of selection draws."""
    candidates = plan["candidates"]
    groups = defaultdict(list)
    for item in candidates:
        if item["prefix"]:
            groups[(item["cluster"], item["quartile"])].append(item)
    categories = defaultdict(list)
    for item in selection["selected"]:
        if not item["prefix"]:
            continue
        gold = boxed(by_id[item["id"]]["output"])
        if prefix_leak_reason(item["prefix"], gold):
            raise ValueError("selected demonstration exposes its target answer")
        partners = [x for x in groups[(item["cluster"], item["quartile"])]
                    if x["id"] != item["id"] and not prefix_leak_reason(x["prefix"], gold)]
        if partners:
            categories[item["category"]].append((item, min(partners, key=lambda x: x["id"])))
    chosen = []
    for name, count in [("retention", 86), ("manageable", 85), ("missed", 85)]:
        ordered = sorted(categories[name], key=lambda pair: (
            label_map[pair[0]["id"]]["language"], pair[0]["quartile"],
            -label_map[pair[0]["id"]]["composition_depth"],
            text_hash("audit:" + pair[0]["id"])))
        # Round-robin language/length strata so a large stratum cannot take all slots.
        strata = defaultdict(list)
        for pair in ordered:
            strata[(label_map[pair[0]["id"]]["language"], pair[0]["quartile"])].append(pair)
        picked = []
        while len(picked) < count:
            previous = len(picked)
            for key in sorted(strata):
                if strata[key] and len(picked) < count:
                    picked.append(strata[key].pop(0))
            if len(picked) == previous:
                raise ValueError("insufficient matched-prefix control capacity")
        chosen.extend(picked)
    requests = []
    for item, partner in chosen:
        for condition, prefix in [("audit_normal", ""), ("audit_assisted", item["prefix"]),
                                  ("audit_mismatch", partner["prefix"])]:
            for sample in range(2):
                requests.append({"key": request_key(item["id"], condition, sample), "id": item["id"],
                                 "condition": condition, "sample": sample, "prefix": prefix,
                                 "seed": stable_seed(42, item["id"], condition, sample)})
    return requests


def audit(args):
    plan = load_plan(args.plan)
    rows, _, labels = load_source(args.pool, args.features, args.labels)
    by_id, label_map = {r["id"]: r for r in rows}, {r["id"]: r for r in labels}
    out = Path(args.output)
    selection = json.loads((out / "selection_manifest.json").read_text())
    if selection["plan_sha256"] != digest(args.plan) or selection["ledger_sha256"] != digest(args.ledger):
        raise ValueError("selection does not match frozen preparation")
    requests = audit_requests(plan, selection, label_map, by_id)
    ledger = str(Path(args.ledger).with_name("audit_generations.jsonl"))
    records = run_requests(args, plan, requests, by_id, ledger)
    ids = list(dict.fromkeys(r["id"] for r in requests))
    pairs_normal, pairs_mismatch, per_id = [], [], []
    for row_id in ids:
        values, complete = {}, True
        for condition in ("audit_normal", "audit_assisted", "audit_mismatch"):
            samples = [records[request_key(row_id, condition, j)] for j in range(2)]
            complete &= all(x["finish_reason"] == "stop" and x["supported"] for x in samples)
            values[condition] = sum(x["finish_reason"] == "stop" and x["correct"] for x in samples)
        per_id.append({"id": row_id, "all_complete_supported": complete, **values})
        if complete:
            pairs_normal.append(values["audit_assisted"] - values["audit_normal"])
            pairs_mismatch.append(values["audit_assisted"] - values["audit_mismatch"])
    normal_test, mismatch_test = sign_test(pairs_normal), sign_test(pairs_mismatch)
    report = {"schema": SCHEMA, "fresh_audit_ids": ids, "results": per_id,
              "complete_supported_ids": len(pairs_normal), "versus_unassisted": normal_test,
              "versus_mismatched_prefix": mismatch_test, "alpha_per_test": .025,
              "selection_sha256": digest(out / "selection_manifest.json"),
              "audit_ledger_sha256": digest(ledger),
              "note": "Conditional answering evidence does not prove reasoning steps or SFT transfer."}
    report["passed"] = len(pairs_normal) >= 32 and normal_test["passed"] and mismatch_test["passed"]
    write_json(out / "audit_receipt.json", report)
    if not report["passed"]:
        raise ValueError("fresh demonstration audit inconclusive or negative; full training is not authorized by receipt")
    print(json.dumps({k: v for k, v in report.items() if k not in ("results", "fresh_audit_ids")}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "score", "export", "audit"))
    p.add_argument("--pool", default="artifacts/prepared_extended/pool.json")
    p.add_argument("--features", default="artifacts/features_extended.json")
    p.add_argument("--labels", default="artifacts/skillgraph-asft-15k/skill_labels.jsonl")
    p.add_argument("--model")
    p.add_argument("--model-manifest", default="artifacts/model.json")
    p.add_argument("--plan", default="artifacts/demonstration-response/plan.json")
    p.add_argument("--ledger", default="artifacts/demonstration-response/generations.jsonl")
    p.add_argument("--output", default="artifacts/datasets-demonstration-response-15k")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch size must be positive")
    if args.stage == "plan":
        if not args.model:
            p.error("plan requires pinned model path")
        make_plan(args)
    elif args.stage == "score":
        plan = load_plan(args.plan)
        rows, _, _ = load_source(args.pool, args.features, args.labels)
        run_requests(args, plan, standard_requests(plan), {r["id"]: r for r in rows}, args.ledger)
    elif args.stage == "audit":
        audit(args)
    else:
        export(args)


if __name__ == "__main__":
    main()
