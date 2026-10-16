"""Add batched base-model response NLL to an existing feature file.

The scorer preserves the existing diversity clusters and static quality features.  It
only adds a teacher-forced response NLL from the pinned base model.  Progress is
append-only JSONL so a preempted ARC job can resume without rescoring completed rows.
"""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path

from .data import digest, read_rows, write_json
from .features import token_ids


def response_nll(logits, input_ids, prompt_lengths, response_lengths):
    """Return one mean causal NLL per example over its unpadded response tokens."""
    import torch
    import torch.nn.functional as F

    shifted_logits = logits[:, :-1].contiguous()
    shifted_targets = input_ids[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shifted_logits.float().view(-1, shifted_logits.shape[-1]),
        shifted_targets.view(-1), reduction="none"
    ).view(shifted_targets.shape)
    positions = torch.arange(shifted_targets.shape[1], device=input_ids.device)[None, :]
    # shifted target position p predicts original token p+1.  Response begins at
    # original prompt_lengths, hence p >= prompt_lengths-1.
    starts = torch.as_tensor(prompt_lengths, device=input_ids.device)[:, None] - 1
    ends = starts + torch.as_tensor(response_lengths, device=input_ids.device)[:, None]
    mask = (positions >= starts) & (positions < ends)
    counts = mask.sum(1)
    if (counts <= 0).any():
        raise ValueError("response NLL received an empty response span")
    return (token_losses * mask).sum(1) / counts


def make_batches(records, max_batch_tokens):
    """Greedily batch tokenized records under a padded-token budget."""
    batch, longest = [], 0
    for record in records:
        length = len(record[2])
        candidate_longest = max(longest, length)
        if batch and candidate_longest * (len(batch) + 1) > max_batch_tokens:
            yield batch
            batch, longest = [], 0
        batch.append(record)
        longest = max(longest, length)
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--nll-tokens", type=int, default=512)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--flush-every", type=int, default=256)
    args = parser.parse_args()
    if min(args.nll_tokens, args.max_batch_tokens, args.flush_every) < 1:
        parser.error("token and flush limits must be positive")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        parser.error("base-model NLL scoring requires CUDA")

    rows, features = read_rows(args.input), read_rows(args.features)
    if len(rows) != len(features) or [r["id"] for r in rows] != [f["id"] for f in features]:
        raise ValueError("features must match the exact pool in order")
    source_meta = json.loads(Path(args.features + ".meta.json").read_text())
    if source_meta["input_sha256"] != digest(args.input):
        raise ValueError("static features are stale for the requested pool")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).to("cuda").eval()

    run_key = {
        "schema": "base-response-nll-v1",
        "input_sha256": digest(args.input),
        "features_sha256": digest(args.features),
        "model": args.model,
        "revision": args.revision,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "resolved_tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
        "nll_tokens": args.nll_tokens,
        "max_batch_tokens": args.max_batch_tokens,
    }
    partial = Path(args.output + ".partial.jsonl")
    header = Path(args.output + ".partial.meta.json")
    completed = {}
    if partial.exists() or header.exists():
        if not (partial.exists() and header.exists()):
            raise ValueError("incomplete NLL resume files")
        if json.loads(header.read_text()) != run_key:
            raise ValueError("NLL partial belongs to a different run")
        for line in partial.read_text().splitlines():
            item = json.loads(line)
            if item["id"] in completed:
                raise ValueError("duplicate ID in NLL partial")
            completed[item["id"]] = item["nll"]
    else:
        write_json(header, run_key)

    pending = []
    for row, feature in zip(rows, features):
        if row["id"] in completed:
            continue
        prompt, full = token_ids(tokenizer, row)
        end = min(len(full), len(prompt) + args.nll_tokens)
        if end <= len(prompt) or end > model.config.max_position_embeddings:
            raise ValueError(f"invalid NLL span for {row['id']}")
        pending.append((row["id"], len(prompt), full[:end]))
    # Sorting only changes execution order, never the final feature ordering.
    pending.sort(key=lambda item: (len(item[2]), item[0]))

    buffered = []
    with partial.open("a", encoding="utf-8") as stream:
        for batch in make_batches(pending, args.max_batch_tokens):
            longest = max(len(item[2]) for item in batch)
            ids = torch.full((len(batch), longest), tokenizer.pad_token_id,
                             dtype=torch.long, device="cuda")
            attention = torch.zeros_like(ids)
            prompt_lengths, response_lengths = [], []
            for index, (_, prompt_length, sequence) in enumerate(batch):
                ids[index, :len(sequence)] = torch.tensor(sequence, device="cuda")
                attention[index, :len(sequence)] = 1
                prompt_lengths.append(prompt_length)
                response_lengths.append(len(sequence) - prompt_length)
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=attention).logits
                losses = response_nll(logits, ids, prompt_lengths, response_lengths)
            for (row_id, _, _), loss in zip(batch, losses.tolist()):
                if not math.isfinite(loss):
                    raise ValueError(f"nonfinite NLL for {row_id}")
                buffered.append({"id": row_id, "nll": float(loss)})
            if len(buffered) >= args.flush_every:
                for item in buffered:
                    stream.write(json.dumps(item, separators=(",", ":")) + "\n")
                    completed[item["id"]] = item["nll"]
                stream.flush()
                buffered.clear()
                print(f"{len(completed)}/{len(rows)}", flush=True)
        for item in buffered:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
            completed[item["id"]] = item["nll"]
        stream.flush()

    if len(completed) != len(rows):
        raise ValueError(f"NLL coverage is incomplete: {len(completed)}/{len(rows)}")
    output = [dict(feature, nll=completed[feature["id"]]) for feature in features]
    write_json(args.output, output)
    write_json(args.output + ".meta.json", {
        "input_sha256": digest(args.input),
        "arguments": vars(args),
        "provenance": {
            "source_features_sha256": digest(args.features),
            "source_features_meta_sha256": digest(args.features + ".meta.json"),
            "code_sha256": digest(__file__),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        },
        **run_key,
    })
    partial.unlink()
    header.unlink()


if __name__ == "__main__":
    main()
