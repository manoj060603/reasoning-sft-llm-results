"""Export original dataset rows and auditable, response-masked training caches."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import json
import multiprocessing
import os
from pathlib import Path

from reasoning.data import alpaca, digest, read_rows, write_json
from reasoning.features import token_ids
from deadline.prepare_transfer import write_jsonl


_WORKER_TOKENIZER = None
_CHUNK_SIZE = 16


def _initialize_tokenizer(tokenizer_path):
    # Spawn workers load only the tokenizer, never a model or CUDA runtime.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


def _tokenize_chunk(rows):
    return [token_ids(_WORKER_TOKENIZER, row) for row in rows]


def _ordered_chunks(executor, rows, workers):
    """Keep at most two 16-row chunks per worker outstanding, in source order."""
    rows, pending = iter(rows), deque()

    def submit():
        chunk = list(islice(rows, _CHUNK_SIZE))
        if chunk:
            pending.append(executor.submit(_tokenize_chunk, chunk))
        return bool(chunk)

    for _ in range(2 * workers):
        if not submit():
            break
    while pending:
        yield from pending.popleft().result()
        submit()


def _tokenized(selected, tokenizer, workers, tokenizer_path):
    if workers == 1:
        for row in selected:
            yield token_ids(tokenizer, row)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize_tokenizer,
                             initargs=(str(Path(tokenizer_path).resolve()),)) as executor:
        yield from _ordered_chunks(executor, selected, workers)


def export(pool, identifiers, tokenizer, destination, role, cutoff=8192, expected_rows=15000,
           *, workers=1, tokenizer_path=None):
    """Export unchanged originals; optional workers load the same local tokenizer.

    Parallelism changes execution only: serialization, masks, ordering and the
    returned metadata remain byte-identical to serial export. The caller must
    supply the local tokenizer directory corresponding to ``tokenizer``.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("Tokenization workers must be a positive integer.")
    if workers > 1 and (tokenizer_path is None or not Path(tokenizer_path).is_dir()):
        raise ValueError("Parallel tokenization requires an existing local tokenizer_path.")
    if len(identifiers) != expected_rows or len(set(identifiers)) != expected_rows:
        raise ValueError("The training budget requires exactly 15,000 unique prompt IDs.")
    by_id = {row["id"]: row for row in pool}
    if len(by_id) != len(pool) or any(identifier not in by_id for identifier in identifiers):
        raise ValueError("Training IDs must uniquely identify unchanged source records.")
    destination = Path(destination)
    if (destination / "cache_meta.json").exists() or (destination / "training.jsonl").exists():
        raise ValueError("Refusing to overwrite an existing training export.")
    selected = [by_id[identifier] for identifier in identifiers]
    truncated = []
    original_tokens = 0
    supervised_tokens = 0

    def records():
        nonlocal original_tokens, supervised_tokens
        for row, (prompt, full) in zip(selected, _tokenized(selected, tokenizer, workers, tokenizer_path), strict=True):
            if len(prompt) >= cutoff:
                raise ValueError("Prompt exceeds the declared training capacity.")
            original_tokens += len(full)
            if len(full) > cutoff:
                if role == "advanced":
                    raise ValueError("Advanced selection must preserve complete original targets.")
                truncated.append(row["id"])
            cached_ids = full[:cutoff]
            record = {"id": row["id"], "input_ids": cached_ids,
                      "labels": [-100] * len(prompt) + cached_ids[len(prompt):],
                      "fold": -1, "question_sha256": row["id"]}
            record["prefix_truncated"] = len(full) > cutoff
            supervised_tokens += sum(label != -100 for label in record["labels"])
            yield record

    write_jsonl(destination / "training.jsonl", records())
    write_json(destination / "dataset.json", alpaca(selected))
    write_json(destination / "selected_ids.json", identifiers)
    metadata = {"cache_sha256": digest(destination / "training.jsonl"),
        "dataset_sha256": digest(destination / "dataset.json"),
        "selected_ids_sha256": digest(destination / "selected_ids.json"),
        "rows": len(selected), "expected_rows": len(selected), "cutoff": cutoff,
        "dataset_role": role, "truncation_policy": "prefix_truncate" if truncated else "none",
        "truncated_rows": len(truncated), "truncated_ids": truncated,
        "original_serialized_tokens": original_tokens, "supervised_training_tokens": supervised_tokens,
        "unique_rows_dropped": 0, "target_rewriting": False,
        "note": "Dataset export retains original responses; random-arm training may use declared prefixes."}
    write_json(destination / "cache_meta.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--ids", required=True, help="JSON ID list or selection manifest with datasets.<role>.selected_ids")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--role", choices=["random", "advanced"], required=True)
    parser.add_argument("--cutoff", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    identifiers = json.loads(Path(args.ids).read_text())
    if isinstance(identifiers, dict):
        identifiers = identifiers["datasets"][args.role]["selected_ids"]
    metadata = export(read_rows(args.pool), identifiers, tokenizer, args.output, args.role, args.cutoff,
                      workers=args.workers, tokenizer_path=args.model)
    metadata.update(pool_sha256=digest(args.pool), selection_source_sha256=digest(args.ids), model=args.model)
    write_json(Path(args.output) / "cache_meta.json", metadata)
    print(json.dumps({key: value for key, value in metadata.items() if key != "truncated_ids"}, indent=2))


if __name__ == "__main__":
    main()
