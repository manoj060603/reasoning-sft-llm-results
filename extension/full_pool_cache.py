"""Unadmitted private complete-pair cache format; stdlib and caller-owned I/O.

Nothing is loaded, written or executed on import. A separately admitted caller
owns compute/source/tokenizer/plan/exclusion guards and the original native
serialization. This helper verifies supplied tokens/hashes and physical bytes;
it cannot prove an unseen original string, tokenizer, prefix or source file.
There is no tokenizer, model, fit, draw, scoring, CLI, admission, resume or repair.

write_cache creates ONE new child directory of an existing owner0700 directory.
Each ascending-ID shard has at most256 rows: raw little-endian u32 full tokens,
u64 token offsets N+1, u32 prompt lengths and a bounded private JSON row index.
No labels/text are stored; derive_labels produces unshifted -100-prefix labels
on demand. Native EOS and the caller's exact plan-bound tail are never appended
or altered. A final atomic-exclusive manifest FILE is the commit marker.
Failures retain private partial files and cannot reuse the directory.

open_cache requires independent expected IDs/identity/suffix and a manifest
FILE SHA. ALL files and every binary/index row are verified before any row is
yielded. A single ordered pass is then reverified shard-by-shard and at close.
Consumers must finish/close successfully before claiming complete consumption.
Ordinary file changes are detected via open descriptors, hashes and stat
snapshots; hostile same-user/out-of-band upstream mutation is not an admission
guarantee. The caller must preserve its own original source snapshot.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct


SCHEMA = "extension-full-pool-complete-cache-v1"
INDEX_SCHEMA = "extension-full-pool-cache-shard-index-v1"
PROTOCOL_SHA256 = "160948a0477c6f234a587ac2cc32297f5dc229ba6c0e3a83cea89242fd3adc29"
APPROVAL_SHA256 = "785b445913cc7c9c76c56474db4c89eb01a39c65bfbb50d1e8f87f7a6d0bfdc6"
SHARD_ROWS, VOCAB_SIZE, EOS_TOKEN_ID = 256, 151936, 151645
CONTEXT_TOKENS, RESERVE_TOKENS, MAX_TOKENS = 32768, 32, 32736
NATIVE_ROWS, MAX_IO_BYTES, MAX_JSON_BYTES = 82961, 1 << 20, 2 << 20
MAX_ROW_TOKEN_BYTES = 130944
FORMAT = {"shard_rows": SHARD_ROWS, "tokens": "uint32_little_endian", "offsets": "uint64_little_endian_N_plus_1",
    "prompt_lengths": "uint32_little_endian", "offset_unit": "tokens", "vocabulary": VOCAB_SIZE,
    "eos_token_id": EOS_TOKEN_ID, "context_tokens": CONTEXT_TOKENS, "unused_context_reserve": RESERVE_TOKENS,
    "maximum_complete_tokens": MAX_TOKENS, "maximum_row_token_bytes": MAX_ROW_TOKEN_BYTES,
    "maximum_io_bytes": MAX_IO_BYTES, "labels_stored": False, "causal_shift_applied": False,
    "mask_value_encoding": "canonical_JSON_integer_array_0_prompt_1_active"}
IDENTITY_KEYS = frozenset({"protocol_file_sha256", "root_approval_file_sha256", "preparation_plan_file_sha256",
    "source_identity_sha256", "tokenizer_identity_sha256", "base_identity_file_sha256", "pool_file_sha256",
    "raw_source_file_sha256", "exclusion_receipt_file_sha256", "exclusion_receipt_value_sha256",
    "eligible_sorted_ids_value_sha256", "native_suffix_value_sha256"})
FIXED_IDENTITIES = {"protocol_file_sha256": PROTOCOL_SHA256, "root_approval_file_sha256": APPROVAL_SHA256,
    "base_identity_file_sha256": "81af59dbb365b8559ad0d853dbda245aa4ac3ff0bedc03428e50fed65d7f7ae2",
    "pool_file_sha256": "4e8ac972757555c2c0c1c0d31f87f49fc79d19815fb89b49bb748cee3ea145b0",
    "raw_source_file_sha256": "f064d1a5fb39b9f493f678d5d6350a9acbfb2a2fa85400d0947fe72578ff28d8",
    "exclusion_receipt_file_sha256": "0066e69a996cf963173018d68196d7ff69e658420c40b250941c2fe8bb6da406"}
ROW_INPUT_KEYS = frozenset({"id", "input_ids", "prompt_tokens", "original_pair_sha256", "prompt_token_ids_sha256",
    "full_token_ids_sha256", "labels_sha256", "mask_sha256", "native_suffix_sha256"})
ROW_META_KEYS = frozenset({"id", "shard_index", "row_in_shard", "token_offset", "prompt_tokens", "total_tokens",
    "active_tokens", "eos_tokens", "non_eos_tokens", "tail_tokens", "original_pair_sha256", "prompt_token_ids_sha256",
    "full_token_ids_sha256", "labels_sha256", "mask_sha256", "native_suffix_sha256", "native_tail_sha256",
    "token_bytes_sha256", "cache_record_sha256"})
MANIFEST_KEYS = frozenset({"schema", "complete", "fixture_only", "format", "identity", "native_suffix", "rows",
    "eligible_sorted_ids_value_sha256", "total_tokens", "active_tokens", "eos_tokens", "tail_tokens", "shards",
    "native_population_provenance_or_compute_admitted"})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _int(value, low, high=None):
    return type(value) is int and value >= low and (high is None or value <= high)


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _plain(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_plain(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _plain(item) for key, item in value.items())
    return False


def canonical_value_bytes(value):
    _require(_plain(value), "Finite plain JSON values only")
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def value_sha256(value):
    return hashlib.sha256(canonical_value_bytes(value)).hexdigest()


def _parse(payload):
    _require(type(payload) is bytes and 0 < len(payload) <= MAX_JSON_BYTES, "Bounded JSON bytes required")
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON value")))
    _require(_plain(value), "Unsupported JSON value")
    return value


def _ids(expected_ids, fixture_only):
    _require(type(fixture_only) is bool and type(expected_ids) in (list, tuple), "Explicit expected ID sequence/fixture policy required")
    ids = tuple(expected_ids)
    _require(0 < len(ids) <= NATIVE_ROWS and all(_sha(identifier) for identifier in ids) and
        all(a < b for a, b in zip(ids, ids[1:])) and (fixture_only or len(ids) == NATIVE_ROWS),
        "Exact unique ascending expected IDs required; small counts must be explicit NONNATIVE fixtures")
    return ids


def _suffix(native_suffix):
    _require(type(native_suffix) in (list, tuple) and 1 <= len(native_suffix) <= 32 and
        all(_int(token, 0, VOCAB_SIZE - 1) for token in native_suffix) and native_suffix[0] == EOS_TOKEN_ID and
        list(native_suffix).count(EOS_TOKEN_ID) == 1, "Exact plan-supplied native EOS-plus-tail tuple required")
    return tuple(native_suffix)


def validate_identity(identity, *, expected_ids, native_suffix, fixture_only=False):
    ids, suffix = _ids(expected_ids, fixture_only), _suffix(native_suffix)
    _require(type(identity) is dict and set(identity) == IDENTITY_KEYS and all(_sha(value) for value in identity.values()) and
        all(identity[name] == value for name, value in FIXED_IDENTITIES.items()) and
        identity["eligible_sorted_ids_value_sha256"] == value_sha256(list(ids)) and
        identity["native_suffix_value_sha256"] == value_sha256(list(suffix)),
        "Supplied caller identity/expected-ID/suffix/protocol bindings differ")
    return json.loads(canonical_value_bytes(identity))


def _tokens(input_ids, prompt_tokens):
    _require(type(input_ids) in (list, tuple) and 2 <= len(input_ids) <= MAX_TOKENS and
        all(_int(token, 0, VOCAB_SIZE - 1) for token in input_ids) and
        _int(prompt_tokens, 1, len(input_ids) - 1), "Complete native vocabulary IDs, positive prompt and active target required")
    tokens = tuple(input_ids)
    _require(len(tokens) * 4 <= MAX_ROW_TOKEN_BYTES, "Row exceeds fixed130944 token-byte bound")
    return tokens


def derive_labels(input_ids, prompt_tokens):
    """Unshifted labels on demand; never append/truncate/rewrite a target."""
    ids = _tokens(input_ids, prompt_tokens)
    return [-100] * prompt_tokens + list(ids[prompt_tokens:])


def cache_row(identifier, input_ids, prompt_tokens, original_pair_sha256, *, native_suffix):
    """Bind supplied native tokens; this cannot verify unseen original text."""
    ids, suffix = _tokens(input_ids, prompt_tokens), _suffix(native_suffix)
    _require(_sha(identifier) and _sha(original_pair_sha256) and len(ids) - prompt_tokens >= len(suffix) and
        ids[-len(suffix):] == suffix, "Original pair/ID or exact complete native suffix differs")
    return {"id": identifier, "input_ids": list(ids), "prompt_tokens": prompt_tokens,
        "original_pair_sha256": original_pair_sha256, "prompt_token_ids_sha256": value_sha256(list(ids[:prompt_tokens])),
        "full_token_ids_sha256": value_sha256(list(ids)), "labels_sha256": value_sha256(derive_labels(ids, prompt_tokens)),
        "mask_sha256": value_sha256([0] * prompt_tokens + [1] * (len(ids) - prompt_tokens)),
        "native_suffix_sha256": value_sha256(list(suffix))}


def _freeze_row(row, suffix):
    _require(type(row) is dict and set(row) == ROW_INPUT_KEYS, "Unsupported cache input-row schema")
    tokens = _tokens(row["input_ids"], row["prompt_tokens"])
    _require(_sha(row["id"]) and all(_sha(value) for key, value in row.items() if key.endswith("sha256")),
        "Exact native row/source/hash scalar types required")
    before = canonical_value_bytes({**row, "input_ids": list(tokens)})
    expected = cache_row(row["id"], tokens, row["prompt_tokens"], row["original_pair_sha256"], native_suffix=suffix)
    _require(before and before == canonical_value_bytes(expected) and
        before == canonical_value_bytes({**row, "input_ids": list(row["input_ids"])}), "Input row changed or native token/mask/label hash differs")
    return tuple(expected["input_ids"]), {key: value for key, value in expected.items() if key != "input_ids"}


def _metadata(ids, info, suffix, shard, row, offset):
    packed = struct.pack("<" + str(len(ids)) + "I", *ids)
    active = len(ids) - info["prompt_tokens"]
    eos = ids[info["prompt_tokens"]:].count(EOS_TOKEN_ID)
    result = {**info, "shard_index": shard, "row_in_shard": row, "token_offset": offset,
        "total_tokens": len(ids), "active_tokens": active, "eos_tokens": eos, "non_eos_tokens": active - eos,
        "tail_tokens": len(suffix) - 1, "native_tail_sha256": value_sha256(list(suffix[1:])),
        "token_bytes_sha256": hashlib.sha256(packed).hexdigest()}
    result["cache_record_sha256"] = value_sha256(result)
    return result, packed


def _validate_metadata(metadata):
    _require(type(metadata) is dict and set(metadata) == ROW_META_KEYS and _sha(metadata["id"]) and
        all(_sha(value) for key, value in metadata.items() if key.endswith("sha256")) and
        _int(metadata["shard_index"], 0, (NATIVE_ROWS - 1) // SHARD_ROWS) and
        _int(metadata["row_in_shard"], 0, SHARD_ROWS - 1) and
        metadata["shard_index"] * SHARD_ROWS + metadata["row_in_shard"] < NATIVE_ROWS and
        _int(metadata["token_offset"], 2 * metadata["row_in_shard"], MAX_TOKENS * metadata["row_in_shard"]) and
        _int(metadata["total_tokens"], 2, MAX_TOKENS) and _int(metadata["prompt_tokens"], 1, metadata["total_tokens"] - 1) and
        _int(metadata["active_tokens"], 1, metadata["total_tokens"] - 1) and
        metadata["active_tokens"] == metadata["total_tokens"] - metadata["prompt_tokens"] and
        _int(metadata["eos_tokens"], 1, metadata["active_tokens"]) and
        _int(metadata["non_eos_tokens"], 0, metadata["active_tokens"] - 1) and
        metadata["eos_tokens"] + metadata["non_eos_tokens"] == metadata["active_tokens"] and
        _int(metadata["tail_tokens"], 0, min(31, metadata["non_eos_tokens"])) and
        metadata["cache_record_sha256"] == value_sha256({key: value for key, value in metadata.items() if key != "cache_record_sha256"}),
        "Exact hash-bound complete-cache row metadata required")
    return metadata


def selection_row(metadata):
    """Pure selector ROW_KEYS4 bridge; physical/native provenance stays caller-owned."""
    _validate_metadata(metadata)
    return {key: metadata[key] for key in ("total_tokens", "active_tokens", "original_pair_sha256", "cache_record_sha256")}


def _absolute(path):
    path = Path(path)
    _require(path.is_absolute() and ".." not in path.parts, "Explicit absolute cache path without traversal required")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.exists() or current.is_symlink():
            _require(not current.is_symlink(), "Symlink cache ancestor refused")
    return path


def _directory_stat(path):
    value = path.lstat()
    _require(stat.S_ISDIR(value.st_mode) and value.st_uid == os.getuid() and stat.S_IMODE(value.st_mode) == 0o700,
        "Existing owner0700 directory required")
    return value


def _fingerprint(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns,
        value.st_uid, stat.S_IMODE(value.st_mode), value.st_nlink)


class _Directory:
    def __init__(self, path):
        self.fd = None
        self.path = _absolute(path)
        before = _directory_stat(self.path)
        _require(all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")), "No-follow directory/file APIs required")
        self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            self.identity = (before.st_dev, before.st_ino)
            self.check()
        except BaseException:
            self.close()
            raise

    def check(self):
        _absolute(self.path)
        value, actual = _directory_stat(self.path), os.fstat(self.fd)
        _require((value.st_dev, value.st_ino) == self.identity == (actual.st_dev, actual.st_ino) and
            stat.S_ISDIR(actual.st_mode) and actual.st_uid == os.getuid() and stat.S_IMODE(actual.st_mode) == 0o700,
            "Cache directory identity/owner/mode changed")

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def names(self):
        self.check()
        return set(os.listdir(self.fd))

    def open(self, name, *, write=False):
        _require(type(name) is str and re.fullmatch(r"[A-Za-z0-9_.-]+", name) and name not in (".", ".."), "Unsafe cache filename")
        self.check()
        flags = os.O_NOFOLLOW | os.O_NONBLOCK | (os.O_WRONLY | os.O_CREAT | os.O_EXCL if write else os.O_RDONLY)
        fd = os.open(name, flags, 0o600, dir_fd=self.fd)
        try:
            value = os.fstat(fd)
            _require(stat.S_ISREG(value.st_mode) and value.st_uid == os.getuid() and
                stat.S_IMODE(value.st_mode) == 0o600 and value.st_nlink == 1, "Owner0600 ordinary single-link file required")
            self.check()
        except BaseException:
            os.close(fd)
            raise
        return fd


def _write(fd, payload):
    _require(type(payload) is bytes, "Only explicit byte buffers may be written")
    view, offset = memoryview(payload), 0
    while offset < len(view):
        count = os.write(fd, view[offset:offset + MAX_IO_BYTES])
        _require(_int(count, 1, min(MAX_IO_BYTES, len(view) - offset)), "Incomplete bounded cache write")
        offset += count


def _read_exact(fd, size):
    _require(_int(size, 1, MAX_ROW_TOKEN_BYTES), "Bounded binary token-row read required")
    pieces, remaining = [], size
    while remaining:
        chunk = os.read(fd, min(remaining, MAX_IO_BYTES))
        _require(bool(chunk), "Truncated binary token row")
        pieces.append(chunk)
        remaining -= len(chunk)
    return b"".join(pieces)


def _file_record(directory, name, *, expected_bytes=None, maximum=SHARD_ROWS * MAX_ROW_TOKEN_BYTES):
    _require(_int(maximum, 1, SHARD_ROWS * MAX_ROW_TOKEN_BYTES) and
        (expected_bytes is None or _int(expected_bytes, 1, maximum)), "Bounded physical FILE size policy required")
    fd = directory.open(name)
    try:
        before, digest, size = os.fstat(fd), hashlib.sha256(), 0
        _require(0 < before.st_size <= maximum and (expected_bytes is None or before.st_size == expected_bytes),
            "Physical cache file exceeds or differs from its exact byte envelope")
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, MAX_IO_BYTES))
            _require(bool(chunk), "Truncated cache file while hashing")
            digest.update(chunk)
            size += len(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        _require(not os.read(fd, 1) and _fingerprint(before) == _fingerprint(after) and size == before.st_size,
            "Cache file changed while hashing or is empty")
        return {"sha256": digest.hexdigest(), "bytes": size}, _fingerprint(after)
    finally:
        os.close(fd)


def _small_file(directory, name, maximum=MAX_JSON_BYTES):
    fd = directory.open(name)
    try:
        before = os.fstat(fd)
        _require(0 < before.st_size <= maximum, "Metadata/binary index exceeds its fixed byte bound")
        pieces, remaining = [], before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, MAX_IO_BYTES))
            _require(bool(chunk), "Truncated cache metadata/index")
            pieces.append(chunk)
            remaining -= len(chunk)
        _require(not os.read(fd, 1) and _fingerprint(before) == _fingerprint(os.fstat(fd)), "Cache metadata/index changed while reading")
        return b"".join(pieces)
    finally:
        os.close(fd)


def _save_file(directory, name, payload):
    fd = directory.open(name, write=True)
    try:
        _write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    record, fingerprint = _file_record(directory, name, expected_bytes=len(payload), maximum=MAX_JSON_BYTES)
    _require(record == {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}, "Written cache bytes differ")
    return record, fingerprint


def _part_names(index):
    base = "shard-" + str(index).zfill(6)
    return {"tokens": base + ".tokens.u32le", "offsets": base + ".offsets.u64le",
        "prompts": base + ".prompts.u32le", "index": base + ".index.json"}


def _unchanged(directory, fingerprints, names):
    _require(directory.names() == set(names), "Cache inventory changed or has unexpected files")
    for name, expected in fingerprints.items():
        fd = directory.open(name)
        try:
            _require(_fingerprint(os.fstat(fd)) == expected, "Cache part/manifest changed after verification")
        finally:
            os.close(fd)


def _publish_manifest(directory, manifest, fingerprints):
    payload = canonical_value_bytes(manifest) + b"\n"
    _require(len(payload) <= MAX_JSON_BYTES, "Manifest exceeds fixed byte bound")
    temporary, terminal, linked, inode = ".manifest.pending", "manifest.json", False, None
    try:
        _, pending_stat = _save_file(directory, temporary, payload)
        inode = pending_stat[:2]
        _unchanged(directory, fingerprints, set(fingerprints) | {temporary})
        os.link(temporary, terminal, src_dir_fd=directory.fd, dst_dir_fd=directory.fd, follow_symlinks=False)
        linked = True
        os.unlink(temporary, dir_fd=directory.fd)
        os.fsync(directory.fd)
        result, terminal_stat = _file_record(directory, terminal, expected_bytes=len(payload), maximum=MAX_JSON_BYTES)
        _require(result == {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}, "Committed manifest bytes differ")
        _unchanged(directory, {**fingerprints, terminal: terminal_stat}, set(fingerprints) | {terminal})
        return result
    except BaseException:
        if linked:
            try:
                value = os.stat(terminal, dir_fd=directory.fd, follow_symlinks=False)
                if (value.st_dev, value.st_ino) == inode:
                    os.unlink(terminal, dir_fd=directory.fd)
                    os.fsync(directory.fd)
            except OSError:
                pass
        raise


def write_cache(output, rows, *, expected_ids, identity, native_suffix, fixture_only=False):
    """Write a fresh cache; rows is a one-pass iterable of exact cache_row dicts.

    Output must not exist, even empty. Its parent must already be owner0700.
    No supplied source text or labels are stored, and no invalid row is skipped.
    Returning complete means format/byte construction only, not native admission.
    """
    ids, suffix = _ids(expected_ids, fixture_only), _suffix(native_suffix)
    identity = validate_identity(identity, expected_ids=ids, native_suffix=suffix, fixture_only=fixture_only)
    output = _absolute(output)
    _directory_stat(output.parent)
    _require(not output.exists() and not output.is_symlink(), "Cache output already exists; no reuse/resume")
    os.mkdir(output, 0o700)
    directory = _Directory(output)
    fingerprints, shards, totals = {}, [], {"total_tokens": 0, "active_tokens": 0, "eos_tokens": 0, "tail_tokens": 0}
    try:
        iterator, position = iter(rows), 0
        for shard in range((len(ids) + SHARD_ROWS - 1) // SHARD_ROWS):
            count = min(SHARD_ROWS, len(ids) - position)
            names, offsets, prompts, index_rows = _part_names(shard), [0], [], []
            token_fd = directory.open(names["tokens"], write=True)
            token_digest = hashlib.sha256()
            try:
                for row_index in range(count):
                    try:
                        row = next(iterator)
                    except StopIteration as error:
                        raise ValueError("Missing input row; no drop/refill/partial completion") from error
                    tokens, info = _freeze_row(row, suffix)
                    _require(info["id"] == ids[position], "Input IDs differ from exact expected ascending order")
                    metadata, packed = _metadata(tokens, info, suffix, shard, row_index, offsets[-1])
                    _write(token_fd, packed)
                    token_digest.update(packed)
                    offsets.append(offsets[-1] + len(tokens))
                    prompts.append(info["prompt_tokens"])
                    index_rows.append(metadata)
                    for key in totals:
                        totals[key] += metadata[key]
                    position += 1
                os.fsync(token_fd)
            finally:
                os.close(token_fd)
            token_record, token_stat = _file_record(directory, names["tokens"], expected_bytes=offsets[-1] * 4)
            _require(token_record == {"sha256": token_digest.hexdigest(), "bytes": offsets[-1] * 4}, "Complete token shard bytes differ")
            parts = {names["tokens"]: token_record}
            fingerprints[names["tokens"]] = token_stat
            payloads = {"offsets": struct.pack("<" + str(len(offsets)) + "Q", *offsets),
                "prompts": struct.pack("<" + str(len(prompts)) + "I", *prompts),
                "index": canonical_value_bytes({"schema": INDEX_SCHEMA, "shard_index": shard, "rows": index_rows}) + b"\n"}
            _require(len(payloads["index"]) <= MAX_JSON_BYTES, "Private256-row JSON index exceeds fixed byte bound")
            for kind, payload in payloads.items():
                parts[names[kind]], fingerprints[names[kind]] = _save_file(directory, names[kind], payload)
            shards.append({"index": shard, "rows": count, "first_id": index_rows[0]["id"], "last_id": index_rows[-1]["id"],
                "token_count": offsets[-1], "files": parts})
        sentinel = object()
        _require(next(iterator, sentinel) is sentinel, "Extra input row; no duplicate or population substitution")
        _require(position == len(ids), "Complete input coverage required")
        manifest = {"schema": SCHEMA, "complete": True, "fixture_only": fixture_only, "format": FORMAT,
            "identity": identity, "native_suffix": list(suffix), "rows": len(ids),
            "eligible_sorted_ids_value_sha256": value_sha256(list(ids)), **totals, "shards": shards,
            "native_population_provenance_or_compute_admitted": False}
        terminal = _publish_manifest(directory, manifest, fingerprints)
        return {"schema": SCHEMA, "complete": True, "fixture_only": fixture_only, "rows": len(ids), "shards": len(shards),
            "manifest_file_sha256": terminal["sha256"], "manifest_file_bytes": terminal["bytes"],
            "manifest_value_sha256": value_sha256(manifest), "identity_value_sha256": value_sha256(identity), **totals,
            "native_population_provenance_or_compute_admitted": False}
    finally:
        directory.close()


def _manifest(directory, expected_sha, ids, identity, suffix, fixture_only):
    _require(_sha(expected_sha), "Caller-pinned manifest physical FILE SHA required")
    payload = _small_file(directory, "manifest.json")
    _require(hashlib.sha256(payload).hexdigest() == expected_sha, "Manifest FILE SHA differs")
    value = _parse(payload)
    _require(type(value) is dict and set(value) == MANIFEST_KEYS and value["schema"] == SCHEMA and value["complete"] is True and
        value["fixture_only"] is fixture_only and canonical_value_bytes(value["format"]) == canonical_value_bytes(FORMAT) and
        value["identity"] == identity and type(value["native_suffix"]) is list and _suffix(value["native_suffix"]) == suffix and
        _int(value["rows"], len(ids), len(ids)) and
        value["eligible_sorted_ids_value_sha256"] == value_sha256(list(ids)) and
        value["native_population_provenance_or_compute_admitted"] is False and
        _int(value["total_tokens"], 2 * len(ids), MAX_TOKENS * len(ids)) and
        _int(value["active_tokens"], len(ids), (MAX_TOKENS - 1) * len(ids)) and
        _int(value["eos_tokens"], len(ids), value["active_tokens"]) and
        _int(value["tail_tokens"], (len(suffix) - 1) * len(ids), (len(suffix) - 1) * len(ids)) and
        type(value["shards"]) is list and len(value["shards"]) == (len(ids) + SHARD_ROWS - 1) // SHARD_ROWS,
        "Unsupported/incomplete manifest or independent caller identity differs")
    files = {"manifest.json"}
    for index, shard in enumerate(value["shards"]):
        names = _part_names(index)
        expected_count = min(SHARD_ROWS, len(ids) - index * SHARD_ROWS)
        _require(type(shard) is dict and set(shard) == {"index", "rows", "first_id", "last_id", "token_count", "files"} and
            _int(shard["index"], index, index) and _int(shard["rows"], expected_count, expected_count) and
            shard["first_id"] == ids[index * SHARD_ROWS] and shard["last_id"] == ids[index * SHARD_ROWS + expected_count - 1] and
            _int(shard["token_count"], 2 * expected_count, MAX_TOKENS * expected_count) and
            type(shard["files"]) is dict and set(shard["files"]) == set(names.values()), "Shard boundaries/order/path/count differ")
        for name, item in shard["files"].items():
            _require(type(item) is dict and set(item) == {"sha256", "bytes"} and _sha(item["sha256"]) and _int(item["bytes"], 1),
                "Exact shard physical file record required")
            files.add(name)
        _require(shard["files"][names["tokens"]]["bytes"] == shard["token_count"] * 4 and
            shard["files"][names["offsets"]]["bytes"] == (expected_count + 1) * 8 and
            shard["files"][names["prompts"]]["bytes"] == expected_count * 4 and
            shard["files"][names["index"]]["bytes"] <= MAX_JSON_BYTES, "Shard FILE byte envelopes differ")
    _require(directory.names() == files, "Unexpected/missing/partial cache inventory")
    return value, files


def _verified_shard(directory, shard, suffix, expected_ids):
    names, fingerprints = _part_names(shard["index"]), {}
    for name, expected in shard["files"].items():
        actual, fingerprints[name] = _file_record(directory, name, expected_bytes=expected["bytes"])
        _require(actual == expected, "Pinned shard physical FILE hash/length differs")
    count = shard["rows"]
    offset_bytes = _small_file(directory, names["offsets"], (SHARD_ROWS + 1) * 8)
    prompt_bytes = _small_file(directory, names["prompts"], SHARD_ROWS * 4)
    _require(len(offset_bytes) == (count + 1) * 8 and len(prompt_bytes) == count * 4, "Binary offset/prompt index lengths differ")
    offsets, prompts = struct.unpack("<" + str(count + 1) + "Q", offset_bytes), struct.unpack("<" + str(count) + "I", prompt_bytes)
    _require(offsets[0] == 0 and offsets[-1] == shard["token_count"] and all(a < b for a, b in zip(offsets, offsets[1:])) and
        shard["files"][names["tokens"]]["bytes"] == offsets[-1] * 4, "Corrupt/nonmonotonic/end-offset token index")
    index = _parse(_small_file(directory, names["index"]))
    _require(type(index) is dict and set(index) == {"schema", "shard_index", "rows"} and index["schema"] == INDEX_SCHEMA and
        _int(index["shard_index"], shard["index"], shard["index"]) and type(index["rows"]) is list and len(index["rows"]) == count,
        "Unsupported private row-index schema/count")
    fd = directory.open(names["tokens"])
    try:
        for row_number, metadata in enumerate(index["rows"]):
            length = offsets[row_number + 1] - offsets[row_number]
            _require(2 <= length <= MAX_TOKENS and 1 <= prompts[row_number] < length, "Row token offset/prompt envelope differs")
            payload = _read_exact(fd, length * 4)
            tokens = struct.unpack("<" + str(length) + "I", payload)
            _validate_metadata(metadata)
            row = cache_row(expected_ids[row_number], tokens, prompts[row_number], metadata["original_pair_sha256"], native_suffix=suffix)
            _, info = _freeze_row(row, suffix)
            expected, _ = _metadata(tokens, info, suffix, shard["index"], row_number, offsets[row_number])
            _require(metadata == expected, "Row source/token/mask/suffix/count/hash/index metadata differs")
            _require(_fingerprint(os.fstat(fd)) == fingerprints[names["tokens"]], "Token file changed while reading rows")
            yield {"id": metadata["id"], "input_ids": tuple(tokens), "prompt_tokens": prompts[row_number], "metadata": dict(metadata)}
        _require(not os.read(fd, 1), "Extra token bytes after final offset")
    finally:
        os.close(fd)
    for name, expected in fingerprints.items():
        actual_fd = directory.open(name)
        try:
            _require(_fingerprint(os.fstat(actual_fd)) == expected, "Shard changed during verified stream")
        finally:
            os.close(actual_fd)


class VerifiedCache:
    """Construct only with open_cache; all parts verify before ordered iteration.

    One iteration starts at row0; no seek/resume/selection interface exists.
    Close (or context exit) rechecks the complete pinned inventory. A caller
    must not promote partial consumption or use this class as source admission.
    """
    def __init__(self, directory, manifest, files, fingerprints, ids, suffix, manifest_sha256):
        self._directory, self._manifest, self._files = directory, manifest, files
        self._fingerprints, self._ids, self._suffix = fingerprints, ids, suffix
        self.manifest_file_sha256 = manifest_sha256
        self._used = self._closed = False
        self._iterator, self.rows_yielded = None, 0

    @property
    def manifest(self):
        return json.loads(canonical_value_bytes(self._manifest))

    def _unchanged(self):
        _require(not self._closed, "Closed cache reader cannot be reused")
        _unchanged(self._directory, self._fingerprints, self._files)

    def iter_rows(self):
        _require(not self._used, "Reader is one-pass; no resume/reuse")
        self._unchanged()
        self._used = True
        self._iterator = self._rows()
        return self._iterator

    def _rows(self):
        for shard in self._manifest["shards"]:
            start = shard["index"] * SHARD_ROWS
            iterator = _verified_shard(self._directory, shard, self._suffix, self._ids[start:start + shard["rows"]])
            try:
                for row in iterator:
                    self.rows_yielded += 1
                    yield row
            finally:
                iterator.close()
        self._unchanged()

    def close(self):
        if self._closed:
            return
        try:
            if self._iterator is not None:
                self._iterator.close()
            self._unchanged()
        finally:
            self._directory.close()
            self._closed = True

    def __enter__(self):
        self._unchanged()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException:
                pass
        return False


def open_cache(path, *, expected_manifest_sha256, expected_ids, expected_identity, native_suffix, fixture_only=False):
    """Validate ALL physical parts/rows before returning a one-pass reader."""
    ids, suffix = _ids(expected_ids, fixture_only), _suffix(native_suffix)
    identity = validate_identity(expected_identity, expected_ids=ids, native_suffix=suffix, fixture_only=fixture_only)
    directory = _Directory(path)
    try:
        manifest, files = _manifest(directory, expected_manifest_sha256, ids, identity, suffix, fixture_only)
        expected_files = {name: record for shard in manifest["shards"] for name, record in shard["files"].items()}
        fingerprints = {}
        for name in files:
            if name == "manifest.json":
                actual, fingerprints[name] = _file_record(directory, name, maximum=MAX_JSON_BYTES)
                _require(actual["sha256"] == expected_manifest_sha256, "Manifest changed after parsing")
            else:
                actual, fingerprints[name] = _file_record(directory, name, expected_bytes=expected_files[name]["bytes"])
                _require(actual == expected_files[name], "Shard changed after manifest parsing")
        totals = {"total_tokens": 0, "active_tokens": 0, "eos_tokens": 0, "tail_tokens": 0}
        for shard in manifest["shards"]:
            start = shard["index"] * SHARD_ROWS
            for row in _verified_shard(directory, shard, suffix, ids[start:start + shard["rows"]]):
                for key in totals:
                    totals[key] += row["metadata"][key]
        _require(all(totals[key] == manifest[key] for key in totals), "Complete cache aggregate counts differ")
        _unchanged(directory, fingerprints, files)
        return VerifiedCache(directory, manifest, files, fingerprints, ids, suffix, expected_manifest_sha256)
    except BaseException:
        directory.close()
        raise
