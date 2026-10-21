"""Pure preparation of unchanged, complete canonical source pairs.

No files, tokenizers, models, scores or selectors are loaded here. Callers must
verify the original source and tokenizer bytes on the intended native compute
allocation before supplying them. These helpers bind caller-supplied identities;
an in-memory object cannot establish its own origin.

Use ``prepare_pool`` BEFORE any pool scoring or ID selection. Its exclusions
must be the union of separately verified reserved-source-evaluation and final-
benchmark lexical audits of this exact population. ``serialize_selected`` repeats
that gate on the full population and rejects, rather than drops or replaces, any
requested excluded ID. Canonical pool prompt IDs are not independent labeled
record IDs. External data needs its own lexical audit; matching these IDs is not
proof of semantic decontamination or complete benchmark coverage.

Serialization follows reasoning.features.token_ids: original system/user text,
the native assistant prefix, and original assistant text. Unlike the historical
random-arm exporter, every target must fit its explicit context envelope with
the verified native EOS and following template tail intact. No truncation,
rewriting, EOS append, refill, causal shift or experiment recipe occurs here.
The returned labels are unshifted; the preserving trainer shifts exactly once.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from reasoning.data import prompt_id


POOL_ID_NAMESPACE = "canonical_pool_prompt_sha256_v1"
EXCLUSION_SCHEMA = "preserving-pool-exclusions-v1"
CONTEXT_ROLES = frozenset({"diagnostic", "pilot", "full"})
_ROW_KEYS = frozenset({"id", "instruction", "input", "output", "system"})
_SCOPES = {"reserved_source_evaluation": "all_reserved_source_splits",
           "final_benchmark_questions": "all_nine_final_benchmarks"}
_SHA = re.compile(r"[0-9a-f]{64}")


class PreservationDataError(ValueError):
    """Invalid supplied source, exclusion binding or native serialization."""


class CompleteContextError(PreservationDataError):
    """A complete original pair cannot fit; ``failure`` is an auditable receipt."""

    def __init__(self, failure):
        self.failure = _copy_json(failure)
        super().__init__("Complete original pair exceeds its declared context: " +
                         failure["row_id"])


def _require(condition, message):
    if not condition:
        raise PreservationDataError(message)


def _json_value(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_json_value(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _json_value(item) for key, item in value.items())
    return False


def _canonical(value):
    _require(_json_value(value), "Only finite plain JSON values may enter provenance")
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def value_sha256(value):
    """Stable canonical-JSON identity, distinct from a file-byte SHA256."""
    return hashlib.sha256(_canonical(value)).hexdigest()


def text_sha256(text):
    _require(type(text) is str, "Text hashing requires an unchanged string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _copy_json(value):
    return json.loads(_canonical(value))


def _sha(value):
    _require(type(value) is str and _SHA.fullmatch(value), "Expected a lowercase SHA256 identity")
    return value


def _identity(value, name):
    _require(type(value) is dict and bool(value), "Supply a nonempty caller-verified " + name)
    return _copy_json(value)


def _population(rows):
    _require(type(rows) in (list, tuple) and bool(rows), "Supply the full nonempty canonical pool")
    copied, by_id = [], {}
    for row in rows:
        _require(type(row) is dict and set(row) == _ROW_KEYS,
                 "Original canonical pool rows must have exactly id/instruction/input/output/system")
        _sha(row["id"])
        _require(all(type(row[key]) is str for key in ("instruction", "input", "output", "system")) and
                 all(row[key].strip() for key in ("instruction", "output", "system")) and row["input"] == "",
                 "Preserve nonempty canonical text and the original empty input; do not recanonicalize")
        _require(row["id"] == prompt_id(row["instruction"]),
                 "Pool IDs must be canonical prompt IDs, not independent labeled record IDs")
        _require(row["id"] not in by_id, "Duplicate canonical pool prompt ID")
        unchanged = dict(row)
        copied.append(unchanged)
        by_id[row["id"]] = unchanged
    return copied, by_id


def _exclusions(by_id, excluded_pool_ids, receipt):
    _require(type(excluded_pool_ids) in (set, frozenset), "Supply the explicit union set of excluded pool IDs")
    _require(all(type(identifier) is str and _SHA.fullmatch(identifier) for identifier in excluded_pool_ids),
             "Excluded IDs must use the canonical pool prompt namespace")
    _require(excluded_pool_ids <= set(by_id),
             "Every excluded ID must resolve in the full pool; labeled record IDs are a different namespace")
    _require(type(receipt) is dict and set(receipt) ==
             {"schema", "id_namespace", "population_ids_sha256", "components"} and
             receipt["schema"] == EXCLUSION_SCHEMA and receipt["id_namespace"] == POOL_ID_NAMESPACE and
             receipt["population_ids_sha256"] == value_sha256(sorted(by_id)),
             "Exclusion receipt must bind this full canonical pool and its namespace")
    components = receipt["components"]
    _require(type(components) is dict and set(components) == set(_SCOPES),
             "Bind both reserved-source and all-nine-benchmark lexical audit components")
    union, criteria = set(), set()
    for purpose, scope in _SCOPES.items():
        component = components[purpose]
        _require(type(component) is dict and set(component) ==
                 {"scope", "pool_ids", "audit_receipt_sha256", "criterion_sha256", "question_inventory_sha256"} and
                 component["scope"] == scope, "Unexpected or incomplete exclusion audit scope")
        for key in ("audit_receipt_sha256", "criterion_sha256", "question_inventory_sha256"):
            _sha(component[key])
        identifiers = component["pool_ids"]
        _require(type(identifiers) is list and
                 all(type(identifier) is str and _SHA.fullmatch(identifier) for identifier in identifiers) and
                 identifiers == sorted(set(identifiers)) and set(identifiers) <= set(by_id),
                 "Audit component IDs must be sorted, unique, and resolved canonical pool IDs")
        union.update(identifiers)
        criteria.add(component["criterion_sha256"])
    _require(len(criteria) == 1 and union == excluded_pool_ids,
             "Both audits must bind the same lexical criterion and exactly the supplied union")
    return _copy_json(receipt)


def prepare_pool(rows, *, excluded_pool_ids, exclusion_receipt, source_identity):
    """Apply the supplied audit union to the FULL pool before scoring/selection.

    The returned rows retain input order and exact strings. Exclusion receipts
    are structurally checked and hashed; their external audit bytes and claims
    still require independent caller verification. No hidden ID draw occurs.
    """
    source = _identity(source_identity, "source_identity")
    copied, by_id = _population(rows)
    audit = _exclusions(by_id, excluded_pool_ids, exclusion_receipt)
    eligible = [row for row in copied if row["id"] not in excluded_pool_ids]
    _require(bool(eligible), "Exclusion union leaves no eligible original pairs")
    metadata = {"schema": "preserving-eligible-pool-v1", "id_namespace": POOL_ID_NAMESPACE,
        "input_rows": len(copied), "eligible_rows": len(eligible), "excluded_rows": len(excluded_pool_ids),
        "input_ordered_ids_sha256": value_sha256([row["id"] for row in copied]),
        "input_rows_sha256": value_sha256([value_sha256(row) for row in copied]),
        "eligible_ordered_ids_sha256": value_sha256([row["id"] for row in eligible]),
        "eligible_rows_sha256": value_sha256([value_sha256(row) for row in eligible]),
        "excluded_ids_sha256": value_sha256(sorted(excluded_pool_ids)),
        "source_identity": source, "source_identity_sha256": value_sha256(source),
        "exclusion_receipt": audit, "exclusion_receipt_sha256": value_sha256(audit),
        "audit_contents_independently_verified_here": False,
        "semantic_decontamination_claimed": False, "selection_or_scoring_performed": False}
    return {"rows": eligible, "provenance": metadata}


def _tokens(value, vocab_size, name):
    _require(type(value) is list and bool(value) and
             all(type(token) is int and 0 <= token < vocab_size for token in value),
             name + " must be a nonempty native list of in-vocabulary integer IDs")
    return list(value)


def _tokenizer_contract(tokenizer, identity, eos_token_id, terminal_suffix, terminal_text, vocab_size):
    identity = _identity(identity, "tokenizer_identity")
    _require(type(vocab_size) is int and vocab_size > 0 and type(eos_token_id) is int and
             0 <= eos_token_id < vocab_size, "Supply verified model vocabulary size and native EOS")
    _require(type(terminal_suffix) is tuple and bool(terminal_suffix) and terminal_suffix[0] == eos_token_id and
             terminal_suffix.count(eos_token_id) == 1 and
             all(type(token) is int and 0 <= token < vocab_size for token in terminal_suffix) and
             type(terminal_text) is str and bool(terminal_text), "Supply exact native EOS-plus-tail tokens and text")
    template = getattr(tokenizer, "chat_template", None)
    _require(type(template) is str and bool(template) and
             identity.get("chat_template_sha256") == text_sha256(template) and
             type(identity.get("eos_token_id")) is int and identity["eos_token_id"] == eos_token_id and
             type(getattr(tokenizer, "eos_token_id", None)) is int and tokenizer.eos_token_id == eos_token_id,
             "Caller identity, actual native template and EOS differ")
    files = identity.get("files")
    _require(type(files) is dict and bool(files) and all(type(name) is str and bool(name) and
             type(checksum) is str and _SHA.fullmatch(checksum) for name, checksum in files.items()),
             "Bind the caller-verified tokenizer file SHA inventory")
    _require(callable(getattr(tokenizer, "apply_chat_template", None)) and
             callable(getattr(tokenizer, "encode", None)), "A supplied native tokenizer object is required")
    encoded_tail = _tokens(tokenizer.encode(terminal_text, add_special_tokens=False), vocab_size, "Native tail")
    _require(encoded_tail == list(terminal_suffix), "Native terminal text and token suffix disagree")
    return identity


def _serialize(row, tokenizer, *, cutoff, context_role, eos_token_id, terminal_suffix,
               terminal_text, vocab_size, source_sha256, tokenizer_sha256, exclusions_sha256):
    prompt_messages = [{"role": "system", "content": row["system"]},
                       {"role": "user", "content": row["instruction"]}]
    full_messages = prompt_messages + [{"role": "assistant", "content": row["output"]}]
    prompt = _tokens(tokenizer.apply_chat_template(prompt_messages, tokenize=True, return_dict=False,
        add_generation_prompt=True), vocab_size, "Native prompt")
    full = _tokens(tokenizer.apply_chat_template(full_messages, tokenize=True, return_dict=False,
        add_generation_prompt=False), vocab_size, "Native full pair")
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    _require(type(prompt_text) is str and type(full_text) is str and
             row["system"] in prompt_text and row["instruction"] in prompt_text and
             full_text == prompt_text + row["output"] + terminal_text,
             "Native rendered text must preserve the complete original pair and exact terminal text")
    _require(prompt == _tokens(tokenizer.encode(prompt_text, add_special_tokens=False), vocab_size, "Rendered prompt") and
             full == _tokens(tokenizer.encode(full_text, add_special_tokens=False), vocab_size, "Rendered full pair"),
             "Native template token IDs disagree with its exact rendered text")
    _require(len(full) > len(prompt) and full[:len(prompt)] == prompt,
             "Native prompt is not an exact token prefix; response masking would be invalid")
    response = full[len(prompt):]
    _require(len(response) > len(terminal_suffix) and full[-len(terminal_suffix):] == list(terminal_suffix),
             "Complete original response lacks its actual native EOS-plus-tail; never append or repair it")
    labels = [-100] * len(prompt) + response
    provenance = {"id_namespace": POOL_ID_NAMESPACE, "row_id": row["id"],
        "row_sha256": value_sha256(row), "pair_sha256": value_sha256(full_messages),
        "prompt_messages_sha256": value_sha256(prompt_messages),
        "system_text_sha256": text_sha256(row["system"]), "user_text_sha256": text_sha256(row["instruction"]),
        "input_text_sha256": text_sha256(row["input"]), "output_text_sha256": text_sha256(row["output"]),
        "rendered_prompt_sha256": text_sha256(prompt_text), "rendered_full_sha256": text_sha256(full_text),
        "prompt_token_ids_sha256": value_sha256(prompt), "full_token_ids_sha256": value_sha256(full),
        "labels_sha256": value_sha256(labels), "total_tokens": len(full), "prompt_tokens": len(prompt),
        "active_tokens": len(response), "active_eos_tokens": response.count(eos_token_id),
        "eos_token_id": eos_token_id, "native_terminal_suffix": list(terminal_suffix),
        "native_terminal_text_sha256": text_sha256(terminal_text), "cutoff": cutoff, "context_role": context_role,
        "source_identity_sha256": source_sha256, "tokenizer_identity_sha256": tokenizer_sha256,
        "exclusion_receipt_sha256": exclusions_sha256, "complete_original_target": True,
        "truncated": False, "target_rewritten": False, "causal_shift_applied": False}
    if len(full) > cutoff:
        raise CompleteContextError({"schema": "preserving-complete-context-failure-v1",
            "reason": "prompt_has_no_response_capacity" if len(prompt) >= cutoff else "complete_pair_exceeds_cutoff",
            **provenance})
    return {"id": row["id"], "input_ids": full, "labels": labels,
        "attention_mask": [1] * len(full), "original_row": dict(row), "provenance": provenance}


def serialize_selected(rows, ordered_ids, tokenizer, *, excluded_pool_ids, exclusion_receipt,
                       source_identity, tokenizer_identity, cutoff, context_role,
                       eos_token_id, terminal_suffix, terminal_text, vocab_size):
    """Serialize explicitly supplied IDs or fail the entire call, without I/O.

    ``rows`` is the full original population, not a prefiltered subset. The
    repeated union gate occurs before tokenization. Returned records are plain
    unshifted examples; callers must choose model batch keys and atomically bind
    any later cache files themselves. A failure returns no partial cohort.
    Diagnostic, pilot and full envelopes are explicitly distinct provenance;
    success at a shorter envelope does not admit a longer training workload.
    """
    _require(type(cutoff) is int and cutoff >= 2 and type(context_role) is str and context_role in CONTEXT_ROLES,
             "Declare a positive complete-pair cutoff and diagnostic/pilot/full envelope")
    prepared = prepare_pool(rows, excluded_pool_ids=excluded_pool_ids,
        exclusion_receipt=exclusion_receipt, source_identity=source_identity)
    _require(type(ordered_ids) in (list, tuple) and bool(ordered_ids) and
             all(type(identifier) is str and _SHA.fullmatch(identifier) for identifier in ordered_ids) and
             len(set(ordered_ids)) == len(ordered_ids), "Supply explicit nonempty unique ordered canonical pool IDs")
    _require(not set(ordered_ids).intersection(excluded_pool_ids),
             "Requested reserved or benchmark-overlapping ID; do not drop, replace, score or tokenize it")
    by_id = {row["id"]: row for row in prepared["rows"]}
    _require(set(ordered_ids) <= set(by_id), "A requested ID is missing from the eligible original pool")
    identity = _tokenizer_contract(tokenizer, tokenizer_identity, eos_token_id,
        terminal_suffix, terminal_text, vocab_size)
    pool_proof = prepared["provenance"]
    examples = [_serialize(by_id[identifier], tokenizer, cutoff=cutoff, context_role=context_role,
        eos_token_id=eos_token_id, terminal_suffix=terminal_suffix, terminal_text=terminal_text, vocab_size=vocab_size,
        source_sha256=pool_proof["source_identity_sha256"], tokenizer_sha256=value_sha256(identity),
        exclusions_sha256=pool_proof["exclusion_receipt_sha256"]) for identifier in ordered_ids]
    metadata = {"schema": "preserving-complete-pairs-v1", "id_namespace": POOL_ID_NAMESPACE,
        "rows": len(examples), "ordered_ids_sha256": value_sha256(list(ordered_ids)),
        "ordered_pair_sha256": value_sha256([example["provenance"]["pair_sha256"] for example in examples]),
        "ordered_token_sha256": value_sha256([example["provenance"]["full_token_ids_sha256"] for example in examples]),
        "total_tokens": sum(example["provenance"]["total_tokens"] for example in examples),
        "active_tokens": sum(example["provenance"]["active_tokens"] for example in examples),
        "cutoff": cutoff, "context_role": context_role, "vocab_size": vocab_size,
        "eos_token_id": eos_token_id, "native_terminal_suffix": list(terminal_suffix),
        "native_terminal_text_sha256": text_sha256(terminal_text), "pool_provenance": pool_proof,
        "tokenizer_identity": identity, "tokenizer_identity_sha256": value_sha256(identity),
        "tokenizer_files_independently_verified_here": False,
        "truncated_rows": 0, "rewritten_rows": 0, "requested_rows_dropped": 0,
        "replacement_rows_added": 0, "complete_original_targets": True,
        "model_memory_admission_established": False}
    return {"examples": examples, "provenance": metadata}


def verify_serialized_examples(examples, tokenizer, *, source_identity, tokenizer_identity,
                               exclusion_receipt_sha256, cutoff, context_role,
                               eos_token_id, terminal_suffix, terminal_text, vocab_size):
    """Replay at most four already-bound private cache witnesses, without I/O.

    This callable API does not make witness rows public. It reserializes exact
    supplied originals and compares the entire prepared example, including all
    tokens, masks and hashes. The caller must first verify the private CPU cache
    receipt, selection and full-pool lexical/exclusion evidence. Only the supplied
    exclusion receipt identity is bound here; no pool scan, row choice, semantic
    claim, independent file verification or model admission is performed.
    """
    _require(type(examples) in (list, tuple) and 1 <= len(examples) <= 4 and
             all(type(example) is dict and set(example) ==
                 {"id", "input_ids", "labels", "attention_mask", "original_row", "provenance"}
                 for example in examples), "Replay only one to four exact prepared private examples")
    _require(type(cutoff) is int and cutoff >= 2 and type(context_role) is str and context_role in CONTEXT_ROLES,
             "Declare the exact witness context envelope")
    _sha(exclusion_receipt_sha256)
    source = _identity(source_identity, "source_identity")
    rows, _ = _population([example["original_row"] for example in examples])
    identity = _tokenizer_contract(tokenizer, tokenizer_identity, eos_token_id,
        terminal_suffix, terminal_text, vocab_size)
    for example, row in zip(examples, rows, strict=True):
        expected = _serialize(row, tokenizer, cutoff=cutoff, context_role=context_role,
            eos_token_id=eos_token_id, terminal_suffix=terminal_suffix, terminal_text=terminal_text,
            vocab_size=vocab_size, source_sha256=value_sha256(source), tokenizer_sha256=value_sha256(identity),
            exclusions_sha256=exclusion_receipt_sha256)
        # JSON identities reject tensor coercions, tuple/list substitutions and
        # bool/int equivalence that ordinary Python dictionary equality accepts.
        _require(_canonical(example) == _canonical(expected),
                 "Prepared witness differs from exact original/native-token replay: " + row["id"])
    return {"schema": "preserving-native-witness-replay-v1", "examples_verified": len(examples),
        "ordered_ids_sha256": value_sha256([row["id"] for row in rows]),
        "examples_sha256": value_sha256(list(examples)), "source_identity_sha256": value_sha256(source),
        "tokenizer_identity_sha256": value_sha256(identity),
        "exclusion_receipt_sha256": exclusion_receipt_sha256, "cutoff": cutoff, "context_role": context_role,
        "full_pool_audit_reestablished": False, "selection_performed": False,
        "raw_witnesses_remain_private": True, "model_memory_admission_established": False}
