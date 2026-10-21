"""Prospective independent synthetic binary-cache fixtures; authoring is not execution.

Every row/ID/identity is fabricated, with explicit fixture_only=True. Files live
only in owner-only TemporaryDirectory children on a future admitted CPU worker.
No corpus, tokenizer, model, topic fit, real draw or scheduler admission occurs.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import tempfile
import unittest
from unittest import mock

from extension import full_pool_cache as cache


def payload(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(payload(value)).hexdigest()


def filepin(path):
    b = path.read_bytes()
    return {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def pin(label):
    return hashlib.sha256(label.encode()).hexdigest()


def identifier(number):
    return format(number, "064x")


SUFFIX = [151645, 198]
FIXED = {
    "protocol_file_sha256": "160948a0477c6f234a587ac2cc32297f5dc229ba6c0e3a83cea89242fd3adc29",
    "root_approval_file_sha256": "785b445913cc7c9c76c56474db4c89eb01a39c65bfbb50d1e8f87f7a6d0bfdc6",
    "base_identity_file_sha256": "81af59dbb365b8559ad0d853dbda245aa4ac3ff0bedc03428e50fed65d7f7ae2",
    "pool_file_sha256": "4e8ac972757555c2c0c1c0d31f87f49fc79d19815fb89b49bb748cee3ea145b0",
    "raw_source_file_sha256": "f064d1a5fb39b9f493f678d5d6350a9acbfb2a2fa85400d0947fe72578ff28d8",
    "exclusion_receipt_file_sha256": "0066e69a996cf963173018d68196d7ff69e658420c40b250941c2fe8bb6da406"}


def own_row(number, tokens=None, prompt=2):
    tokens = [700 + number, 701, 99 + number, *SUFFIX] if tokens is None else list(tokens)
    return {"id": identifier(number), "input_ids": tokens, "prompt_tokens": prompt,
        "original_pair_sha256": pin("synthetic-original:" + str(number)),
        "prompt_token_ids_sha256": digest(tokens[:prompt]), "full_token_ids_sha256": digest(tokens),
        "labels_sha256": digest([-100] * prompt + tokens[prompt:]),
        "mask_sha256": digest([0] * prompt + [1] * (len(tokens) - prompt)),
        "native_suffix_sha256": digest(SUFFIX)}


def own_identity(ids):
    return {**FIXED, "preparation_plan_file_sha256": pin("fixture-only-plan"),
        "source_identity_sha256": pin("fixture-only-source"),
        "tokenizer_identity_sha256": pin("fixture-only-tokenizer"),
        "exclusion_receipt_value_sha256": pin("fixture-only-exclusion-value"),
        "eligible_sorted_ids_value_sha256": digest(ids), "native_suffix_value_sha256": digest(SUFFIX)}


class FullPoolCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.root.chmod(0o700)
        self.serial = 0

    def case(self, count=3, *, rows=None):
        rows = [own_row(i + 1) for i in range(count)] if rows is None else rows
        ids = [row["id"] for row in rows]
        self.serial += 1
        output = self.root / ("cache-" + str(self.serial))
        identity = own_identity(ids)
        result = cache.write_cache(output, iter(rows), expected_ids=ids,
            identity=identity, native_suffix=SUFFIX, fixture_only=True)
        return output, ids, identity, result, rows

    def reader(self, case, **changes):
        output, ids, identity, receipt, _ = case
        args = dict(expected_manifest_sha256=receipt["manifest_file_sha256"],
            expected_ids=ids, expected_identity=identity, native_suffix=SUFFIX, fixture_only=True)
        args.update(changes)
        return cache.open_cache(output, **args)

    def manifest(self, case):
        return json.loads((case[0] / "manifest.json").read_bytes())

    def replace_manifest(self, case, value):
        data = payload(value) + b"\n"
        (case[0] / "manifest.json").write_bytes(data)
        case[3]["manifest_file_sha256"] = hashlib.sha256(data).hexdigest()

    def rebound_part(self, case, name, data, *, index=0):
        (case[0] / name).write_bytes(data)
        value = self.manifest(case)
        value["shards"][index]["files"][name] = filepin(case[0] / name)
        self.replace_manifest(case, value)

    def metadata(self, case):
        with self.reader(case) as reader:
            return list(reader.iter_rows())[0]["metadata"]

    def test_01_unshifted_labels_match_independent_native_target(self):
        self.assertEqual(cache.derive_labels([42, 43, 44, 151645, 198], 2), [-100, -100, 44, 151645, 198])
        self.assertEqual(cache.derive_labels([42, 151645, 198], 1), [-100, 151645, 198])

    def test_02_row_hashes_match_independent_canonical_oracle(self):
        expected = own_row(1)
        actual = cache.cache_row(expected["id"], expected["input_ids"], 2,
            expected["original_pair_sha256"], native_suffix=SUFFIX)
        self.assertEqual(actual, expected)
        self.assertNotEqual(actual["full_token_ids_sha256"], hashlib.sha256(payload(expected["input_ids"]) + b"\n").hexdigest())

    def test_03_roundtrip_permissions_private_scope_and_complete_hashes(self):
        case = self.case()
        self.assertEqual(stat.S_IMODE(case[0].stat().st_mode), 0o700)
        self.assertTrue(all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in case[0].iterdir()))
        self.assertEqual(case[3]["manifest_file_sha256"], filepin(case[0] / "manifest.json")["sha256"])
        self.assertNotEqual(case[3]["manifest_file_sha256"], case[3]["manifest_value_sha256"])
        self.assertIs(case[3]["fixture_only"], True)
        self.assertIs(case[3]["native_population_provenance_or_compute_admitted"], False)
        with self.reader(case) as reader:
            rows = list(reader.iter_rows())
            self.assertEqual(reader.rows_yielded, 3)
            self.assertEqual([r["id"] for r in rows], case[1])
            self.assertEqual([list(r["input_ids"]) for r in rows], [r["input_ids"] for r in case[4]])
            self.assertEqual([r["prompt_tokens"] for r in rows], [2, 2, 2])
            self.assertEqual(reader.manifest["total_tokens"], 15)
            self.assertEqual(reader.manifest["active_tokens"], 9)
            self.assertEqual(reader.manifest["eos_tokens"], 3)
            self.assertEqual(reader.manifest["tail_tokens"], 3)

    def test_04_raw_parts_are_little_endian_and_exact_offsets(self):
        case = self.case(2)
        self.assertEqual((case[0] / "shard-000000.tokens.u32le").read_bytes(),
            struct.pack("<10I", *(case[4][0]["input_ids"] + case[4][1]["input_ids"])))
        self.assertEqual((case[0] / "shard-000000.offsets.u64le").read_bytes(), struct.pack("<3Q", 0, 5, 10))
        self.assertEqual((case[0] / "shard-000000.prompts.u32le").read_bytes(), struct.pack("<2I", 2, 2))
        self.assertEqual(set(self.manifest(case)["format"]), set(cache.FORMAT))
        self.assertFalse(any("label" in p.name or "text" in p.name for p in case[0].iterdir()))

    def test_05_fixed_256_row_sharding_restarts_offsets_and_preserves_all_rows(self):
        case = self.case(257)
        value = self.manifest(case)
        self.assertEqual([s["rows"] for s in value["shards"]], [256, 1])
        self.assertEqual([s["index"] for s in value["shards"]], [0, 1])
        self.assertEqual((case[0] / "shard-000001.offsets.u64le").read_bytes(), struct.pack("<2Q", 0, 5))
        with self.reader(case) as reader:
            rows = list(reader.iter_rows())
            self.assertEqual(len(rows), 257)
            self.assertEqual(rows[-1]["metadata"]["row_in_shard"], 0)
            self.assertEqual(rows[-1]["metadata"]["shard_index"], 1)
            self.assertEqual(rows[-1]["metadata"]["token_offset"], 0)

    def test_06_maximum_complete_row_preserves_32_unused_context_tokens(self):
        row = own_row(1, tokens=[7] * 32734 + SUFFIX, prompt=11)
        case = self.case(rows=[row])
        with self.reader(case) as reader:
            actual = list(reader.iter_rows())[0]
            self.assertEqual(actual["metadata"]["total_tokens"], 32736)
            self.assertEqual(actual["metadata"]["active_tokens"], 32725)
            self.assertEqual(filepin(case[0] / "shard-000000.tokens.u32le")["bytes"], 130944)
            self.assertEqual(actual["input_ids"][-2:], tuple(SUFFIX))

    def test_07_overlength_is_rejected_without_truncation_or_terminal_marker(self):
        row = own_row(1, tokens=[7] * 32735 + SUFFIX)
        output, ids = self.root / "overlong", [identifier(1)]
        with self.assertRaises(ValueError):
            cache.write_cache(output, [row], expected_ids=ids, identity=own_identity(ids), native_suffix=SUFFIX, fixture_only=True)
        self.assertFalse((output / "manifest.json").exists())

    def test_08_invalid_vocabulary_and_bool_or_float_tokens_rejected(self):
        for token in (-1, 151936, True, 1.0):
            with self.subTest(token=token), self.assertRaises(ValueError):
                cache.cache_row(identifier(1), [10, token, *SUFFIX], 1, pin("pair"), native_suffix=SUFFIX)

    def test_09_prompt_and_active_bounds_are_exact_plain_integers(self):
        for prompt in (0, 5, -1, True, 2.0):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                cache.derive_labels([1, 2, 3, *SUFFIX], prompt)

    def test_10_supplied_native_suffix_is_exact_and_never_appended(self):
        for suffix in ([], [198], [151645, 151645], [151645, True], [151645] * 33):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                cache.cache_row(identifier(1), [1, 2, *SUFFIX], 1, pin("pair"), native_suffix=suffix)
        with self.assertRaises(ValueError):
            cache.cache_row(identifier(1), [1, 2, 151645], 1, pin("pair"), native_suffix=SUFFIX)

    def test_11_input_hash_schema_tampering_stops_before_completion(self):
        for key in ("prompt_token_ids_sha256", "full_token_ids_sha256", "labels_sha256", "mask_sha256", "native_suffix_sha256"):
            row = own_row(1); row[key] = pin("wrong")
            output, ids = self.root / key, [identifier(1)]
            with self.subTest(key=key), self.assertRaises(ValueError):
                cache.write_cache(output, [row], expected_ids=ids, identity=own_identity(ids), native_suffix=SUFFIX, fixture_only=True)
            self.assertFalse((output / "manifest.json").exists())

    def test_12_duplicate_and_wrong_row_order_never_refilled(self):
        ids = [identifier(1), identifier(2)]
        for index, rows in enumerate(([own_row(2), own_row(1)], [own_row(1), own_row(1)])):
            output = self.root / ("order" + str(index))
            with self.assertRaises(ValueError):
                cache.write_cache(output, rows, expected_ids=ids, identity=own_identity(ids), native_suffix=SUFFIX, fixture_only=True)
            self.assertFalse((output / "manifest.json").exists())

    def test_13_missing_extra_or_generator_failure_retains_uncommitted_partials(self):
        ids = [identifier(1), identifier(2)]
        def broken():
            yield own_row(1)
            raise RuntimeError("synthetic generator failure")
        for index, rows, error in ((0, [own_row(1)], ValueError), (1, [own_row(1), own_row(2), own_row(3)], ValueError), (2, broken(), RuntimeError)):
            output = self.root / ("coverage" + str(index))
            with self.assertRaises(error):
                cache.write_cache(output, rows, expected_ids=ids, identity=own_identity(ids), native_suffix=SUFFIX, fixture_only=True)
            self.assertTrue(output.is_dir())
            self.assertFalse((output / "manifest.json").exists())

    def test_14_existing_success_and_partial_outputs_never_reused(self):
        case = self.case()
        for output in (case[0], self.root / "partial"):
            if not output.exists(): output.mkdir(mode=0o700)
            with self.assertRaises(ValueError):
                cache.write_cache(output, case[4], expected_ids=case[1], identity=case[2], native_suffix=SUFFIX, fixture_only=True)

    def test_15_owner_permissions_are_checked_on_directories_and_files(self):
        case = self.case()
        (case[0] / "manifest.json").chmod(0o640)
        with self.assertRaises(ValueError): self.reader(case)
        (case[0] / "manifest.json").chmod(0o600)
        case[0].chmod(0o750)
        with self.assertRaises(ValueError): self.reader(case)
        case[0].chmod(0o700)
        parent = self.root / "nonprivate"; parent.mkdir(mode=0o750); parent.chmod(0o750)
        with self.assertRaises(ValueError):
            cache.write_cache(parent / "cache", case[4], expected_ids=case[1], identity=case[2], native_suffix=SUFFIX, fixture_only=True)

    def test_16_symlink_and_hardlink_aliases_rejected(self):
        case = self.case()
        alias = self.root / "alias"; alias.symlink_to(case[0], target_is_directory=True)
        with self.assertRaises(ValueError):
            cache.open_cache(alias, expected_manifest_sha256=case[3]["manifest_file_sha256"], expected_ids=case[1], expected_identity=case[2], native_suffix=SUFFIX, fixture_only=True)
        target = case[0] / "shard-000000.tokens.u32le"
        os.link(target, self.root / "hardlink")
        with self.assertRaises(ValueError): self.reader(case)

    def test_17_independent_manifest_file_pin_not_value_or_self_trust(self):
        case = self.case()
        for value in (case[3]["manifest_value_sha256"], pin("wrong"), None):
            with self.subTest(pin=value), self.assertRaises(ValueError):
                self.reader(case, expected_manifest_sha256=value)

    def test_18_unexpected_and_missing_parts_are_rejected_before_rows(self):
        case = self.case()
        extra = case[0] / "extra.json"; extra.write_bytes(b"{}\n"); extra.chmod(0o600)
        with self.assertRaises(ValueError): self.reader(case)
        extra.unlink(); (case[0] / "shard-000000.prompts.u32le").unlink()
        with self.assertRaises(ValueError): self.reader(case)

    def test_19_physical_token_corruption_is_detected_before_any_yield(self):
        case = self.case()
        name = "shard-000000.tokens.u32le"; path = case[0] / name
        data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
        with self.assertRaises(ValueError): self.reader(case)

    def test_20_rehashed_nonmonotonic_and_wrong_end_offsets_are_rejected(self):
        for offsets in ((0, 5, 5, 15), (0, 5, 10, 14), (1, 5, 10, 15)):
            case = self.case()
            self.rebound_part(case, "shard-000000.offsets.u64le", struct.pack("<4Q", *offsets))
            with self.subTest(offsets=offsets), self.assertRaises(ValueError): self.reader(case)

    def test_21_rehashed_invalid_prompt_binary_is_rejected(self):
        for prompt in (0, 5, 2**32 - 1):
            case = self.case()
            self.rebound_part(case, "shard-000000.prompts.u32le", struct.pack("<3I", prompt, 2, 2))
            with self.subTest(prompt=prompt), self.assertRaises(ValueError): self.reader(case)

    def test_22_rehashed_private_index_count_and_row_source_tampering_rejected(self):
        for field, value in (("active_tokens", True), ("active_tokens", 3.0), ("total_tokens", 5.0), ("id", identifier(999))):
            case = self.case()
            name = "shard-000000.index.json"; index = json.loads((case[0] / name).read_bytes())
            row = index["rows"][0]; row[field] = value
            # Keep record checksum consistent to exercise semantic validation.
            row["cache_record_sha256"] = digest({k: v for k, v in row.items() if k != "cache_record_sha256"})
            self.rebound_part(case, name, payload(index) + b"\n")
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): self.reader(case)

    def test_23_rebound_format_schema_and_alias_types_are_rejected(self):
        for change in (lambda m: m.update(schema="other"), lambda m: m["format"].update(labels_stored=True),
                       lambda m: m.update(rows=3.0), lambda m: m["shards"][0].update(index=False)):
            case = self.case(); m = self.manifest(case); change(m); self.replace_manifest(case, m)
            with self.assertRaises(ValueError): self.reader(case)

    def test_24_caller_source_tokenizer_and_sorted_expected_ids_bindings_required(self):
        case = self.case()
        identity = copy.deepcopy(case[2]); identity["tokenizer_identity_sha256"] = pin("other-tokenizer")
        with self.assertRaises(ValueError): self.reader(case, expected_identity=identity)
        with self.assertRaises(ValueError): self.reader(case, expected_ids=case[1][::-1])
        identity = copy.deepcopy(case[2]); identity["pool_file_sha256"] = pin("other-pool")
        with self.assertRaises(ValueError): self.reader(case, expected_identity=identity)

    def test_25_rebound_aggregate_counts_must_match_complete_binary_rows(self):
        for name in ("total_tokens", "active_tokens", "eos_tokens", "tail_tokens"):
            case = self.case(); m = self.manifest(case); m[name] += 1; self.replace_manifest(case, m)
            with self.subTest(field=name), self.assertRaises(ValueError): self.reader(case)

    def test_26_rehashed_out_of_vocabulary_and_suffix_token_bytes_rejected(self):
        for position, token in ((0, 151936), (4, 199)):
            case = self.case(); data = bytearray((case[0] / "shard-000000.tokens.u32le").read_bytes())
            data[position * 4:position * 4 + 4] = struct.pack("<I", token)
            self.rebound_part(case, "shard-000000.tokens.u32le", bytes(data))
            with self.assertRaises(ValueError): self.reader(case)

    def test_27_one_pass_and_closed_reader_cannot_be_reused(self):
        case = self.case()
        with self.reader(case) as reader:
            first = reader.iter_rows(); self.assertEqual(len(list(first)), 3)
            with self.assertRaises(ValueError): reader.iter_rows()
            changed = reader.manifest; changed["rows"] = 0
            self.assertEqual(reader.manifest["rows"], 3)
        with self.assertRaises(ValueError): reader.iter_rows()
        reader.close()  # Idempotent close is not a second consumption.

    def test_28_midstream_and_close_time_mutations_abort(self):
        case = self.case()
        reader = self.reader(case); stream = reader.iter_rows(); next(stream)
        path = case[0] / "shard-000000.tokens.u32le"; data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        with self.assertRaises(ValueError): list(stream)
        with self.assertRaises(ValueError): reader.close()
        self.assertTrue(reader._closed)

    def test_29_bounded_write_fault_preserves_partial_and_closes_descriptors(self):
        ids = [identifier(1)]; output = self.root / "write-failed"
        opened, closed = [], []
        real_open, real_close = os.open, os.close
        def track_open(*args, **kwargs):
            fd = real_open(*args, **kwargs); opened.append(fd); return fd
        def track_close(fd):
            closed.append(fd); return real_close(fd)
        with mock.patch.object(cache.os, "open", side_effect=track_open), mock.patch.object(cache.os, "close", side_effect=track_close), mock.patch.object(cache, "_write", side_effect=OSError("synthetic full disk")):
            with self.assertRaises(OSError):
                cache.write_cache(output, [own_row(1)], expected_ids=ids, identity=own_identity(ids), native_suffix=SUFFIX, fixture_only=True)
        self.assertTrue(output.is_dir()); self.assertFalse((output / "manifest.json").exists())
        self.assertEqual(sorted(opened), sorted(closed))
        with self.assertRaises(ValueError):
            cache.write_cache(output, [own_row(1)], expected_ids=ids, identity=own_identity(ids), native_suffix=SUFFIX, fixture_only=True)

    def test_30_atomic_manifest_commit_failure_leaves_no_success_marker(self):
        ids = [identifier(1)]; output = self.root / "link-failed"
        with mock.patch.object(cache.os, "link", side_effect=OSError("synthetic commit failure")):
            with self.assertRaises(OSError):
                cache.write_cache(output, [own_row(1)], expected_ids=ids, identity=own_identity(ids), native_suffix=SUFFIX, fixture_only=True)
        self.assertTrue(output.is_dir()); self.assertFalse((output / "manifest.json").exists())
        self.assertTrue((output / ".manifest.pending").exists())

    def test_31_selector_bridge_matches_four_hash_bound_fields_and_strict_types(self):
        case = self.case(); row = self.metadata(case)
        expected = {k: row[k] for k in ("total_tokens", "active_tokens", "original_pair_sha256", "cache_record_sha256")}
        self.assertEqual(cache.selection_row(row), expected)
        for field, value in (("active_tokens", 3.0), ("prompt_tokens", True), ("eos_tokens", 1.0), ("row_in_shard", False)):
            changed = copy.deepcopy(row); changed[field] = value
            changed["cache_record_sha256"] = digest({k: v for k, v in changed.items() if k != "cache_record_sha256"})
            with self.subTest(field=field), self.assertRaises(ValueError): cache.selection_row(changed)
        changed = copy.deepcopy(row); changed["original_pair_sha256"] = pin("wrong")
        with self.assertRaises(ValueError): cache.selection_row(changed)

    def test_32_small_count_seams_never_establish_native_population_or_admission(self):
        case = self.case()
        with self.assertRaises(ValueError):
            cache.open_cache(case[0], expected_manifest_sha256=case[3]["manifest_file_sha256"], expected_ids=case[1], expected_identity=case[2], native_suffix=SUFFIX)
        for fixture in (1, "true", None):
            with self.subTest(fixture=fixture), self.assertRaises(ValueError): self.reader(case, fixture_only=fixture)
        identity = copy.deepcopy(case[2]); identity["extra"] = pin("unsupported")
        with self.assertRaises(ValueError): self.reader(case, expected_identity=identity)
        self.assertIs(self.manifest(case)["native_population_provenance_or_compute_admitted"], False)
