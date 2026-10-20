"""Offline fixtures only; no native model/data work or generation."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from deadline.recovery import screen as screen_module
from reasoning.data import SYSTEM, prompt_id
from deadline.recovery.screen import (MAX_GPU_SECONDS, PROTOCOL, SIZE, _safe_file_map, canonical_rows,
    check_disjoint, choose_screen, deadline_guard, fixed_gate, load_prepared,
    render_prompt, strict_ids, summarize_records, text_digest, validate_summary)
from deadline.recovery.common import digest, write_new_json


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "system:" + messages[0]["content"] + "\nuser:" + messages[1]["content"] + "\nassistant:"

    def encode(self, text, **kwargs):
        if "oversize" in text:
            return [1] * 2049
        return list(text.encode())


def source_rows():
    rows = []
    for i in range(1000):
        question = "Original source question " + str(i)
        rows.append(dict(id=prompt_id(question), instruction=question,
                         input="", output=r"Teacher derivation. \boxed{42}", system=SYSTEM))
    return rows


def gate_metrics(base=40, advanced=35, recovery=38, advanced_length=12, recovery_length=6):
    return dict(base=dict(rows=64, teacher_agreement=base, length_finishes=0),
                advanced=dict(rows=64, teacher_agreement=advanced, length_finishes=advanced_length),
                recovery=dict(rows=64, teacher_agreement=recovery, length_finishes=recovery_length))


def sealed_screen_fixture(root, recovery_agreement=38):
    """Real tiny JSON/hash chain; no mocked validator or model runtime."""
    root = Path(root)
    output = root / "development"
    output.mkdir()
    models = {}
    for role in screen_module.ROLES:
        model = root / "models" / role / "model"
        model.mkdir(parents=True)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
            (model / name).write_text("small offline identity fixture: " + role + name)
        models[role] = dict(path=str(model.resolve()), files={
            path.name: digest(path) for path in model.iterdir()})
    advanced = Path(models["advanced"]["path"]).parent
    write_new_json(advanced / "manifest.json", dict(offline="training identity only"))
    write_new_json(advanced / "complete.json", dict(offline="training receipt identity only"))
    screen_file = Path(screen_module.__file__).resolve()
    merge = root / "merge-complete.json"
    write_new_json(merge, dict(schema="deadline-base-anchor-merge-v1", complete=True,
        alpha_advanced=.25,
        inputs=dict(base=models["base"], advanced=dict(models["advanced"],
            training_manifest_sha256=digest(advanced / "manifest.json"),
            training_receipt_sha256=digest(advanced / "complete.json"))),
        output=dict(models["recovery"], all_finite=True, finite_fp32_tensors=434),
        primary_plan_sha256=digest(screen_file.parents[1] / "experiment_plan.json"),
        recovery_plan_sha256=digest(screen_file.with_name("plan.json"))))
    selected, _ = choose_screen(source_rows(), FakeTokenizer())
    write_new_json(output / "selected_ids.json", [row["id"] for row in selected])
    write_new_json(output / "selected.json", selected)
    prepared = dict(schema=screen_module.SCHEMA, phase="prepare", complete=True,
        protocol=PROTOCOL, models=models, merge_receipt=str(merge), merge_receipt_sha256=digest(merge),
        selected_ids_sha256=digest(output / "selected_ids.json"),
        selected_sha256=digest(output / "selected.json"), code_sha256=digest(screen_file))
    write_new_json(output / "prepared.json", prepared)
    allocation = dict(job_id="12345", time_limit_seconds=MAX_GPU_SECONDS,
        reserve_seconds=screen_module.RESERVE_SECONDS, deadline_utc=screen_module.DEADLINE.isoformat())
    write_new_json(output / "gpu_allocation.json", allocation)
    counts = gate_metrics(recovery=recovery_agreement)
    for role in screen_module.ROLES:
        directory = output / role
        records = [dict(id=row["id"], teacher_answer_sha256=row["teacher_answer_sha256"],
            teacher_agreement=i < counts[role]["teacher_agreement"],
            finish_reason="length" if i < counts[role]["length_finishes"] else "stop",
            boxed_completion=True, repetition_8gram=0., output_tokens=20,
            output_sha256=text_digest("omitted offline output " + role + str(i)))
            for i, row in enumerate(selected)]
        write_new_json(directory / "observations.json", records)
        write_new_json(directory / "manifest.json", dict(schema=screen_module.SCHEMA, phase="run",
            role=role, protocol=PROTOCOL, model=models[role], allocation=allocation,
            prepared_sha256=digest(output / "prepared.json"), versions=dict(vllm="offline-fixture")))
        write_new_json(directory / "complete.json", dict(schema=screen_module.SCHEMA, complete=True,
            role=role, metrics=summarize_records(records, [row["id"] for row in selected]),
            observations_sha256=digest(directory / "observations.json"),
            manifest_sha256=digest(directory / "manifest.json")))
    write_new_json(output / "summary.json", screen_module._summary_payload(output))
    return output


class RecoveryScreenTests(unittest.TestCase):
    def test_native_prompt_and_system(self):
        text, tokens = render_prompt(FakeTokenizer(), source_rows()[0])
        self.assertIn(SYSTEM, text)
        self.assertEqual(tokens, list(text.encode()))

    def test_selection_exact_64_deterministic_under_reordering(self):
        rows = source_rows()
        selected, eligible = choose_screen(rows, FakeTokenizer())
        again, again_eligible = choose_screen(list(reversed(rows)), FakeTokenizer())
        self.assertEqual(selected, again)
        self.assertEqual((len(selected), eligible, again_eligible), (64, 1000, 1000))
        self.assertEqual(len({r["id"] for r in selected}), 64)
        self.assertTrue(all(r["prompt_tokens"] <= 2048 for r in selected))

    def test_teacher_solutions_and_prompt_text_not_exported(self):
        selected, _ = choose_screen(source_rows(), FakeTokenizer())
        serialized = json.dumps(selected)
        self.assertNotIn("Teacher derivation", serialized)
        self.assertNotIn("Original source question", serialized)
        self.assertNotIn("boxed", serialized)
        self.assertTrue(all(r["teacher_answer_sha256"] == text_digest("42") for r in selected))

    def test_eligibility_excludes_oversize_unboxed_and_incomplete(self):
        rows = source_rows()
        rows[0]["output"] = "No boxed teacher answer"
        rows[1]["output"] = r"Incomplete \boxed{42"
        rows[2]["instruction"] += " oversize"
        rows[2]["id"] = prompt_id(rows[2]["instruction"])
        selected, eligible = choose_screen(rows, FakeTokenizer())
        self.assertEqual(eligible, 997)
        self.assertTrue({r["id"] for r in rows[:3]}.isdisjoint(r["id"] for r in selected))

    def test_shortage_and_wrong_heldout_count_refused(self):
        rows = source_rows()
        with self.assertRaises(ValueError):
            choose_screen(rows[:-1], FakeTokenizer())
        for row in rows[63:]:
            row["output"] = "No box"
        with self.assertRaises(ValueError):
            choose_screen(rows, FakeTokenizer())

    def test_canonical_mismatch_duplicate_and_rewritten_system_refused(self):
        rows = source_rows()
        for change in (lambda r: r.update(id="0" * 64), lambda r: r.update(system="Changed"),
                       lambda r: r.update(input="Unfolded")):
            broken = copy.deepcopy(rows)
            change(broken[0])
            with self.assertRaises(ValueError):
                canonical_rows(broken, 1000)
        with self.assertRaises(ValueError):
            canonical_rows(rows + [rows[0]])

    def test_disjointness_refuses_any_training_or_probe_overlap(self):
        heldout, training = [prompt_id("heldout")], [prompt_id("training")]
        self.assertEqual(check_disjoint(heldout, dict(training=training)), dict(training=1))
        for name in ("pool", "random", "advanced", "probeA", "probeB"):
            with self.assertRaises(ValueError):
                check_disjoint(heldout, {name: training + heldout})
        with self.assertRaises(ValueError):
            check_disjoint(heldout, {})

    def test_ids_reject_missing_duplicates_noncanonical(self):
        for values in ([], ["bad"], ["a" * 64] * 2, [True]):
            with self.assertRaises(ValueError):
                strict_ids(values)

    def test_gate_all_boundaries(self):
        self.assertTrue(fixed_gate(gate_metrics())["accepted"])
        self.assertFalse(fixed_gate(gate_metrics(recovery=37))["accepted"])
        self.assertFalse(fixed_gate(gate_metrics(advanced=39))["accepted"])
        self.assertFalse(fixed_gate(gate_metrics(recovery_length=7))["accepted"])
        self.assertTrue(fixed_gate(gate_metrics(advanced_length=0, recovery_length=0))["accepted"])
        self.assertFalse(fixed_gate(gate_metrics(advanced_length=0, recovery_length=1))["accepted"])
        self.assertFalse(fixed_gate(gate_metrics(advanced_length=11, recovery_length=6))["accepted"])

    def test_gate_invalid_count_or_missing_role_refused(self):
        metrics = gate_metrics()
        metrics["recovery"]["teacher_agreement"] = True
        with self.assertRaises(ValueError):
            fixed_gate(metrics)
        metrics = gate_metrics()
        del metrics["base"]
        with self.assertRaises(ValueError):
            fixed_gate(metrics)

    def test_diagnostics_refuse_raw_generations_and_bad_types(self):
        identifier = prompt_id("heldout")
        row = dict(id=identifier, teacher_answer_sha256=text_digest("42"), teacher_agreement=True,
                   finish_reason="length", boxed_completion=False, repetition_8gram=.25,
                   output_tokens=4096, output_sha256=text_digest("omitted generation"))
        metrics = summarize_records([row], [identifier])
        self.assertEqual(metrics["teacher_agreement"], 1)
        self.assertEqual(metrics["length_finishes"], 1)
        for change in (dict(generated_response="solution"), dict(output_tokens=True),
                       dict(finish_reason="abort"), dict(repetition_8gram=float("nan")),
                       dict(teacher_agreement=1)):
            with self.assertRaises(ValueError):
                summarize_records([dict(row, **change)], [identifier])

    def test_immutable_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sealed.json"
            write_new_json(path, dict(frozen=True))
            original = digest(path)
            with self.assertRaises((ValueError, FileExistsError, RuntimeError)):
                write_new_json(path, dict(frozen=False))
            self.assertEqual(digest(path), original)

    def test_model_inventory_binds_notice_and_allows_only_unloaded_trainer_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            names = ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "model.safetensors", "LICENSE", "NOTICE")
            for name in names:
                (model / name).write_text("tiny offline fixture: " + name)
            files = {name: digest(model / name) for name in names}
            (model / "training_args.bin").write_bytes(b"never loaded")
            self.assertEqual(_safe_file_map(model, files)["files"], files)
            (model / "model.bin").write_bytes(b"forbidden pickle weights")
            with self.assertRaises(ValueError):
                _safe_file_map(model, files)
            (model / "model.bin").unlink()
            (model / "NOTICE").write_text("changed")
            with self.assertRaises(ValueError):
                _safe_file_map(model, files)

    def test_unsealed_output_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            write_new_json(path / "prepared.json", dict(schema="wrong", complete=True))
            with self.assertRaises(ValueError):
                load_prepared(path)

    def test_explicit_deadline_reserve(self):
        deadline_guard(MAX_GPU_SECONDS, datetime(2026, 10, 20, 23, 0, tzinfo=timezone.utc))
        with self.assertRaises(ValueError):
            deadline_guard(MAX_GPU_SECONDS, datetime(2026, 10, 21, 1, 0, tzinfo=timezone.utc))

    def test_validate_summary_recomputes_counts_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = sealed_screen_fixture(temporary)
            before = {str(path): digest(path) for path in output.rglob("*.json")}
            validated = validate_summary(output)
            self.assertTrue(validated["gate"]["accepted"])
            self.assertEqual(validated["metrics"]["recovery"]["teacher_agreement"], 38)
            self.assertEqual(validated["metrics"]["advanced"]["length_finishes"], 12)
            self.assertEqual(before, {str(path): digest(path) for path in output.rglob("*.json")})

    def test_validate_summary_rejects_tampered_stored_accepted_flag_and_counts(self):
        for field in ("accepted", "count", "accepted_type", "count_type"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                output = sealed_screen_fixture(temporary)
                path = output / "summary.json"
                summary = json.loads(path.read_text())
                if field == "accepted":
                    summary["gate"]["accepted"] = False
                elif field == "accepted_type":
                    summary["gate"]["accepted"] = 1
                elif field == "count_type":
                    summary["metrics"]["recovery"]["teacher_agreement"] = 38.0
                else:
                    summary["metrics"]["recovery"]["teacher_agreement"] = 39
                path.write_text(json.dumps(summary))
                with self.assertRaises(ValueError):
                    validate_summary(output)

    def test_validate_summary_rejects_falsely_accepted_failed_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = sealed_screen_fixture(temporary, recovery_agreement=37)
            path = output / "summary.json"
            summary = json.loads(path.read_text())
            self.assertFalse(summary["gate"]["accepted"])
            with self.assertRaises(ValueError):
                validate_summary(output)
            summary["gate"]["accepted"] = True
            path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                validate_summary(output)

    def test_validate_summary_rejects_role_count_tampering_and_split_allocation(self):
        for field in ("count", "allocation"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                output = sealed_screen_fixture(temporary)
                receipt_path = output / "advanced/complete.json"
                receipt = json.loads(receipt_path.read_text())
                if field == "count":
                    receipt["metrics"]["teacher_agreement"] += 1
                else:
                    manifest_path = output / "advanced/manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    manifest["allocation"]["job_id"] = "54321"
                    manifest_path.write_text(json.dumps(manifest))
                    receipt["manifest_sha256"] = digest(manifest_path)
                receipt_path.write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):
                    validate_summary(output)


if __name__ == "__main__":
    unittest.main()
