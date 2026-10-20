"""Synthetic CPU checks for source-only selection; no benchmark data or model scores."""
import copy
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from deadline.compatible_coverage import (COLUMNS, allocate, content_sha256,
                                         select_datasets, select_random_control)
from reasoning import features as feature_module
from reasoning.data import digest, prompt_id, write_json


def fixture(n=40):
    rows, features = [], []
    for index in range(n):
        question = f"Q{index}: compute {index} + 37."
        row = dict(id=prompt_id(question), instruction=question, input="",
                   output=f"Add the two integers. \\boxed{{{index + 37}}}",
                   system=r"Please reason step by step, and put your final answer within \boxed{}.")
        response = (400, 1200, 2500)[index % 3]
        rows.append(row)
        features.append(dict(id=row["id"], cluster=index % 4, total_tokens=response + 80,
                             response_tokens=response, complete_box=True, repetition=.01))
    meta = dict(input_sha256="a" * 64,
                resolved_tokenizer_commit="b" * 40,
                provenance=dict(code_sha256=digest(feature_module.__file__)))
    heldout = [dict(instruction="Entirely different reserved source task", id="reserved-source-id")]
    return rows, features, meta, heldout


def transfer_fixture(rows):
    probe_a, probe_b = rows[-2]["id"], rows[-1]["id"]
    reservoir = [row["id"] for row in rows[:-2]]
    folds = {identifier: str(i % 2) for i, identifier in enumerate(reservoir)}
    folds.update({probe_a: "0", probe_b: "1"})
    routes = {"0": "probe_b", "1": "probe_a"}
    scores = []
    for i, identifier in enumerate(reservoir):
        gain = (i % 8 - 2) / 10
        scores.append(dict(id=identifier, fold=folds[identifier], probe_id=routes[folds[identifier]],
                           base_ce=2., probe_ce=2. * (1 - gain), gain=gain))
    meta = dict(input_sha256="a" * 64, base_model_sha256="c" * 64,
                tokenizer_sha256="d" * 64, reservoir_ids=reservoir,
                fold_assignments=folds, fold_to_probe=routes,
                probe_training_ids={"probe_a": [probe_a], "probe_b": [probe_b]},
                probe_initialization_sha256={"probe_a": "c" * 64, "probe_b": "c" * 64},
                probe_checkpoint_sha256={"probe_a": "e" * 64, "probe_b": "f" * 64})
    return scores, meta


class DeadlineSelectionTests(unittest.TestCase):
    def run_selection(self, rows, features, meta, heldout, **kwargs):
        return select_datasets(rows, features, feature_meta=meta, pool_sha256="a" * 64,
                               heldout_rows=heldout, size=kwargs.pop("size", 12), **kwargs)

    def test_crossfit_count_gain_cap_and_unchanged_targets(self):
        rows, features, meta, heldout = fixture()
        scores, transfer = transfer_fixture(rows)
        before = copy.deepcopy((rows, features, meta, heldout, scores, transfer))
        result = self.run_selection(rows, features, meta, heldout,
                                    transfer_scores=scores, transfer_meta=transfer)
        lookup = {row["id"]: row for row in rows}
        reserved = {rows[-2]["id"], rows[-1]["id"]}
        for selected in result["datasets"].values():
            self.assertEqual(len(selected), 12)
            self.assertEqual(len({r["id"] for r in selected}), 12)
            self.assertFalse({r["id"] for r in selected} & reserved)
            self.assertTrue(all(r == lookup[r["id"]] for r in selected))
        self.assertTrue(all(c["gain_leaders"] <= c["quota"] // 2
                            for c in result["manifest"]["cells"].values()))
        self.assertEqual(before, (rows, features, meta, heldout, scores, transfer))
        result["datasets"]["advanced"][0]["output"] = "mutated return value"
        self.assertEqual(before[0], rows)
        for info in result["dataset_info"].values():
            self.assertEqual(info["columns"], COLUMNS)

    def test_determinism_and_input_order_invariance(self):
        rows, features, meta, heldout = fixture()
        scores, transfer = transfer_fixture(rows)
        first = self.run_selection(rows, features, meta, heldout,
                                   transfer_scores=scores, transfer_meta=transfer)
        second = self.run_selection(list(reversed(rows)), list(reversed(features)), meta,
                                    list(reversed(heldout)), transfer_scores=list(reversed(scores)),
                                    transfer_meta=transfer)
        self.assertEqual(first, second)
        manifest = copy.deepcopy(first["manifest"])
        expected = manifest.pop("selection_sha256")
        self.assertEqual(expected, content_sha256(manifest))

    def test_full_15000_budget_and_fixed_length_cells(self):
        rows, features, meta, heldout = fixture(15010)
        result = self.run_selection(rows, features, meta, heldout, size=15000, strategy="coverage")
        self.assertEqual(len(result["datasets"]["advanced"]), 15000)
        self.assertEqual(len(result["datasets"]["random"]), 15000)
        cells = result["manifest"]["cells"]
        self.assertEqual(sum(c["quota"] for c in cells.values()), 15000)
        self.assertEqual({key.split(":")[1] for key in cells}, {"1024", "2048", "4096"})
        self.assertIn("no student-aware", result["manifest"]["policy_label"])

    def test_source_heldout_reserved_from_both_arms(self):
        rows, features, meta, heldout = fixture()
        heldout.append(rows[0])
        result = self.run_selection(rows, features, meta, heldout, strategy="coverage")
        self.assertEqual(result["manifest"]["source_heldout_excluded_ids"], [rows[0]["id"]])
        for selected in result["datasets"].values():
            self.assertNotIn(rows[0]["id"], {r["id"] for r in selected})
        with self.assertRaisesRegex(ValueError, "held-out"):
            self.run_selection(rows, features, meta, [], strategy="coverage")

    def test_source_near_duplicates_are_also_reserved(self):
        rows, features, meta, heldout = fixture()
        question = "A distinctive source-only question with a long sequence of different words that must stay reserved."
        rows[0]["instruction"] = question + " Please solve."
        rows[0]["id"] = prompt_id(rows[0]["instruction"])
        features[0]["id"] = rows[0]["id"]
        heldout.append(dict(instruction=question))
        result = self.run_selection(rows, features, meta, heldout, strategy="coverage")
        self.assertIn(rows[0]["id"], result["manifest"]["source_heldout_excluded_ids"])

    def test_structure_length_repetition_and_final_box_filters(self):
        rows, features, meta, heldout = fixture()
        rows[0]["output"] = "No final answer"
        features[0]["complete_box"] = False
        rows[1]["output"] += " Now \\boxed{unfinished"
        features[1]["complete_box"] = False
        features[2]["repetition"] = .151
        features[3]["response_tokens"] = 127
        features[4]["total_tokens"] = 4096 - 31
        rows[5]["output"] = "<think>unfinished " + rows[5]["output"]
        result = self.run_selection(rows, features, meta, heldout, strategy="coverage")
        rejected = {row["id"] for row in rows[:6]}
        self.assertFalse(rejected & {r["id"] for r in result["datasets"]["advanced"]})
        # Exactly on the safety bound is eligible; response text is not cropped.
        features[4]["total_tokens"] = 4096 - 32
        result = self.run_selection(rows, features, meta, heldout, size=35, strategy="coverage")
        self.assertIn(rows[4]["id"], {r["id"] for r in result["datasets"]["advanced"]})

    def test_uniform_random_retains_overlength_original_and_reports_it(self):
        rows, features, meta, heldout = fixture()
        expected = random.Random(42).sample(sorted(rows, key=lambda r: r["id"]), 12)
        overlength_id = expected[0]["id"]
        for feature in features:
            if feature["id"] == overlength_id:
                feature["total_tokens"] = 10000
        result = self.run_selection(rows, features, meta, heldout, strategy="coverage")
        self.assertEqual(result["datasets"]["random"], expected)
        self.assertEqual(result["manifest"]["datasets"]["random"]["over_cutoff_ids"], [overlength_id])
        self.assertNotIn(overlength_id, {r["id"] for r in result["datasets"]["advanced"]})

    def test_shortage_fails_instead_of_relaxing_or_silent_drop(self):
        rows, features, meta, heldout = fixture(12)
        with self.assertRaisesRegex(ValueError, "random pool shortage"):
            self.run_selection(rows, features, meta, heldout, size=13, strategy="coverage")
        features[0]["repetition"] = .9
        with self.assertRaisesRegex(ValueError, "eligible reservoir shortage"):
            self.run_selection(rows, features, meta, heldout, size=12, strategy="coverage")
        scores, transfer = transfer_fixture(rows)
        features[0]["repetition"] = .01
        omitted = scores[-1]["id"]
        transfer["reservoir_ids"].remove(omitted)
        transfer["fold_assignments"].pop(omitted)
        with self.assertRaisesRegex(ValueError, "eligible reservoir shortage"):
            self.run_selection(rows, features, meta, heldout, size=10,
                               transfer_scores=scores[:-1], transfer_meta=transfer)

    def test_stale_features_and_identity_mismatch_fail(self):
        rows, features, meta, heldout = fixture()
        for changes in (dict(input_sha256="0" * 64), dict(provenance={"code_sha256": "0" * 64})):
            bad_meta = dict(meta, **changes)
            with self.assertRaisesRegex(ValueError, "stale features"):
                self.run_selection(rows, features, bad_meta, heldout, strategy="coverage")
        with self.assertRaisesRegex(ValueError, "exact canonical pool"):
            self.run_selection(rows, features[:-1], meta, heldout, strategy="coverage")
        features[0]["complete_box"] = False
        with self.assertRaisesRegex(ValueError, "boxed-answer flag"):
            self.run_selection(rows, features, meta, heldout, strategy="coverage")

    def test_crossfit_leakage_and_probe_holdout_membership_fail(self):
        rows, features, meta, heldout = fixture()
        scores, transfer = transfer_fixture(rows)
        bad = copy.deepcopy(transfer)
        bad["fold_to_probe"] = {"0": "probe_a", "1": "probe_b"}
        with self.assertRaisesRegex(ValueError, "trained on its candidate fold"):
            self.run_selection(rows, features, meta, heldout, transfer_scores=scores, transfer_meta=bad)
        with self.assertRaisesRegex(ValueError, "disjoint from all held-out"):
            self.run_selection(rows, features, meta, heldout + [rows[-1]],
                               transfer_scores=scores, transfer_meta=transfer)
        bad = copy.deepcopy(transfer)
        bad["reservoir_ids"].append(rows[-1]["id"])
        with self.assertRaisesRegex(ValueError, "reserved from the final"):
            self.run_selection(rows, features, meta, heldout, transfer_scores=scores, transfer_meta=bad)

    def test_partial_duplicate_nonfinite_or_wrong_gain_scores_fail(self):
        rows, features, meta, heldout = fixture()
        scores, transfer = transfer_fixture(rows)
        for bad_scores, message in ((scores[:-1], "every reservoir"),
                                    (scores + [scores[0]], "duplicate transfer")):
            with self.assertRaisesRegex(ValueError, message):
                self.run_selection(rows, features, meta, heldout,
                                   transfer_scores=bad_scores, transfer_meta=transfer)
        for change, message in ((dict(base_ce=float("nan")), "nonfinite"),
                                (dict(gain=.999), "disagrees"),
                                (dict(probe_id="probe_a" if scores[0]["probe_id"] != "probe_a" else "probe_b"), "mismatch")):
            bad = copy.deepcopy(scores)
            bad[0].update(change)
            with self.assertRaisesRegex(ValueError, message):
                self.run_selection(rows, features, meta, heldout, transfer_scores=bad, transfer_meta=transfer)

    def test_old_probe_initialization_and_stale_transfer_fail(self):
        rows, features, meta, heldout = fixture()
        scores, transfer = transfer_fixture(rows)
        bad = copy.deepcopy(transfer)
        bad["probe_initialization_sha256"]["probe_a"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "original base"):
            self.run_selection(rows, features, meta, heldout, transfer_scores=scores, transfer_meta=bad)
        bad = dict(transfer, input_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "stale transfer"):
            self.run_selection(rows, features, meta, heldout, transfer_scores=scores, transfer_meta=bad)
        for key, message in (("probe_initialization_sha256", "original base"),
                             ("probe_checkpoint_sha256", "checkpoint SHA")):
            bad = dict(transfer)
            bad.pop(key)
            with self.assertRaisesRegex(ValueError, message):
                self.run_selection(rows, features, meta, heldout, transfer_scores=scores, transfer_meta=bad)

    def test_prescore_random_control_exactly_matches_final_export(self):
        rows, features, meta, heldout = fixture()
        scores, transfer = transfer_fixture(rows)
        result = self.run_selection(rows, features, meta, heldout,
                                    transfer_scores=scores, transfer_meta=transfer)
        prescore = select_random_control(rows, heldout_rows=heldout,
                                         probe_ids=[rows[-2]["id"], rows[-1]["id"]], size=12)
        self.assertEqual(prescore["rows"], result["datasets"]["random"])
        self.assertEqual(prescore["sampling_pool_rows"], result["manifest"]["random_sampling_pool_rows"])

    def test_explicit_fallback_and_unscored_outside_reservoir(self):
        rows, features, meta, heldout = fixture()
        with self.assertRaisesRegex(ValueError, "requires scores"):
            self.run_selection(rows, features, meta, heldout)
        scores, transfer = transfer_fixture(rows)
        removed = {s["id"] for s in scores[-4:]}
        transfer["reservoir_ids"] = [identifier for identifier in transfer["reservoir_ids"] if identifier not in removed]
        transfer["fold_assignments"] = {identifier: fold for identifier, fold in transfer["fold_assignments"].items()
                                         if identifier not in removed}
        result = self.run_selection(rows, features, meta, heldout,
                                    transfer_scores=scores[:-4], transfer_meta=transfer)
        self.assertFalse(removed & {r["id"] for r in result["datasets"]["advanced"]})
        self.assertEqual(result["manifest"]["random_sampling_pool_rows"], len(rows) - 2)
        with self.assertRaisesRegex(ValueError, "silently ignore"):
            self.run_selection(rows, features, meta, heldout, strategy="coverage",
                               transfer_scores=scores, transfer_meta=transfer)

    def test_integer_fold_labels_and_no_positive_gains_are_auditable(self):
        rows, features, meta, heldout = fixture()
        scores, transfer = transfer_fixture(rows)
        transfer["fold_assignments"] = {identifier: int(fold)
                                         for identifier, fold in transfer["fold_assignments"].items()}
        for score in scores:
            score.update(fold=int(score["fold"]), base_ce=2., probe_ce=2.2, gain=-.1)
        result = self.run_selection(rows, features, meta, heldout,
                                    transfer_scores=scores, transfer_meta=transfer)
        summary = result["manifest"]["transfer_summary"]
        self.assertEqual(summary["gain_leaders"], 0)
        self.assertEqual(summary["positive_gain_candidates"], 0)
        self.assertEqual(summary["coverage_rows"], 12)

    def test_cluster_sqrt_allocation_and_length_capacity(self):
        quotas = allocate({"common": 100, "rare": 4}, 12, square_root=True)
        self.assertGreater(quotas["rare"], 0)
        self.assertLessEqual(quotas["rare"], 4)
        self.assertEqual(sum(quotas.values()), 12)
        self.assertEqual(allocate({"a": 2, "b": 3}, 0), {"a": 0, "b": 0})
        with self.assertRaisesRegex(ValueError, "shortage"):
            allocate({"a": 2}, 3)

    def test_cli_merges_score_shards_and_explicit_system_mapping(self):
        rows, features, meta, heldout = fixture()
        scores, transfer = transfer_fixture(rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "pool.json", rows)
            meta["input_sha256"] = transfer["input_sha256"] = digest(root / "pool.json")
            for name, value in (("features.json", features), ("features.json.meta.json", meta),
                                ("heldout.json", heldout), ("transfer.json", transfer)):
                write_json(root / name, value)
            for fold in ("0", "1"):
                (root / f"scores-{fold}.jsonl").write_text("\n".join(json.dumps(s) for s in scores if s["fold"] == fold))
            command = [sys.executable, "-m", "deadline.compatible_coverage", "--pool", str(root / "pool.json"),
                       "--features", str(root / "features.json"), "--heldout", str(root / "heldout.json"),
                       "--transfer-meta", str(root / "transfer.json"), "--size", "12", "--output", str(root / "out")]
            for fold in ("0", "1"):
                command += ["--transfer-scores", str(root / f"scores-{fold}.jsonl")]
            run = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            for name in ("random", "advanced"):
                exported = json.loads((root / "out" / f"{name}.json").read_text())
                self.assertEqual(len(exported), 12)
                self.assertTrue(all(set(row) == {"instruction", "input", "output", "system"} for row in exported))
            info = json.loads((root / "out" / "dataset_info.json").read_text())
            self.assertEqual(info["advanced"]["columns"], COLUMNS)
            repeat = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertIn("refusing to overwrite", repeat.stderr)


if __name__ == "__main__":
    unittest.main()
