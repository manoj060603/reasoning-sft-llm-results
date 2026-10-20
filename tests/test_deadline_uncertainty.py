"""Synthetic raw-question bootstrap checks; no measured benchmark scores."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from deadline.audit_results import EXPECTED, OFFICIAL, PROTOCOL, REQUIRED_SETTINGS
from deadline.compare_uncertainty import compare_uncertainty, paired_cluster_bootstrap
from reasoning.benchmark import record_ids
from reasoning.data import prompt_id, write_json


def make_runs(root, *, counts=None, omitted=(), aggregates=(), math_only_gain=False, identical=False):
    counts = {name: 3 for name in EXPECTED} if counts is None else counts
    paths = {}
    for role in ("original", "random", "advanced"):
        path = root / role
        paths[role] = path
        manifest = dict(REQUIRED_SETTINGS, seed=0, max_model_len=32768, generation_config="vllm",
                        top_k=-1, min_p=0., repetition_penalty=1., batch_size=1024,
                        tensor_parallel_size=1, max_num_seqs=128, gpu_memory_utilization=.85,
                        versions={"synthetic_fixture": "1"}, grader_sha256={"grader.py": "synthetic"},
                        evaluator_sha256="synthetic-evaluator-code-hash",
                        problem_counts={}, problem_ids={}, data={})
        metrics = {}
        for benchmark, count in counts.items():
            if benchmark in omitted:
                continue
            rows = []
            for index in range(count):
                if math_only_gain:
                    flags = [role == "advanced" and benchmark == "math"] * 8
                elif identical:
                    flags = [sample < index % 4 for sample in range(8)]
                else:
                    correct = (index + {"original": 0, "random": 1, "advanced": 2}[role]) % 5
                    flags = [sample < correct for sample in range(8)]
                question = ("Kaoyan repeated question" if benchmark == "kaoyan" and index < 2 else
                            f"{benchmark}: source question {index}")
                rows.append(dict(question=question, answers_correctness=flags,
                    generated_responses=["Recorded answer." for _ in range(8)],
                    finish_reasons=["length" if sample == 7 else "stop" for sample in range(8)],
                    output_tokens=[10 + index] * 8))
            for row, identifier in zip(rows, record_ids(rows)):
                row["problem_id"] = identifier
            manifest["problem_counts"][benchmark] = count
            manifest["problem_ids"][benchmark] = [row["problem_id"] for row in rows]
            manifest["data"][benchmark] = f"synthetic-data-digest-{benchmark}"
            metrics[benchmark] = dict(problems=count, samples_per_problem=8,
                pass_at_1=sum(sum(row["answers_correctness"]) for row in rows) / (8 * count),
                pass_at_8=sum(any(row["answers_correctness"]) for row in rows) / count,
                truncated_samples=count)
            if benchmark not in aggregates:
                write_json(path / benchmark / "batch_000000.json", rows)
        write_json(path / "manifest.json", manifest)
        write_json(path / "metrics.json", metrics)
    return paths, counts


def paired_record(identifier, question, p1, p8):
    return dict(problem_id=identifier, question=question, question_cluster=prompt_id(question),
                pass_at_1=p1, pass_at_8=p8)


class UncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_identical_models_have_exact_zero_delta_and_intervals(self):
        paths, counts = make_runs(self.root, identical=True)
        result = compare_uncertainty(paths, expected=counts, resamples=1000)
        for comparison in result["comparisons"].values():
            for benchmark in comparison["benchmarks"].values():
                for metric in benchmark["metrics"].values():
                    self.assertEqual(metric["delta"], 0.)
                    self.assertEqual(metric["ci95"], [0., 0.])
            for macro in comparison["macros"].values():
                self.assertEqual(macro["status"], "complete")
                self.assertEqual(macro["metrics"]["pass_at_8"]["ci95"], [0., 0.])

    def test_duplicate_normalized_questions_always_travel_together(self):
        questions = ("Repeated Problem", "REPEATED   problem", "Different problem")
        candidate = [paired_record(str(i), q, 1. if i < 2 else 0., 1. if i < 2 else 0.)
                     for i, q in enumerate(questions)]
        comparator = [paired_record(str(i), q, 0. if i < 2 else 1., 0. if i < 2 else 1.)
                      for i, q in enumerate(questions)]
        result = paired_cluster_bootstrap(candidate, comparator, resamples=1000)
        self.assertEqual(result["question_clusters"], 2)
        self.assertEqual(result["cluster_record_counts"], [1, 2])
        self.assertEqual(result["duplicate_question_records"], 1)
        # Sampling two clusters permits AA, AB or BB only. A naive record
        # bootstrap would also generate -1/3 by splitting the duplicate pair.
        self.assertEqual({round(value, 8) for value in result["draws"]["pass_at_8"]},
                         {-1., round(1 / 3, 8), 1.})
        self.assertAlmostEqual(result["metrics"]["pass_at_8"]["delta"], 1 / 3)

    def test_zero_average_from_opposing_effects_retains_uncertainty(self):
        candidate = [paired_record("a", "First question", 1., 1.),
                     paired_record("b", "Second question", 0., 0.)]
        comparator = [paired_record("a", "First question", 0., 0.),
                      paired_record("b", "Second question", 1., 1.)]
        result = paired_cluster_bootstrap(candidate, comparator, resamples=1000)
        self.assertEqual(result["metrics"]["pass_at_8"]["delta"], 0.)
        self.assertEqual(result["metrics"]["pass_at_8"]["ci95"], [-1., 1.])

    def test_macros_are_equal_benchmark_weighted_and_stratified(self):
        counts = {name: (10 if name == "math" else 1) for name in EXPECTED}
        paths, counts = make_runs(self.root, counts=counts, math_only_gain=True)
        result = compare_uncertainty(paths, expected=counts, resamples=1000)
        for comparison in result["comparisons"].values():
            official = comparison["macros"]["official_eight_macro"]
            self.assertEqual(official["metrics"]["pass_at_8"]["delta"], 1 / 8)
            self.assertEqual(official["metrics"]["pass_at_8"]["ci95"], [1 / 8, 1 / 8])
            historical = comparison["macros"]["supplemented_historical_seven_macro"]
            self.assertEqual(historical["metrics"]["pass_at_8"]["delta"], 0.)
            pooled = official["pooled_generation_diagnostics"]["advanced"]
            self.assertEqual(pooled["samples"], 17 * 8)
            self.assertEqual(pooled["truncation_rate"], 1 / 8)

    def test_partial_or_aggregate_only_history_never_creates_a_macro(self):
        paths, counts = make_runs(self.root, aggregates=("aime25",))
        result = compare_uncertainty(paths, expected=counts, resamples=1000)
        for comparison in result["comparisons"].values():
            self.assertEqual(comparison["macros"]["official_eight_macro"]["status"], "complete")
            macro = comparison["macros"]["supplemented_historical_seven_macro"]
            self.assertEqual(macro["status"], "unavailable")
            self.assertIsNone(macro["metrics"])
            self.assertEqual(macro["missing_raw_matched_benchmarks"], ["aime25"])
        self.assertIn("aggregate receipt only", result["unavailable_benchmarks"]["aime25"][0])

    def test_incomplete_official_benchmark_refuses_partial_average(self):
        paths, counts = make_runs(self.root, omitted=("gpqa",))
        result = compare_uncertainty(paths, expected=counts, resamples=1000)
        for comparison in result["comparisons"].values():
            self.assertNotIn("gpqa", comparison["benchmarks"])
            self.assertIsNone(comparison["macros"]["official_eight_macro"]["metrics"])
            self.assertIsNone(comparison["macros"]["supplemented_historical_seven_macro"]["metrics"])

    def test_protocol_and_benchmark_data_mismatch_are_fatal(self):
        paths, counts = make_runs(self.root)
        path = paths["advanced"] / "manifest.json"
        original = json.loads(path.read_text())
        changed = copy.deepcopy(original)
        changed["seed"] = 7
        write_json(path, changed)
        with self.assertRaisesRegex(ValueError, "Unmatched evaluator protocol"):
            compare_uncertainty(paths, expected=counts, resamples=1000)
        changed = copy.deepcopy(original)
        changed["data"]["math"] = "different"
        write_json(path, changed)
        with self.assertRaisesRegex(ValueError, "Unmatched math evidence"):
            compare_uncertainty(paths, expected=counts, resamples=1000)

    def test_missing_or_mismatched_raw_question_is_fatal(self):
        paths, counts = make_runs(self.root)
        path = paths["advanced"] / "math" / "batch_000000.json"
        rows = json.loads(path.read_text())
        original = copy.deepcopy(rows)
        rows[0].pop("question")
        write_json(path, rows)
        with self.assertRaisesRegex(ValueError, "raw question text is required"):
            compare_uncertainty(paths, expected=counts, resamples=1000)
        original[0]["question"] = "A different actual question"
        write_json(path, original)
        with self.assertRaisesRegex(ValueError, "raw question identities differ"):
            compare_uncertainty(paths, expected=counts, resamples=1000)

    def test_all_roles_cannot_change_questions_under_unchanged_manifest_ids(self):
        paths, counts = make_runs(self.root)
        for path in paths.values():
            raw_path = path / "math" / "batch_000000.json"
            rows = json.loads(raw_path.read_text())
            rows[0]["question"] = "Identical tampering in every run with the old problem ID"
            write_json(raw_path, rows)
        with self.assertRaisesRegex(ValueError, "raw question identities differ"):
            compare_uncertainty(paths, expected=counts, resamples=1000)

    def test_real_duplicate_record_ids_bind_clusters_and_original_positions(self):
        paths, counts = make_runs(self.root)
        result = compare_uncertainty(paths, expected=counts, resamples=1000)
        raw = json.loads((paths["original"] / "kaoyan" / "batch_000000.json").read_text())
        ids = [row["problem_id"] for row in raw]
        self.assertEqual(record_ids(raw), ids)
        self.assertNotEqual(ids[0], ids[1])
        self.assertEqual(prompt_id(raw[0]["question"]), prompt_id(raw[1]["question"]))
        estimate = result["comparisons"]["advanced_vs_random"]["benchmarks"]["kaoyan"]
        self.assertEqual(estimate["question_clusters"], 2)
        self.assertEqual(estimate["duplicate_question_records"], 1)

    def test_raw_row_order_does_not_change_pairing_or_fixed_seed_results(self):
        paths, counts = make_runs(self.root)
        first = compare_uncertainty(paths, expected=counts, resamples=1000)
        path = paths["random"] / "kaoyan" / "batch_000000.json"
        write_json(path, list(reversed(json.loads(path.read_text()))))
        second = compare_uncertainty(paths, expected=counts, resamples=1000)
        self.assertEqual(first["comparisons"], second["comparisons"])
        self.assertEqual(first["generation_diagnostics"], second["generation_diagnostics"])
        self.assertNotEqual(first["sources"], second["sources"])

    def test_missing_generation_diagnostics_are_null_not_imputed(self):
        paths, counts = make_runs(self.root)
        path = paths["advanced"] / "math" / "batch_000000.json"
        rows = json.loads(path.read_text())
        rows[0].pop("output_tokens")
        rows[0].pop("finish_reasons")
        write_json(path, rows)
        result = compare_uncertainty(paths, expected=counts, resamples=1000)
        diagnostic = result["generation_diagnostics"]["math"]["advanced"]
        self.assertIsNone(diagnostic["truncation_rate"])
        self.assertIsNone(diagnostic["mean_output_tokens"])
        self.assertEqual(diagnostic["mean_output_characters"], len("Recorded answer."))
        pooled = result["comparisons"]["advanced_vs_random"]["macros"]["official_eight_macro"]["pooled_generation_diagnostics"]["advanced"]
        self.assertIsNone(pooled["truncation_rate"])
        self.assertIsNone(pooled["mean_output_tokens"])

    def test_cli_creates_only_json_and_refuses_overwrite(self):
        paths, _ = make_runs(self.root, counts=EXPECTED, identical=True)
        output = self.root / "analysis" / "uncertainty.json"
        command = [sys.executable, "-m", "deadline.compare_uncertainty", "--output", str(output), "--resamples", "1000"]
        for role, path in paths.items():
            command += ["--" + role, str(path)]
        run = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(output.read_text())
        self.assertEqual(result["seed"], 20261019)
        self.assertFalse(result["multiple_comparison_adjustment"])
        self.assertIn("training-seed", result["conditioning"])
        self.assertFalse(list(output.parent.glob("*.md")))
        second = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("Refusing to overwrite", second.stderr)


if __name__ == "__main__":
    unittest.main()
