import json
import tempfile
import unittest
from pathlib import Path

from reasoning.skill_graph import coverage_select, label_items, load_labels, validate_label


def label(skills, domain="algebra"):
    return dict(domain=domain, skills=skills, composition_depth=2,
                representation="symbolic", language="english", answer_form="integer")


class SkillGraphTests(unittest.TestCase):
    def test_schema_and_normalized_pairs(self):
        x = validate_label(label(["factoring", "equations"]))
        self.assertEqual(x["skills"], ["equations", "factoring"])
        self.assertIn("p:equations+factoring", label_items(x, 3))
        with self.assertRaises(ValueError):
            validate_label(label(["equations", "equations"]))
        with self.assertRaises(ValueError):
            validate_label(label(["equations", "invented_skill"]))

    def test_ledger_requires_exact_order(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "labels.jsonl"
            p.write_text(json.dumps({"id": "b", "label": label(["equations", "factoring"])}) + "\n")
            with self.assertRaises(ValueError):
                load_labels(p, ["a", "b"], allow_partial=True)

    def test_greedy_selection_is_distinct_and_covers_rare_skill(self):
        rows = [dict(id=str(i), instruction=f"Question {i}", input="", output="x", system="x")
                for i in range(10)]
        features = [dict(id=str(i), cluster=i % 2, total_tokens=1000,
                         response_tokens=500, complete_box=True, repetition=0)
                    for i in range(10)]
        labels = [dict(id=str(i), **validate_label(label(
            ["equations", "circle_geometry"] if i == 9 else ["equations", "factoring"],
            "geometry" if i == 9 else "algebra"))) for i in range(10)]
        selected, receipt = coverage_select(rows, features, labels, 3, 8192)
        self.assertEqual(len({r["id"] for r in selected}), 3)
        self.assertIn("9", {r["id"] for r in selected})
        self.assertEqual(receipt["selected_ids"], [r["id"] for r in selected])
        self.assertEqual(coverage_select(rows, features, labels, 3, 8192)[1]["selected_ids"],
                         receipt["selected_ids"])


if __name__ == "__main__":
    unittest.main()
