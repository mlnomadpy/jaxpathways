"""The evidence gate recomputes aggregates from per-question raw results."""

import sys
from pathlib import Path
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tests"))

from check import verify_qa_aggregate
from squad_scoring import score


class EvidenceReconstructionTests(unittest.TestCase):
    def test_fabricated_perfect_aggregate_is_rejected_against_raw_rows(self):
        ids = ["q1", "q2"]
        answers = {"q1": ["Paris"], "q2": ["blue whale"]}
        predictions = {"q1": "Paris", "q2": "wrong"}
        scores = {
            question_id: score(predictions[question_id], answers[question_id])
            for question_id in ids
        }
        report = {
            "questions": 2,
            "articles": 1,
            "exact_match": 0.5,
            "token_f1": 0.5,
            "per_question": [
                {"question_id": question_id, **value}
                for question_id, value in scores.items()
            ],
        }
        self.assertEqual(
            verify_qa_aggregate("branch", scores, report, ids, 1),
            {"questions": 2, "exact_match": 0.5, "token_f1": 0.5},
        )
        report["exact_match"] = 1.0
        report["token_f1"] = 1.0
        with self.assertRaisesRegex(AssertionError, "not reconstructed"):
            verify_qa_aggregate("branch", scores, report, ids, 1)


if __name__ == "__main__":
    unittest.main()
