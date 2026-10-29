"""Synthetic smoke cases for the generic phase-separated runtime plot."""

from __future__ import annotations

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from plots import loss_figure


class LossFigureTests(unittest.TestCase):
    def render(self, records):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "loss.svg"
        loss_figure(path, records)
        source = path.read_text(encoding="utf-8")
        ET.fromstring(source)
        return source

    def test_variable_phase_counts_use_separate_units_and_update_axes(self):
        records = [
            {"stage": "phase1_mlm", "step": step, "loss_nats_per_masked_token": value}
            for step, value in zip((2, 5, 9), (3.0, 2.0, 1.5))
        ] + [
            {
                "stage": "phase2_contrastive_only",
                "step": step,
                "contrastive_plus_weighted_mlm_loss": value,
            }
            for step, value in enumerate((0.0, 2.0), 1)
        ]
        svg = self.render(records)
        self.assertIn("3 updates", svg)
        self.assertIn("2 updates", svg)
        self.assertIn("nats per selected masked token", svg)
        self.assertIn("contrastive objective units", svg)
        self.assertIn("Update within phase (2–9)", svg)
        self.assertIn("Update within phase (1–2)", svg)
        self.assertGreaterEqual(svg.count(">0</text>"), 2)

    def test_single_all_zero_phase_has_finite_zero_preserving_panel(self):
        svg = self.render(
            [
                {
                    "stage": "phase3_human_pairs",
                    "step": 1,
                    "multi_positive_loss": 0.0,
                    "query_ids": ["q1"],
                    "positive_pairs": 1,
                    "candidate_doc_ids": ["d1"],
                }
            ]
        )
        self.assertIn("1 updates", svg)
        self.assertIn("multi-positive contrastive objective units", svg)
        self.assertIn("Update within phase (1–1)", svg)
        self.assertIn(
            "Final phase-3 batch: 1 query, 1 positive pair, and 1 candidate", svg
        )
        self.assertNotIn("nan", svg.lower())
        self.assertNotIn("inf", svg.lower())

    def test_phase_three_note_requires_observed_singleton_batch(self):
        svg = self.render(
            [
                {
                    "stage": "phase3_human_pairs",
                    "step": 1,
                    "multi_positive_loss": 0.0,
                    "query_ids": ["q1", "q2"],
                    "positive_pairs": 2,
                    "candidate_doc_ids": ["d1", "d2"],
                }
            ]
        )
        self.assertNotIn("Final phase-3 batch:", svg)


if __name__ == "__main__":
    unittest.main()
