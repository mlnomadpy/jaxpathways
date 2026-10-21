"""Prevent a package from combining checkpoints with stale evaluation receipts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from export_consumer import _candidate_hash, _hash, _validate_evaluation_bindings


class ExportBindingTests(unittest.TestCase):
    def fixture(self, root):
        data = root / "data"
        stage = root / "run" / "branch-a"
        data.mkdir()
        stage.mkdir(parents=True)
        (data / "frozen_recipe.json").write_text('{"recipe":1}')
        (data / "tokenizer.json").write_text('{"tokenizer":1}')
        (stage / "final-state.npz").write_bytes(b"checkpoint bytes")
        identity = {
            "candidate_source_sha256": _candidate_hash(),
            "recipe_sha256": _hash(data / "frozen_recipe.json"),
            "processor_sha256": _hash(data / "tokenizer.json"),
        }
        identity_path = stage / "identity.json"
        identity_path.write_text(json.dumps(identity, sort_keys=True))
        metrics = {
            "recipe_sha256": identity["recipe_sha256"],
            "tokenizer_sha256": identity["processor_sha256"],
            "evaluator_source_sha256": _hash(PROJECT / "evaluate_pilot.py"),
            "scorer": {"reference_bytes_sha256": _hash(PROJECT / "squad_scoring.py")},
            "branches": {
                "branch-a": {
                    "checkpoint_sha256": _hash(stage / "final-state.npz"),
                    "checkpoint_identity_sha256": _hash(identity_path),
                    "candidate_source_sha256": identity["candidate_source_sha256"],
                }
            },
        }
        return data, stage, metrics

    def test_matching_receipt_binds_all_frozen_inputs_and_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            data, stage, metrics = self.fixture(Path(directory))
            result = _validate_evaluation_bindings(data, stage, "branch-a", metrics)
            self.assertEqual(
                result["checkpoint_sha256"],
                metrics["branches"]["branch-a"]["checkpoint_sha256"],
            )
            self.assertEqual(result["tokenizer_sha256"], metrics["tokenizer_sha256"])

    def test_stale_or_wrong_evaluation_receipts_are_rejected(self):
        cases = (
            ("recipe", "recipe_sha256", "evaluation recipe_sha256"),
            ("tokenizer", "tokenizer_sha256", "evaluation tokenizer_sha256"),
            (
                "evaluator",
                "evaluator_source_sha256",
                "evaluation evaluator_source_sha256",
            ),
        )
        for label, field, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                data, stage, metrics = self.fixture(Path(directory))
                metrics[field] = "0" * 64
                with self.assertRaisesRegex(ValueError, message):
                    _validate_evaluation_bindings(data, stage, "branch-a", metrics)

        with tempfile.TemporaryDirectory() as directory:
            data, stage, metrics = self.fixture(Path(directory))
            metrics["scorer"]["reference_bytes_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "scorer.reference_bytes_sha256"):
                _validate_evaluation_bindings(data, stage, "branch-a", metrics)

        with tempfile.TemporaryDirectory() as directory:
            data, stage, metrics = self.fixture(Path(directory))
            metrics["branches"]["branch-a"]["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "checkpoint digest"):
                _validate_evaluation_bindings(data, stage, "branch-a", metrics)

        with tempfile.TemporaryDirectory() as directory:
            data, stage, metrics = self.fixture(Path(directory))
            metrics["branches"]["branch-a"]["checkpoint_identity_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                _validate_evaluation_bindings(data, stage, "branch-a", metrics)

        with tempfile.TemporaryDirectory() as directory:
            data, stage, metrics = self.fixture(Path(directory))
            metrics["branches"]["branch-a"]["candidate_source_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "candidate identity"):
                _validate_evaluation_bindings(data, stage, "branch-a", metrics)

        with tempfile.TemporaryDirectory() as directory:
            data, stage, metrics = self.fixture(Path(directory))
            identity_path = stage / "identity.json"
            identity = json.loads(identity_path.read_text())
            identity["recipe_sha256"] = "0" * 64
            identity_path.write_text(json.dumps(identity, sort_keys=True))
            metrics["branches"]["branch-a"]["checkpoint_identity_sha256"] = _hash(
                identity_path
            )
            with self.assertRaisesRegex(ValueError, "checkpoint recipe or tokenizer"):
                _validate_evaluation_bindings(data, stage, "branch-a", metrics)


if __name__ == "__main__":
    unittest.main()
