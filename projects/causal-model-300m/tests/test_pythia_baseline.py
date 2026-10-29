"""Pythia scoring windows cover each per-record target once."""

from pathlib import Path
import sys
import hashlib
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from evaluate_pythia_baseline import _windows
import evaluate_pythia_baseline
import evaluate_pythia_qa
from prepare import PYTHIA_FILES_SHA256
from pythia_pin import verify_pythia_snapshot


class TinyTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) for character in text]


class PythiaBaselineWindowTests(unittest.TestCase):
    def test_windows_restart_each_record_and_score_each_token_and_eos_once(self):
        records = [("first", "abc"), ("second", "de")]
        windows, byte_count, target_count = _windows(records, TinyTokenizer(), 2)
        self.assertEqual(byte_count, 5)
        self.assertEqual(target_count, 7)
        self.assertEqual([row[0] for row in windows], ["first"] * 2 + ["second"] * 2)
        self.assertEqual(windows[0][1:], ([99, ord("a")], [ord("a"), ord("b")]))
        self.assertEqual(windows[1][1:], ([ord("b"), ord("c")], [ord("c"), 99]))
        self.assertEqual(windows[2][1:], ([99, ord("d")], [ord("d"), ord("e")]))
        self.assertEqual(windows[3][1:], ([ord("e")], [99]))

    def test_snapshot_verifier_requires_all_seven_exact_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {}
            for index, relative in enumerate(sorted(PYTHIA_FILES_SHA256)):
                payload = f"pinned-fixture-{index}".encode()
                (root / relative).write_bytes(payload)
                expected[relative] = hashlib.sha256(payload).hexdigest()
            self.assertEqual(verify_pythia_snapshot(root, expected), expected)

            for relative in expected:
                with self.subTest(relative=relative):
                    path = root / relative
                    original = path.read_bytes()
                    path.write_bytes(original + b" altered")
                    with self.assertRaisesRegex(ValueError, "hash mismatch"):
                        verify_pythia_snapshot(root, expected)
                    path.write_bytes(original)
                    path.unlink()
                    with self.assertRaisesRegex(ValueError, "missing"):
                        verify_pythia_snapshot(root, expected)
                    path.write_bytes(original)

    def test_both_scorers_reject_a_bad_snapshot_before_any_loader_call(self):
        missing_data = Path("/missing/c02-data")
        missing_model = Path("/missing/pythia")
        with (
            patch.object(
                evaluate_pythia_baseline,
                "snapshot_identity",
                side_effect=ValueError("bad frozen pin"),
            ),
            patch.object(
                evaluate_pythia_baseline.AutoTokenizer,
                "from_pretrained",
            ) as tokenizer_loader,
            patch.object(
                evaluate_pythia_baseline.AutoModelForCausalLM,
                "from_pretrained",
            ) as model_loader,
        ):
            with self.assertRaisesRegex(ValueError, "bad frozen pin"):
                evaluate_pythia_baseline.evaluate(
                    missing_data, missing_model, Path("/tmp/unused.json")
                )
            tokenizer_loader.assert_not_called()
            model_loader.assert_not_called()

        with (
            patch.object(
                evaluate_pythia_qa,
                "snapshot_identity",
                side_effect=ValueError("bad frozen pin"),
            ),
            patch.object(
                evaluate_pythia_qa.AutoTokenizer,
                "from_pretrained",
            ) as tokenizer_loader,
            patch.object(
                evaluate_pythia_qa.AutoModelForCausalLM,
                "from_pretrained",
            ) as model_loader,
        ):
            with self.assertRaisesRegex(ValueError, "bad frozen pin"):
                evaluate_pythia_qa.evaluate(
                    missing_data, missing_model, Path("/tmp/unused.json")
                )
            tokenizer_loader.assert_not_called()
            model_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
