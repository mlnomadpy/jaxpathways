"""Focused contracts for answer-only likelihood and train-only byte baseline."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "data"))

from evaluate_qa_diagnostics import _byte_unigram, _response_batch


class Encoded:
    def __init__(self, ids):
        self.ids = ids


class SmallTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text.startswith("Question:"):
            question = text.splitlines()[0].removeprefix("Question: ")
            return Encoded([int(item) for item in question.split(",")])
        return Encoded([int(item) for item in text.split(",") if item])


class QADiagnosticTests(unittest.TestCase):
    def test_response_mask_excludes_prompt_and_keeps_answer_eos(self):
        tokenizer = SmallTokenizer()
        rows = [
            {
                "question_id": "fits",
                "question": "2,3,4",
                "context": "",
                "answers": ["5,6"],
            },
            {
                "question_id": "too-long",
                "question": "1,2,3,4,5,6,7,8",
                "context": "",
                "answers": ["9"],
            },
        ]
        inputs, targets, masks, kept, excluded = _response_batch(rows, tokenizer, 8)
        # The prompt renderer includes the context and labels; this fake tokenizer
        # maps its text fields to comma-separated IDs and the answer IDs directly.
        self.assertEqual(kept, ["fits"])
        self.assertEqual(excluded, ["too-long"])
        self.assertEqual(len(inputs[0]), len(targets[0]))
        self.assertEqual(masks[0], [False, False, False, True, True, True])

    def test_train_only_byte_unigram_reports_exact_shared_byte_denominator(self):
        result = _byte_unigram(
            [{"text": "ab"}],
            [{"text": "ab"}, {"text": "c"}],
            "tiny heldout fixture",
        )
        self.assertEqual(result["normalized_utf8_bytes"], 3)
        self.assertEqual(result["training_bytes"], 2)
        self.assertEqual(
            result["training_source"].split(";")[0], "base-train.jsonl only"
        )
        self.assertTrue(np.isfinite(result["nll_bits_per_normalized_utf8_byte"]))


if __name__ == "__main__":
    unittest.main()
