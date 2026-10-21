"""SQuAD scoring and batched RoPE generation parity controls."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import jax

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import model
from config import ModelConfig
from evaluate_pilot import (
    generate_with_candidate,
    normalize_answer,
    score_answer,
    token_f1,
)


class EvaluationContractTests(unittest.TestCase):
    def test_squad_normalization_empty_answers_and_multiple_gold_answers(self):
        self.assertEqual(normalize_answer("The U.S.A., and a test!"), "usa and test")
        self.assertEqual(token_f1("red red blue", "blue red"), 0.8)
        self.assertEqual(
            score_answer("", ["", "an answer"]), {"exact_match": 1.0, "token_f1": 0.0}
        )
        self.assertEqual(
            score_answer("the answer", ["wrong", "an answer"])["exact_match"], 1.0
        )

    def test_evaluator_generation_uses_candidate_cached_generation_api(self):
        config = ModelConfig(
            vocab_size=32,
            context=16,
            width=8,
            layers=2,
            heads=2,
            mlp_width=12,
            dropout=0.0,
        )
        params = model.initialize(jax.random.PRNGKey(123), config)
        sentinel = [9, 10, 2]
        with patch.object(
            model, "generate", return_value=sentinel
        ) as selected_candidate:
            actual = generate_with_candidate(params, [1, 4, 5], config, 8)
        self.assertEqual(actual, sentinel)
        selected_candidate.assert_called_once_with(
            params, [1, 4, 5], config, max_new_tokens=8, eos_token_id=2
        )


if __name__ == "__main__":
    unittest.main()
