"""Native harness adapter uses the selected decoder's logits and generation."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import model
from config import ModelConfig
from consumer import LocalCausalModel
from lm_eval_adapter import C02PilotLM


class HarnessAdapterTests(unittest.TestCase):
    def setUp(self):
        config = ModelConfig(
            vocab_size=16,
            context=16,
            width=8,
            layers=2,
            heads=2,
            mlp_width=12,
            dropout=0.0,
        )
        params = model.initialize(jax.random.PRNGKey(77), config)
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        self.loaded = LocalCausalModel(
            Path("."), {"branch": "test"}, config, params, tokenizer
        )
        self.adapter = C02PilotLM.__new__(C02PilotLM)
        from lm_eval.api.model import LM

        LM.__init__(self.adapter)
        self.adapter.loaded = self.loaded
        self.adapter.max_length = config.context
        self.adapter.tokenizer = tokenizer

    def test_loglikelihood_matches_candidate_forward_targets(self):
        context, continuation = [4, 5], [6, 7]
        actual = self.adapter._score(context, continuation)
        logits = model.logits(
            self.loaded.params,
            jnp.asarray([[4, 5, 6]], dtype=jnp.int32),
            self.loaded.config,
        )[0]
        log_probs = jax.nn.log_softmax(logits[1:3], axis=-1)
        expected = float(log_probs[0, 6] + log_probs[1, 7])
        self.assertAlmostEqual(actual[0], expected, places=5)
        self.assertEqual(
            actual[1],
            bool(jnp.all(jnp.argmax(logits[1:3], axis=-1) == jnp.array(continuation))),
        )

    def test_generation_uses_local_consumer_and_applies_until(self):
        self.loaded.generate = Mock(return_value={"text": "answer<STOP>suffix"})
        result = self.adapter.generate_until(
            [
                SimpleNamespace(
                    args=("question", {"until": ["<STOP>"], "max_gen_toks": 7})
                )
            ]
        )
        self.assertEqual(result, ["answer"])
        self.loaded.generate.assert_called_once_with("question", 7)

    def test_context_continuation_pair_scores_byte_bpe_boundary_merge(self):
        tokenizer = Tokenizer(
            models.BPE(
                vocab={"[UNK]": 0, "x": 1, "Ġ": 2, "a": 3, "Ġa": 4},
                merges=[("Ġ", "a")],
                unk_token="[UNK]",
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        self.adapter.tokenizer = tokenizer

        context_ids, target_ids = self.adapter._context_and_target_ids("x ", "a")
        joined = tokenizer.encode("x a", add_special_tokens=False)
        separate = tokenizer.encode("x ", add_special_tokens=False).ids
        self.assertEqual(joined.tokens, ["x", "Ġa"])
        self.assertEqual(separate, [1, 2])
        self.assertEqual(context_ids, [1])
        self.assertEqual(target_ids, [4])

        empty_context, target = self.adapter._context_and_target_ids("", "a")
        self.assertEqual(empty_context, [])
        self.assertEqual(target, [3])

    def test_boundary_offsets_use_unicode_characters_and_preserve_trailing_space(self):
        class Encoded:
            ids = [11, 12]
            offsets = [(0, 1), (1, 3)]

        class CharacterOffsetTokenizer:
            def encode(self, _text, add_special_tokens=False):
                return Encoded()

        self.adapter.tokenizer = CharacterOffsetTokenizer()
        context_ids, target_ids = self.adapter._context_and_target_ids("é", "x ")
        self.assertEqual(context_ids, [11])
        self.assertEqual(target_ids, [12])

    def test_rolling_loglikelihood_scores_every_target_once_across_windows(self):
        self.adapter.max_length = 4
        targets = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        inputs_seen = []

        def candidate_logits(_params, input_ids, config, **_kwargs):
            inputs_seen.append(np.asarray(input_ids).tolist())
            return jnp.zeros((1, input_ids.shape[1], config.vocab_size))

        def candidate_log_probs(logits):
            token_costs = jnp.arange(logits.shape[-1], dtype=jnp.float32)
            return -jnp.broadcast_to(token_costs, logits.shape)

        with (
            patch("lm_eval_adapter.model.logits", side_effect=candidate_logits),
            patch("lm_eval_adapter.jax_log_softmax", side_effect=candidate_log_probs),
            patch.object(self.adapter, "_ids", return_value=targets),
        ):
            score = self.adapter.loglikelihood_rolling(
                [SimpleNamespace(args=("the fixed target stream",))]
            )[0]
        self.assertAlmostEqual(score, -sum(targets), places=5)
        self.assertEqual([len(row[0]) for row in inputs_seen], [4] * 8)
        self.assertEqual(inputs_seen[0][0], [1, 4, 5, 6])
        self.assertEqual(inputs_seen[1][0], [4, 5, 6, 7])
        self.assertEqual(inputs_seen[-1][0], [10, 11, 12, 13])

    def test_singleton_final_tail_uses_all_context_tokens(self):
        self.adapter.max_length = 4
        seen = []

        def context_sensitive_logits(_params, input_ids, config, **_kwargs):
            values = np.asarray(input_ids)
            seen.append(values.tolist())
            logits = np.zeros((1, values.shape[1], config.vocab_size), dtype=np.float32)
            if int(values[0, 0]) == 4:
                logits[0, -1, 9] = 3.0
            return jnp.asarray(logits)

        with patch(
            "lm_eval_adapter.model.logits", side_effect=context_sensitive_logits
        ):
            actual, greedy = self.adapter._score([4, 5, 6, 7], [9, 10])

        first = np.zeros(self.loaded.config.vocab_size, dtype=np.float64)
        first[9] = 3.0
        expected = float(jax.nn.log_softmax(jnp.asarray(first))[9]) - np.log(
            self.loaded.config.vocab_size
        )
        self.assertAlmostEqual(actual, expected, places=5)
        self.assertFalse(greedy)
        self.assertEqual(seen, [[[4, 5, 6, 7]], [[5, 6, 7, 9]]])


if __name__ == "__main__":
    unittest.main()
