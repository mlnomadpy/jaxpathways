"""Independent small-array oracles for the local C02 RoPE decoder."""

import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from config import ModelConfig
import model


def _np(value):
    return np.asarray(value, dtype=np.float64)


def _norm(x, scale, bias):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + 1e-5) * scale + bias


def _rope(x, positions, base):
    batch, heads, length, width = x.shape
    result = np.empty_like(x)
    for index in range(width // 2):
        frequency = base ** (-2 * index / width)
        angle = positions[:, None, :] * frequency
        even = x[..., 2 * index]
        odd = x[..., 2 * index + 1]
        result[..., 2 * index] = even * np.cos(angle) - odd * np.sin(angle)
        result[..., 2 * index + 1] = even * np.sin(angle) + odd * np.cos(angle)
    return result


def _heads(x, config):
    batch, length, _ = x.shape
    return x.reshape(
        batch, length, config.heads, config.width // config.heads
    ).transpose(0, 2, 1, 3)


def _join(x):
    batch, heads, length, width = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, length, heads * width)


def numpy_reference(params, tokens, config, valid_mask, segments, positions):
    """Independent NumPy implementation of the stated forward equations."""
    p = jax.tree.map(_np, params)
    x = p["token_embed"][tokens]
    batch, length = tokens.shape
    causal = np.arange(length)[:, None] >= np.arange(length)[None, :]
    allowed = (
        causal[None, None]
        & valid_mask[:, None, :, None]
        & valid_mask[:, None, None, :]
        & (segments[:, None, :, None] == segments[:, None, None, :])
        & (segments[:, None, :, None] > 0)
    )
    for block in p["blocks"]:
        h = _norm(x, block["norm1_scale"], block["norm1_bias"])
        q = _rope(_heads(h @ block["q_proj"], config), positions, config.rope_base)
        k = _rope(_heads(h @ block["k_proj"], config), positions, config.rope_base)
        v = _heads(h @ block["v_proj"], config)
        scores = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(q.shape[-1])
        weights = np.where(allowed, scores, -1e30)
        weights = np.exp(weights - np.max(weights, axis=-1, keepdims=True))
        weights = np.where(allowed, weights, 0.0)
        denominator = weights.sum(axis=-1, keepdims=True)
        weights = weights / np.where(denominator == 0, 1.0, denominator)
        x = x + _join(np.einsum("bhqk,bhkd->bhqd", weights, v)) @ block["o_proj"]
        h = _norm(x, block["norm2_scale"], block["norm2_bias"])
        preactivation = h @ block["up"]
        gelu = (
            0.5
            * preactivation
            * (
                1.0
                + np.tanh(
                    np.sqrt(2.0 / np.pi) * (preactivation + 0.044715 * preactivation**3)
                )
            )
        )
        x = x + gelu @ block["down"]
    return _norm(x, p["final_scale"], p["final_bias"]) @ p["token_embed"].T


class DecoderMathTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            vocab_size=24,
            context=16,
            width=8,
            layers=2,
            heads=2,
            mlp_width=12,
            dropout=0.0,
        )
        self.params = model.initialize(jax.random.PRNGKey(37), self.config)

    def test_actual_tree_matches_rope_parameter_formula_and_independent_forward(self):
        actual = sum(np.prod(leaf.shape) for leaf in jax.tree.leaves(self.params))
        expected = 24 * 8 + 2 * (4 * 8 * 8 + 2 * 8 * 12 + 4 * 8) + 2 * 8
        self.assertEqual(model.parameter_count(self.config), expected)
        self.assertEqual(actual, expected)
        tokens = np.asarray([[1, 4, 5, 6]], dtype=np.int32)
        valid = np.ones_like(tokens, dtype=bool)
        segments = np.ones_like(tokens, dtype=np.int32)
        positions = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
        actual_logits = model.logits(
            self.params,
            tokens,
            self.config,
            valid_mask=valid,
            segment_ids=segments,
            position_ids=positions,
        )
        expected_logits = numpy_reference(
            self.params, tokens, self.config, valid, segments, positions
        )
        np.testing.assert_allclose(actual_logits, expected_logits, rtol=2e-5, atol=2e-5)

    def test_target_architecture_count_uses_abstract_initialized_tree(self):
        target = ModelConfig(
            vocab_size=49152,
            context=2048,
            width=1024,
            layers=24,
            heads=16,
            mlp_width=3072,
            dropout=0.1,
        )
        abstract_params = jax.eval_shape(
            lambda key: model.initialize(key, target), jax.random.PRNGKey(3)
        )
        abstract_count = sum(
            int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(abstract_params)
        )
        self.assertEqual(abstract_count, 302_090_240)
        self.assertEqual(model.parameter_count(target), abstract_count)

    def test_future_and_other_document_tokens_cannot_change_current_logits(self):
        tokens = jnp.asarray([[1, 4, 5, 6, 7]], dtype=jnp.int32)
        segments = jnp.asarray([[1, 1, 2, 2, 2]], dtype=jnp.int32)
        positions = jnp.asarray([[0, 1, 0, 1, 2]], dtype=jnp.int32)
        baseline = model.logits(
            self.params,
            tokens,
            self.config,
            segment_ids=segments,
            position_ids=positions,
        )
        future_changed = tokens.at[0, 4].set(8)
        changed_future = model.logits(
            self.params,
            future_changed,
            self.config,
            segment_ids=segments,
            position_ids=positions,
        )
        np.testing.assert_allclose(baseline[:, :4], changed_future[:, :4], atol=1e-6)
        earlier_doc_changed = tokens.at[0, :2].set(jnp.asarray([9, 10]))
        changed_prior = model.logits(
            self.params,
            earlier_doc_changed,
            self.config,
            segment_ids=segments,
            position_ids=positions,
        )
        np.testing.assert_allclose(baseline[:, 2:], changed_prior[:, 2:], atol=1e-6)

    def test_shifted_target_mask_excludes_padding_and_document_transition(self):
        tokens = jnp.asarray([[1, 4, 2, 1, 5, 2, 0]], dtype=jnp.int32)
        segments = jnp.asarray([[1, 1, 1, 2, 2, 2, 0]], dtype=jnp.int32)
        _, targets, _, mask = model.shifted_causal_targets(tokens, segments)
        self.assertEqual(targets.tolist(), [[4, 2, 1, 5, 2, 0]])
        self.assertEqual(mask.tolist(), [[True, True, False, True, True, False]])
        numerator, count = model.token_loss(self.params, tokens, segments, self.config)
        self.assertTrue(np.isfinite(float(numerator)))
        self.assertEqual(float(count), 4.0)

    def test_all_padding_is_finite_and_has_zero_loss_count(self):
        tokens = jnp.zeros((1, 4), dtype=jnp.int32)
        segments = jnp.zeros_like(tokens)
        output = model.logits(self.params, tokens, self.config)
        numerator, count = model.token_loss(self.params, tokens, segments, self.config)
        self.assertTrue(bool(jnp.all(jnp.isfinite(output))))
        self.assertEqual(float(numerator), 0.0)
        self.assertEqual(float(count), 0.0)

    def test_cached_prefill_and_each_decode_step_match_full_logits(self):
        prompt = [1, 4, 5]
        _, cache = model.prefill(self.params, prompt, self.config)
        for appended in (6, 7, 8):
            full = model.logits(
                self.params,
                jnp.asarray([prompt + [appended]], dtype=jnp.int32),
                self.config,
            )[0, -1]
            cached, cache = model.decode_step(self.params, appended, cache, self.config)
            np.testing.assert_allclose(cached[0], full, rtol=1e-5, atol=1e-5)
            prompt.append(appended)
        self.assertEqual(len(cache["keys"]), self.config.layers)
        with self.assertRaisesRegex(ValueError, "different model configuration"):
            model.decode_step(
                self.params,
                9,
                cache,
                self.config.__class__(
                    context=8,
                    vocab_size=24,
                    width=8,
                    layers=2,
                    heads=2,
                    mlp_width=12,
                    dropout=0.0,
                ),
            )

    def test_changed_batch_prompts_match_independent_single_prompt_calls(self):
        prompts = np.asarray([[1, 4, 5], [1, 6, 7]], dtype=np.int32)
        _, cache = model.prefill(self.params, prompts, self.config)
        batched, cache = model.decode_step(
            self.params, np.asarray([8, 9]), cache, self.config
        )
        for index, prompt in enumerate(prompts):
            single, single_cache = model.prefill(
                self.params, prompt.tolist(), self.config
            )
            single, _ = model.decode_step(
                self.params, 8 + index, single_cache, self.config
            )
            np.testing.assert_allclose(batched[index], single[0], rtol=1e-5, atol=1e-5)
        self.assertEqual(cache["position"], 4)

    def test_qv_lora_zero_delta_merged_parity_and_changed_delta(self):
        adapter = model.initialize_lora(jax.random.PRNGKey(41), self.config)
        adapter_count = sum(np.prod(leaf.shape) for leaf in jax.tree.leaves(adapter))
        self.assertEqual(adapter_count, model.lora_parameter_count(self.config))
        self.assertEqual(adapter_count, 2 * 2 * 4 * (8 + 8))
        tokens = jnp.asarray([[1, 4, 5, 6]], dtype=jnp.int32)
        base_logits = model.logits(self.params, tokens, self.config)
        adapter_logits = model.logits(
            self.params, tokens, self.config, adapters=adapter
        )
        merged = model.merge_lora(self.params, adapter, self.config)
        merged_logits = model.logits(merged, tokens, self.config)
        np.testing.assert_array_equal(adapter_logits, base_logits)
        np.testing.assert_array_equal(merged_logits, base_logits)
        changed_layers = []
        for layer in adapter["blocks"]:
            changed_layers.append(
                {
                    name: {
                        "A": values["A"],
                        "B": jnp.full_like(values["B"], 0.03),
                    }
                    for name, values in layer.items()
                }
            )
        changed_adapter = {"blocks": tuple(changed_layers)}
        changed_logits = model.logits(
            self.params, tokens, self.config, adapters=changed_adapter
        )
        changed_merged = model.merge_lora(self.params, changed_adapter, self.config)
        changed_merged_logits = model.logits(changed_merged, tokens, self.config)
        self.assertGreater(float(jnp.max(jnp.abs(changed_logits - base_logits))), 1e-8)
        np.testing.assert_allclose(
            changed_logits, changed_merged_logits, rtol=1e-5, atol=1e-6
        )

    def test_nonzero_lora_uses_frozen_alpha_over_rank_in_forward_merge_and_cache(self):
        adapter = model.initialize_lora(jax.random.PRNGKey(53), self.config)
        layers = []
        expected_blocks = []
        legacy_blocks = []
        for base, adapter_layer in zip(
            self.params["blocks"], adapter["blocks"], strict=True
        ):
            adapter_values = {}
            expected = dict(base)
            legacy = dict(base)
            for name in ("q_proj", "v_proj"):
                a = jnp.sin(
                    jnp.arange(
                        adapter_layer[name]["A"].size, dtype=jnp.float32
                    ).reshape(adapter_layer[name]["A"].shape)
                    * 0.7
                )
                b = 0.8 * jnp.cos(
                    jnp.arange(
                        adapter_layer[name]["B"].size, dtype=jnp.float32
                    ).reshape(adapter_layer[name]["B"].shape)
                    * 0.43
                )
                adapter_values[name] = {"A": a, "B": b}
                delta = a @ b
                expected[name] = base[name] + 2.0 * delta
                legacy[name] = base[name] + 0.5 * delta
            layers.append(adapter_values)
            expected_blocks.append(expected)
            legacy_blocks.append(legacy)
        changed = {"blocks": tuple(layers)}
        independent = {**self.params, "blocks": tuple(expected_blocks)}
        wrong_scale = {**self.params, "blocks": tuple(legacy_blocks)}
        tokens = jnp.asarray([[1, 4, 5, 6]], dtype=jnp.int32)
        adapted = model.logits(self.params, tokens, self.config, adapters=changed)
        expected = model.logits(independent, tokens, self.config)
        wrong = model.logits(wrong_scale, tokens, self.config)
        np.testing.assert_allclose(adapted, expected, rtol=1e-5, atol=1e-6)
        self.assertGreater(float(jnp.max(jnp.abs(adapted - wrong))), 1e-5)
        merged = model.merge_lora(self.params, changed, self.config)
        np.testing.assert_allclose(
            model.logits(merged, tokens, self.config), expected, rtol=1e-5, atol=1e-6
        )
        next_logits, cache = model.prefill(
            self.params, [1, 4, 5], self.config, adapters=changed
        )
        full_next = model.logits(
            self.params,
            jnp.asarray([[1, 4, 5]], dtype=jnp.int32),
            self.config,
            adapters=changed,
        )[:, -1]
        np.testing.assert_allclose(next_logits, full_next, rtol=1e-5, atol=1e-6)
        decoded, _ = model.decode_step(
            self.params, 6, cache, self.config, adapters=changed
        )
        full_after_decode = model.logits(
            self.params,
            jnp.asarray([[1, 4, 5, 6]], dtype=jnp.int32),
            self.config,
            adapters=changed,
        )[:, -1]
        np.testing.assert_allclose(decoded, full_after_decode, rtol=1e-5, atol=1e-6)

    def test_cache_binds_params_and_adapter_and_later_layer_corruption(self):
        prompt = [1, 4, 5]
        adapter = model.initialize_lora(jax.random.PRNGKey(47), self.config)
        _, cache = model.prefill(self.params, prompt, self.config, adapters=adapter)
        changed_params = {
            **self.params,
            "token_embed": self.params["token_embed"].at[4, 0].add(0.1),
        }
        with self.assertRaisesRegex(ValueError, "different model configuration"):
            model.decode_step(changed_params, 6, cache, self.config, adapters=adapter)
        changed_adapter = model.initialize_lora(jax.random.PRNGKey(48), self.config)
        with self.assertRaisesRegex(ValueError, "different model configuration"):
            model.decode_step(
                self.params, 6, cache, self.config, adapters=changed_adapter
            )
        base_logits, base_cache = model.prefill(self.params, prompt, self.config)
        expected, _ = model.decode_step(self.params, 6, base_cache, self.config)
        corrupted = model.corrupt_cache_layer(base_cache, 1, 0.7)
        wrong, _ = model.decode_step(self.params, 6, corrupted, self.config)
        self.assertGreater(float(jnp.max(jnp.abs(expected - wrong))), 1e-8)

    def test_gradient_matches_finite_difference_on_embedding_leaf(self):
        tokens = jnp.asarray([[1, 4, 5, 2]], dtype=jnp.int32)
        segments = jnp.ones_like(tokens)

        def objective(params):
            numerator, count = model.token_loss(params, tokens, segments, self.config)
            return numerator / count

        gradient = jax.grad(objective)(self.params)["token_embed"]
        index = (4, 2)
        epsilon = 1e-3
        plus = self.params["token_embed"].at[index].add(epsilon)
        minus = self.params["token_embed"].at[index].add(-epsilon)
        finite_difference = (
            float(objective({**self.params, "token_embed": plus}))
            - float(objective({**self.params, "token_embed": minus}))
        ) / (2 * epsilon)
        self.assertAlmostEqual(float(gradient[index]), finite_difference, delta=3e-3)


if __name__ == "__main__":
    unittest.main()
