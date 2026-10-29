"""Focused AdamW and fresh-process full-state recovery contracts."""

from __future__ import annotations

import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import model
from config import ModelConfig, TrainConfig
from training import (
    load_checkpoint,
    make_state,
    save_checkpoint,
    schedule_rate,
    update_from_batches,
)


def fixture():
    config = ModelConfig(
        vocab_size=16, context=8, width=8, layers=2, heads=2, mlp_width=16, dropout=0.1
    )
    train_config = TrainConfig(
        batch_size=2,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_steps=1,
        total_steps=2,
        seed=17,
    )
    params = model.initialize(jax.random.PRNGKey(7), config)
    state = make_state(params, None, jax.random.PRNGKey(19), [0, 1, 2, 3])
    state["step"] = 1
    state["schedule_step"] = 1
    state["cursor"] = 2
    state["seen_ids"] = ["a", "b"]
    identity = {
        "stage": "recovery-test",
        "recipe_sha256": "fixed-recipe",
        "processor_sha256": "fixed-processor",
        "candidate_source_sha256": "fixed-candidate",
        "parent_checkpoint_sha256": "fixed-parent",
        "schedule": {"updates": 2, "dropout": 0.1},
        "schedule_sample_ids": ["a", "b", "c", "d"],
        "schedule_offsets": [0, 2, 4],
        "model_config": config.as_dict(),
        "train_config": train_config.as_dict(),
        "adapter_spec": None,
    }
    batch = {
        "input_ids": jnp.asarray(
            [[1, 4, 5, 6, 0, 0, 0], [1, 7, 8, 9, 10, 0, 0]], dtype=jnp.int32
        ),
        "target_ids": jnp.asarray(
            [[4, 5, 6, 2, 0, 0, 0], [7, 8, 9, 10, 2, 0, 0]], dtype=jnp.int32
        ),
        "target_mask": jnp.asarray(
            [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]], dtype=jnp.bool_
        ),
        "segment_ids": jnp.asarray(
            [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]], dtype=jnp.int32
        ),
        "position_ids": jnp.asarray(
            [[0, 1, 2, 3, 0, 0, 0], [0, 1, 2, 3, 4, 0, 0]], dtype=jnp.int32
        ),
    }
    return config, train_config, state, identity, batch


def tree_equal(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            tree_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if left is None:
        return right is None
    if isinstance(left, list):
        return left == right
    if isinstance(left, (int, float, str, bool)):
        return left == right
    return np.array_equal(np.asarray(left), np.asarray(right))


class RecoveryTests(unittest.TestCase):
    def test_cosine_scheduler_uses_declared_minimum_rate(self):
        at_floor = schedule_rate(1e-3, step=1, total=2, warmup=1, minimum_rate=3e-4)
        frozen_floor = schedule_rate(1e-3, step=1, total=2, warmup=1, minimum_rate=1e-4)
        self.assertAlmostEqual(at_floor, 3e-4)
        self.assertAlmostEqual(frozen_floor, 1e-4)
        self.assertNotEqual(at_floor, frozen_floor)

    def test_fresh_process_restores_dropout_adam_and_next_update_exactly(self):
        config, train_config, state, identity, batch = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, identity_path, child_path = (
                root / "before.npz",
                root / "identity.json",
                root / "child.npz",
            )
            identity_path.write_text(json.dumps(identity))
            save_checkpoint(checkpoint, state, identity)
            expected, _ = update_from_batches(
                state, [batch], config, train_config, 1e-3
            )
            expected["cursor"] = 4
            expected["seen_ids"].extend(["c", "d"])
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT / "tests" / "resume_worker.py"),
                    str(checkpoint),
                    str(identity_path),
                    str(child_path),
                ],
                check=True,
                env={**os.environ, "JAX_PLATFORMS": "cpu"},
                capture_output=True,
                text=True,
            )
            restored = load_checkpoint(
                child_path, identity, expected_order=[0, 1, 2, 3]
            )
            self.assertTrue(tree_equal(expected, restored))

    def test_rejects_changed_candidate_or_schedule_identity(self):
        _, _, state, identity, _ = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.npz"
            save_checkpoint(path, state, identity)
            altered = {**identity, "candidate_source_sha256": "different"}
            with self.assertRaisesRegex(ValueError, "identity"):
                load_checkpoint(path, altered)

    def test_rejects_self_consistent_but_semantically_invalid_optimizer_metadata(self):
        _, _, state, identity, _ = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary) / "original.npz"
            save_checkpoint(original, state, identity)
            controls = (
                ("integer-moment", self._integer_first_moment, "first moment"),
                ("negative-variance", self._negative_second_moment, "nonnegative"),
                ("negative-epoch", self._negative_epoch, "epoch"),
                ("past-horizon", self._past_horizon, "horizon"),
                ("cursor-early", self._cursor_early, "update boundary"),
                ("cursor-late", self._cursor_late, "update boundary"),
            )
            for name, mutate, message in controls:
                path = Path(temporary) / f"{name}.npz"
                path.write_bytes(mutate(original.read_bytes()))
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, message
                ):
                    load_checkpoint(path, identity)

    @staticmethod
    def _rewrite(payload, transform):
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        meta = json.loads(arrays["__meta__"].tobytes())
        transform(arrays, meta)
        arrays["__meta__"] = np.frombuffer(
            json.dumps(meta, sort_keys=True, separators=(",", ":")).encode(),
            dtype=np.uint8,
        )
        output = io.BytesIO()
        np.savez_compressed(output, **arrays)
        return output.getvalue()

    @classmethod
    def _integer_first_moment(cls, payload):
        def mutate(arrays, meta):
            name = cls._first_array_name(meta["tree_specs"]["first"])
            arrays[name] = arrays[name].astype(np.int32)
            cls._change_declared_dtype(meta["tree_specs"]["first"], name, "int32")

        return cls._rewrite(payload, mutate)

    @classmethod
    def _negative_second_moment(cls, payload):
        def mutate(arrays, meta):
            name = cls._first_array_name(meta["tree_specs"]["second"])
            arrays[name].flat[0] = -1.0

        return cls._rewrite(payload, mutate)

    @classmethod
    def _negative_epoch(cls, payload):
        return cls._rewrite(payload, lambda arrays, meta: meta.__setitem__("epoch", -1))

    @classmethod
    def _past_horizon(cls, payload):
        def mutate(arrays, meta):
            meta["step"] = meta["schedule_step"] = 9999

        return cls._rewrite(payload, mutate)

    @classmethod
    def _cursor_early(cls, payload):
        def mutate(arrays, meta):
            meta["cursor"] = 1
            meta["seen_ids"] = ["a"]

        return cls._rewrite(payload, mutate)

    @classmethod
    def _cursor_late(cls, payload):
        def mutate(arrays, meta):
            meta["cursor"] = 4
            meta["seen_ids"] = ["a", "b", "c", "d"]

        return cls._rewrite(payload, mutate)

    @staticmethod
    def _first_array_name(spec):
        if "__array__" in spec:
            return spec["__array__"]
        children = spec.get("items", []) if "__sequence__" in spec else spec.values()
        for child in children:
            if isinstance(child, dict):
                try:
                    return RecoveryTests._first_array_name(child)
                except StopIteration:
                    pass
        raise StopIteration

    @staticmethod
    def _change_declared_dtype(spec, name, dtype):
        if spec.get("__array__") == name:
            spec["dtype"] = dtype
            return True
        children = spec.get("items", []) if "__sequence__" in spec else spec.values()
        return any(
            RecoveryTests._change_declared_dtype(child, name, dtype)
            for child in children
            if isinstance(child, dict)
        )


if __name__ == "__main__":
    unittest.main()
