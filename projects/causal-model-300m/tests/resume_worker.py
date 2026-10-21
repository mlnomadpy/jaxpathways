"""One-update worker used to verify a truly fresh-process checkpoint replay."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import model
from config import ModelConfig, TrainConfig
from training import load_checkpoint, save_checkpoint, update_from_batches


def main():
    checkpoint, identity_path, output = map(Path, sys.argv[1:4])
    identity = json.loads(identity_path.read_text())
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
    state = load_checkpoint(checkpoint, identity, expected_order=[0, 1, 2, 3])
    state, metrics = update_from_batches(state, [batch], config, train_config, 1e-3)
    state["cursor"] += 2
    state["seen_ids"].extend(["c", "d"])
    save_checkpoint(output, state, identity)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
