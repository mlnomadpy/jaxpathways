"""Check portability of the retained physical-TPU checkpoint on one CPU."""

import hashlib
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "checkpoint-files.json").read_text())
for filename, expected in manifest["publishedFiles"].items():
    assert (
        hashlib.sha256((ROOT / "checkpoint-step-4" / filename).read_bytes()).hexdigest()
        == expected
    )

sharding = NamedSharding(Mesh(np.asarray(jax.devices("cpu")[:1]), ("data",)), P())
target = {
    "weights": jax.ShapeDtypeStruct((8,), jnp.float32, sharding=sharding),
    "momentum": jax.ShapeDtypeStruct((8,), jnp.float32, sharding=sharding),
    "step": jax.ShapeDtypeStruct((), jnp.int32, sharding=sharding),
}
with ocp.StandardCheckpointer() as checkpointer:
    state = checkpointer.restore(ROOT / "checkpoint-step-4", target=target)

# Independently evaluate the documented fixture and four momentum updates.
x = np.random.default_rng(173).normal(size=(64, 8)).astype(np.float32)
truth = np.linspace(-0.3, 0.4, 8, dtype=np.float32)
y = x @ truth + np.random.default_rng(991).normal(0, 0.01, 64).astype(np.float32)
weights = np.zeros(8, np.float32)
momentum = np.zeros(8, np.float32)
for _ in range(4):
    gradient = np.float32(2 / 64) * (x.T @ (x @ weights - y))
    momentum = np.float32(0.8) * momentum + gradient
    weights = weights - np.float32(0.04) * momentum
assert int(state["step"]) == 4
np.testing.assert_allclose(state["weights"], weights, rtol=3e-5, atol=3e-6)
np.testing.assert_allclose(state["momentum"], momentum, rtol=3e-5, atol=3e-6)
recorded = json.loads((ROOT / "checkpoint-cpu-restore.json").read_text())
for name, array in state.items():
    np.testing.assert_array_equal(np.asarray(array), recorded["state"][name])
print("PASS: retained TPU checkpoint restored on CPU; step, weights and momentum match")
