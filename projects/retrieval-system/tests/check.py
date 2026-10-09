"""Verification runner for Exact & Partitioned Vector Retrieval Service (retrieval-system)."""
import argparse
import importlib.util
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(implementation: str):
    if implementation in ("starter", "solution"):
        path = ROOT / implementation / "model.py"
    else:
        path = Path(implementation).resolve()
    spec = importlib.util.spec_from_file_location("capstone_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_stage_1(mod):
    params = mod.init_params(seed=7, d_model=16, d_out=8)
    assert set(params.keys()) >= {"w_enc", "b_enc", "temperature"}
    assert params["w_enc"].shape == (16, 8)
    assert params["b_enc"].shape == (8,)
    print("PASS stage 1: parameter initialization and shape contract (retrieval-system)")


def check_stage_2(mod):
    params = mod.init_params(seed=7, d_model=16, d_out=8)
    x = jnp.ones((4, 16), dtype=jnp.float32)
    mask = jnp.array([True, True, False, True])
    out = mod.forward_batch(params, x, mask=mask)
    assert out["embeddings"].shape == (4, 8)
    assert np.allclose(np.asarray(out["embeddings"][2]), 0.0, atol=1e-6)
    norms = np.linalg.norm(np.asarray(out["embeddings"])[[0, 1, 3]], axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    print("PASS stage 2: masked forward representation and unit normalization (retrieval-system)")


def check_stage_3(mod):
    params = mod.init_params(seed=7, d_model=16, d_out=8)
    x = jax.random.normal(jax.random.key(3), (5, 16), dtype=jnp.float32)
    targets = jax.random.normal(jax.random.key(4), (5, 8), dtype=jnp.float32)
    loss_val, grads = mod.compute_loss_and_grads(params, x, targets)
    assert jnp.isfinite(loss_val)
    assert grads["w_enc"].shape == (16, 8)
    eps = 1e-3
    plus = {**params, "b_enc": params["b_enc"].at[0].add(eps)}
    minus = {**params, "b_enc": params["b_enc"].at[0].add(-eps)}
    l_plus, _ = mod.compute_loss_and_grads(plus, x, targets)
    l_minus, _ = mod.compute_loss_and_grads(minus, x, targets)
    fd = float((l_plus - l_minus) / (2.0 * eps))
    assert np.allclose(float(grads["b_enc"][0]), fd, atol=1e-3, rtol=1e-2)
    print("PASS stage 3: autodiff vs finite-difference gradient check (retrieval-system)")


def check_stage_4(mod):
    rep = mod.qualification_report(seed=7, steps=25, lr=0.1)
    assert rep["finite"] is True
    assert rep["final_loss"] < rep["initial_loss"]
    assert np.isfinite(rep["eval_loss"])
    print("PASS stage 4: compiled scan training and held-out qualification (retrieval-system)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", default="solution")
    parser.add_argument("--stage", default="all")
    args = parser.parse_args()
    mod = load_module(args.implementation)
    stages = {"1": check_stage_1, "2": check_stage_2, "3": check_stage_3, "4": check_stage_4}
    if args.stage == "all":
        for fn in stages.values():
            fn(mod)
    else:
        stages[args.stage](mod)


if __name__ == "__main__":
    main()
