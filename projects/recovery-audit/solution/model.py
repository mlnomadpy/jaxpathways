"""Reference solution for Verify checkpoint integrity, NaN guards, and deterministic replay (recovery-audit)."""
import jax
import jax.numpy as jnp
import numpy as np


def init_params(seed: int = 0, d_model: int = 16, d_out: int = 8) -> dict:
    """Stage 1: Initialize deterministic model parameters with explicit PRNG keys."""
    # Create explicit PRNG key and split into independent parameter streams.
    key = jax.random.key(seed)
    k_w, k_p = jax.random.split(key)
    scale = 1.0 / jnp.sqrt(jnp.float32(d_model))
    w_enc = jax.random.normal(k_w, (d_model, d_out), dtype=jnp.float32) * scale
    b_enc = jnp.zeros((d_out,), dtype=jnp.float32)
    proj = jax.random.normal(k_p, (d_out, d_out), dtype=jnp.float32) * (1.0 / jnp.sqrt(jnp.float32(d_out)))
    return {
        "w_enc": w_enc,
        "b_enc": b_enc,
        "proj": proj,
        "temperature": jnp.float32(0.07),
    }


def forward_batch(params: dict, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> dict:
    """Stage 2: Compute L2-normalized representations and masked projections."""
    # Guard input tensor rank and feature dimension contract.
    if x.ndim != 2 or x.shape[1] != params["w_enc"].shape[0]:
        raise ValueError(f"Expected 2D input with feature dim {params['w_enc'].shape[0]}, got {x.shape}")
    raw = jnp.tanh(x @ params["w_enc"] + params["b_enc"]) @ params["proj"]
    if mask is not None:
        if mask.shape != (x.shape[0],):
            raise ValueError("Mask must have shape (B,)")
        raw = jnp.where(mask[:, None], raw, 0.0)
    norms = jnp.linalg.norm(raw, axis=-1, keepdims=True)
    embeddings = raw / jnp.maximum(norms, 1e-6)
    if mask is not None:
        embeddings = jnp.where(mask[:, None], embeddings, 0.0)
    return {
        "raw": raw,
        "embeddings": embeddings,
        "norms": jnp.squeeze(norms, axis=-1),
    }


def _objective(params: dict, x: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute mean squared alignment error between normalized embeddings and targets."""
    if x.shape[0] != targets.shape[0] or targets.shape[1] != params["w_enc"].shape[1]:
        raise ValueError("Targets shape must match batch output shape (B, d_out)")
    out = forward_batch(params, x)["embeddings"]
    target_norms = jnp.maximum(jnp.linalg.norm(targets, axis=-1, keepdims=True), 1e-6)
    unit_targets = targets / target_norms
    return jnp.mean((out - unit_targets) ** 2)


def compute_loss_and_grads(params: dict, x: jnp.ndarray, targets: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    """Stage 3: Evaluate the scalar training objective and parameter gradients via jax.value_and_grad."""
    return jax.value_and_grad(_objective)(params, x, targets)


def qualification_report(seed: int = 0, steps: int = 25, lr: float = 0.1) -> dict:
    """Stage 4: Execute a compiled scan training loop and return a verification report."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    params = init_params(seed=seed, d_model=16, d_out=8)
    data_key = jax.random.key(seed + 101)
    kx, kt, kh = jax.random.split(data_key, 3)
    x_train = jax.random.normal(kx, (12, 16), dtype=jnp.float32)
    true_w = jax.random.normal(kt, (16, 8), dtype=jnp.float32) / 4.0
    y_train = x_train @ true_w
    x_eval = jax.random.normal(kh, (6, 16), dtype=jnp.float32)
    y_eval = x_eval @ true_w

    def step_fn(carry, _):
        loss_val, grads = jax.value_and_grad(_objective)(carry, x_train, y_train)
        next_carry = jax.tree.map(lambda p, g: p - lr * g, carry, grads)
        return next_carry, loss_val

    fitted, history = jax.lax.scan(step_fn, params, None, length=steps)
    eval_loss = float(_objective(fitted, x_eval, y_eval))
    return {
        "project": "recovery-audit",
        "initial_loss": float(history[0]),
        "final_loss": float(history[-1]),
        "eval_loss": eval_loss,
        "finite": bool(jnp.all(jnp.isfinite(history))),
        "steps": int(steps),
        "backend": jax.default_backend(),
    }
