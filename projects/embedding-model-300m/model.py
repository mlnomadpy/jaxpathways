"""Pre-LN bidirectional RoPE Transformer and tied masked-language head.

Adapted from the explicit JAX parameter-tree and independent-shape-check pattern
in ``projects/text-harness/foundations/model.py``. The attention mask and
training objective here are a new bidirectional encoder implementation.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

try:
    from .config import EncoderConfig
except ImportError:
    from config import EncoderConfig


def _config(config):
    cfg = config if isinstance(config, EncoderConfig) else EncoderConfig(**config)
    cfg.validate()
    return cfg


def _is_traced(value):
    return isinstance(value, jax.core.Tracer)


def _normal(key, shape, scale, dtype=jnp.float32):
    return jax.random.normal(key, shape, dtype=dtype) * scale


def initialize(key, config):
    """Initialize a true two-matrix GELU Transformer with tied MLM embeddings."""
    cfg = _config(config)
    keys = iter(jax.random.split(key, 3 + cfg.layers * 6))
    width = cfg.width
    params = {
        "token_embed": _normal(next(keys), (cfg.vocab_size, width), 0.02),
        "blocks": (),
        "final_scale": jnp.ones((width,), dtype=jnp.float32),
        "final_bias": jnp.zeros((width,), dtype=jnp.float32),
    }
    blocks = []
    for _ in range(cfg.layers):
        blocks.append(
            {
                "norm1_scale": jnp.ones((width,), dtype=jnp.float32),
                "norm1_bias": jnp.zeros((width,), dtype=jnp.float32),
                "q": _normal(next(keys), (width, width), 1 / math.sqrt(width)),
                "k": _normal(next(keys), (width, width), 1 / math.sqrt(width)),
                "v": _normal(next(keys), (width, width), 1 / math.sqrt(width)),
                "o": _normal(next(keys), (width, width), 1 / math.sqrt(width)),
                "norm2_scale": jnp.ones((width,), dtype=jnp.float32),
                "norm2_bias": jnp.zeros((width,), dtype=jnp.float32),
                "up": _normal(next(keys), (width, cfg.mlp_width), 1 / math.sqrt(width)),
                "down": _normal(
                    next(keys), (cfg.mlp_width, width), 1 / math.sqrt(cfg.mlp_width)
                ),
            }
        )
    params["blocks"] = tuple(blocks)
    params["mlm"] = {
        "dense": _normal(next(keys), (width, width), 1 / math.sqrt(width)),
        "dense_bias": jnp.zeros((width,), dtype=jnp.float32),
        "norm_scale": jnp.ones((width,), dtype=jnp.float32),
        "norm_bias": jnp.zeros((width,), dtype=jnp.float32),
        "vocab_bias": jnp.zeros((cfg.vocab_size,), dtype=jnp.float32),
    }
    actual = sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(params))
    expected = parameter_count(cfg, include_mlm=True)
    if actual != expected:
        raise AssertionError(
            f"parameter formula {expected} disagrees with tree {actual}"
        )
    return params


def parameter_count(config, *, include_mlm=False):
    cfg = _config(config)
    d, f, vocab = cfg.width, cfg.mlp_width, cfg.vocab_size
    total = vocab * d + 2 * d
    total += cfg.layers * (4 * d * d + 2 * d * f + 4 * d)
    if include_mlm:
        total += d * d + d + 2 * d + vocab
    return total


def layer_norm(x, scale, bias, epsilon=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + epsilon) * scale + bias


def _split_heads(x, cfg):
    batch, length, _ = x.shape
    return x.reshape(batch, length, cfg.heads, cfg.width // cfg.heads).transpose(
        0, 2, 1, 3
    )


def _join_heads(x):
    batch, heads, length, dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, length, heads * dim)


def _rope(x):
    """Apply rotary position embedding to [batch, heads, time, head_dim]."""
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("RoPE requires an even attention head width")
    position = jnp.arange(x.shape[-2], dtype=jnp.float32)
    inv = 1.0 / (10000.0 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    angle = position[:, None] * inv[None, :]
    cos, sin = jnp.cos(angle)[None, None, :, :], jnp.sin(angle)[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return jnp.stack((even * cos - odd * sin, even * sin + odd * cos), axis=-1).reshape(
        x.shape
    )


def _dropout(x, key, rate):
    if key is None or rate == 0:
        return x
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


def hidden_states(params, input_ids, config, valid_mask=None, *, dropout_key=None):
    """Return final normalized token states with full bidirectional attention."""
    cfg = _config(config)
    ids = jnp.asarray(input_ids, dtype=jnp.int32)
    if ids.ndim != 2 or ids.shape[1] < 1 or ids.shape[1] > cfg.max_length:
        raise ValueError("input_ids must be [batch, 1..max_length]")
    if valid_mask is None:
        valid = ids != cfg.pad_id
    else:
        valid = jnp.asarray(valid_mask, dtype=jnp.bool_)
        if valid.shape != ids.shape:
            raise ValueError("valid_mask must match input_ids")
        if not _is_traced(ids) and bool(
            np.any(np.asarray(valid & (ids == cfg.pad_id)))
        ):
            raise ValueError("padding IDs cannot be marked valid")
    if not _is_traced(ids) and bool(
        np.any(np.asarray((ids < 0) | (ids >= cfg.vocab_size)))
    ):
        raise ValueError("token ID is outside the configured vocabulary")

    x = params["token_embed"][ids]
    # Invalid query rows are masked after each block; valid queries can attend
    # to every valid key in the same input. There is intentionally no triangle.
    allowed = valid[:, None, None, :] & valid[:, None, :, None]
    keys = (
        jax.random.split(dropout_key, max(1, cfg.layers * 2))
        if dropout_key is not None
        else None
    )
    for index, block in enumerate(params["blocks"]):
        h = layer_norm(x, block["norm1_scale"], block["norm1_bias"])
        q, k, v = (_split_heads(h @ block[name], cfg) for name in ("q", "k", "v"))
        q, k = _rope(q), _rope(k)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(cfg.width // cfg.heads)
        has_key = jnp.any(allowed, axis=-1, keepdims=True)
        fallback = (~has_key) & (jnp.arange(ids.shape[1]) == 0)
        safe_allowed = allowed | fallback
        scores = jnp.where(safe_allowed, scores, jnp.asarray(-1e30, jnp.float32))
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        weights = jnp.where(has_key, weights, 0.0)
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        branch = _join_heads(attended) @ block["o"]
        branch = _dropout(
            branch, None if keys is None else keys[2 * index], cfg.dropout
        )
        x = jnp.where(valid[..., None], x + branch, 0.0)
        h = layer_norm(x, block["norm2_scale"], block["norm2_bias"])
        branch = jax.nn.gelu(h @ block["up"]) @ block["down"]
        branch = _dropout(
            branch, None if keys is None else keys[2 * index + 1], cfg.dropout
        )
        x = jnp.where(valid[..., None], x + branch, 0.0)
    return jnp.where(
        valid[..., None],
        layer_norm(x, params["final_scale"], params["final_bias"]),
        0.0,
    )


def mlm_logits(params, input_ids, config, valid_mask=None, *, dropout_key=None):
    cfg = _config(config)
    hidden = hidden_states(params, input_ids, cfg, valid_mask, dropout_key=dropout_key)
    head = params["mlm"]
    transformed = jax.nn.gelu(hidden @ head["dense"] + head["dense_bias"])
    transformed = layer_norm(transformed, head["norm_scale"], head["norm_bias"])
    return transformed @ params["token_embed"].T + head["vocab_bias"]


def pool(params, input_ids, config, valid_mask=None, *, dropout_key=None):
    """Mean-pool content tokens; PAD/BOS/EOS/UNK/MASK never enter the mean."""
    cfg = _config(config)
    ids = jnp.asarray(input_ids, dtype=jnp.int32)
    valid = (
        ids != cfg.pad_id if valid_mask is None else jnp.asarray(valid_mask, jnp.bool_)
    )
    content = valid
    for token_id in (cfg.bos_id, cfg.eos_id, cfg.unk_id, cfg.mask_id):
        content = content & (ids != token_id)
    hidden = hidden_states(params, ids, cfg, valid, dropout_key=dropout_key)
    count = jnp.sum(content, axis=1, keepdims=True)
    mean = jnp.sum(jnp.where(content[..., None], hidden, 0.0), axis=1) / jnp.maximum(
        count, 1
    )
    norm = jnp.linalg.norm(mean, axis=-1, keepdims=True)
    return jnp.where((count > 0) & (norm > 0), mean / jnp.maximum(norm, 1e-12), 0.0)


def masked_token_loss(params, input_ids, labels, loss_mask, config, dropout_key=None):
    """Selected-token summed cross entropy and count, with an empty-mask error."""
    cfg = _config(config)
    ids = jnp.asarray(input_ids, jnp.int32)
    labels = jnp.asarray(labels, jnp.int32)
    selected = jnp.asarray(loss_mask, jnp.bool_)
    if ids.shape != labels.shape or ids.shape != selected.shape:
        raise ValueError("MLM ids, labels, and loss_mask must have equal shapes")
    if not _is_traced(ids) and bool(
        np.any(
            np.asarray(
                selected
                & ((ids == cfg.pad_id) | (labels < 5) | (labels >= cfg.vocab_size))
            )
        )
    ):
        raise ValueError("selected MLM targets must be content tokens, not padding")
    count = jnp.sum(selected)
    if not _is_traced(count) and int(np.asarray(count)) < 1:
        raise ValueError("MLM loss mask must select at least one token")
    logits = mlm_logits(params, ids, cfg, ids != cfg.pad_id, dropout_key=dropout_key)
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    nll = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
    total = jnp.sum(jnp.where(selected, nll, 0.0))
    return total / jnp.maximum(count, 1), total, jnp.asarray(count, jnp.int32)
