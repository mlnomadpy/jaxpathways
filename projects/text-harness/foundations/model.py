"""Configurable pre-LN causal decoder, explicit masks, cache and token loss."""

from dataclasses import asdict
import math

import jax
import jax.numpy as jnp
import numpy as np

try:
    from .config import ModelConfig
except ImportError:  # The public checker also loads this file by path.
    from config import ModelConfig


def _config(config):
    result = config if isinstance(config, ModelConfig) else ModelConfig(**config)
    result.validate()
    return result


def parameter_count(config):
    """Return the exact trainable parameter count implied by model initialization."""
    cfg = _config(config)
    width, mlp, vocab = cfg.width, cfg.mlp_width, cfg.vocab_size
    # Token/position embeddings, two affine norms per block, four attention
    # projections, two MLP projections, final norm, and optional output head.
    count = vocab * width + cfg.context * width + 2 * width
    count += cfg.layers * (
        4 * width * width + 4 * width + width * mlp + mlp * width + mlp + width
    )
    if not cfg.tied_embeddings:
        count += width * vocab + vocab
    return count


def _normal(key, shape, scale):
    return jax.random.normal(key, shape, dtype=jnp.float32) * scale


def initialize(key, config):
    cfg = _config(config)
    keys = iter(jax.random.split(key, 3 + cfg.layers * 8))
    width, mlp = cfg.width, cfg.mlp_width
    params = {
        "token_embed": _normal(next(keys), (cfg.vocab_size, width), 0.02),
        "position": _normal(next(keys), (cfg.context, width), 0.01),
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
                "up": _normal(next(keys), (width, mlp), 1 / math.sqrt(width)),
                "up_bias": jnp.zeros((mlp,), dtype=jnp.float32),
                "down": _normal(next(keys), (mlp, width), 1 / math.sqrt(mlp)),
                "down_bias": jnp.zeros((width,), dtype=jnp.float32),
            }
        )
    params["blocks"] = tuple(blocks)
    if not cfg.tied_embeddings:
        params["head"] = _normal(
            next(keys), (width, cfg.vocab_size), 1 / math.sqrt(width)
        )
        params["head_bias"] = jnp.zeros((cfg.vocab_size,), dtype=jnp.float32)
    actual = sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(params))
    if actual != parameter_count(cfg):
        raise AssertionError(
            f"parameter formula {parameter_count(cfg)} disagrees with tree {actual}"
        )
    return params


def layer_norm(x, scale, bias, epsilon=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + epsilon) * scale + bias


def scaled_dot_attention(q, k, v, allowed):
    """Stable attention; an empty row gets zero weights rather than NaNs."""
    width = q.shape[-1]
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(
        jnp.asarray(width, jnp.float32)
    )
    allowed = jnp.asarray(allowed, dtype=jnp.bool_)
    any_allowed = jnp.any(allowed, axis=-1, keepdims=True)
    positions = jnp.arange(k.shape[-2])
    fallback = (~any_allowed) & (positions == 0)
    safe_allowed = allowed | fallback
    masked = jnp.where(safe_allowed, scores, jnp.asarray(-1e30, jnp.float32))
    weights = jax.nn.softmax(masked, axis=-1)
    weights = jnp.where(any_allowed, weights, 0.0)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v), weights


def _dropout(x, key, rate):
    if rate <= 0:
        return x
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


def _split_heads(x, cfg):
    batch, length, _ = x.shape
    return x.reshape(batch, length, cfg.heads, cfg.width // cfg.heads).transpose(
        0, 2, 1, 3
    )


def _join_heads(x):
    batch, heads, length, dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, length, heads * dim)


def _mask(valid, segments):
    batch, length = valid.shape
    positions = jnp.arange(length)
    causal = positions[:, None] >= positions[None, :]
    return (
        causal[None, None, :, :]
        & valid[:, None, None, :]
        & valid[:, None, :, None]
        & (segments[:, None, :, None] == segments[:, None, None, :])
        & (segments[:, None, :, None] > 0)
    )


def logits(
    params,
    tokens,
    config,
    valid_mask=None,
    segment_ids=None,
    dropout_key=None,
    return_attention=False,
):
    """Apply the decoder to `[batch, time]` IDs, using explicit masks and FP32 math."""
    cfg = _config(config)
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    if tokens.ndim != 2 or tokens.shape[1] > cfg.context:
        raise ValueError("tokens must be rank two and fit the configured context")
    valid = (
        tokens != 0 if valid_mask is None else jnp.asarray(valid_mask, dtype=jnp.bool_)
    )
    segments = (
        jnp.where(valid, 1, 0)
        if segment_ids is None
        else jnp.asarray(segment_ids, dtype=jnp.int32)
    )
    if valid.shape != tokens.shape or segments.shape != tokens.shape:
        raise ValueError("valid and segment masks must match token shape")
    x = params["token_embed"][tokens] + params["position"][None, : tokens.shape[1], :]
    allowed = _mask(valid, segments)
    keys = (
        jax.random.split(dropout_key, max(1, cfg.layers * 2))
        if dropout_key is not None
        else None
    )
    attentions = []
    for index, block in enumerate(params["blocks"]):
        h = layer_norm(x, block["norm1_scale"], block["norm1_bias"])
        q, k, v = (_split_heads(h @ block[name], cfg) for name in ("q", "k", "v"))
        attended, weights = scaled_dot_attention(q, k, v, allowed)
        branch = _join_heads(attended) @ block["o"]
        if keys is not None:
            branch = _dropout(branch, keys[2 * index], cfg.dropout)
        x = x + branch
        h = layer_norm(x, block["norm2_scale"], block["norm2_bias"])
        branch = (
            jax.nn.gelu(h @ block["up"] + block["up_bias"]) @ block["down"]
            + block["down_bias"]
        )
        if keys is not None:
            branch = _dropout(branch, keys[2 * index + 1], cfg.dropout)
        x = x + branch
        attentions.append(weights)
    x = layer_norm(x, params["final_scale"], params["final_bias"])
    if cfg.tied_embeddings:
        output = x @ params["token_embed"].T
    else:
        output = x @ params["head"] + params["head_bias"]
    return (output, tuple(attentions)) if return_attention else output


def causal_targets(tokens, segment_ids):
    """Return shifted tensors and the target-side same-document scoring mask."""
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    segments = jnp.asarray(segment_ids, dtype=jnp.int32)
    if tokens.shape != segments.shape or tokens.ndim != 2 or tokens.shape[1] < 2:
        raise ValueError(
            "packed tokens and segment IDs must be matching rank-two arrays"
        )
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    eligible = (
        (targets != 0) & (segments[:, :-1] == segments[:, 1:]) & (segments[:, :-1] > 0)
    )
    return inputs, targets, segments[:, :-1], eligible


def token_loss(params, tokens, segment_ids, config, key=None):
    inputs, targets, input_segments, eligible = causal_targets(tokens, segment_ids)
    valid = inputs != 0
    predictions = logits(params, inputs, config, valid, input_segments, key)
    log_probs = jax.nn.log_softmax(predictions.astype(jnp.float32), axis=-1)
    selected = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    count = jnp.sum(eligible)
    return -jnp.sum(jnp.where(eligible, selected, 0.0)) / jnp.maximum(count, 1)


def prefill(params, prompt, config):
    """Prefill per-layer projected K/V; cache position is an absolute next index."""
    cfg = _config(config)
    ids = np.asarray(prompt, dtype=np.int32)
    if ids.ndim != 1 or len(ids) < 1 or len(ids) > cfg.context:
        raise ValueError("prompt must be nonempty and fit the configured context")
    if np.any(ids < 1) or np.any(ids >= cfg.vocab_size):
        raise ValueError("prompt contains an invalid token id")
    length = len(ids)
    tokens = jnp.asarray(ids[None, :])
    x = params["token_embed"][tokens] + params["position"][None, :length, :]
    keys, values = [], []
    for block in params["blocks"]:
        h = layer_norm(x, block["norm1_scale"], block["norm1_bias"])
        q, k, v = (_split_heads(h @ block[name], cfg) for name in ("q", "k", "v"))
        keys.append(k)
        values.append(v)
        position = jnp.arange(length)
        allowed = (position[:, None] >= position[None, :])[None, None, :, :]
        attended, _ = scaled_dot_attention(q, k, v, allowed)
        x = x + _join_heads(attended) @ block["o"]
        h = layer_norm(x, block["norm2_scale"], block["norm2_bias"])
        x = (
            x
            + jax.nn.gelu(h @ block["up"] + block["up_bias"]) @ block["down"]
            + block["down_bias"]
        )
    x = layer_norm(x, params["final_scale"], params["final_bias"])
    last = x[:, -1, :]
    output = (
        last @ params["token_embed"].T
        if cfg.tied_embeddings
        else last @ params["head"] + params["head_bias"]
    )
    cap_shape = (1, cfg.heads, cfg.context, cfg.width // cfg.heads)
    padded_keys = tuple(
        jnp.zeros(cap_shape, jnp.float32).at[:, :, :length, :].set(k) for k in keys
    )
    padded_values = tuple(
        jnp.zeros(cap_shape, jnp.float32).at[:, :, :length, :].set(v) for v in values
    )
    return output[0], {
        "keys": padded_keys,
        "values": padded_values,
        "position": length,
        "capacity": cfg.context,
        "config_hash": cfg.fingerprint(),
    }


def decode_step(params, token, cache, config):
    """Append one token at the absolute position, returning new per-layer K/V state."""
    cfg = _config(config)
    position = int(cache["position"])
    if (
        cache.get("config_hash") != cfg.fingerprint()
        or cache.get("capacity") != cfg.context
    ):
        raise ValueError("cache belongs to a different model configuration")
    if position < 1 or position >= cfg.context:
        raise ValueError("cache context capacity exhausted or invalid")
    if not isinstance(token, (int, np.integer)) or token < 1 or token >= cfg.vocab_size:
        raise ValueError("invalid next token id")
    x = params["token_embed"][jnp.asarray(token, dtype=jnp.int32)][None, None, :]
    x = x + params["position"][None, position : position + 1, :]
    next_keys, next_values = [], []
    for index, block in enumerate(params["blocks"]):
        h = layer_norm(x, block["norm1_scale"], block["norm1_bias"])
        q, k, v = (_split_heads(h @ block[name], cfg) for name in ("q", "k", "v"))
        k_cache = cache["keys"][index].at[:, :, position : position + 1, :].set(k)
        v_cache = cache["values"][index].at[:, :, position : position + 1, :].set(v)
        next_keys.append(k_cache)
        next_values.append(v_cache)
        allowed = jnp.arange(cfg.context)[None, None, None, :] < position + 1
        attended, _ = scaled_dot_attention(q, k_cache, v_cache, allowed)
        x = x + _join_heads(attended) @ block["o"]
        h = layer_norm(x, block["norm2_scale"], block["norm2_bias"])
        x = (
            x
            + jax.nn.gelu(h @ block["up"] + block["up_bias"]) @ block["down"]
            + block["down_bias"]
        )
    x = layer_norm(x, params["final_scale"], params["final_bias"])
    last = x[:, -1, :]
    output = (
        last @ params["token_embed"].T
        if cfg.tied_embeddings
        else last @ params["head"] + params["head_bias"]
    )
    return output[0], {
        "keys": tuple(next_keys),
        "values": tuple(next_values),
        "position": position + 1,
        "capacity": cfg.context,
        "config_hash": cfg.fingerprint(),
    }


def corrupt_cache_layer(cache, layer, amount=1.0):
    """Copy a cache with a deliberate K corruption for controlled parity tests."""
    if not 0 <= layer < len(cache["keys"]):
        raise ValueError("cache layer index is out of range")
    keys = list(cache["keys"])
    keys[layer] = keys[layer].at[:, :, 0, 0].add(amount)
    return {**cache, "keys": tuple(keys)}


def generate(params, prompt, config, max_new_tokens):
    """Greedy cached generation with an explicit request-token budget."""
    cfg = _config(config)
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    ids = [int(token) for token in prompt]
    if not ids:
        raise ValueError("generation requires a nonempty prompt")
    next_logits, cache = prefill(params, ids, cfg)
    generated = []
    for _ in range(max_new_tokens):
        choices = next_logits.at[0].set(-jnp.inf)
        choices = choices.at[1].set(-jnp.inf)
        token = int(jnp.argmax(choices))
        generated.append(token)
        if token == 2:
            break
        if cache["position"] >= cfg.context:
            if len(generated) < max_new_tokens:
                raise ValueError("generation request exceeds absolute context capacity")
            break
        next_logits, cache = decode_step(params, token, cache, cfg)
    return generated


def configuration_payload(config):
    return asdict(_config(config))
