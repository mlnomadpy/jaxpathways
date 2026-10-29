"""C02 pre-LN causal decoder with RoPE and per-layer incremental KV state.

This module locally adapts the accepted L05 `projects/text-harness/foundations/model.py`
and `config.py` (source hashes are in `data/foundation-reuse.json`). C02 replaces
learned absolute positions with RoPE and exposes explicit aligned targets so
packed article/passage boundaries cannot invent next-token transitions.
"""

from dataclasses import asdict
import math
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

try:
    from .config import ModelConfig
except ImportError:  # The public checker loads this file by path.
    from config import ModelConfig

LORA_RANK = 4
LORA_ALPHA = 8.0
LORA_SCALE = LORA_ALPHA / LORA_RANK


def _config(config):
    result = config if isinstance(config, ModelConfig) else ModelConfig(**config)
    result.validate()
    return result


def parameter_count(config):
    """Count the unique parameter leaves, tying embeddings exactly once."""
    cfg = _config(config)
    width, mlp, vocab = cfg.width, cfg.mlp_width, cfg.vocab_size
    return (
        vocab * width
        + cfg.layers * (4 * width * width + 2 * width * mlp + 4 * width)
        + 2 * width
    )


def _normal(key, shape, scale):
    return jax.random.normal(key, shape, dtype=jnp.float32) * scale


def initialize(key, config):
    """Initialize a bias-free projection tree and verify its derived count."""
    cfg = _config(config)
    keys = iter(jax.random.split(key, 1 + cfg.layers * 6))
    width, mlp = cfg.width, cfg.mlp_width
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
                "q_proj": _normal(next(keys), (width, width), 1 / math.sqrt(width)),
                "k_proj": _normal(next(keys), (width, width), 1 / math.sqrt(width)),
                "v_proj": _normal(next(keys), (width, width), 1 / math.sqrt(width)),
                "o_proj": _normal(next(keys), (width, width), 1 / math.sqrt(width)),
                "norm2_scale": jnp.ones((width,), dtype=jnp.float32),
                "norm2_bias": jnp.zeros((width,), dtype=jnp.float32),
                "up": _normal(next(keys), (width, mlp), 1 / math.sqrt(width)),
                "down": _normal(next(keys), (mlp, width), 1 / math.sqrt(mlp)),
            }
        )
    params["blocks"] = tuple(blocks)
    actual = sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(params))
    if actual != parameter_count(cfg):
        raise AssertionError(
            f"parameter formula {parameter_count(cfg)} disagrees with initialized tree {actual}"
        )
    return params


def initialize_lora(key, config, rank=4):
    """Create zero-delta Q/V adapters for every layer with frozen rank four."""
    cfg = _config(config)
    if rank != 4:
        raise ValueError("the frozen C02 pilot uses LoRA rank four")
    keys = iter(jax.random.split(key, cfg.layers * 2))
    adapters = []
    for _ in range(cfg.layers):
        layer = {}
        for name in ("q_proj", "v_proj"):
            layer[name] = {
                "A": jax.random.normal(next(keys), (cfg.width, rank), dtype=jnp.float32)
                * 0.02,
                "B": jnp.zeros((rank, cfg.width), dtype=jnp.float32),
            }
        adapters.append(layer)
    return {"blocks": tuple(adapters)}


def lora_parameter_count(config, rank=4):
    cfg = _config(config)
    if rank != 4:
        raise ValueError("the frozen C02 pilot uses LoRA rank four")
    return cfg.layers * 2 * rank * (cfg.width + cfg.width)


def merge_lora(params, adapters, config):
    """Return a new base tree with each Q/V low-rank delta merged once."""
    cfg = _config(config)
    if len(adapters.get("blocks", ())) != cfg.layers:
        raise ValueError("adapter layer count does not match the model")
    merged_blocks = []
    for block, adapter in zip(params["blocks"], adapters["blocks"], strict=True):
        copied = dict(block)
        for name in ("q_proj", "v_proj"):
            if name not in adapter:
                raise ValueError(f"adapter is missing {name}")
            copied[name] = block[name] + LORA_SCALE * (
                adapter[name]["A"] @ adapter[name]["B"]
            )
        merged_blocks.append(copied)
    return {**params, "blocks": tuple(merged_blocks)}


def state_fingerprint(params, adapters=None):
    """Hash config-independent array identities for cache and adapter binding."""
    digest = hashlib.sha256()
    for group_name, tree in (("params", params), ("adapters", adapters)):
        digest.update(group_name.encode())
        if tree is None:
            continue
        path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
        for path, leaf in path_leaves:
            value = np.asarray(leaf)
            digest.update(str(path).encode())
            digest.update(str(value.dtype).encode())
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def layer_norm(x, scale, bias, epsilon=1e-5):
    mean = jnp.mean(x.astype(jnp.float32), axis=-1, keepdims=True)
    variance = jnp.mean(
        jnp.square(x.astype(jnp.float32) - mean), axis=-1, keepdims=True
    )
    return (x - mean) * jax.lax.rsqrt(variance + epsilon) * scale + bias


def rotary(x, position_ids, base=10000.0):
    """Apply pairwise RoPE to `[batch, heads, time, head_width]` queries/keys."""
    width = x.shape[-1]
    positions = jnp.asarray(position_ids, dtype=jnp.float32)
    if positions.ndim == 1:
        positions = jnp.broadcast_to(
            positions[None, :], (x.shape[0], positions.shape[0])
        )
    if positions.shape != (x.shape[0], x.shape[2]):
        raise ValueError("position_ids must match batch and sequence dimensions")
    frequency = jnp.power(
        jnp.asarray(base, jnp.float32),
        -jnp.arange(0, width, 2, dtype=jnp.float32) / width,
    )
    angle = positions[..., None] * frequency[None, None, :]
    cosine, sine = jnp.cos(angle)[:, None, :, :], jnp.sin(angle)[:, None, :, :]
    paired = x.reshape(*x.shape[:-1], width // 2, 2)
    even, odd = paired[..., 0], paired[..., 1]
    return jnp.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), axis=-1
    ).reshape(x.shape)


def scaled_dot_attention(q, k, v, allowed):
    """Stable attention; all-masked rows return zero weights rather than NaNs."""
    scores = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32))
    scores = scores / jnp.sqrt(jnp.asarray(q.shape[-1], jnp.float32))
    allowed = jnp.asarray(allowed, dtype=jnp.bool_)
    any_allowed = jnp.any(allowed, axis=-1, keepdims=True)
    fallback_key = jnp.arange(k.shape[-2]) == 0
    safe_allowed = allowed | ((~any_allowed) & fallback_key[None, None, None, :])
    masked = jnp.where(safe_allowed, scores, jnp.asarray(-1e30, jnp.float32))
    weights = jax.nn.softmax(masked, axis=-1)
    weights = jnp.where(any_allowed, weights, 0.0)
    output = jnp.einsum("bhqk,bhkd->bhqd", weights, v.astype(jnp.float32))
    return output, weights


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


def attention_mask(valid_mask, segment_ids):
    """Return causal, valid-query/key and same-positive-segment connectivity."""
    valid = jnp.asarray(valid_mask, dtype=jnp.bool_)
    segments = jnp.asarray(segment_ids, dtype=jnp.int32)
    if valid.ndim != 2 or segments.shape != valid.shape:
        raise ValueError("valid_mask and segment_ids must be matching rank-two arrays")
    length = valid.shape[1]
    positions = jnp.arange(length)
    causal = positions[:, None] >= positions[None, :]
    return (
        causal[None, None, :, :]
        & valid[:, None, :, None]
        & valid[:, None, None, :]
        & (segments[:, None, :, None] == segments[:, None, None, :])
        & (segments[:, None, :, None] > 0)
    )


def _effective_kernel(block, name, adapter_block=None, scale=None):
    kernel = block[name]
    if adapter_block is None or name not in adapter_block:
        return kernel
    adapter = adapter_block[name]
    return kernel + scale * (adapter["A"] @ adapter["B"])


def logits(
    params,
    tokens,
    config,
    valid_mask=None,
    segment_ids=None,
    position_ids=None,
    dropout_key=None,
    adapters=None,
    return_attention=False,
):
    """Run a causal decoder on `[batch,time]` IDs using FP32 attention math."""
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
    positions = (
        jnp.broadcast_to(jnp.arange(tokens.shape[1]), tokens.shape)
        if position_ids is None
        else jnp.asarray(position_ids, dtype=jnp.int32)
    )
    if positions.shape != tokens.shape:
        raise ValueError("position_ids must match tokens")
    x = params["token_embed"][tokens]
    allowed = attention_mask(valid, segments)
    keys = (
        jax.random.split(dropout_key, max(1, cfg.layers * 2))
        if dropout_key is not None
        else None
    )
    attentions = []
    adapter_scale = LORA_SCALE
    for index, block in enumerate(params["blocks"]):
        adapter_block = adapters["blocks"][index] if adapters is not None else None
        h = layer_norm(x, block["norm1_scale"], block["norm1_bias"])
        q = _split_heads(
            h @ _effective_kernel(block, "q_proj", adapter_block, adapter_scale), cfg
        )
        k = _split_heads(
            h @ _effective_kernel(block, "k_proj", adapter_block, adapter_scale), cfg
        )
        v = _split_heads(
            h @ _effective_kernel(block, "v_proj", adapter_block, adapter_scale), cfg
        )
        q = rotary(q, positions, cfg.rope_base)
        k = rotary(k, positions, cfg.rope_base)
        attended, weights = scaled_dot_attention(q, k, v, allowed)
        branch = _join_heads(attended) @ block["o_proj"]
        if keys is not None:
            branch = _dropout(branch, keys[2 * index], cfg.dropout)
        x = x + branch
        h = layer_norm(x, block["norm2_scale"], block["norm2_bias"])
        branch = jax.nn.gelu(h @ block["up"]) @ block["down"]
        if keys is not None:
            branch = _dropout(branch, keys[2 * index + 1], cfg.dropout)
        x = x + branch
        attentions.append(weights)
    x = layer_norm(x, params["final_scale"], params["final_bias"])
    output = x @ params["token_embed"].T
    return (output, tuple(attentions)) if return_attention else output


def aligned_token_loss(
    params,
    input_ids,
    target_ids,
    target_mask,
    segment_ids,
    position_ids,
    config,
    key=None,
    adapters=None,
):
    """Return the raw numerator/count for already aligned target arrays."""
    inputs = jnp.asarray(input_ids, dtype=jnp.int32)
    targets = jnp.asarray(target_ids, dtype=jnp.int32)
    mask = jnp.asarray(target_mask, dtype=jnp.bool_)
    if inputs.shape != targets.shape or inputs.shape != mask.shape:
        raise ValueError("aligned inputs, targets and target mask must match")
    predictions = logits(
        params,
        inputs,
        config,
        valid_mask=jnp.ones(inputs.shape, dtype=jnp.bool_),
        segment_ids=segment_ids,
        position_ids=position_ids,
        dropout_key=key,
        adapters=adapters,
    ).astype(jnp.float32)
    log_probs = jax.nn.log_softmax(predictions, axis=-1)
    selected = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    count = jnp.sum(mask.astype(jnp.float32))
    numerator = -jnp.sum(jnp.where(mask, selected, 0.0))
    return numerator, count


def shifted_causal_targets(tokens, segment_ids, valid_mask=None):
    """Build shifted targets with target-side padding and document-boundary mask."""
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    segments = jnp.asarray(segment_ids, dtype=jnp.int32)
    valid = (
        tokens != 0 if valid_mask is None else jnp.asarray(valid_mask, dtype=jnp.bool_)
    )
    if (
        tokens.shape != segments.shape
        or tokens.shape != valid.shape
        or tokens.ndim != 2
        or tokens.shape[1] < 2
    ):
        raise ValueError(
            "tokens, segment IDs and validity must be matching rank-two arrays"
        )
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    target_valid = (
        valid[:, 1:]
        & valid[:, :-1]
        & (segments[:, :-1] == segments[:, 1:])
        & (segments[:, 1:] > 0)
    )
    return inputs, targets, segments[:, :-1], target_valid


def token_loss(
    params, tokens, segment_ids, config, valid_mask=None, key=None, adapters=None
):
    """Return shifted-loss numerator/count, masking each next-token target."""
    inputs, targets, input_segments, eligible = shifted_causal_targets(
        tokens, segment_ids, valid_mask
    )
    valid = (
        (inputs != 0)
        if valid_mask is None
        else jnp.asarray(valid_mask, dtype=jnp.bool_)[:, :-1]
    )
    predictions = logits(
        params,
        inputs,
        config,
        valid_mask=valid,
        segment_ids=input_segments,
        dropout_key=key,
        adapters=adapters,
    ).astype(jnp.float32)
    log_probs = jax.nn.log_softmax(predictions, axis=-1)
    selected = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    count = jnp.sum(eligible.astype(jnp.float32))
    numerator = -jnp.sum(jnp.where(eligible, selected, 0.0))
    return numerator, count


def prefill(params, prompt, config, valid_mask=None, adapters=None):
    """Prefill per-layer RoPE K/V for one or more equal-length prompts."""
    cfg = _config(config)
    ids = np.asarray(prompt, dtype=np.int32)
    if ids.ndim == 1:
        ids = ids[None, :]
    if ids.ndim != 2 or ids.shape[1] < 1 or ids.shape[1] > cfg.context:
        raise ValueError("prompt must have shape [batch,time] and fit the context")
    if np.any(ids < 1) or np.any(ids >= cfg.vocab_size):
        raise ValueError("prompt contains an invalid token id")
    batch, length = ids.shape
    valid = (
        np.ones((batch, length), dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != ids.shape or np.any(~valid.any(axis=1)):
        raise ValueError(
            "each prompt needs a matching validity mask with a valid token"
        )
    positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
    tokens = jnp.asarray(ids)
    segments = jnp.where(jnp.asarray(valid), 1, 0)
    x = params["token_embed"][tokens]
    allowed = attention_mask(valid, segments)
    keys, values = [], []
    adapter_scale = LORA_SCALE
    for index, block in enumerate(params["blocks"]):
        adapter_block = adapters["blocks"][index] if adapters is not None else None
        h = layer_norm(x, block["norm1_scale"], block["norm1_bias"])
        q = _split_heads(
            h @ _effective_kernel(block, "q_proj", adapter_block, adapter_scale), cfg
        )
        k = _split_heads(
            h @ _effective_kernel(block, "k_proj", adapter_block, adapter_scale), cfg
        )
        v = _split_heads(
            h @ _effective_kernel(block, "v_proj", adapter_block, adapter_scale), cfg
        )
        q, k = rotary(q, positions, cfg.rope_base), rotary(k, positions, cfg.rope_base)
        keys.append(k)
        values.append(v)
        attended, _ = scaled_dot_attention(q, k, v, allowed)
        x = x + _join_heads(attended) @ block["o_proj"]
        h = layer_norm(x, block["norm2_scale"], block["norm2_bias"])
        x = x + jax.nn.gelu(h @ block["up"]) @ block["down"]
    x = layer_norm(x, params["final_scale"], params["final_bias"])
    last = x[:, -1, :] @ params["token_embed"].T
    capacity_shape = (batch, cfg.heads, cfg.context, cfg.width // cfg.heads)
    padded_keys = tuple(
        jnp.zeros(capacity_shape, jnp.float32).at[:, :, :length, :].set(k) for k in keys
    )
    padded_values = tuple(
        jnp.zeros(capacity_shape, jnp.float32).at[:, :, :length, :].set(v)
        for v in values
    )
    valid_cache = (
        jnp.zeros((batch, cfg.context), dtype=jnp.bool_)
        .at[:, :length]
        .set(jnp.asarray(valid))
    )
    return last, {
        "keys": padded_keys,
        "values": padded_values,
        "valid": valid_cache,
        "position": length,
        "capacity": cfg.context,
        "config_hash": cfg.fingerprint(),
        "state_hash": state_fingerprint(params, adapters),
    }


def decode_step(params, token, cache, config, adapters=None):
    """Append a same-position batch of tokens, updating every decoder layer."""
    cfg = _config(config)
    position = int(cache["position"])
    if (
        cache.get("config_hash") != cfg.fingerprint()
        or cache.get("capacity") != cfg.context
        or cache.get("state_hash") != state_fingerprint(params, adapters)
    ):
        raise ValueError("cache belongs to a different model configuration")
    if position < 1 or position >= cfg.context:
        raise ValueError("cache context capacity exhausted or invalid")
    token = np.asarray(token, dtype=np.int32)
    if token.ndim == 0:
        token = token[None]
    if (
        token.ndim != 1
        or token.shape[0] != cache["valid"].shape[0]
        or np.any(token < 1)
        or np.any(token >= cfg.vocab_size)
    ):
        raise ValueError("next token must be one valid ID per cached prompt")
    batch = token.shape[0]
    x = params["token_embed"][jnp.asarray(token)][:, None, :]
    positions = jnp.full((batch, 1), position, dtype=jnp.int32)
    next_keys, next_values = [], []
    adapter_scale = LORA_SCALE
    for index, block in enumerate(params["blocks"]):
        adapter_block = adapters["blocks"][index] if adapters is not None else None
        h = layer_norm(x, block["norm1_scale"], block["norm1_bias"])
        q = _split_heads(
            h @ _effective_kernel(block, "q_proj", adapter_block, adapter_scale), cfg
        )
        k = _split_heads(
            h @ _effective_kernel(block, "k_proj", adapter_block, adapter_scale), cfg
        )
        v = _split_heads(
            h @ _effective_kernel(block, "v_proj", adapter_block, adapter_scale), cfg
        )
        q, k = rotary(q, positions, cfg.rope_base), rotary(k, positions, cfg.rope_base)
        key_cache = cache["keys"][index].at[:, :, position : position + 1, :].set(k)
        value_cache = cache["values"][index].at[:, :, position : position + 1, :].set(v)
        next_keys.append(key_cache)
        next_values.append(value_cache)
        valid = cache["valid"].at[:, position].set(True)
        allowed = valid[:, None, None, :] & (
            jnp.arange(cfg.context)[None, None, None, :] <= position
        )
        attended, _ = scaled_dot_attention(q, key_cache, value_cache, allowed)
        x = x + _join_heads(attended) @ block["o_proj"]
        h = layer_norm(x, block["norm2_scale"], block["norm2_bias"])
        x = x + jax.nn.gelu(h @ block["up"]) @ block["down"]
    x = layer_norm(x, params["final_scale"], params["final_bias"])
    output = x[:, -1, :] @ params["token_embed"].T
    next_valid = cache["valid"].at[:, position].set(True)
    return output, {
        **cache,
        "keys": tuple(next_keys),
        "values": tuple(next_values),
        "valid": next_valid,
        "position": position + 1,
    }


def corrupt_cache_layer(cache, layer, amount=1.0):
    """Copy a cache with a deliberate key corruption for a negative oracle."""
    if not 0 <= layer < len(cache["keys"]):
        raise ValueError("cache layer index is out of range")
    keys = list(cache["keys"])
    keys[layer] = keys[layer].at[:, :, 0, 0].add(amount)
    return {**cache, "keys": tuple(keys)}


def generate(params, prompt, config, max_new_tokens, eos_token_id=2, adapters=None):
    """Greedy cached generation bounded by both output and model context."""
    cfg = _config(config)
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    ids = [int(token) for token in prompt]
    if not ids:
        raise ValueError("generation requires a nonempty prompt")
    next_logits, cache = prefill(params, ids, cfg, adapters=adapters)
    generated = []
    for _ in range(max_new_tokens):
        choices = next_logits.at[:, 0].set(-jnp.inf)
        token = int(jnp.argmax(choices[0]))
        generated.append(token)
        if token == eos_token_id:
            break
        if cache["position"] >= cfg.context:
            if len(generated) < max_new_tokens:
                raise ValueError("generation request exceeds absolute context capacity")
            break
        next_logits, cache = decode_step(params, token, cache, cfg, adapters=adapters)
    return generated


def configuration_payload(config):
    return asdict(_config(config))
