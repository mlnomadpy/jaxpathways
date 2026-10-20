"""Reference implementations for the causal objective contract."""

import jax
import jax.numpy as jnp


def causal_attention_mask(token_ids, pad_id=0):
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, sequence]")
    positions = jnp.arange(token_ids.shape[1])
    causal = positions[:, None] >= positions[None, :]
    valid_keys = token_ids != pad_id
    return causal[None, None, :, :] & valid_keys[:, None, None, :]


def shift_tokens(sequences, target_start_indices, pad_id=0):
    if sequences.ndim != 2:
        raise ValueError("sequences must have shape [batch, sequence]")
    if target_start_indices.shape != (sequences.shape[0],):
        raise ValueError("one target start index is required per sequence")
    inputs = sequences[:, :-1]
    targets = sequences[:, 1:]
    target_positions = jnp.arange(1, sequences.shape[1])[None, :]
    mask = (targets != pad_id) & (target_positions >= target_start_indices[:, None])
    return inputs, targets, mask


def masked_causal_cross_entropy(logits, targets, target_mask):
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError(
            "logits and targets must have shapes [batch, time, vocab] and [batch, time]"
        )
    if target_mask.shape != targets.shape:
        raise ValueError("target mask must match target shape")
    weights = target_mask.astype(logits.dtype)
    count = jnp.sum(weights)
    if not isinstance(count, jax.core.Tracer) and float(count) == 0:
        raise ValueError("at least one next-token target must be selected")
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[
        ..., 0
    ]
    return -jnp.sum(target_log_probs * weights) / jnp.maximum(count, 1.0)


def sample_next_token(logits, key, temperature=1.0, top_k=0):
    if logits.ndim != 1:
        raise ValueError("sampling expects one vocabulary-logit vector")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k < 0 or top_k > logits.shape[0]:
        raise ValueError("top_k must be zero or no larger than the vocabulary")
    filtered = logits / temperature
    if top_k:
        threshold = jnp.sort(filtered)[-top_k]
        filtered = jnp.where(filtered >= threshold, filtered, -jnp.inf)
    return jax.random.categorical(key, filtered).astype(jnp.int32)
