"""Learner functions for the tiny bidirectional encoder in the lineage pilot."""
import jax
import jax.numpy as jnp
import optax


def encode(model, params, token_batch):
    """Return one L2-normalized bidirectional vector per passage."""
    pooled, _ = model.apply({'params': params}, jnp.asarray(token_batch, dtype=jnp.int32))
    return pooled / jnp.maximum(jnp.linalg.norm(pooled, axis=-1, keepdims=True), 1e-8)


def masked_lm_loss(model, params, corrupted, targets, selected):
    """Predict only selected original word IDs, normalized by selected count."""
    _, logits = model.apply({'params': params}, jnp.asarray(corrupted, jnp.int32))
    token_losses = optax.softmax_cross_entropy_with_integer_labels(
        logits, jnp.asarray(targets, dtype=jnp.int32)
    )
    mask = jnp.asarray(selected, dtype=jnp.float32)
    count = jnp.sum(mask)
    # An empty selection is an invalid batch. NaN makes it fail loudly under jit.
    return jnp.sum(token_losses * mask) / jnp.where(count > 0, count, jnp.nan)


def contrastive_loss(model, params, left_tokens, right_tokens, temperature=0.15):
    """Symmetric InfoNCE over two independently masked views of each passage."""
    left = encode(model, params, left_tokens)
    right = encode(model, params, right_tokens)
    logits = left @ right.T / temperature
    labels = jnp.arange(logits.shape[0])
    return 0.5 * (
        jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        + jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits.T, labels))
    )


def supervised_contrastive_loss(embeddings, labels, temperature=0.15):
    """Attract same-chapter passages and repel other chapter labels in the batch."""
    vectors = jnp.asarray(embeddings)
    vectors = vectors / jnp.maximum(jnp.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)
    labels = jnp.asarray(labels)
    similarities = vectors @ vectors.T / temperature
    size = similarities.shape[0]
    nonself = ~jnp.eye(size, dtype=bool)
    positives = (labels[:, None] == labels[None, :]) & nonself
    logits = jnp.where(nonself, similarities, -1e9)
    positive_logits = jnp.where(positives, similarities, -1e9)
    log_probability = logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)
    positive_count = jnp.sum(positives, axis=1)
    per_anchor = -jnp.sum(jnp.where(positives, log_probability, 0.0), axis=1) / jnp.maximum(positive_count, 1)
    valid = positive_count > 0
    return jnp.sum(jnp.where(valid, per_anchor, 0.0)) / jnp.maximum(jnp.sum(valid), 1)
