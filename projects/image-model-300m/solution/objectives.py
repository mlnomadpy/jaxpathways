"""Visible-only masked reconstruction and multi-positive image contrast."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import model as model_api

from model import _encoder_block, _linear_init, encode_patch_vectors, patchify


def cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "logits and labels must have shapes (batch, classes) and (batch,)"
        )
    selected = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    return -jnp.mean(jnp.take_along_axis(selected, labels[:, None], axis=1))


def random_patch_mask(
    key: jax.Array, batch: int, patches: int, mask_ratio: float = 0.5
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Select a fixed visible count per image; return visible, hidden, and binary mask."""
    if not 0.0 < mask_ratio < 1.0 or batch <= 0 or patches < 2:
        raise ValueError(
            "mask ratio must leave at least one visible and one hidden patch"
        )
    hidden_count = int(round(patches * mask_ratio))
    hidden_count = min(max(hidden_count, 1), patches - 1)
    visible_count = patches - hidden_count
    noise = jax.random.uniform(key, (batch, patches), dtype=jnp.float32)
    order = jnp.argsort(noise, axis=1)
    visible = order[:, :visible_count]
    hidden = order[:, visible_count:]
    mask = (
        jnp.zeros((batch, patches), dtype=jnp.float32)
        .at[jnp.arange(batch)[:, None], hidden]
        .set(1.0)
    )
    return visible, hidden, mask


def init_mae_decoder(
    key: jax.Array,
    width: int = 32,
    heads: int = 4,
    mlp_dim: int = 64,
    layers: int = 1,
    patches: int = 64,
    patch_values: int = 192,
    *,
    model_module: Any = model_api,
) -> dict[str, Any]:
    if width % heads or min(width, heads, mlp_dim, layers, patches, patch_values) <= 0:
        raise ValueError("invalid decoder dimensions")
    keys = iter(jax.random.split(key, 3 + layers * 4))
    blocks = []
    for _ in range(layers):
        qkv_kernel, qkv_bias = model_module._linear_init(next(keys), width, 3 * width)
        out_kernel, out_bias = model_module._linear_init(next(keys), width, width)
        mlp_in_kernel, mlp_in_bias = model_module._linear_init(
            next(keys), width, mlp_dim
        )
        mlp_out_kernel, mlp_out_bias = model_module._linear_init(
            next(keys), mlp_dim, width
        )
        blocks.append(
            {
                "ln1_scale": jnp.ones((width,), jnp.float32),
                "ln1_bias": jnp.zeros((width,), jnp.float32),
                "qkv_kernel": qkv_kernel,
                "qkv_bias": qkv_bias,
                "out_kernel": out_kernel,
                "out_bias": out_bias,
                "ln2_scale": jnp.ones((width,), jnp.float32),
                "ln2_bias": jnp.zeros((width,), jnp.float32),
                "mlp_in_kernel": mlp_in_kernel,
                "mlp_in_bias": mlp_in_bias,
                "mlp_out_kernel": mlp_out_kernel,
                "mlp_out_bias": mlp_out_bias,
            }
        )
    pred_kernel, pred_bias = model_module._linear_init(next(keys), width, patch_values)
    return {
        "mask_token": 0.02
        * jax.random.normal(next(keys), (1, 1, width), dtype=jnp.float32),
        "position_embedding": 0.02
        * jax.random.normal(next(keys), (1, patches + 1, width), dtype=jnp.float32),
        "blocks": tuple(blocks),
        "heads": heads,
        "epsilon": 1e-5,
        "dropout_rate": 0.1,
        "prediction_kernel": pred_kernel,
        "prediction_bias": pred_bias,
    }


def mae_forward(
    encoder: dict[str, Any],
    decoder: dict[str, Any],
    images: jax.Array,
    mask_key: jax.Array,
    *,
    dropout_key: jax.Array | None = None,
    mask_ratio: float = 0.5,
    train: bool = False,
    model_module: Any = model_api,
) -> dict[str, jax.Array]:
    cfg = encoder["config"]
    patch_size = int(cfg["patch_size"])
    targets = model_module.patchify(images, patch_size)
    batch, patch_count, _ = targets.shape
    visible, hidden, mask = random_patch_mask(mask_key, batch, patch_count, mask_ratio)
    if dropout_key is None:
        encoder_dropout_key = decoder_dropout_key = None
    else:
        encoder_dropout_key, decoder_dropout_key = jax.random.split(dropout_key)
    encoded = model_module.encode_patch_vectors(
        encoder,
        targets,
        visible,
        dropout_key=encoder_dropout_key,
        train=train,
        return_tokens=True,
    )
    visible_latents = encoded[:, 1:, :]
    restored = jnp.broadcast_to(
        decoder["mask_token"], (batch, patch_count, visible_latents.shape[-1])
    )
    restored = restored.at[jnp.arange(batch)[:, None], visible].set(visible_latents)
    cls = encoded[:, :1, :] + decoder["position_embedding"][:, :1, :]
    patches = restored + decoder["position_embedding"][:, 1:, :]
    tokens = jnp.concatenate((cls, patches), axis=1)
    block_keys = (
        (None,) * len(decoder["blocks"])
        if decoder_dropout_key is None
        else tuple(jax.random.split(decoder_dropout_key, len(decoder["blocks"])))
    )
    for block, key in zip(decoder["blocks"], block_keys):
        tokens = model_module._encoder_block(
            tokens,
            block,
            int(decoder["heads"]),
            float(decoder["epsilon"]),
            key,
            float(decoder["dropout_rate"]),
            train,
        )
    prediction = (
        tokens[:, 1:, :] @ decoder["prediction_kernel"] + decoder["prediction_bias"]
    )
    squared = jnp.square(prediction - targets)
    hidden_elements = mask[..., None] * jnp.ones_like(squared)
    loss = jnp.sum(squared * hidden_elements) / jnp.maximum(
        jnp.sum(hidden_elements), 1.0
    )
    return {
        "loss": loss,
        "prediction": prediction,
        "targets": targets,
        "mask": mask,
        "visible_indices": visible,
        "hidden_indices": hidden,
        "visible_encoder_tokens": encoded,
    }


def init_projector(
    key: jax.Array, input_dim: int = 32, hidden_dim: int = 64, output_dim: int = 32
) -> dict[str, jax.Array]:
    keys = jax.random.split(key, 2)
    first_kernel, first_bias = _linear_init(keys[0], input_dim, hidden_dim)
    second_kernel, second_bias = _linear_init(keys[1], hidden_dim, output_dim)
    return {
        "first_kernel": first_kernel,
        "first_bias": first_bias,
        "second_kernel": second_kernel,
        "second_bias": second_bias,
    }


def project_features(projector: dict[str, jax.Array], features: jax.Array) -> jax.Array:
    hidden = jax.nn.relu(features @ projector["first_kernel"] + projector["first_bias"])
    projected = hidden @ projector["second_kernel"] + projector["second_bias"]
    norm = jnp.linalg.norm(projected, axis=-1, keepdims=True)
    return projected / jnp.maximum(norm, 1e-12)


def multi_positive_nt_xent(
    projected: jax.Array,
    source_positive: jax.Array,
    temperature: float = 0.2,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """InfoNCE mass objective with known same-source/hash neighbors as positives."""
    if projected.ndim != 2 or projected.shape[0] < 4 or projected.shape[0] % 2:
        raise ValueError(
            "projected features must contain two views for at least two images"
        )
    image_count = projected.shape[0] // 2
    if source_positive.shape != (image_count, image_count):
        raise ValueError(
            "source_positive must be an image-level square relation matrix"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    positive = jnp.block(
        [[source_positive, source_positive], [source_positive, source_positive]]
    )
    identity = jnp.eye(projected.shape[0], dtype=bool)
    positive = positive | jnp.concatenate(
        (
            jnp.concatenate(
                (
                    jnp.zeros((image_count, image_count), dtype=bool),
                    jnp.eye(image_count, dtype=bool),
                ),
                axis=1,
            ),
            jnp.concatenate(
                (
                    jnp.eye(image_count, dtype=bool),
                    jnp.zeros((image_count, image_count), dtype=bool),
                ),
                axis=1,
            ),
        ),
        axis=0,
    )
    positive = positive & ~identity
    if not isinstance(positive, jax.core.Tracer) and bool(
        jnp.any(jnp.sum(positive, axis=1) == 0)
    ):
        raise ValueError("every view must have at least its paired view as a positive")
    similarities = (projected @ projected.T) / temperature
    denominator = jnp.where(identity, -jnp.inf, similarities)
    numerator = jnp.where(positive, similarities, -jnp.inf)
    loss_by_anchor = jax.scipy.special.logsumexp(
        denominator, axis=1
    ) - jax.scipy.special.logsumexp(numerator, axis=1)
    loss = jnp.mean(loss_by_anchor)
    return loss, {
        "similarities": similarities,
        "positive_mask": positive,
        "negative_mask": (~positive) & (~identity),
        "positive_similarity_mean": jnp.sum(jnp.where(positive, similarities, 0.0))
        / jnp.maximum(jnp.sum(positive), 1),
        "negative_similarity_mean": jnp.sum(
            jnp.where((~positive) & (~identity), similarities, 0.0)
        )
        / jnp.maximum(jnp.sum((~positive) & (~identity)), 1),
        "per_anchor_loss": loss_by_anchor,
    }


def positive_relation(
    image_ids: list[str], known_neighbor_pairs: set[tuple[str, str]] | None = None
) -> jax.Array:
    count = len(image_ids)
    known = (
        set()
        if known_neighbor_pairs is None
        else {tuple(sorted(pair)) for pair in known_neighbor_pairs}
    )
    matrix = jnp.eye(count, dtype=bool)
    for left in range(count):
        for right in range(left + 1, count):
            if tuple(sorted((image_ids[left], image_ids[right]))) in known:
                matrix = matrix.at[left, right].set(True).at[right, left].set(True)
    return matrix
