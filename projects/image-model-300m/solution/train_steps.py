"""Pure loss functions over array-only trainable trees for each pilot arm."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

import model
import objectives


def split_vit(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate static model metadata from the differentiable parameter tree."""
    static = {"config": dict(params["config"])}
    trainable = {name: value for name, value in params.items() if name != "config"}
    return trainable, static


def assemble_vit(trainable: dict[str, Any], static: dict[str, Any]) -> dict[str, Any]:
    return {**trainable, "config": dict(static["config"])}


def split_decoder(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    static_names = ("heads", "epsilon", "dropout_rate")
    static = {name: params[name] for name in static_names}
    trainable = {
        name: value for name, value in params.items() if name not in static_names
    }
    return trainable, static


def assemble_decoder(
    trainable: dict[str, Any], static: dict[str, Any]
) -> dict[str, Any]:
    return {**trainable, **static}


def _cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    log_probabilities = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probabilities, labels[:, None], axis=1))


def vit_supervised_loss(
    trainable: dict[str, Any],
    head: dict[str, jax.Array],
    static: dict[str, Any],
    images: jax.Array,
    labels: jax.Array,
    dropout_key: jax.Array,
    *,
    model_api: Any = model,
) -> jax.Array:
    params = assemble_vit(trainable, static)
    logits = model_api.classification_logits(
        params, head, images, dropout_key=dropout_key, train=True
    )
    return _cross_entropy(logits, labels)


def cnn_supervised_loss(
    params: dict[str, jax.Array],
    images: jax.Array,
    labels: jax.Array,
    dropout_key: jax.Array,
    *,
    model_api: Any = model,
) -> jax.Array:
    logits = model_api.cnn_logits(params, images, dropout_key=dropout_key, train=True)
    return _cross_entropy(logits, labels)


def frozen_probe_loss(
    head: dict[str, jax.Array],
    vit_trainable: dict[str, Any],
    static: dict[str, Any],
    images: jax.Array,
    labels: jax.Array,
    *,
    model_api: Any = model,
) -> jax.Array:
    params = assemble_vit(vit_trainable, static)
    features = jax.lax.stop_gradient(model_api.encode_images(params, images))
    logits = features @ head["kernel"] + head["bias"]
    return _cross_entropy(logits, labels)


def mae_pretraining_loss(
    vit_trainable: dict[str, Any],
    decoder_trainable: dict[str, Any],
    vit_static: dict[str, Any],
    decoder_static: dict[str, Any],
    images: jax.Array,
    mask_key: jax.Array,
    dropout_key: jax.Array,
    *,
    model_api: Any = model,
    objective_api: Any = objectives,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    encoder = assemble_vit(vit_trainable, vit_static)
    decoder = assemble_decoder(decoder_trainable, decoder_static)
    result = objective_api.mae_forward(
        encoder,
        decoder,
        images,
        mask_key,
        dropout_key=dropout_key,
        mask_ratio=0.5,
        train=True,
    )
    selected_values = jnp.sum(result["mask"]) * result["targets"].shape[-1]
    return result["loss"], {"selected_mae_target_values": selected_values}


def simclr_pretraining_loss(
    vit_trainable: dict[str, Any],
    projector: dict[str, jax.Array],
    vit_static: dict[str, Any],
    paired_views: jax.Array,
    source_positive: jax.Array,
    dropout_key: jax.Array,
    *,
    model_api: Any = model,
    objective_api: Any = objectives,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if paired_views.ndim != 5 or paired_views.shape[1] != 2:
        raise ValueError(
            "paired views must have shape (batch, 2, height, width, channels)"
        )
    encoder = assemble_vit(vit_trainable, vit_static)
    # The objective lays its positive matrix out as [all view 0, all view 1].
    # Keep encoder output in that same view-major order.
    view_major = jnp.transpose(paired_views, (1, 0, 2, 3, 4))
    flat_views = view_major.reshape((-1,) + paired_views.shape[2:])
    features = model_api.encode_images(
        encoder, flat_views, dropout_key=dropout_key, train=True
    )
    projected = objective_api.project_features(projector, features)
    loss, _ = objective_api.multi_positive_nt_xent(
        projected, source_positive, temperature=0.2
    )
    return loss, {"selected_mae_target_values": jnp.asarray(0, dtype=jnp.int32)}


def partial_trainable_vit(params: dict[str, Any]) -> dict[str, Any]:
    """Return only final block and final normalization leaves for partial tuning."""
    if not params["blocks"]:
        raise ValueError("partial tuning requires at least one encoder block")
    return {
        "last_block": params["blocks"][-1],
        "final_ln_scale": params["final_ln_scale"],
        "final_ln_bias": params["final_ln_bias"],
    }


def merge_partial_vit(
    original: dict[str, Any], partial: dict[str, Any]
) -> dict[str, Any]:
    blocks = tuple(original["blocks"][:-1]) + (partial["last_block"],)
    return {
        **original,
        "blocks": blocks,
        "final_ln_scale": partial["final_ln_scale"],
        "final_ln_bias": partial["final_ln_bias"],
    }


def trainable_leaf_inventory(tree: Any) -> list[dict[str, Any]]:
    """List array leaves only; static config must never enter an optimizer."""
    inventory = []

    def visit(value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], path + (key,))
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, path + (index,))
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            inventory.append(
                {
                    "path": "/".join(map(str, path)),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            )
        else:
            raise TypeError(
                f"static or unsupported value leaked into parameter tree at {path}"
            )

    visit(tree, ())
    return inventory
