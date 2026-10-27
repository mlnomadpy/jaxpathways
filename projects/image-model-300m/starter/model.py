"""Learner implementation surface for the real-image pilot."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


PILOT_CONFIG = {
    "image_size": 64,
    "channels": 3,
    "patch_size": 8,
    "width": 32,
    "heads": 4,
    "mlp_dim": 64,
    "layers": 2,
    "classes": 5,
    "dropout_rate": 0.1,
    "layer_norm_epsilon": 1e-5,
}


def patchify(images: jax.Array, patch_size: int = 8) -> jax.Array:
    """Map NHWC images to row-major patches with within-patch H,W,C order."""
    # TODO: Validate divisible image dimensions, then reshape and transpose before flattening.
    raise NotImplementedError("TODO: implement the declared row-major NHWC patch order")


def unpatchify(patches: jax.Array, image_size: int = 64, patch_size: int = 8, channels: int = 3) -> jax.Array:
    """Invert patchify without changing channel order."""
    # TODO: Undo the exact patch grid and local pixel axes used in patchify.
    raise NotImplementedError("TODO: restore the original spatial layout")


def init_vit(key: jax.Array, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create patch projection, class/grid positions, two pre-LN attention blocks and final norm."""
    # TODO: Split keys explicitly and return the documented numeric parameter tree plus config.
    raise NotImplementedError("TODO: initialize the bidirectional ViT parameter tree")


def encode_images(
    params: dict[str, Any], images: jax.Array, *, dropout_key: jax.Array | None = None, train: bool = False
) -> jax.Array:
    """Return the final normalized class-token feature for each RGB image."""
    # TODO: Apply all patches without a causal mask; use an explicit key in training mode.
    raise NotImplementedError("TODO: implement scaled dot-product self-attention and pooling")


def classification_logits(
    params: dict[str, Any],
    head: dict[str, jax.Array],
    images: jax.Array,
    *,
    dropout_key: jax.Array | None = None,
    train: bool = False,
) -> jax.Array:
    """Map image batches to five class logits."""
    # TODO: Apply the candidate encoder and a biased linear task head.
    raise NotImplementedError("TODO: implement the classification head")


def init_cnn(key: jax.Array, classes: int = 5, channels: int = 3) -> dict[str, jax.Array]:
    """Create the two-convolution RGB comparison control."""
    # TODO: Initialize 3x3 convolutions with 16 and 32 channels and a global-pool head.
    raise NotImplementedError("TODO: initialize the CNN control")


def cnn_logits(
    params: dict[str, jax.Array],
    images: jax.Array,
    *,
    dropout_key: jax.Array | None = None,
    train: bool = False,
) -> jax.Array:
    """Return CNN class logits while keeping dropout randomness explicit."""
    # TODO: Implement SAME 3x3 convolution, stride-two reduction, ReLU and spatial mean.
    raise NotImplementedError("TODO: implement the CNN control forward pass")
