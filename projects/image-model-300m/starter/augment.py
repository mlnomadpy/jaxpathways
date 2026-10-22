"""Implement the frozen preprocessing augmentations with explicit JAX keys."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def random_crop_parameters(key: jax.Array, size: int):
    """TODO: sample area/aspect, valid placement, and full-frame fallback."""
    raise NotImplementedError("TODO: implement the recipe's crop sampling rule")


def _bilinear_resize_crop(image: jax.Array, key: jax.Array) -> jax.Array:
    """TODO: resize only within the selected crop's boundaries."""
    raise NotImplementedError("TODO: implement crop-bounded bilinear sampling")


def horizontal_flip(image: jax.Array, key: jax.Array) -> jax.Array:
    """TODO: flip width with Bernoulli probability 0.5."""
    raise NotImplementedError("TODO: implement explicit-key horizontal flip")


def supervised_views(images: jax.Array, keys: jax.Array) -> jax.Array:
    """TODO: one independent horizontal flip per image."""
    raise NotImplementedError("TODO: implement supervised weak augmentation")


def contrastive_view(image: jax.Array, key: jax.Array) -> jax.Array:
    """TODO: apply crop, flip, brightness, contrast, saturation, then clip."""
    raise NotImplementedError("TODO: implement one SimCLR view")


def paired_contrastive_views(images: jax.Array, keys: jax.Array) -> jax.Array:
    """TODO: independently transform two views per image."""
    raise NotImplementedError("TODO: implement paired SimCLR views")
