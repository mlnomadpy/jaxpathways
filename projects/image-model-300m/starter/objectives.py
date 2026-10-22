"""Learner objectives for the photo-pilot encoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Mean stable multiclass cross-entropy for integer class IDs."""
    # TODO: use log-softmax and gather one target class per row.
    raise NotImplementedError("TODO: implement stable cross-entropy")


def random_patch_mask(key: jax.Array, batch: int, patches: int, mask_ratio: float = 0.5):
    """Return distinct visible/hidden patch indices and a binary hidden mask."""
    # TODO: shuffle each image's patch IDs with explicit keys and reject edge ratios.
    raise NotImplementedError("TODO: select a fixed visible population without replacement")


def mae_forward(encoder, decoder, images, mask_key, *, dropout_key=None, mask_ratio=0.5, train=False):
    """Encode visible patches only, restore sequence positions, and score hidden pixels."""
    # TODO: hidden-pixel changes may alter target loss but never visible encoder inputs.
    raise NotImplementedError("TODO: implement the visible-only masked-image path")


def multi_positive_nt_xent(projected, source_positive, temperature=0.2):
    """Return multi-positive log-mass InfoNCE and auditable pair masks."""
    # TODO: exclude self-pairs, include paired views and declared same-source neighbors.
    raise NotImplementedError("TODO: implement two-view contrastive loss")
