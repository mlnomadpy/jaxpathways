"""Explicit-key, label-preserving image views for supervised and SimCLR arms."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def horizontal_flip(image: jax.Array, key: jax.Array) -> jax.Array:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("an image must have shape (height, width, 3)")
    return jax.lax.cond(
        jax.random.bernoulli(key, 0.5),
        lambda value: value[:, ::-1, :],
        lambda value: value,
        image,
    )


def supervised_views(images: jax.Array, keys: jax.Array) -> jax.Array:
    """Apply only independent horizontal flips to a batch of normalized images."""
    if images.ndim != 4 or images.shape[-1] != 3 or keys.shape != (images.shape[0], 2):
        raise ValueError("expected NHWC RGB images and one JAX key per image")
    return jax.vmap(horizontal_flip)(images, keys)


def _bilinear_resize_crop(image: jax.Array, key: jax.Array) -> jax.Array:
    size = image.shape[0]
    crop_top, crop_left, crop_height, crop_width = random_crop_parameters(key, size)
    y = crop_top.astype(jnp.float32) + (jnp.arange(size, dtype=jnp.float32) + 0.5) * crop_height / size - 0.5
    x = crop_left.astype(jnp.float32) + (jnp.arange(size, dtype=jnp.float32) + 0.5) * crop_width / size - 0.5
    y = jnp.clip(y, crop_top, crop_top + crop_height - 1)
    x = jnp.clip(x, crop_left, crop_left + crop_width - 1)
    y0 = jnp.floor(y).astype(jnp.int32)
    x0 = jnp.floor(x).astype(jnp.int32)
    y1 = jnp.minimum(y0 + 1, crop_top + crop_height - 1)
    x1 = jnp.minimum(x0 + 1, crop_left + crop_width - 1)
    wy = (y - jnp.floor(y))[:, None, None]
    wx = (x - jnp.floor(x))[None, :, None]
    top_left = image[y0[:, None], x0[None, :], :]
    top_right = image[y0[:, None], x1[None, :], :]
    bottom_left = image[y1[:, None], x0[None, :], :]
    bottom_right = image[y1[:, None], x1[None, :], :]
    return (1 - wy) * ((1 - wx) * top_left + wx * top_right) + wy * ((1 - wx) * bottom_left + wx * bottom_right)


def random_crop_parameters(key: jax.Array, size: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return top, left, height, width; invalid oversize crops become full-frame."""
    if size <= 0:
        raise ValueError("crop size must be positive")
    area_key, ratio_key, top_key, left_key = jax.random.split(key, 4)
    area_fraction = jax.random.uniform(area_key, (), minval=0.75, maxval=1.0)
    log_ratio = jax.random.uniform(ratio_key, (), minval=jnp.log(0.90), maxval=jnp.log(1.1111111111))
    aspect = jnp.exp(log_ratio)
    sampled_height = jnp.rint(size * jnp.sqrt(area_fraction / aspect)).astype(jnp.int32)
    sampled_width = jnp.rint(size * jnp.sqrt(area_fraction * aspect)).astype(jnp.int32)
    valid = (sampled_height >= 1) & (sampled_width >= 1) & (sampled_height <= size) & (sampled_width <= size)
    crop_height = jnp.where(valid, sampled_height, size)
    crop_width = jnp.where(valid, sampled_width, size)
    top = jax.random.randint(top_key, (), 0, size - crop_height + 1)
    left = jax.random.randint(left_key, (), 0, size - crop_width + 1)
    return top, left, crop_height, crop_width


def contrastive_view(image: jax.Array, key: jax.Array) -> jax.Array:
    """Random-resized crop, flip, brightness, contrast, saturation, then clip."""
    if image.ndim != 3 or image.shape[-1] != 3 or image.shape[0] != image.shape[1]:
        raise ValueError("contrastive views require a square normalized RGB image")
    crop_key, flip_key, brightness_key, contrast_key, saturation_key = jax.random.split(key, 5)
    view = _bilinear_resize_crop(image, crop_key)
    view = horizontal_flip(view, flip_key)
    brightness = jax.random.uniform(brightness_key, (), minval=-0.15, maxval=0.15)
    view = view + brightness
    contrast = jax.random.uniform(contrast_key, (), minval=0.8, maxval=1.2)
    mean = jnp.mean(view)
    view = (view - mean) * contrast + mean
    saturation = jax.random.uniform(saturation_key, (), minval=0.8, maxval=1.2)
    luminance = jnp.sum(view * jnp.asarray([0.299, 0.587, 0.114], dtype=view.dtype), axis=-1, keepdims=True)
    view = luminance + saturation * (view - luminance)
    return jnp.clip(view, -1.0, 1.0)


def paired_contrastive_views(images: jax.Array, keys: jax.Array) -> jax.Array:
    """Return (batch, 2, height, width, channels) deterministic keyed views."""
    if images.ndim != 4 or images.shape[-1] != 3 or keys.shape != (images.shape[0], 2, 2):
        raise ValueError("expected NHWC RGB images and two JAX keys per image")
    flat_images = jnp.repeat(images, 2, axis=0)
    flat_keys = keys.reshape((-1, 2))
    transformed = jax.vmap(contrastive_view)(flat_images, flat_keys)
    return transformed.reshape((images.shape[0], 2) + images.shape[1:])
