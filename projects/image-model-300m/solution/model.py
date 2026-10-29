"""Small, real bidirectional ViT and CNN used by the flower-photo pilot."""

from __future__ import annotations

import math
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
    """Return row-major NHWC patches, preserving within-patch H,W,C order."""
    if images.ndim != 4:
        raise ValueError("images must have shape (batch, height, width, channels)")
    batch, height, width, channels = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("image dimensions must be divisible by patch_size")
    grid_h, grid_w = height // patch_size, width // patch_size
    blocks = images.reshape(batch, grid_h, patch_size, grid_w, patch_size, channels)
    blocks = jnp.transpose(blocks, (0, 1, 3, 2, 4, 5))
    return blocks.reshape(batch, grid_h * grid_w, patch_size * patch_size * channels)


def unpatchify(patches: jax.Array, image_size: int = 64, patch_size: int = 8, channels: int = 3) -> jax.Array:
    """Invert :func:`patchify` for row-major NHWC patch vectors."""
    grid_h = grid_w = image_size // patch_size
    expected = patch_size * patch_size * channels
    if patches.ndim != 3 or patches.shape[1:] != (grid_h * grid_w, expected):
        raise ValueError("patch shape does not match the requested image and patch dimensions")
    batch = patches.shape[0]
    blocks = patches.reshape(batch, grid_h, grid_w, patch_size, patch_size, channels)
    blocks = jnp.transpose(blocks, (0, 1, 3, 2, 4, 5))
    return blocks.reshape(batch, image_size, image_size, channels)


def _linear_init(key: jax.Array, fan_in: int, fan_out: int, dtype=jnp.float32) -> tuple[jax.Array, jax.Array]:
    limit = jnp.sqrt(jnp.asarray(6.0 / (fan_in + fan_out), dtype=dtype))
    weight_key, bias_key = jax.random.split(key)
    weight = jax.random.uniform(weight_key, (fan_in, fan_out), dtype=dtype, minval=-limit, maxval=limit)
    bias = jax.random.uniform(bias_key, (fan_out,), dtype=dtype, minval=-limit, maxval=limit)
    return weight, bias


def init_vit(key: jax.Array, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Initialize a pre-LN ViT encoder with a class token and learned 2-D grid positions."""
    cfg = dict(PILOT_CONFIG if config is None else config)
    size = int(cfg["image_size"])
    patch = int(cfg["patch_size"])
    channels = int(cfg["channels"])
    width = int(cfg["width"])
    heads = int(cfg["heads"])
    mlp_dim = int(cfg["mlp_dim"])
    layers = int(cfg["layers"])
    if size <= 0 or patch <= 0 or size % patch or channels <= 0 or width <= 0 or heads <= 0 or width % heads:
        raise ValueError("invalid image, patch, channel, width, or head dimensions")
    if mlp_dim <= 0 or layers <= 0:
        raise ValueError("mlp_dim and layers must be positive")
    keys = iter(jax.random.split(key, 2 + layers * 5))
    patch_kernel, patch_bias = _linear_init(next(keys), patch * patch * channels, width)
    grid_tokens = (size // patch) ** 2
    cls_token = 0.02 * jax.random.normal(next(keys), (1, 1, width), dtype=jnp.float32)
    position_embedding = 0.02 * jax.random.normal(next(keys), (1, grid_tokens + 1, width), dtype=jnp.float32)
    blocks = []
    for _ in range(layers):
        qkv_kernel, qkv_bias = _linear_init(next(keys), width, width * 3)
        out_kernel, out_bias = _linear_init(next(keys), width, width)
        mlp_in_kernel, mlp_in_bias = _linear_init(next(keys), width, mlp_dim)
        mlp_out_kernel, mlp_out_bias = _linear_init(next(keys), mlp_dim, width)
        blocks.append(
            {
                "ln1_scale": jnp.ones((width,), dtype=jnp.float32),
                "ln1_bias": jnp.zeros((width,), dtype=jnp.float32),
                "qkv_kernel": qkv_kernel,
                "qkv_bias": qkv_bias,
                "out_kernel": out_kernel,
                "out_bias": out_bias,
                "ln2_scale": jnp.ones((width,), dtype=jnp.float32),
                "ln2_bias": jnp.zeros((width,), dtype=jnp.float32),
                "mlp_in_kernel": mlp_in_kernel,
                "mlp_in_bias": mlp_in_bias,
                "mlp_out_kernel": mlp_out_kernel,
                "mlp_out_bias": mlp_out_bias,
            }
        )
    return {
        "patch_kernel": patch_kernel,
        "patch_bias": patch_bias,
        "cls_token": cls_token,
        "position_embedding": position_embedding,
        "blocks": tuple(blocks),
        "final_ln_scale": jnp.ones((width,), dtype=jnp.float32),
        "final_ln_bias": jnp.zeros((width,), dtype=jnp.float32),
        "config": cfg,
    }


def init_linear_head(key: jax.Array, width: int = 32, classes: int = 5) -> dict[str, jax.Array]:
    kernel, bias = _linear_init(key, width, classes)
    return {"kernel": kernel, "bias": bias}


def layer_norm(x: jax.Array, scale: jax.Array, bias: jax.Array, epsilon: float = 1e-5) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normalized = (x - mean) * jax.lax.rsqrt(variance + epsilon)
    return normalized * scale + bias


def _dropout(x: jax.Array, key: jax.Array | None, rate: float, train: bool) -> jax.Array:
    if not train or rate == 0.0:
        return x
    if key is None:
        raise ValueError("training with dropout requires an explicit PRNG key")
    keep = 1.0 - rate
    mask = jax.random.bernoulli(key, keep, x.shape)
    return jnp.where(mask, x / keep, 0.0)


def _encoder_block(
    x: jax.Array, params: dict, heads: int, epsilon: float, key: jax.Array | None, rate: float, train: bool
) -> jax.Array:
    width = x.shape[-1]
    head_width = width // heads
    keys = (None, None) if key is None else tuple(jax.random.split(key, 2))
    normalized = layer_norm(x, params["ln1_scale"], params["ln1_bias"], epsilon)
    qkv = normalized @ params["qkv_kernel"] + params["qkv_bias"]
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, heads, head_width)
    query, key_value, value = (jnp.transpose(qkv[:, :, index], (0, 2, 1, 3)) for index in range(3))
    scores = jnp.einsum("bhqd,bhkd->bhqk", query, key_value) / jnp.sqrt(jnp.asarray(head_width, dtype=x.dtype))
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
    attended = jnp.einsum("bhqk,bhkd->bhqd", weights, value)
    attended = jnp.transpose(attended, (0, 2, 1, 3)).reshape(x.shape)
    attended = attended @ params["out_kernel"] + params["out_bias"]
    x = x + _dropout(attended, keys[0], rate, train)
    normalized = layer_norm(x, params["ln2_scale"], params["ln2_bias"], epsilon)
    hidden = normalized @ params["mlp_in_kernel"] + params["mlp_in_bias"]
    hidden = jax.nn.gelu(hidden, approximate=True)
    hidden = hidden @ params["mlp_out_kernel"] + params["mlp_out_bias"]
    return x + _dropout(hidden, keys[1], rate, train)


def encode_patch_vectors(
    params: dict[str, Any],
    patch_vectors: jax.Array,
    patch_indices: jax.Array | None = None,
    *,
    dropout_key: jax.Array | None = None,
    train: bool = False,
    return_tokens: bool = False,
    allow_position_interpolation: bool = False,
) -> jax.Array:
    """Encode a complete grid or selected visible patches, returning CLS features."""
    cfg = params["config"]
    if patch_vectors.ndim != 3:
        raise ValueError("patch_vectors must have shape (batch, patches, pixels)")
    if patch_vectors.shape[2] != params["patch_kernel"].shape[0]:
        raise ValueError("patch width differs from patch projection")
    if patch_indices is None:
        grid = math.isqrt(patch_vectors.shape[1])
        position_count = params["position_embedding"].shape[1] - 1
        if patch_vectors.shape[1] != position_count:
            if not allow_position_interpolation or grid * grid != patch_vectors.shape[1]:
                raise ValueError("full-grid input has the wrong number of patches")
            old_grid = math.isqrt(position_count)
            if old_grid * old_grid != position_count:
                raise ValueError("learned patch positions do not form a square grid")
            grid_positions = params["position_embedding"][:, 1:, :].reshape(1, old_grid, old_grid, -1)
            grid_positions = jnp.transpose(grid_positions, (0, 3, 1, 2))
            grid_positions = jax.image.resize(
                grid_positions, (1, grid_positions.shape[1], grid, grid), method="bilinear"
            )
            positions = jnp.transpose(grid_positions, (0, 2, 3, 1)).reshape(1, grid * grid, -1)
        else:
            positions = params["position_embedding"][:, 1:, :]
        selected = patch_vectors
    else:
        if (
            patch_indices.ndim != 2
            or patch_indices.shape[0] != patch_vectors.shape[0]
            or patch_indices.shape[1] <= 0
            or patch_indices.shape[1] > patch_vectors.shape[1]
        ):
            raise ValueError("visible patch indices must align with batch and visible patch count")
        gather_indices = jnp.broadcast_to(patch_indices[..., None], (*patch_indices.shape, patch_vectors.shape[-1]))
        selected = jnp.take_along_axis(patch_vectors, gather_indices, axis=1)
        positions = params["position_embedding"][0, 1:][patch_indices]
    tokens = selected @ params["patch_kernel"] + params["patch_bias"]
    tokens = tokens + positions
    cls = jnp.broadcast_to(
        params["cls_token"] + params["position_embedding"][:, :1, :], (tokens.shape[0], 1, tokens.shape[-1])
    )
    tokens = jnp.concatenate((cls, tokens), axis=1)
    if train and dropout_key is None:
        raise ValueError("training requires an explicit dropout key")
    block_keys = (
        (None,) * len(params["blocks"])
        if dropout_key is None
        else tuple(jax.random.split(dropout_key, len(params["blocks"])))
    )
    for block, key in zip(params["blocks"], block_keys):
        tokens = _encoder_block(
            tokens, block, int(cfg["heads"]), float(cfg["layer_norm_epsilon"]), key, float(cfg["dropout_rate"]), train
        )
    tokens = layer_norm(tokens, params["final_ln_scale"], params["final_ln_bias"], float(cfg["layer_norm_epsilon"]))
    return tokens if return_tokens else tokens[:, 0, :]


def encode_images(
    params: dict[str, Any],
    images: jax.Array,
    *,
    dropout_key: jax.Array | None = None,
    train: bool = False,
    allow_position_interpolation: bool = False,
) -> jax.Array:
    cfg = params["config"]
    image_size = int(cfg["image_size"])
    channels = int(cfg["channels"])
    valid_shape = images.ndim == 4 and images.shape[-1] == channels and images.shape[1] == images.shape[2]
    if valid_shape and not allow_position_interpolation:
        valid_shape = images.shape[1:] == (image_size, image_size, channels)
    if valid_shape and allow_position_interpolation:
        valid_shape = images.shape[1] % int(cfg["patch_size"]) == 0
    if not valid_shape:
        raise ValueError("image tensor shape differs from the ViT configuration or interpolation policy")
    vectors = patchify(images, int(cfg["patch_size"]))
    return encode_patch_vectors(
        params, vectors, dropout_key=dropout_key, train=train, allow_position_interpolation=allow_position_interpolation
    )


def classification_logits(
    params: dict[str, Any],
    head: dict[str, jax.Array],
    images: jax.Array,
    *,
    dropout_key: jax.Array | None = None,
    train: bool = False,
    allow_position_interpolation: bool = False,
) -> jax.Array:
    features = encode_images(
        params, images, dropout_key=dropout_key, train=train, allow_position_interpolation=allow_position_interpolation
    )
    return features @ head["kernel"] + head["bias"]


def init_cnn(key: jax.Array, classes: int = 5, channels: int = 3) -> dict[str, jax.Array]:
    """Initialize a small two-convolution RGB control network."""
    keys = jax.random.split(key, 4)
    conv1_kernel = jax.random.normal(keys[0], (3, 3, channels, 16), dtype=jnp.float32) * jnp.sqrt(
        2.0 / (3 * 3 * channels)
    )
    conv2_kernel = jax.random.normal(keys[1], (3, 3, 16, 32), dtype=jnp.float32) * jnp.sqrt(2.0 / (3 * 3 * 16))
    head_kernel, head_bias = _linear_init(keys[2], 32, classes)
    return {
        "conv1_kernel": conv1_kernel,
        "conv1_bias": jnp.zeros((16,), dtype=jnp.float32),
        "conv2_kernel": conv2_kernel,
        "conv2_bias": jnp.zeros((32,), dtype=jnp.float32),
        "head_kernel": head_kernel,
        "head_bias": head_bias,
    }


def cnn_features(
    params: dict[str, jax.Array],
    images: jax.Array,
    *,
    dropout_key: jax.Array | None = None,
    train: bool = False,
) -> jax.Array:
    x = jax.lax.conv_general_dilated(
        images,
        params["conv1_kernel"],
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    x = jax.nn.relu(x + params["conv1_bias"])
    x = jax.lax.conv_general_dilated(
        x,
        params["conv2_kernel"],
        window_strides=(2, 2),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    x = jax.nn.relu(x + params["conv2_bias"])
    features = jnp.mean(x, axis=(1, 2))
    return _dropout(features, dropout_key, 0.1, train)


def cnn_logits(
    params: dict[str, jax.Array],
    images: jax.Array,
    *,
    dropout_key: jax.Array | None = None,
    train: bool = False,
) -> jax.Array:
    return (
        cnn_features(params, images, dropout_key=dropout_key, train=train) @ params["head_kernel"] + params["head_bias"]
    )


def parameter_count(tree: Any) -> int:
    """Count numeric leaves without counting the nonarray config metadata."""
    return sum(math.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(tree) if hasattr(leaf, "shape"))
