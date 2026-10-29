"""Independent NumPy float64 checks for image model forward semantics."""

from __future__ import annotations

import math

import numpy as np


def _array(value):
    return np.asarray(value, dtype=np.float64)


def patchify(images: np.ndarray, patch_size: int) -> np.ndarray:
    batch, height, width, channels = images.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    result = []
    for row in range(grid_h):
        for column in range(grid_w):
            patch = images[
                :, row * patch_size : (row + 1) * patch_size, column * patch_size : (column + 1) * patch_size, :
            ]
            result.append(patch.reshape(batch, -1))
    return np.stack(result, axis=1).astype(np.float64)


def _layer_norm(x, scale, bias, epsilon):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + epsilon) * _array(scale) + _array(bias)


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def _softmax(scores):
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def vit_features(params: dict, images: np.ndarray, patch_indices: np.ndarray | None = None) -> np.ndarray:
    cfg = params["config"]
    patches = patchify(np.asarray(images, dtype=np.float64), int(cfg["patch_size"]))
    if patch_indices is not None:
        patches = np.take_along_axis(patches, patch_indices[..., None], axis=1)
        positions = _array(params["position_embedding"])[0, 1:][patch_indices]
    else:
        positions = _array(params["position_embedding"])[0, 1:, :]
    tokens = patches @ _array(params["patch_kernel"]) + _array(params["patch_bias"])
    tokens = tokens + positions
    cls = np.broadcast_to(
        _array(params["cls_token"]) + _array(params["position_embedding"])[:, :1, :],
        (len(images), 1, int(cfg["width"])),
    )
    tokens = np.concatenate((cls, tokens), axis=1)
    heads = int(cfg["heads"])
    width = int(cfg["width"])
    head_width = width // heads
    epsilon = float(cfg["layer_norm_epsilon"])
    for block in params["blocks"]:
        normalized = _layer_norm(tokens, block["ln1_scale"], block["ln1_bias"], epsilon)
        qkv = normalized @ _array(block["qkv_kernel"]) + _array(block["qkv_bias"])
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, heads, head_width)
        query, key, value = (np.transpose(qkv[:, :, index], (0, 2, 1, 3)) for index in range(3))
        scores = np.einsum("bhqd,bhkd->bhqk", query, key) / math.sqrt(head_width)
        weights = _softmax(scores)
        attended = np.einsum("bhqk,bhkd->bhqd", weights, value)
        attended = np.transpose(attended, (0, 2, 1, 3)).reshape(tokens.shape)
        attended = attended @ _array(block["out_kernel"]) + _array(block["out_bias"])
        tokens = tokens + attended
        normalized = _layer_norm(tokens, block["ln2_scale"], block["ln2_bias"], epsilon)
        hidden = normalized @ _array(block["mlp_in_kernel"]) + _array(block["mlp_in_bias"])
        hidden = _gelu_tanh(hidden)
        hidden = hidden @ _array(block["mlp_out_kernel"]) + _array(block["mlp_out_bias"])
        tokens = tokens + hidden
    return _layer_norm(tokens[:, 0, :], params["final_ln_scale"], params["final_ln_bias"], epsilon)


def vit_logits(params: dict, head: dict, images: np.ndarray) -> np.ndarray:
    return vit_features(params, images) @ _array(head["kernel"]) + _array(head["bias"])


def _conv2d_same(x, kernel, bias, stride):
    batch, height, width, in_channels = x.shape
    kh, kw, _, out_channels = kernel.shape
    output_h = math.ceil(height / stride)
    output_w = math.ceil(width / stride)
    pad_h = max((output_h - 1) * stride + kh - height, 0)
    pad_w = max((output_w - 1) * stride + kw - width, 0)
    padded = np.pad(x, ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)))
    output = np.zeros((batch, output_h, output_w, out_channels), dtype=np.float64)
    for row in range(output_h):
        for column in range(output_w):
            window = padded[:, row * stride : row * stride + kh, column * stride : column * stride + kw, :]
            output[:, row, column, :] = np.einsum("bhwc,hwco->bo", window, kernel) + bias
    return output


def cnn_logits(params: dict, images: np.ndarray) -> np.ndarray:
    x = _conv2d_same(
        np.asarray(images, dtype=np.float64), _array(params["conv1_kernel"]), _array(params["conv1_bias"]), 1
    )
    x = np.maximum(x, 0.0)
    x = _conv2d_same(x, _array(params["conv2_kernel"]), _array(params["conv2_bias"]), 2)
    x = np.maximum(x, 0.0)
    features = np.mean(x, axis=(1, 2))
    return features @ _array(params["head_kernel"]) + _array(params["head_bias"])


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(shifted), axis=1))
    return float(np.mean(log_normalizer - shifted[np.arange(len(labels)), labels]))


def finite_difference(function, value: np.ndarray, index: tuple[int, ...], step: float = 1e-3) -> float:
    plus = value.copy()
    minus = value.copy()
    plus[index] += step
    minus[index] -= step
    return (function(plus) - function(minus)) / (2 * step)
