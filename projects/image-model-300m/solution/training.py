"""Deterministic optimizer and no-drop ID-stream primitives for the pilot."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class AdamWState:
    step: jax.Array
    first_moment: Any
    second_moment: Any


@dataclasses.dataclass
class IDStreamState:
    """Serialized state for a concatenated sequence of shuffled ID epochs."""

    epoch: int
    cursor: int
    permutation: np.ndarray
    seed: int
    example_count: int
    generator: str = "numpy.PCG64"


def adamw_init(params: Any) -> AdamWState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamWState(jnp.asarray(0, dtype=jnp.int32), zeros, zeros)


def _global_norm(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32))) for leaf in leaves))


def adamw_update(
    params: Any,
    gradients: Any,
    state: AdamWState,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 1e-4,
    clip_norm: float = 1.0,
) -> tuple[Any, AdamWState, dict[str, jax.Array]]:
    """Apply clipped AdamW; decay only matrix and higher-rank leaves.

    Biases and normalization vectors are rank one, so they are not decayed.
    Embedding and convolution kernels are rank two or higher and are decayed.
    """
    if learning_rate <= 0 or epsilon <= 0 or clip_norm <= 0:
        raise ValueError("learning rate, epsilon, and clip norm must be positive")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or weight_decay < 0:
        raise ValueError("invalid AdamW coefficients")
    grad_norm = _global_norm(gradients)
    scale = jnp.minimum(1.0, jnp.asarray(clip_norm, jnp.float32) / jnp.maximum(grad_norm, 1e-12))
    clipped = jax.tree_util.tree_map(lambda value: value * scale, gradients)
    step = state.step + jnp.asarray(1, jnp.int32)
    first = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, state.first_moment, clipped)
    second = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1 - beta2) * jnp.square(g), state.second_moment, clipped)
    correction1 = 1 - beta1 ** step.astype(jnp.float32)
    correction2 = 1 - beta2 ** step.astype(jnp.float32)

    def apply(param, m, v):
        estimate = (m / correction1) / (jnp.sqrt(v / correction2) + epsilon)
        decay = weight_decay * param if param.ndim >= 2 else jnp.zeros_like(param)
        return param - learning_rate * (estimate + decay)

    updated = jax.tree_util.tree_map(apply, params, first, second)
    new_state = AdamWState(step, first, second)
    return updated, new_state, {"gradient_norm": grad_norm, "clip_scale": scale}


def _permutation(size: int, seed: int, epoch: int) -> np.ndarray:
    sequence = np.random.SeedSequence([int(seed), int(epoch)])
    generator = np.random.Generator(np.random.PCG64(sequence))
    return generator.permutation(size).astype(np.int32)


def init_id_stream(example_count: int, seed: int) -> IDStreamState:
    if example_count <= 0:
        raise ValueError("example_count must be positive")
    return IDStreamState(0, 0, _permutation(example_count, seed, 0), int(seed), int(example_count))


def next_id_batch(state: IDStreamState, batch_size: int) -> tuple[np.ndarray, IDStreamState, list[dict[str, int]]]:
    """Consume consecutive shuffled IDs, carrying partial tails across epochs."""
    if batch_size <= 0 or state.example_count <= 0:
        raise ValueError("batch size and example count must be positive")
    if state.permutation.shape != (state.example_count,) or not np.array_equal(
        np.sort(state.permutation), np.arange(state.example_count, dtype=np.int32)
    ):
        raise ValueError("sampler permutation is not a bijection of the fit IDs")
    remaining = batch_size
    pieces = []
    epoch_spans = []
    epoch = state.epoch
    cursor = state.cursor
    permutation = state.permutation
    while remaining:
        available = state.example_count - cursor
        take = min(available, remaining)
        pieces.append(permutation[cursor : cursor + take])
        epoch_spans.append({"epoch": epoch, "start": cursor, "count": take})
        cursor += take
        remaining -= take
        if cursor == state.example_count:
            epoch += 1
            cursor = 0
            permutation = _permutation(state.example_count, state.seed, epoch)
    return (
        np.concatenate(pieces),
        IDStreamState(epoch, cursor, permutation, state.seed, state.example_count),
        epoch_spans,
    )


def _identity_digest(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _id_list_digest(ids: list[str]) -> str:
    encoded = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stream_ids(fit_ids: list[str], seed: int, count: int) -> list[str]:
    result = []
    epoch = 0
    while len(result) < count:
        permutation = _permutation(len(fit_ids), seed, epoch)
        take = min(count - len(result), len(permutation))
        result.extend(fit_ids[index] for index in permutation[:take])
        epoch += 1
    return result


def _tree_to_arrays(prefix: str, tree: Any) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    leaves, structure = jax.tree_util.tree_flatten(tree)
    arrays: dict[str, np.ndarray] = {}
    descriptions = []
    for index, leaf in enumerate(leaves):
        value = np.asarray(leaf)
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"nonfinite {prefix} leaf {index}")
        key = f"{prefix}_{index:05d}"
        arrays[key] = value
        descriptions.append({"key": key, "shape": list(value.shape), "dtype": value.dtype.str})
    return arrays, descriptions


def _restore_leaf(template: Any, value: np.ndarray) -> Any:
    if isinstance(template, jax.Array):
        return jnp.asarray(value)
    if isinstance(template, np.ndarray):
        return value.copy()
    if isinstance(template, np.generic):
        return value[()] if value.shape == () else value.copy()
    if np.isscalar(template):
        return value.item() if value.shape == () else value.copy()
    return jnp.asarray(value)


def save_checkpoint(
    path: str | Path,
    *,
    params: Any,
    optimizer: AdamWState,
    dropout_key: jax.Array,
    augmentation_key: jax.Array,
    mask_key: jax.Array,
    contrastive_key: jax.Array,
    sampler: IDStreamState,
    identity: dict[str, Any],
    fit_ids: list[str],
    consumed_ids: list[str],
) -> dict[str, Any]:
    """Write a non-pickle NPZ and a source/recipe-bound JSON sidecar."""
    destination = Path(path).expanduser().resolve()
    npz_path = destination.with_suffix(".npz")
    sidecar = destination.with_suffix(".json")
    if destination.exists() or npz_path.exists() or sidecar.exists():
        raise FileExistsError(destination)
    if len(fit_ids) != sampler.example_count or len(set(fit_ids)) != len(fit_ids):
        raise ValueError("fit ID inventory must be unique and match the sampler size")
    if identity.get("fit_ids_sha256") != _id_list_digest(fit_ids):
        raise ValueError("fit ID inventory differs from checkpoint identity")
    if identity.get("fit_order_seed") is not None and int(identity["fit_order_seed"]) != sampler.seed:
        raise ValueError("sampler seed differs from checkpoint identity")
    if consumed_ids != _stream_ids(fit_ids, sampler.seed, len(consumed_ids)):
        raise ValueError("consumed IDs do not match the deterministic sampler stream")
    if not 0 <= sampler.cursor < sampler.example_count or sampler.epoch < 0:
        raise ValueError("sampler epoch or cursor is outside its valid range")
    expected_epoch, expected_cursor = divmod(len(consumed_ids), sampler.example_count)
    if (sampler.epoch, sampler.cursor) != (expected_epoch, expected_cursor):
        raise ValueError("sampler position does not match consumed ID count")
    if not np.array_equal(sampler.permutation, _permutation(sampler.example_count, sampler.seed, sampler.epoch)):
        raise ValueError("sampler permutation does not match its deterministic epoch seed")
    batch_size = identity.get("batch_size")
    if batch_size is not None:
        expected_count = int(optimizer.step) * int(batch_size)
        if len(consumed_ids) != expected_count:
            raise ValueError("consumed ID count does not match optimizer step and batch size")
        if expected_count != len(consumed_ids):
            raise ValueError("sampler position does not match optimizer step and consumed IDs")
    if identity.get("schedule_total_steps") is not None and int(optimizer.step) > int(identity["schedule_total_steps"]):
        raise ValueError("optimizer step exceeds its source-bound schedule")
    if identity.get("batch_size") is not None and identity.get("schedule_total_steps") is not None:
        expected_count = int(optimizer.step) * int(identity["batch_size"])
        if len(consumed_ids) != expected_count:
            raise ValueError("consumed ID count does not match optimizer step and batch size")
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    descriptions: dict[str, list[dict[str, Any]]] = {}
    for name, tree in (("params", params), ("first", optimizer.first_moment), ("second", optimizer.second_moment)):
        values, metadata = _tree_to_arrays(name, tree)
        arrays.update(values)
        descriptions[name] = metadata
    for name, key in (
        ("dropout_key", dropout_key),
        ("augmentation_key", augmentation_key),
        ("mask_key", mask_key),
        ("contrastive_key", contrastive_key),
    ):
        arrays[name] = np.asarray(key)
    arrays["sampler_permutation"] = np.asarray(sampler.permutation, dtype=np.int32)
    arrays["optimizer_step"] = np.asarray(optimizer.step, dtype=np.int32)
    npz_fd, npz_temp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".npz.tmp", dir=destination.parent)
    sidecar_fd, sidecar_temp_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".json.tmp", dir=destination.parent
    )
    npz_temp, sidecar_temp = Path(npz_temp_name), Path(sidecar_temp_name)
    try:
        with os.fdopen(npz_fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        archive_digest = hashlib.sha256(npz_temp.read_bytes()).hexdigest()
        metadata = {
            "format": "jaxpathways-image-checkpoint.v1",
            "npz_sha256": archive_digest,
            "identity": identity,
            "identity_sha256": _identity_digest(identity),
            "trees": descriptions,
            "sampler": {
                "epoch": sampler.epoch,
                "cursor": sampler.cursor,
                "seed": sampler.seed,
                "example_count": sampler.example_count,
                "generator": sampler.generator,
            },
            "consumed_id_count": len(consumed_ids),
            "consumed_ids_sha256": hashlib.sha256("\n".join(consumed_ids).encode()).hexdigest(),
            "consumed_ids": consumed_ids,
        }
        with os.fdopen(sidecar_fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if destination.exists() or npz_path.exists() or sidecar.exists():
            raise FileExistsError(destination)
        os.replace(npz_temp, npz_path)
        try:
            os.replace(sidecar_temp, sidecar)
        except Exception:
            npz_path.unlink(missing_ok=True)
            raise
    finally:
        npz_temp.unlink(missing_ok=True)
        sidecar_temp.unlink(missing_ok=True)
    return {"npz": str(npz_path), "metadata": str(sidecar), "npz_sha256": archive_digest}


def restore_checkpoint(
    path: str | Path,
    *,
    params_template: Any,
    identity: dict[str, Any],
    fit_ids: list[str],
) -> dict[str, Any]:
    """Validate bytes, source identity, tree semantics, keys and sampler state."""
    prefix = Path(path).expanduser().resolve()
    npz_path = prefix.with_suffix(".npz")
    sidecar = prefix.with_suffix(".json")
    metadata = json.loads(sidecar.read_text())
    if metadata.get("format") != "jaxpathways-image-checkpoint.v1":
        raise ValueError("unsupported checkpoint format")
    if metadata.get("identity") != identity or metadata.get("identity_sha256") != _identity_digest(identity):
        raise ValueError("checkpoint source/recipe identity differs")
    if hashlib.sha256(npz_path.read_bytes()).hexdigest() != metadata.get("npz_sha256"):
        raise ValueError("checkpoint NPZ checksum mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        loaded = {name: archive[name] for name in archive.files}
    tree_names = set(metadata.get("trees", {}))
    if tree_names != {"params", "first", "second"}:
        raise ValueError("checkpoint tree inventory is incomplete or contains unknown trees")
    declared_keys = {
        description["key"]
        for tree_descriptions in metadata.get("trees", {}).values()
        for description in tree_descriptions
    }
    declared_keys.update(
        ("dropout_key", "augmentation_key", "mask_key", "contrastive_key", "sampler_permutation", "optimizer_step")
    )
    if set(loaded) != declared_keys:
        raise ValueError("checkpoint has missing or extra state leaves")
    expected = {"params": params_template}
    output_trees = {}
    for name, template in expected.items():
        leaves, treedef = jax.tree_util.tree_flatten(template)
        descriptions = metadata["trees"].get(name, [])
        if len(descriptions) != len(leaves):
            raise ValueError(f"{name} leaf count differs from the model template")
        rebuilt = []
        for index, (description, template_leaf) in enumerate(zip(descriptions, leaves)):
            if description.get("key") != f"{name}_{index:05d}":
                raise ValueError(f"{name} leaf names or order differ from the canonical tree")
            value = loaded.get(description["key"])
            expected_array = np.asarray(template_leaf)
            if value is None or list(value.shape) != list(expected_array.shape) or value.dtype != expected_array.dtype:
                raise ValueError(f"{name} leaf {index} shape or dtype differs from template")
            if description.get("shape") != list(value.shape) or description.get("dtype") != value.dtype.str:
                raise ValueError(f"{name} leaf {index} metadata shape or dtype differs from its array")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"{name} leaf {index} is nonfinite")
            rebuilt.append(_restore_leaf(template_leaf, value))
        output_trees[name] = jax.tree_util.tree_unflatten(treedef, rebuilt)
    params = output_trees["params"]
    first_template = jax.tree_util.tree_map(jnp.zeros_like, params)
    second_template = jax.tree_util.tree_map(jnp.zeros_like, params)
    for name, template in (("first", first_template), ("second", second_template)):
        leaves, treedef = jax.tree_util.tree_flatten(template)
        descriptions = metadata["trees"].get(name, [])
        if len(descriptions) != len(leaves):
            raise ValueError(f"{name} moment leaf count differs from model template")
        restored = []
        for index, (description, template_leaf) in enumerate(zip(descriptions, leaves)):
            if description.get("key") != f"{name}_{index:05d}":
                raise ValueError(f"{name} moment names or order differ from the canonical tree")
            value = loaded.get(description["key"])
            expected_array = np.asarray(template_leaf)
            if value is None or value.shape != expected_array.shape or value.dtype != expected_array.dtype:
                raise ValueError(f"{name} moment {index} shape or dtype differs from template")
            if description.get("shape") != list(value.shape) or description.get("dtype") != value.dtype.str:
                raise ValueError(f"{name} moment {index} metadata shape or dtype differs from its array")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} moment {index} is nonfinite")
            if name == "second" and np.any(value < 0):
                raise ValueError(f"second moment {index} contains a negative variance")
            restored.append(jnp.asarray(value))
        output_trees[name] = jax.tree_util.tree_unflatten(treedef, restored)
    sampler_info = metadata.get("sampler", {})
    count = int(sampler_info.get("example_count", 0))
    permutation = loaded.get("sampler_permutation")
    if count <= 0 or permutation is None or permutation.dtype != np.int32 or permutation.shape != (count,):
        raise ValueError("checkpoint sampler permutation is malformed")
    if not np.array_equal(np.sort(permutation), np.arange(count, dtype=np.int32)):
        raise ValueError("checkpoint sampler permutation is not a bijection")
    epoch, cursor = int(sampler_info.get("epoch", -1)), int(sampler_info.get("cursor", -1))
    if epoch < 0 or not 0 <= cursor < count:
        raise ValueError("checkpoint sampler position is invalid")
    sampler_seed = int(sampler_info.get("seed", -1))
    if (
        len(fit_ids) != count
        or len(set(fit_ids)) != count
        or identity.get("fit_ids_sha256") != _id_list_digest(fit_ids)
    ):
        raise ValueError("restore fit ID inventory differs from checkpoint identity")
    if identity.get("fit_order_seed") is not None and sampler_seed != int(identity["fit_order_seed"]):
        raise ValueError("checkpoint sampler seed differs from identity")
    if not np.array_equal(permutation, _permutation(count, sampler_seed, epoch)):
        raise ValueError("checkpoint permutation differs from deterministic epoch seed")
    keys = {}
    for name in ("dropout_key", "augmentation_key", "mask_key", "contrastive_key"):
        value = loaded.get(name)
        if value is None or value.shape != (2,) or value.dtype != np.uint32:
            raise ValueError(f"checkpoint {name} has an invalid shape or dtype")
        keys[name] = jnp.asarray(value)
    step_array = loaded.get("optimizer_step")
    if step_array is None or step_array.shape != () or step_array.dtype != np.int32 or int(step_array) < 0:
        raise ValueError("checkpoint optimizer step is invalid")
    schedule_total = identity.get("schedule_total_steps")
    if schedule_total is not None and int(step_array) > int(schedule_total):
        raise ValueError("checkpoint optimizer step exceeds its source-bound schedule")
    if sampler_info.get("generator") != "numpy.PCG64":
        raise ValueError("checkpoint sampler generator is unsupported")
    consumed_ids = metadata.get("consumed_ids")
    if not isinstance(consumed_ids, list) or len(consumed_ids) != metadata.get("consumed_id_count"):
        raise ValueError("checkpoint consumed-ID trace is malformed")
    if hashlib.sha256("\n".join(consumed_ids).encode()).hexdigest() != metadata.get("consumed_ids_sha256"):
        raise ValueError("checkpoint consumed-ID trace checksum mismatch")
    if consumed_ids != _stream_ids(fit_ids, sampler_seed, len(consumed_ids)):
        raise ValueError("checkpoint consumed IDs do not match the deterministic sampler sequence")
    batch_size = identity.get("batch_size")
    if batch_size is not None:
        consumed_count = int(step_array) * int(batch_size)
        if len(consumed_ids) != consumed_count:
            raise ValueError("checkpoint consumed IDs disagree with optimizer step and batch size")
        expected_epoch, expected_cursor = divmod(consumed_count, count)
        if (epoch, cursor) != (expected_epoch, expected_cursor):
            raise ValueError("checkpoint sampler position disagrees with consumed batch count")
    elif len(consumed_ids) != epoch * count + cursor:
        raise ValueError("checkpoint sampler position disagrees with consumed ID count")
    return {
        "params": params,
        "optimizer": AdamWState(jnp.asarray(step_array), output_trees["first"], output_trees["second"]),
        "keys": keys,
        "sampler": IDStreamState(epoch, cursor, permutation.copy(), sampler_seed, count, sampler_info["generator"]),
        "metadata": metadata,
    }
