"""Source-bound update loops for the small image-representation pilot.

The public entry point in ``run.py`` verifies the recipe and data before it
loads training arrays. This module deliberately contains no corpus paths or
download behavior so its numerical steps can be reviewed independently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

import augment
import training

ROOT_SEEDS = {
    "fit_order": 271828,
    "augmentation": 314159,
    "contrastive_view": 173205,
    "dropout": 161803,
    "mask": 141421,
}


def stage_key(root_seed: int, stage_id: str) -> jax.Array:
    """Derive a stable stage-specific key without Python's randomized hash."""
    suffix = int.from_bytes(
        hashlib.sha256(stage_id.encode("utf-8")).digest()[:4], "big"
    )
    return jax.random.fold_in(jax.random.PRNGKey(int(root_seed)), suffix)


def _keys_for_ids(
    root_key: jax.Array, step: int, item_indices: np.ndarray, view_count: int
) -> jax.Array:
    step_key = jax.random.fold_in(root_key, int(step))
    by_item = [jax.random.fold_in(step_key, int(index)) for index in item_indices]
    if view_count == 1:
        return jnp.stack(by_item)
    return jnp.stack(
        [
            jnp.stack([jax.random.fold_in(key, view) for view in range(view_count)])
            for key in by_item
        ]
    )


def _positive_matrix(ids: list[str], phash_by_id: dict[str, str]) -> jax.Array:
    relation = np.eye(len(ids), dtype=bool)
    values = [int(phash_by_id[item], 16) for item in ids]
    for left in range(len(ids)):
        for right in range(left + 1, len(ids)):
            if (values[left] ^ values[right]).bit_count() <= 6:
                relation[left, right] = relation[right, left] = True
    return jnp.asarray(relation)


def _positive_matrix_from_pairs(
    ids: list[str], known_positive_pairs: set[tuple[str, str]]
) -> np.ndarray:
    relation = np.eye(len(ids), dtype=bool)
    for left in range(len(ids)):
        for right in range(left + 1, len(ids)):
            pair = tuple(sorted((ids[left], ids[right])))
            if pair in known_positive_pairs:
                relation[left, right] = relation[right, left] = True
    return relation


def train_updates(
    *,
    stage_id: str,
    params: Any,
    loss_fn: Callable[
        [Any, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], Any
    ],
    images_uint8: np.ndarray,
    labels: np.ndarray,
    fit_ids: list[str],
    phash_by_id: dict[str, str],
    known_positive_pairs: set[tuple[str, str]] | None = None,
    mode: str,
    batch_size: int,
    updates: int,
    learning_rate: float,
    checkpoint_directory: str | Path,
    identity: dict[str, Any],
    resume_prefix: str | Path | None = None,
    stop_after: int | None = None,
) -> dict[str, Any]:
    """Run a fully keyed stage, checkpointing complete state after every update.

    ``loss_fn`` receives the trainable tree, one or two augmented image views,
    labels, dropout key, MAE mask key, and source-positive matrix. Unused
    arguments remain shape-valid and are ignored by the selected objective.
    Checkpoints live only under the caller-selected run directory.
    """
    if mode not in {"flip", "contrastive", "none"}:
        raise ValueError("mode must be flip, contrastive, or none")
    if batch_size <= 0 or updates <= 0 or learning_rate <= 0:
        raise ValueError("batch size, update count, and learning rate must be positive")
    images = np.asarray(images_uint8)
    targets = np.asarray(labels)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (64, 64, 3):
        raise ValueError("training inputs must be uint8 NHWC 64x64 RGB")
    if targets.shape != (len(fit_ids),) or images.shape[0] != len(fit_ids):
        raise ValueError("fit image, label, and ID inventories differ")
    if len(set(fit_ids)) != len(fit_ids):
        raise ValueError("fit IDs must be unique")
    if mode == "contrastive" and any(
        image_id not in phash_by_id for image_id in fit_ids
    ):
        raise ValueError(
            "contrastive mode requires pinned pHash metadata for every fit image"
        )
    if mode == "contrastive" and known_positive_pairs is not None:
        fit_id_set = set(fit_ids)
        if any(
            left == right or left not in fit_id_set or right not in fit_id_set
            for left, right in known_positive_pairs
        ):
            raise ValueError("source-positive pairs must contain distinct fit IDs")

    fit_order = sorted(range(len(fit_ids)), key=lambda index: fit_ids[index])
    sorted_ids = [fit_ids[index] for index in fit_order]
    fit_id_digest = training._id_list_digest(sorted_ids)
    identity = {
        **identity,
        "stage_id": stage_id,
        "fit_ids_sha256": fit_id_digest,
        "fit_order_seed": ROOT_SEEDS["fit_order"],
        "batch_size": batch_size,
        "schedule_total_steps": updates,
        "learning_rate": learning_rate,
        "mode": mode,
        "root_seeds": dict(ROOT_SEEDS),
    }
    if known_positive_pairs is not None:
        pair_digest = hashlib.sha256(
            "\n".join("\t".join(pair) for pair in sorted(known_positive_pairs)).encode()
        ).hexdigest()
        identity["source_positive_pairs_sha256"] = pair_digest
    sorted_images = images[fit_order]
    sorted_labels = targets[fit_order]
    sampler = training.init_id_stream(len(sorted_ids), ROOT_SEEDS["fit_order"])
    optimizer = training.adamw_init(params)
    consumed_ids: list[str] = []
    keys = {
        "dropout_key": stage_key(ROOT_SEEDS["dropout"], stage_id),
        "augmentation_key": stage_key(ROOT_SEEDS["augmentation"], stage_id),
        "mask_key": stage_key(ROOT_SEEDS["mask"], stage_id),
        "contrastive_key": stage_key(ROOT_SEEDS["contrastive_view"], stage_id),
    }
    if resume_prefix is not None:
        restored = training.restore_checkpoint(
            resume_prefix, params_template=params, identity=identity, fit_ids=sorted_ids
        )
        params = restored["params"]
        optimizer = restored["optimizer"]
        sampler = restored["sampler"]
        keys = restored["keys"]
        consumed_ids = list(restored["metadata"]["consumed_ids"])

    checkpoint_root = (
        Path(resume_prefix).expanduser().resolve().parent
        if resume_prefix is not None
        else Path(checkpoint_directory).expanduser().resolve() / stage_id
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    trace_events = checkpoint_root / "trace-events"
    trace_events.mkdir(parents=True, exist_ok=True)
    trace = []
    dropout_root = keys["dropout_key"]
    augmentation_root = keys["augmentation_key"]
    mask_root = keys["mask_key"]
    contrastive_root = keys["contrastive_key"]
    start_step = int(optimizer.step)
    for completed_step in range(1, start_step + 1):
        event_path = trace_events / f"step-{completed_step:06d}.json"
        if not event_path.is_file():
            raise ValueError(
                f"checkpoint step {completed_step} has no durable trace event"
            )
        event = json.loads(event_path.read_text())
        if (
            event.get("stage_id") != stage_id
            or int(event.get("step", -1)) != completed_step
        ):
            raise ValueError(
                "saved trace event does not match its checkpoint stage/update"
            )
        trace.append(event)
    if start_step > updates:
        raise ValueError("restored optimizer step exceeds the declared stage schedule")
    stop_step = updates if stop_after is None else int(stop_after)
    if not start_step <= stop_step <= updates:
        raise ValueError(
            "stop_after must be between the restored step and the declared schedule"
        )

    def objective(
        current_params,
        current_images,
        current_labels,
        dropout_key,
        mask_key,
        positive_matrix,
    ):
        return loss_fn(
            current_params,
            current_images,
            current_labels,
            dropout_key,
            mask_key,
            positive_matrix,
        )

    value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))

    for step in range(start_step, stop_step):
        sampled_indices, sampler, spans = training.next_id_batch(sampler, batch_size)
        selected_ids = [sorted_ids[int(index)] for index in sampled_indices]
        consumed_ids.extend(selected_ids)
        batch = sorted_images[sampled_indices].astype(np.float32) / np.float32(
            127.5
        ) - np.float32(1.0)
        labels_batch = sorted_labels[sampled_indices].astype(np.int32)
        local_item_indices = sampled_indices
        if mode == "flip":
            view_keys = _keys_for_ids(augmentation_root, step, local_item_indices, 1)
            model_images = augment.supervised_views(jnp.asarray(batch), view_keys)
        elif mode == "contrastive":
            view_keys = _keys_for_ids(contrastive_root, step, local_item_indices, 2)
            model_images = augment.paired_contrastive_views(
                jnp.asarray(batch), view_keys
            )
        else:
            model_images = jnp.asarray(batch)
        positive = (
            _positive_matrix(selected_ids, phash_by_id)
            if mode == "contrastive"
            else jnp.eye(batch_size, dtype=bool)
        )
        if mode == "contrastive" and known_positive_pairs is not None:
            expected_positive = _positive_matrix_from_pairs(
                selected_ids, known_positive_pairs
            )
            if not np.array_equal(np.asarray(positive), expected_positive):
                raise ValueError(
                    "pHash runner relation differs from source-enriched pipeline relation"
                )
        dropout_key = jax.random.fold_in(dropout_root, step)
        mask_key = jax.random.fold_in(mask_root, step)

        (loss, objective_metrics), gradients = value_and_grad(
            params,
            model_images,
            jnp.asarray(labels_batch),
            dropout_key,
            mask_key,
            positive,
        )
        loss_value = float(loss)
        if not np.isfinite(loss_value):
            raise FloatingPointError(
                f"{stage_id} produced nonfinite loss at update {step + 1}"
            )
        grad_leaves = jax.tree_util.tree_leaves(gradients)
        if any(not np.isfinite(np.asarray(leaf)).all() for leaf in grad_leaves):
            raise FloatingPointError(
                f"{stage_id} produced nonfinite gradients at update {step + 1}"
            )
        params, optimizer, update_info = training.adamw_update(
            params,
            gradients,
            optimizer,
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=1e-4,
            clip_norm=1.0,
        )
        current_step = int(optimizer.step)
        if not isinstance(objective_metrics, dict):
            raise TypeError(
                "loss_fn must return (loss, objective_metrics) with explicit counters"
            )
        mae_targets = int(
            np.asarray(objective_metrics.get("selected_mae_target_values", 0))
        )
        if mae_targets < 0:
            raise ValueError("objective target-value counters cannot be negative")
        view_count = int(model_images.shape[1]) if model_images.ndim == 5 else 1
        model_view_count = len(selected_ids) * view_count
        record = {
            "stage_id": stage_id,
            "step": current_step,
            "loss": loss_value,
            "gradient_norm": float(update_info["gradient_norm"]),
            "clip_scale": float(update_info["clip_scale"]),
            "sampled_ids": selected_ids,
            "epoch_spans": spans,
            "input_image_count": len(selected_ids),
            "model_view_count": model_view_count,
            "input_pixel_values": int(model_images.size),
            "selected_mae_target_values": mae_targets,
            "source_positive_relation_checked": bool(
                mode == "contrastive" and known_positive_pairs is not None
            ),
            "source_positive_pair_count": (
                len(known_positive_pairs)
                if mode == "contrastive" and known_positive_pairs is not None
                else None
            ),
        }
        trace.append(record)
        training.save_checkpoint(
            checkpoint_root / f"step-{current_step:06d}",
            params=params,
            optimizer=optimizer,
            dropout_key=dropout_root,
            augmentation_key=augmentation_root,
            mask_key=mask_root,
            contrastive_key=contrastive_root,
            sampler=sampler,
            identity=identity,
            fit_ids=sorted_ids,
            consumed_ids=consumed_ids,
        )
        event_path = trace_events / f"step-{current_step:06d}.json"
        temporary_event = event_path.with_suffix(".json.tmp")
        temporary_event.write_text(json.dumps(record, sort_keys=True) + "\n")
        temporary_event.replace(event_path)

    checkpoint_prefix = checkpoint_root / f"step-{int(optimizer.step):06d}"
    if start_step == stop_step and start_step > 0:
        # Resuming a completed stage still loads its serialized state. The
        # checkpoint is authoritative; in-memory state is only the template.
        pass
    restored = training.restore_checkpoint(
        checkpoint_prefix,
        params_template=params,
        identity=identity,
        fit_ids=sorted_ids,
    )
    for name in ("params",):
        for actual, expected in zip(
            jax.tree_util.tree_leaves(restored[name]), jax.tree_util.tree_leaves(params)
        ):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    if int(restored["optimizer"].step) != int(optimizer.step):
        raise ValueError(
            "serialized optimizer step differs from the in-memory stage boundary"
        )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(
            (restored["optimizer"].first_moment, restored["optimizer"].second_moment)
        ),
        jax.tree_util.tree_leaves((optimizer.first_moment, optimizer.second_moment)),
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    if (
        restored["sampler"].epoch != sampler.epoch
        or restored["sampler"].cursor != sampler.cursor
        or not np.array_equal(restored["sampler"].permutation, sampler.permutation)
        or restored["metadata"]["consumed_ids"] != consumed_ids
    ):
        raise ValueError(
            "serialized sampler boundary differs from the in-memory stage boundary"
        )
    for key_name in keys:
        np.testing.assert_array_equal(
            np.asarray(restored["keys"][key_name]), np.asarray(keys[key_name])
        )
    params = restored["params"]
    optimizer = restored["optimizer"]
    sampler = restored["sampler"]
    keys = restored["keys"]
    consumed_ids = list(restored["metadata"]["consumed_ids"])
    return {
        "params": params,
        "optimizer": optimizer,
        "sampler": sampler,
        "keys": keys,
        "consumed_ids": consumed_ids,
        "trace": trace,
        "identity": identity,
        "checkpoint_prefix": str(checkpoint_prefix),
        "complete": int(optimizer.step) == updates,
    }


def stage_source_identity(
    recipe_path: str | Path, source_files: list[Path], parent_digest: str | None
) -> dict[str, Any]:
    recipe_bytes = Path(recipe_path).read_bytes()
    sources = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    return {
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "stage_sources_sha256": sources,
        "parent_checkpoint_sha256": parent_digest,
    }
