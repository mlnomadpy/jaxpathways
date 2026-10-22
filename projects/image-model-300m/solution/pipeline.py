"""Bounded end-to-end flower pilot, called only after strict preflight approval."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageOps

import augment
import consumer
import consumer_runtime
import evaluate
import model
import objectives
import runner
import train_steps


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class InterruptedStage(RuntimeError):
    """Checker-only deterministic stop after a durable stage checkpoint."""


_PHASH_PATTERN = re.compile(r"[0-9a-fA-F]{16}\Z")
FROZEN_CLASS_ORDER = ("daisy", "dandelion", "roses", "sunflowers", "tulips")

_SPLIT_IDENTITY_FIELDS = (
    "id",
    "label",
    "class_index",
    "creator",
    "creator_id",
    "raw_sha256",
    "decoded_pixel_sha256",
    "source_url",
)
_SOURCE_IDENTITY_FIELDS = (
    "id",
    "label",
    "creator",
    "creator_id",
    "raw_sha256",
    "pixel_sha256",
    "source_url",
    "phash",
)


def _source_metadata_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build a strict ID index from the trusted source-manifest inventory."""
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("source manifest has no image record inventory")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        image_id = record.get("id") if isinstance(record, dict) else None
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("source inventory contains an invalid image ID")
        if image_id in by_id:
            raise ValueError(
                f"source inventory contains duplicate image ID: {image_id}"
            )
        by_id[image_id] = record
    return by_id


def load_split_inputs(
    data_directory: str | Path,
    split: str,
    *,
    source_metadata_by_id: dict[str, dict[str, Any]] | None = None,
    class_order: tuple[str, ...] = FROZEN_CLASS_ORDER,
) -> dict[str, Any]:
    """Load one frozen split and enrich rows from the trusted source inventory.

    The split JSON intentionally does not duplicate perceptual hashes. Its IDs
    join to ``source-manifest.json`` exactly once, and the source record is the
    only accepted pHash authority.
    """
    if split not in {"fit", "recipe_dev", "final"}:
        raise ValueError(f"unknown frozen image split: {split}")
    root = Path(data_directory).expanduser().resolve()
    if source_metadata_by_id is None:
        manifest = json.loads((root / "source-manifest.json").read_text())
        source_metadata_by_id = _source_metadata_index(manifest)
    rows = json.loads((root / "splits" / f"{split}.json").read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{split} split rows must be a JSON list")
    if tuple(class_order) != FROZEN_CLASS_ORDER:
        raise ValueError("split class order differs from the frozen image class order")
    seen: set[str] = set()
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{split} split contains a non-object row")
        missing_split_fields = [
            field for field in _SPLIT_IDENTITY_FIELDS if field not in row
        ]
        if missing_split_fields:
            raise ValueError(
                f"{split} split row is missing identity fields {missing_split_fields}"
            )
        image_id = row.get("id") if isinstance(row, dict) else None
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(f"{split} split contains an invalid image ID")
        if image_id in seen:
            raise ValueError(f"{split} split contains duplicate image ID: {image_id}")
        seen.add(image_id)
        source_record = source_metadata_by_id.get(image_id)
        if source_record is None:
            raise ValueError(
                f"{split} split ID is missing from source inventory: {image_id}"
            )
        missing_source_fields = [
            field for field in _SOURCE_IDENTITY_FIELDS if field not in source_record
        ]
        if missing_source_fields:
            raise ValueError(
                f"source inventory record for {image_id} is missing identity fields {missing_source_fields}"
            )
        if "phash" in row:
            raise ValueError(
                f"{split} split row duplicates pHash instead of using source inventory: {image_id}"
            )
        if not isinstance(row["label"], str) or row["label"] not in class_order:
            raise ValueError(f"{split} split row has an unknown class label for {image_id}")
        expected_class_index = class_order.index(row["label"])
        if (
            isinstance(row["class_index"], bool)
            or not isinstance(row["class_index"], int)
            or row["class_index"] != expected_class_index
        ):
            raise ValueError(
                f"split class_index does not match frozen class order for {image_id}"
            )
        for split_field, source_field in (
            ("label", "label"),
            ("creator", "creator"),
            ("creator_id", "creator_id"),
            ("raw_sha256", "raw_sha256"),
            ("decoded_pixel_sha256", "pixel_sha256"),
            ("source_url", "source_url"),
        ):
            if row[split_field] != source_record[source_field]:
                raise ValueError(
                    f"split/source {split_field} mismatch for {image_id}"
                )
        phash = source_record.get("phash")
        if not isinstance(phash, str) or _PHASH_PATTERN.fullmatch(phash) is None:
            raise ValueError(
                f"source inventory has invalid 16-digit pHash for {image_id}"
            )
        enriched_rows.append({**row, "phash": phash})
    images = np.load(
        root / "arrays" / f"{split}_images.npy", mmap_mode="r", allow_pickle=False
    )
    labels = np.load(root / "arrays" / f"{split}_labels.npy", allow_pickle=False)
    if images.shape[0] != len(enriched_rows) or labels.shape != (len(enriched_rows),):
        raise ValueError(f"{split} arrays and split ID inventory differ")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{split} label array must contain integer class indices")
    for index, row in enumerate(enriched_rows):
        if int(labels[index]) != row["class_index"]:
            raise ValueError(
                f"split row/array class label mismatch for {row['id']}"
            )
    return {"images": images, "labels": labels, "rows": enriched_rows}


def _tree_digest(tree: Any) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _head(
    model_id: str, width: int, classes: int, *, model_api=model, runner_api=runner
) -> dict[str, jax.Array]:
    return model_api.init_linear_head(
        runner_api.stage_key(1729, f"head:{model_id}"), width, classes
    )


def _vit_supervised(
    encoder: dict, head: dict, static: dict, loss_name: str, *, model_api=model
):
    def loss(params, images, labels, dropout_key, _mask_key, _positive):
        value = train_steps.vit_supervised_loss(
            params["encoder"],
            params["head"],
            static,
            images,
            labels,
            dropout_key,
            model_api=model_api,
        )
        return value, {"selected_mae_target_values": jnp.asarray(0, dtype=jnp.int32)}

    return loss


def _train_vit(
    stage: dict,
    encoder: dict,
    head: dict,
    *,
    images: np.ndarray,
    labels: np.ndarray,
    fit_ids: list[str],
    phash_by_id: dict[str, str],
    output: Path,
    recipe_sha: str,
    parent_hash: str,
    mode: str = "flip",
    model_api=model,
    runner_api=runner,
    resume_prefix: str | Path | None = None,
    stop_after: int | None = None,
) -> dict:
    trainable, static = train_steps.split_vit(encoder)
    initial = {"encoder": trainable, "head": head}
    stage_id = stage["id"]
    loss_name = stage["objective"]
    loss_fn = _vit_supervised(encoder, head, static, loss_name, model_api=model_api)
    identity = {
        "recipe_sha256": recipe_sha,
        "parent_parameters_sha256": parent_hash,
        "model_config": static["config"],
        "objective": loss_name,
        "stage_spec_sha256": hashlib.sha256(
            json.dumps(stage, sort_keys=True).encode()
        ).hexdigest(),
    }
    return runner_api.train_updates(
        stage_id=stage_id,
        params=initial,
        loss_fn=loss_fn,
        images_uint8=images,
        labels=labels,
        fit_ids=fit_ids,
        phash_by_id=phash_by_id,
        mode=mode,
        batch_size=int(stage["batch_size"]),
        updates=int(stage["updates"]),
        learning_rate=float(stage["learning_rate"]),
        checkpoint_directory=output / "checkpoints",
        identity=identity,
        resume_prefix=resume_prefix,
        stop_after=stop_after,
    ) | {"static": static, "head": None}


def _train_cnn(
    stage,
    params,
    *,
    images,
    labels,
    fit_ids,
    output,
    recipe_sha,
    model_api=model,
    runner_api=runner,
    resume_prefix=None,
    stop_after=None,
):
    def loss(current, batch, targets, dropout_key, _mask_key, _positive):
        value = train_steps.cnn_supervised_loss(
            current, batch, targets, dropout_key, model_api=model_api
        )
        return value, {"selected_mae_target_values": jnp.asarray(0, dtype=jnp.int32)}

    return runner_api.train_updates(
        stage_id=stage["id"],
        params=params,
        loss_fn=loss,
        images_uint8=images,
        labels=labels,
        fit_ids=fit_ids,
        phash_by_id={},
        mode="flip",
        batch_size=int(stage["batch_size"]),
        updates=int(stage["updates"]),
        learning_rate=float(stage["learning_rate"]),
        checkpoint_directory=output / "checkpoints",
        identity={
            "recipe_sha256": recipe_sha,
            "objective": stage["objective"],
            "parent": stage["parent"],
        },
        resume_prefix=resume_prefix,
        stop_after=stop_after,
    )


def _positive_pairs(rows: list[dict]) -> set[tuple[str, str]]:
    ordered = sorted(rows, key=lambda row: row["id"])
    hashes = []
    for row in ordered:
        phash = row.get("phash")
        if not isinstance(phash, str) or _PHASH_PATTERN.fullmatch(phash) is None:
            raise ValueError(f"invalid joined 16-digit pHash for {row.get('id')}")
        hashes.append(int(phash, 16))
    return {
        (ordered[left]["id"], ordered[right]["id"])
        for left in range(len(ordered))
        for right in range(left + 1, len(ordered))
        if (hashes[left] ^ hashes[right]).bit_count() <= 6
    }


def _train_mae(
    stage,
    encoder,
    images,
    labels,
    fit_ids,
    output,
    recipe_sha,
    parent_hash,
    *,
    model_api=model,
    objective_api=objectives,
    runner_api=runner,
    resume_prefix=None,
    stop_after=None,
):
    encoder_trainable, encoder_static = train_steps.split_vit(encoder)
    config = encoder["config"]
    patch_values = int(config["patch_size"]) ** 2 * int(config["channels"])
    patch_count = (int(config["image_size"]) // int(config["patch_size"])) ** 2
    decoder_width = int(config["width"])
    decoder = objective_api.init_mae_decoder(
        runner_api.stage_key(141421, stage["id"]),
        width=decoder_width,
        heads=int(config["heads"]),
        mlp_dim=max(2, decoder_width * 2),
        layers=1,
        patches=patch_count,
        patch_values=patch_values,
    )
    decoder_trainable, decoder_static = train_steps.split_decoder(decoder)

    def loss(params, batch, _labels, dropout_key, mask_key, _positive):
        return train_steps.mae_pretraining_loss(
            params["encoder"],
            params["decoder"],
            encoder_static,
            decoder_static,
            batch,
            mask_key,
            dropout_key,
            model_api=model_api,
            objective_api=objective_api,
        )

    result = runner_api.train_updates(
        stage_id=stage["id"],
        params={"encoder": encoder_trainable, "decoder": decoder_trainable},
        loss_fn=loss,
        images_uint8=images,
        labels=labels,
        fit_ids=fit_ids,
        phash_by_id={},
        mode="flip",
        batch_size=int(stage["batch_size"]),
        updates=int(stage["updates"]),
        learning_rate=float(stage["learning_rate"]),
        checkpoint_directory=output / "checkpoints",
        identity={
            "recipe_sha256": recipe_sha,
            "parent_parameters_sha256": parent_hash,
            "objective": stage["objective"],
            "encoder_static": encoder_static,
            "decoder_static": decoder_static,
        },
        resume_prefix=resume_prefix,
        stop_after=stop_after,
    )
    return result | {
        "encoder_static": encoder_static,
        "decoder_static": decoder_static,
        "encoder": train_steps.assemble_vit(
            result["params"]["encoder"], encoder_static
        ),
    }


def _train_simclr(
    stage,
    encoder,
    images,
    labels,
    fit_ids,
    rows,
    output,
    recipe_sha,
    parent_hash,
    *,
    model_api=model,
    objective_api=objectives,
    runner_api=runner,
    resume_prefix=None,
    stop_after=None,
):
    encoder_trainable, encoder_static = train_steps.split_vit(encoder)
    encoder_width = int(encoder["config"]["width"])
    projector = objective_api.init_projector(
        runner_api.stage_key(173205, stage["id"]),
        encoder_width,
        max(2, encoder_width * 2),
        encoder_width,
    )
    row_by_id = {row["id"]: row for row in rows}
    phash_by_id = {image_id: row_by_id[image_id]["phash"] for image_id in fit_ids}
    known_positive_pairs = _positive_pairs(rows)

    def loss(params, views, _labels, dropout_key, _mask_key, positive):
        return train_steps.simclr_pretraining_loss(
            params["encoder"],
            params["projector"],
            encoder_static,
            views,
            positive,
            dropout_key,
            model_api=model_api,
            objective_api=objective_api,
        )

    result = runner_api.train_updates(
        stage_id=stage["id"],
        params={"encoder": encoder_trainable, "projector": projector},
        loss_fn=loss,
        images_uint8=images,
        labels=labels,
        fit_ids=fit_ids,
        phash_by_id=phash_by_id,
        known_positive_pairs=known_positive_pairs,
        mode="contrastive",
        batch_size=int(stage["batch_size"]),
        updates=int(stage["updates"]),
        learning_rate=float(stage["learning_rate"]),
        checkpoint_directory=output / "checkpoints",
        identity={
            "recipe_sha256": recipe_sha,
            "parent_parameters_sha256": parent_hash,
            "objective": stage["objective"],
            "encoder_static": encoder_static,
            "pHash_positive_threshold": 6,
        },
        resume_prefix=resume_prefix,
        stop_after=stop_after,
    )
    return result | {
        "encoder_static": encoder_static,
        "encoder": train_steps.assemble_vit(
            result["params"]["encoder"], encoder_static
        ),
        "projector": result["params"]["projector"],
    }


def _probe(
    stage,
    encoder,
    images,
    labels,
    fit_ids,
    output,
    recipe_sha,
    parent_hash,
    *,
    model_api=model,
    runner_api=runner,
    resume_prefix=None,
    stop_after=None,
):
    encoder_trainable, static = train_steps.split_vit(encoder)
    head = _head(
        stage["id"],
        int(encoder["config"]["width"]),
        int(encoder["config"]["classes"]),
        model_api=model_api,
        runner_api=runner_api,
    )

    def loss(current_head, batch, targets, _dropout, _mask, _positive):
        value = train_steps.frozen_probe_loss(
            current_head, encoder_trainable, static, batch, targets, model_api=model_api
        )
        return value, {"selected_mae_target_values": jnp.asarray(0, dtype=jnp.int32)}

    result = runner_api.train_updates(
        stage_id=stage["id"],
        params=head,
        loss_fn=loss,
        images_uint8=images,
        labels=labels,
        fit_ids=fit_ids,
        phash_by_id={},
        mode="flip",
        batch_size=int(stage["batch_size"]),
        updates=int(stage["updates"]),
        learning_rate=float(stage["learning_rate"]),
        checkpoint_directory=output / "checkpoints",
        identity={
            "recipe_sha256": recipe_sha,
            "parent_parameters_sha256": parent_hash,
            "objective": stage["objective"],
            "frozen_encoder_sha256": _tree_digest(encoder_trainable),
        },
        resume_prefix=resume_prefix,
        stop_after=stop_after,
    )
    return result | {"encoder": encoder, "head": result["params"]}


def _finetune(
    stage,
    encoder,
    images,
    labels,
    fit_ids,
    output,
    recipe_sha,
    parent_hash,
    *,
    model_api=model,
    runner_api=runner,
    resume_prefix=None,
    stop_after=None,
):
    head = _head(
        stage["id"],
        int(encoder["config"]["width"]),
        int(encoder["config"]["classes"]),
        model_api=model_api,
        runner_api=runner_api,
    )
    result = _train_vit(
        stage,
        encoder,
        head,
        images=images,
        labels=labels,
        fit_ids=fit_ids,
        phash_by_id={},
        output=output,
        recipe_sha=recipe_sha,
        parent_hash=parent_hash,
        mode="flip",
        model_api=model_api,
        runner_api=runner_api,
        resume_prefix=resume_prefix,
        stop_after=stop_after,
    )
    full_encoder = train_steps.assemble_vit(
        result["params"]["encoder"], result["static"]
    )
    return result | {"encoder": full_encoder, "head": result["params"]["head"]}


def _partial(
    stage,
    encoder,
    images,
    labels,
    fit_ids,
    output,
    recipe_sha,
    parent_hash,
    *,
    model_api=model,
    runner_api=runner,
    resume_prefix=None,
    stop_after=None,
):
    full_trainable, static = train_steps.split_vit(encoder)
    partial = train_steps.partial_trainable_vit(encoder)
    head = _head(
        stage["id"],
        int(encoder["config"]["width"]),
        int(encoder["config"]["classes"]),
        model_api=model_api,
        runner_api=runner_api,
    )

    def loss(params, batch, targets, dropout_key, _mask, _positive):
        updated = train_steps.merge_partial_vit(full_trainable, params["partial"])
        value = train_steps.vit_supervised_loss(
            updated,
            params["head"],
            static,
            batch,
            targets,
            dropout_key,
            model_api=model_api,
        )
        return value, {"selected_mae_target_values": jnp.asarray(0, dtype=jnp.int32)}

    result = runner_api.train_updates(
        stage_id=stage["id"],
        params={"partial": partial, "head": head},
        loss_fn=loss,
        images_uint8=images,
        labels=labels,
        fit_ids=fit_ids,
        phash_by_id={},
        mode="flip",
        batch_size=int(stage["batch_size"]),
        updates=int(stage["updates"]),
        learning_rate=float(stage["learning_rate"]),
        checkpoint_directory=output / "checkpoints",
        identity={
            "recipe_sha256": recipe_sha,
            "parent_parameters_sha256": parent_hash,
            "objective": stage["objective"],
            "trainable_leaf_inventory": train_steps.trainable_leaf_inventory(
                result_partial := {"partial": partial, "head": head}
            ),
        },
        resume_prefix=resume_prefix,
        stop_after=stop_after,
    )
    full_encoder_trainable = train_steps.merge_partial_vit(
        full_trainable, result["params"]["partial"]
    )
    return result | {
        "encoder": train_steps.assemble_vit(full_encoder_trainable, static),
        "head": result["params"]["head"],
        "static": static,
    }


def _logits(encoder, head, images, *, model_api=model):
    return np.asarray(
        model_api.classification_logits(encoder, head, jnp.asarray(images))
    )


def _features(encoder, images, *, model_api=model):
    return np.asarray(model_api.encode_images(encoder, jnp.asarray(images)))


def _component_by_id(manifest: dict) -> dict[str, str]:
    return {
        image_id: component_id
        for component_id, details in manifest["grouping"]["component_members"].items()
        for image_id in details["eligible_ids"]
    }


def _read_no_letterbox(archive_path: Path, image_ids: list[str]) -> np.ndarray:
    selected = set(image_ids)
    pixels = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if len(path.parts) != 3 or path.parts[0] != "flower_photos":
                continue
            image_id = PurePosixPath(*path.parts[1:]).as_posix()
            if image_id not in selected:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"missing held-out image bytes: {image_id}")
            with Image.open(stream) as source:
                display = ImageOps.exif_transpose(source).convert("RGB")
                pixels[image_id] = np.asarray(
                    display.resize((64, 64), resample=Image.Resampling.LANCZOS),
                    dtype=np.uint8,
                )
    if pixels.keys() != selected:
        raise ValueError(
            "archive did not provide the complete final inventory for resize control"
        )
    return np.stack([pixels[image_id] for image_id in image_ids])


def _read_archive_bytes(archive_path: Path, image_ids: list[str]) -> dict[str, bytes]:
    selected = set(image_ids)
    found = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if len(path.parts) != 3 or path.parts[0] != "flower_photos":
                continue
            image_id = PurePosixPath(*path.parts[1:]).as_posix()
            if image_id in selected:
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"missing held-out image bytes: {image_id}")
                found[image_id] = stream.read()
    if found.keys() != selected:
        raise ValueError(
            "archive did not provide the complete raw-image consumer canary inventory"
        )
    return found


def _write_stage_receipts(
    records: dict[str, Any], output_directory: Path
) -> dict[str, Any]:
    receipts = {}
    for name, record in records.items():
        prefix = Path(record["checkpoint_prefix"])
        npz_path, metadata_path = prefix.with_suffix(".npz"), prefix.with_suffix(
            ".json"
        )
        if not npz_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"completed stage {name} has no final checkpoint pair")
        trace_path = output_directory / "trace" / f"{name}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(record["trace"], indent=2, sort_keys=True) + "\n"
        )
        receipts[name] = {
            "checkpoint_npz": str(npz_path.relative_to(output_directory)),
            "checkpoint_npz_sha256": _sha(npz_path),
            "checkpoint_metadata": str(metadata_path.relative_to(output_directory)),
            "checkpoint_metadata_sha256": _sha(metadata_path),
            "trace": str(trace_path.relative_to(output_directory)),
            "trace_sha256": _sha(trace_path),
            "consumed_id_count": len(record["consumed_ids"]),
            "unique_consumed_ids": len(set(record["consumed_ids"])),
            "consumed_id_history_sha256": hashlib.sha256(
                "\n".join(record["consumed_ids"]).encode()
            ).hexdigest(),
            "completed_updates": int(record["optimizer"].step),
        }
    return receipts


def execute(
    *,
    recipe_path: Path,
    data_directory: Path,
    archive_path: Path,
    output_directory: Path,
    implementation_source: Path,
    api_modules: dict[str, Any] | None = None,
    candidate_source_paths: dict[str, Path] | None = None,
    candidate_source_sha256: dict[str, str] | None = None,
    resume: bool = False,
    interrupt_after_stage: str | None = None,
) -> dict:
    """Run the complete predeclared stage graph after caller verifies approval."""
    recipe_bytes = recipe_path.read_bytes()
    recipe_sha = hashlib.sha256(recipe_bytes).hexdigest()
    recipe = json.loads(recipe_bytes)
    apis = {
        "model": model,
        "objectives": objectives,
        "augment": augment,
        "evaluate": evaluate,
        "runner": runner,
    }
    if api_modules:
        apis.update(api_modules)
    model_api = apis["model"]
    objective_api = apis["objectives"]
    eval_api = apis["evaluate"]
    runner_api = apis["runner"]
    declared_checker_config = recipe.get("checker_fixture")
    if declared_checker_config is not None:
        if (
            declared_checker_config.get("synthetic_only") is not True
            or recipe.get("data", {}).get("upstream", {}).get("name")
            != "checker-owned synthetic RGB graph fixture"
        ):
            raise ValueError(
                "model config override is restricted to checker synthetic data"
            )
        model_api.PILOT_CONFIG = dict(declared_checker_config["model_config"])
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=resume)
    manifest = json.loads((data_directory / "source-manifest.json").read_text())
    source_metadata_by_id = _source_metadata_index(manifest)
    api_source_paths = {
        name: Path(module.__file__).resolve()
        for name, module in apis.items()
        if hasattr(module, "__file__")
    }
    owned_source_paths = {
        "pipeline": Path(__file__).resolve(),
        "train_steps": Path(train_steps.__file__).resolve(),
        "consumer": Path(consumer.__file__).resolve(),
        "consumer_runtime": Path(consumer_runtime.__file__).resolve(),
        "processor": (Path(__file__).parents[1] / "data" / "processor.py").resolve(),
    }
    if candidate_source_paths is None or candidate_source_sha256 is None:
        raise ValueError("approved candidate source inventory is required")
    candidate_source_paths = {
        str(name): Path(path).expanduser().resolve()
        for name, path in candidate_source_paths.items()
    }
    if set(candidate_source_paths) != set(candidate_source_sha256):
        raise ValueError("candidate source paths and approved hashes differ")
    current_candidate_hashes = {
        name: _sha(path) for name, path in sorted(candidate_source_paths.items())
    }
    if current_candidate_hashes != candidate_source_sha256:
        raise ValueError("candidate API source inventory changed before initialization")
    for module_name, module_path in api_source_paths.items():
        resolved_module = module_path.resolve()
        relative_matches = [
            relative
            for relative, candidate_file in candidate_source_paths.items()
            if candidate_file == resolved_module
        ]
        if len(relative_matches) != 1:
            raise ValueError(
                f"candidate API module {module_name} is outside its approved source inventory"
            )
        if current_candidate_hashes[relative_matches[0]] != _sha(resolved_module):
            raise ValueError(
                f"candidate API module {module_name} differs from its approved source hash"
            )
    required_candidate_files = set(recipe.get("candidate_api_required_files", []))
    if not required_candidate_files <= set(current_candidate_hashes):
        raise ValueError("recipe candidate API source inventory is incomplete")
    run_identity = {
        "recipe_sha256": recipe_sha,
        "source_manifest_sha256": _sha(data_directory / "source-manifest.json"),
        "api_source_sha256": {
            name: _sha(path) for name, path in sorted(api_source_paths.items())
        },
        "owned_source_sha256": {
            name: _sha(path) for name, path in sorted(owned_source_paths.items())
        },
        "implementation_source_sha256": _sha(implementation_source),
        "candidate_api_source_sha256": current_candidate_hashes,
        "effective_model_config": dict(model_api.PILOT_CONFIG),
    }
    journal_path = output_directory / "pipeline-journal.json"
    if resume:
        if not journal_path.is_file():
            raise ValueError("--resume requires the original pipeline journal")
        journal = json.loads(journal_path.read_text())
        if journal.get("run_identity") != run_identity:
            raise ValueError(
                "pipeline resume identity differs from the frozen source/data/recipe"
            )
    else:
        if any(output_directory.iterdir()):
            raise FileExistsError("new pipeline output directory must be empty")
        journal = {
            "schema": "jaxpathways.image-pipeline-journal.v1",
            "run_identity": run_identity,
            "stages": {},
        }
        _write_json_atomic(journal_path, journal)

    def run_stage(stage_id, function):
        entry = journal["stages"].get(stage_id, {})
        checkpoint_root = output_directory / "checkpoints" / stage_id
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        resume_prefix = None
        if entry.get("status") == "complete":
            recorded_prefix = Path(entry["checkpoint_prefix"])
            resume_prefix = (
                recorded_prefix
                if recorded_prefix.is_absolute()
                else output_directory / recorded_prefix
            )
            if (
                not resume_prefix.with_suffix(".npz").is_file()
                or not resume_prefix.with_suffix(".json").is_file()
            ):
                raise ValueError(f"completed stage {stage_id} lost its checkpoint pair")
            if _sha(resume_prefix.with_suffix(".npz")) != entry.get(
                "checkpoint_npz_sha256"
            ):
                raise ValueError(
                    f"completed stage {stage_id} checkpoint bytes changed after journal freeze"
                )
            if _sha(resume_prefix.with_suffix(".json")) != entry.get(
                "checkpoint_metadata_sha256"
            ):
                raise ValueError(
                    f"completed stage {stage_id} checkpoint metadata changed after journal freeze"
                )
            frozen_trace = output_directory / "trace" / f"{stage_id}.json"
            if not frozen_trace.is_file() or _sha(frozen_trace) != entry.get(
                "trace_sha256"
            ):
                raise ValueError(
                    f"completed stage {stage_id} trace changed after journal freeze"
                )
        elif resume:
            previous = entry.get("latest_checkpoint_prefix")
            if previous:
                recorded_prefix = Path(previous)
                resume_prefix = (
                    recorded_prefix
                    if recorded_prefix.is_absolute()
                    else output_directory / recorded_prefix
                )
            else:
                candidates = sorted(checkpoint_root.glob("step-*.json"))
                if candidates:
                    resume_prefix = candidates[-1].with_suffix("")
            if resume_prefix is not None and entry.get("latest_checkpoint_prefix") in {
                str(resume_prefix),
                str(resume_prefix.relative_to(output_directory)),
            }:
                npz_file, json_file = resume_prefix.with_suffix(
                    ".npz"
                ), resume_prefix.with_suffix(".json")
                if (
                    entry.get("checkpoint_npz_sha256")
                    and _sha(npz_file) != entry["checkpoint_npz_sha256"]
                ):
                    raise ValueError(
                        f"interrupted stage {stage_id} checkpoint bytes changed after journal freeze"
                    )
                if (
                    entry.get("checkpoint_metadata_sha256")
                    and _sha(json_file) != entry["checkpoint_metadata_sha256"]
                ):
                    raise ValueError(
                        f"interrupted stage {stage_id} metadata changed after journal freeze"
                    )
        journal["stages"][stage_id] = {
            **entry,
            "status": "running",
            "latest_checkpoint_prefix": (
                str(resume_prefix.relative_to(output_directory))
                if resume_prefix
                else entry.get("latest_checkpoint_prefix")
            ),
        }
        _write_json_atomic(journal_path, journal)
        stop_after = (
            1
            if interrupt_after_stage == stage_id and entry.get("status") != "complete"
            else None
        )
        record = function(resume_prefix=resume_prefix, stop_after=stop_after)
        prefix = Path(record["checkpoint_prefix"])
        if not record["complete"]:
            if (
                not prefix.with_suffix(".npz").is_file()
                or not prefix.with_suffix(".json").is_file()
            ):
                raise ValueError(
                    f"interrupted stage {stage_id} has no durable checkpoint pair"
                )
            journal["stages"][stage_id] = {
                **journal["stages"][stage_id],
                "status": "running",
                "latest_checkpoint_prefix": str(prefix.relative_to(output_directory)),
                "checkpoint_npz_sha256": _sha(prefix.with_suffix(".npz")),
                "checkpoint_metadata_sha256": _sha(prefix.with_suffix(".json")),
            }
            _write_json_atomic(journal_path, journal)
            raise InterruptedStage(
                f"intentional checker interruption after {stage_id} update"
            )
        trace_path = output_directory / "trace" / f"{stage_id}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(trace_path, record["trace"])
        journal["stages"][stage_id] = {
            "status": "complete",
            "checkpoint_prefix": str(prefix.relative_to(output_directory)),
            "checkpoint_npz_sha256": _sha(prefix.with_suffix(".npz")),
            "checkpoint_metadata_sha256": _sha(prefix.with_suffix(".json")),
            "trace_sha256": _sha(trace_path),
            "completed_updates": int(record["optimizer"].step),
            "consumed_ids_sha256": hashlib.sha256(
                "\n".join(record["consumed_ids"]).encode()
            ).hexdigest(),
        }
        _write_json_atomic(journal_path, journal)
        return record

    def invoke(stage_id, function, *args, **kwargs):
        return run_stage(
            stage_id,
            lambda resume_prefix, stop_after: function(
                *args, **kwargs, resume_prefix=resume_prefix, stop_after=stop_after
            ),
        )

    def freeze_decision(name, decision):
        decisions = journal.setdefault("decisions", {})
        if name in decisions and decisions[name] != decision:
            raise ValueError(f"frozen {name} decision changed on pipeline resume")
        decisions[name] = decision
        _write_json_atomic(output_directory / f"{name}-selection.json", decision)
        event_name = f"{name}_selection_frozen"
        events = journal.setdefault("events", [])
        if event_name not in [item.get("event") for item in events]:
            events.append(
                {
                    "event": event_name,
                    "decision_sha256": _sha(
                        output_directory / f"{name}-selection.json"
                    ),
                }
            )
        _write_json_atomic(journal_path, journal)

    arrays = {}

    def load_split(split):
        return load_split_inputs(
            data_directory,
            split,
            source_metadata_by_id=source_metadata_by_id,
            class_order=tuple(recipe["data"]["class_order"]),
        )

    for split in ("fit", "recipe_dev"):
        arrays[split] = load_split(split)
    fit = arrays["fit"]
    fit_ids = [row["id"] for row in fit["rows"]]
    phash_by_id = {row["id"]: row["phash"] for row in fit["rows"]}
    stages = {stage["id"]: stage for stage in recipe["pilot"]["stages"]}
    rng_seeds = recipe["pilot"]["seeds"]
    initial_vit = model_api.init_vit(jax.random.PRNGKey(int(rng_seeds["initial_vit"])))
    initial_cnn = model_api.init_cnn(jax.random.PRNGKey(int(rng_seeds["initial_cnn"])))
    source_paths = [
        implementation_source,
        Path(__file__),
        Path(runner_api.__file__),
        Path(runner_api.training.__file__),
        Path(train_steps.__file__),
        Path(objective_api.__file__),
        Path(apis["augment"].__file__),
        Path(eval_api.__file__),
        Path(consumer.__file__),
        Path(consumer_runtime.__file__),
        Path(__file__).parents[1] / "data" / "processor.py",
    ]
    records: dict[str, Any] = {}
    baseline_cnn = invoke(
        "baseline-cnn",
        _train_cnn,
        stages["baseline-cnn"],
        initial_cnn,
        images=fit["images"],
        labels=fit["labels"],
        fit_ids=fit_ids,
        output=output_directory,
        recipe_sha=recipe_sha,
        model_api=model_api,
        runner_api=runner_api,
    )
    records["baseline-cnn"] = baseline_cnn
    initial_trainable, initial_static = train_steps.split_vit(initial_vit)
    baseline_head = _head(
        "baseline-vit",
        int(initial_vit["config"]["width"]),
        int(initial_vit["config"]["classes"]),
    )
    baseline_parent = _tree_digest(initial_trainable)
    baseline = invoke(
        "baseline-vit",
        _train_vit,
        stages["baseline-vit"],
        initial_vit,
        baseline_head,
        images=fit["images"],
        labels=fit["labels"],
        fit_ids=fit_ids,
        phash_by_id=phash_by_id,
        output=output_directory,
        recipe_sha=recipe_sha,
        parent_hash=baseline_parent,
        model_api=model_api,
        runner_api=runner_api,
    )
    records["baseline-vit"] = baseline | {
        "encoder": train_steps.assemble_vit(
            baseline["params"]["encoder"], baseline["static"]
        ),
        "head": baseline["params"]["head"],
    }

    mae_initial = model_api.init_vit(jax.random.PRNGKey(int(rng_seeds["initial_vit"])))
    mae = invoke(
        "mae-pretrain",
        _train_mae,
        stages["mae-pretrain"],
        mae_initial,
        fit["images"],
        fit["labels"],
        fit_ids,
        output_directory,
        recipe_sha,
        _tree_digest(train_steps.split_vit(mae_initial)[0]),
        model_api=model_api,
        objective_api=objective_api,
        runner_api=runner_api,
    )
    records["mae-pretrain"] = mae
    simclr_initial = model_api.init_vit(
        jax.random.PRNGKey(int(rng_seeds["initial_vit"]))
    )
    simclr = invoke(
        "simclr-pretrain",
        _train_simclr,
        stages["simclr-pretrain"],
        simclr_initial,
        fit["images"],
        fit["labels"],
        fit_ids,
        fit["rows"],
        output_directory,
        recipe_sha,
        _tree_digest(train_steps.split_vit(simclr_initial)[0]),
        model_api=model_api,
        objective_api=objective_api,
        runner_api=runner_api,
    )
    records["simclr-pretrain"] = simclr

    probe_specs = {
        "probe-random": initial_vit,
        "probe-mae": mae["encoder"],
        "probe-simclr": simclr["encoder"],
        "probe-supervised": records["baseline-vit"]["encoder"],
    }
    for name, parent in probe_specs.items():
        parent_digest = _tree_digest(train_steps.split_vit(parent)[0])
        records[name] = invoke(
            name,
            _probe,
            stages[name],
            parent,
            fit["images"],
            fit["labels"],
            fit_ids,
            output_directory,
            recipe_sha,
            parent_digest,
            model_api=model_api,
            runner_api=runner_api,
        )

    dev = arrays["recipe_dev"]
    probe_scores = {}
    for probe_name in ("probe-mae", "probe-simclr"):
        state = records[probe_name]
        probe_scores[probe_name] = eval_api.classification_metrics(
            _logits(
                state["encoder"],
                state["head"],
                dev["images"].astype(np.float32) / 127.5 - 1,
                model_api=model_api,
            ),
            dev["labels"],
            int(model_api.PILOT_CONFIG["classes"]),
        )["macro_balanced_accuracy"]
    selected_pretrain = (
        "mae-pretrain"
        if probe_scores["probe-mae"] >= probe_scores["probe-simclr"]
        else "simclr-pretrain"
    )
    freeze_decision(
        "pretraining",
        {
            "schema": "jaxpathways.image-pretraining-selection.v1",
            "probe_scores": probe_scores,
            "selected": selected_pretrain,
            "tie_rule": "MAE wins an exact development balanced-accuracy tie",
            "basis": "recipe-development probes only",
        },
    )
    partial_stage = dict(stages["partial-selected"])
    partial_stage["parent"] = selected_pretrain
    selected_encoder = records[selected_pretrain]["encoder"]

    records["finetune-mae"] = invoke(
        "finetune-mae",
        _finetune,
        stages["finetune-mae"],
        mae["encoder"],
        fit["images"],
        fit["labels"],
        fit_ids,
        output_directory,
        recipe_sha,
        _tree_digest(train_steps.split_vit(mae["encoder"])[0]),
        model_api=model_api,
        runner_api=runner_api,
    )
    records["finetune-simclr"] = invoke(
        "finetune-simclr",
        _finetune,
        stages["finetune-simclr"],
        simclr["encoder"],
        fit["images"],
        fit["labels"],
        fit_ids,
        output_directory,
        recipe_sha,
        _tree_digest(train_steps.split_vit(simclr["encoder"])[0]),
        model_api=model_api,
        runner_api=runner_api,
    )
    records["partial-selected"] = invoke(
        "partial-selected",
        _partial,
        partial_stage,
        selected_encoder,
        fit["images"],
        fit["labels"],
        fit_ids,
        output_directory,
        recipe_sha,
        _tree_digest(train_steps.split_vit(selected_encoder)[0]),
        model_api=model_api,
        runner_api=runner_api,
    )

    class_count = len(recipe["data"]["class_order"])
    dev_images = dev["images"].astype(np.float32) / 127.5 - 1
    dev_reports, temperatures = {}, {}
    dev_logits_by_name = {}
    for name, state in records.items():
        if "encoder" not in state or "head" not in state:
            continue
        dev_logits = _logits(
            state["encoder"], state["head"], dev_images, model_api=model_api
        )
        temperature = eval_api.fit_temperature(dev_logits, dev["labels"])
        dev_logits_by_name[name] = dev_logits
        dev_reports[name] = {
            "raw": eval_api.classification_metrics(
                dev_logits, dev["labels"], class_count
            ),
            "calibrated": eval_api.classification_metrics(
                dev_logits,
                dev["labels"],
                class_count,
                float(temperature["temperature"]),
            ),
        }
        temperatures[name] = temperature

    cnn_dev_logits = np.asarray(
        model_api.cnn_logits(records["baseline-cnn"]["params"], jnp.asarray(dev_images))
    )
    cnn_temperature = eval_api.fit_temperature(cnn_dev_logits, dev["labels"])
    dev_logits_by_name["baseline-cnn"] = cnn_dev_logits
    dev_reports["baseline-cnn"] = {
        "raw": eval_api.classification_metrics(
            cnn_dev_logits, dev["labels"], class_count
        ),
        "calibrated": eval_api.classification_metrics(
            cnn_dev_logits, dev["labels"], class_count, cnn_temperature["temperature"]
        ),
    }
    temperatures["baseline-cnn"] = cnn_temperature

    fit_normalized = fit["images"].astype(np.float32) / 127.5 - 1
    color = eval_api.fit_color_centroids(fit_normalized, fit["labels"], class_count)
    color_dev_logits = eval_api.color_logits(color, dev_images)
    majority_class = int(np.argmax(np.bincount(fit["labels"], minlength=class_count)))
    uniform_dev_logits = np.zeros((len(dev["labels"]), class_count), dtype=np.float32)
    majority_dev_logits = np.full_like(uniform_dev_logits, -1000.0)
    majority_dev_logits[:, majority_class] = 0.0
    controls = {
        "uniform": uniform_dev_logits,
        "fit-majority": majority_dev_logits,
        "fit-color-centroid": color_dev_logits,
    }
    for name, dev_logits in controls.items():
        dev_logits_by_name[name] = dev_logits
        dev_reports[name] = {
            "raw": eval_api.classification_metrics(
                dev_logits, dev["labels"], class_count
            )
        }

    deploy_order = recipe["pilot"]["selection"]["deploy_choice"]["tie_order"]
    selected_deploy = max(
        deploy_order,
        key=lambda name: (
            dev_reports[name]["raw"]["macro_balanced_accuracy"],
            -deploy_order.index(name),
        ),
    )
    selection_record = {
        "schema": "jaxpathways.image-pilot-dev-selection.v1",
        "candidate_order": list(deploy_order),
        "recipe_dev_ids": [row["id"] for row in dev["rows"]],
        "dev_balanced_accuracy": {
            name: float(dev_reports[name]["raw"]["macro_balanced_accuracy"])
            for name in deploy_order
        },
        "temperatures_selected_on_recipe_dev": {
            name: float(temperatures[name]["temperature"]) for name in deploy_order
        },
        "selected": selected_deploy,
        "selection_frozen_before_final_split_load": True,
        "tie_rule": "recipe order resolves exact development-score ties",
    }
    selection_path = output_directory / "selection-lock.json"
    freeze_decision("deploy", selection_record)
    _write_json_atomic(selection_path, selection_record)
    selection_lock_sha256 = _sha(selection_path)

    # The final split is not even loaded until all model choices and calibration
    # temperatures are durably frozen from recipe-development outputs.
    arrays["final"] = load_split("final")
    events = journal.setdefault("events", [])
    if not any(item.get("event") == "final_split_loaded" for item in events):
        events.append(
            {
                "event": "final_split_loaded",
                "ids_sha256": hashlib.sha256(
                    "\n".join(row["id"] for row in arrays["final"]["rows"]).encode()
                ).hexdigest(),
            }
        )
    if [item["event"] for item in events].index("deploy_selection_frozen") > [
        item["event"] for item in events
    ].index("final_split_loaded"):
        raise ValueError("final split was loaded before the development selection lock")
    journal["final_split_loaded_after_deploy_selection"] = True
    _write_json_atomic(journal_path, journal)
    final = arrays["final"]
    final_images = final["images"].astype(np.float32) / 127.5 - 1
    final_reports, final_logits_by_name = {}, {}
    for name, state in records.items():
        if "encoder" not in state or "head" not in state:
            continue
        final_logits = _logits(
            state["encoder"], state["head"], final_images, model_api=model_api
        )
        final_logits_by_name[name] = final_logits
        final_reports[name] = {
            "raw": eval_api.classification_metrics(
                final_logits, final["labels"], class_count
            ),
            "calibrated": eval_api.classification_metrics(
                final_logits,
                final["labels"],
                class_count,
                float(temperatures[name]["temperature"]),
            ),
        }
    cnn_final_logits = np.asarray(
        model_api.cnn_logits(
            records["baseline-cnn"]["params"], jnp.asarray(final_images)
        )
    )
    final_logits_by_name["baseline-cnn"] = cnn_final_logits
    final_reports["baseline-cnn"] = {
        "raw": eval_api.classification_metrics(
            cnn_final_logits, final["labels"], class_count
        ),
        "calibrated": eval_api.classification_metrics(
            cnn_final_logits,
            final["labels"],
            class_count,
            temperatures["baseline-cnn"]["temperature"],
        ),
    }
    color_final_logits = eval_api.color_logits(color, final_images)
    uniform_final_logits = np.zeros(
        (len(final["labels"]), class_count), dtype=np.float32
    )
    majority_final_logits = np.full_like(uniform_final_logits, -1000.0)
    majority_final_logits[:, majority_class] = 0.0
    final_controls = {
        "uniform": uniform_final_logits,
        "fit-majority": majority_final_logits,
        "fit-color-centroid": color_final_logits,
    }
    for name, final_logits in final_controls.items():
        final_logits_by_name[name] = final_logits
        final_reports[name] = {
            "raw": eval_api.classification_metrics(
                final_logits, final["labels"], class_count
            )
        }

    final_ids = [row["id"] for row in final["rows"]]
    component_map = _component_by_id(manifest)
    final_components = [component_map[image_id] for image_id in final_ids]
    retrieval_reports = {}
    retrieval_feature_arrays = {
        "ids": np.asarray(final_ids),
        "labels": np.asarray(final["labels"], dtype=np.int32),
        "component_ids": np.asarray(final_components),
    }
    for name in deploy_order:
        state = records[name]
        features = _features(state["encoder"], final_images, model_api=model_api)
        retrieval_feature_arrays[f"features__{name}"] = np.asarray(
            features, dtype=np.float32
        )
        retrieval_reports[name] = eval_api.retrieval_metrics(
            features, final_ids, final["labels"], final_components
        )

    selected_temperature = float(temperatures[selected_deploy]["temperature"])
    selected_state = records[selected_deploy]
    corruptions = eval_api.corruptions(final_images)
    resize_control = (
        _read_no_letterbox(archive_path, final_ids).astype(np.float32) / 127.5 - 1
    )
    corruptions["resize_without_letterbox"] = resize_control
    original_balanced = final_reports[selected_deploy]["raw"]["macro_balanced_accuracy"]
    selected_conditions = {}
    condition_logits = {}
    for condition, images in corruptions.items():
        logits = (
            final_logits_by_name[selected_deploy]
            if condition == "original"
            else _logits(
                selected_state["encoder"],
                selected_state["head"],
                images,
                model_api=model_api,
            )
        )
        raw_metrics = eval_api.classification_metrics(
            logits, final["labels"], class_count
        )
        calibrated_metrics = eval_api.classification_metrics(
            logits, final["labels"], class_count, selected_temperature
        )
        selected_conditions[condition] = {
            "image_count": len(final_ids),
            "per_class_count": {
                class_name: int(np.sum(final["labels"] == index))
                for index, class_name in enumerate(recipe["data"]["class_order"])
            },
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
            "balanced_accuracy_delta_vs_original": raw_metrics[
                "macro_balanced_accuracy"
            ]
            - original_balanced,
        }
        condition_logits[f"logits__{condition}"] = np.asarray(logits, dtype=np.float32)

    floors = recipe["evaluation"]["precommitted_quality_floors"]
    selected_metrics = final_reports[selected_deploy]
    color_balanced = final_reports["fit-color-centroid"]["raw"][
        "macro_balanced_accuracy"
    ]
    selected_gate = {
        "final_macro_balanced_accuracy": selected_metrics["raw"][
            "macro_balanced_accuracy"
        ]
        >= float(floors["final_macro_balanced_accuracy"].removeprefix(">=")),
        "beats_fixed_color_centroid": selected_metrics["raw"]["macro_balanced_accuracy"]
        > color_balanced,
        "brightness_drop": selected_metrics["raw"]["macro_balanced_accuracy"]
        - selected_conditions["brightness_plus_0.05"]["raw"]["macro_balanced_accuracy"]
        <= float(
            floors["brightness_plus_0.05_absolute_balanced_accuracy_drop"].removeprefix(
                "<="
            )
        ),
        "calibration_nll_delta": selected_metrics["calibrated"]["nll"]
        - selected_metrics["raw"]["nll"]
        <= float(floors["calibrated_nll_minus_raw_nll"].removeprefix("<=")),
    }
    bootstrap_spec = recipe["evaluation"]["component_bootstrap"]
    selected_bootstrap = {
        "raw": eval_api.component_bootstrap(
            final_logits_by_name[selected_deploy],
            final["labels"],
            final_components,
            classes=class_count,
            replicates=int(bootstrap_spec["replicates"]),
            seed=int(bootstrap_spec["seed"]),
        ),
        "calibrated": eval_api.component_bootstrap(
            final_logits_by_name[selected_deploy],
            final["labels"],
            final_components,
            classes=class_count,
            temperature=selected_temperature,
            replicates=int(bootstrap_spec["replicates"]),
            seed=int(bootstrap_spec["seed"]),
        ),
    }
    stage_receipts = _write_stage_receipts(records, output_directory)

    processor_source = Path(__file__).parents[1] / "data" / "processor.py"
    package_path = output_directory / "local-consumer"
    package_report = consumer.export_local_bundle(
        package_path,
        model_module=model_api,
        model_source=implementation_source,
        processor_source=processor_source,
        encoder_params=selected_state["encoder"],
        head=selected_state["head"],
        provenance={
            "dataset": recipe["data"]["upstream"]["name"],
            "archive_sha256": recipe["data"]["upstream"]["sha256"],
            "license_notice": "CC BY 2.0 image-level credits are in the upstream attribution notice; no photo pixels are packaged.",
            "classes": recipe["data"]["class_order"],
            "selected_stage": selected_deploy,
            "recipe_sha256": recipe_sha,
            "weight_rights_statement": "Private correctness bundle only; no standalone weight redistribution license is asserted.",
        },
    )
    package_model = consumer_runtime.LocalImageClassifier.from_pretrained(
        package_path, local_files_only=True
    )
    canary_ids = final_ids[: min(3, len(final_ids))]
    canary_raw = _read_archive_bytes(archive_path, canary_ids)
    package_canaries = []
    for image_id in canary_ids:
        raw_pixels, processor_info = package_model.processor.preprocess_bytes(
            canary_raw[image_id], 64
        )
        normalized = package_model.processor.normalize_uint8(raw_pixels[None, ...])
        direct_logits = _logits(
            selected_state["encoder"],
            selected_state["head"],
            normalized,
            model_api=model_api,
        )
        packaged = package_model.predict_bytes(canary_raw[image_id])
        np.testing.assert_allclose(
            packaged["logits"], direct_logits[0], rtol=0.0, atol=0.0
        )
        package_canaries.append(
            {
                "id": image_id,
                "true_class": int(final["labels"][final_ids.index(image_id)]),
                "predicted_class": packaged["class_index"],
                "source_dimensions": [
                    processor_info["source_width"],
                    processor_info["source_height"],
                ],
                "logits": packaged["logits"],
            }
        )

    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    comparison_arrays = {
        "dev_ids": np.asarray([row["id"] for row in dev["rows"]]),
        "dev_labels": np.asarray(dev["labels"], dtype=np.int32),
        "final_ids": np.asarray(final_ids),
        "final_labels": np.asarray(final["labels"], dtype=np.int32),
    }
    comparison_keys = {}
    for name in sorted(set(dev_logits_by_name) | set(final_logits_by_name)):
        if name not in dev_logits_by_name or name not in final_logits_by_name:
            raise ValueError(
                f"candidate/control {name} lacks a development or final logit array"
            )
        dev_key, final_key = f"dev_logits__{name}", f"final_logits__{name}"
        comparison_arrays[dev_key] = np.asarray(
            dev_logits_by_name[name], dtype=np.float32
        )
        comparison_arrays[final_key] = np.asarray(
            final_logits_by_name[name], dtype=np.float32
        )
        comparison_keys[name] = {"development": dev_key, "final": final_key}
    np.savez_compressed(
        raw_directory / "all-classifier-comparisons.npz", **comparison_arrays
    )
    _write_json_atomic(
        raw_directory / "classifier-comparison-index.json",
        {
            "schema": "jaxpathways.image-classifier-comparison.v1",
            "arrays": comparison_keys,
        },
    )
    np.savez_compressed(
        raw_directory / "final-retrieval-features.npz", **retrieval_feature_arrays
    )
    _write_json_atomic(
        raw_directory / "final-retrieval-rankings.json", retrieval_reports
    )
    np.savez_compressed(
        raw_directory / "selected-final-condition-logits.npz", **condition_logits
    )
    probabilities = np.exp(
        final_logits_by_name[selected_deploy]
        - np.max(final_logits_by_name[selected_deploy], axis=1, keepdims=True)
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    np.savez_compressed(
        raw_directory / "selected-final-predictions.npz",
        ids=np.asarray(final_ids),
        labels=np.asarray(final["labels"], dtype=np.int32),
        logits=np.asarray(final_logits_by_name[selected_deploy], dtype=np.float32),
        probabilities=np.asarray(probabilities, dtype=np.float32),
    )
    raw_artifact_hashes = {
        str(path.relative_to(output_directory)): _sha(path)
        for path in sorted(raw_directory.rglob("*"))
        if path.is_file()
    }
    report = {
        "schema": "jaxpathways.image-pilot-run.v1",
        "recipe_sha256": recipe_sha,
        "pipeline_run_identity": run_identity,
        "candidate_model_source_sha256": run_identity["api_source_sha256"].get("model"),
        "source_manifest_sha256": _sha(data_directory / "source-manifest.json"),
        "dataset_archive_sha256": recipe["data"]["upstream"]["sha256"],
        "recipe_source_sha256": recipe["source_sha256"],
        "stage_order": [stage["id"] for stage in recipe["pilot"]["stages"]],
        "stage_traces": {name: value["trace"] for name, value in records.items()},
        "stage_checkpoints": stage_receipts,
        "stage_execution": {
            name: {
                "objective": stages[name]["objective"],
                "parent": stages[name]["parent"],
                "updates": int(stages[name]["updates"]),
                "batch_size": int(stages[name]["batch_size"]),
                "learning_rate": float(stages[name]["learning_rate"]),
                "measured_input_images": int(
                    sum(row["input_image_count"] for row in record["trace"])
                ),
                "measured_model_views": int(
                    sum(row["model_view_count"] for row in record["trace"])
                ),
                "measured_pixel_values": int(
                    sum(row["input_pixel_values"] for row in record["trace"])
                ),
                "measured_selected_mae_target_values": int(
                    sum(row["selected_mae_target_values"] for row in record["trace"])
                ),
                "actual_parent_parameters_sha256": record["identity"].get(
                    "parent_parameters_sha256"
                ),
                "objective_units_are_stage_specific": True,
            }
            for name, record in records.items()
        },
        "dev_metrics": dev_reports,
        "final_metrics": final_reports,
        "temperature_selection": temperatures,
        "probe_based_pretrain_selection": {
            "scores": probe_scores,
            "selected": selected_pretrain,
        },
        "deploy_selection": {
            "dev_only_selected": selected_deploy,
            "candidate_order": deploy_order,
            "final_results_cannot_reselect": True,
            "frozen_lock_sha256": selection_lock_sha256,
            "frozen_before_final_split_load": True,
        },
        "retrieval": retrieval_reports,
        "selected_final_component_bootstrap": selected_bootstrap,
        "selected_final_conditions": selected_conditions,
        "fixed_baselines": ["uniform", "fit-majority", "fit-color-centroid"],
        "final_class_counts": {
            class_name: int(np.sum(final["labels"] == index))
            for index, class_name in enumerate(recipe["data"]["class_order"])
        },
        "selected_prediction_arrays": "raw/selected-final-predictions.npz (final IDs, labels, logits and probabilities)",
        "all_comparison_arrays": {
            "classifiers": "raw/all-classifier-comparisons.npz",
            "classifier_index": "raw/classifier-comparison-index.json",
            "retrieval_features": "raw/final-retrieval-features.npz",
            "retrieval_rankings": "raw/final-retrieval-rankings.json",
            "selected_corruption_logits": "raw/selected-final-condition-logits.npz",
        },
        "raw_artifact_sha256": raw_artifact_hashes,
        "consumer_package": {
            "manifest_sha256": package_report["manifest_sha256"],
            "parameter_count": package_report["parameter_count"],
            "contains_source_images": package_report["contains_source_images"],
            "weight_rights_statement": package_report["weight_rights_statement"],
            "raw_image_canaries": package_canaries,
        },
        "quality_gate": {
            "selected_candidate_only": selected_gate,
            "all_selected_floors_pass": bool(all(selected_gate.values())),
            "precommitted_floors": floors,
            "classification_only_pilot": True,
            "human_acceptance": "pending",
            "target_300m_qualification": "pending",
            "public_weight_release": "pending rights review; no standalone weight license asserted",
        },
        "limitations": recipe["evaluation"]["limits"],
    }
    (output_directory / "run-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
