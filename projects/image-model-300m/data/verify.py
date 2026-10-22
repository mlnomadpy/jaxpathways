"""Verify the source-bound prepared flower data used by the pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, ImageOps


PROJECT = Path(__file__).resolve().parents[1]
CLASS_ORDER = ("daisy", "dandelion", "roses", "sunflowers", "tulips")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_sources(recipe: dict[str, Any]) -> None:
    declared = recipe.get("source_sha256")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("recipe has no source hash inventory")
    for relative, expected in declared.items():
        path = (PROJECT / relative).resolve()
        if not path.is_relative_to(PROJECT) or not path.is_file():
            raise ValueError(f"source hash path escapes project or is missing: {relative}")
        if sha256_file(path) != expected:
            raise ValueError(f"recipe-bound source changed: {relative}")


def _verify_archive(archive: Path, manifest: dict[str, Any], recipe: dict[str, Any]) -> None:
    source = recipe["data"]["upstream"]
    if archive.stat().st_size != source["bytes"] or sha256_file(archive) != source["sha256"]:
        raise ValueError("upstream archive bytes differ from the pinned source")
    expected_members = manifest.get("archive_members")
    if not isinstance(expected_members, list):
        raise ValueError("source manifest has no full archive-member inventory")
    observed = []
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "flower_photos":
                raise ValueError(f"unexpected archive member path: {member.name}")
            if member.isdir():
                observed.append({"path": member.name, "type": "directory"})
                continue
            if not member.isfile():
                raise ValueError(f"archive contains a nonregular member: {member.name}")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"archive member has no bytes: {member.name}")
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            observed.append({"path": member.name, "type": "file", "bytes": size, "sha256": digest.hexdigest()})
    if observed != expected_members:
        raise ValueError("archive member identities/bytes differ from the source manifest")


def _oracle_pixels(raw: bytes) -> np.ndarray:
    """Independent Pillow path matching the frozen 64-pixel processor contract."""
    import io

    with Image.open(io.BytesIO(raw)) as source:
        if source.format != "JPEG" or getattr(source, "n_frames", 1) != 1:
            raise ValueError("pinned archive image is not a single-frame JPEG")
        display = ImageOps.exif_transpose(source).convert("RGB")
        contained = ImageOps.contain(display, (64, 64), method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (64, 64), (127, 127, 127))
        canvas.paste(contained, ((64 - contained.width) // 2, (64 - contained.height) // 2))
        return np.asarray(canvas, dtype=np.uint8)


def _verify_prepared_pixels(archive: Path, data_path: Path) -> None:
    rows_by_id = {}
    arrays_by_split = {}
    for split in ("fit", "recipe_dev", "final"):
        rows = json.loads((data_path / "splits" / f"{split}.json").read_text())
        arrays_by_split[split] = np.load(
            data_path / "arrays" / f"{split}_images.npy", mmap_mode="r", allow_pickle=False
        )
        for index, row in enumerate(rows):
            if row["id"] in rows_by_id:
                raise ValueError("prepared image ID is duplicated across splits")
            rows_by_id[row["id"]] = (split, index)
    missing = set(rows_by_id)
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if len(path.parts) != 3 or path.parts[0] != "flower_photos":
                continue
            image_id = PurePosixPath(*path.parts[1:]).as_posix()
            if image_id not in rows_by_id:
                continue
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"selected archive image has no bytes: {image_id}")
            raw = stream.read()
            split, index = rows_by_id[image_id]
            expected = _oracle_pixels(raw)
            actual = np.asarray(arrays_by_split[split][index])
            if not np.array_equal(actual, expected):
                raise ValueError(f"prepared RGB pixels differ from independent archive decode: {image_id}")
            missing.remove(image_id)
    if missing:
        raise ValueError(f"selected IDs were absent from raw archive: {sorted(missing)[:5]}")


def verify_data(
    data_directory: str | Path,
    *,
    recipe_path: str | Path | None = None,
    archive_path: str | Path | None = None,
    verify_sources: bool = True,
) -> dict[str, Any]:
    data_path = Path(data_directory).expanduser().resolve()
    recipe_file = Path(recipe_path).expanduser().resolve() if recipe_path else PROJECT / "data" / "recipe-review.json"
    recipe = json.loads(recipe_file.read_text())
    if recipe.get("schema") != "jaxpathways.image-pilot-recipe-review.v1":
        raise ValueError("unsupported image pilot recipe schema")
    if verify_sources:
        _verify_sources(recipe)
    manifest_path = data_path / "source-manifest.json"
    if sha256_file(manifest_path) != recipe["data"]["prepared_data"]["source_manifest_sha256"]:
        raise ValueError("prepared source manifest differs from recipe")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source", {}).get("archive_sha256") != recipe["data"]["upstream"]["sha256"]:
        raise ValueError("prepared manifest claims a different upstream archive")
    membership_path = PROJECT / recipe["data"]["membership_artifact"]["path"]
    if sha256_file(membership_path) != recipe["data"]["membership_artifact"]["sha256"]:
        raise ValueError("frozen component assignment changed")
    prepared_sources = manifest.get("materialization_sources", {})
    for field, expected in (
        ("preparer_sha256", recipe["source_sha256"]["data/prepare.py"]),
        ("processor_sha256", recipe["source_sha256"]["data/processor.py"]),
        ("verifier_sha256", recipe["source_sha256"]["data/verify.py"]),
        ("membership_sha256", recipe["data"]["membership_artifact"]["sha256"]),
        ("requirements_sha256", recipe["source_sha256"]["requirements-image.txt"]),
        ("requirements_lock_sha256", recipe["source_sha256"]["requirements-image-lock.txt"]),
    ):
        if prepared_sources.get(field) != expected:
            raise ValueError(f"prepared manifest is not bound to recipe {field}")
    expected_rows = recipe["data"]["prepared_data"].get("split_rows")
    manifest_rows = manifest.get("prepared_split_rows")
    if not isinstance(expected_rows, dict) or set(expected_rows) != {
        f"splits/{name}.json" for name in ("fit", "recipe_dev", "final")
    }:
        raise ValueError("recipe has no complete source-bound split-row inventory")
    if manifest_rows != expected_rows:
        raise ValueError("prepared manifest split-row inventory differs from the trusted recipe")
    for relative, declared in expected_rows.items():
        path = data_path / relative
        if (
            not path.is_file()
            or path.stat().st_size != declared.get("bytes")
            or sha256_file(path) != declared.get("sha256")
        ):
            raise ValueError(f"split identity metadata differs from its source-bound bytes: {relative}")
    if archive_path is not None:
        pinned_archive = Path(archive_path).expanduser().resolve()
        _verify_archive(pinned_archive, manifest, recipe)
        _verify_prepared_pixels(pinned_archive, data_path)

    summary = manifest["grouping"]["split_summary"]
    role_by_component = {}
    member = manifest["grouping"]["component_members"]
    expected_file_splits = {}
    for role in ("fit", "recipe_dev", "final"):
        component_ids = manifest["grouping"]["split_components"][role]
        if len(component_ids) != summary[role]["component_count"]:
            raise ValueError(f"{role} component count differs from manifest summary")
        for component_id in component_ids:
            if component_id in role_by_component:
                raise ValueError("a source component occurs in multiple splits")
            role_by_component[component_id] = role
            details = member.get(component_id)
            if details is None:
                raise ValueError("split component has no membership details")
            for image_id in details["eligible_ids"]:
                if image_id in expected_file_splits:
                    raise ValueError("eligible image ID appears in multiple source components")
                expected_file_splits[image_id] = role

    counts = {}
    for split in ("fit", "recipe_dev", "final"):
        split_rows = json.loads((data_path / "splits" / f"{split}.json").read_text())
        ids = [row["id"] for row in split_rows]
        if len(ids) != len(set(ids)) or any(expected_file_splits.get(image_id) != split for image_id in ids):
            raise ValueError(f"{split} row IDs duplicate or violate fixed source-component membership")
        expected_count = summary[split]["image_count"]
        if len(ids) != expected_count:
            raise ValueError(f"{split} row count differs from frozen summary")
        labels = np.asarray([row["class_index"] for row in split_rows], dtype=np.int32)
        if np.any(labels < 0) or np.any(labels >= len(CLASS_ORDER)):
            raise ValueError(f"{split} class index is out of range")
        class_counts = {name: int(np.sum(labels == index)) for index, name in enumerate(CLASS_ORDER)}
        if class_counts != summary[split]["class_counts"]:
            raise ValueError(f"{split} per-class counts differ from frozen summary")
        counts[split] = {
            "images": len(ids),
            "components": len(manifest["grouping"]["split_components"][split]),
            "per_class": class_counts,
        }

        images = np.load(data_path / "arrays" / f"{split}_images.npy", mmap_mode="r", allow_pickle=False)
        stored_labels = np.load(data_path / "arrays" / f"{split}_labels.npy", mmap_mode="r", allow_pickle=False)
        if images.shape != (len(ids), 64, 64, 3) or images.dtype != np.uint8:
            raise ValueError(f"{split} image array has an invalid shape or dtype")
        if (
            stored_labels.shape != labels.shape
            or stored_labels.dtype != np.int32
            or not np.array_equal(stored_labels, labels)
        ):
            raise ValueError(f"{split} label array does not match the declared row order")
        for index, array_name in (("images", f"{split}_images.npy"), ("labels", f"{split}_labels.npy")):
            relative = f"arrays/{array_name}"
            expected_hash = recipe["data"]["prepared_data"]["arrays"][array_name]
            if (
                sha256_file(data_path / relative) != expected_hash
                or manifest["prepared_array_files"][relative]["sha256"] != expected_hash
            ):
                raise ValueError(f"{split} {index} array differs from its source-bound hash")

    if len(expected_file_splits) != recipe["data"]["counts"]["eligible_images"]:
        raise ValueError("component membership does not cover the eligible image inventory")
    return {
        "status": "pass",
        "archive_sha256": recipe["data"]["upstream"]["sha256"],
        "source_manifest_sha256": sha256_file(manifest_path),
        "membership_sha256": recipe["data"]["membership_artifact"]["sha256"],
        "counts": counts,
        "archive_reconstructed": archive_path is not None,
        "source_files_verified": len(recipe["source_sha256"]) if verify_sources else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--archive", type=Path, help="official source archive to independently rehash")
    parser.add_argument("--recipe", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(verify_data(args.data, recipe_path=args.recipe, archive_path=args.archive), indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
