"""Build a deterministic, source-auditable TensorFlow flower_photos pilot set.

The archive is parsed as a stream and images are never extracted to a caller-
controlled path. Generated pixel arrays and memberships must live outside this
project because the photographs are third-party licensed data.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

import numpy as np
import PIL
import scipy
from PIL import Image, ImageOps
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from processor import decode_display_rgb, letterbox_rgb


ARCHIVE_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
ARCHIVE_BYTES = 228_813_984
ARCHIVE_SHA256 = "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c"
LICENSE_SHA256 = "701e91b9e9f46d1f9623fc793a00fd444fe2821f66b01f3a0ff4a73d82f49507"
CLASSES = ("daisy", "dandelion", "roses", "sunflowers", "tulips")
FRACTIONS = {"fit": 0.70, "recipe_dev": 0.15, "final": 0.15}
SPLITS = tuple(FRACTIONS)
IMAGE_SIZE = 64
SEED_LABEL = "flower-creator-groups-v1"

# Exclusions are derived from the archive's own attribution ledger and exact
# file/pixel identity checks. The unmatched attribution files remain recorded.
CONFLICTING_LABEL_IDS = (
    "roses/15922772266_1167a06620.jpg",
    "tulips/15922772266_1167a06620.jpg",
)
REDUNDANT_SAME_CLASS_IDS = (
    "sunflowers/15066430311_fb57fa92b0_m.jpg",
    "sunflowers/15072973261_73e2912ef2_n.jpg",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source(path: Path) -> str:
    return sha256_file(path)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def attribution_rows(license_bytes: bytes) -> dict[str, dict[str, str]]:
    text = license_bytes.decode("utf-8")
    pattern = re.compile(r"(.+\.(?:jpg|jpeg|png)) CC-BY by (.*?) - (https?://\S+)$", re.I)
    result: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        match = pattern.fullmatch(line)
        if match:
            image_id, creator, source_url = match.groups()
            if image_id in result:
                raise ValueError(f"duplicate attribution entry for {image_id}")
            result[image_id] = {
                "creator": creator,
                "creator_id": " ".join(creator.casefold().split()),
                "source_url": source_url,
                "source_account_id": PurePosixPath(source_url.split("?", 1)[0]).parts[-2],
                "source_photo_id": PurePosixPath(source_url.split("?", 1)[0]).parts[-1],
            }
    if sha256_bytes(license_bytes) != LICENSE_SHA256:
        raise ValueError("archive LICENSE.txt does not match the reviewed license bytes")
    return result


def decode_record(image_id: str, raw: bytes, label: str, attribution: dict[str, str] | None) -> dict:
    raw_hash = sha256_bytes(raw)
    if attribution is None:
        return {
            "id": image_id,
            "label": label,
            "raw_sha256": raw_hash,
            "rights_status": "unresolved_attribution_excluded",
        }
    display, info = decode_display_rgb(raw)
    if info["source_format"] != "JPEG":
        raise ValueError(f"unexpected image format: {image_id}")
    width, height = display.size
    pixels = display.tobytes()
    pixel_hash = sha256_bytes(width.to_bytes(4, "big") + height.to_bytes(4, "big") + pixels)
    gray = ImageOps.fit(display.convert("L"), (32, 32), method=Image.Resampling.BILINEAR)
    grid = np.asarray(gray, dtype=np.float64)
    n = 32
    basis = np.cos(np.pi / n * (np.arange(n)[:, None] + 0.5) * np.arange(8)[None, :])
    dct_low = (basis.T @ grid @ basis)[:8, :8]
    median = np.median(dct_low.ravel()[1:])
    phash = sum(1 << i for i, value in enumerate(dct_low.ravel()) if value > median)
    return {
        "id": image_id,
        "label": label,
        "raw_sha256": raw_hash,
        "pixel_sha256": pixel_hash,
        "width": width,
        "height": height,
        "source_format": info["source_format"],
        "frame_count": 1,
        "exif_orientation": info["exif_orientation"],
        "phash": f"{phash:016x}",
        **attribution,
        "rights_status": "cc-by-2.0-attributed",
    }


def load_archive(archive: Path) -> tuple[bytes, list[dict], list[dict]]:
    if archive.stat().st_size != ARCHIVE_BYTES or sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError("archive size or SHA-256 does not match the frozen upstream checksum")
    records: list[dict] = []
    members: list[dict] = []
    image_member_count = 0
    license_bytes = b""
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "flower_photos":
                raise ValueError(f"unsafe or unexpected archive path: {member.name!r}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"non-regular archive member is not accepted: {member.name!r}")
            if member.isdir():
                members.append({"path": member.name, "type": "directory"})
                continue
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"archive member has no bytes: {member.name}")
            data = stream.read()
            members.append({"path": member.name, "type": "file", "bytes": len(data), "sha256": sha256_bytes(data)})
            relative = PurePosixPath(*path.parts[1:])
            if relative.as_posix() == "LICENSE.txt":
                license_bytes = data
                continue
            if len(relative.parts) != 2 or relative.parts[0] not in CLASSES:
                raise ValueError(f"unexpected file in archive: {member.name}")
            image_member_count += 1
            attribution = None
            # Lookup is deferred until the license is streamed and parsed.
            records.append({"id": relative.as_posix(), "label": relative.parts[0], "raw": data})
    if image_member_count != 3670 or not license_bytes:
        raise ValueError(f"unexpected archive inventory: {image_member_count} images")
    attrs = attribution_rows(license_bytes)
    for record in records:
        record.update(decode_record(record["id"], record.pop("raw"), record["label"], attrs.get(record["id"])))
    image_ids = {record["id"] for record in records}
    if set(attrs) - image_ids:
        raise ValueError("license lists files absent from the pinned image archive")
    if len(attrs) != 3661 or image_ids - set(attrs) != {
        "dandelion/12094442595_297494dba4_m.jpg",
        "dandelion/14283011_3e7452c5b2_n.jpg",
        "dandelion/4721773235_429acdf496_n.jpg",
        "roses/2448812029_047d981092_m.jpg",
        "roses/3634244527_e72c47842c_n.jpg",
        "roses/3903276582_fe05bf84c7_n.jpg",
        "roses/8437935944_aab997560a_n.jpg",
        "sunflowers/20753711039_0b11d24b50_n.jpg",
        "tulips/4558912791_084e440365_m.jpg",
    }:
        raise ValueError("attribution coverage differs from the inspected archive")
    return license_bytes, records, members


def phash_candidates(records: list[dict]) -> list[dict]:
    eligible = [r for r in records if r["rights_status"] == "cc-by-2.0-attributed"]
    pairs = []
    for index, left in enumerate(eligible):
        a = int(left["phash"], 16)
        for right in eligible[index + 1 :]:
            distance = (a ^ int(right["phash"], 16)).bit_count()
            if distance <= 6:
                pairs.append({"distance": distance, "left": left["id"], "right": right["id"]})
    return sorted(pairs, key=lambda item: (item["distance"], item["left"], item["right"]))


def prepare_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    raw_groups: dict[str, list[str]] = collections.defaultdict(list)
    pixel_groups: dict[str, list[str]] = collections.defaultdict(list)
    for record in records:
        if record["rights_status"] == "cc-by-2.0-attributed":
            raw_groups[record["raw_sha256"]].append(record["id"])
            pixel_groups[record["pixel_sha256"]].append(record["id"])
    expected_duplicate_groups = {
        frozenset(CONFLICTING_LABEL_IDS),
        frozenset(("sunflowers/14889392928_9742aed45b_m.jpg", REDUNDANT_SAME_CLASS_IDS[0])),
        frozenset(("sunflowers/15069459615_7e0fd61914_n.jpg", REDUNDANT_SAME_CLASS_IDS[1])),
    }
    observed_raw_groups = {frozenset(v) for v in raw_groups.values() if len(v) > 1}
    observed_pixel_groups = {frozenset(v) for v in pixel_groups.values() if len(v) > 1}
    if observed_raw_groups != expected_duplicate_groups or observed_pixel_groups != expected_duplicate_groups:
        raise ValueError("exact duplicate identity groups differ from the reviewed inventory")
    excluded = set(CONFLICTING_LABEL_IDS) | set(REDUNDANT_SAME_CLASS_IDS)
    for image_id in excluded:
        if not any(record["id"] == image_id for record in records):
            raise ValueError(f"reviewed duplicate exclusion absent: {image_id}")
    eligible = [r for r in records if r["rights_status"] == "cc-by-2.0-attributed" and r["id"] not in excluded]
    if len(eligible) != 3657:
        raise ValueError(f"expected 3,657 reviewed records, received {len(eligible)}")
    return eligible, records


def source_components(all_records: list[dict], eligible: list[dict]) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    attributed = [r for r in all_records if r["rights_status"] == "cc-by-2.0-attributed"]
    parent = list(range(len(attributed)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for field in ("creator_id", "raw_sha256", "pixel_sha256"):
        owners: dict[str, int] = {}
        for index, record in enumerate(attributed):
            key = record[field]
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index
    full_groups: dict[int, list[dict]] = collections.defaultdict(list)
    for index, record in enumerate(attributed):
        full_groups[find(index)].append(record)
    eligible_ids = {record["id"] for record in eligible}
    components: dict[str, list[dict]] = {}
    detail: dict[str, dict] = {}
    for rows in full_groups.values():
        all_ids = sorted(record["id"] for record in rows)
        component_id = hashlib.sha256("\n".join(all_ids).encode("utf-8")).hexdigest()
        kept = [record for record in rows if record["id"] in eligible_ids]
        if kept:
            components[component_id] = kept
            detail[component_id] = {
                "all_attributed_ids": all_ids,
                "eligible_ids": sorted(record["id"] for record in kept),
                "excluded_ids": sorted(set(all_ids) - {record["id"] for record in kept}),
                "creator_ids": sorted({record["creator_id"] for record in rows}),
            }
    if len(components) != 1528:
        raise ValueError(f"expected 1,528 source components, received {len(components)}")
    return components, detail


def assign_groups(components: dict[str, list[dict]]) -> tuple[dict[str, list[str]], dict]:
    group_ids = sorted(components)
    if len(group_ids) != 1528:
        raise ValueError(f"expected 1,528 source components, received {len(group_ids)}")
    groups = len(group_ids)
    splits = len(SPLITS)
    class_counts = {group_id: collections.Counter(r["label"] for r in components[group_id]) for group_id in group_ids}
    total = collections.Counter(r["label"] for rows in components.values() for r in rows)
    variable_count = groups * splits + splits * len(CLASSES) * 2
    matrix = lil_matrix((groups + splits * len(CLASSES), variable_count), dtype=np.float64)
    lower = np.zeros(matrix.shape[0], dtype=np.float64)
    upper = np.zeros(matrix.shape[0], dtype=np.float64)
    for group_index in range(groups):
        for split_index in range(splits):
            matrix[group_index, group_index * splits + split_index] = 1.0
        lower[group_index] = upper[group_index] = 1.0

    def deviation_index(split_index: int, class_index: int, positive: bool) -> int:
        return groups * splits + 2 * (split_index * len(CLASSES) + class_index) + (0 if positive else 1)

    for split_index, split in enumerate(SPLITS):
        for class_index, label in enumerate(CLASSES):
            row = groups + split_index * len(CLASSES) + class_index
            for group_index, group_id in enumerate(group_ids):
                matrix[row, group_index * splits + split_index] = class_counts[group_id][label]
            matrix[row, deviation_index(split_index, class_index, True)] = -1.0
            matrix[row, deviation_index(split_index, class_index, False)] = 1.0
            target = total[label] * FRACTIONS[split]
            lower[row] = upper[row] = target
    objective = np.zeros(variable_count, dtype=np.float64)
    for group_index, group_id in enumerate(group_ids):
        for split_index, split in enumerate(SPLITS):
            stable = int(hashlib.sha256(f"{SEED_LABEL}|{group_id}|{split}".encode()).hexdigest()[:8], 16)
            objective[group_index * splits + split_index] = stable / 0xFFFFFFFF * 1e-7
    for split_index, split in enumerate(SPLITS):
        for class_index, label in enumerate(CLASSES):
            scale = 1.0 / (total[label] * FRACTIONS[split])
            objective[deviation_index(split_index, class_index, True)] = scale
            objective[deviation_index(split_index, class_index, False)] = scale
    integrality = np.zeros(variable_count, dtype=np.int32)
    integrality[: groups * splits] = 1
    bounds = Bounds(
        np.zeros(variable_count), np.r_[np.ones(groups * splits), np.full(splits * len(CLASSES) * 2, np.inf)]
    )
    solution = milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 120.0},
    )
    if not solution.success or solution.x is None:
        raise RuntimeError(f"deterministic group stratification failed: {solution.message}")
    memberships = {split: [] for split in SPLITS}
    for group_index, group_id in enumerate(group_ids):
        selected = int(np.argmax(solution.x[group_index * splits : (group_index + 1) * splits]))
        memberships[SPLITS[selected]].append(group_id)
    detail = {
        split: {
            "component_count": len(memberships[split]),
            "image_count": sum(len(components[g]) for g in memberships[split]),
            "class_counts": dict(collections.Counter(r["label"] for g in memberships[split] for r in components[g])),
        }
        for split in SPLITS
    }
    if min(detail[s]["class_counts"][c] for s in ("recipe_dev", "final") for c in CLASSES) < 16:
        raise ValueError("group constraints make the required development/final class coverage infeasible")
    return memberships, {"groups": components, "summary": detail, "objective": float(solution.fun)}


def load_frozen_memberships(
    eligible: list[dict],
    components: dict[str, list[dict]],
    component_detail: dict[str, dict],
    membership_path: Path,
) -> tuple[dict[str, list[str]], dict]:
    """Validate and use the reviewed assignment without invoking a solver."""
    frozen = json.loads(membership_path.read_text())
    if frozen.get("status") != "reviewed_split_content_pending_complete_recipe":
        raise ValueError("project membership artifact is not in the reviewed state")
    if frozen.get("archive_sha256") != ARCHIVE_SHA256 or frozen.get("license_sha256") != LICENSE_SHA256:
        raise ValueError("membership artifact is bound to a different archive or credit ledger")
    if frozen.get("component_members") != component_detail:
        raise ValueError("archive-derived source component membership differs from the reviewed lock")
    if frozen.get("class_order") != list(CLASSES):
        raise ValueError("class order differs from the reviewed membership lock")
    memberships = frozen.get("split_components")
    if not isinstance(memberships, dict) or set(memberships) != set(SPLITS):
        raise ValueError("membership artifact must contain exactly the three reviewed splits")
    flattened = [group for split in SPLITS for group in memberships[split]]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(components):
        raise ValueError("split memberships must partition every source component exactly once")
    summary = {}
    for split in SPLITS:
        selected_groups = memberships[split]
        selected_rows = [record for group_id in selected_groups for record in components[group_id]]
        class_counts = collections.Counter(record["label"] for record in selected_rows)
        summary[split] = {
            "component_count": len(selected_groups),
            "image_count": len(selected_rows),
            "class_counts": dict(class_counts),
        }
    if summary != frozen.get("split_summary"):
        raise ValueError("computed split counts differ from the reviewed membership summary")
    if len(eligible) != sum(value["image_count"] for value in summary.values()):
        raise ValueError("the split does not cover all eligible labeled images")
    if min(summary[split]["class_counts"][label] for split in ("recipe_dev", "final") for label in CLASSES) < 16:
        raise ValueError("reviewed split no longer meets minimum per-class coverage")
    return memberships, {"groups": components, "summary": summary, "objective": frozen["solver"]["objective_value"]}


def resize_letterbox(raw: bytes) -> np.ndarray:
    display, _ = decode_display_rgb(raw)
    return letterbox_rgb(display, IMAGE_SIZE)


def download_archive(cache: Path) -> Path:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and cache.stat().st_size == ARCHIVE_BYTES and sha256_file(cache) == ARCHIVE_SHA256:
        return cache
    partial = cache.with_suffix(cache.suffix + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "jaxpathways-image-preparer/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as target:
        shutil.copyfileobj(response, target, length=1024 * 1024)
    if partial.stat().st_size != ARCHIVE_BYTES or sha256_file(partial) != ARCHIVE_SHA256:
        partial.unlink(missing_ok=True)
        raise ValueError("download did not match the pinned archive bytes")
    partial.replace(cache)
    return cache


def build_output(archive: Path, output: Path) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    resolved_output = output.expanduser().resolve()
    if resolved_output == project_root or project_root in resolved_output.parents:
        raise ValueError("prepared photo bytes must be written outside the project source tree")
    license_bytes, all_records, members = load_archive(archive)
    candidates = phash_candidates(all_records)
    eligible, _ = prepare_records(all_records)
    components, component_detail = source_components(all_records, eligible)
    membership_path = Path(__file__).with_name("frozen-memberships.json")
    memberships, split_info = load_frozen_memberships(eligible, components, component_detail, membership_path)
    excluded_unattributed = [r for r in all_records if r["rights_status"] == "unresolved_attribution_excluded"]
    excluded_duplicate = [
        {
            "id": image_id,
            "reason": "exact duplicate with conflicting class label; excluded from all labeled experiments",
        }
        for image_id in CONFLICTING_LABEL_IDS
    ] + [
        {"id": image_id, "reason": "redundant exact duplicate within same class; retained deterministic counterpart"}
        for image_id in REDUNDANT_SAME_CLASS_IDS
    ]
    split_of_group = {group: split for split, group_list in memberships.items() for group in group_list}
    component_by_id = {record["id"]: component_id for component_id, rows in components.items() for record in rows}
    excluded_candidate_ids = set(CONFLICTING_LABEL_IDS) | set(REDUNDANT_SAME_CLASS_IDS)
    excluded_candidate_ids.update(r["id"] for r in excluded_unattributed)
    for candidate in candidates:
        if candidate["left"] in excluded_candidate_ids or candidate["right"] in excluded_candidate_ids:
            candidate["decision"] = "excluded_by_attribution_or_exact_duplicate_rule"
            candidate["component_split"] = None
        else:
            candidate["decision"] = "same_creator_component_kept_together"
            candidate["component_split"] = split_of_group[component_by_id[candidate["left"]]]
        candidate["review_limit"] = "pHash is a candidate screen, not proof of all near-duplicate absence"
    ids_by_split = {
        split: sorted(image_id for image_id, group_id in component_by_id.items() if split_of_group[group_id] == split)
        for split in SPLITS
    }
    records_by_id = {r["id"]: r for r in all_records}
    label_index = {label: i for i, label in enumerate(CLASSES)}
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{resolved_output.name}-", dir=resolved_output.parent))
    try:
        (staging / "splits").mkdir()
        (staging / "arrays").mkdir()
        npy_files: dict[str, dict[str, str]] = {}
        split_row_files: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            ids = ids_by_split[split]
            # The archive member hashes are already verified; build a small map
            # of raw bytes for the prepared subset directly from the tar stream.
            # Data are not copied into the source tree.
            raw_by_id = read_selected_raw_members(archive, set(ids))
            images = np.stack([resize_letterbox(raw_by_id[image_id]) for image_id in ids]).astype(np.uint8)
            labels = np.asarray([label_index[records_by_id[image_id]["label"]] for image_id in ids], dtype=np.int32)
            for filename, array in (("images", images), ("labels", labels)):
                path = staging / "arrays" / f"{split}_{filename}.npy"
                with path.open("wb") as stream:
                    np.lib.format.write_array(stream, array, allow_pickle=False)
                npy_files[f"arrays/{split}_{filename}.npy"] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
            split_rows = []
            for image_id in ids:
                record = records_by_id[image_id]
                split_rows.append(
                    {
                        "id": image_id,
                        "label": record["label"],
                        "class_index": label_index[record["label"]],
                        "creator_id": record["creator_id"],
                        "raw_sha256": record["raw_sha256"],
                        "decoded_pixel_sha256": record["pixel_sha256"],
                        "source_url": record["source_url"],
                        "creator": record["creator"],
                    }
                )
            row_path = staging / "splits" / f"{split}.json"
            write_json(row_path, split_rows)
            split_row_files[f"splits/{split}.json"] = {
                "bytes": row_path.stat().st_size,
                "sha256": sha256_file(row_path),
                "row_count": len(split_rows),
            }
        write_json(
            staging / "source-manifest.json",
            {
                "source": {
                    "archive_url": ARCHIVE_URL,
                    "archive_bytes": ARCHIVE_BYTES,
                    "archive_sha256": ARCHIVE_SHA256,
                    "license_file_sha256": LICENSE_SHA256,
                    "license": "Upstream LICENSE.txt notice and per-image Flickr attribution entries state CC BY 2.0; the file is a credit ledger, not the complete legal-code text.",
                    "upstream_builder": "tensorflow/datasets tf_flowers",
                    "upstream_revision": "7b86414cb6408640c84620ea8c401dbac79138eb",
                },
                "software": {
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "pillow": PIL.__version__,
                    "scipy": scipy.__version__,
                },
                "archive_members": members,
                "records": all_records,
                "unresolved_attribution_exclusions": [
                    {
                        "id": r["id"],
                        "raw_sha256": r["raw_sha256"],
                        "reason": "no per-image creator/source entry in upstream LICENSE.txt; not used in any fit or evaluation",
                    }
                    for r in excluded_unattributed
                ],
                "exact_duplicate_exclusions": excluded_duplicate,
                "perceptual_neighbor_candidates_le6": candidates,
                "grouping": {
                    "component_key": "casefolded exact creator attribution name, connected with exact raw and EXIF-transposed RGB pixel identities before any split",
                    "component_count": len(split_info["groups"]),
                    "largest_component": max(len(v) for v in split_info["groups"].values()),
                    "assignment": "project-owned reviewed exact component membership; materialization validates the archive-derived component set and does not re-solve",
                    "assignment_seed_label": SEED_LABEL,
                    "split_components": memberships,
                    "component_members": component_detail,
                    "split_summary": split_info["summary"],
                    "objective": split_info["objective"],
                },
                "materialization_sources": {
                    "preparer_sha256": sha256_source(Path(__file__).resolve()),
                    "processor_sha256": sha256_file(Path(__file__).with_name("processor.py")),
                    "verifier_sha256": sha256_file(Path(__file__).with_name("verify.py")),
                    "membership_path": str(membership_path.relative_to(Path(__file__).resolve().parents[1])),
                    "membership_sha256": sha256_file(membership_path),
                    "requirements_sha256": sha256_file(Path(__file__).resolve().parents[1] / "requirements-image.txt"),
                    "requirements_lock_sha256": sha256_file(
                        Path(__file__).resolve().parents[1] / "requirements-image-lock.txt"
                    ),
                    "solver_provenance": json.loads(membership_path.read_text())["solver"],
                },
                "processor": {
                    "operation": "EXIF transpose; Pillow convert RGB using display channel values; aspect-preserving Lanczos contain; centered RGB(127,127,127) letterbox to 64x64; uint8 NHWC; no fitted normalization",
                    "icc_policy": "embedded profiles are not color-managed; exact source bytes and converted pixel bytes are recorded",
                    "transparency_policy": "Pillow RGB conversion behavior; archive images are verified JPEGs",
                    "image_size": [IMAGE_SIZE, IMAGE_SIZE, 3],
                },
                "candidate_status": "data and split candidate only; optimization prohibited until Sol reviews complete immutable training recipe",
                "eligible_records": len(eligible),
                "prepared_array_files": npy_files,
                "prepared_split_rows": split_row_files,
            },
        )
        (staging / "LICENSE.txt").write_bytes(license_bytes)
        if resolved_output.exists():
            raise FileExistsError(f"output already exists; use a new external directory: {resolved_output}")
        staging.replace(resolved_output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest_path = resolved_output / "source-manifest.json"
    return {
        "output": str(resolved_output),
        "manifest_sha256": sha256_file(manifest_path),
        "split_summary": split_info["summary"],
    }


def read_selected_raw_members(archive: Path, selected_ids: set[str]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if len(path.parts) != 3 or path.parts[0] != "flower_photos":
                continue
            image_id = PurePosixPath(*path.parts[1:]).as_posix()
            if image_id in selected_ids:
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError(f"missing bytes for selected image {image_id}")
                found[image_id] = stream.read()
    if found.keys() != selected_ids:
        raise ValueError(f"selected archive rows are missing: {sorted(selected_ids - found.keys())[:5]}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="verified local archive; use --download to acquire it")
    parser.add_argument("--download", action="store_true", help="download the pinned official archive")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/tmp")) / "flower_photos.tgz",
        help="archive cache path used with --download",
    )
    parser.add_argument("--output", type=Path, required=True, help="new external directory for data and memberships")
    args = parser.parse_args()
    if args.download and args.archive:
        parser.error("choose either --download or --archive")
    if args.download:
        archive = download_archive(args.cache.expanduser())
    elif args.archive:
        archive = args.archive.expanduser()
    else:
        parser.error("provide --archive PATH or --download")
    print(json.dumps(build_output(archive, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
