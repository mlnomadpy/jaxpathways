"""Pack compact, source-bound C01 evidence without model-weight payloads."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
from tokenizers import Tokenizer

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = Path("/private/tmp/jaxpathways-embedding-c01-check-final8/full-run")
DEFAULT_OUTPUT = PROJECT / "reference/evidence/final8"
EXPECTED_RECEIPT_SHA256 = (
    "514287dcfd125d3127f93c43d0c20540b6325c9d4d39f4a803ea2fa57aa942b8"
)
METHODS = (
    "bm25",
    "external_minilm_l6_v2",
    "phase1_mlm",
    "phase2_contrastive_only",
    "phase2_contrastive_plus_retained_mlm",
    "phase3_human_pairs",
    "random_initialization",
)
FINAL_PROTOCOL = "final_official_dev_custom_protocol"
METRIC_KEYS = (
    "recall@1",
    "recall@5",
    "recall@10",
    "ndcg@1",
    "ndcg@5",
    "ndcg@10",
    "mrr",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _copy_verified(run: Path, stage: Path, relative: str, receipt: dict) -> dict:
    source = run / relative
    expected = receipt[relative]
    actual = _sha256(source)
    if actual != expected:
        raise ValueError(f"source hash differs from final8 receipt: {relative}")
    destination = stage / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    copied = _sha256(destination)
    if copied != actual:
        raise ValueError(f"copied hash changed: {relative}")
    return {"sha256": actual, "size_bytes": source.stat().st_size}


def _stable_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a NumPy-readable archive with fixed ZIP metadata and member order."""
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(
                array_buffer, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                array_buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output.getvalue())


def _independent_metrics(query_ids, document_ids, ordinal_ranks, qrels):
    """Recompute macro Recall/MRR/nDCG from full ranks and qrels only."""
    query_ids = [int(value) for value in query_ids]
    document_ids = [int(value) for value in document_ids]
    if len(set(query_ids)) != len(query_ids) or len(set(document_ids)) != len(
        document_ids
    ):
        raise ValueError("query and corpus ID inventories must be unique")
    if ordinal_ranks.shape != (len(query_ids), len(document_ids)):
        raise ValueError("rank matrix must cover every query and candidate document")
    expected_ordinals = np.arange(len(document_ids), dtype=np.uint16)
    if not np.all(np.sort(ordinal_ranks, axis=1) == expected_ordinals[None, :]):
        raise ValueError("each full ranking must be a complete document permutation")
    if set(map(str, query_ids)) != set(qrels):
        raise ValueError("shared qrels must cover each query exactly once")

    rows = []
    for row_index, query_id in enumerate(query_ids):
        relevant = set(map(int, qrels[str(query_id)]))
        ranked = [document_ids[int(index)] for index in ordinal_ranks[row_index]]
        if not relevant <= set(ranked):
            raise ValueError(f"query {query_id} ranking omitted a qrel positive")
        if not relevant:
            rows.append(
                {
                    "query_id": query_id,
                    "judged": False,
                    "relevant_count": 0,
                    "first_relevant_rank": None,
                }
            )
            continue
        ranks = [
            i + 1 for i, document_id in enumerate(ranked) if document_id in relevant
        ]
        row = {
            "query_id": query_id,
            "judged": True,
            "relevant_count": len(relevant),
            "first_relevant_rank": min(ranks),
        }
        for cutoff in (1, 5, 10):
            gains = [rank for rank in ranks if rank <= cutoff]
            row[f"recall@{cutoff}"] = len(gains) / len(relevant)
            dcg = sum(1.0 / math.log2(rank + 1.0) for rank in gains)
            ideal = sum(
                1.0 / math.log2(rank + 1.0)
                for rank in range(1, min(len(relevant), cutoff) + 1)
            )
            row[f"ndcg@{cutoff}"] = dcg / ideal
        row["mrr"] = 1.0 / min(ranks)
        rows.append(row)

    judged_rows = [row for row in rows if row["judged"]]
    result = {
        "queries_total": len(rows),
        "queries_judged": len(judged_rows),
        "queries_unjudged": len(rows) - len(judged_rows),
        "per_query": rows,
    }
    for key in METRIC_KEYS:
        result[key] = (
            0.0
            if not judged_rows
            else sum(float(row[key]) for row in judged_rows) / len(judged_rows)
        )
    return result


def _assert_metrics_close(recomputed: dict, saved: dict, method: str) -> float:
    maximum_difference = 0.0
    for key in (*METRIC_KEYS, "queries_total", "queries_judged", "queries_unjudged"):
        actual = float(recomputed[key])
        expected = float(saved[key])
        maximum_difference = max(maximum_difference, abs(actual - expected))
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"{method} metric differs for {key}: {actual} != {expected}"
            )
    if len(recomputed["per_query"]) != len(saved["per_query"]):
        raise ValueError(f"{method} per-query row count differs")
    for actual_row, expected_row in zip(recomputed["per_query"], saved["per_query"]):
        for key in actual_row:
            actual, expected = actual_row[key], expected_row.get(key)
            if isinstance(actual, float):
                maximum_difference = max(
                    maximum_difference, abs(actual - float(expected))
                )
                if not math.isclose(
                    actual, float(expected), rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(f"{method} per-query {key} differs")
            elif actual != expected:
                raise ValueError(f"{method} per-query {key} differs")
    return maximum_difference


def _rank_source(run: Path, method: str):
    folder = run / "raw" / method / FINAL_PROTOCOL
    filename = "rankings.npz" if method == "bm25" else "retrieval-arrays.npz"
    path = folder / filename
    with np.load(path, allow_pickle=False) as data:
        required = {"query_ids", "document_ids", "ranked_doc_ids"}
        if not required <= set(data.files):
            raise ValueError(f"missing rank fields in {path}")
        return (
            np.asarray(data["query_ids"], dtype=np.int64),
            np.asarray(data["document_ids"], dtype=np.int64),
            np.asarray(data["ranked_doc_ids"], dtype=np.uint32),
            path,
        )


def _build(run: Path, stage: Path, packager_path: Path) -> None:
    receipt_path = run / "sourcebound-receipt.json"
    receipt_sha = _sha256(receipt_path)
    if receipt_sha != EXPECTED_RECEIPT_SHA256:
        raise ValueError("input is not the immutable accepted final8 source-bound run")
    receipt = json.loads(receipt_path.read_text())
    independent_path = run.parent / "independent-check.json"
    independent = json.loads(independent_path.read_text())
    if independent.get("status") != "pass":
        raise ValueError("historical independent-check record is not passing")

    source_files = {}
    copied_paths = [
        "pilot-report.json",
        "training-trace.json",
        "figure-data.json",
        "weak-pair-inventory.json",
        "tokenizer-info.json",
        "frozen-recipe-input.json",
        "baseline-receipt.json",
        "model-config.json",
        "mlm-input-ids.npy",
        "cold-vectors.npy",
        "figures/final-retrieval.svg",
        "figures/training-objectives.svg",
    ]
    for relative in copied_paths:
        source_files[relative] = _copy_verified(
            run, stage, relative, receipt["run_artifacts_sha256"]
        )

    checkpoint_paths = []
    for stage_name in (
        "phase1-mlm",
        "phase2_contrastive_only",
        "phase2_contrastive_plus_retained_mlm",
        "phase3-human-pairs",
    ):
        relative = f"checkpoints/{stage_name}/checkpoint.json"
        checkpoint_paths.append(relative)
        source_files[relative] = _copy_verified(
            run, stage, relative, receipt["run_artifacts_sha256"]
        )

    for relative, source, historical_sha in (
        ("sourcebound-receipt.json", receipt_path, receipt_sha),
        (
            "independent-check.json",
            independent_path,
            _sha256(independent_path),
        ),
    ):
        destination = stage / relative
        shutil.copyfile(source, destination)
        source_files[relative] = {
            "sha256": historical_sha,
            "size_bytes": source.stat().st_size,
        }

    package_files = (
        "README.md",
        "DATA_PROVENANCE.md",
        "MEASURED_EVIDENCE.json",
        "package.json",
        "config.json",
    )
    for filename in package_files:
        relative = f"local-package/{filename}"
        source_files[relative] = _copy_verified(
            run, stage, relative, receipt["run_artifacts_sha256"]
        )

    # Keep a numeric canary but not source text or the trained tokenizer file.
    cold_input_path = run / "cold-input.json"
    cold_input_sha = _sha256(cold_input_path)
    if cold_input_sha != receipt["run_artifacts_sha256"]["cold-input.json"]:
        raise ValueError("cold canary text input differs from historical receipt")
    tokenizer_path = run / "local-package/tokenizer.json"
    tokenizer_sha = _sha256(tokenizer_path)
    expected_tokenizer_sha = receipt["run_artifacts_sha256"][
        "local-package/tokenizer.json"
    ]
    if tokenizer_sha != expected_tokenizer_sha:
        raise ValueError("package tokenizer differs from historical receipt")
    package_config = json.loads((run / "local-package/config.json").read_text())
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cold_texts = json.loads(cold_input_path.read_text())["texts"]
    token_rows = []
    for text in cold_texts:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        values = [package_config["bos_id"], *map(int, ids), package_config["eos_id"]]
        if len(values) > package_config["max_length"]:
            values = values[: package_config["max_length"] - 1] + [
                package_config["eos_id"]
            ]
        token_rows.append(values)
    token_lengths = np.asarray([len(row) for row in token_rows], dtype=np.uint16)
    token_width = max(map(len, token_rows), default=1)
    token_inputs = np.full(
        (len(token_rows), token_width), package_config["pad_id"], dtype=np.int32
    )
    for index, row in enumerate(token_rows):
        token_inputs[index, : len(row)] = row
    _stable_npz(
        stage / "cold-token-input.npz",
        {"input_ids": token_inputs, "lengths": token_lengths},
    )
    source_files["cold-token-input.npz"] = {
        "sha256": _sha256(stage / "cold-token-input.npz"),
        "size_bytes": (stage / "cold-token-input.npz").stat().st_size,
        "derived_from_text_sha256": cold_input_sha,
        "tokenizer_sha256": tokenizer_sha,
        "contains_source_text": False,
        "text_count": len(cold_texts),
        "max_length": package_config["max_length"],
    }

    report = json.loads((stage / "pilot-report.json").read_text())
    qrels_sources = {}
    rank_input_sources = {}
    metrics_checks = {}
    shared_query_ids = None
    shared_document_ids = None
    document_ordinal = None
    qrels_bytes = None
    output_metrics_dir = stage / "rankings/metrics"
    output_metrics_dir.mkdir(parents=True, exist_ok=True)

    for method in METHODS:
        raw_query_ids, document_ids, ranked_doc_ids, source_path = _rank_source(
            run, method
        )
        if shared_query_ids is None:
            shared_query_ids = raw_query_ids
            shared_document_ids = document_ids
            document_ordinal = {
                int(document_id): index
                for index, document_id in enumerate(document_ids)
            }
            if len(document_ordinal) != len(document_ids):
                raise ValueError("source corpus document IDs are not unique")
        elif not np.array_equal(raw_query_ids, shared_query_ids) or not np.array_equal(
            document_ids, shared_document_ids
        ):
            raise ValueError(f"{method} rank inventory differs from shared IDs")
        if ranked_doc_ids.shape != (len(shared_query_ids), len(shared_document_ids)):
            raise ValueError(f"{method} rank matrix is not full-corpus/full-query")
        ordinal = np.empty(ranked_doc_ids.shape, dtype=np.uint16)
        for row in range(ranked_doc_ids.shape[0]):
            try:
                ordinal[row] = [
                    document_ordinal[int(value)] for value in ranked_doc_ids[row]
                ]
            except KeyError as error:
                raise ValueError(
                    f"{method} ranking contains an unknown document ID"
                ) from error
        npz_path = stage / f"rankings/{method}.npz"
        _stable_npz(npz_path, {"ranked_document_indices": ordinal})
        rank_input_hash = _sha256(source_path)
        rank_input_rel = source_path.relative_to(run).as_posix()
        if receipt["run_artifacts_sha256"].get(rank_input_rel) != rank_input_hash:
            raise ValueError(f"rank input differs from receipt: {rank_input_rel}")
        rank_input_sources[method] = {
            "path": rank_input_rel,
            "sha256": rank_input_hash,
            "size_bytes": source_path.stat().st_size,
        }

        metric_rel = f"raw/{method}/{FINAL_PROTOCOL}/metrics.json"
        qrels_rel = f"raw/{method}/{FINAL_PROTOCOL}/qrels.json"
        metric_path = run / metric_rel
        qrels_path = run / qrels_rel
        if _sha256(metric_path) != receipt["run_artifacts_sha256"][metric_rel]:
            raise ValueError(f"metric source differs from receipt: {metric_rel}")
        if _sha256(qrels_path) != receipt["run_artifacts_sha256"][qrels_rel]:
            raise ValueError(f"qrels source differs from receipt: {qrels_rel}")
        metrics_destination = output_metrics_dir / f"{method}.json"
        shutil.copyfile(metric_path, metrics_destination)
        saved_metrics = json.loads(metric_path.read_text())
        metrics_checks[method] = _assert_metrics_close(
            _independent_metrics(
                shared_query_ids,
                shared_document_ids,
                ordinal,
                json.loads(qrels_path.read_text()),
            ),
            saved_metrics,
            method,
        )
        if qrels_bytes is None:
            qrels_bytes = qrels_path.read_bytes()
        elif qrels_bytes != qrels_path.read_bytes():
            raise ValueError("per-method qrels differ; cannot deduplicate")
        qrels_sources[method] = {
            "path": qrels_rel,
            "sha256": _sha256(qrels_path),
            "size_bytes": qrels_path.stat().st_size,
        }
        source_files[f"rankings/metrics/{method}.json"] = {
            "sha256": _sha256(metrics_destination),
            "size_bytes": metrics_destination.stat().st_size,
            "source_path": metric_rel,
        }

    _stable_npz(
        stage / "rankings/shared-inventory.npz",
        {"query_ids": shared_query_ids, "document_ids": shared_document_ids},
    )
    qrels_destination = stage / "rankings/shared-qrels.json"
    qrels_destination.write_bytes(qrels_bytes)
    source_files["rankings/shared-qrels.json"] = {
        "sha256": _sha256(qrels_destination),
        "size_bytes": qrels_destination.stat().st_size,
        "source_hashes_by_method": qrels_sources,
    }
    inventory_path = stage / "rankings/shared-inventory.npz"
    source_files["rankings/shared-inventory.npz"] = {
        "sha256": _sha256(inventory_path),
        "size_bytes": inventory_path.stat().st_size,
        "source_paths": [rank_input_sources[method]["path"] for method in METHODS],
        "query_count": len(shared_query_ids),
        "document_count": len(shared_document_ids),
    }
    derived_ranks = {}
    for method in METHODS:
        rank_path = stage / f"rankings/{method}.npz"
        derived_ranks[method] = {
            "sha256": _sha256(rank_path),
            "size_bytes": rank_path.stat().st_size,
            "input_path": rank_input_sources[method]["path"],
            "input_sha256": rank_input_sources[method]["sha256"],
            "field": "ranked_doc_ids",
            "transformation": "Map every ranked document ID to its ordinal in shared-inventory.document_ids; store full ordering as uint16 indices.",
            "shape": [int(len(shared_query_ids)), int(len(shared_document_ids))],
            "dtype": "uint16",
        }

    reconstruction = {
        "schema": "jaxpathways.c01.independent-rank-metric-check.v1",
        "protocol": FINAL_PROTOCOL,
        "query_count": len(shared_query_ids),
        "document_count": len(shared_document_ids),
        "judged_query_count": sum(
            bool(value) for value in json.loads(qrels_bytes).values()
        ),
        "methods": {
            method: {
                "metrics_sha256": source_files[f"rankings/metrics/{method}.json"][
                    "sha256"
                ],
                "rank_indices_sha256": derived_ranks[method]["sha256"],
                "max_absolute_metric_or_per_query_difference": metrics_checks[method],
                "all_full_rank_permutations_valid": True,
                "metrics_reconstructed_from_full_ranks_and_shared_qrels": True,
            }
            for method in METHODS
        },
    }
    _write_json(stage / "rankings/independent-metric-check.json", reconstruction)
    source_files["rankings/independent-metric-check.json"] = {
        "sha256": _sha256(stage / "rankings/independent-metric-check.json"),
        "size_bytes": (stage / "rankings/independent-metric-check.json").stat().st_size,
    }

    # Explain every original run artifact not included byte-for-byte in the packet.
    copied_original = (
        set(copied_paths)
        | set(checkpoint_paths)
        | {
            "raw/phase1_mlm/final_official_dev_custom_protocol/metrics.json",
            "raw/phase2_contrastive_only/final_official_dev_custom_protocol/metrics.json",
            "raw/phase2_contrastive_plus_retained_mlm/final_official_dev_custom_protocol/metrics.json",
            "raw/phase3_human_pairs/final_official_dev_custom_protocol/metrics.json",
            "raw/random_initialization/final_official_dev_custom_protocol/metrics.json",
            "raw/external_minilm_l6_v2/final_official_dev_custom_protocol/metrics.json",
            "raw/bm25/final_official_dev_custom_protocol/metrics.json",
            "local-package/README.md",
            "local-package/DATA_PROVENANCE.md",
            "local-package/MEASURED_EVIDENCE.json",
            "local-package/package.json",
            "local-package/config.json",
            f"raw/{METHODS[0]}/{FINAL_PROTOCOL}/qrels.json",
        }
    )
    rank_input_paths = {value["path"] for value in rank_input_sources.values()}
    omitted = {}
    for relative, sha in sorted(receipt["run_artifacts_sha256"].items()):
        if relative in copied_original:
            continue
        if relative.startswith("checkpoints/") and relative.endswith("state.npz"):
            reason = "Trained optimizer/model state retained only in the external complete review packet pending a standalone-weight rights review."
        elif relative.startswith("baseline_snapshot/"):
            reason = "Pinned third-party baseline snapshot omitted; exact revision and every snapshot hash remain in baseline-receipt.json and the source-bound receipt."
        elif relative in rank_input_paths:
            reason = "Full input rank/score/vector file represented by a smaller derived full-order uint16 rank array; original hash retained for provenance."
        elif relative.startswith("raw/") and "/recipe_development/" in relative:
            reason = "Recipe-development arrays are summarized in the report and figure data; the compact packet retains final-protocol full rankings only."
        elif relative.startswith("raw/") and relative.endswith("qrels.json"):
            reason = "Byte-identical qrels deduplicated to rankings/shared-qrels.json."
        elif relative == "cold-input.json":
            reason = "Original biomedical canary text is omitted; deterministic numeric token IDs and expected vectors are retained with their source hashes."
        elif relative.startswith("local-package/") and relative.endswith(
            ("model.safetensors", "tokenizer.json")
        ):
            reason = "Trained package weight/tokenizer payload omitted pending rights review; source hashes remain in package.json and the immutable receipt."
        elif relative.startswith("local-package/"):
            reason = "Package loader/source payload is not a loadable public package without its reviewed weights and tokenizer; project source remains available separately."
        elif "__pycache__/" in relative or relative.startswith("cold-process/"):
            reason = "Ephemeral interpreter or cold-process output; reproducible validation results and numeric canary vector are retained."
        elif relative.startswith(("bad-", "candidate-pool-sentinel")):
            reason = "Checker mutation fixture; its acceptance result is summarized by the independent-check record and report."
        elif relative.endswith(".npz") and relative.startswith("raw/"):
            reason = "Large development score/vector arrays omitted; final protocol retains complete full-corpus rank order and all reported metrics."
        else:
            reason = "Checker operational artifact not needed to interpret retained metrics, plots, split identities, or checkpoint ancestry."
        omitted_entry = {"sha256": sha, "reason": reason}
        omitted_source = run / relative
        if omitted_source.is_file():
            omitted_entry["size_bytes"] = omitted_source.stat().st_size
        omitted[relative] = omitted_entry

    baseline_files = receipt["baseline_file_sha256"]
    omitted["baseline_snapshot_files"] = {
        "files": baseline_files,
        "reason": "Pinned external baseline model/tokenizer snapshot is intentionally not included; baseline-receipt.json records revision, license metadata, and content hashes. A rerun must fetch the exact pinned revision or provide the exact external snapshot.",
    }

    packager_sha = _sha256(packager_path)
    manifest = {
        "schema": "jaxpathways.embedding-c01.compact-evidence.v1",
        "historical_run": {
            "name": "final8",
            "receipt_sha256": receipt_sha,
            "independent_check_sha256": source_files["independent-check.json"][
                "sha256"
            ],
            "checker_sha256_at_historical_run": receipt["source_files_sha256"][
                "tests/check.py"
            ],
            "amendment_path": "../checker-attention-amendment.json",
            "current_checker_sha256": json.loads(
                (
                    PROJECT / "reference/evidence/checker-attention-amendment.json"
                ).read_text()
            )["amendment"]["current_checker_sha256"],
        },
        "packager": {
            "project_path": packager_path.relative_to(PROJECT).as_posix(),
            "sha256": packager_sha,
            "output_is_deterministic": True,
        },
        "historical_source_files_sha256": receipt["source_files_sha256"],
        "included_files": source_files,
        "rank_input_files": rank_input_sources,
        "derived_full_rank_files": derived_ranks,
        "omitted_artifacts": omitted,
        "metric_reconstruction": {
            "path": "rankings/independent-metric-check.json",
            "sha256": _sha256(stage / "rankings/independent-metric-check.json"),
            "method": "Independent Python implementation of Recall@1/5/10, reciprocal rank, and binary nDCG@1/5/10 from full rank lists and the shared qrels; checked every per-query row and macro value against saved metrics.json.",
        },
        "total_bytes_excludes": [
            "This manifest's own bytes, to avoid circular self-hashing.",
            "README.md and evidence-index.json at the parent reference directory.",
            "checker-attention-amendment.json at the parent evidence directory.",
        ],
    }
    _write_json(stage / "retention-manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite evidence packet: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".c01-evidence-", dir=output.parent))
    try:
        _build(run, stage, Path(__file__).resolve())
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"wrote {output} ({total} bytes)")


if __name__ == "__main__":
    main()
