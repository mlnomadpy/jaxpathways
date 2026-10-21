"""Render a phase-separated teaching plot bound to immutable final8 evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from plots import loss_figure

REFERENCE = PROJECT / "reference"
FINAL8 = REFERENCE / "evidence/final8"
OUTPUT_DIR = REFERENCE / "teaching"
OUTPUT_SVG = OUTPUT_DIR / "objective-panels.svg"
MANIFEST = OUTPUT_DIR / "manifest.json"
TRACE_PATH = FINAL8 / "training-trace.json"
FIGURE_DATA_PATH = FINAL8 / "figure-data.json"
HISTORICAL_PLOT = FINAL8 / "figures/training-objectives.svg"

EXPECTED_INPUTS = {
    "training_trace_sha256": "a9a8016bf4672083859daafe1a548c436bc5ebbbb7c463bf2ef265778917e5d1",
    "figure_data_sha256": "1ca4d365ca5f1179ecd858647d6eb781cab504bc3c1fd20e23968b79c9ce6766",
    "historical_plot_sha256": "485b691ed3ddf9b11239de4d6f4a30a0a31b433d5be5c6330189824776bfed72",
}
STAGES = (
    "phase1_mlm",
    "phase2_contrastive_only",
    "phase2_contrastive_plus_retained_mlm",
    "phase3_human_pairs",
)
LOSS_FIELDS = {
    "phase1_mlm": "loss_nats_per_masked_token",
    "phase2_contrastive_only": "contrastive_plus_weighted_mlm_loss",
    "phase2_contrastive_plus_retained_mlm": "contrastive_plus_weighted_mlm_loss",
    "phase3_human_pairs": "multi_positive_loss",
}
UNITS = {
    "phase1_mlm": "nats per selected masked token",
    "phase2_contrastive_only": "contrastive objective units",
    "phase2_contrastive_plus_retained_mlm": "mixed contrastive + weighted MLM units",
    "phase3_human_pairs": "multi-positive contrastive objective units",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_and_validate() -> tuple[dict[str, list[dict]], dict[str, dict]]:
    actual_hashes = {
        "training_trace_sha256": _sha256(TRACE_PATH),
        "figure_data_sha256": _sha256(FIGURE_DATA_PATH),
        "historical_plot_sha256": _sha256(HISTORICAL_PLOT),
    }
    if actual_hashes != EXPECTED_INPUTS:
        raise ValueError("immutable final8 figure inputs differ from accepted hashes")

    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    figure_data = json.loads(FIGURE_DATA_PATH.read_text(encoding="utf-8"))
    recorded_traces = figure_data.get("training_objectives", {}).get("traces")
    if trace != recorded_traces:
        raise ValueError("figure-data trace rows differ from the immutable trace")

    grouped = {stage: [] for stage in STAGES}
    for row in trace:
        stage = row.get("stage")
        if stage not in grouped:
            raise ValueError(f"unexpected stage in final8 trace: {stage}")
        grouped[stage].append(row)

    expected_counts = {
        "phase1_mlm": 8,
        "phase2_contrastive_only": 8,
        "phase2_contrastive_plus_retained_mlm": 8,
        "phase3_human_pairs": 71,
    }
    if {stage: len(rows) for stage, rows in grouped.items()} != expected_counts:
        raise ValueError("final8 update counts differ from the teaching panel contract")
    for stage, rows in grouped.items():
        if [row.get("step") for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"{stage} steps are not contiguous and one-based")
        field = LOSS_FIELDS[stage]
        for row in rows:
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"missing/non-finite {field} in {stage}")
    tail = grouped["phase3_human_pairs"][-1]
    if not (
        tail["multi_positive_loss"] == 0.0
        and len(tail["query_ids"]) == 1
        and tail["positive_pairs"] == 1
        and len(tail["candidate_doc_ids"]) == 1
    ):
        raise ValueError("phase 3 final-batch interpretation no longer matches trace")
    return grouped, actual_hashes


def render() -> dict:
    grouped, inputs = _load_and_validate()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loss_figure(OUTPUT_SVG, [row for stage in STAGES for row in grouped[stage]])
    values = {
        stage: {
            "updates": len(rows),
            "loss_field": LOSS_FIELDS[stage],
            "unit": UNITS[stage],
            "minimum": min(float(row[LOSS_FIELDS[stage]]) for row in rows),
            "maximum": max(float(row[LOSS_FIELDS[stage]]) for row in rows),
            "final_value": float(rows[-1][LOSS_FIELDS[stage]]),
            "zero_preserving_vertical_axis": True,
        }
        for stage, rows in grouped.items()
    }
    tail = grouped["phase3_human_pairs"][-1]
    manifest = {
        "schema": "jaxpathways.embedding-c01.teaching-figure.v2",
        "purpose": "Four phase-specific teaching panels from the same generic renderer used by learner runs; historical final8 trace, plot and receipt are preserved.",
        "historical_inputs": {
            "training_trace": {
                "path": "reference/evidence/final8/training-trace.json",
                "sha256": inputs["training_trace_sha256"],
            },
            "figure_data": {
                "path": "reference/evidence/final8/figure-data.json",
                "sha256": inputs["figure_data_sha256"],
                "trace_rows_equal_training_trace": True,
            },
            "historical_combined_plot": {
                "path": "reference/evidence/final8/figures/training-objectives.svg",
                "sha256": inputs["historical_plot_sha256"],
                "preserved_unchanged": True,
            },
            "historical_receipt": {
                "path": "reference/evidence/final8/sourcebound-receipt.json",
                "sha256": _sha256(FINAL8 / "sourcebound-receipt.json"),
            },
        },
        "renderer": {
            "path": "tools/render_teaching_objectives.py",
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "runtime_renderer": {
            "path": "plots.py",
            "sha256": _sha256(PROJECT / "plots.py"),
            "shared_function": "loss_figure",
        },
        "outputs": {
            "phase_panels": {
                "path": "reference/teaching/objective-panels.svg",
                "sha256": _sha256(OUTPUT_SVG),
                "size_bytes": OUTPUT_SVG.stat().st_size,
            }
        },
        "panels": values,
        "phase3_final_batch": {
            "step": tail["step"],
            "query_count": len(tail["query_ids"]),
            "positive_pair_count": tail["positive_pairs"],
            "candidate_count": len(tail["candidate_doc_ids"]),
            "loss": tail["multi_positive_loss"],
            "interpretation": "The last partial batch contains one query, one positive pair, and one candidate. Its zero loss is not evidence of retrieval quality.",
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    result = render()
    print(
        f"rendered {result['outputs']['phase_panels']['path']} "
        f"sha256={result['outputs']['phase_panels']['sha256']}"
    )
