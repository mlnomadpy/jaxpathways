"""Render a measured, explicitly limited card from a completed run report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_model_card(report: dict[str, Any], recipe: dict[str, Any]) -> str:
    """Create a card whose claims are derived from measured output fields."""
    if report.get("schema") != "jaxpathways.image-pilot-run.v1":
        raise ValueError("model card requires a completed image-pilot run report")
    if report.get("optimizer_updates", 1) == 0 or not report.get("stage_checkpoints"):
        raise ValueError(
            "model card cannot claim measured training without stage checkpoint receipts"
        )
    quality = report["quality_gate"]
    gates = quality["selected_candidate_only"]
    failed = [name for name, passed in gates.items() if not passed]
    gate_status = "passed" if not failed else "failed"
    selected = report["deploy_selection"]["dev_only_selected"]
    selected_metrics = report["final_metrics"][selected]
    source = recipe["data"]["upstream"]
    final_counts = report["final_class_counts"]
    exposures = report["stage_execution"]
    stage_rows = []
    for name in report["stage_order"]:
        entry = exposures[name]
        stage_rows.append(
            f"| `{name}` | {entry['objective']} | {entry['updates']} × {entry['batch_size']} | "
            f"{entry['measured_input_images']} | {entry['measured_model_views']} | {entry['measured_pixel_values']} | "
            f"{entry['measured_selected_mae_target_values']} |"
        )
    failed_lines = (
        "\n".join(f"- `{name}`: failed" for name in failed) if failed else "- None."
    )
    class_counts = ", ".join(f"{name}: {count}" for name, count in final_counts.items())
    return f"""# Measured model card: bounded flower-image CPU pilot

This report describes a bounded, small-architecture CPU pilot. It is not a 300M model, a broad visual-representation benchmark, or deployment qualification. The selected candidate is `{selected}`. Its precommitted pilot quality floors **{gate_status}** on the fixed final inventory. Floor status is derived from the run report, not assumed from the reference outcome.

## Data and rights

- Source: {source['name']} ([upstream archive]({source['url']})), SHA-256 `{source['sha256']}`.
- The archive's `LICENSE.txt` is an attribution notice with per-image CC BY 2.0 credits, not the complete legal code. Nine unattributed images and four exact-content conflict/redundancy exclusions are omitted from the eligible split.
- Class order: {', '.join(recipe['data']['class_order'])}.
- Final held-out counts: {class_counts}; {sum(final_counts.values())} images total.
- Recipe SHA-256: `{report['recipe_sha256']}`. Prepared source-manifest SHA-256: `{report['source_manifest_sha256']}`.
- The final split is source-component-disjoint from fit and recipe-development. The selected checkpoint ancestry and every stage's measured IDs are recorded in the run report, per-stage traces and checkpoint sidecars.

## Training exposure

| Stage | Objective (units are stage-specific) | Updates × batch | Source image presentations | Model views | RGB pixel values presented | MAE target values scored |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(stage_rows)}

These are measured presentations, not the prepared inventory size. A repeated image in a later epoch is a new presentation. MAE counts only its declared hidden-patch target values. Loss magnitudes from cross-entropy, pixel MSE and contrastive NT-Xent have different units and must not be compared as a common progress scale.

## Final evaluation

- Raw selected-candidate macro balanced accuracy: {selected_metrics['raw']['macro_balanced_accuracy']:.6f}.
- Raw NLL: {selected_metrics['raw']['nll']:.6f}; recipe-development-calibrated NLL: {selected_metrics['calibrated']['nll']:.6f}.
- The complete source-component bootstrap intervals, corruption results, retrieval ranks, class counts and all candidate/control metrics are in `run-report.json` and its referenced raw artifacts.
- Selected-candidate floor failures:
{failed_lines}

The final inventory was reserved for measurement but was inspected in this bounded engineering run. Its results are not presented as a newly blind estimate. Final labels do not select the branch, tune temperature, or change the recipe.

## Limitations and release status

- A small five-class flower corpus does not establish general image understanding or transfer quality.
- The CPU pilot's small ViT does not establish full target-architecture memory, performance, or training feasibility.
- Human task acceptance, target-scale qualification, service/device behavior, and any public model release remain pending.
- No standalone weight redistribution license is asserted. The local consumer bundle is private correctness evidence only; confirm rights and attribution requirements before any redistribution.

Use the exact run receipt, immutable recipe, source pins, checkpoint ancestry and raw metrics to reproduce or audit these results. Do not treat this card by itself as an acceptance or release certificate.
"""


def write_model_card(report_path: str | Path, recipe_path: str | Path) -> Path:
    report_file = Path(report_path).resolve()
    recipe_file = Path(recipe_path).resolve()
    report = json.loads(report_file.read_text())
    recipe = json.loads(recipe_file.read_text())
    content = render_model_card(report, recipe)
    destination = report_file.parent / "MODEL_CARD.md"
    destination.write_text(content)
    return destination
