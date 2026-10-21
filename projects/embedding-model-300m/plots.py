"""Small dependency-free SVG figures from actual saved pilot measurements."""

from __future__ import annotations

import html
import math
from pathlib import Path


def _svg_start(title, subtitle, width=900, height=520):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf9"/>',
        f'<text x="78" y="42" font-size="23" font-family="sans-serif" font-weight="700" fill="#202626">{html.escape(title)}</text>',
        f'<text x="78" y="68" font-size="13" font-family="sans-serif" fill="#596563">{html.escape(subtitle)}</text>',
    ]


def loss_figure(path: Path, records):
    """Render one independent, zero-based panel for each observed objective phase."""
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "phase1_mlm": "#286b8e",
        "phase2_contrastive_only": "#b46c2a",
        "phase2_contrastive_plus_retained_mlm": "#49835d",
        "phase3_human_pairs": "#884c79",
    }
    titles = {
        "phase1_mlm": "Phase 1 · masked language modeling",
        "phase2_contrastive_only": "Phase 2 · contrastive-only",
        "phase2_contrastive_plus_retained_mlm": "Phase 2 · contrastive + retained MLM",
        "phase3_human_pairs": "Phase 3 · human-evidence pairs",
    }
    units = {
        "phase1_mlm": "nats per selected masked token",
        "phase2_contrastive_only": "contrastive objective units",
        "phase2_contrastive_plus_retained_mlm": "mixed contrastive + weighted MLM units",
        "phase3_human_pairs": "multi-positive contrastive objective units",
    }
    fields = {
        "phase1_mlm": "loss_nats_per_masked_token",
        "phase2_contrastive_only": "contrastive_plus_weighted_mlm_loss",
        "phase2_contrastive_plus_retained_mlm": "contrastive_plus_weighted_mlm_loss",
        "phase3_human_pairs": "multi_positive_loss",
    }
    series = {}
    for row in records:
        stage = row["stage"]
        if stage not in fields:
            raise ValueError(f"unsupported objective stage: {stage}")
        field = fields[stage]
        value = row.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"missing or non-finite {field} for {stage}")
        stage_series = series.setdefault(stage, [])
        step = row.get("step", len(stage_series) + 1)
        if not isinstance(step, int) or step < 1:
            raise ValueError(f"invalid update step for {stage}: {step!r}")
        if stage_series and step <= stage_series[-1][0]:
            raise ValueError(f"updates are not strictly increasing for {stage}")
        stage_series.append((step, float(value), row))
    if not series:
        return
    stages = list(series)
    columns = 2
    rows_count = math.ceil(len(stages) / columns)
    width, panel_width, panel_height = 1240, 545, 340
    top_margin, row_gap = 104, 20
    phase3_note = None
    tail = series.get("phase3_human_pairs", [])
    if tail:
        row = tail[-1][2]
        if (
            float(row.get("multi_positive_loss", float("nan"))) == 0.0
            and len(row.get("query_ids", [])) == 1
            and row.get("positive_pairs") == 1
            and len(row.get("candidate_doc_ids", [])) == 1
        ):
            phase3_note = (
                f"Final phase-3 batch: {len(row['query_ids'])} query, "
                f"{row['positive_pairs']} positive pair, and "
                f"{len(row['candidate_doc_ids'])} candidate; zero loss is a "
                "single-example result, not retrieval-quality evidence."
            )
    footer_height = 58 if phase3_note else 22
    height = (
        top_margin
        + rows_count * panel_height
        + (rows_count - 1) * row_gap
        + footer_height
    )
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Training objectives by phase</title>',
        '<desc id="desc">Each observed phase has an independent update axis and a vertical axis starting at zero. Objective units differ by phase, so compare values within a panel only.</desc>',
        '<rect width="100%" height="100%" fill="#fbfbf9"/>',
        '<text x="54" y="38" font-size="23" font-family="sans-serif" font-weight="700" fill="#202626">Training objectives, separated by phase</text>',
        '<text x="54" y="64" font-size="13" font-family="sans-serif" fill="#596563">Each panel uses its own phase-local update axis and zero-based loss scale.</text>',
    ]
    for index, stage in enumerate(stages):
        x0 = 50 + (index % columns) * 595
        y0 = top_margin + (index // columns) * (panel_height + row_gap)
        panel = series[stage]
        values = [entry[1] for entry in panel]
        raw_max = max(values)
        y_max = max(0.1, math.ceil(raw_max * 1.12 * 10.0) / 10.0)
        left, right, top, bottom = x0 + 80, x0 + panel_width - 24, y0 + 72, y0 + 267
        count = len(panel)
        chunks.extend(
            [
                f'<rect x="{x0}" y="{y0}" width="{panel_width}" height="{panel_height}" rx="10" fill="#ffffff" stroke="#b8c8ca"/>',
                f'<text x="{x0 + 18}" y="{y0 + 29}" font-family="sans-serif" font-size="16" font-weight="700" fill="#182b30">{html.escape(titles[stage])}</text>',
                f'<text x="{x0 + 18}" y="{y0 + 51}" font-family="sans-serif" font-size="12" fill="#435b61">{html.escape(units[stage])} · {count} updates</text>',
            ]
        )
        for tick in range(5):
            fraction = tick / 4
            y = bottom - (bottom - top) * fraction
            value = y_max * fraction
            chunks.extend(
                [
                    f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#dbe4e4"/>',
                    f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#435b61">{value:.3g}</text>',
                ]
            )
        chunks.extend(
            [
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#526a70"/>',
                f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#526a70"/>',
            ]
        )
        points = []
        first_step, last_step = panel[0][0], panel[-1][0]
        for step, value, _ in panel:
            x = left + (right - left) * (step - first_step) / max(
                last_step - first_step, 1
            )
            y = bottom - (bottom - top) * value / y_max
            points.append((x, y, step, value))
        chunks.append(
            f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y, _, _ in points)}" fill="none" stroke="{colors[stage]}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y, step, value in points:
            chunks.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{colors[stage]}"><title>Update {step}: {value:.8g}</title></circle>'
            )
        step_ticks = sorted({first_step, panel[len(panel) // 2][0], last_step})
        for step in step_ticks:
            point = next(item for item in points if item[2] == step)
            chunks.append(
                f'<text x="{point[0]:.2f}" y="{bottom + 18}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#435b61">{step}</text>'
            )
        chunks.append(
            f'<text x="{(left + right) / 2:.2f}" y="{bottom + 43}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#263c41">Update within phase ({first_step}–{last_step})</text>'
        )
    if phase3_note:
        footer_y = height - 24
        chunks.append(
            f'<text x="54" y="{footer_y}" font-family="sans-serif" font-size="12" fill="#713c23">{html.escape(phase3_note)}</text>'
        )
    chunks.append("</svg>")
    path.write_text("\n".join(chunks) + "\n")


def retrieval_figure(path: Path, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(metrics)
    values = [
        (float(metrics[name]["recall@10"]), float(metrics[name]["ndcg@10"]))
        for name in labels
    ]
    maximum = max([1.0] + [v for pair in values for v in pair])
    chunks = _svg_start(
        "SciFact frozen-dev exact retrieval",
        "Recall@10 and nDCG@10; 188 judged queries from official dev, 300 inventoried.",
    )
    left, top, bottom = 275, 112, 440
    scale = 430 / maximum
    for tick in range(6):
        value = maximum * tick / 5
        x = left + scale * value
        chunks.extend(
            [
                f'<line x1="{x:.1f}" y1="{top-9}" x2="{x:.1f}" y2="{bottom}" stroke="#d5dcda"/>',
                f'<text x="{x:.1f}" y="{bottom+20}" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#596563">{value:.2f}</text>',
            ]
        )
    for index, (label, (recall, ndcg)) in enumerate(zip(labels, values)):
        y = top + index * 48
        chunks.append(
            f'<text x="{left-12}" y="{y+12}" text-anchor="end" font-size="11" font-family="sans-serif" fill="#202626">{html.escape(label)}</text>'
        )
        chunks.extend(
            [
                f'<rect x="{left}" y="{y-2}" width="{recall*scale:.1f}" height="15" fill="#286b8e"/>',
                f'<rect x="{left}" y="{y+16}" width="{ndcg*scale:.1f}" height="15" fill="#b46c2a"/>',
                f'<text x="{left+recall*scale+5:.1f}" y="{y+10}" font-size="10" font-family="sans-serif" fill="#202626">{recall:.3f}</text>',
                f'<text x="{left+ndcg*scale+5:.1f}" y="{y+28}" font-size="10" font-family="sans-serif" fill="#202626">{ndcg:.3f}</text>',
            ]
        )
    chunks.extend(
        [
            f'<line x1="{left}" y1="{bottom}" x2="{left+430}" y2="{bottom}" stroke="#596563"/>',
            f'<text x="{left+215}" y="{bottom+44}" font-size="13" text-anchor="middle" font-family="sans-serif" fill="#202626">metric value</text>',
            '<rect x="83" y="472" width="14" height="12" fill="#286b8e"/><text x="104" y="482" font-size="11" font-family="sans-serif">Recall@10</text><rect x="203" y="472" width="14" height="12" fill="#b46c2a"/><text x="224" y="482" font-size="11" font-family="sans-serif">nDCG@10</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(chunks) + "\n")
