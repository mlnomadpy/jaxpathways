"""Render the retained foundations pilot's three measured metric views."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPORT = HERE / "run/pilot-results.json"
DATA = HERE / "pilot-figure-data.json"
SVG = HERE / "pilot-metrics.svg"
PNG = HERE / "pilot-metrics.png"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = json.loads(REPORT.read_text())
    steps = report["training"]["steps"]
    dev_bytes = report["data"]["validation"]["normalized_paragraph_bytes"]
    baseline = report["metrics"]["train_only_byte_unigram_dev"]
    figure_data = {
        "source_report": "run/pilot-results.json",
        "source_report_sha256": sha256_file(REPORT),
        "units": {
            "training_step_loss": "nats per predicted BPE token, minibatch objective",
            "evaluation_nll": "nats per predicted BPE token, full split",
            "dev_byte_rate": "bits per normalized paragraph UTF-8 byte",
        },
        "training_steps": [
            {
                "step": int(item["step"]),
                "loss_nats_per_token": float(item["loss_nats_per_token"]),
            }
            for item in steps
        ],
        "evaluation_token_nll": {
            "train_before": float(
                report["training"]["text_token_nll_before"]["nll_nats_per_token"]
            ),
            "train_after": float(
                report["training"]["text_token_nll_after"]["nll_nats_per_token"]
            ),
            "dev_before": float(report["metrics"]["dev_before"]["nll_nats_per_token"]),
            "dev_after": float(report["metrics"]["dev_after"]["nll_nats_per_token"]),
            "test_final": float(report["metrics"]["final_test"]["nll_nats_per_token"]),
        },
        "development_bits_per_normalized_paragraph_byte": {
            "train_only_byte_unigram": float(
                baseline["bits_per_normalized_paragraph_byte"]
            ),
            "trained_bpe": float(
                report["metrics"]["dev_bits_per_normalized_paragraph_byte"]
            ),
            "normalized_paragraph_utf8_bytes": int(dev_bytes),
        },
        "scope": "Six-update FP32 CPU pilot on one public-domain novel; held-out chapters are fixed and this is not a blind benchmark.",
    }
    DATA.write_text(json.dumps(figure_data, indent=2, ensure_ascii=False) + "\n")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(14.2, 4.35), gridspec_kw={"width_ratios": [1.05, 1.15, 1]}
    )
    fig.suptitle(
        "Text-harness foundations: measured small pilot",
        fontsize=14,
        fontweight="bold",
        x=0.04,
        ha="left",
    )

    step_ids = [item["step"] for item in figure_data["training_steps"]]
    step_losses = [
        item["loss_nats_per_token"] for item in figure_data["training_steps"]
    ]
    axes[0].plot(step_ids, step_losses, color="#267C78", marker="o", linewidth=2)
    axes[0].set_title("Training minibatch objective")
    axes[0].set_xlabel("Update")
    axes[0].set_ylabel("Nats per predicted BPE token")
    axes[0].set_xticks(step_ids)
    axes[0].grid(axis="y", alpha=0.25)

    evaluation = figure_data["evaluation_token_nll"]
    names = [
        "Train\nbefore",
        "Train\nafter",
        "Dev\nbefore",
        "Dev\nafter",
        "Test\nfinal",
    ]
    keys = ["train_before", "train_after", "dev_before", "dev_after", "test_final"]
    evaluation_values = [evaluation[key] for key in keys]
    axes[1].scatter(
        range(len(keys)),
        evaluation_values,
        c=["#9ABDB8", "#267C78", "#9ABDB8", "#267C78", "#D18A38"],
        s=62,
        zorder=3,
    )
    axes[1].set_title("Full-split token NLL")
    axes[1].set_ylabel("Nats per predicted BPE token")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(range(len(keys)), names)
    axes[1].set_ylim(min(evaluation_values) - 0.012, max(evaluation_values) + 0.012)
    for index, value in enumerate(evaluation_values):
        axes[1].annotate(
            f"{value:.3f}",
            (index, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    rates = figure_data["development_bits_per_normalized_paragraph_byte"]
    rate_bars = axes[2].bar(
        ["Byte unigram\n(train-only)", "BPE model\n(after 6 updates)"],
        [rates["train_only_byte_unigram"], rates["trained_bpe"]],
        color=["#8C9AA6", "#267C78"],
    )
    axes[2].set_title("Development byte rate")
    axes[2].set_ylabel("Bits per normalized paragraph UTF-8 byte")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].bar_label(rate_bars, fmt="%.3f", padding=3, fontsize=8)
    axes[2].text(
        0.5,
        0.02,
        f"Same {rates['normalized_paragraph_utf8_bytes']:,} normalized bytes",
        transform=axes[2].transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
    )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.04,
        0.01,
        "One-book CPU pilot; six updates. Test chapters were scored at the end of each run and reused during engineering validation.",
        fontsize=8,
        color="#4D5963",
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.99, 0.91), w_pad=1.8)
    fig.savefig(SVG, bbox_inches="tight")
    fig.savefig(PNG, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
