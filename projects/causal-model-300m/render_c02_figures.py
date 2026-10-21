"""Render plots from frozen C02 run/evaluation JSON and bind their input hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BRANCHES = (
    "base",
    "continuation_mixture",
    "continuation_general_control",
    "sft_full_from_domain",
    "sft_lora_from_domain",
    "sft_full_from_general",
)
LABELS = {
    "base": "Base",
    "continuation_mixture": "80/20 continuation",
    "continuation_general_control": "General control",
    "sft_full_from_domain": "Full SFT, domain parent",
    "sft_lora_from_domain": "Q/V LoRA, domain parent",
    "sft_full_from_general": "Full SFT, general parent",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(
    run_dir: Path,
    evaluation_dir: Path,
    pythia_path: Path,
    pythia_baseline_path: Path,
    diagnostics_path: Path,
    output_dir: Path,
):
    run_dir, evaluation_dir, output_dir = map(
        Path.resolve, (run_dir, evaluation_dir, output_dir)
    )
    pythia_path, pythia_baseline_path, diagnostics_path = (
        Path(pythia_path).resolve(),
        Path(pythia_baseline_path).resolve(),
        Path(diagnostics_path).resolve(),
    )
    metrics_path = evaluation_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    diagnostics = json.loads(diagnostics_path.read_text())
    pythia = json.loads(pythia_path.read_text())
    pythia_baseline = json.loads(pythia_baseline_path.read_text())
    branch_traces = {
        branch: json.loads((run_dir / branch / "metrics.json").read_text())["updates"]
        for branch in BRANCHES
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titleweight": "semibold"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(BRANCHES)))
    for branch, color in zip(BRANCHES, colors, strict=True):
        points = branch_traces[branch]
        axes[0].plot(
            [row["update"] for row in points],
            [row["loss"] for row in points],
            marker=".",
            linewidth=1.35,
            markersize=4,
            color=color,
            label=LABELS[branch],
        )
    axes[0].set_title("Training minibatches")
    axes[0].set_xlabel("Update within phase")
    axes[0].set_ylabel("Token NLL (nats / valid target)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="best")

    qa_labels = [LABELS[branch] for branch in BRANCHES] + ["Pythia-14M"]
    qa_em = [
        metrics["branches"][branch]["custom_squad_qa"]["exact_match"]
        for branch in BRANCHES
    ]
    qa_f1 = [
        metrics["branches"][branch]["custom_squad_qa"]["token_f1"]
        for branch in BRANCHES
    ]
    qa_em.append(pythia["exact_match"])
    qa_f1.append(pythia["token_f1"])
    x = np.arange(len(qa_labels))
    width = 0.36
    axes[1].bar(x - width / 2, qa_em, width, label="Exact match", color="#3366aa")
    axes[1].bar(x + width / 2, qa_f1, width, label="Token F1", color="#dc8d24")
    axes[1].axhline(
        0.05, color="#3366aa", linestyle="--", linewidth=1, label="EM floor 0.05"
    )
    axes[1].axhline(
        0.10, color="#dc8d24", linestyle=":", linewidth=1, label="F1 floor 0.10"
    )
    axes[1].set_ylim(0, 0.12)
    axes[1].set_xticks(x, qa_labels, rotation=30, ha="right")
    axes[1].set_ylabel("Score on frozen 128-question custom set")
    axes[1].set_title("Generated-answer QA scores")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=7, loc="upper left")
    qa_path = output_dir / "c02-training-and-qa.png"
    fig.savefig(qa_path, dpi=180)
    fig.savefig(qa_path.with_suffix(".svg"))
    plt.close(fig)

    splits = (
        ("WikiText held-out", "general_wikitext_nll", "general_wikitext"),
        ("SQuAD final contexts", "domain_squad_context_nll", "squad_final_contexts"),
    )
    byte_labels = [LABELS[branch] for branch in BRANCHES] + [
        "Pythia-14M",
        "Byte unigram",
    ]
    x = np.arange(len(byte_labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 4.6), constrained_layout=True)
    for index, (split_label, metrics_key, baseline_key) in enumerate(splits):
        values = [
            metrics["branches"][branch][metrics_key][
                "nll_bits_per_normalized_utf8_byte"
            ]
            for branch in BRANCHES
        ]
        values.append(
            pythia_baseline[
                (
                    "general_wikitext_heldout"
                    if index == 0
                    else "domain_squad_final_contexts"
                )
            ]["nll_bits_per_normalized_utf8_byte"]
        )
        values.append(
            diagnostics["simple_byte_unigram"][baseline_key][
                "nll_bits_per_normalized_utf8_byte"
            ]
        )
        offset = (-0.5 if index == 0 else 0.5) * width
        ax.bar(x + offset, values, width, label=split_label)
    ax.set_xticks(x, byte_labels, rotation=28, ha="right")
    ax.set_ylabel("NLL (bits / normalized UTF-8 byte)")
    ax.set_title("Likelihood rate on the same normalized held-out bytes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    byte_path = output_dir / "c02-byte-rates.png"
    fig.savefig(byte_path, dpi=180)
    fig.savefig(byte_path.with_suffix(".svg"))
    plt.close(fig)

    sources = {
        "pilot_metrics": metrics_path,
        "pythia_qa": pythia_path,
        "pythia_nll": pythia_baseline_path,
        "supplemental_diagnostics": diagnostics_path,
        "render_script": Path(__file__),
        **{
            f"{branch}_training_trace": run_dir / branch / "metrics.json"
            for branch in BRANCHES
        },
    }
    index = {
        "status": "plots of the retained bounded CPU run; no additional fit or score selection",
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
        "plots": {
            path.name: sha256(path)
            for base in (qa_path, byte_path)
            for path in (base, base.with_suffix(".svg"))
        },
        "interpretation": {
            "training_and_qa": "per-update dropout minibatches and fixed-set generated answer metrics use different denominators and are shown in separate panels; dashed lines are precommitted QA floors",
            "byte_rates": "each model is scored on the same normalized UTF-8 byte denominator within each split; token NLLs across different tokenizers are not plotted together",
        },
    }
    index_path = output_dir / "figure-index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(
        json.dumps({"index": str(index_path), "sha256": sha256(index_path)}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--pythia-qa", type=Path, required=True)
    parser.add_argument("--pythia-baseline", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(
        args.run,
        args.evaluation,
        args.pythia_qa,
        args.pythia_baseline,
        args.diagnostics,
        args.output,
    )


if __name__ == "__main__":
    main()
