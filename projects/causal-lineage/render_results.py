"""Plot the retained CPU run; never substitute illustrative training values."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    output = Path(__file__).resolve().parent / "outputs"
    result = json.loads((output / "pilot-results.json").read_text())
    figure, axes = plt.subplots(3, 1, figsize=(8, 13.5), layout="constrained")
    colors = ["#6240ad", "#087c83", "#bc532c"]
    for axis, stage, title, color in zip(
        axes[:2],
        ("pretraining", "adaptation"),
        ("Next-token pretraining", "Response-only continuation adaptation"),
        colors,
    ):
        history = result["history"][stage]
        axis.plot(
            [row["update"] for row in history],
            [row["loss"] for row in history],
            color=color,
        )
        axis.set(
            xlabel="Update within this stage",
            ylabel="Training minibatch CE (nats/token)",
            title=title,
        )
        axis.grid(alpha=0.2)
    evaluation = result["result"]["evaluation"]["test"]
    values = [
        evaluation[name]["perplexity"]
        for name in (
            "pretrained_transformer",
            "adapted_transformer",
            "unigram_baseline",
        )
    ]
    bars = axes[2].bar(["Pretrained", "Adapted", "Unigram"], values, color=colors)
    axes[2].bar_label(bars, fmt="%.1f", padding=3)
    axes[2].set(
        ylabel="Held-out next-token perplexity (lower is better)",
        title="Same 4,954 chapter-held-out targets",
        ylim=(0, max(values) * 1.18),
    )
    figure.suptitle("70,704 parameters · small CPU pilot · within one novel")
    for suffix in ("svg", "png"):
        figure.savefig(
            output / f"learning-and-evaluation.{suffix}",
            dpi=160,
            metadata={"Date": None} if suffix == "svg" else {},
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
