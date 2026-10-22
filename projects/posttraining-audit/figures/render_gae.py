"""Render the illustrative GAE figure independently of the learner implementation.

Install projects/posttraining-audit/requirements.txt, then run this file from any working directory.
Use --check to verify the hand calculation without importing plotting libraries.
"""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def calculate(fixture):
    """Scalar finite-horizon calculation, including separate bootstrap/trace stops."""
    gamma, lam = fixture["gamma"], fixture["lambda"]
    results = []
    for sequence in fixture["sequences"]:
        rewards = sequence["rewards"]
        values = sequence["values"]
        next_values = sequence["nextValues"]
        assert len(rewards) == len(values) == len(next_values)
        assert sequence["stop"] in ("eos", "timeout")
        deltas, advantages = [], [0.0] * len(rewards)
        for index, reward in enumerate(rewards):
            terminal = index == len(rewards) - 1 and sequence["stop"] == "eos"
            bootstrap = 0.0 if terminal else next_values[index]
            deltas.append(reward + gamma * bootstrap - values[index])
        carry = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            # Both observed stops end this trace. Timeout still used V above.
            trace = index < len(rewards) - 1
            carry = deltas[index] + gamma * lam * trace * carry
            advantages[index] = carry
        bootstrap = 0.0 if sequence["stop"] == "eos" else next_values[-1]
        for actual, expected in zip(deltas, sequence["expectedDeltas"], strict=True):
            assert abs(actual - expected) < 1e-12
        for actual, expected in zip(
            advantages, sequence["expectedAdvantages"], strict=True
        ):
            assert abs(actual - expected) < 1e-12
        assert bootstrap == sequence["expectedBootstrap"]
        results.append(
            {"deltas": deltas, "advantages": advantages, "bootstrap": bootstrap}
        )
    return results


def render(fixture, results, source_hash):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {"font.size": 12, "svg.fonttype": "none", "svg.hashsalt": source_hash}
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.4), sharey=True)
    colors = ("#c9692c", "#287e83", "#6841a6")
    width = 0.24
    for ax, sequence, result in zip(axes, fixture["sequences"], results, strict=True):
        positions = np.arange(1, len(sequence["rewards"]) + 1)
        groups = (sequence["rewards"], sequence["values"], result["advantages"])
        for offset, values, color, label in zip(
            (-width, 0, width),
            groups,
            colors,
            ("Observed reward", "Old value estimate", "GAE advantage"),
            strict=True,
        ):
            bars = ax.bar(
                positions + offset, values, width, color=color, label=label, zorder=3
            )
            ax.bar_label(
                bars,
                labels=[f"{value:.3g}" for value in values],
                padding=4,
                fontsize=10,
            )
        final_next = sequence["nextValues"][-1]
        rule = (
            f"EOS: ignore next value {final_next:g}; bootstrap = 0"
            if sequence["stop"] == "eos"
            else f"Timeout: include next value {final_next:g}; bootstrap = {final_next:g}"
        )
        ax.set_title(sequence["name"], fontsize=15, fontweight="bold", pad=56)
        ax.text(
            0.5,
            1.035,
            rule + "\nBoth stops end the advantage trace.",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.5,
        )
        labels = [str(position) for position in positions]
        labels[-1] += "\n" + ("EOS" if sequence["stop"] == "eos" else "time limit")
        ax.set_xticks(positions, labels)
        ax.set_xlabel("Response action position")
        ax.set_ylim(0, 2.45)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Reward, value and advantage (reward units)")
    figure.suptitle(
        "Illustrative GAE: termination and truncation",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.915,
        "Discount gamma = 0.9; trace lambda = 0.8. No KL shaping. Not a training result.",
        ha="center",
        fontsize=12,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
    )
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.69, wspace=0.15)
    figure.savefig(HERE / "gae-eos-timeout.svg", metadata={"Date": None})
    figure.savefig(HERE / "gae-eos-timeout.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Verify numeric fixture without rendering"
    )
    args = parser.parse_args()
    source = (HERE / "gae-eos-timeout-fixture.json").read_bytes()
    fixture = json.loads(source)
    results = calculate(fixture)
    if not args.check:
        render(fixture, results, hashlib.sha256(source).hexdigest())
    print(
        json.dumps(
            {"fixtureSha256": hashlib.sha256(source).hexdigest(), "results": results},
            indent=2,
        )
    )
