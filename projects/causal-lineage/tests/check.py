"""Independently check causal masking/loss, then run the staged pilot."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from network import (
    ROOT,
    SPECIAL,
    canonical_hash,
    load_data,
    make_model,
    make_tokenizer,
)
from run import load_candidate, load_checkpoint, source_identity

CONFIG = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))


def check_candidate(candidate) -> None:
    tokens = jnp.asarray([[SPECIAL["<bos>"], 7, 8, SPECIAL["<pad>"]]])
    mask = np.asarray(candidate.causal_attention_mask(tokens, SPECIAL["<pad>"]))
    expected = np.asarray(
        [
            [
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, True, False],
                    [True, True, True, False],
                ]
            ]
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected)

    sequences = jnp.asarray(
        [[SPECIAL["<bos>"], 7, 8, SPECIAL["<eos>"], SPECIAL["<pad>"]]]
    )
    inputs, targets, target_mask = candidate.shift_tokens(
        sequences, jnp.asarray([1]), SPECIAL["<pad>"]
    )
    np.testing.assert_array_equal(
        np.asarray(inputs), [[SPECIAL["<bos>"], 7, 8, SPECIAL["<eos>"]]]
    )
    np.testing.assert_array_equal(
        np.asarray(targets), [[7, 8, SPECIAL["<eos>"], SPECIAL["<pad>"]]]
    )
    np.testing.assert_array_equal(np.asarray(target_mask), [[True, True, True, False]])

    rng = np.random.default_rng(90210)
    logits = rng.normal(size=(2, 3, 5)).astype(np.float32)
    labels = np.asarray([[1, 3, 0], [4, 2, 1]], dtype=np.int32)
    selected = np.asarray([[True, False, True], [False, True, True]])
    actual = float(
        candidate.masked_causal_cross_entropy(
            jnp.asarray(logits), jnp.asarray(labels), jnp.asarray(selected)
        )
    )
    log_probs = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    expected_loss = -log_probs[np.arange(2)[:, None], np.arange(3), labels][
        selected
    ].mean()
    np.testing.assert_allclose(actual, expected_loss, rtol=1e-6, atol=1e-6)
    try:
        candidate.masked_causal_cross_entropy(
            jnp.asarray(logits), jnp.asarray(labels), jnp.zeros_like(labels, dtype=bool)
        )
    except ValueError:
        pass
    else:
        raise AssertionError("an empty next-token target mask was accepted")

    sample_logits = jnp.asarray([0.0, -2.0, 3.0, 1.0])
    sampled = int(
        candidate.sample_next_token(sample_logits, jax.random.key(5), top_k=1)
    )
    if sampled != 2:
        raise AssertionError("top_k=1 did not select the highest-logit token")

    passages, instructions, _, _ = load_data()
    tokenizer = make_tokenizer(passages, instructions, CONFIG["max_vocabulary"])
    model = make_model(CONFIG, tokenizer)
    sample = jnp.asarray([[SPECIAL["<bos>"], 7, 8, 9, 10, 11]], dtype=jnp.int32)
    params = model.init(
        jax.random.key(4),
        sample,
        candidate.causal_attention_mask(sample, SPECIAL["<pad>"]),
    )["params"]
    original_logits = model.apply(
        {"params": params}, sample, candidate.causal_attention_mask(sample)
    )
    changed = sample.at[0, 4].set(13)
    changed_logits = model.apply(
        {"params": params}, changed, candidate.causal_attention_mask(changed)
    )
    np.testing.assert_array_equal(
        np.asarray(original_logits[:, :4]), np.asarray(changed_logits[:, :4])
    )


def check_swapped_parent(
    output: Path, data_sha: str, tokenizer: dict, candidate_path: Path
) -> None:
    candidate, candidate_path = load_candidate(candidate_path)
    source_code, _ = source_identity(candidate_path)
    stage2 = output / "checkpoints/stage-2"
    with TemporaryDirectory(prefix="causal-lineage-invalid-parent-") as temp:
        altered = Path(temp) / "stage-2"
        shutil.copytree(stage2, altered)
        manifest_path = altered / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["parent_checkpoint_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        try:
            load_checkpoint(
                altered,
                stage=2,
                config=CONFIG,
                tokenizer=tokenizer,
                data_sha=data_sha,
                source_code=source_code,
                expected_parent=json.loads(
                    (output / "checkpoints/stage-1/manifest.json").read_text()
                )["checkpoint_sha256"],
                mask_fn=candidate.causal_attention_mask,
            )
        except ValueError as error:
            if "parent" not in str(error):
                raise
        else:
            raise AssertionError("swapped predecessor checkpoint was accepted")


def reference_run(implementation: str, stage: str, output_argument: str | None) -> None:
    implementation_path = (
        PROJECT_ROOT / f"{implementation}/model.py"
        if implementation in ("starter", "solution")
        else Path(implementation).resolve()
    )
    candidate, _ = load_candidate(implementation_path)
    check_candidate(candidate)
    with TemporaryDirectory(prefix="causal-lineage-reference-") as temp:
        output = (
            Path(output_argument).resolve() if output_argument else (Path(temp) / "run")
        )
        command = [
            sys.executable,
            str(PROJECT_ROOT / "run.py"),
            "--implementation",
            str(implementation_path),
            "--output",
            str(output),
            "--stage",
            stage,
        ]
        environment = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=True,
            timeout=240,
        )
        if stage == "1":
            stage1 = json.loads(
                (output / "checkpoints/stage-1/manifest.json").read_text()
            )
            validation = json.loads((output / "stage-1.json").read_text())
            if stage1["parent_checkpoint_sha256"] is not None:
                raise AssertionError("pretraining checkpoint unexpectedly has a parent")
            if validation["validation_next_token"]["tokens"] <= 0:
                raise AssertionError("validation pretraining metric had no targets")
            print(
                "PASS causal-lineage stage 1: causal checks, next-token pretraining, "
                "and held-out validation receipt"
            )
            return
        if stage == "2":
            if not (output / "checkpoints/stage-1/manifest.json").is_file():
                raise AssertionError(
                    "stage 2 did not preserve its required predecessor"
                )
        evaluation = json.loads((output / "evaluation.json").read_text())
        result = json.loads((output / "result.json").read_text())
        stage1 = json.loads((output / "checkpoints/stage-1/manifest.json").read_text())
        stage2 = json.loads((output / "checkpoints/stage-2/manifest.json").read_text())
        if stage1["parent_checkpoint_sha256"] is not None:
            raise AssertionError("pretraining checkpoint unexpectedly has a parent")
        if stage2["parent_checkpoint_sha256"] != stage1["checkpoint_sha256"]:
            raise AssertionError(
                "adaptation did not inherit the actual pretraining checkpoint"
            )
        if (
            stage2["objective"]
            != "instruction-style next-paragraph continuation adaptation"
        ):
            raise AssertionError("adaptation objective is not explicitly scoped")
        if (
            evaluation["adaptation"]["preference_optimization"]
            or evaluation["adaptation"]["rlhf"]
        ):
            raise AssertionError(
                "this project must not claim preference optimization or RLHF"
            )
        if evaluation["test"]["pretrained_transformer"]["tokens"] <= 0:
            raise AssertionError("held-out language-model evaluation had no targets")
        if (
            evaluation["test"]["unigram_baseline"]["tokens"]
            != evaluation["test"]["pretrained_transformer"]["tokens"]
        ):
            raise AssertionError(
                "unigram baseline and model were scored on different targets"
            )
        if not result["export"]["generated_ids"]:
            raise AssertionError(
                "exported model did not produce sampled inference tokens"
            )
        passages, instructions, _, data_sha = load_data()
        tokenizer = make_tokenizer(passages, instructions, CONFIG["max_vocabulary"])
        check_swapped_parent(output, data_sha, tokenizer, implementation_path)
        print(
            "PASS causal-lineage all: independent causal checks, real-data stages, "
            "checkpoint predecessor rejection, held-out unigram comparison, and fresh-process export sampling"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", default="starter")
    parser.add_argument("--stage", choices=("1", "2", "all"), required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    reference_run(args.implementation, args.stage, args.output)


if __name__ == "__main__":
    main()
