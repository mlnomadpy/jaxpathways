"""Export one already-evaluated pilot branch to a local inference package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import jax
import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE / "data") not in sys.path:
    sys.path.insert(0, str(HERE / "data"))

import model
import prepare
from config import ModelConfig
from run_pilot import _candidate_hash as _trained_candidate_hash, _hash
from training import load_checkpoint


def _candidate_hash(model_source: Path | None = None):
    """Hash the package decoder plus its fixed runtime dependencies."""
    digest = hashlib.sha256()
    sources = (
        ("config.py", HERE / "config.py"),
        ("model.py", model_source or HERE / "model.py"),
        ("training.py", HERE / "training.py"),
        ("run_pilot.py", HERE / "run_pilot.py"),
    )
    for name, path in sources:
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _encode(tree, arrays, name="params"):
    if tree is None or isinstance(tree, (bool, int, float, str)):
        return tree
    if isinstance(tree, dict):
        return {
            key: _encode(value, arrays, f"{name}/{key}")
            for key, value in sorted(tree.items())
        }
    if isinstance(tree, (tuple, list)):
        return {
            "tuple": [
                _encode(value, arrays, f"{name}/{index}")
                for index, value in enumerate(tree)
            ]
        }
    array = np.asarray(tree)
    key = f"a{len(arrays):04d}"
    arrays[key] = array
    return {"array": key, "shape": list(array.shape), "dtype": str(array.dtype)}


def _validate_evaluation_bindings(data, stage, branch, metrics):
    checkpoint_path = stage / "final-state.npz"
    identity_path = stage / "identity.json"
    if not checkpoint_path.is_file() or not identity_path.is_file():
        raise ValueError("selected evaluation checkpoint or identity is missing")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    branch_metrics = metrics["branches"][branch]
    recipe_sha = _hash(data / "frozen_recipe.json")
    tokenizer_sha = _hash(data / "tokenizer.json")
    evaluator_sha = _hash(HERE / "evaluate_pilot.py")
    scorer_sha = _hash(HERE / "squad_scoring.py")
    expected = {
        "recipe_sha256": recipe_sha,
        "tokenizer_sha256": tokenizer_sha,
        "evaluator_source_sha256": evaluator_sha,
        "scorer.reference_bytes_sha256": scorer_sha,
    }
    actual = {
        "recipe_sha256": metrics.get("recipe_sha256"),
        "tokenizer_sha256": metrics.get("tokenizer_sha256"),
        "evaluator_source_sha256": metrics.get("evaluator_source_sha256"),
        "scorer.reference_bytes_sha256": metrics.get("scorer", {}).get(
            "reference_bytes_sha256"
        ),
    }
    for identity_name, digest in expected.items():
        if actual[identity_name] != digest:
            raise ValueError(
                f"evaluation {identity_name} differs from the frozen package inputs"
            )
    checkpoint_sha = _hash(checkpoint_path)
    identity_sha = _hash(identity_path)
    if branch_metrics.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("evaluation checkpoint digest differs from selected branch")
    if branch_metrics.get("checkpoint_identity_sha256") != identity_sha:
        raise ValueError("evaluation checkpoint identity differs from selected branch")
    if branch_metrics.get("candidate_source_sha256") != identity.get(
        "candidate_source_sha256"
    ):
        raise ValueError("evaluation candidate identity differs from checkpoint")
    if (
        identity["recipe_sha256"] != recipe_sha
        or identity["processor_sha256"] != tokenizer_sha
    ):
        raise ValueError(
            "checkpoint recipe or tokenizer differs from frozen evaluation"
        )
    if identity["candidate_source_sha256"] != _trained_candidate_hash():
        raise ValueError("checkpoint candidate source differs from current decoder")
    return {
        "identity": identity,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_identity_sha256": identity_sha,
        "recipe_sha256": recipe_sha,
        "tokenizer_sha256": tokenizer_sha,
        "evaluator_source_sha256": evaluator_sha,
        "scorer_sha256": scorer_sha,
    }


def export(
    data_dir,
    run_dir,
    evaluation_dir,
    branch,
    output_dir,
    model_module=None,
    model_source=None,
):
    data, run, evaluation, output = map(
        lambda item: Path(item).resolve(),
        (data_dir, run_dir, evaluation_dir, output_dir),
    )
    prepare.verify_prepared(data, require_project_lock=True)
    metrics = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
    if branch not in metrics["branches"]:
        raise ValueError("branch is absent from the completed held-out evaluation")
    stage = run / branch
    binding = _validate_evaluation_bindings(data, stage, branch, metrics)
    identity = binding["identity"]
    model_module = model if model_module is None else model_module
    model_source = (
        HERE / "model.py" if model_source is None else Path(model_source).resolve()
    )
    if not model_source.is_file():
        raise ValueError("selected model implementation source is missing")
    inference_source_sha256 = _candidate_hash(model_source)
    checkpoint_path = stage / "final-state.npz"
    state = load_checkpoint(
        checkpoint_path,
        identity,
        expected_order=list(range(len(identity["schedule_sample_ids"]))),
        expected_seen_ids=identity["schedule_sample_ids"],
    )
    params = state["params"]
    if state["adapters"] is not None:
        params = model_module.merge_lora(
            params, state["adapters"], identity["model_config"]
        )
    output.mkdir(parents=True, exist_ok=False)
    tokenizer_path = output / "tokenizer.json"
    shutil.copyfile(data / "tokenizer.json", tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != identity["model_config"]["vocab_size"]:
        raise ValueError("tokenizer vocabulary and checkpoint architecture differ")
    config = ModelConfig(**identity["model_config"])
    (output / "config.json").write_text(
        json.dumps(config.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    arrays = {}
    parameter_spec = _encode(params, arrays)
    with (output / "parameters.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    (output / "parameters.json").write_text(
        json.dumps(parameter_spec, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (output / "RIGHTS.md").write_text(
        (HERE / "data" / "RIGHTS.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    runtime_files = (
        "consumer.py",
        "config.py",
        "model.py",
        "run_pilot.py",
        "serve_local.py",
        "training.py",
    )
    for name in runtime_files:
        shutil.copyfile(
            model_source if name == "model.py" else HERE / name, output / name
        )
    shutil.copyfile(HERE / "requirements-c02.txt", output / "requirements.txt")
    (output / "README.md").write_text(
        "# Local C02 pilot consumer\n\n"
        f"This package contains the evaluated `{branch}` bounded CPU pilot checkpoint. "
        "It is a small mechanics and task pilot, not the planned 300M capstone, a broad assistant, or a qualified release. "
        "The model was trained on the frozen WikiText-2/SQuAD recipe; see RIGHTS.md and the project guide for source limits.\n\n"
        "This folder is self-contained apart from the packages in requirements.txt. From this folder, use `from consumer import load_package`, then call `load_package('.').generate(prompt, max_new_tokens=32)`. "
        "The package verifies file hashes and the decoder source identity before inference. Requests are limited to 32 generated tokens and the declared context. "
        "To start the bounded loopback HTTP wrapper, run `python serve_local.py --package . --port 8127`; it binds only to 127.0.0.1, starts a 30-second request budget at admission, enforces that budget while parsing the prompt and between generation steps, admits at most two active requests, and rejects overload with HTTP 503. Cancellation is cooperative: an in-progress native JAX operation must return before the next cancellation check, so it can delay completion beyond the request budget.\n",
        encoding="utf-8",
    )
    files = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in (
            "README.md",
            "RIGHTS.md",
            "config.py",
            "config.json",
            "consumer.py",
            "model.py",
            "parameters.json",
            "parameters.npz",
            "requirements.txt",
            "run_pilot.py",
            "serve_local.py",
            "tokenizer.json",
            "training.py",
        )
    }
    manifest = {
        "format": "c02-custom-consumer-v1",
        "status": "bounded pilot consumer; not full capstone or release qualification",
        "branch": branch,
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "checkpoint_identity_sha256": binding["checkpoint_identity_sha256"],
        "candidate_source_sha256": inference_source_sha256,
        "checkpoint_candidate_source_sha256": identity["candidate_source_sha256"],
        "recipe_sha256": binding["recipe_sha256"],
        "tokenizer_sha256": binding["tokenizer_sha256"],
        "evaluator_source_sha256": binding["evaluator_source_sha256"],
        "scorer_source_sha256": binding["scorer_sha256"],
        "evaluation_metrics_sha256": _hash(evaluation / "metrics.json"),
        "full_parameters": model_module.parameter_count(config),
        "files": files,
        "parameter_arrays": sorted(arrays),
        "entrypoint": "consumer.load_package",
        "rights_notice": "RIGHTS.md",
        "max_new_tokens": 32,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "package": str(output),
                "manifest_sha256": _hash(output / "manifest.json"),
                "parameters": manifest["full_parameters"],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation", type=Path)
    args = parser.parse_args()
    candidate = model
    if args.implementation is not None:
        source = args.implementation.resolve()
        spec = importlib.util.spec_from_file_location("model", source)
        if spec is None or spec.loader is None:
            parser.error(f"cannot import selected model implementation: {source}")
        candidate = importlib.util.module_from_spec(spec)
        sys.modules["model"] = candidate
        spec.loader.exec_module(candidate)
    export(
        args.data,
        args.run,
        args.evaluation,
        args.branch,
        args.output,
        model_module=candidate,
        model_source=args.implementation,
    )


if __name__ == "__main__":
    main()
