"""Train one small causal-LM stage with source-bound checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from network import (
    ROOT,
    SPECIAL,
    canonical_hash,
    encode_text,
    init_params,
    load_data,
    load_params,
    make_model,
    make_tokenizer,
    parameter_count,
    sha256_bytes,
    token_words,
    tree_bytes,
)

CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def load_candidate(path: str | Path):
    candidate_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(
        "causal_lineage_candidate", candidate_path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import implementation file: {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, candidate_path


def source_identity(candidate_path: Path) -> tuple[dict, str]:
    tracked = [
        ROOT / "network.py",
        ROOT / "run.py",
        ROOT / "probe_export.py",
        ROOT / "data/build_data.py",
        candidate_path,
    ]
    files = {}
    for path in tracked:
        try:
            name = path.resolve().relative_to(ROOT).as_posix()
        except ValueError as error:
            raise ValueError(
                "implementation must live inside the project folder"
            ) from error
        files[name] = sha256_bytes(path.read_bytes())
    return {
        "files": files,
        "sha256": canonical_hash(files),
    }, candidate_path.relative_to(ROOT).as_posix()


def document_sequences(passages: list[dict], tokenizer: dict, max_length: int):
    sequences = []
    for passage in passages:
        ids = (
            [SPECIAL["<bos>"]]
            + encode_text(passage["text"], tokenizer)
            + [SPECIAL["<eos>"]]
        )
        for start in range(0, len(ids) - 1, max_length):
            chunk = ids[start : start + max_length + 1]
            if len(chunk) < 2:
                continue
            chunk += [SPECIAL["<pad>"]] * (max_length + 1 - len(chunk))
            sequences.append((chunk, 1))
    return sequences


def instruction_sequences(instructions: list[dict], tokenizer: dict, config: dict):
    max_length = config["max_length"]
    output = []
    for record in instructions:
        prompt_ids = encode_text(record["instruction"] + " Context:", tokenizer)
        context_ids = encode_text(record["context"], tokenizer)[
            -config["instruction_context_tokens"] :
        ]
        response_ids = encode_text(record["response"], tokenizer)[
            : config["instruction_response_tokens"]
        ]
        prefix = [SPECIAL["<bos>"]] + prompt_ids + context_ids + [SPECIAL["<sep>"]]
        response_start = len(prefix)
        sequence = prefix + response_ids + [SPECIAL["<eos>"]]
        if len(sequence) > max_length + 1:
            overflow = len(sequence) - (max_length + 1)
            context_ids = context_ids[min(overflow, len(context_ids)) :]
            prefix = [SPECIAL["<bos>"]] + prompt_ids + context_ids + [SPECIAL["<sep>"]]
            response_start = len(prefix)
            sequence = prefix + response_ids + [SPECIAL["<eos>"]]
        if len(sequence) > max_length + 1:
            raise ValueError("instruction prompt exceeds the fixed context window")
        sequence += [SPECIAL["<pad>"]] * (max_length + 1 - len(sequence))
        output.append((sequence, response_start))
    return output


def arrays_for(rows: list[tuple[list[int], int]], candidate):
    sequences = jnp.asarray(np.asarray([row[0] for row in rows], dtype=np.int32))
    starts = jnp.asarray(np.asarray([row[1] for row in rows], dtype=np.int32))
    inputs, targets, target_mask = candidate.shift_tokens(
        sequences, starts, SPECIAL["<pad>"]
    )
    if not bool(jnp.all(jnp.sum(target_mask, axis=1) > 0)):
        raise ValueError("every training/evaluation row must have a next-token target")
    return inputs, targets, target_mask


def model_hash(config: dict, tokenizer: dict) -> str:
    architecture = {
        "name": "one-block-pre-layernorm-causal-transformer-v1",
        "vocab_size": tokenizer["vocabulary_size"],
        "max_length": config["max_length"],
        "width": config["width"],
        "layers": config["layers"],
        "heads": config["attention_heads"],
        "mlp_width": config["mlp_width"],
        "causal_mask_contract": "query position may attend to non-PAD keys at or before itself",
    }
    return canonical_hash(architecture)


def objective_loss(model, params, inputs, targets, mask, candidate):
    attention = candidate.causal_attention_mask(inputs, SPECIAL["<pad>"])
    logits = model.apply({"params": params}, inputs, attention)
    return candidate.masked_causal_cross_entropy(logits, targets, mask)


def evaluate(model, params, rows, candidate, batch_size):
    inputs, targets, target_mask = arrays_for(rows, candidate)

    @jax.jit
    def evaluate_batch(batch_inputs, batch_targets, batch_mask):
        loss = objective_loss(
            model, params, batch_inputs, batch_targets, batch_mask, candidate
        )
        count = jnp.sum(batch_mask)
        return loss * count, count

    total_nll = 0.0
    token_count = 0
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        nll_sum, count = evaluate_batch(
            inputs[start:stop], targets[start:stop], target_mask[start:stop]
        )
        total_nll += float(nll_sum)
        token_count += int(count)
    mean_nll = total_nll / token_count
    return {
        "nll": mean_nll,
        "perplexity": float(np.exp(mean_nll)),
        "tokens": token_count,
    }


def train_stage(model, params, rows, candidate, *, learning_rate, updates, seed):
    inputs, targets, target_mask = arrays_for(rows, candidate)
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def update(
        current_params, current_opt_state, batch_inputs, batch_targets, batch_mask
    ):
        def loss_fn(p):
            return objective_loss(
                model, p, batch_inputs, batch_targets, batch_mask, candidate
            )

        loss, gradients = jax.value_and_grad(loss_fn)(current_params)
        updates_tree, new_opt_state = optimizer.update(
            gradients, current_opt_state, current_params
        )
        new_params = optax.apply_updates(current_params, updates_tree)
        leaves = jax.tree.leaves(gradients)
        grad_norm = jnp.sqrt(
            sum(jnp.sum(leaf.astype(jnp.float32) ** 2) for leaf in leaves)
        )
        finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm)
        finite = finite & jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves])
        )
        return new_params, new_opt_state, loss, grad_norm, finite

    rng = np.random.default_rng(seed)
    history = []
    for step in range(1, updates + 1):
        indices = rng.integers(0, len(rows), size=CONFIG["batch_size"])
        params, optimizer_state, loss, grad_norm, finite = update(
            params,
            optimizer_state,
            inputs[indices],
            targets[indices],
            target_mask[indices],
        )
        if not bool(finite):
            raise FloatingPointError(f"nonfinite loss or gradient at update {step}")
        history.append(
            {
                "update": step,
                "loss": float(loss),
                "gradient_norm": float(grad_norm),
                "learning_rate": learning_rate,
                "target_tokens": int(jnp.sum(target_mask[indices])),
            }
        )
    return params, history


def unigram_baseline(train_rows, evaluation_rows, tokenizer, alpha):
    vocabulary_size = tokenizer["vocabulary_size"]
    counts = np.zeros(vocabulary_size, dtype=np.float64)
    for sequence, _ in train_rows:
        values = np.asarray(sequence[1:], dtype=np.int64)
        values = values[values != SPECIAL["<pad>"]]
        counts += np.bincount(values, minlength=vocabulary_size)
    probabilities = (counts + alpha) / (counts.sum() + alpha * vocabulary_size)
    total_nll = 0.0
    total_tokens = 0
    for sequence, target_start in evaluation_rows:
        targets = np.asarray(sequence[1:], dtype=np.int64)
        positions = np.arange(1, len(sequence))
        selected = (targets != SPECIAL["<pad>"]) & (positions >= target_start)
        total_nll -= np.log(probabilities[targets[selected]]).sum()
        total_tokens += int(selected.sum())
    mean_nll = float(total_nll / total_tokens)
    return {
        "nll": mean_nll,
        "perplexity": float(np.exp(mean_nll)),
        "tokens": total_tokens,
    }


def save_checkpoint(
    directory: Path,
    params,
    *,
    stage: int,
    objective: str,
    config: dict,
    tokenizer: dict,
    data_sha: str,
    source_code: dict,
    parent_checkpoint_sha: str | None,
    history: list[dict],
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    parameter_bytes = tree_bytes(params)
    manifest = {
        "schemaVersion": 1,
        "projectId": "causal-lineage",
        "stage": stage,
        "objective": objective,
        "config_sha256": canonical_hash(config),
        "tokenizer_sha256": canonical_hash(tokenizer),
        "model_sha256": model_hash(config, tokenizer),
        "data_sha256": data_sha,
        "source_code_identity": source_code,
        "parent_checkpoint_sha256": parent_checkpoint_sha,
        "optimizer_state": "fresh Adam state per stage; no mid-stage resume is claimed",
        "updates": len(history),
        "training_history": history,
        "parameter_count": parameter_count(params),
        "params_sha256": sha256_bytes(parameter_bytes),
    }
    manifest["checkpoint_sha256"] = hashlib.sha256(
        parameter_bytes + canonical_hash(manifest).encode("ascii")
    ).hexdigest()
    (directory / "params.msgpack").write_bytes(parameter_bytes)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_checkpoint(
    directory: Path,
    *,
    stage: int,
    config: dict,
    tokenizer: dict,
    data_sha: str,
    source_code: dict,
    expected_parent: str | None,
    mask_fn,
):
    params_path = directory / "params.msgpack"
    manifest_path = directory / "manifest.json"
    if not params_path.is_file() or not manifest_path.is_file():
        raise ValueError(
            f"checkpoint stage {stage} is missing its parameter or manifest file"
        )
    params_bytes = params_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "stage": (manifest.get("stage"), stage),
        "config": (manifest.get("config_sha256"), canonical_hash(config)),
        "tokenizer": (manifest.get("tokenizer_sha256"), canonical_hash(tokenizer)),
        "model": (manifest.get("model_sha256"), model_hash(config, tokenizer)),
        "data": (manifest.get("data_sha256"), data_sha),
        "source": (manifest.get("source_code_identity"), source_code),
        "parent": (manifest.get("parent_checkpoint_sha256"), expected_parent),
        "parameters": (manifest.get("params_sha256"), sha256_bytes(params_bytes)),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"checkpoint {label} identity mismatch")
    payload = {
        key: value for key, value in manifest.items() if key != "checkpoint_sha256"
    }
    expected_checkpoint = hashlib.sha256(
        params_bytes + canonical_hash(payload).encode("ascii")
    ).hexdigest()
    if expected_checkpoint != manifest.get("checkpoint_sha256"):
        raise ValueError("checkpoint manifest checksum mismatch")
    model = make_model(config, tokenizer)
    template = init_params(model, config["seed"], config["max_length"], mask_fn)
    return load_params(template, params_bytes), manifest


def validate_rejects_swapped_parent(output: Path, manifest, params_bytes, args):
    with TemporaryDirectory(prefix="causal-lineage-stale-parent-") as temp:
        directory = Path(temp) / "stage-2"
        directory.mkdir()
        changed = dict(manifest)
        changed["parent_checkpoint_sha256"] = "0" * 64
        (directory / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")
        (directory / "params.msgpack").write_bytes(params_bytes)
        try:
            load_checkpoint(
                directory,
                stage=2,
                config=args["config"],
                tokenizer=args["tokenizer"],
                data_sha=args["data_sha"],
                source_code=args["source_code"],
                expected_parent=args["expected_parent"],
                mask_fn=args["candidate"].causal_attention_mask,
            )
        except ValueError as error:
            if "parent" not in str(error):
                raise
        else:
            raise AssertionError("a checkpoint with a swapped parent was accepted")


def tokenize_prompt(text: str, tokenizer: dict, max_length: int) -> np.ndarray:
    ids = [SPECIAL["<bos>"]] + encode_text(text, tokenizer)
    if len(ids) > max_length:
        ids = ids[-max_length:]
    result = np.full((max_length,), SPECIAL["<pad>"], dtype=np.int32)
    result[: len(ids)] = ids
    return result


def export_and_probe(
    output: Path,
    params,
    *,
    config: dict,
    tokenizer: dict,
    data_sha: str,
    source_code: dict,
    candidate_rel: str,
    model,
    candidate,
) -> dict:
    export = output / "export"
    export.mkdir(parents=True, exist_ok=True)
    params_bytes = tree_bytes(params)
    (export / "params.msgpack").write_bytes(params_bytes)
    (export / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (export / "tokenizer.json").write_text(
        json.dumps(tokenizer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prompt = "Alice wondered what might happen next."
    prompt_tokens = tokenize_prompt(prompt, tokenizer, config["max_length"])
    valid_length = int(np.sum(prompt_tokens != SPECIAL["<pad>"]))
    inputs = jnp.asarray(prompt_tokens[None, :])
    attention = candidate.causal_attention_mask(inputs, SPECIAL["<pad>"])
    logits = model.apply({"params": params}, inputs, attention)
    prompt_logits = np.asarray(logits[0, valid_length - 1], dtype="<f4")
    expected_logits_sha = sha256_bytes(prompt_logits.tobytes())
    file_hashes = {
        name: sha256_bytes((export / name).read_bytes())
        for name in ("params.msgpack", "config.json", "tokenizer.json")
    }
    stage2_manifest = json.loads(
        (output / "checkpoints/stage-2/manifest.json").read_text(encoding="utf-8")
    )
    export_manifest = {
        "schemaVersion": 1,
        "projectId": "causal-lineage",
        "source_checkpoint_sha256": stage2_manifest["checkpoint_sha256"],
        "checkpoint_sha256": stage2_manifest["checkpoint_sha256"],
        "params_sha256": sha256_bytes(params_bytes),
        "config_sha256": canonical_hash(config),
        "tokenizer_sha256": canonical_hash(tokenizer),
        "model_sha256": model_hash(config, tokenizer),
        "data_sha256": data_sha,
        "source_code_identity": source_code,
        "implementation_path": candidate_rel,
        "files": file_hashes,
        "inference_contract": "CPU next-token top-k sampling; educational continuation only",
    }
    (export / "manifest.json").write_text(
        json.dumps(export_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    command = [
        sys.executable,
        str(ROOT / "probe_export.py"),
        "--export",
        str(export),
        "--implementation",
        str(ROOT / candidate_rel),
        "--prompt",
        prompt,
        "--seed",
        str(config["seed"]),
        "--max-new-tokens",
        "16",
    ]
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    probes = []
    with TemporaryDirectory(prefix="causal-lineage-reload-check-") as temp:
        expected_path = Path(temp) / "expected-prompt-logits.npy"
        np.save(expected_path, prompt_logits)
        command.extend(("--expected-logits", str(expected_path)))
        for _ in range(2):
            result = subprocess.run(
                command,
                cwd=ROOT.parent.parent,
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            probes.append(json.loads(result.stdout.strip().splitlines()[-1]))
    if probes[0] != probes[1]:
        raise AssertionError("fresh-process export probes were not deterministic")
    if not probes[0]["prompt_logits_close"]:
        raise AssertionError(
            "fresh-process logits do not match the in-process model: "
            f"max absolute difference {probes[0]['prompt_logits_max_abs_difference']}"
        )
    if not probes[0]["generated_ids"]:
        raise AssertionError("exported inference did not produce a sampled token")
    return {
        "prompt": prompt,
        "prompt_logits_sha256": expected_logits_sha,
        "fresh_process_logits_max_abs_difference": probes[0][
            "prompt_logits_max_abs_difference"
        ],
        "fresh_process_probes": 2,
        "generated_ids": probes[0]["generated_ids"],
        "sample_text": probes[0]["sample_text"],
        "export_manifest_sha256": sha256_bytes((export / "manifest.json").read_bytes()),
    }


def run_stage(stage: int, output: Path, candidate, candidate_path: Path):
    passages, instruction_records, data_manifest, data_sha = load_data()
    tokenizer = make_tokenizer(passages, instruction_records, CONFIG["max_vocabulary"])
    model = make_model(CONFIG, tokenizer)
    source_code, candidate_rel = source_identity(candidate_path)
    train_passages = [row for row in passages if row["split"] == "train"]
    validation_passages = [row for row in passages if row["split"] == "validation"]
    test_passages = [row for row in passages if row["split"] == "test"]
    train_rows = document_sequences(train_passages, tokenizer, CONFIG["max_length"])
    validation_rows = document_sequences(
        validation_passages, tokenizer, CONFIG["max_length"]
    )
    test_rows = document_sequences(test_passages, tokenizer, CONFIG["max_length"])
    instruction_rows = instruction_sequences(instruction_records, tokenizer, CONFIG)
    if not train_rows or not validation_rows or not test_rows or not instruction_rows:
        raise ValueError(
            "train, validation, test and adaptation splits must be nonempty"
        )

    output.mkdir(parents=True, exist_ok=True)
    stage1_dir = output / "checkpoints/stage-1"
    stage2_dir = output / "checkpoints/stage-2"
    base_metrics = None
    stage1_manifest = None
    if stage == 1:
        params = init_params(
            model, CONFIG["seed"], CONFIG["max_length"], candidate.causal_attention_mask
        )
        params, history = train_stage(
            model,
            params,
            train_rows,
            candidate,
            learning_rate=CONFIG["pretrain_learning_rate"],
            updates=CONFIG["pretrain_updates"],
            seed=CONFIG["seed"] + 1,
        )
        stage1_manifest = save_checkpoint(
            stage1_dir,
            params,
            stage=1,
            objective="chapter-isolated next-token pretraining",
            config=CONFIG,
            tokenizer=tokenizer,
            data_sha=data_sha,
            source_code=source_code,
            parent_checkpoint_sha=None,
            history=history,
        )
        validation = evaluate(
            model, params, validation_rows, candidate, CONFIG["batch_size"]
        )
        result = {
            "stage": "pretraining",
            "parameter_count": parameter_count(params),
            "train_passages": len(train_passages),
            "train_sequences": len(train_rows),
            "validation_passages": len(validation_passages),
            "validation_next_token": validation,
            "updates": len(history),
            "training_loss_first_last": [history[0]["loss"], history[-1]["loss"]],
            "checkpoint_sha256": stage1_manifest["checkpoint_sha256"],
            "parent_checkpoint_sha256": None,
            "data_sha256": data_sha,
            "tokenizer_sha256": canonical_hash(tokenizer),
            "model_sha256": model_hash(CONFIG, tokenizer),
        }
        (output / "stage-1.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        return result

    if not stage1_dir.is_dir():
        raise ValueError("stage 2 requires the verified stage-1 predecessor checkpoint")
    base_params, stage1_manifest = load_checkpoint(
        stage1_dir,
        stage=1,
        config=CONFIG,
        tokenizer=tokenizer,
        data_sha=data_sha,
        source_code=source_code,
        expected_parent=None,
        mask_fn=candidate.causal_attention_mask,
    )
    if stage == 1:
        raise ValueError("internal stage selection error")
    params, history = train_stage(
        model,
        base_params,
        instruction_rows,
        candidate,
        learning_rate=CONFIG["adaptation_learning_rate"],
        updates=CONFIG["adaptation_updates"],
        seed=CONFIG["seed"] + 2,
    )
    stage2_manifest = save_checkpoint(
        stage2_dir,
        params,
        stage=2,
        objective="instruction-style next-paragraph continuation adaptation",
        config=CONFIG,
        tokenizer=tokenizer,
        data_sha=data_sha,
        source_code=source_code,
        parent_checkpoint_sha=stage1_manifest["checkpoint_sha256"],
        history=history,
    )
    base_validation = evaluate(
        model, base_params, validation_rows, candidate, CONFIG["batch_size"]
    )
    adapted_validation = evaluate(
        model, params, validation_rows, candidate, CONFIG["batch_size"]
    )
    baseline_validation = unigram_baseline(
        train_rows, validation_rows, tokenizer, CONFIG["unigram_alpha"]
    )
    # Test chapters are scored only after both training stages are complete.
    base_test = evaluate(model, base_params, test_rows, candidate, CONFIG["batch_size"])
    adapted_test = evaluate(model, params, test_rows, candidate, CONFIG["batch_size"])
    baseline_test = unigram_baseline(
        train_rows, test_rows, tokenizer, CONFIG["unigram_alpha"]
    )
    evaluation = {
        "scope": "small CPU educational pilot; perplexity is not a benchmark or quality claim",
        "split_unit": data_manifest["split_unit"],
        "chapter_split_map": data_manifest["split_map"],
        "train_passages": len(train_passages),
        "validation_passages": len(validation_passages),
        "test_passages": len(test_passages),
        "adaptation_examples": len(instruction_records),
        "tokenizer_sha256": canonical_hash(tokenizer),
        "data_sha256": data_sha,
        "parameter_count": parameter_count(params),
        "validation": {
            "pretrained_transformer": base_validation,
            "adapted_transformer": adapted_validation,
            "unigram_baseline": baseline_validation,
        },
        "test": {
            "pretrained_transformer": base_test,
            "adapted_transformer": adapted_test,
            "unigram_baseline": baseline_test,
        },
        "adaptation": {
            "objective": "response-only next-token cross-entropy on adjacent training-chapter passage continuations",
            "labels": "source continuation text, not human preference or alignment labels",
            "preference_optimization": False,
            "rlhf": False,
            "updates": len(history),
            "training_loss_first_last": [history[0]["loss"], history[-1]["loss"]],
        },
        "base_checkpoint_sha256": stage1_manifest["checkpoint_sha256"],
        "adapted_checkpoint_sha256": stage2_manifest["checkpoint_sha256"],
        "checkpoint_parent_sha256": stage2_manifest["parent_checkpoint_sha256"],
        "claims": CONFIG["claims"],
    }
    (output / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    export_receipt = export_and_probe(
        output,
        params,
        config=CONFIG,
        tokenizer=tokenizer,
        data_sha=data_sha,
        source_code=source_code,
        candidate_rel=candidate_rel,
        model=model,
        candidate=candidate,
    )
    result = {"evaluation": evaluation, "export": export_receipt}
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", default="solution")
    parser.add_argument("--stage", choices=("1", "2", "all"), required=True)
    parser.add_argument("--output", default="runs/causal-lineage/reference")
    args = parser.parse_args()
    if args.implementation == "solution":
        candidate_path = ROOT / "solution/model.py"
    elif args.implementation == "starter":
        candidate_path = ROOT / "starter/model.py"
    else:
        candidate_path = Path(args.implementation)
        if not candidate_path.is_absolute():
            candidate_path = Path.cwd() / candidate_path
    candidate, candidate_path = load_candidate(candidate_path)
    output = Path(args.output)
    if not output.is_absolute():
        output = Path.cwd() / output
    stages = (1, 2) if args.stage == "all" else (int(args.stage),)
    for stage in stages:
        run_stage(stage, output, candidate, candidate_path)


if __name__ == "__main__":
    main()
