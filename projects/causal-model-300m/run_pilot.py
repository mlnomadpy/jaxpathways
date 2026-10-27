#!/usr/bin/env python3
"""Execute the locked small CPU lineage; never creates a target-scale model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE / "data") not in sys.path:
    sys.path.insert(0, str(HERE / "data"))

import model
import prepare as prepare_data
from config import ModelConfig, TrainConfig
from training import (
    identity_payload,
    load_blocks,
    load_checkpoint,
    make_state,
    phase_step,
    read_jsonl,
    save_checkpoint,
    sft_batch,
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _candidate_hash():
    digest = hashlib.sha256()
    for path in (
        HERE / "config.py",
        HERE / "model.py",
        HERE / "training.py",
        HERE / "run_pilot.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _tree_copy(tree):
    return jax.tree.map(lambda value: jnp.array(value), tree)


def _stage_seed(name, root_seed):
    digest = hashlib.sha256(f"{root_seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _same_tree(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_tree(left[key], right[key]) for key in left
        )
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            _same_tree(a, b) for a, b in zip(left, right, strict=True)
        )
    if left is None:
        return right is None
    if isinstance(left, list):
        return left == right
    if isinstance(left, (int, float, str, bool)):
        return left == right
    return np.array_equal(np.asarray(left), np.asarray(right))


def _write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def run(prepared, output):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    # Verify every source, prepared output, tokenizer, schedule and target array
    # against the project-owned lock before initialization or optimizer work.
    prepare_data.verify_prepared(prepared, require_project_lock=True)
    config = ModelConfig()
    train_config = TrainConfig()
    config.validate()
    train_config.validate()
    recipe = json.loads((prepared / "frozen_recipe.json").read_text())
    if recipe["status"] != "frozen-data-protocol-pending-independent-review":
        raise ValueError("unexpected C02 recipe identity")
    if _hash(HERE / "data" / "frozen_recipe.json") != _hash(
        prepared / "frozen_recipe.json"
    ):
        raise ValueError(
            "prepared data does not match the project-owned accepted recipe"
        )
    if _hash(prepared / "tokenizer.json") != recipe["tokenizer"]["processor_sha256"]:
        raise ValueError("prepared tokenizer does not match the locked processor")
    output.mkdir(parents=True, exist_ok=True)

    batch_rows = read_jsonl(prepared / "pilot-batches.jsonl")
    schedules = {}
    for row in batch_rows:
        schedules.setdefault(row["branch"], []).append(row)
    for rows in schedules.values():
        rows.sort(key=lambda row: row["update"])

    base_arrays, base_index = load_blocks(prepared, "base-train")
    domain_arrays, domain_index = load_blocks(prepared, "continuation-domain")
    general_arrays, general_index = load_blocks(prepared, "continuation-general")
    fit_rows = read_jsonl(prepared / "sft-fit.jsonl")
    fit_by_id = {row["id"]: row for row in fit_rows}
    tokenizer = Tokenizer.from_file(str(prepared / "tokenizer.json"))

    root_seed = int(recipe["pilot_schedule"]["randomness"]["root_seed"])
    init_key, _ = jax.random.split(jax.random.PRNGKey(root_seed))
    initial_params = model.initialize(init_key, config)
    initial_hash = model.state_fingerprint(initial_params)
    parents = {
        "base": (initial_params, None, "seeded-random-initialization", initial_hash)
    }
    phase_reports = {}
    runtime_start = time.perf_counter()

    specs = [
        ("base", "base", base_arrays, base_index, "base-train"),
        ("continuation_mixture", "base", None, None, None),
        ("continuation_general_control", "base", None, None, None),
        ("sft_full_from_domain", "continuation_mixture", None, None, None),
        ("sft_lora_from_domain", "continuation_mixture", None, None, None),
        ("sft_full_from_general", "continuation_general_control", None, None, None),
    ]
    rates = recipe["pilot_schedule"]["optimizer"]["learning_rates"]
    step_counts = recipe["pilot_schedule"]["updates"]
    parent_checkpoint_hashes = {"seeded-random-initialization": initial_hash}

    for stage, parent_key, fixed_arrays, fixed_index, block_stem in specs:
        parent_params, parent_adapters, parent_stage, parent_hash = parents[parent_key]
        params = _tree_copy(parent_params)
        is_lora = stage == "sft_lora_from_domain"
        adapters = (
            model.initialize_lora(
                jax.random.PRNGKey(_stage_seed(stage + ":lora", root_seed)), config
            )
            if is_lora
            else None
        )
        scheduled = schedules[stage]
        total_steps = step_counts[stage]
        if len(scheduled) != total_steps:
            raise ValueError(
                f"locked {stage} schedule has {len(scheduled)} rows, expected {total_steps}"
            )
        stage_train_config = TrainConfig(
            batch_size=4,
            learning_rate=rates[stage],
            min_learning_rate=rates[stage] * 0.1,
            warmup_steps=max(1, int(0.1 * total_steps)),
            total_steps=total_steps,
            seed=_stage_seed(stage, root_seed),
            beta1=train_config.beta1,
            beta2=train_config.beta2,
            epsilon=train_config.epsilon,
            weight_decay=train_config.weight_decay,
            gradient_clip_norm=train_config.gradient_clip_norm,
        )
        stage_key = jax.random.PRNGKey(_stage_seed(stage + ":dropout", root_seed))
        chosen_ids = []
        for row in scheduled:
            if "block_ids" in row:
                chosen_ids.extend(row["block_ids"])
            if "domain_block_ids" in row:
                chosen_ids.extend(row["domain_block_ids"] + row["general_block_ids"])
            if "question_ids" in row:
                chosen_ids.extend(row["question_ids"])
        identity = identity_payload(
            stage,
            parent_hash,
            config,
            stage_train_config,
            prepared,
            _candidate_hash(),
        )
        identity["schedule_sha256"] = hashlib.sha256(
            json.dumps(scheduled, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        identity["schedule_sample_ids"] = chosen_ids
        identity["schedule_offsets"] = [0]
        for scheduled_row in scheduled:
            consumed_in_update = sum(
                len(scheduled_row.get(key, []))
                for key in (
                    "block_ids",
                    "domain_block_ids",
                    "general_block_ids",
                    "question_ids",
                )
            )
            identity["schedule_offsets"].append(
                identity["schedule_offsets"][-1] + consumed_in_update
            )
        if identity["schedule_offsets"][-1] != len(chosen_ids):
            raise AssertionError(
                "frozen schedule ID count does not match its update offsets"
            )
        identity["parent_stage"] = parent_stage
        identity["adapter_spec"] = recipe["lora"] if is_lora else None
        identity["target_mask_policy"] = (
            "answer tokens plus EOS only"
            if stage.startswith("sft")
            else "all aligned source next-token targets"
        )
        identity["prompt_template"] = recipe["final_selection"]["generation"][
            "template"
        ]
        # The stage order itself is immutable and checkpoint-validated.
        state = make_state(
            params,
            adapters,
            stage_key,
            list(range(len(chosen_ids))),
        )
        stage_dir = output / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        _write_json(stage_dir / "identity.json", identity)
        logs = []
        for update, row in enumerate(scheduled):
            if stage == "base":
                indices = [fixed_index[item] for item in row["block_ids"]]
                batches = [
                    {
                        key: value
                        for key, value in _stack_blocks(fixed_arrays, indices).items()
                    }
                ]
            elif stage.startswith("continuation_"):
                if stage == "continuation_mixture":
                    domain_ids, general_ids = (
                        row["domain_block_ids"],
                        row["general_block_ids"],
                    )
                    if len(domain_ids) != 4 or len(general_ids) != 1:
                        raise ValueError(
                            "locked continuation mixture must accumulate [4,1] blocks"
                        )
                    batches = [
                        _stack_blocks(
                            domain_arrays, [domain_index[item] for item in domain_ids]
                        ),
                        _stack_blocks(
                            general_arrays,
                            [general_index[item] for item in general_ids],
                        ),
                    ]
                else:
                    block_ids = row["block_ids"]
                    batches = [
                        _stack_blocks(
                            general_arrays,
                            [general_index[item] for item in block_ids[:4]],
                        ),
                        _stack_blocks(
                            general_arrays,
                            [general_index[item] for item in block_ids[4:]],
                        ),
                    ]
            else:
                examples = [fit_by_id[item] for item in row["question_ids"]]
                batches = [
                    sft_batch(examples, tokenizer, config, prepare_data.qa_prompt)
                ]
            if update == total_steps - 1:
                save_checkpoint(stage_dir / "resume-before-final.npz", state, identity)
            state, metrics = phase_step(
                state,
                batches,
                config,
                stage_train_config,
                rates[stage],
                total_steps,
                stage_train_config.warmup_steps,
            )
            consumed = len(row.get("block_ids", []))
            consumed += len(row.get("domain_block_ids", [])) + len(
                row.get("general_block_ids", [])
            )
            consumed += len(row.get("question_ids", []))
            state["cursor"] += consumed
            state["seen_ids"].extend(row.get("block_ids", []))
            state["seen_ids"].extend(row.get("domain_block_ids", []))
            state["seen_ids"].extend(row.get("general_block_ids", []))
            state["seen_ids"].extend(row.get("question_ids", []))
            logs.append({"update": update + 1, **metrics})
        checkpoint_hash = save_checkpoint(
            stage_dir / "final-state.npz", state, identity
        )
        replay_path = stage_dir / "fresh-process-replay.npz"
        subprocess.run(
            [
                sys.executable,
                str(HERE / "tests" / "replay_branch.py"),
                str(prepared),
                str(stage_dir),
                str(replay_path),
            ],
            check=True,
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
            capture_output=True,
            text=True,
        )
        replayed = load_checkpoint(
            replay_path,
            identity,
            expected_order=list(range(len(chosen_ids))),
            expected_seen_ids=chosen_ids,
        )
        if not _same_tree(state, replayed):
            raise AssertionError(f"fresh-process full-state replay differs for {stage}")
        parent_checkpoint_hashes[stage] = checkpoint_hash
        if is_lora:
            effective = model.merge_lora(params, adapters, config)
            parents[stage] = (effective, _tree_copy(adapters), stage, checkpoint_hash)
        else:
            parents[stage] = (_tree_copy(state["params"]), None, stage, checkpoint_hash)
        _write_json(
            stage_dir / "metrics.json",
            {
                "stage": stage,
                "parent_stage": parent_stage,
                "parent_checkpoint_sha256": parent_hash,
                "checkpoint_sha256": checkpoint_hash,
                "fresh_process_replay_sha256": _hash(replay_path),
                "fresh_process_replay_passed": True,
                "identity": identity,
                "updates": logs,
                "elapsed_seconds": time.perf_counter() - runtime_start,
                "final_optimizer_step": state["step"],
                "final_valid_target_count": sum(
                    row["valid_target_count"] for row in logs
                ),
                "adapter_parameters": (
                    model.lora_parameter_count(config) if is_lora else 0
                ),
                "full_parameters": model.parameter_count(config),
            },
        )
        phase_reports[stage] = {"checkpoint_sha256": checkpoint_hash, "metrics": logs}
        print(
            json.dumps(
                {
                    "stage": stage,
                    "loss_first": logs[0]["loss"],
                    "loss_last": logs[-1]["loss"],
                    "checkpoint_sha256": checkpoint_hash,
                }
            ),
            flush=True,
        )

    _write_json(
        output / "pilot-index.json",
        {
            "status": "bounded CPU pilot complete; no target-scale qualification",
            "recipe_sha256": _hash(prepared / "frozen_recipe.json"),
            "source_manifest_sha256": _hash(prepared / "source-manifest.json"),
            "tokenizer_sha256": _hash(prepared / "tokenizer.json"),
            "model_config": config.as_dict(),
            "parameter_count": model.parameter_count(config),
            "runtime_seconds": time.perf_counter() - runtime_start,
            "branches": phase_reports,
        },
    )


def _stack_blocks(arrays, indices):
    return {
        key: jnp.asarray(arrays[key][indices])
        for key in (
            "input_ids",
            "target_ids",
            "target_mask",
            "segment_ids",
            "position_ids",
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == "__main__":
    main()
