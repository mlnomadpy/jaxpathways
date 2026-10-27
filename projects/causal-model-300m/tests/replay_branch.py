"""Replay the final update of one actual stage in a fresh Python process."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import jax
from tokenizers import Tokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "data"))
candidate_path = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else None
if candidate_path is not None:
    candidate_spec = importlib.util.spec_from_file_location("model", candidate_path)
    if candidate_spec is None or candidate_spec.loader is None:
        raise ImportError(f"cannot load candidate implementation: {candidate_path}")
    candidate_module = importlib.util.module_from_spec(candidate_spec)
    sys.modules["model"] = candidate_module
    candidate_spec.loader.exec_module(candidate_module)
import model
import prepare as prepare_data
from config import ModelConfig, TrainConfig
from training import (
    load_blocks,
    load_checkpoint,
    phase_step,
    read_jsonl,
    save_checkpoint,
    sft_batch,
)


def main():
    data, stage_dir, output = map(Path, sys.argv[1:4])
    identity_path = stage_dir / "identity.json"
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text())
    else:
        import numpy as np

        with np.load(stage_dir / "final-state.npz", allow_pickle=False) as archive:
            metadata = json.loads(
                np.asarray(archive["__meta__"], dtype=np.uint8).tobytes()
            )
        identity = metadata["identity"]
    stage = identity["stage"]
    config = ModelConfig(**identity["model_config"])
    train_config = TrainConfig(**identity["train_config"])
    recipe = json.loads((data / "frozen_recipe.json").read_text())
    schedules = {}
    for row in read_jsonl(data / "pilot-batches.jsonl"):
        schedules.setdefault(row["branch"], []).append(row)
    schedule = sorted(schedules[stage], key=lambda row: row["update"])
    row = schedule[-1]
    order_length = sum(
        len(item.get("block_ids", []))
        + len(item.get("domain_block_ids", []))
        + len(item.get("general_block_ids", []))
        + len(item.get("question_ids", []))
        for item in schedule
    )
    expected_seen_ids = [
        sample_id
        for item in schedule[:-1]
        for key in (
            "block_ids",
            "domain_block_ids",
            "general_block_ids",
            "question_ids",
        )
        for sample_id in item.get(key, [])
    ]
    frozen_seen_ids = [
        sample_id
        for item in schedule
        for key in (
            "block_ids",
            "domain_block_ids",
            "general_block_ids",
            "question_ids",
        )
        for sample_id in item.get(key, [])
    ]
    if frozen_seen_ids != identity.get("schedule_sample_ids"):
        raise ValueError(
            "checkpoint identity does not match the verified source schedule"
        )
    state = load_checkpoint(
        stage_dir / "resume-before-final.npz",
        identity,
        expected_order=list(range(order_length)),
        expected_seen_ids=identity["schedule_sample_ids"],
    )
    if state["seen_ids"] != expected_seen_ids:
        raise ValueError(
            "checkpoint consumed-ID cursor differs from the scheduled prefix"
        )
    if stage == "base":
        arrays, index = load_blocks(data, "base-train")
        batches = [_blocks(arrays, [index[item] for item in row["block_ids"]])]
    elif stage == "continuation_mixture":
        domain, domain_index = load_blocks(data, "continuation-domain")
        general, general_index = load_blocks(data, "continuation-general")
        batches = [
            _blocks(domain, [domain_index[item] for item in row["domain_block_ids"]]),
            _blocks(
                general, [general_index[item] for item in row["general_block_ids"]]
            ),
        ]
    elif stage == "continuation_general_control":
        general, index = load_blocks(data, "continuation-general")
        ids = row["block_ids"]
        batches = [
            _blocks(general, [index[item] for item in ids[:4]]),
            _blocks(general, [index[item] for item in ids[4:]]),
        ]
    else:
        tokenizer = Tokenizer.from_file(str(data / "tokenizer.json"))
        by_id = {item["id"]: item for item in read_jsonl(data / "sft-fit.jsonl")}
        batches = [
            sft_batch(
                [by_id[item] for item in row["question_ids"]],
                tokenizer,
                config,
                prepare_data.qa_prompt,
            )
        ]
    rate = recipe["pilot_schedule"]["optimizer"]["learning_rates"][stage]
    state, metrics = phase_step(
        state,
        batches,
        config,
        train_config,
        rate,
        recipe["pilot_schedule"]["updates"][stage],
        train_config.warmup_steps,
    )
    consumed = (
        len(row.get("block_ids", []))
        + len(row.get("domain_block_ids", []))
        + len(row.get("general_block_ids", []))
        + len(row.get("question_ids", []))
    )
    state["cursor"] += consumed
    for key in ("block_ids", "domain_block_ids", "general_block_ids", "question_ids"):
        state["seen_ids"].extend(row.get(key, []))
    digest = save_checkpoint(output, state, identity)
    print(
        json.dumps(
            {"stage": stage, "metrics": metrics, "checkpoint_sha256": digest},
            sort_keys=True,
        )
    )


def _blocks(arrays, rows):
    import jax.numpy as jnp

    return {
        key: jnp.asarray(arrays[key][rows])
        for key in (
            "input_ids",
            "target_ids",
            "target_mask",
            "segment_ids",
            "position_ids",
        )
    }


if __name__ == "__main__":
    main()
