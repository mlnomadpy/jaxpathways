"""Continue one serialized phase checkpoint in a fresh Python process."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import run
from config import EncoderConfig, TrainConfig
from trainer import (
    initialize_state,
    load_checkpoint,
    save_checkpoint,
    train_human_contrastive,
    train_mlm,
    train_weak_contrastive,
)


def main(instruction_path):
    instruction = json.loads(Path(instruction_path).read_text())
    candidate, _ = run.load_implementation(instruction["candidate"])
    config = EncoderConfig(**instruction["config"])
    train_config = TrainConfig(**instruction["train_config"])
    payload_path = Path(instruction["payload"])
    with np.load(payload_path, allow_pickle=False) as payload:
        rows = payload["rows"]
        state_count = int(instruction["count"])
        template = initialize_state(
            candidate,
            config,
            state_count,
            seed=instruction["seed"],
            train_config=train_config,
        )
        state, _ = load_checkpoint(
            instruction["checkpoint"], template, instruction["identity"]
        )
        kind = instruction["kind"]
        if kind == "mlm":
            state, stats = train_mlm(
                candidate,
                state,
                rows,
                config,
                train_config,
                steps=1,
                tag="resume-control",
            )
        elif kind in {"weak", "weak_only"}:
            state, stats = train_weak_contrastive(
                candidate,
                state,
                payload["left"],
                payload["right"],
                payload["groups"],
                config,
                train_config,
                steps=1,
                retained_mlm_weight=float(instruction["retained_mlm_weight"]),
                mlm_examples=rows,
                tag="resume-weak-control",
            )
        elif kind == "human":
            queries = json.loads(Path(instruction["queries"]).read_text())
            documents = json.loads(Path(instruction["documents"]).read_text())
            for query in queries:
                query["ids"] = np.asarray(query["ids"], np.int32)
            for document in documents:
                document["ids"] = np.asarray(document["ids"], np.int32)
            state, stats = train_human_contrastive(
                candidate,
                state,
                queries,
                documents,
                config,
                train_config,
                steps=1,
                max_positive_per_query=4,
            )
        else:
            raise ValueError(f"unknown resume fixture: {kind}")
        Path(instruction["trace_output"]).write_text(
            json.dumps(stats["trace"], sort_keys=True)
        )
        save_checkpoint(instruction["output"], state, instruction["identity"])


if __name__ == "__main__":
    main(sys.argv[1])
