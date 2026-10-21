"""Evaluate a pinned local Pythia-14M checkpoint on the frozen text splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE / "data") not in sys.path:
    sys.path.insert(0, str(HERE / "data"))

import prepare
from evaluate_pilot import _wikitext_records
from pythia_pin import snapshot_identity


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _windows(records, tokenizer, context):
    windows = []
    byte_count = 0
    target_count = 0
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Pythia tokenizer has no EOS token")
    for record_id, text in records:
        normalized = prepare.model_text(text)
        byte_count += len(normalized.encode("utf-8"))
        targets = tokenizer.encode(normalized, add_special_tokens=False) + [eos]
        sequence = [eos, *targets]
        for start in range(0, len(sequence) - 1, context):
            labels = sequence[start + 1 : start + context + 1]
            inputs = sequence[start : start + len(labels)]
            if not labels or len(inputs) != len(labels):
                continue
            windows.append((record_id, inputs, labels))
            target_count += len(labels)
    return windows, byte_count, target_count


def _evaluate(records, tokenizer, model, context, batch_size):
    windows, byte_count, target_count = _windows(records, tokenizer, context)
    numerator = 0.0
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    for start in range(0, len(windows), batch_size):
        selected = windows[start : start + batch_size]
        input_ids = np.full((len(selected), context), pad, dtype=np.int64)
        attention = np.zeros((len(selected), context), dtype=np.int64)
        labels = np.full((len(selected), context), -100, dtype=np.int64)
        for row, (_, inputs, targets) in enumerate(selected):
            length = len(inputs)
            input_ids[row, :length] = inputs
            attention[row, :length] = 1
            labels[row, :length] = targets
        with torch.inference_mode():
            logits = model(
                input_ids=torch.from_numpy(input_ids),
                attention_mask=torch.from_numpy(attention),
                use_cache=False,
            ).logits
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                torch.from_numpy(labels).reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            numerator += float(loss)
    if target_count == 0 or byte_count == 0:
        raise ValueError("Pythia evaluation split has no targets or source bytes")
    return {
        "nll_sum_nats": numerator,
        "valid_token_targets": target_count,
        "nll_nats_per_token": numerator / target_count,
        "normalized_utf8_bytes": byte_count,
        "nll_bits_per_normalized_utf8_byte": numerator / (byte_count * np.log(2)),
        "records": len(records),
        "windows": len(windows),
    }


def evaluate(data_dir, model_dir, output_path, batch_size=8):
    data, model_path, output = map(
        lambda value: Path(value).resolve(), (data_dir, model_dir, output_path)
    )
    pinned_snapshot = snapshot_identity(model_path)
    prepare.verify_prepared(data, require_project_lock=True)
    memberships = json.loads((data / "memberships.json").read_text())
    wiki_records, _ = _wikitext_records(data, memberships)
    final = [
        json.loads(line)
        for line in (data / "qa-final.jsonl").read_text().splitlines()
        if line.strip()
    ]
    contexts = {}
    for row in final:
        contexts.setdefault(row["paragraph_id"], row["context"])
    domain_records = sorted(contexts.items())
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    if tokenizer.vocab_size != 50254 or len(tokenizer) != 50277:
        raise ValueError("local Pythia-14M tokenizer identity is unexpected")
    candidate = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.float32
    ).to("cpu")
    candidate.eval()
    context = 512
    report = {
        "status": "external pretrained baseline; local Pythia weights were not trained or modified",
        "model": "EleutherAI/pythia-14m",
        "revision": pinned_snapshot["revision"],
        "snapshot_files_sha256": pinned_snapshot["files_sha256"],
        "checkpoint_sha256": _hash(model_path / "model.safetensors"),
        "tokenizer_sha256": _hash(model_path / "tokenizer.json"),
        "evaluator_sha256": _hash(Path(__file__)),
        "pin_verifier_sha256": _hash(HERE / "pythia_pin.py"),
        "device": "CPU FP32",
        "context": context,
        "window_contract": "per-record EOS start; next-token targets are covered exactly once in contiguous context-512 windows; no cross-record target; final EOS is included",
        "byte_denominator": "UTF-8 bytes after the frozen whitespace collapse and trim; tokenizer targets and normalized bytes are reported separately",
        "general_wikitext_heldout": _evaluate(
            wiki_records, tokenizer, candidate, context, batch_size
        ),
        "domain_squad_final_contexts": _evaluate(
            domain_records, tokenizer, candidate, context, batch_size
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "sha256": _hash(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    evaluate(args.data, args.model, args.output, args.batch_size)


if __name__ == "__main__":
    main()
