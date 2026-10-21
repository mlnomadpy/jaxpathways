"""Response-only held-out likelihood and a train-only byte unigram control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

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
import prepare
from config import ModelConfig
from evaluate_pilot import _nll, _wikitext_records
from run_pilot import _candidate_hash, _hash
from training import load_checkpoint

BRANCHES = (
    "base",
    "continuation_mixture",
    "continuation_general_control",
    "sft_full_from_domain",
    "sft_lora_from_domain",
    "sft_full_from_general",
)


def _response_batch(rows, tokenizer, context):
    inputs, targets, masks, kept, excluded = [], [], [], [], []
    for row in rows:
        prompt_ids = tokenizer.encode(
            prepare.qa_prompt(row["question"], row["context"]),
            add_special_tokens=False,
        ).ids
        answer_ids = tokenizer.encode(
            prepare.model_text(row["answers"][0]), add_special_tokens=False
        ).ids
        # Match SFT: BOS predicts the prompt, answer-only targets follow, and
        # EOS is also a response target. No prompt or answer is truncated.
        sequence_targets = [*prompt_ids, *answer_ids, 2]
        sequence_inputs = [1, *prompt_ids, *answer_ids]
        if len(sequence_inputs) != len(sequence_targets):
            raise AssertionError("aligned response inputs and targets diverged")
        if len(sequence_inputs) > context:
            excluded.append(row["question_id"])
            continue
        target_mask = [False] * len(prompt_ids) + [True] * (len(answer_ids) + 1)
        inputs.append(sequence_inputs)
        targets.append(sequence_targets)
        masks.append(target_mask)
        kept.append(row["question_id"])
    return inputs, targets, masks, kept, excluded


def _answer_nll(params, rows, tokenizer, config, branch):
    inputs, targets, masks, kept, excluded = _response_batch(
        rows, tokenizer, config.context
    )
    if not kept:
        raise ValueError("no final answers fit without truncation")
    padded_inputs = np.zeros((len(kept), config.context), dtype=np.int32)
    padded_targets = np.zeros_like(padded_inputs)
    valid = np.zeros_like(padded_inputs, dtype=bool)
    response = np.zeros_like(padded_inputs, dtype=bool)
    positions = np.zeros_like(padded_inputs)
    segments = np.zeros_like(padded_inputs)
    for index, (x, y, mask) in enumerate(zip(inputs, targets, masks, strict=True)):
        length = len(x)
        padded_inputs[index, :length] = x
        padded_targets[index, :length] = y
        valid[index, :length] = True
        response[index, :length] = mask
        positions[index, :length] = np.arange(length)
        segments[index, :length] = 1
    logits = model.logits(
        params,
        jnp.asarray(padded_inputs),
        config,
        valid_mask=jnp.asarray(valid),
        segment_ids=jnp.asarray(segments),
        position_ids=jnp.asarray(positions),
    ).astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    selected = jnp.take_along_axis(
        log_probs, jnp.asarray(padded_targets)[..., None], axis=-1
    )[..., 0]
    target_count = int(np.sum(response))
    numerator = -float(jnp.sum(jnp.where(jnp.asarray(response), selected, 0.0)))
    return {
        "questions_scored": len(kept),
        "eligible_question_ids": kept,
        "questions_excluded_for_untruncated_context_answer_fit": len(excluded),
        "excluded_question_ids": excluded,
        "response_token_targets_including_eos": target_count,
        "nll_sum_nats": numerator,
        "nll_nats_per_response_target": numerator / target_count,
        "branch": branch,
        "answer_reference": "first published answer string, matching the frozen SFT selection rule",
    }


def _byte_unigram(train_records, heldout_records, split_name):
    counts = np.ones(256, dtype=np.float64)
    for row in train_records:
        counts += np.bincount(
            np.frombuffer(
                prepare.model_text(row["text"]).encode("utf-8"), dtype=np.uint8
            ),
            minlength=256,
        )
    total = float(counts.sum())
    log_probability = np.log2(counts / total)
    loss_bits = 0.0
    byte_count = 0
    for record in heldout_records:
        payload = prepare.model_text(record["text"]).encode("utf-8")
        values = np.frombuffer(payload, dtype=np.uint8)
        byte_count += len(values)
        loss_bits -= float(log_probability[values].sum())
    if not byte_count:
        raise ValueError(f"{split_name} has no normalized UTF-8 bytes")
    return {
        "split": split_name,
        "nll_bits_per_normalized_utf8_byte": loss_bits / byte_count,
        "normalized_utf8_bytes": byte_count,
        "training_bytes": int(total - 256),
        "training_source": "base-train.jsonl only; add-one smoothing over all 256 byte values",
    }


def evaluate(data_dir, run_dir, evaluation_dir, output_path):
    data, run, evaluation, output = map(
        lambda value: Path(value).resolve(),
        (data_dir, run_dir, evaluation_dir, output_path),
    )
    prepare.verify_prepared(data, require_project_lock=True)
    recipe = json.loads((data / "frozen_recipe.json").read_text())
    tokenizer_path = data / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    architecture = recipe["pilot_schedule"]["architecture"]
    config = ModelConfig(
        vocab_size=architecture["vocab_size"],
        context=architecture["context"],
        width=architecture["width"],
        layers=architecture["layers"],
        heads=architecture["heads"],
        mlp_width=architecture["mlp_width"],
        dropout=architecture["dropout"],
        tied_embeddings=True,
        rope_base=10000.0,
    )
    final_rows = [
        json.loads(line)
        for line in (data / "qa-final.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(final_rows) != 128:
        raise ValueError("final QA rows differ from the frozen 128-question set")
    train_rows = [
        json.loads(line)
        for line in (data / "base-train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    membership = json.loads((data / "memberships.json").read_text())
    wiki_records, _ = _wikitext_records(data, membership)
    wiki_test = [{"id": record_id, "text": text} for record_id, text in wiki_records]
    domain_test_by_id = {}
    for row in final_rows:
        domain_test_by_id[row["paragraph_id"]] = row["context"]
    domain_test = [
        {"id": key, "text": value} for key, value in sorted(domain_test_by_id.items())
    ]
    domain_records = [(row["id"], row["text"]) for row in domain_test]
    result = {
        "status": "supplemental held-out diagnostics on the frozen pilot; not tuned on final results",
        "perplexity_scope": "exp(response-only NLL) is reported only for this frozen 512-token tokenizer and vocabulary; it is not comparable across tokenizers",
        "recipe_sha256": _hash(data / "frozen_recipe.json"),
        "tokenizer_sha256": _hash(tokenizer_path),
        "checkpoint_directory": str(run),
        "main_evaluation_sha256": _hash(evaluation / "metrics.json"),
        "evaluator_sha256": _hash(Path(__file__)),
        "source_hashes": {
            "model.py": _hash(HERE / "model.py"),
            "training.py": _hash(HERE / "training.py"),
            "run_pilot.py": _hash(HERE / "run_pilot.py"),
            "config.py": _hash(HERE / "config.py"),
            "evaluate_pilot.py": _hash(HERE / "evaluate_pilot.py"),
            "squad_scoring.py": _hash(HERE / "squad_scoring.py"),
            "prepare.py": _hash(HERE / "data" / "prepare.py"),
            "source-manifest.json": _hash(data / "source-manifest.json"),
            "tokenizer.json": _hash(tokenizer_path),
            "base-train.jsonl": _hash(data / "base-train.jsonl"),
            "qa-final.jsonl": _hash(data / "qa-final.jsonl"),
            "frozen_recipe.json": _hash(data / "frozen_recipe.json"),
            "memberships.json": _hash(data / "memberships.json"),
            "metrics.json": _hash(evaluation / "metrics.json"),
        },
        "branch_bindings": {},
        "simple_byte_unigram": {
            "general_wikitext": _byte_unigram(
                train_rows, wiki_test, "WikiText-2 validation+test"
            ),
            "squad_final_contexts": _byte_unigram(
                train_rows, domain_test, "SQuAD final contexts"
            ),
        },
        "answer_only_likelihood": {},
    }
    root_seed = int(recipe["pilot_schedule"]["randomness"]["root_seed"])
    initialization_key, _ = jax.random.split(jax.random.PRNGKey(root_seed))
    initial_params = model.initialize(initialization_key, config)
    initial_logits = jax.jit(
        lambda params, input_ids, valid_mask, segment_ids, position_ids: model.logits(
            params,
            input_ids,
            config,
            valid_mask=valid_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
        )
    )
    result["random_initialization"] = {
        "parameter_tree_sha256": model.state_fingerprint(initial_params),
        "general_wikitext_nll": _nll(
            initial_params, wiki_records, tokenizer, config, initial_logits
        ),
        "domain_squad_context_nll": _nll(
            initial_params,
            domain_records,
            tokenizer,
            config,
            initial_logits,
        ),
        "answer_only_likelihood": _answer_nll(
            initial_params, final_rows, tokenizer, config, "random_initialization"
        ),
    }
    for branch in BRANCHES:
        identity = json.loads((run / branch / "identity.json").read_text())
        if identity["candidate_source_sha256"] != _candidate_hash():
            raise ValueError(
                f"{branch} checkpoint does not bind the current candidate source"
            )
        state = load_checkpoint(
            run / branch / "final-state.npz",
            identity,
            expected_order=list(range(len(identity["schedule_sample_ids"]))),
            expected_seen_ids=identity["schedule_sample_ids"],
        )
        params = state["params"]
        if state["adapters"] is not None:
            params = model.merge_lora(params, state["adapters"], config)
        result["answer_only_likelihood"][branch] = _answer_nll(
            params, final_rows, tokenizer, config, branch
        )
        result["answer_only_likelihood"][branch]["same_tokenizer_perplexity"] = float(
            np.exp(
                result["answer_only_likelihood"][branch]["nll_nats_per_response_target"]
            )
        )
        result["branch_bindings"][branch] = {
            "identity_sha256": _hash(run / branch / "identity.json"),
            "checkpoint_sha256": _hash(run / branch / "final-state.npz"),
            "candidate_source_sha256": identity["candidate_source_sha256"],
            "parent_checkpoint_sha256": identity["parent_checkpoint_sha256"],
            "recipe_sha256": identity["recipe_sha256"],
            "processor_sha256": identity["processor_sha256"],
            "source_manifest_sha256": identity["source_manifest_sha256"],
        }
    excluded_sets = {
        tuple(report["excluded_question_ids"])
        for report in result["answer_only_likelihood"].values()
    }
    if len(excluded_sets) != 1:
        raise AssertionError(
            "answer-likelihood branches did not use one shared eligibility set"
        )
    eligible_sets = {
        tuple(report["eligible_question_ids"])
        for report in result["answer_only_likelihood"].values()
    }
    if len(eligible_sets) != 1:
        raise AssertionError(
            "answer-likelihood branches did not use one shared eligible question set"
        )
    eligible_ids = list(next(iter(eligible_sets)))
    eligible_hash = hashlib.sha256(
        json.dumps(eligible_ids, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    result["answer_only_eligibility"] = {
        "rule": "score only if BOS + complete frozen question/context prompt + first published answer + EOS fit context; never crop; apply identically to all six checkpoints",
        "eligible_question_ids": eligible_ids,
        "eligible_question_ids_sha256": eligible_hash,
        "eligible_question_count": len(eligible_ids),
        "excluded_question_ids": list(next(iter(excluded_sets))),
        "exclusion_reason": "complete prompt, answer and EOS exceed the 512-token context",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "sha256": _hash(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.data, args.run, args.evaluation, args.output)


if __name__ == "__main__":
    main()
