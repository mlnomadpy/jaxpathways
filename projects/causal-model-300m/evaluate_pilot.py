"""Reloaded held-out evaluation for the frozen C02 six-branch CPU pilot."""

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
from run_pilot import _candidate_hash, _hash
from squad_scoring import normalize_answer, score as score_answer, token_f1
from training import load_checkpoint, read_jsonl


def _target_chunks(text: str, tokenizer: Tokenizer, context: int):
    ids = tokenizer.encode(prepare.model_text(text), add_special_tokens=False).ids
    if not ids:
        return [], len(prepare.model_text(text).encode("utf-8"))
    ids.append(2)
    chunks = []
    for start in range(0, len(ids), context):
        target = ids[start : start + context]
        inputs = [1, *target[:-1]]
        chunks.append((inputs, target))
    return chunks, len(prepare.model_text(text).encode("utf-8"))


def _batches(records, tokenizer, context, batch_size=8):
    chunks = []
    byte_count = 0
    target_count = 0
    for record_id, text in records:
        record_chunks, nbytes = _target_chunks(text, tokenizer, context)
        byte_count += nbytes
        for inputs, targets in record_chunks:
            chunks.append((record_id, inputs, targets))
            target_count += len(targets)
    for start in range(0, len(chunks), batch_size):
        selected = chunks[start : start + batch_size]
        input_ids = np.zeros((len(selected), context), dtype=np.int32)
        target_ids = np.zeros_like(input_ids)
        target_mask = np.zeros_like(input_ids, dtype=bool)
        segments = np.zeros_like(input_ids)
        positions = np.zeros_like(input_ids)
        for row, (_, inputs, targets) in enumerate(selected):
            length = len(targets)
            input_ids[row, :length] = inputs
            target_ids[row, :length] = targets
            target_mask[row, :length] = True
            segments[row, :length] = 1
            positions[row, :length] = np.arange(length)
        yield {
            "input_ids": jnp.asarray(input_ids),
            "target_ids": jnp.asarray(target_ids),
            "target_mask": jnp.asarray(target_mask),
            "segment_ids": jnp.asarray(segments),
            "position_ids": jnp.asarray(positions),
        }
    return byte_count, target_count


def _nll(params, records, tokenizer, config, logits_function):
    # Materialize the generator's batches while retaining exact denominators.
    batches = list(_batches(records, tokenizer, config.context))
    byte_count = sum(
        len(prepare.model_text(text).encode("utf-8")) for _, text in records
    )
    targets = 0
    numerator = 0.0
    for batch in batches:
        logits = logits_function(
            params,
            batch["input_ids"],
            valid_mask=batch["input_ids"] != 0,
            segment_ids=batch["segment_ids"],
            position_ids=batch["position_ids"],
        ).astype(jnp.float32)
        log_probs = jnp.asarray(jax_log_softmax(logits))
        selected = jnp.take_along_axis(
            log_probs, batch["target_ids"][..., None], axis=-1
        )[..., 0]
        numerator -= float(jnp.sum(jnp.where(batch["target_mask"], selected, 0.0)))
        targets += int(jnp.sum(batch["target_mask"]))
    if not targets:
        raise ValueError("held-out evaluation has no tokenizer targets")
    return {
        "nll_sum_nats": numerator,
        "valid_token_targets": targets,
        "nll_nats_per_token": numerator / targets,
        "normalized_utf8_bytes": byte_count,
        "nll_bits_per_normalized_utf8_byte": numerator / (byte_count * np.log(2)),
    }


def jax_log_softmax(logits):
    # A small wrapper keeps the evaluation numerator in FP32 explicitly.
    import jax

    return jax.nn.log_softmax(logits, axis=-1)


def generate_with_candidate(params, prompt, config, max_new_tokens):
    """Use the selected decoder's own cached prefill/decode generation API."""
    return model.generate(
        params,
        prompt,
        config,
        max_new_tokens=max_new_tokens,
        eos_token_id=2,
    )


def _wikitext_records(data, memberships):
    records = []
    for split in ("validation", "test"):
        source = next(
            path
            for path in (data / "source" / "wikitext-2-raw-v1").glob("*.parquet")
            if split in path.name
        )
        articles = prepare._wiki_articles(source, split)
        records.extend((row["record_id"], row["model_text"]) for row in articles)
    inventory = {
        row["record_id"]: row
        for row in memberships["articles"]
        if row["source"] == "wikitext" and row["split"] in {"validation", "test"}
    }
    if {record_id for record_id, _ in records} != set(inventory):
        raise ValueError("reparsed WikiText held-out article IDs differ from freeze")
    return records, inventory


def _load_branch(data, run_dir, name):
    stage_dir = run_dir / name
    identity = json.loads((stage_dir / "identity.json").read_text())
    if identity["candidate_source_sha256"] != _candidate_hash():
        raise ValueError(f"{name} checkpoint candidate no longer matches source")
    state = load_checkpoint(
        stage_dir / "final-state.npz",
        identity,
        expected_order=list(range(len(identity["schedule_sample_ids"]))),
        expected_seen_ids=identity["schedule_sample_ids"],
    )
    return identity, state


def evaluate(data_dir, run_dir, output_dir):
    data, run, output = (
        Path(data_dir).resolve(),
        Path(run_dir).resolve(),
        Path(output_dir).resolve(),
    )
    prepare.verify_prepared(data, require_project_lock=True)
    recipe = json.loads((data / "frozen_recipe.json").read_text())
    source_manifest = json.loads((data / "source-manifest.json").read_text())
    tokenizer_path = data / "tokenizer.json"
    if _hash(tokenizer_path) != recipe["tokenizer"]["processor_sha256"]:
        raise ValueError("evaluation tokenizer differs from the frozen processor")
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
    candidate_logits = jax.jit(
        lambda params, input_ids, valid_mask, segment_ids, position_ids: model.logits(
            params,
            input_ids,
            config,
            valid_mask=valid_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
        )
    )
    final_rows = read_jsonl(data / "qa-final.jsonl")
    if len(final_rows) != 128:
        raise ValueError(
            "custom final QA split differs from the frozen 128-question inventory"
        )
    final_ids = {row["question_id"] for row in final_rows}
    if final_ids != set(
        json.loads((data / "memberships.json").read_text())["final_question_ids"]
    ):
        raise ValueError("final QA question membership differs from the frozen split")
    memberships = json.loads((data / "memberships.json").read_text())
    wiki_records, wiki_inventory = _wikitext_records(data, memberships)
    contexts = {}
    for row in final_rows:
        contexts.setdefault(row["paragraph_id"], row["context"])
    domain_records = sorted(contexts.items())
    branches = [
        "base",
        "continuation_mixture",
        "continuation_general_control",
        "sft_full_from_domain",
        "sft_lora_from_domain",
        "sft_full_from_general",
    ]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "reloaded bounded CPU pilot evaluation; no target-scale or general assistant qualification",
        "recipe_sha256": _hash(data / "frozen_recipe.json"),
        "tokenizer_sha256": _hash(tokenizer_path),
        "evaluator_source_sha256": _hash(Path(__file__)),
        "scorer": {
            "name": "SQuAD v1.1 exact match and token F1",
            "upstream_reference": "https://github.com/huggingface/evaluate/blob/main/metrics/squad/compute_score.py",
            "upstream_reference_commit": "main is mutable; local semantic port is pinned by reference_bytes_sha256",
            "reference_bytes_path": "projects/causal-model-300m/squad_scoring.py",
            "reference_bytes_sha256": _hash(HERE / "squad_scoring.py"),
            "data_repository_revision": source_manifest["sources"]["squad"]["revision"],
            "empty_token_overlap_f1": 0.0,
            "empty_normalized_strings_exact_match": 1.0,
            "empty_gold_answer_count": sum(
                not normalize_answer(answer)
                for row in final_rows
                for answer in row["answers"]
            ),
        },
        "evaluation_source": "frozen WikiText-2 raw validation+test articles and 128 filtered official SQuAD 1.1 development questions",
        "wiki_heldout_membership": "All 122 WikiText validation/test articles are held out from every training stage and have the frozen quarantine-final-evaluation-overlap role. Reconstructed article components contain zero observed SQuAD articles and zero SQuAD dev questions; this is an exact membership-edge result, not proof against semantic overlap.",
        "simple_baseline": {
            "exact_match": 0.0,
            "token_f1": 0.0,
            "definition": "empty-answer prediction",
        },
        "branches": {},
    }
    generations_path = output / "qa-generations.jsonl"
    with generations_path.open("w", encoding="utf-8") as generations_file:
        for branch in branches:
            print(f"evaluating {branch}", flush=True)
            identity, state = _load_branch(data, run, branch)
            adapters = state["adapters"]
            params = state["params"]
            if branch == "sft_lora_from_domain":
                merged_params = model.merge_lora(params, adapters, config)
                # The evaluated merged model is checked against the unmerged path below.
                for probe in ([1, 5, 6], [1, 7, 8, 9]):
                    a = model.logits(
                        params, jnp.asarray([probe]), config, adapters=adapters
                    )
                    b = model.logits(merged_params, jnp.asarray([probe]), config)
                    np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)
            evaluated_params = (
                model.merge_lora(params, adapters, config)
                if adapters is not None
                else params
            )
            wiki_metrics = _nll(
                evaluated_params, wiki_records, tokenizer, config, candidate_logits
            )
            domain_metrics = _nll(
                evaluated_params, domain_records, tokenizer, config, candidate_logits
            )
            scored, branch_rows = [], []
            for row in final_rows:
                prompt = prepare.qa_prompt(row["question"], row["context"])
                prompt_ids = [
                    1,
                    *tokenizer.encode(prompt, add_special_tokens=False).ids,
                ]
                if len(prompt_ids) + 32 > config.context:
                    raise ValueError(
                        f"frozen final prompt no longer fits: {row['question_id']}"
                    )
                generated_ids = generate_with_candidate(
                    evaluated_params, prompt_ids, config, max_new_tokens=32
                )
                prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
                metrics = score_answer(prediction, row["answers"])
                scored.append(metrics)
                record = {
                    "branch": branch,
                    "question_id": row["question_id"],
                    "prompt": prompt,
                    "prompt_token_ids": prompt_ids,
                    "generated_token_ids": generated_ids,
                    "prediction": prediction,
                    "finish_reason": (
                        "eos"
                        if generated_ids and generated_ids[-1] == 2
                        else "max_new_tokens"
                    ),
                    "human_answers": row["answers"],
                    **metrics,
                }
                generations_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                branch_rows.append(record)
            branch_report = {
                "checkpoint_sha256": _hash(run / branch / "final-state.npz"),
                "checkpoint_identity_sha256": hashlib.sha256(
                    (run / branch / "identity.json").read_bytes()
                ).hexdigest(),
                "candidate_source_sha256": identity["candidate_source_sha256"],
                "general_wikitext_nll": wiki_metrics,
                "domain_squad_context_nll": domain_metrics,
                "custom_squad_qa": {
                    "questions": len(scored),
                    "articles": len({row["article_title"] for row in final_rows}),
                    "exact_match": float(
                        np.mean([item["exact_match"] for item in scored])
                    ),
                    "token_f1": float(np.mean([item["token_f1"] for item in scored])),
                    "per_question": [
                        {"question_id": record["question_id"], **score}
                        for record, score in zip(branch_rows, scored, strict=True)
                    ],
                },
                "adapter_parameters": (
                    model.lora_parameter_count(config) if adapters is not None else 0
                ),
                "full_parameters": model.parameter_count(config),
                "merged_lora_evaluated": adapters is not None,
            }
            report["branches"][branch] = branch_report
    report["general_retention_comparison"] = {
        "domain_mixture_minus_matched_general_control_nats_per_token": report[
            "branches"
        ]["continuation_mixture"]["general_wikitext_nll"]["nll_nats_per_token"]
        - report["branches"]["continuation_general_control"]["general_wikitext_nll"][
            "nll_nats_per_token"
        ],
        "domain_full_sft_minus_general_parent_full_sft_nats_per_token": report[
            "branches"
        ]["sft_full_from_domain"]["general_wikitext_nll"]["nll_nats_per_token"]
        - report["branches"]["sft_full_from_general"]["general_wikitext_nll"][
            "nll_nats_per_token"
        ],
        "precommitted_regression_floor_nats_per_token": recipe[
            "precommitted_quality_floors"
        ][
            "maximum_general_retention_regression_nats_per_valid_token_vs_matched_general_control"
        ],
    }
    report["quality_floors"] = {
        "qa_floor": recipe["precommitted_quality_floors"],
        "no_branch_selected_by_final_score": True,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {"metrics": output / "metrics.json", "generations": generations_path},
            default=str,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.data, args.run, args.output)


if __name__ == "__main__":
    main()
