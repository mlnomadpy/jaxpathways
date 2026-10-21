#!/usr/bin/env python3
"""Staged independent checks for a learner's C02 decoder candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "data"))
sys.path.insert(0, str(PROJECT / "tests"))


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("model", path.resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["model"] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_equal(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            tree_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if left is None:
        return right is None
    if isinstance(left, list):
        return left == right
    if isinstance(left, (int, float, str, bool)):
        return left == right
    return np.array_equal(np.asarray(left), np.asarray(right))


def numpy_greedy(candidate, params, prompt, config, maximum, eos_token_id=2):
    """Greedy decode using the independent NumPy full-forward oracle each step."""
    from test_model_math import numpy_reference

    context = [int(value) for value in prompt]
    generated = []
    for _ in range(maximum):
        input_ids = np.asarray([context], dtype=np.int32)
        valid = np.ones_like(input_ids, dtype=bool)
        segments = np.ones_like(input_ids, dtype=np.int32)
        positions = np.broadcast_to(np.arange(len(context)), input_ids.shape)
        scores = numpy_reference(params, input_ids, config, valid, segments, positions)[
            0, -1
        ]
        scores = np.asarray(scores).copy()
        scores[0] = -np.inf
        token = int(np.argmax(scores))
        generated.append(token)
        if token == eos_token_id:
            break
        if len(context) >= config.context:
            if len(generated) < maximum:
                raise ValueError("generation request exceeds absolute context capacity")
            break
        if len(generated) < maximum:
            context.append(token)
    return generated


def check_model(candidate):
    from config import ModelConfig
    from test_model_math import numpy_reference

    config = ModelConfig(
        vocab_size=24,
        context=16,
        width=8,
        layers=2,
        heads=2,
        mlp_width=12,
        dropout=0.0,
    )
    params = candidate.initialize(jax.random.PRNGKey(91), config)
    actual_count = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(params))
    if candidate.parameter_count(config) != actual_count:
        raise AssertionError(
            "parameter formula differs from initialized candidate tree"
        )
    tokens = np.asarray([[1, 4, 5, 6, 7]], dtype=np.int32)
    valid = np.ones_like(tokens, dtype=bool)
    segments = np.asarray([[1, 1, 2, 2, 2]], dtype=np.int32)
    positions = np.asarray([[0, 1, 0, 1, 2]], dtype=np.int32)
    actual = candidate.logits(
        params,
        tokens,
        config,
        valid_mask=valid,
        segment_ids=segments,
        position_ids=positions,
    )
    expected = numpy_reference(params, tokens, config, valid, segments, positions)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    changed_prior = tokens.copy()
    changed_prior[0, :2] = (9, 10)
    altered = candidate.logits(
        params,
        changed_prior,
        config,
        valid_mask=valid,
        segment_ids=segments,
        position_ids=positions,
    )
    np.testing.assert_allclose(
        np.asarray(actual)[:, 2:], np.asarray(altered)[:, 2:], atol=1e-6
    )

    tokens_with_pad = jnp.asarray([[1, 4, 2, 1, 5, 2, 0]], dtype=jnp.int32)
    segment_ids = jnp.asarray([[1, 1, 1, 2, 2, 2, 0]], dtype=jnp.int32)
    _, targets, _, target_mask = candidate.shifted_causal_targets(
        tokens_with_pad, segment_ids
    )
    np.testing.assert_array_equal(np.asarray(targets), [[4, 2, 1, 5, 2, 0]])
    np.testing.assert_array_equal(
        np.asarray(target_mask), [[True, True, False, True, True, False]]
    )
    numerator, count = candidate.token_loss(
        params, tokens_with_pad, segment_ids, config
    )
    if not np.isfinite(float(numerator)) or float(count) != 4.0:
        raise AssertionError(
            "candidate target mask/loss does not preserve valid target ownership"
        )

    prompt = [1, 4, 5]
    _, cache = candidate.prefill(params, prompt, config)
    for token in (6, 7):
        full = candidate.logits(params, jnp.asarray([prompt + [token]]), config)[0, -1]
        cached, cache = candidate.decode_step(params, token, cache, config)
        np.testing.assert_allclose(cached[0], full, rtol=1e-5, atol=1e-5)
        prompt.append(token)
        independent = numpy_reference(
            params,
            np.asarray([prompt], dtype=np.int32),
            config,
            np.ones((1, len(prompt)), dtype=bool),
            np.ones((1, len(prompt)), dtype=np.int32),
            np.arange(len(prompt))[None, :],
        )[0, -1]
        np.testing.assert_allclose(cached[0], independent, rtol=2e-5, atol=2e-5)

    # The generation result is graded against independent full-forward greedy
    # decoding, not just a routing sentinel. Cover distinct prompts, token
    # budgets, a forced EOS and the exact model-context boundary.
    decode_prompts = ([1, 4, 5], [1, 9, 10, 11])
    for generation_prompt in decode_prompts:
        for maximum in (0, 1, 4):
            expected = numpy_greedy(
                candidate, params, generation_prompt, config, maximum
            )
            actual = candidate.generate(
                params, generation_prompt, config, maximum, eos_token_id=2
            )
            if actual != expected:
                raise AssertionError(
                    "candidate generation differs from independent greedy full-forward tokens"
                )
            if len(actual) > maximum:
                raise AssertionError("candidate generation exceeded its token budget")
    eos_probe = [1, 4, 5]
    first_logits = numpy_reference(
        params,
        np.asarray([eos_probe], dtype=np.int32),
        config,
        np.ones((1, len(eos_probe)), dtype=bool),
        np.ones((1, len(eos_probe)), dtype=np.int32),
        np.arange(len(eos_probe))[None, :],
    )[0, -1]
    first_logits = np.asarray(first_logits).copy()
    first_logits[0] = -np.inf
    forced_eos = int(np.argmax(first_logits))
    eos_expected = numpy_greedy(candidate, params, eos_probe, config, 5, forced_eos)
    eos_actual = candidate.generate(
        params, eos_probe, config, 5, eos_token_id=forced_eos
    )
    if eos_actual != eos_expected or eos_actual != [forced_eos]:
        raise AssertionError("candidate generation did not stop at the forced EOS")
    boundary_prompt = [1, *range(4, 4 + config.context - 2)]
    boundary_expected = numpy_greedy(candidate, params, boundary_prompt, config, 2)
    boundary_actual = candidate.generate(
        params, boundary_prompt, config, 2, eos_token_id=2
    )
    if boundary_actual != boundary_expected or len(boundary_actual) != 2:
        raise AssertionError("candidate failed the exact-context token budget")
    full_context_prompt = [1, *range(4, 4 + config.context - 1)]
    exact_boundary = candidate.generate(
        params, full_context_prompt, config, 1, eos_token_id=2
    )
    if exact_boundary != numpy_greedy(
        candidate, params, full_context_prompt, config, 1
    ):
        raise AssertionError("candidate failed a full-context one-token budget")
    try:
        numpy_greedy(candidate, params, full_context_prompt, config, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("independent oracle accepted an impossible context budget")
    try:
        candidate.generate(params, full_context_prompt, config, 2, eos_token_id=2)
    except ValueError as error:
        if "context capacity" not in str(error):
            raise
    else:
        raise AssertionError("candidate accepted an impossible context budget")

    # Make candidate routing observable through the learner-facing training
    # and generation entry points instead of trusting a source hash alone.
    import training
    import evaluate_pilot

    training.model = candidate
    candidate_loss = candidate.aligned_token_loss

    def sentinel(*_args, **_kwargs):
        raise RuntimeError("candidate loss sentinel")

    candidate.aligned_token_loss = sentinel
    batch = {
        "input_ids": jnp.asarray([[1, 4, 5, 0]], dtype=jnp.int32),
        "target_ids": jnp.asarray([[4, 5, 0, 0]], dtype=jnp.int32),
        "target_mask": jnp.asarray([[True, True, False, False]]),
        "segment_ids": jnp.asarray([[1, 1, 0, 0]], dtype=jnp.int32),
        "position_ids": jnp.asarray([[0, 1, 0, 0]], dtype=jnp.int32),
    }
    try:
        try:
            training._numerator(
                params, params, batch, config, jax.random.PRNGKey(3), False
            )
        except RuntimeError as error:
            if str(error) != "candidate loss sentinel":
                raise
        else:
            raise AssertionError("training path bypassed the submitted candidate loss")
    finally:
        candidate.aligned_token_loss = candidate_loss

    candidate_generate = candidate.generate
    candidate.generate = sentinel
    evaluate_pilot.model = candidate
    try:
        try:
            evaluate_pilot.generate_with_candidate(params, [1, 4], config, 1)
        except RuntimeError as error:
            if str(error) != "candidate loss sentinel":
                raise
        else:
            raise AssertionError(
                "evaluation generation bypassed the submitted candidate"
            )
    finally:
        candidate.generate = candidate_generate
    print("model: independent NumPy, masks, cache and candidate-routing checks passed")


def check_data(data_dir: Path):
    import prepare

    prepare.verify_prepared(data_dir, require_project_lock=True)
    recipe = json.loads((data_dir / "frozen_recipe.json").read_text())
    if recipe["tokenizer"]["processor_sha256"] != sha256(data_dir / "tokenizer.json"):
        raise AssertionError("prepared tokenizer differs from frozen recipe")
    print("data: strict raw-source, prepared-array, schedule and tokenizer lock passed")


def check_resume(candidate, data_dir: Path, run_dir: Path):
    import training

    training.model = candidate
    accepted = 0
    for branch in (
        "base",
        "continuation_mixture",
        "continuation_general_control",
        "sft_full_from_domain",
        "sft_lora_from_domain",
        "sft_full_from_general",
    ):
        stage_dir = run_dir / branch
        identity = json.loads((stage_dir / "identity.json").read_text())
        expected = training.load_checkpoint(
            stage_dir / "final-state.npz",
            identity,
            expected_order=list(range(len(identity["schedule_sample_ids"]))),
            expected_seen_ids=identity["schedule_sample_ids"],
        )
        with tempfile.TemporaryDirectory(prefix="c02-candidate-replay-") as temp:
            child = Path(temp) / f"{branch}.npz"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT / "tests" / "replay_branch.py"),
                    str(data_dir),
                    str(stage_dir),
                    str(child),
                    str(Path(candidate.__file__).resolve()),
                ],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "JAX_PLATFORMS": "cpu"},
            )
            actual = training.load_checkpoint(child, identity)
        if not tree_equal(expected, actual):
            raise AssertionError(f"candidate fresh-process replay differs in {branch}")
        accepted += 1
    print(
        f"resume: candidate-bound fresh-process final updates matched all {accepted} branches"
    )


def _cold_package_generation(package_dir: Path, prompt: str, maximum: int = 3):
    code = (
        "import json,sys; from consumer import load_package; "
        "p=load_package(sys.argv[1]); r=p.generate(sys.argv[2], int(sys.argv[3])); "
        "print(json.dumps({'token_ids':r['token_ids'],'finish_reason':r['finish_reason'],"
        "'prompt_tokens':r['prompt_tokens']}))"
    )
    child_env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    child_env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", code, str(package_dir), prompt, str(maximum)],
        cwd=package_dir,
        check=True,
        capture_output=True,
        text=True,
        env=child_env,
    )
    return json.loads(completed.stdout.splitlines()[-1])


def verify_qa_aggregate(branch, raw_scores, reported, expected_ids, article_count):
    per_question = {row["question_id"]: row for row in reported.get("per_question", [])}
    if set(per_question) != set(expected_ids):
        raise AssertionError(f"aggregate per-question IDs differ for {branch}")
    if set(raw_scores) != set(expected_ids):
        raise AssertionError(f"raw score IDs differ for {branch}")
    for question_id, scores in raw_scores.items():
        if any(
            per_question[question_id].get(key) != value for key, value in scores.items()
        ):
            raise AssertionError(
                f"reported per-question score differs for {branch}/{question_id}"
            )
    exact = float(np.mean([item["exact_match"] for item in raw_scores.values()]))
    f1 = float(np.mean([item["token_f1"] for item in raw_scores.values()]))
    if (
        reported.get("questions") != len(expected_ids)
        or reported.get("articles") != article_count
    ):
        raise AssertionError(f"QA denominator differs for {branch}")
    if not np.isclose(reported.get("exact_match"), exact, rtol=0, atol=1e-12):
        raise AssertionError(
            f"QA exact match is not reconstructed from raw rows: {branch}"
        )
    if not np.isclose(reported.get("token_f1"), f1, rtol=0, atol=1e-12):
        raise AssertionError(
            f"QA token F1 is not reconstructed from raw rows: {branch}"
        )
    return {"questions": len(expected_ids), "exact_match": exact, "token_f1": f1}


def check_evidence(
    candidate,
    data_dir: Path,
    run_dir: Path,
    evaluation_dir: Path,
    package_dir: Path,
):
    import prepare
    from config import ModelConfig
    from export_consumer import _candidate_hash as package_source_hash, export
    from run_pilot import _candidate_hash as trained_source_hash
    from squad_scoring import score
    from tokenizers import Tokenizer
    import training

    data, run, evaluation, package = map(
        Path.resolve, (data_dir, run_dir, evaluation_dir, package_dir)
    )
    prepare.verify_prepared(data, require_project_lock=True)
    metrics_path = evaluation / "metrics.json"
    generations_path = evaluation / "qa-generations.jsonl"
    metrics = json.loads(metrics_path.read_text())
    generations = [
        json.loads(line)
        for line in generations_path.read_text().splitlines()
        if line.strip()
    ]
    recipe = json.loads((data / "frozen_recipe.json").read_text())
    tokenizer_path = data / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    final_rows = [
        json.loads(line)
        for line in (data / "qa-final.jsonl").read_text().splitlines()
        if line.strip()
    ]
    expected_ids = [row["question_id"] for row in final_rows]
    expected_by_id = {row["question_id"]: row for row in final_rows}
    if len(expected_ids) != 128 or len(set(expected_ids)) != len(expected_ids):
        raise AssertionError("frozen QA set must have 128 unique question IDs")
    if set(metrics.get("branches", {})) != set(
        (
            "base",
            "continuation_mixture",
            "continuation_general_control",
            "sft_full_from_domain",
            "sft_lora_from_domain",
            "sft_full_from_general",
        )
    ):
        raise AssertionError("reloaded evaluation is missing a frozen branch")
    if metrics.get("recipe_sha256") != sha256(data / "frozen_recipe.json"):
        raise AssertionError("evaluation recipe identity differs from frozen data")
    if metrics.get("tokenizer_sha256") != sha256(tokenizer_path):
        raise AssertionError("evaluation tokenizer identity differs from frozen data")
    if metrics.get("evaluator_source_sha256") != sha256(PROJECT / "evaluate_pilot.py"):
        raise AssertionError("evaluation source identity is stale")
    if metrics.get("scorer", {}).get("reference_bytes_sha256") != sha256(
        PROJECT / "squad_scoring.py"
    ):
        raise AssertionError("evaluation scorer identity is stale")

    expected_pairs = {
        (branch, question_id)
        for branch in metrics["branches"]
        for question_id in expected_ids
    }
    if len(generations) != len(expected_pairs):
        raise AssertionError("raw final QA generations do not cover 6 x 128 rows")
    observed_pairs = set()
    per_branch = {branch: {} for branch in metrics["branches"]}
    for record in generations:
        branch, question_id = record.get("branch"), record.get("question_id")
        pair = (branch, question_id)
        if pair not in expected_pairs or pair in observed_pairs:
            raise AssertionError("raw QA generations have an unknown or duplicate ID")
        observed_pairs.add(pair)
        source = expected_by_id[question_id]
        expected_prompt = prepare.qa_prompt(source["question"], source["context"])
        if record.get("prompt") != expected_prompt:
            raise AssertionError(
                f"raw prompt differs from frozen question {question_id}"
            )
        prompt_ids = [
            1,
            *tokenizer.encode(expected_prompt, add_special_tokens=False).ids,
        ]
        if record.get("prompt_token_ids") != prompt_ids:
            raise AssertionError(f"raw prompt token IDs differ for {question_id}")
        if record.get("human_answers") != source["answers"]:
            raise AssertionError(f"raw answer references differ for {question_id}")
        generated = record.get("generated_token_ids")
        if (
            not isinstance(generated, list)
            or len(generated) > 32
            or any(
                type(token) is not int or not 1 <= token < tokenizer.get_vocab_size()
                for token in generated
            )
        ):
            raise AssertionError(f"invalid generated token IDs for {question_id}")
        if len(prompt_ids) + 32 > recipe["pilot_schedule"]["architecture"]["context"]:
            raise AssertionError(
                f"frozen prompt no longer reserves the generation cap: {question_id}"
            )
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        if record.get("prediction") != decoded:
            raise AssertionError(
                f"decoded prediction differs from raw IDs for {question_id}"
            )
        if record.get("finish_reason") == "eos":
            if not generated or generated[-1] != 2:
                raise AssertionError(
                    f"EOS finish reason is inconsistent for {question_id}"
                )
        elif record.get("finish_reason") == "max_new_tokens":
            if len(generated) != 32 or 2 in generated:
                raise AssertionError(
                    f"token-budget finish reason is inconsistent for {question_id}"
                )
        else:
            raise AssertionError(f"unknown generation finish reason for {question_id}")
        independent = score(decoded, source["answers"])
        if any(record.get(key) != value for key, value in independent.items()):
            raise AssertionError(
                f"raw per-question SQuAD score is fabricated for {question_id}"
            )
        per_branch[branch][question_id] = independent

    if observed_pairs != expected_pairs:
        raise AssertionError("raw QA question/branch matrix is incomplete")

    for branch, records in per_branch.items():
        branch_report = metrics["branches"][branch]
        stage = run / branch
        identity_path = stage / "identity.json"
        checkpoint_path = stage / "final-state.npz"
        identity = json.loads(identity_path.read_text())
        if branch_report.get("checkpoint_sha256") != sha256(checkpoint_path):
            raise AssertionError(f"evaluation checkpoint binding differs for {branch}")
        if branch_report.get("checkpoint_identity_sha256") != sha256(identity_path):
            raise AssertionError(f"evaluation checkpoint identity differs for {branch}")
        if (
            branch_report.get("candidate_source_sha256")
            != identity.get("candidate_source_sha256")
            or identity.get("candidate_source_sha256") != trained_source_hash()
        ):
            raise AssertionError(f"evaluation candidate binding differs for {branch}")
        if (
            identity.get("recipe_sha256") != metrics["recipe_sha256"]
            or identity.get("processor_sha256") != metrics["tokenizer_sha256"]
        ):
            raise AssertionError(
                f"evaluation recipe/tokenizer binding differs for {branch}"
            )
        verify_qa_aggregate(
            branch,
            records,
            branch_report["custom_squad_qa"],
            expected_ids,
            len({row["article_title"] for row in final_rows}),
        )

    floors = recipe["precommitted_quality_floors"]
    if metrics.get("quality_floors", {}).get("qa_floor") != floors:
        raise AssertionError("reported quality floors differ from frozen recipe")
    mixture_nll = metrics["branches"]["continuation_mixture"]["general_wikitext_nll"]
    control_nll = metrics["branches"]["continuation_general_control"][
        "general_wikitext_nll"
    ]
    for value in (mixture_nll, control_nll):
        if value["valid_token_targets"] <= 0 or not np.isclose(
            value["nll_nats_per_token"],
            value["nll_sum_nats"] / value["valid_token_targets"],
            rtol=0,
            atol=1e-12,
        ):
            raise AssertionError("general retention NLL denominator is inconsistent")
    retention = mixture_nll["nll_nats_per_token"] - control_nll["nll_nats_per_token"]
    stored_retention = metrics["general_retention_comparison"][
        "domain_mixture_minus_matched_general_control_nats_per_token"
    ]
    if not np.isclose(stored_retention, retention, rtol=0, atol=1e-12):
        raise AssertionError("general-retention comparison is not reconstructed")
    qa_pass = {}
    for branch, records in per_branch.items():
        exact = float(np.mean([item["exact_match"] for item in records.values()]))
        f1 = float(np.mean([item["token_f1"] for item in records.values()]))
        qa_pass[branch] = (
            exact >= floors["custom_qa_exact_match_minimum"]
            and f1 >= floors["custom_qa_token_f1_minimum"]
        )
    if any(qa_pass.values()):
        raise AssertionError(
            "the retained run's failed QA floor was unexpectedly promoted"
        )
    if (
        retention
        > floors[
            "maximum_general_retention_regression_nats_per_valid_token_vs_matched_general_control"
        ]
    ):
        raise AssertionError("retained continuation exceeds its frozen retention floor")

    # Check the supplied package in a genuinely cold interpreter and produce a
    # second package directly from the selected candidate source. This makes
    # export and consumer use of the same submitted decoder observable.
    package_manifest = json.loads((package / "manifest.json").read_text())
    branch = package_manifest.get("branch")
    if branch not in metrics["branches"]:
        raise AssertionError("cold consumer branch has no matching evaluation")
    branch_report = metrics["branches"][branch]
    if package_manifest.get("checkpoint_sha256") != branch_report["checkpoint_sha256"]:
        raise AssertionError("cold package checkpoint differs from evaluated branch")
    if (
        package_manifest.get("checkpoint_identity_sha256")
        != branch_report["checkpoint_identity_sha256"]
    ):
        raise AssertionError("cold package identity differs from evaluated branch")
    if (
        package_manifest.get(
            "checkpoint_candidate_source_sha256",
            package_manifest.get("candidate_source_sha256"),
        )
        != branch_report["candidate_source_sha256"]
    ):
        raise AssertionError("cold package candidate lineage differs from evaluation")
    if (
        package_manifest.get("recipe_sha256") != metrics["recipe_sha256"]
        or package_manifest.get("tokenizer_sha256") != metrics["tokenizer_sha256"]
        or package_manifest.get("evaluation_metrics_sha256") != sha256(metrics_path)
    ):
        raise AssertionError("cold package recipe/tokenizer/evaluation binding differs")

    source = Path(candidate.__file__).resolve()
    prompt = "Question: Which country?\nContext: A short pilot context.\nAnswer:"
    prompt_ids = [1, *tokenizer.encode(prompt, add_special_tokens=False).ids]
    identity_path = run / branch / "identity.json"
    identity = json.loads(identity_path.read_text())
    training.model = candidate
    state = training.load_checkpoint(
        run / branch / "final-state.npz",
        identity,
        expected_order=list(range(len(identity["schedule_sample_ids"]))),
        expected_seen_ids=identity["schedule_sample_ids"],
    )
    params = state["params"]
    if state["adapters"] is not None:
        params = candidate.merge_lora(
            params, state["adapters"], ModelConfig(**identity["model_config"])
        )
    config = ModelConfig(**identity["model_config"])
    expected_ids = numpy_greedy(candidate, params, prompt_ids, config, 3)
    candidate_ids = candidate.generate(params, prompt_ids, config, 3, eos_token_id=2)
    if candidate_ids != expected_ids:
        raise AssertionError(
            "selected candidate generation differs from the independent oracle"
        )
    supplied_cold = _cold_package_generation(package, prompt, 3)
    if supplied_cold["token_ids"] != expected_ids:
        raise AssertionError(
            "cold exported package generation differs from candidate oracle"
        )

    with tempfile.TemporaryDirectory(prefix="c02-candidate-export-") as directory:
        candidate_package = Path(directory) / "package"
        export(
            data,
            run,
            evaluation,
            branch,
            candidate_package,
            model_module=candidate,
            model_source=source,
        )
        candidate_manifest = json.loads(
            (candidate_package / "manifest.json").read_text()
        )
        if sha256(candidate_package / "model.py") != sha256(source):
            raise AssertionError("export did not copy the selected candidate source")
        if candidate_manifest.get("candidate_source_sha256") != package_source_hash(
            source
        ):
            raise AssertionError(
                "export candidate identity does not bind its copied source"
            )
        if (
            candidate_manifest.get("checkpoint_candidate_source_sha256")
            != branch_report["candidate_source_sha256"]
        ):
            raise AssertionError(
                "export lost the checkpoint's training candidate lineage"
            )
        candidate_cold = _cold_package_generation(candidate_package, prompt, 3)
        if candidate_cold["token_ids"] != expected_ids:
            raise AssertionError(
                "cold consumer bypassed the selected exported candidate"
            )

    print(
        "evaluation/consumer: raw 6x128 scores, source/checkpoint bindings, retention arithmetic, selected-candidate export and cold generation passed"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("model", "data", "resume", "all"), required=True
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--package", type=Path)
    args = parser.parse_args()
    candidate = load_candidate(args.implementation)
    if args.stage in {"model", "all"}:
        check_model(candidate)
    if args.stage in {"data", "all"}:
        if args.data is None:
            parser.error("--data is required for data and all stages")
        check_data(args.data.resolve())
    if args.stage in {"resume", "all"}:
        if args.data is None or args.run is None:
            parser.error("--data and --run are required for resume and all stages")
        check_resume(candidate, args.data.resolve(), args.run.resolve())
    if args.stage == "all":
        if args.evaluation is None or args.package is None:
            parser.error("--evaluation and --package are required for the all stage")
        if args.run is None:
            parser.error("--run is required for evidence and candidate package checks")
        check_evidence(
            candidate,
            args.data.resolve(),
            args.run.resolve(),
            args.evaluation.resolve(),
            args.package.resolve(),
        )


if __name__ == "__main__":
    main()
