"""Real-data decoder checks, fresh-process resume and inference-package canaries."""

import argparse
import hashlib
import importlib.util
import json
import os
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jax
import jax.numpy as jnp
import numpy as np

FOUNDATIONS = Path(__file__).resolve().parents[1]
ROOT = FOUNDATIONS.parent
sys.path.insert(0, str(FOUNDATIONS))
import data
import train
from config import ModelConfig, TrainConfig


def load_implementation(path):
    spec = importlib.util.spec_from_file_location("model", Path(path).resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules["model"] = module
    spec.loader.exec_module(module)
    return module


def require_rejects(call, message):
    try:
        call()
    except (ValueError, OSError, json.JSONDecodeError):
        return
    raise AssertionError(message)


def require_raises_type(call, exception_type, message):
    try:
        call()
    except exception_type:
        return
    raise AssertionError(message)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_candidate(candidate):
    """Make every training utility resolve the selected model module."""
    train.model = candidate
    train._compiled_gradient.cache_clear()


def dataset_hash(examples):
    digest = hashlib.sha256()
    for name in ("tokens", "segments", "loss_mask", "text_mask", "document_index"):
        array = np.ascontiguousarray(examples[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    digest.update(
        json.dumps(examples["stats"]["record_ids"], separators=(",", ":")).encode()
    )
    return digest.hexdigest()


def compare_states(left, right):
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    assert len(left_leaves) == len(right_leaves)
    for a, b in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def child_resume(args, module, examples, model_config, train_config, identity):
    template = train.initialize_training(examples, model_config, train_config)
    state, _ = train.load_checkpoint(args.resume_worker, template, identity)
    state, item = train.step(state, examples, model_config, train_config)
    train.save_checkpoint(args.worker_output, state, identity)
    Path(args.worker_output, "trace.json").write_text(json.dumps(item, sort_keys=True))
    raise SystemExit(0)


def child_package(args, module, identity):
    params, cfg, _ = train.load_model_package(args.reload_worker, identity)
    from tokenizers import Tokenizer

    processor = Tokenizer.from_file(str(Path(args.reload_worker) / "tokenizer.json"))
    prompt = json.loads(args.prompt_ids)
    first, cache = module.prefill(params, prompt, cfg)
    next_logits, next_cache = module.decode_step(params, int(args.next_id), cache, cfg)
    full = module.logits(
        params, jnp.asarray([prompt + [int(args.next_id)]], jnp.int32), cfg
    )[0, -1]
    np.testing.assert_allclose(next_logits, full, atol=2e-5, rtol=2e-5)
    np.savez(
        Path(args.worker_output) / "canary.npz",
        first=np.asarray(first),
        next=np.asarray(next_logits),
        decoded=np.asarray(processor.decode(prompt)),
    )
    Path(args.worker_output, "package.json").write_text(
        json.dumps({"processor_size": processor.get_vocab_size()})
    )
    raise SystemExit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", default=str(FOUNDATIONS / "model.py"))
    parser.add_argument(
        "--output", default="/private/tmp/jaxpathways-text-foundations-w2"
    )
    parser.add_argument("--stage", choices=("all",), default="all")
    parser.add_argument("--resume-worker")
    parser.add_argument("--reload-worker")
    parser.add_argument("--worker-output")
    parser.add_argument("--prompt-ids", default="[]")
    parser.add_argument("--next-id", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    module = load_implementation(args.implementation)
    bind_candidate(module)
    candidate_hash = sha256_file(args.implementation)

    records, source_manifest = data.load_records()
    train_records = [row for row in records if row.split == "train"]
    dev_records = [row for row in records if row.split == "validation"]
    test_records = [row for row in records if row.split == "test"]
    assert {row.chapter for row in train_records} == {"I", "II", "III", "IV"}
    assert {row.chapter for row in dev_records} == {"V"}
    assert {row.chapter for row in test_records} == {"VI", "VII", "VIII"}
    for rows in (train_records, dev_records, test_records):
        assert len({row.text for row in rows}) == len(
            rows
        ), "deduplication must precede tokenization"
    groups = [
        set(row.chapter for row in rows)
        for rows in (train_records, dev_records, test_records)
    ]
    assert (
        not groups[0] & groups[1]
        and not groups[0] & groups[2]
        and not groups[1] & groups[2]
    )
    assert (
        hashlib.sha256(data.SOURCE.read_bytes()).hexdigest()
        == source_manifest["source_sha256"]
    )
    assert b"Project Gutenberg License" in data.SOURCE.read_bytes()

    tokenizer_path = out / "tokenizer.json"
    info_path = out / "tokenizer-info.json"
    if args.resume_worker or args.reload_worker:
        from tokenizers import Tokenizer

        processor = Tokenizer.from_file(str(tokenizer_path))
        tokenizer_info = json.loads(info_path.read_text())
    else:
        processor, tokenizer_info = data.train_bpe(
            records, vocab_size=384, save_path=tokenizer_path
        )
        info_path.write_text(json.dumps(tokenizer_info, sort_keys=True, indent=2))

    assert tokenizer_info["trained_record_ids"] == [
        row.record_id for row in train_records
    ]
    assert set(tokenizer_info["trained_record_ids"]).isdisjoint(
        row.record_id for row in dev_records + test_records
    )
    assert [
        processor.token_to_id(token) for token in ("[PAD]", "[BOS]", "[EOS]", "[UNK]")
    ] == [0, 1, 2, 3]
    unknown_text = "café 🧪 Ω — byte fallback works"
    bpe_ids = data.encode_text(processor, unknown_text)
    assert data.decode_text(processor, bpe_ids) == unknown_text
    byte = data.ByteTokenizer()
    assert byte.decode(byte.encode(unknown_text)) == unknown_text
    assert (
        tokenizer_info["processor_sha256"]
        == hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    )

    model_config = ModelConfig(
        vocab_size=processor.get_vocab_size(),
        context=24,
        width=16,
        layers=2,
        heads=4,
        mlp_width=32,
        dropout=0.1,
        tied_embeddings=True,
    )
    train_config = TrainConfig(
        batch_size=4,
        learning_rate=0.001,
        min_learning_rate=0.0003,
        warmup_steps=2,
        total_steps=6,
        seed=20261020,
    )
    model_config.validate()
    train_config.validate()
    train_examples = data.pack_records(train_records, processor, model_config.context)
    dev_examples = data.pack_records(dev_records, processor, model_config.context)
    test_examples = data.pack_records(test_records, processor, model_config.context)
    train_examples["stats"]["dataset_sha256"] = dataset_hash(train_examples)
    for examples in (train_examples, dev_examples, test_examples):
        examples["stats"]["source_sha256"] = source_manifest["source_sha256"]
        assert examples["stats"]["tokens_truncated"] == 0
        assert examples["stats"]["tokens_exposed"] == examples["stats"]["text_tokens"]
    identity = train.experiment_identity(
        source_manifest,
        tokenizer_info,
        train_examples["stats"],
        model_config,
        train_config,
        candidate_model_sha256=candidate_hash,
    )

    if args.resume_worker:
        if not args.worker_output:
            raise ValueError("--worker-output is required for a resume worker")
        child_resume(args, module, train_examples, model_config, train_config, identity)
    if args.reload_worker:
        if not args.worker_output:
            raise ValueError("--worker-output is required for a package worker")
        child_package(args, module, identity)

    # Config changes are real model variations, not metadata-only toggles.
    changed_config = ModelConfig(
        vocab_size=model_config.vocab_size,
        context=24,
        width=24,
        layers=3,
        heads=4,
        mlp_width=48,
        dropout=0.0,
        tied_embeddings=False,
    )
    changed_params = module.initialize(jax.random.key(99), changed_config)
    assert module.parameter_count(changed_config) == sum(
        int(np.prod(x.shape)) for x in jax.tree.leaves(changed_params)
    )
    changed_logits = module.logits(
        changed_params, jnp.asarray([[1, 30, 2]], jnp.int32), changed_config
    )
    assert changed_logits.shape == (1, 3, changed_config.vocab_size)

    params = module.initialize(jax.random.key(17), model_config)
    count = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(params))
    assert count == module.parameter_count(model_config)
    assert len(params["blocks"]) == 2
    assert all(np.asarray(leaf).dtype == np.float32 for leaf in jax.tree.leaves(params))

    # Independent NumPy token-NLL/count metric on a fresh held-out subset.
    metric_fixture = data.pack_records(
        dev_records[:12], processor, model_config.context
    )
    measured = train.evaluate(
        params, metric_fixture, model_config, batch_size=64, text_only=True
    )
    metric_inputs = jnp.asarray(metric_fixture["inputs"])
    metric_logits = np.asarray(module.logits(params, metric_inputs, model_config))
    metric_targets = metric_fixture["targets"]
    metric_mask = metric_fixture["text_mask"]
    row_max = metric_logits.max(axis=-1, keepdims=True)
    log_partition = row_max + np.log(
        np.exp(metric_logits - row_max).sum(axis=-1, keepdims=True)
    )
    chosen = np.take_along_axis(metric_logits, metric_targets[..., None], axis=-1)[
        ..., 0
    ]
    oracle_nll = float(
        np.sum(np.where(metric_mask, log_partition[..., 0] - chosen, 0.0))
    )
    oracle_count = int(metric_mask.sum())
    assert measured["tokens"] == oracle_count
    np.testing.assert_allclose(measured["nll_nats"], oracle_nll, atol=2e-3, rtol=2e-6)

    # Independent NumPy attention oracle, including a completely blocked row.
    rng = np.random.default_rng(91)
    q = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
    k = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
    v = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
    allowed = np.asarray([[[[1, 0, 0], [1, 1, 0], [0, 0, 0]]]], dtype=bool)
    actual, weights = module.scaled_dot_attention(
        jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), jnp.asarray(allowed)
    )
    oracle_weights = np.zeros((1, 2, 3, 3), dtype=np.float32)
    oracle_output = np.zeros((1, 2, 3, 4), dtype=np.float32)
    for batch in range(1):
        for head in range(2):
            for query in range(3):
                selected = np.flatnonzero(allowed[batch, 0, query])
                if not len(selected):
                    continue
                scores = k[batch, head, selected] @ q[batch, head, query] / np.sqrt(4.0)
                probabilities = np.exp(scores - np.max(scores))
                probabilities /= probabilities.sum()
                oracle_weights[batch, head, query, selected] = probabilities
                oracle_output[batch, head, query] = (
                    probabilities @ v[batch, head, selected]
                )
    np.testing.assert_allclose(weights, oracle_weights, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(actual, oracle_output, atol=2e-6, rtol=2e-6)
    assert np.all(np.isfinite(actual)) and np.all(np.asarray(weights)[:, :, 2] == 0)

    # Two packed documents prove both attention isolation and shifted-loss
    # eligibility against independent NumPy oracles.
    toy = jnp.asarray([[1, 31, 32, 1, 42, 43, 0]], jnp.int32)
    seg = jnp.asarray([[1, 1, 1, 2, 2, 2, 0]], jnp.int32)
    original, attention = module.logits(
        params, toy, model_config, segment_ids=seg, return_attention=True
    )
    changed_first_document = toy.at[:, :3].set(jnp.asarray([[1, 88, 89]]))
    changed = module.logits(
        params, changed_first_document, model_config, segment_ids=seg
    )
    np.testing.assert_allclose(original[:, 3:6], changed[:, 3:6], atol=1e-6, rtol=1e-6)
    assert len(attention) == model_config.layers
    attention0 = np.asarray(attention[0])[0]
    assert np.all(np.triu(attention0[:, :6, :6], k=1) == 0)
    assert np.all(attention0[:, 3:6, :3] == 0), "attention crossed a document boundary"
    _, targets, _, eligible = module.causal_targets(toy, seg)
    target_ids = np.asarray(targets)
    seg_np = np.asarray(seg)
    oracle_eligible = (
        (target_ids != 0) & (seg_np[:, :-1] == seg_np[:, 1:]) & (seg_np[:, :-1] > 0)
    )
    np.testing.assert_array_equal(np.asarray(eligible), oracle_eligible)
    assert oracle_eligible[0, 1] and not oracle_eligible[0, 2]
    logits_for_loss = np.asarray(original[:, :-1])
    stable = logits_for_loss - logits_for_loss.max(axis=-1, keepdims=True)
    log_partition = np.log(np.exp(stable).sum(axis=-1))
    chosen = np.take_along_axis(stable, target_ids[..., None], axis=-1)[..., 0]
    independent_loss = (
        -np.where(oracle_eligible, chosen - log_partition, 0).sum()
        / oracle_eligible.sum()
    )
    np.testing.assert_allclose(
        module.token_loss(params, toy, seg, model_config),
        independent_loss,
        atol=2e-5,
        rtol=2e-5,
    )
    target_valid_only = target_ids != 0
    boundary_unmasked_loss = (
        -np.where(target_valid_only, chosen - log_partition, 0.0).sum()
        / target_valid_only.sum()
    )
    assert not np.isclose(
        independent_loss, boundary_unmasked_loss, atol=1e-6
    ), "cross-segment shifted target was not distinguishable in this fixture"
    # Padding remains inert even when the padded token IDs are changed.
    changed_padding = toy.at[:, 6].set(3)
    padded_change = module.logits(
        params, changed_padding, model_config, segment_ids=seg
    )
    np.testing.assert_allclose(
        original[:, :6], padded_change[:, :6], atol=1e-6, rtol=1e-6
    )
    generated = module.generate(params, [1, 31, 32], model_config, max_new_tokens=3)
    assert len(generated) <= 3 and all(token not in (0, 1) for token in generated)
    all_pad_loss = module.token_loss(
        params, jnp.zeros((2, 5), jnp.int32), jnp.zeros((2, 5), jnp.int32), model_config
    )
    assert np.isfinite(float(all_pad_loss)) and float(all_pad_loss) == 0.0

    # Deterministic iterator coverage, including a partial tail that rolls into
    # the next epoch. The expected order is built independently from its stored
    # document permutation and the fixture's document membership.
    coverage_cfg = ModelConfig(
        vocab_size=260, context=4, width=4, layers=1, heads=1, mlp_width=4
    )
    coverage_train_cfg = TrainConfig(batch_size=4, total_steps=4, seed=13)
    coverage_examples = {
        "tokens": np.tile(np.asarray([[1, 5, 2, 0, 0]], np.int32), (9, 1)),
        "segments": np.tile(np.asarray([[1, 1, 1, 0, 0]], np.int32), (9, 1)),
        "document_index": np.asarray([0, 0, 0, 1, 1, 2, 2, 2, 2], np.int32),
        "stats": {"record_ids": ["a", "b", "c"]},
    }
    coverage_state = train.initialize_training(
        coverage_examples, coverage_cfg, coverage_train_cfg
    )
    first_order = np.asarray(coverage_state["order"])
    seen_first_epoch = []
    for _ in range(2):
        coverage_state, _, chosen_indices, _ = train.take_batch(
            coverage_state, coverage_examples, coverage_train_cfg
        )
        seen_first_epoch.extend(chosen_indices.tolist())
    assert seen_first_epoch == first_order[:8].tolist()
    coverage_state, _, crossed_indices, _ = train.take_batch(
        coverage_state, coverage_examples, coverage_train_cfg
    )
    next_order = np.asarray(coverage_state["order"])
    expected_crossing = first_order[8:].tolist() + next_order[:3].tolist()
    assert crossed_indices.tolist() == expected_crossing
    assert int(coverage_state["epoch"]) == 1 and int(coverage_state["cursor"]) == 3
    assert sorted(seen_first_epoch + first_order[8:].tolist()) == list(
        range(9)
    ), "an epoch dropped or duplicated examples"
    assert len(next_order[:3]) == 3 and len(set(next_order[:3].tolist())) == 3
    coverage_identity = {"train_config_sha256": coverage_train_cfg.fingerprint()}
    coverage_checkpoint = out / "coverage-checkpoint"
    train.save_checkpoint(coverage_checkpoint, coverage_state, coverage_identity)
    coverage_template = train.initialize_training(
        coverage_examples, coverage_cfg, coverage_train_cfg
    )
    restored_coverage, _ = train.load_checkpoint(
        coverage_checkpoint, coverage_template, coverage_identity
    )
    continued, _, expected_next_indices, _ = train.take_batch(
        coverage_state, coverage_examples, coverage_train_cfg
    )
    resumed, _, actual_next_indices, _ = train.take_batch(
        restored_coverage, coverage_examples, coverage_train_cfg
    )
    np.testing.assert_array_equal(actual_next_indices, expected_next_indices)
    compare_states(continued, resumed)
    wrong_batch_identity = {
        "train_config_sha256": TrainConfig(
            batch_size=3, total_steps=4, seed=13
        ).fingerprint()
    }
    require_rejects(
        lambda: train.load_checkpoint(
            coverage_checkpoint, coverage_template, wrong_batch_identity
        ),
        "resume with a changed batch size must be rejected",
    )

    # Training and evaluation must dispatch through the selected candidate.
    original_token_loss, original_logits = module.token_loss, module.logits
    candidate_train_state = train.initialize_training(
        train_examples, model_config, train_config
    )
    module.token_loss = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("candidate token_loss reached")
    )
    train._compiled_gradient.cache_clear()
    require_raises_type(
        lambda: train.step(
            candidate_train_state, train_examples, model_config, train_config
        ),
        RuntimeError,
        "training utilities bypassed the selected model candidate",
    )
    module.token_loss = original_token_loss
    train._compiled_gradient.cache_clear()
    module.logits = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("candidate logits reached")
    )
    require_raises_type(
        lambda: train.evaluate(params, metric_fixture, model_config, batch_size=64),
        RuntimeError,
        "evaluation utilities bypassed the selected model candidate",
    )
    module.logits = original_logits
    train._compiled_gradient.cache_clear()

    # A fresh-process resume must be driven by this exact candidate source.
    # Identity carries its file hash in addition to the reusable project code IDs.
    assert identity["candidate_model_sha256"] == candidate_hash

    # Causal output independence and one-coordinate central-difference gradient check.
    prefix = jnp.asarray([[1, 41, 42, 43]], jnp.int32)
    future_changed = prefix.at[:, 3].set(88)
    np.testing.assert_allclose(
        module.logits(params, prefix, model_config)[:, :3],
        module.logits(params, future_changed, model_config)[:, :3],
        atol=1e-6,
        rtol=1e-6,
    )
    batch_tokens = jnp.asarray(train_examples["tokens"][:2, :13])
    batch_segments = jnp.asarray(train_examples["segments"][:2, :13])
    gradient = jax.grad(
        lambda p: module.token_loss(p, batch_tokens, batch_segments, model_config)
    )
    grad_tree = gradient(params)
    leaf = np.asarray(grad_tree["blocks"][1]["q"])
    coordinate = (0, 1)
    epsilon = 0.002
    plus = jax.tree.map(lambda value: value, params)
    minus = jax.tree.map(lambda value: value, params)
    plus_block = dict(plus["blocks"][1])
    minus_block = dict(minus["blocks"][1])
    plus_block["q"] = plus_block["q"].at[coordinate].add(epsilon)
    minus_block["q"] = minus_block["q"].at[coordinate].add(-epsilon)
    plus = {**plus, "blocks": (*plus["blocks"][:1], plus_block)}
    minus = {**minus, "blocks": (*minus["blocks"][:1], minus_block)}
    finite_difference = (
        module.token_loss(plus, batch_tokens, batch_segments, model_config)
        - module.token_loss(minus, batch_tokens, batch_segments, model_config)
    ) / (2 * epsilon)
    np.testing.assert_allclose(
        leaf[coordinate], finite_difference, atol=3e-3, rtol=0.08
    )

    # Every cache layer is compared with full recomputation after nonzero offsets.
    prompts = [
        [1, *data.encode_text(processor, "Alice")[:5]],
        [1, *data.encode_text(processor, "The rabbit")[:4]],
    ]
    for prompt in prompts:
        current_logits, cache = module.prefill(params, prompt, model_config)
        full_last = module.logits(params, jnp.asarray([prompt]), model_config)[0, -1]
        np.testing.assert_allclose(current_logits, full_last, atol=2e-5, rtol=2e-5)
        for token in [31, 77, 152]:
            current_logits, cache = module.decode_step(
                params, token, cache, model_config
            )
            prompt.append(token)
            full_last = module.logits(params, jnp.asarray([prompt]), model_config)[
                0, -1
            ]
            np.testing.assert_allclose(current_logits, full_last, atol=3e-5, rtol=3e-5)
    original_prompt = prompts[0][:6]
    _, baseline_cache = module.prefill(params, original_prompt, model_config)
    corrupted_cache = module.corrupt_cache_layer(baseline_cache, layer=1)
    clean_next, _ = module.decode_step(params, 61, baseline_cache, model_config)
    corrupt_next, _ = module.decode_step(params, 61, corrupted_cache, model_config)
    assert not np.allclose(
        clean_next, corrupt_next, atol=1e-5, rtol=1e-5
    ), "second-layer K/V corruption must alter output"
    require_rejects(
        lambda: module.decode_step(
            params,
            61,
            baseline_cache,
            ModelConfig(**{**model_config.as_dict(), "width": 32, "heads": 4}),
        ),
        "cache/config mismatch must fail",
    )
    require_rejects(
        lambda: module.decode_step(
            params,
            61,
            module.prefill(params, [1] * model_config.context, model_config)[1],
            model_config,
        ),
        "capacity must reject a further token",
    )

    if args.resume_worker or args.reload_worker:
        return
    # Train-only byte baseline and BPE metrics use matched selected held-out bytes.
    byte_baseline = data.byte_unigram_bpb(train_records, dev_records)
    state = train.initialize_training(train_examples, model_config, train_config)
    before_train = train.evaluate(
        state["params"], train_examples, model_config, text_only=True
    )
    before_dev = train.evaluate(
        state["params"], dev_examples, model_config, text_only=True
    )
    history = []
    for _ in range(6):
        state, item = train.step(state, train_examples, model_config, train_config)
        history.append(item)
    after_train = train.evaluate(
        state["params"], train_examples, model_config, text_only=True
    )
    after_dev = train.evaluate(
        state["params"], dev_examples, model_config, text_only=True
    )
    # Test is frozen until this one final evaluation; no recipe choice uses its result.
    final_test = train.evaluate(
        state["params"], test_examples, model_config, text_only=True
    )
    dev_bits_per_byte = (
        after_dev["nll_nats"]
        / np.log(2)
        / dev_examples["stats"]["normalized_paragraph_bytes"]
    )
    test_bits_per_byte = (
        final_test["nll_nats"]
        / np.log(2)
        / test_examples["stats"]["normalized_paragraph_bytes"]
    )
    report = {
        "source": {
            "title": source_manifest["title"],
            "edition": source_manifest["edition"],
            "source_sha256": source_manifest["source_sha256"],
            "paragraph_normalization": source_manifest["paragraph_normalization"],
            "rights": source_manifest["rights"],
            "chapter_splits": source_manifest["split_map"],
        },
        "tokenizer": tokenizer_info,
        "architecture": model_config.as_dict(),
        "changed_config_sanity": {
            "config": changed_config.as_dict(),
            "parameters": module.parameter_count(changed_config),
            "observed_logits_shape": list(changed_logits.shape),
            "forward_pass_finite": bool(
                np.all(np.isfinite(np.asarray(changed_logits)))
            ),
        },
        "parameters": count,
        "target_config_parameter_count_only": module.parameter_count(
            ModelConfig(
                vocab_size=32000,
                context=2048,
                width=1024,
                layers=21,
                heads=16,
                mlp_width=4096,
            )
        ),
        "training": {
            "config": train_config.as_dict(),
            "steps": history,
            "actual_updates": len(history),
            "precision": "FP32 CPU",
            "text_token_nll_before": before_train,
            "text_token_nll_after": after_train,
            "bits_per_normalized_paragraph_byte_after": after_train["nll_nats"]
            / np.log(2)
            / train_examples["stats"]["normalized_paragraph_bytes"],
        },
        "data": {
            "train": train_examples["stats"],
            "validation": dev_examples["stats"],
            "test": test_examples["stats"],
        },
        "metrics": {
            "dev_before": before_dev,
            "dev_after": after_dev,
            "dev_bits_per_normalized_paragraph_byte": dev_bits_per_byte,
            "final_test": final_test,
            "test_bits_per_normalized_paragraph_byte": test_bits_per_byte,
            "train_only_byte_unigram_dev": byte_baseline,
        },
        "interpretation": "Small one-book CPU pipeline pilot only; tokenizer NLL/perplexity is vocabulary-specific. Byte rates use UTF-8 bytes of whitespace-normalized selected paragraphs, not byte spans in the Gutenberg file. Test chapters were scored once after the recipe was frozen. No model-quality, 300M training, or accelerator claim.",
        "candidate_model_sha256": candidate_hash,
    }
    (out / "pilot-results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    # Fresh-process resume proves next samples, dropout key, moments and update match.
    checkpoint = out / "resume-input"
    resume_expected = out / "resume-expected"
    resume_actual = out / "resume-child"
    template = train.initialize_training(train_examples, model_config, train_config)
    train.save_checkpoint(checkpoint, state, identity)
    expected_state, expected_item = train.step(
        state, train_examples, model_config, train_config
    )
    train.save_checkpoint(resume_expected, expected_state, identity)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--implementation",
        str(Path(args.implementation).resolve()),
        "--output",
        str(out),
        "--resume-worker",
        str(checkpoint),
        "--worker-output",
        str(resume_actual),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise AssertionError(
            "fresh-process resume failed:\n" + result.stdout + result.stderr
        )
    restored, _ = train.load_checkpoint(resume_actual, template, identity)
    compare_states(expected_state, restored)
    assert json.loads((resume_actual / "trace.json").read_text()) == expected_item

    # An incompatible config/source/tokenizer identity and corrupt bytes are rejected.
    for field in (
        "source_sha256",
        "processor_sha256",
        "dataset_sha256",
        "model_config_sha256",
        "training_config_sha256",
        "code_sha256",
    ):
        changed_identity = dict(identity)
        changed_identity[field] = "changed-identity"
        require_rejects(
            lambda changed_identity=changed_identity: train.load_checkpoint(
                checkpoint, template, changed_identity
            ),
            f"changed {field} identity must fail",
        )
    copied = out / "corrupt-state"
    shutil.copytree(checkpoint, copied, dirs_exist_ok=True)
    (copied / "state.npz").write_bytes((copied / "state.npz").read_bytes() + b"corrupt")
    require_rejects(
        lambda: train.load_checkpoint(copied, template, identity),
        "corrupt checkpoint must fail",
    )

    # Separate clean-process model/processor export and numerical canary.
    package = out / "inference-package"
    train.save_model_package(package, state, model_config, tokenizer_path, identity)
    canary_prompt = [1, *data.encode_text(processor, "Alice")[:5]]
    expected_first, expected_cache = module.prefill(
        state["params"], canary_prompt, model_config
    )
    expected_next, _ = module.decode_step(
        state["params"], 2, expected_cache, model_config
    )
    package_output = out / "package-child"
    package_output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--implementation",
        str(Path(args.implementation).resolve()),
        "--output",
        str(out),
        "--reload-worker",
        str(package),
        "--worker-output",
        str(package_output),
        "--prompt-ids",
        json.dumps(canary_prompt),
        "--next-id",
        "2",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise AssertionError(
            "fresh-process package canary failed:\n" + result.stdout + result.stderr
        )
    with np.load(package_output / "canary.npz", allow_pickle=False) as values:
        np.testing.assert_allclose(
            values["first"], expected_first, atol=2e-6, rtol=2e-6
        )
        np.testing.assert_allclose(values["next"], expected_next, atol=2e-6, rtol=2e-6)
        assert values["decoded"].item() == data.decode_text(processor, canary_prompt)
    corrupt_package = out / "corrupt-package"
    shutil.copytree(package, corrupt_package, dirs_exist_ok=True)
    (corrupt_package / "tokenizer.json").write_bytes(
        (corrupt_package / "tokenizer.json").read_bytes() + b"corrupt"
    )
    require_rejects(
        lambda: train.load_model_package(corrupt_package, identity),
        "corrupt processor package must fail its checksum",
    )
    bound_inputs = {
        "checker": Path(__file__).resolve(),
        "candidate_model": Path(args.implementation).resolve(),
        "train": FOUNDATIONS / "train.py",
        "config": FOUNDATIONS / "config.py",
        "data_pipeline": FOUNDATIONS / "data.py",
        "requirements": FOUNDATIONS / "requirements-foundations.txt",
        "source": data.SOURCE,
        "source_manifest": data.SOURCE_MANIFEST,
        "license": FOUNDATIONS / "data/LICENSE.md",
        "tokenizer": tokenizer_path,
        "tokenizer_info": info_path,
        "raw_metrics": out / "pilot-results.json",
        "resume_checkpoint_manifest": checkpoint / "checkpoint.json",
        "resume_checkpoint_state": checkpoint / "state.npz",
        "inference_package_manifest": package / "package.json",
        "inference_package_parameters": package / "params.npz",
        "inference_package_tokenizer": package / "tokenizer.json",
    }
    receipt = {
        "schema": 1,
        "candidate_model_path": str(Path(args.implementation).resolve()),
        "candidate_model_sha256": candidate_hash,
        "experiment_identity": identity,
        "normalized_byte_denominator": "UTF-8 bytes after collapsing whitespace runs to one ASCII space and trimming each selected paragraph; not raw Gutenberg file byte spans.",
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in bound_inputs.items()
        },
    }
    (out / "sourcebound-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False)
    )
    print(
        "PASS foundations: source split and license hashes, train-only BPE/byte roundtrips, masks, 2-layer cache parity, FP32 gradient/training, fresh-process exact resume and package canary"
    )
    print(
        json.dumps(
            {
                "parameters": count,
                "target_parameter_count_only": report[
                    "target_config_parameter_count_only"
                ],
                "records": {
                    key: report["data"][key]["documents"]
                    for key in ("train", "validation", "test")
                },
                "exposed_tokens": {
                    key: report["data"][key]["tokens_exposed"]
                    for key in ("train", "validation", "test")
                },
                "dev_before": before_dev,
                "dev_after": after_dev,
                "dev_bits_per_normalized_paragraph_byte": dev_bits_per_byte,
                "test": final_test,
                "test_bits_per_normalized_paragraph_byte": test_bits_per_byte,
                "byte_unigram_dev": byte_baseline,
            },
            indent=2,
        )
    )
    print("Saved measured pilot report to", out / "pilot-results.json")
    print("Saved source-bound receipt to", out / "sourcebound-receipt.json")


if __name__ == "__main__":
    main()
