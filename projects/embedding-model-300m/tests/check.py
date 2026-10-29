"""Independent data, model, objective, resume, retrieval and package checks."""

from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jax
import jax.numpy as jnp
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent.parent
sys.path.insert(0, str(PROJECT))

from config import EncoderConfig, TARGET_CONFIG, TrainConfig
from baseline import load_pinned_baseline
from data import dataset
import evaluation
import run as lifecycle
from trainer import (
    initialize_state,
    load_checkpoint,
    next_indices,
    save_checkpoint,
    train_mlm,
    train_human_contrastive,
    train_weak_contrastive,
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _load_candidate(selection):
    return lifecycle.load_implementation(selection)


def _parameter_count_formula(config, include_mlm=False):
    d, f, vocab = config.width, config.mlp_width, config.vocab_size
    count = vocab * d + 2 * d + config.layers * (4 * d * d + 2 * d * f + 4 * d)
    if include_mlm:
        count += d * d + d + 2 * d + vocab
    return count


def _numpy_layer_norm(values, scale, bias, epsilon=1e-5):
    mean = values.mean(axis=-1, keepdims=True)
    variance = np.square(values - mean).mean(axis=-1, keepdims=True)
    return (values - mean) / np.sqrt(variance + epsilon) * scale + bias


def _numpy_rope(values):
    dim = values.shape[-1]
    positions = np.arange(values.shape[-2], dtype=np.float64)
    inverse = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    angle = positions[:, None] * inverse[None, :]
    cosine = np.cos(angle)[None, None, :, :]
    sine = np.sin(angle)[None, None, :, :]
    even, odd = values[..., 0::2], values[..., 1::2]
    rotated = np.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), axis=-1
    )
    return rotated.reshape(values.shape)


def _numpy_encoder_oracle(params, ids, config):
    """Independent float64 forward pass for the declared masked RoPE encoder."""
    ids = np.asarray(ids, np.int64)
    valid = ids != config.pad_id
    batch, length = ids.shape
    heads = config.heads
    head_dim = config.width // heads
    x = np.asarray(params["token_embed"])[ids].astype(np.float64)
    for block in params["blocks"]:
        h = _numpy_layer_norm(
            x,
            np.asarray(block["norm1_scale"], np.float64),
            np.asarray(block["norm1_bias"], np.float64),
        )
        projected = {
            name: (h @ np.asarray(block[name], np.float64))
            .reshape(batch, length, heads, head_dim)
            .transpose(0, 2, 1, 3)
            for name in ("q", "k", "v")
        }
        q, k = (_numpy_rope(projected[name]) for name in ("q", "k"))
        v = projected["v"]
        scores = np.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(head_dim)
        allowed = valid[:, None, None, :] & valid[:, None, :, None]
        scores = np.where(allowed, scores, -1.0e30)
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-30)
        weights = np.where(valid[:, None, :, None], weights, 0.0)
        attended = np.einsum("bhqk,bhkd->bhqd", weights, v)
        joined = attended.transpose(0, 2, 1, 3).reshape(batch, length, config.width)
        branch = joined @ np.asarray(block["o"], np.float64)
        x = np.where(valid[..., None], x + branch, 0.0)
        h = _numpy_layer_norm(
            x,
            np.asarray(block["norm2_scale"], np.float64),
            np.asarray(block["norm2_bias"], np.float64),
        )
        pre_gelu = h @ np.asarray(block["up"], np.float64)
        gelu_gate = np.tanh(np.sqrt(2.0 / np.pi) * (pre_gelu + 0.044715 * pre_gelu**3))
        branch = 0.5 * pre_gelu * (1.0 + gelu_gate)
        branch = branch @ np.asarray(block["down"], np.float64)
        x = np.where(valid[..., None], x + branch, 0.0)
    return np.where(
        valid[..., None],
        _numpy_layer_norm(
            x,
            np.asarray(params["final_scale"], np.float64),
            np.asarray(params["final_bias"], np.float64),
        ),
        0.0,
    )


def _numpy_mlm_logits(params, ids, config):
    """Independent float64 tied-head forward pass over oracle encoder states."""
    hidden = _numpy_encoder_oracle(params, ids, config)
    head = params["mlm"]
    transformed = hidden @ np.asarray(head["dense"], np.float64)
    transformed += np.asarray(head["dense_bias"], np.float64)
    transformed = (
        0.5
        * transformed
        * (
            1.0
            + np.tanh(np.sqrt(2.0 / np.pi) * (transformed + 0.044715 * transformed**3))
        )
    )
    transformed = _numpy_layer_norm(
        transformed,
        np.asarray(head["norm_scale"], np.float64),
        np.asarray(head["norm_bias"], np.float64),
    )
    return transformed @ np.asarray(params["token_embed"], np.float64).T + np.asarray(
        head["vocab_bias"], np.float64
    )


def _numpy_masked_loss(logits, labels, mask):
    logits = np.asarray(logits, np.float64)
    labels = np.asarray(labels, np.int64)
    mask = np.asarray(mask, bool)
    maximum = logits.max(axis=-1, keepdims=True)
    log_partition = np.log(np.exp(logits - maximum).sum(axis=-1)) + maximum[..., 0]
    chosen = np.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    losses = log_partition - chosen
    count = int(mask.sum())
    return float(losses[mask].sum() / count), float(losses[mask].sum()), count


def _replace_tree_leaf(tree, path, value):
    """Copy a nested dict/tuple parameter tree with one leaf replaced."""
    if not path:
        return value
    key, *tail = path
    if isinstance(tree, dict):
        result = dict(tree)
        result[key] = _replace_tree_leaf(tree[key], tail, value)
        return result
    if isinstance(tree, tuple):
        result = list(tree)
        result[key] = _replace_tree_leaf(tree[key], tail, value)
        return tuple(result)
    if isinstance(tree, list):
        result = list(tree)
        result[key] = _replace_tree_leaf(tree[key], tail, value)
        return result
    raise TypeError(f"cannot descend through parameter leaf at {path!r}")


def _check_mlm_oracle_and_gradients(module, params, ids, labels, selected, config):
    """Grade raw MLM outputs and gradients against independent NumPy math."""
    raw_logits = np.asarray(module.mlm_logits(params, ids, config), np.float64)
    oracle_logits = _numpy_mlm_logits(params, np.asarray(ids), config)
    if not np.all(np.isfinite(raw_logits)):
        raise AssertionError("candidate MLM logits are non-finite")
    np.testing.assert_allclose(raw_logits, oracle_logits, rtol=3e-5, atol=3e-6)

    expected_loss = _numpy_masked_loss(oracle_logits, labels, selected)
    candidate_loss = module.masked_token_loss(params, ids, labels, selected, config)
    np.testing.assert_allclose(
        np.asarray(candidate_loss, np.float64),
        np.asarray(expected_loss, np.float64),
        rtol=3e-5,
        atol=3e-6,
    )

    gradient = jax.grad(
        lambda current: module.masked_token_loss(
            current, ids, labels, selected, config
        )[0]
    )(params)
    checks = []
    paths = (
        ("mlm", "dense"),
        ("mlm", "norm_scale"),
        ("mlm", "norm_bias"),
        ("mlm", "vocab_bias"),
        ("blocks", 0, "q"),
        ("blocks", 1, "v"),
        ("blocks", 0, "up"),
        ("blocks", 1, "down"),
        ("token_embed",),
    )
    epsilon_values = (2e-3, 1e-3)
    for index, path in enumerate(paths):
        leaf = params
        grad_leaf = gradient
        for part in path:
            leaf = leaf[part]
            grad_leaf = grad_leaf[part]
        leaf_array = np.asarray(leaf, np.float64)
        direction = np.random.default_rng(7301 + index).normal(size=leaf_array.shape)
        direction /= np.linalg.norm(direction)

        def independent_loss(offset):
            changed = _replace_tree_leaf(
                params,
                path,
                jnp.asarray(leaf_array + offset * direction, dtype=jnp.float32),
            )
            return _numpy_masked_loss(
                _numpy_mlm_logits(changed, np.asarray(ids), config),
                labels,
                selected,
            )[0]

        finite_differences = [
            (independent_loss(epsilon) - independent_loss(-epsilon)) / (2.0 * epsilon)
            for epsilon in epsilon_values
        ]
        if not all(np.isfinite(finite_differences)):
            raise AssertionError(f"non-finite independent MLM derivative for {path}")
        if max(abs(value) for value in finite_differences) < 1e-4:
            raise AssertionError(f"MLM fixture has no sensitivity for {path}")
        fd_spread = abs(finite_differences[0] - finite_differences[1])
        np.testing.assert_allclose(
            finite_differences[0], finite_differences[1], rtol=0.02, atol=1e-7
        )
        candidate_derivative = float(
            np.sum(np.asarray(grad_leaf, np.float64) * direction)
        )
        derivative_tolerance = max(
            4.0 * fd_spread, 1e-6, 0.05 * abs(finite_differences[1])
        )
        if abs(candidate_derivative) < 0.5 * abs(finite_differences[1]):
            raise AssertionError(
                f"candidate MLM derivative is disconnected from {path}: "
                f"{candidate_derivative} versus independent "
                f"{finite_differences[1]}"
            )
        if abs(candidate_derivative - finite_differences[1]) > derivative_tolerance:
            raise AssertionError(
                f"candidate MLM derivative for {path} differs by "
                f"{abs(candidate_derivative - finite_differences[1])}, "
                f"exceeding finite-difference-aware tolerance "
                f"{derivative_tolerance}"
            )
        checks.append(
            {
                "path": list(path),
                "candidate_autodiff": candidate_derivative,
                "independent_numpy_finite_differences": finite_differences,
                "independent_fd_spread": fd_spread,
                "candidate_derivative_tolerance": derivative_tolerance,
                "direction_norm": float(np.linalg.norm(direction)),
            }
        )

    head_norm = float(
        np.sqrt(
            sum(
                np.sum(np.square(np.asarray(gradient["mlm"][name], np.float64)))
                for name in (
                    "dense",
                    "dense_bias",
                    "norm_scale",
                    "norm_bias",
                    "vocab_bias",
                )
            )
        )
    )
    encoder_norm = float(
        np.sqrt(
            sum(
                np.sum(np.square(np.asarray(leaf, np.float64)))
                for leaf in jax.tree.leaves(gradient["blocks"])
            )
        )
    )
    if (
        not np.isfinite(head_norm + encoder_norm)
        or min(head_norm, encoder_norm) <= 1e-4
    ):
        raise AssertionError("MLM must have finite, nonzero head and encoder gradients")
    return {
        "selected_loss_matches_independent_numpy": True,
        "raw_logits_match_independent_numpy": True,
        "selected_count": int(expected_loss[2]),
        "head_activation": "tanh-approximate GELU (0.044715 cubic coefficient)",
        "layer_norm_variance_epsilon": 1e-5,
        "head_gradient_norm": head_norm,
        "encoder_gradient_norm": encoder_norm,
        "derivatives": checks,
    }


def _independent_contrastive(
    query, docs, positive, candidate_mask=None, temperature=0.05
):
    query = np.asarray(query, np.float64)
    docs = np.asarray(docs, np.float64)
    positive = np.asarray(positive, bool)
    candidate_mask = (
        np.ones(docs.shape[0], bool)
        if candidate_mask is None
        else np.asarray(candidate_mask, bool)
    )
    logits = query @ docs.T / temperature
    rows = []
    for index, values in enumerate(logits):
        row_positive = positive[index] & candidate_mask
        if not row_positive.any():
            raise ValueError("oracle query has no positive candidate")
        denom = np.logaddexp.reduce(values[candidate_mask])
        numerator = np.logaddexp.reduce(values[row_positive])
        rows.append(denom - numerator)
    return float(np.mean(rows))


def _check_data(data_dir):
    frozen = dataset.verify_frozen_data(data_dir)
    ids = frozen["inventories"]["ids"]
    corpus = frozen["corpus"]
    train = frozen["original_train"]
    final = frozen["final"]
    if len(corpus) != 5183 or len(train) != 809 or len(final) != 300:
        raise AssertionError(
            "official SciFact inventory sizes differ from frozen recipe"
        )

    mutated_data = Path(tempfile.mkdtemp(prefix="jaxpathways-c01-data-mutation-"))
    shutil.copytree(data_dir, mutated_data / "frozen")
    mutated_dir = mutated_data / "frozen"
    fit_file = mutated_dir / "claims_fit.jsonl"
    fit_rows = [json.loads(line) for line in fit_file.read_text().splitlines()]
    fit_rows[0]["claim"] = "REVIEW CANARY: not part of the pinned archive"
    fit_file.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in fit_rows)
    )
    mutation_manifest_path = mutated_dir / "frozen-manifest.json"
    mutation_manifest = json.loads(mutation_manifest_path.read_text())
    mutation_manifest["outputs"]["claims_fit.jsonl_sha256"] = _sha(
        fit_file.read_bytes()
    )
    mutation_manifest_path.write_text(json.dumps(mutation_manifest))
    try:
        dataset.verify_frozen_data(mutated_dir)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "data verifier accepted a text mutation with a recomputed local manifest"
        )

    def norm(text):
        return " ".join(re.sub(r"\W+", " ", text.casefold()).split())

    parent = list(range(len(train) + len(final)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    by_text = {}
    duplicate_doc_groups = {}
    for row in corpus:
        content = norm(
            (row.get("title", "") or "") + " " + " ".join(row.get("abstract") or [])
        )
        if content in by_text:
            duplicate_doc_groups.setdefault(content, [by_text[content]]).append(
                int(row["doc_id"])
            )
        else:
            by_text[content] = int(row["doc_id"])
    doc_alias = {int(row["doc_id"]): f"doc:{int(row['doc_id'])}" for row in corpus}
    for group_index, group in enumerate(duplicate_doc_groups.values()):
        for doc_id in group:
            doc_alias[doc_id] = f"exactdoc:{group_index}"

    linked, owner = [], {}
    combined = train + final
    for index, claim in enumerate(combined):
        docs = set(map(int, claim.get("cited_doc_ids", []))) | set(
            map(int, claim.get("evidence", {}).keys())
        )
        linked.append(docs)
        keys = {doc_alias[doc_id] for doc_id in docs}
        keys.add("claim:" + norm(claim["claim"]))
        for key in keys:
            if key in owner:
                union(index, owner[key])
            else:
                owner[key] = index
    final_roots = {find(len(train) + index) for index in range(len(final))}
    final_claim_roots = {root for root in final_roots}
    expected_quarantine = {
        int(train[index]["id"])
        for index in range(len(train))
        if find(index) in final_claim_roots
    }
    fit_ids = {int(row["id"]) for row in frozen["fit"]}
    dev_ids = {int(row["id"]) for row in frozen["recipe_dev"]}
    original_ids = {int(row["id"]) for row in train}
    if fit_ids & dev_ids or fit_ids | dev_ids | expected_quarantine != original_ids:
        raise AssertionError(
            "fit/development/quarantine is not an exact original-train partition"
        )
    if expected_quarantine & (fit_ids | dev_ids):
        raise AssertionError("final-connected claim component leaked into train split")
    final_component_docs = set().union(
        *(
            linked[index]
            for index in range(len(combined))
            if find(index) in final_claim_roots
        )
    )
    pretraining_ids = set(map(int, ids["pretraining_doc_ids"]))
    if (
        pretraining_ids
        != set(int(row["doc_id"]) for row in corpus) - final_component_docs
    ):
        raise AssertionError(
            "tokenizer/MLM inventory does not exclude the complete final-connected document group"
        )
    if sorted(final_component_docs) != ids["final_heldout_doc_ids"]:
        raise AssertionError("heldout document-group inventory is wrong")

    def assert_no_shared_components(rows_a, rows_b, label):
        map_a = {int(row["id"]): row for row in rows_a}
        map_b = {int(row["id"]): row for row in rows_b}
        root_a = {find(i) for i, row in enumerate(combined) if int(row["id"]) in map_a}
        root_b = {find(i) for i, row in enumerate(combined) if int(row["id"]) in map_b}
        if root_a & root_b:
            raise AssertionError(f"group leakage across {label}")

    assert_no_shared_components(
        frozen["fit"], frozen["recipe_dev"], "fit and recipe-development"
    )
    if len(final_component_docs) != 295 or len(expected_quarantine) != 278:
        raise AssertionError(
            "transitive final-component counts differ from frozen observed values"
        )
    for claim in final:
        evidence_ids = set(map(int, claim.get("evidence", {}).keys()))
        if not evidence_ids <= {int(row["doc_id"]) for row in corpus}:
            raise AssertionError("final positive missing from full candidate corpus")
        for annotations in claim.get("evidence", {}).values():
            for annotation in annotations:
                if annotation["label"] not in {
                    "SUPPORT",
                    "CONTRADICT",
                } or not annotation.get("sentences"):
                    raise AssertionError("original human annotation was not preserved")
    return frozen


def _check_model(candidate):
    module, candidate_path = candidate
    config = EncoderConfig(
        vocab_size=32,
        max_length=12,
        width=16,
        layers=2,
        heads=4,
        mlp_width=24,
        dropout=0.1,
    )
    params = module.initialize(jax.random.key(17), config)
    actual_encoder = sum(
        int(np.prod(leaf.shape))
        for leaf in jax.tree.leaves({k: v for k, v in params.items() if k != "mlm"})
    )
    actual_all = sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(params))
    if (
        module.parameter_count(config) != actual_encoder
        or module.parameter_count(config, include_mlm=True) != actual_all
    ):
        raise AssertionError("candidate parameter count disagrees with its real tree")
    if (
        module.parameter_count(TARGET_CONFIG) != 302090240
        or module.parameter_count(TARGET_CONFIG, include_mlm=True) != 303191040
    ):
        raise AssertionError("target architecture count is not the frozen target count")
    external_dir = Path(tempfile.mkdtemp(prefix="jaxpathways-c01-candidate-")).resolve()
    external_candidate = external_dir / "learner-model.py"
    shutil.copyfile(candidate_path, external_candidate)
    _, external_candidate = lifecycle.load_implementation(external_candidate)
    external_hashes = lifecycle._hash_source_tree(external_candidate)
    if external_hashes.get("candidate_external/learner-model.py") != _sha(
        external_candidate.read_bytes()
    ):
        raise AssertionError("external learner candidate path was not source-bound")
    abstract = jax.eval_shape(
        lambda key: module.initialize(key, TARGET_CONFIG), jax.random.key(2)
    )
    abstract_count = sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(abstract))
    if abstract_count != 303191040:
        raise AssertionError(
            "abstract target parameter tree does not include the tied MLM head exactly once"
        )

    ids = jnp.asarray([[2, 5, 6, 7, 3, 0], [2, 9, 8, 7, 3, 0]], jnp.int32)
    hidden = np.asarray(module.hidden_states(params, ids, config))
    oracle_hidden = _numpy_encoder_oracle(params, np.asarray(ids), config)
    np.testing.assert_allclose(hidden, oracle_hidden, rtol=2e-5, atol=2e-6)
    content = np.asarray(ids) != config.pad_id
    for token_id in (config.bos_id, config.eos_id, config.unk_id, config.mask_id):
        content &= np.asarray(ids) != token_id
    pooled_mean = np.sum(
        np.where(content[..., None], oracle_hidden, 0.0), axis=1
    ) / np.maximum(content.sum(axis=1, keepdims=True), 1)
    pooled_norm = np.linalg.norm(pooled_mean, axis=-1, keepdims=True)
    oracle_pooled = np.where(
        (content.sum(axis=1, keepdims=True) > 0) & (pooled_norm > 0),
        pooled_mean / np.maximum(pooled_norm, 1e-12),
        0.0,
    )
    np.testing.assert_allclose(
        np.asarray(module.pool(params, ids, config)),
        oracle_pooled,
        rtol=2e-5,
        atol=2e-6,
    )

    # Ensure the fixed NumPy fixture rejects an encoder that discards QK logits
    # and spreads attention uniformly over the valid keys.
    reference_path = PROJECT / "model.py"
    reference_source = reference_path.read_text(encoding="utf-8")
    softmax_source = "weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)"
    if softmax_source not in reference_source:
        raise AssertionError("owned reference fixture lacks its attention expression")
    uniform_source = reference_source.replace(
        softmax_source,
        "weights = safe_allowed.astype(jnp.float32) / "
        "jnp.maximum(jnp.sum(safe_allowed.astype(jnp.float32), axis=-1, keepdims=True), 1.0)",
        1,
    )
    uniform_path = Path(tempfile.mkdtemp(prefix="jaxpathways-c01-uniform-attn-")) / (
        "uniform_attention.py"
    )
    uniform_path.write_text(uniform_source)
    uniform_module, _ = _load_candidate(uniform_path)
    reference_module, _ = _load_candidate(reference_path)
    reference_params = reference_module.initialize(jax.random.key(17), config)
    reference_ids = np.asarray(ids)
    reference_oracle = _numpy_encoder_oracle(reference_params, reference_ids, config)
    uniform_hidden = np.asarray(
        uniform_module.hidden_states(reference_params, ids, config)
    )
    if np.allclose(uniform_hidden, reference_oracle, rtol=2e-5, atol=2e-6):
        raise AssertionError("NumPy oracle did not reject uniform masked attention")
    future_changed = ids.at[0, 3].set(10)
    changed = np.asarray(module.hidden_states(params, future_changed, config))
    if np.allclose(hidden[0, 1], changed[0, 1], rtol=1e-6, atol=1e-7):
        raise AssertionError(
            "bidirectional encoder did not expose a future-token change to an earlier state"
        )
    longer = jnp.pad(ids, ((0, 0), (0, 3)))
    padded_hidden = np.asarray(module.hidden_states(params, longer, config))
    np.testing.assert_allclose(
        hidden, padded_hidden[:, : ids.shape[1]], rtol=1e-5, atol=1e-6
    )
    second_changed = ids.at[1, 1].set(11)
    changed_rows = np.asarray(module.hidden_states(params, second_changed, config))
    np.testing.assert_array_equal(hidden[0], changed_rows[0])
    pooled = np.asarray(module.pool(params, ids, config))
    np.testing.assert_allclose(
        np.linalg.norm(pooled, axis=1), np.ones(2), rtol=1e-5, atol=1e-6
    )
    empty = np.asarray(module.pool(params, jnp.asarray([[2, 3, 0]], jnp.int32), config))
    np.testing.assert_array_equal(empty, np.zeros_like(empty))
    stochastic_config = replace(config, dropout=0.5)
    stochastic_a = np.asarray(
        module.pool(params, ids, stochastic_config, dropout_key=jax.random.key(61))
    )
    stochastic_b = np.asarray(
        module.pool(params, ids, stochastic_config, dropout_key=jax.random.key(62))
    )
    if np.allclose(stochastic_a, stochastic_b, rtol=1e-6, atol=1e-7):
        raise AssertionError("nonzero encoder dropout ignored distinct PRNG keys")
    try:
        module.hidden_states(
            params,
            jnp.asarray([[2, 0]], jnp.int32),
            config,
            jnp.asarray([[1, 1]], bool),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid valid-mask/padding combination was accepted")

    input_ids = jnp.asarray([[2, 4, 9, 3], [2, 6, 4, 3]], jnp.int32)
    labels = jnp.asarray([[2, 7, 9, 3], [2, 6, 8, 3]], jnp.int32)
    selected = jnp.asarray([[0, 1, 0, 0], [0, 0, 1, 0]], bool)
    mlm_params = copy.deepcopy(params)
    mlm_params["mlm"]["dense_bias"] = jnp.linspace(-0.13, 0.17, config.width)
    mlm_params["mlm"]["norm_scale"] = jnp.linspace(0.63, 1.37, config.width)
    mlm_params["mlm"]["norm_bias"] = jnp.linspace(0.09, -0.11, config.width)
    mlm_params["mlm"]["vocab_bias"] = jnp.linspace(-0.21, 0.24, config.vocab_size)
    mlm_checks = _check_mlm_oracle_and_gradients(
        module, mlm_params, input_ids, labels, selected, config
    )
    # BERT-style selected targets still contribute when corruption chooses a
    # random vocabulary token or leaves the input unchanged (the 10%/10% cases).
    unmasked_selected = input_ids.at[0, 1].set(12).at[1, 2].set(8)
    unmasked_loss, _, unmasked_count = module.masked_token_loss(
        mlm_params, unmasked_selected, labels, selected, config
    )
    unmasked_oracle = _numpy_masked_loss(
        _numpy_mlm_logits(mlm_params, np.asarray(unmasked_selected), config),
        labels,
        selected,
    )
    np.testing.assert_allclose(
        [unmasked_loss, unmasked_count],
        [unmasked_oracle[0], unmasked_oracle[2]],
        rtol=2e-5,
        atol=2e-6,
    )
    try:
        module.masked_token_loss(
            mlm_params, input_ids, labels, jnp.zeros_like(selected), config
        )
    except ValueError:
        pass
    else:
        raise AssertionError("empty masked-token loss set was accepted")

    # The false-pass mutations are derived from the owned implementation, not
    # from source-string assumptions about a submitted learner implementation.
    owned_zero, _ = _load_candidate(PROJECT / "model.py")
    zero_params = owned_zero.initialize(jax.random.key(17), config)
    zero_params["mlm"]["dense_bias"] = mlm_params["mlm"]["dense_bias"]
    zero_params["mlm"]["norm_scale"] = mlm_params["mlm"]["norm_scale"]
    zero_params["mlm"]["norm_bias"] = mlm_params["mlm"]["norm_bias"]
    zero_params["mlm"]["vocab_bias"] = mlm_params["mlm"]["vocab_bias"]
    owned_zero.mlm_logits = (
        lambda p, token_ids, cfg, valid_mask=None, **kwargs: jnp.zeros(
            (*jnp.asarray(token_ids).shape, cfg.vocab_size), dtype=jnp.float32
        )
    )
    try:
        _check_mlm_oracle_and_gradients(
            owned_zero, zero_params, input_ids, labels, selected, config
        )
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "owned constant-zero MLM head passed the independent oracle"
        )

    owned_stopped, _ = _load_candidate(PROJECT / "model.py")
    stopped_hidden_states = owned_stopped.hidden_states
    owned_stopped.hidden_states = lambda *args, **kwargs: jax.lax.stop_gradient(
        stopped_hidden_states(*args, **kwargs)
    )
    stopped_params = owned_stopped.initialize(jax.random.key(17), config)
    stopped_params["mlm"]["dense_bias"] = mlm_params["mlm"]["dense_bias"]
    stopped_params["mlm"]["norm_scale"] = mlm_params["mlm"]["norm_scale"]
    stopped_params["mlm"]["norm_bias"] = mlm_params["mlm"]["norm_bias"]
    stopped_params["mlm"]["vocab_bias"] = mlm_params["mlm"]["vocab_bias"]
    try:
        _check_mlm_oracle_and_gradients(
            owned_stopped, stopped_params, input_ids, labels, selected, config
        )
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "owned forward-identical stopped-encoder-gradient mutation passed"
        )

    owned_stopped_head, _ = _load_candidate(PROJECT / "model.py")
    original_mlm_logits = owned_stopped_head.mlm_logits
    owned_stopped_head.mlm_logits = lambda *args, **kwargs: jax.lax.stop_gradient(
        original_mlm_logits(*args, **kwargs)
    )
    stopped_head_params = owned_stopped_head.initialize(jax.random.key(17), config)
    for name in ("dense_bias", "norm_scale", "norm_bias", "vocab_bias"):
        stopped_head_params["mlm"][name] = mlm_params["mlm"][name]
    try:
        _check_mlm_oracle_and_gradients(
            owned_stopped_head,
            stopped_head_params,
            input_ids,
            labels,
            selected,
            config,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("owned stopped-head-gradient mutation passed")

    class StopDownProjection(ast.NodeTransformer):
        def __init__(self):
            self.changed = 0

        def visit_FunctionDef(self, node):
            if node.name != "hidden_states":
                return node
            return self.generic_visit(node)

        def visit_Subscript(self, node):
            self.generic_visit(node)
            key = node.slice
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "block"
                and isinstance(key, ast.Constant)
                and key.value == "down"
            ):
                self.changed += 1
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="jax", ctx=ast.Load()),
                            attr="lax",
                            ctx=ast.Load(),
                        ),
                        attr="stop_gradient",
                        ctx=ast.Load(),
                    ),
                    args=[node],
                    keywords=[],
                )
            return node

    stopped_down_tree = ast.parse(reference_path.read_text(encoding="utf-8"))
    stop_down = StopDownProjection()
    stopped_down_tree = stop_down.visit(stopped_down_tree)
    if stop_down.changed != 1:
        raise AssertionError(
            "owned stopped-down fixture did not isolate one projection"
        )
    stopped_down_path = (
        Path(tempfile.mkdtemp(prefix="jaxpathways-c01-stopped-down-"))
        / "stopped_down.py"
    )
    stopped_down_path.write_text(
        ast.unparse(ast.fix_missing_locations(stopped_down_tree)) + "\n",
        encoding="utf-8",
    )
    owned_stopped_down, _ = _load_candidate(stopped_down_path)
    stopped_down_params = owned_stopped_down.initialize(jax.random.key(17), config)
    for name in ("dense_bias", "norm_scale", "norm_bias", "vocab_bias"):
        stopped_down_params["mlm"][name] = mlm_params["mlm"][name]
    try:
        _check_mlm_oracle_and_gradients(
            owned_stopped_down,
            stopped_down_params,
            input_ids,
            labels,
            selected,
            config,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("owned stopped-down-gradient mutation passed")

    query_vectors = jnp.asarray([[0.8, 0.6], [0.6, 0.8]], jnp.float32)
    doc_vectors = jnp.asarray(
        [[0.8, 0.6], [0.6, 0.8], [1.0, 0.0], [0.0, 1.0]], jnp.float32
    )
    positive = jnp.asarray([[1, 1, 0, 0], [0, 1, 0, 0]], bool)
    candidate_mask = jnp.asarray([1, 1, 1, 0], bool)
    from objectives import multi_positive_contrastive
    from objectives import retrieval_metrics

    contrastive = multi_positive_contrastive(
        query_vectors, doc_vectors, positive, candidate_mask=candidate_mask
    )
    contrastive_oracle = _independent_contrastive(
        query_vectors, doc_vectors, positive, candidate_mask
    )
    np.testing.assert_allclose(
        float(contrastive), contrastive_oracle, rtol=1e-6, atol=1e-6
    )
    query_grad = jax.grad(
        lambda q: multi_positive_contrastive(
            q, doc_vectors, positive, candidate_mask=candidate_mask
        )
    )(query_vectors)
    eps = 1e-3
    q_plus, q_minus = np.asarray(query_vectors).copy(), np.asarray(query_vectors).copy()
    q_plus[0, 0] += eps
    q_minus[0, 0] -= eps
    fd = (
        _independent_contrastive(q_plus, doc_vectors, positive, candidate_mask)
        - _independent_contrastive(q_minus, doc_vectors, positive, candidate_mask)
    ) / (2 * eps)
    np.testing.assert_allclose(float(query_grad[0, 0]), fd, rtol=2e-3, atol=2e-4)

    try:
        multi_positive_contrastive(
            query_vectors,
            doc_vectors,
            jnp.zeros_like(positive),
            candidate_mask=candidate_mask,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("no-positive contrastive row was accepted")
    reference_rankings = {101: [8, 9, 10], 102: [9, 8, 10]}
    reference_relevance = {101: [8], 102: []}
    if (
        retrieval_metrics(reference_rankings, reference_relevance)["queries_judged"]
        != 1
    ):
        raise AssertionError(
            "retrieval metrics lost the expected judged-query denominator"
        )
    for bad_rankings, bad_relevance in (
        ({101: [8, 8], 102: [9, 8, 10]}, reference_relevance),
        ({101: [8, 9, 10]}, reference_relevance),
        ({101: [9, 10], 102: [9, 8, 10]}, reference_relevance),
    ):
        try:
            retrieval_metrics(bad_rankings, bad_relevance)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "retrieval metrics accepted incomplete/duplicate ranks"
            )
    changed_seed = module.initialize(jax.random.key(18), config)
    if np.allclose(
        np.asarray(module.pool(params, ids, config)),
        np.asarray(module.pool(changed_seed, ids, config)),
    ):
        raise AssertionError(
            "changed initialization seed did not change encoder vectors"
        )
    changed_config = replace(config, layers=3)
    changed_params = module.initialize(jax.random.key(17), changed_config)
    if module.parameter_count(changed_config, include_mlm=True) == actual_all:
        raise AssertionError("changed model shape did not alter parameter count")
    if module.hidden_states(changed_params, ids, changed_config).shape != hidden.shape:
        raise AssertionError("changed layer count changed output shape")
    return {
        "target_encoder_params": 302090240,
        "target_with_mlm_params": 303191040,
        "bidirectional_future_context": True,
        "numpy_encoder_attention_rope_mask_pooling_oracle": True,
        "uniform_attention_mutation_rejected": True,
        "padding_and_batch_isolation": True,
        "nonzero_dropout_responds_to_prng": True,
        "mlm_numpy_oracle_and_full_gradient_fd": mlm_checks,
        "mlm_expected_head_conventions": {
            "activation": "tanh-approximate GELU (0.044715 cubic coefficient)",
            "layer_norm_variance_epsilon": 1e-5,
        },
        "constant_zero_head_mutation_rejected": True,
        "stopped_encoder_gradient_mutation_rejected": True,
        "stopped_head_gradient_mutation_rejected": True,
        "stopped_down_gradient_mutation_rejected": True,
        "multi_positive_numpy_oracle_and_gradient_fd": True,
        "candidate": str(candidate_path),
    }


def _compare_state(left, right):
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if left_tree != right_tree or len(left_leaves) != len(right_leaves):
        raise AssertionError("resume changed pytree structure")
    for first, second in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def _check_resume(candidate, out_dir):
    model, candidate_path = candidate
    config = EncoderConfig(
        vocab_size=32,
        max_length=12,
        width=16,
        layers=2,
        heads=4,
        mlp_width=24,
        dropout=0.1,
    )
    train_config = TrainConfig(
        batch_size=4,
        learning_rate=0.0002,
        weight_decay=0.01,
        warmup_steps=1,
        total_steps=4,
        seed=913,
    )
    rows = np.asarray(
        [[2, 5 + (i % 10), 7, 3, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(10)], np.int32
    )
    initial = initialize_state(
        model, config, len(rows), seed=912, train_config=train_config
    )
    identity = {
        "code": _sha(Path(candidate_path).read_bytes()),
        "dataset": _sha(rows.tobytes()),
        "purpose": "resume negative-control",
    }
    state, _ = train_mlm(
        model, initial, rows, config, train_config, steps=2, tag="resume-control"
    )
    checkpoint = Path(out_dir) / "resume-input"
    save_checkpoint(checkpoint, state, identity)
    expected, expected_trace = train_mlm(
        model, state, rows, config, train_config, steps=1, tag="resume-control"
    )

    worker = Path(out_dir) / "resume-worker"
    worker.mkdir(parents=True, exist_ok=True)
    rows_file = Path(out_dir) / "resume-rows.npy"
    np.save(rows_file, rows)
    instruction = {
        "candidate": str(candidate_path),
        "config": config.as_dict(),
        "train_config": train_config.as_dict(),
        "count": len(rows),
        "identity": identity,
        "checkpoint": str(checkpoint),
        "rows": str(rows_file),
        "output": str(worker),
        "trace_output": str(Path(out_dir) / "mlm-resume-trace.json"),
    }
    instruction_path = Path(out_dir) / "resume-instruction.json"
    instruction_path.write_text(json.dumps(instruction))
    code = (
        "import json,sys,numpy as np;from pathlib import Path;"
        "sys.path.insert(0,str(Path(sys.argv[1]).resolve()));"
        "from config import EncoderConfig,TrainConfig;import run;from trainer import initialize_state,load_checkpoint,train_mlm,save_checkpoint;"
        "a=json.load(open(sys.argv[2]));m,_=run.load_implementation(a['candidate']);"
        "c=EncoderConfig(**a['config']);t=TrainConfig(**a['train_config']);"
        "template=initialize_state(m,c,a['count'],seed=912,train_config=t);"
        "s,_=load_checkpoint(a['checkpoint'],template,a['identity']);"
        "s,stats=train_mlm(m,s,np.load(a['rows']),c,t,steps=1,tag='resume-control');"
        "json.dump(stats['trace'],open(a['trace_output'],'w'),sort_keys=True);"
        "save_checkpoint(a['output'],s,a['identity'])"
    )
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    subprocess.run(
        [sys.executable, "-c", code, str(PROJECT), str(instruction_path)],
        cwd=worker,
        env=env,
        check=True,
        timeout=180,
    )
    template = initialize_state(
        model, config, len(rows), seed=912, train_config=train_config
    )
    restored, _ = load_checkpoint(worker, template, identity)
    _compare_state(expected, restored)
    resumed_trace = json.loads(Path(instruction["trace_output"]).read_text())
    for key in ("indices", "dropout_key", "mask_key", "learning_rate"):
        if resumed_trace[0][key] != expected_trace["trace"][0][key]:
            raise AssertionError(f"fresh MLM resume changed its next {key}")

    # Independently exercise both phase-2 branch mechanics and the supervised
    # phase-3 candidate-union path across a process boundary. A phase-1-only
    # replay test cannot certify those stage-specific sampler/objective inputs.
    replay_fixtures = []
    left = rows[:8].copy()
    right = rows[:8].copy()
    right[:, 2] = np.where(right[:, 2] == 7, 8, 7)
    groups = np.repeat(np.arange(4, dtype=np.int32), 2)
    weak_arrays = {"rows": rows, "left": left, "right": right, "groups": groups}
    replay_fixtures.append(("weak", weak_arrays, None, None, len(left)))
    replay_fixtures.append(("weak_only", weak_arrays, None, None, len(left)))
    queries = [
        {
            "query_id": index,
            "ids": rows[index].tolist(),
            "positive_doc_ids": [100 + index % 4],
        }
        for index in range(8)
    ]
    documents = [{"id": 100 + index, "ids": rows[index].tolist()} for index in range(4)]
    replay_fixtures.append(("human", {"rows": rows}, queries, documents, len(queries)))
    for fixture_index, (kind, arrays, query_json, document_json, count) in enumerate(
        replay_fixtures
    ):
        seed = 930 + fixture_index
        retained_weight = 0.0 if kind == "weak_only" else 0.1
        stage_state = initialize_state(
            model, config, count, seed=seed, train_config=train_config
        )
        stage_identity = {
            **identity,
            "dataset": _sha(
                json.dumps(
                    arrays, sort_keys=True, default=lambda x: x.tolist()
                ).encode()
            ),
            "purpose": f"{kind}-resume-control",
        }
        payload_path = Path(out_dir) / f"{kind}-payload.npz"
        np.savez_compressed(payload_path, **arrays)
        queries_path = Path(out_dir) / f"{kind}-queries.json"
        documents_path = Path(out_dir) / f"{kind}-documents.json"
        queries_path.write_text(json.dumps(query_json or []))
        documents_path.write_text(json.dumps(document_json or []))

        def train_stage(current, steps):
            if kind in {"weak", "weak_only"}:
                return train_weak_contrastive(
                    model,
                    current,
                    left,
                    right,
                    groups,
                    config,
                    train_config,
                    steps=steps,
                    retained_mlm_weight=retained_weight,
                    mlm_examples=rows,
                    tag="resume-weak-control",
                )
            return train_human_contrastive(
                model,
                current,
                [{**row, "ids": np.asarray(row["ids"], np.int32)} for row in queries],
                [{**row, "ids": np.asarray(row["ids"], np.int32)} for row in documents],
                config,
                train_config,
                steps=steps,
                max_positive_per_query=4,
            )

        stage_state, _ = train_stage(stage_state, 2)
        stage_checkpoint = Path(out_dir) / f"{kind}-resume-input"
        save_checkpoint(stage_checkpoint, stage_state, stage_identity)
        expected_stage, expected_stats = train_stage(stage_state, 1)
        stage_output = Path(out_dir) / f"{kind}-resume-worker"
        stage_instruction = {
            "candidate": str(candidate_path),
            "config": config.as_dict(),
            "train_config": train_config.as_dict(),
            "count": count,
            "seed": seed,
            "identity": stage_identity,
            "checkpoint": str(stage_checkpoint),
            "payload": str(payload_path),
            "queries": str(queries_path),
            "documents": str(documents_path),
            "output": str(stage_output),
            "kind": kind,
            "retained_mlm_weight": retained_weight,
            "trace_output": str(Path(out_dir) / f"{kind}-resumed-trace.json"),
        }
        stage_instruction_path = Path(out_dir) / f"{kind}-instruction.json"
        stage_instruction_path.write_text(json.dumps(stage_instruction))
        stage_child = dict(os.environ, JAX_PLATFORMS="cpu")
        subprocess.run(
            [
                sys.executable,
                str(PROJECT / "tests" / "resume_worker.py"),
                str(stage_instruction_path),
            ],
            cwd=stage_output.parent,
            env=stage_child,
            check=True,
            timeout=180,
        )
        stage_template = initialize_state(
            model, config, count, seed=seed, train_config=train_config
        )
        stage_restored, _ = load_checkpoint(
            stage_output, stage_template, stage_identity
        )
        _compare_state(expected_stage, stage_restored)
        resumed_stage_trace = json.loads(
            Path(stage_instruction["trace_output"]).read_text()
        )
        stage_expected_trace = expected_stats["trace"][0]
        keys = (
            ("indices", "dropout_key", "learning_rate")
            if kind in {"weak", "weak_only"}
            else ("query_indices", "dropout_key", "candidate_doc_ids", "learning_rate")
        )
        for key in keys:
            if resumed_stage_trace[0][key] != stage_expected_trace[key]:
                raise AssertionError(f"fresh {kind} resume changed its next {key}")

    # The epoch tail is consumed as its own partial batch; the next step starts a
    # fresh shuffled epoch, so every row appears once per traversal.
    cursor_state = initialize_state(
        model, config, 10, seed=12, train_config=train_config
    )
    seen = []
    batch_lengths = []
    for _ in range(3):
        cursor_state, indices, _ = next_indices(cursor_state, 10, 4)
        seen.extend(map(int, indices))
        batch_lengths.append(len(indices))
    if (
        sorted(seen) != list(range(10))
        or len(set(seen)) != 10
        or batch_lengths != [4, 4, 2]
    ):
        raise AssertionError("partial-tail sampler skipped or duplicated an example")
    cursor_state, indices, _ = next_indices(cursor_state, 10, 4)
    if int(cursor_state["epoch"]) != 1 or int(cursor_state["cursor"]) != 4:
        raise AssertionError(
            "epoch boundary did not advance cursor/permutation exactly once"
        )
    wrong = dict(identity, dataset="mutated")
    try:
        load_checkpoint(checkpoint, template, wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("checkpoint accepted a changed source/data identity")

    def leaf_index_for(key):
        for index, (path, _) in enumerate(
            jax.tree_util.tree_flatten_with_path(template)[0]
        ):
            names = [part.key for part in path if hasattr(part, "key")]
            if key in names:
                return index
        raise AssertionError(f"template has no {key} checkpoint leaf")

    mutation_indices = {
        "moment_nan": leaf_index_for("m"),
        "moment_dtype": leaf_index_for("m"),
        "permutation": leaf_index_for("permutation"),
        "cursor": leaf_index_for("cursor"),
    }
    for name, target in mutation_indices.items():
        corrupted = Path(out_dir) / f"mutated-{name}"
        shutil.copytree(checkpoint, corrupted)
        with np.load(corrupted / "state.npz", allow_pickle=False) as archive:
            leaves = {key: archive[key].copy() for key in archive.files}
        target_name = f"leaf_{target:04d}"
        if name == "moment_nan":
            leaves[target_name].flat[0] = np.nan
        elif name == "moment_dtype":
            leaves[target_name] = leaves[target_name].astype(np.int32)
        elif name == "permutation":
            leaves[target_name] = np.zeros_like(leaves[target_name])
        else:
            leaves[target_name] = np.asarray(-1, dtype=leaves[target_name].dtype)
        np.savez(corrupted / "state.npz", **leaves)
        manifest_path = corrupted / "checkpoint.json"
        corrupted_manifest = json.loads(manifest_path.read_text())
        corrupted_manifest["state_sha256"] = _sha(
            (corrupted / "state.npz").read_bytes()
        )
        manifest_path.write_text(json.dumps(corrupted_manifest))
        try:
            load_checkpoint(corrupted, template, identity)
        except ValueError:
            pass
        else:
            raise AssertionError(f"checkpoint accepted semantic mutation: {name}")
    return {
        "fresh_process_next_update_exact": True,
        "fresh_process_resume_stages": [
            "phase1_mlm",
            "phase2_contrastive_only",
            "phase2_joint",
            "phase3_human",
        ],
        "fresh_process_next_sample_and_dropout_keys_equal": True,
        "partial_tail_coverage": batch_lengths,
        "changed_identity_rejected": True,
        "rechecksummed_semantic_mutations_rejected": sorted(mutation_indices),
        "next_batch_indices": expected_trace["trace"][0]["indices"],
    }


def _independent_metrics(raw_scores, query_ids, doc_ids, claims):
    relevance = {
        int(row["id"]): set(map(int, row.get("evidence", {}))) for row in claims
    }
    per_query = []
    for row_index, query_id in enumerate(map(int, query_ids)):
        relevant = relevance.get(query_id, set())
        if not relevant:
            per_query.append(
                {
                    "query_id": query_id,
                    "judged": False,
                    "relevant_count": 0,
                    "first_relevant_rank": None,
                }
            )
            continue
        order = np.lexsort((doc_ids, -raw_scores[row_index]))
        relevant_ranks = [
            rank + 1
            for rank, position in enumerate(order)
            if int(doc_ids[position]) in relevant
        ]
        first = min(relevant_ranks) if relevant_ranks else None
        row = {
            "query_id": query_id,
            "judged": True,
            "relevant_count": len(relevant),
            "first_relevant_rank": first,
            "mrr": 0.0 if first is None else 1.0 / first,
        }
        for k in (1, 5, 10):
            hits = sum(rank <= k for rank in relevant_ranks)
            row[f"recall@{k}"] = hits / len(relevant)
            dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks if rank <= k)
            ideal = sum(
                1.0 / math.log2(rank + 1)
                for rank in range(1, min(len(relevant), k) + 1)
            )
            row[f"ndcg@{k}"] = dcg / ideal
        per_query.append(row)
    judged = [row for row in per_query if row["judged"]]
    result = {
        "queries_total": len(per_query),
        "queries_judged": len(judged),
        "queries_unjudged": len(per_query) - len(judged),
        "per_query": per_query,
        "judgment_source": "SciFact claims[*].evidence document IDs only",
        "unjudged_queries_excluded": len(per_query) - len(judged),
    }
    for metric in (
        "recall@1",
        "recall@5",
        "recall@10",
        "ndcg@1",
        "ndcg@5",
        "ndcg@10",
        "mrr",
    ):
        result[metric] = (
            float(np.mean([row[metric] for row in judged])) if judged else 0.0
        )
    return result


def _compare_metrics(saved, actual, label):
    for key in (
        "queries_total",
        "queries_judged",
        "queries_unjudged",
        "judgment_source",
    ):
        if saved.get(key) != actual.get(key):
            raise AssertionError(f"{label} metric inventory mismatch for {key}")
    for key in (
        "recall@1",
        "recall@5",
        "recall@10",
        "ndcg@1",
        "ndcg@5",
        "ndcg@10",
        "mrr",
    ):
        np.testing.assert_allclose(
            saved[key], actual[key], rtol=1e-7, atol=1e-8, err_msg=f"{label} {key}"
        )
    if len(saved.get("per_query", [])) != len(actual["per_query"]):
        raise AssertionError(f"{label} per-query inventory length mismatch")
    for saved_row, actual_row in zip(saved["per_query"], actual["per_query"]):
        for key in ("query_id", "judged", "relevant_count", "first_relevant_rank"):
            if saved_row.get(key) != actual_row.get(key):
                raise AssertionError(f"{label} per-query {key} mismatch")
        if actual_row["judged"]:
            for key in (
                "recall@1",
                "recall@5",
                "recall@10",
                "ndcg@1",
                "ndcg@5",
                "ndcg@10",
                "mrr",
            ):
                np.testing.assert_allclose(
                    saved_row[key],
                    actual_row[key],
                    rtol=1e-7,
                    atol=1e-8,
                    err_msg=f"{label} query {actual_row['query_id']} {key}",
                )


def _independent_bm25(documents, claims):
    def terms(text):
        return re.findall(r"\w+", text.casefold())

    tokenized = [
        terms((row.get("title", "") or "") + " " + " ".join(row.get("abstract") or []))
        for row in documents
    ]
    counters = [dict((term, row.count(term)) for term in set(row)) for row in tokenized]
    doc_ids = np.asarray([int(row["doc_id"]) for row in documents], np.int64)
    lengths = np.asarray([len(row) for row in tokenized], np.float64)
    average = float(lengths.mean())
    df = {}
    for counts in counters:
        for term in counts:
            df[term] = df.get(term, 0) + 1
    result = np.zeros((len(claims), len(documents)), np.float64)
    for qi, claim in enumerate(claims):
        for term in terms(claim["claim"]):
            frequency = df.get(term, 0)
            inverse = math.log(
                1 + (len(documents) - frequency + 0.5) / (frequency + 0.5)
            )
            for di, counts in enumerate(counters):
                tf = counts.get(term, 0)
                denom = tf + 1.5 * (0.25 + 0.75 * lengths[di] / average)
                result[qi, di] += inverse * tf * 2.5 / max(denom, 1e-12)
    return np.asarray([int(row["id"]) for row in claims], np.int64), doc_ids, result


def _verify_population_raw(path, model, params, config, docs, claims, tokenizer, label):
    data = np.load(path, allow_pickle=False)
    independent = evaluation.encode_model(
        model, params, config, docs, claims, tokenizer
    )
    scores = (
        independent["query_vectors"].astype(np.float64)
        @ independent["document_vectors"].astype(np.float64).T
    )
    np.testing.assert_allclose(
        data["query_vectors"], independent["query_vectors"], rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        data["document_vectors"], independent["document_vectors"], rtol=2e-5, atol=2e-6
    )
    np.testing.assert_array_equal(data["query_ids"], independent["query_ids"])
    np.testing.assert_array_equal(data["document_ids"], independent["document_ids"])
    np.testing.assert_allclose(data["scores"], scores, rtol=2e-5, atol=2e-5)
    actual = _independent_metrics(
        scores, independent["query_ids"], independent["document_ids"], claims
    )
    saved = json.loads(Path(path).with_name("metrics.json").read_text())
    _compare_metrics(saved, actual, label)
    for row, query_id in enumerate(independent["query_ids"]):
        order = np.lexsort((independent["document_ids"], -scores[row]))
        expected = independent["document_ids"][order]
        np.testing.assert_array_equal(
            data["ranked_doc_ids"][row], expected.astype(np.uint32)
        )
    return actual


def _independent_selected_token_count(ids, key_data, vocab_size, pad_id):
    values = jnp.asarray(ids, jnp.int32)
    key = jax.random.wrap_key_data(jnp.asarray(key_data, jnp.uint32))
    choose_key, _, _ = jax.random.split(key, 3)
    eligible = values >= 5
    selected = jax.random.bernoulli(choose_key, 0.15, values.shape) & eligible
    missing = ~jnp.any(selected, axis=1)
    first = jnp.argmax(eligible, axis=1)
    fallback = jax.nn.one_hot(first, values.shape[1], dtype=jnp.bool_) & eligible
    selected = selected | (missing[:, None] & fallback)
    if not np.array_equal(np.asarray(values == pad_id), np.asarray(values == 0)):
        raise AssertionError("frozen pilot PAD token ID is not the expected zero")
    return int(np.asarray(selected).sum())


def _independent_measured_exposure(
    traces,
    selected_ids,
    mlm_values,
    pad_id,
    pair_inventory,
    recipe_dev_doc_ids,
    fit_claims,
):
    """Rebuild consumed IDs and token totals from the recorded step inputs."""
    phase1 = [row for row in traces if row["stage"] == "phase1_mlm"]
    weak = [row for row in traces if row["stage"] == "phase2_contrastive_only"]
    joint = [
        row for row in traces if row["stage"] == "phase2_contrastive_plus_retained_mlm"
    ]
    human = [row for row in traces if row["stage"] == "phase3_human_pairs"]
    phase1_docs = sorted(
        {int(selected_ids[index]) for row in phase1 for index in row["indices"]}
    )
    joint_docs = sorted(
        {int(selected_ids[index]) for row in joint for index in row["mlm_indices"]}
    )
    weak_indices = sorted({int(index) for row in weak for index in row["indices"]})
    weak_docs = sorted(
        {int(pair_inventory[index]["group_id"]) for index in weak_indices}
    )
    joint_indices = sorted({int(index) for row in joint for index in row["indices"]})
    query_ids = sorted(
        {int(query_id) for row in human for query_id in row["query_ids"]}
    )
    query_to_positive = {
        int(row["id"]): set(map(int, row.get("evidence", {}))) for row in fit_claims
    }
    positive_docs = sorted(
        {doc_id for query_id in query_ids for doc_id in query_to_positive[query_id]}
    )
    return {
        "phase1_mlm": {
            "consumed_document_ids": phase1_docs,
            "unique_documents": len(phase1_docs),
            "recipe_development_documents_consumed": sorted(
                set(phase1_docs) & recipe_dev_doc_ids
            ),
            "recipe_development_documents_consumed_count": len(
                set(phase1_docs) & recipe_dev_doc_ids
            ),
            "nonpadding_input_tokens": sum(
                int(np.count_nonzero(mlm_values[row["indices"]] != pad_id))
                for row in phase1
            ),
            "selected_masked_tokens": sum(row["selected_tokens"] for row in phase1),
        },
        "phase2_contrastive_only": {
            "consumed_weak_pair_indices": weak_indices,
            "weak_pair_source_document_ids": weak_docs,
            "unique_weak_pairs": len(weak_indices),
            "nonpadding_pair_input_tokens": sum(
                row["pair_input_tokens"] for row in weak
            ),
        },
        "phase2_contrastive_plus_retained_mlm": {
            "consumed_weak_pair_indices": joint_indices,
            "unique_weak_pairs": len(joint_indices),
            "weak_pair_source_document_ids": sorted(
                {
                    int(pair_inventory[index]["group_id"])
                    for row in joint
                    for index in row["indices"]
                }
            ),
            "retained_mlm_consumed_document_ids": joint_docs,
            "unique_retained_mlm_documents": len(joint_docs),
            "recipe_development_documents_consumed": sorted(
                set(joint_docs) & recipe_dev_doc_ids
            ),
            "recipe_development_documents_consumed_count": len(
                set(joint_docs) & recipe_dev_doc_ids
            ),
            "nonpadding_pair_input_tokens": sum(
                row["pair_input_tokens"] for row in joint
            ),
            "nonpadding_retained_mlm_input_tokens": sum(
                row["mlm_input_tokens"] for row in joint
            ),
            "selected_retained_mlm_tokens": sum(
                row["mlm_selected_tokens"] for row in joint
            ),
        },
        "phase3_human_pairs": {
            "consumed_fit_query_ids": query_ids,
            "unique_fit_queries": len(query_ids),
            "human_evidence_positive_document_ids": positive_docs,
            "unique_positive_documents": len(positive_docs),
            "nonpadding_query_input_tokens": sum(
                row["query_input_tokens"] for row in human
            ),
            "nonpadding_candidate_document_input_tokens": sum(
                row["candidate_document_input_tokens"] for row in human
            ),
        },
    }


def _load_checkpoint_for_eval(
    model, folder, config, row_count, seed, train_cfg, parent_params=None, global_step=0
):
    template = initialize_state(
        model,
        config,
        row_count,
        seed=seed,
        train_config=train_cfg,
        initial_params=parent_params,
        initial_global_step=global_step,
    )
    manifest = json.loads((Path(folder) / "checkpoint.json").read_text())
    state, _ = load_checkpoint(folder, template, manifest["identity"])
    return state["params"], manifest


def _check_full_run(data_dir, output, implementation, frozen, baseline_snapshot=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    offline_baseline = baseline_snapshot is not None
    if offline_baseline:
        shutil.copytree(baseline_snapshot, output / "baseline_snapshot")
    report = lifecycle.run(data_dir, output, implementation, "all", offline_baseline)
    report_path = output / "pilot-report.json"
    receipt_path = output / "sourcebound-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    reported = json.loads(report_path.read_text())
    recipe_bytes = (PROJECT / "data" / "frozen_recipe.json").read_bytes()
    if reported["frozen_recipe_input_sha256"] != _sha(recipe_bytes):
        raise AssertionError("run did not snapshot the exact current frozen recipe")
    if receipt["frozen_recipe_sha256"] != _sha(recipe_bytes):
        raise AssertionError("source-bound receipt is not bound to the frozen recipe")
    for relative, expected_hash in receipt["run_artifacts_sha256"].items():
        if _sha((output / relative).read_bytes()) != expected_hash:
            raise AssertionError(f"run artifact changed after receipt: {relative}")
    package_dir = output / "local-package"
    package_card = (package_dir / "README.md").read_bytes()
    package_metadata = json.loads((package_dir / "package.json").read_text())
    card_hash = _sha(package_card)
    card_text = package_card.decode("utf-8")
    provenance_bytes = (package_dir / "DATA_PROVENANCE.md").read_bytes()
    evidence_bytes = (package_dir / "MEASURED_EVIDENCE.json").read_bytes()
    if (
        package_metadata.get("model_card_sha256") != card_hash
        or package_metadata.get("data_provenance_sha256") != _sha(provenance_bytes)
        or package_metadata.get("measured_evidence_sha256") != _sha(evidence_bytes)
        or reported.get("local_package", {}).get("model_card_sha256") != card_hash
        or "S2ORC under ODC-By 1.0" not in card_text
        or "not a blinded benchmark" not in card_text
        or "LICENSE.md" not in provenance_bytes.decode("utf-8")
    ):
        raise AssertionError(
            "consumer model card is missing its measured result, rights or limits"
        )
    exposure = reported["exposure"]
    if (
        exposure["recipe_development_documents_in_tokenizer"] != 29
        or exposure["recipe_development_documents_in_mlm_sample"] != 3
        or exposure["recipe_development_query_text_training_examples"] != 0
        or exposure["recipe_development_human_pair_training_examples"] != 0
    ):
        raise AssertionError(
            "training-exposure disclosure differs from frozen membership"
        )
    model, candidate_path = _load_candidate(implementation)
    out = Path(output)
    recipe = json.loads((PROJECT / "data" / "frozen_recipe.json").read_text())
    processor = __import__("tokenizers").Tokenizer.from_file(
        str(out / "tokenizer.json")
    )
    config = EncoderConfig(**json.loads((out / "model-config.json").read_text()))
    corpus, fit, recipe_dev, final = (
        frozen["corpus"],
        frozen["fit"],
        frozen["recipe_dev"],
        frozen["final"],
    )
    pretraining_ids = frozen["inventories"]["ids"]["pretraining_doc_ids"]
    prepared_pilot_docs = dataset.select_pilot_documents(corpus, pretraining_ids, 128)
    selected_ids = [int(row["doc_id"]) for row in prepared_pilot_docs]
    if reported["selected_mlm_document_ids"] != selected_ids:
        raise AssertionError(
            "prepared MLM document inventory differs from frozen selection"
        )
    mlm_values = np.load(out / "mlm-input-ids.npy", allow_pickle=False)
    if mlm_values.shape != (128, config.max_length):
        raise AssertionError("prepared MLM tensor has an unexpected shape")
    trace = json.loads((out / "training-trace.json").read_text())
    phase1_trace = [row for row in trace if row["stage"] == "phase1_mlm"]
    joint_trace = [
        row for row in trace if row["stage"] == "phase2_contrastive_plus_retained_mlm"
    ]
    for row in phase1_trace:
        batch = mlm_values[row["indices"]]
        if int(np.count_nonzero(batch != config.pad_id)) != row["input_tokens"]:
            raise AssertionError("phase-1 trace input-token count is not reproducible")
        selected_count = _independent_selected_token_count(
            batch, row["mask_key"], config.vocab_size, config.pad_id
        )
        if selected_count != row["selected_tokens"]:
            raise AssertionError(
                "phase-1 trace selected-token count is not reproducible"
            )
    for row in joint_trace:
        expected_indices = [int(index) % len(mlm_values) for index in row["indices"]]
        if row["mlm_indices"] != expected_indices or row["mlm_key"] is None:
            raise AssertionError("retained-MLM sampler indices or key are absent")
        batch = mlm_values[expected_indices]
        if int(np.count_nonzero(batch != config.pad_id)) != row["mlm_input_tokens"]:
            raise AssertionError("retained-MLM input-token count is not reproducible")
        selected_count = _independent_selected_token_count(
            batch, row["mlm_key"], config.vocab_size, config.pad_id
        )
        if selected_count != row["mlm_selected_tokens"]:
            raise AssertionError(
                "retained-MLM selected-token count is not reproducible"
            )
    pair_inventory = json.loads((out / "weak-pair-inventory.json").read_text())["pairs"]
    expected_exposure = _independent_measured_exposure(
        trace,
        selected_ids,
        mlm_values,
        config.pad_id,
        pair_inventory,
        set(map(int, frozen["inventories"]["ids"]["recipe_dev_train_doc_ids"])),
        fit,
    )
    if reported["measured_training_exposure"] != expected_exposure:
        raise AssertionError(
            "run report training exposure differs from consumed traces"
        )
    packaged_evidence = json.loads((package_dir / "MEASURED_EVIDENCE.json").read_text())
    if packaged_evidence[
        "measured_training_exposure"
    ] != expected_exposure or packaged_evidence["evaluation"] != {
        key: value
        for key, value in reported["evaluation"].items()
        if key.endswith("/final_official_dev_custom_protocol")
    }:
        raise AssertionError("consumer bundle omitted measured exposure or results")
    base = TrainConfig(
        batch_size=4,
        learning_rate=0.0003,
        weight_decay=0.01,
        warmup_steps=1,
        total_steps=8,
        seed=20261021,
    )
    phase1_cfg = replace(base, total_steps=8, learning_rate=0.0003)
    phase2_cfg = replace(base, total_steps=8, learning_rate=0.0001)
    phase3_cfg = replace(base, total_steps=71, learning_rate=0.00005)
    states, manifests = {}, {}
    states["phase1_mlm"], manifests["phase1_mlm"] = _load_checkpoint_for_eval(
        model, out / "checkpoints" / "phase1-mlm", config, 128, 20261021, phase1_cfg
    )
    states["phase2_contrastive_only"], manifests["phase2_contrastive_only"] = (
        _load_checkpoint_for_eval(
            model,
            out / "checkpoints" / "phase2_contrastive_only",
            config,
            32,
            20261022,
            phase2_cfg,
            states["phase1_mlm"],
            8,
        )
    )
    (
        states["phase2_contrastive_plus_retained_mlm"],
        manifests["phase2_contrastive_plus_retained_mlm"],
    ) = _load_checkpoint_for_eval(
        model,
        out / "checkpoints" / "phase2_contrastive_plus_retained_mlm",
        config,
        32,
        20261022,
        phase2_cfg,
        states["phase1_mlm"],
        8,
    )
    states["phase3_human_pairs"], manifests["phase3_human_pairs"] = (
        _load_checkpoint_for_eval(
            model,
            out / "checkpoints" / "phase3-human-pairs",
            config,
            281,
            20261023,
            phase3_cfg,
            states["phase2_contrastive_plus_retained_mlm"],
            16,
        )
    )
    random_state = initialize_state(
        model, config, 128, seed=20261021, train_config=phase1_cfg
    )
    states["random_initialization"] = random_state["params"]
    expected_hashes = report["source_hashes"]
    independently_verified_final = {}
    for name, state in states.items():
        for pop_name, claims in (
            ("recipe_development", recipe_dev),
            ("final_official_dev_custom_protocol", final),
        ):
            raw_path = out / "raw" / name / pop_name / "retrieval-arrays.npz"
            independent_metrics = _verify_population_raw(
                raw_path,
                model,
                state,
                config,
                corpus,
                claims,
                processor,
                name + "/" + pop_name,
            )
            if pop_name == "final_official_dev_custom_protocol":
                independently_verified_final[name] = independent_metrics
            report_metrics = report["evaluation"].get(name + "/" + pop_name)
            if report_metrics is None:
                raise AssertionError(
                    f"run report omits evaluated population {name}/{pop_name}"
                )
            _compare_metrics(
                report_metrics,
                independent_metrics,
                "run report " + name + "/" + pop_name,
            )

    # Branches begin with identical optimizer/sampler/dropout state and consume
    # exactly the same 32 weak pairs; only the retained-MLM objective changes.
    trace = json.loads((out / "training-trace.json").read_text())
    branch_a = [row for row in trace if row["stage"] == "phase2_contrastive_only"]
    branch_b = [
        row for row in trace if row["stage"] == "phase2_contrastive_plus_retained_mlm"
    ]
    if len(branch_a) != 8 or len(branch_b) != 8:
        raise AssertionError(
            "phase 2 ablation did not consume the frozen eight updates per branch"
        )
    for a, b in zip(branch_a, branch_b):
        for key in ("indices", "dropout_key", "learning_rate"):
            if a[key] != b[key]:
                raise AssertionError(
                    f"phase 2 branches did not match pair exposure for {key}"
                )
    p3_indices = [
        index
        for row in trace
        if row["stage"] == "phase3_human_pairs"
        for index in row["query_indices"]
    ]
    if (
        len(p3_indices) != 281
        or len(set(p3_indices)) != 281
        or sorted(p3_indices) != list(range(281))
    ):
        raise AssertionError(
            "phase 3 did not cover every judged fit query exactly once"
        )

    # Independently regenerate the external baseline and BM25 raw results.
    baseline, baseline_receipt = load_pinned_baseline(
        out / "baseline_snapshot", offline=True
    )
    doc_texts = [dataset.document_text(row) for row in corpus]
    baseline_docs = baseline.encode(
        doc_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    for pop_name, claims in (
        ("recipe_development", recipe_dev),
        ("final_official_dev_custom_protocol", final),
    ):
        path = out / "raw" / "external_minilm_l6_v2" / pop_name / "retrieval-arrays.npz"
        raw = np.load(path, allow_pickle=False)
        qv = baseline.encode(
            [row["claim"] for row in claims],
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores = np.asarray(qv, np.float64) @ np.asarray(baseline_docs, np.float64).T
        np.testing.assert_allclose(raw["query_vectors"], qv, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            raw["document_vectors"], baseline_docs, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(raw["scores"], scores, rtol=2e-5, atol=2e-5)
        metrics = _independent_metrics(
            scores, raw["query_ids"], raw["document_ids"], claims
        )
        _compare_metrics(
            json.loads(path.with_name("metrics.json").read_text()),
            metrics,
            "external baseline",
        )
        _compare_metrics(
            report["evaluation"]["external_minilm_l6_v2/" + pop_name],
            metrics,
            "run report external baseline",
        )

        qids, doc_ids, bm25 = _independent_bm25(corpus, claims)
        bm25_raw = np.load(
            out / "raw" / "bm25" / pop_name / "rankings.npz", allow_pickle=False
        )
        np.testing.assert_array_equal(bm25_raw["query_ids"], qids)
        np.testing.assert_array_equal(bm25_raw["document_ids"], doc_ids)
        np.testing.assert_allclose(bm25_raw["scores"], bm25, rtol=1e-6, atol=1e-6)
        rankings = np.asarray(
            [doc_ids[np.lexsort((doc_ids, -row))] for row in bm25], np.uint32
        )
        np.testing.assert_array_equal(bm25_raw["ranked_doc_ids"], rankings)
        bm_metrics = _independent_metrics(bm25, qids, doc_ids, claims)
        _compare_metrics(
            json.loads((out / "raw" / "bm25" / pop_name / "metrics.json").read_text()),
            bm_metrics,
            "BM25",
        )
        _compare_metrics(
            report["evaluation"]["bm25/" + pop_name],
            bm_metrics,
            "run report BM25",
        )

    source_member = json.loads((out / "pilot-report.json").read_text())
    if source_member.get("target_300m_qualified") or source_member.get(
        "public_release_qualified"
    ):
        raise AssertionError("bounded pilot falsely claims target qualification")
    if (
        source_member["target_parameter_count"] != 302090240
        or source_member["target_with_mlm_head_parameter_count"] != 303191040
    ):
        raise AssertionError("run receipt target architecture count changed")
    if (
        source_member["processor_info"]["training_doc_ids"]
        != frozen["inventories"]["ids"]["pretraining_doc_ids"]
    ):
        raise AssertionError(
            "tokenizer included reserved documents or omitted eligible inventory"
        )
    if (
        source_member["exposure"]["final_judged_queries"] != 188
        or source_member["exposure"]["final_query_inventory"] != 300
    ):
        raise AssertionError(
            "final evaluation inventory misstated judged versus unjudged queries"
        )
    if len(baseline_receipt["snapshot_files_sha256"]) < 5:
        raise AssertionError("external baseline receipt lacks pinned local model files")
    phase3 = independently_verified_final["phase3_human_pairs"]
    phase1 = independently_verified_final["phase1_mlm"]
    for label, name in (
        ("Phase 3 human-evidence pilot", "phase3_human_pairs"),
        ("Phase 1 MLM", "phase1_mlm"),
        ("Frozen MiniLM-L6-v2 baseline", "external_minilm_l6_v2"),
        ("BM25 lexical baseline", "bm25"),
    ):
        metrics = report["evaluation"][name + "/final_official_dev_custom_protocol"]
        expected_row = "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            label,
            metrics["recall@1"],
            metrics["recall@10"],
            metrics["mrr"],
            metrics["ndcg@10"],
        )
        if expected_row not in card_text:
            raise AssertionError("consumer model card omits its measured metric row")
    floors = {"ndcg@10": 0.10, "recall@10": 0.30}
    floors_pass = (
        phase3["ndcg@10"] >= floors["ndcg@10"]
        and phase3["recall@10"] >= floors["recall@10"]
    )
    retention_drop = phase1["ndcg@10"] - phase3["ndcg@10"]
    gate_pass = floors_pass and retention_drop <= 0.05
    quality = source_member["quality_gates"]
    if (
        quality["precommitted_floors"] != floors
        or quality["phase3_final_floors_pass"] != floors_pass
        or not math.isclose(
            quality["phase1_to_phase3_ndcg10_drop"],
            retention_drop,
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
        or quality["pilot_quality_gate_pass"] != gate_pass
        or ("has passed" in card_text) != floors_pass
        or ("**passed** these floors" in card_text) != floors_pass
        or ("**failed** these floors" in card_text) == floors_pass
    ):
        raise AssertionError(
            "report/model-card gate claims disagree with independently recomputed retrieval metrics"
        )
    return {
        "full_lifecycle": report["status"],
        "candidate_stages_independently_ranked": list(states),
        "external_baseline_revision": baseline_receipt["revision"],
        "independent_exact_scores_and_metrics": True,
        "bm25_scores_regenerated": True,
        "matched_phase2_exposure": 32,
        "phase3_fit_queries_covered": 281,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", default="solution")
    parser.add_argument("--stage", choices=("all",), default="all")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--baseline-snapshot",
        type=Path,
        default=None,
        help="optional exact pinned local baseline snapshot for offline checking",
    )
    args = parser.parse_args()
    output = args.output or Path(
        tempfile.mkdtemp(prefix="jaxpathways-embedding-c01-check-")
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = _check_data(args.data.resolve())
    candidate = _load_candidate(args.implementation)
    model_checks = _check_model(candidate)
    resume_checks = _check_resume(candidate, output / "resume")
    full_checks = _check_full_run(
        args.data.resolve(),
        output / "full-run",
        args.implementation,
        frozen,
        None if args.baseline_snapshot is None else args.baseline_snapshot.resolve(),
    )
    result = {
        "status": "pass",
        "stage": args.stage,
        "output": str(output),
        "data": {
            "archive_sha256": frozen["manifest"]["source"]["archive_sha256"],
            "corpus_docs": len(frozen["corpus"]),
            "fit_claims": len(frozen["fit"]),
            "recipe_development_claims": len(frozen["recipe_dev"]),
            "final_claims": len(frozen["final"]),
            "final_judged_claims": sum(
                bool(row.get("evidence")) for row in frozen["final"]
            ),
        },
        "model": model_checks,
        "resume": resume_checks,
        "lifecycle": full_checks,
        "limitations": [
            "CPU pilot is not 300M target training",
            "one seed is insufficient for a training-variance claim",
            "SciFact dev is the custom final evaluation because official test lacks public labels",
            "MTEB and human review remain unrun",
        ],
    }
    (output / "independent-check.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
