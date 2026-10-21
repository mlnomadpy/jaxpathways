"""FP32 Adam, deterministic example order, complete checkpoint state and metrics."""

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

try:
    from .config import ModelConfig, TrainConfig
    from . import model
except ImportError:  # Loaded from a project-local script or explicit module path.
    from config import ModelConfig, TrainConfig
    import model


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def code_identity():
    folder = Path(__file__).resolve().parent
    names = ("config.py", "data.py", "model.py", "train.py")
    return {name: _digest((folder / name).read_bytes()) for name in names}


def experiment_identity(
    source_manifest,
    processor_info,
    packed_stats,
    config,
    train_config,
    candidate_model_sha256=None,
):
    return {
        "source_sha256": source_manifest["source_sha256"],
        "processor_sha256": processor_info["processor_sha256"],
        "dataset_sha256": packed_stats["dataset_sha256"],
        "model_config_sha256": config.fingerprint(),
        "training_config_sha256": train_config.fingerprint(),
        "code_sha256": code_identity(),
        "candidate_model_sha256": candidate_model_sha256,
        "jax_version": jax.__version__,
        "platform": jax.default_backend(),
    }


def _expand_document_order(document_order, document_index):
    return jnp.asarray(
        np.concatenate(
            [
                np.flatnonzero(np.asarray(document_index) == int(document))
                for document in np.asarray(document_order)
            ]
        ),
        dtype=jnp.int32,
    )


def initialize_training(examples, model_config, train_config):
    cfg, tcfg = model_config, train_config
    cfg.validate()
    tcfg.validate()
    num_examples = len(examples["tokens"])
    num_documents = len(examples["stats"]["record_ids"])
    if num_examples < tcfg.batch_size or num_documents < 1:
        raise ValueError("training split must contain at least one full minibatch")
    seed = jax.random.key(tcfg.seed)
    init_key, order_key, state_key = jax.random.split(seed, 3)
    params = model.initialize(init_key, cfg)
    moments = jax.tree.map(jnp.zeros_like, params)
    document_order = jax.random.permutation(order_key, num_documents).astype(jnp.int32)
    order = _expand_document_order(document_order, examples["document_index"])
    state = {
        "params": params,
        "m": moments,
        "v": moments,
        "step": jnp.asarray(0, jnp.int32),
        "key": jax.random.key_data(state_key),
        "order": order,
        "document_order": document_order,
        "epoch": jnp.asarray(0, jnp.int32),
        "cursor": jnp.asarray(0, jnp.int32),
    }
    return state


def learning_rate(step, config):
    config.validate()
    step = int(step)
    if config.warmup_steps and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    decay_steps = max(1, config.total_steps - config.warmup_steps)
    progress = min(1.0, max(0.0, (step - config.warmup_steps) / decay_steps))
    return config.learning_rate + progress * (
        config.min_learning_rate - config.learning_rate
    )


def take_batch(state, examples, config):
    """Select the next full batch, refreshing a new deterministic epoch on boundary."""
    n = len(examples["tokens"])
    batch_size = config.batch_size
    cursor, epoch = int(state["cursor"]), int(state["epoch"])
    key = jax.random.wrap_key_data(jnp.asarray(state["key"], dtype=jnp.uint32))
    order = state["order"]
    document_order = state["document_order"]
    chosen_parts = []
    while sum(len(part) for part in chosen_parts) < batch_size:
        if cursor >= n:
            key, shuffle_key = jax.random.split(key)
            document_order = jax.random.permutation(
                shuffle_key, len(examples["stats"]["record_ids"])
            ).astype(jnp.int32)
            order = _expand_document_order(document_order, examples["document_index"])
            cursor, epoch = 0, epoch + 1
        available = n - cursor
        take = min(batch_size - sum(len(part) for part in chosen_parts), available)
        chosen_parts.append(np.asarray(order[cursor : cursor + take], dtype=np.int32))
        cursor += take
    chosen = np.concatenate(chosen_parts)
    key, dropout_key = jax.random.split(key)
    next_state = {
        **state,
        "key": jax.random.key_data(key),
        "order": order,
        "document_order": document_order,
        "epoch": jnp.asarray(epoch, jnp.int32),
        # `cursor` already includes every selected example, including the tail
        # before an epoch rollover. Adding batch_size here skips records.
        "cursor": jnp.asarray(cursor, jnp.int32),
    }
    batch = {
        name: jnp.asarray(examples[name][chosen]) for name in ("tokens", "segments")
    }
    return next_state, batch, chosen, dropout_key


def _adam_update(params, m, v, grads, step, learning_rate_value, config):
    count = int(step) + 1
    beta1, beta2 = config.beta1, config.beta2
    m = jax.tree.map(lambda old, grad: beta1 * old + (1.0 - beta1) * grad, m, grads)
    v = jax.tree.map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad), v, grads
    )
    correction1, correction2 = 1.0 - beta1**count, 1.0 - beta2**count
    params = jax.tree.map(
        lambda param, first, second: param
        - learning_rate_value
        * (first / correction1)
        / (jnp.sqrt(second / correction2) + config.epsilon),
        params,
        m,
        v,
    )
    return params, m, v


@lru_cache(maxsize=8)
def _compiled_gradient(model_config):
    return jax.jit(
        jax.value_and_grad(
            lambda params, tokens, segments, key: model.token_loss(
                params, tokens, segments, model_config, key
            )
        )
    )


def step(state, examples, model_config, train_config):
    next_state, batch, indices, dropout_key = take_batch(state, examples, train_config)
    loss, grads = _compiled_gradient(model_config)(
        state["params"], batch["tokens"], batch["segments"], dropout_key
    )
    rate = learning_rate(state["step"], train_config)
    params, m, v = _adam_update(
        state["params"],
        state["m"],
        state["v"],
        grads,
        state["step"],
        rate,
        train_config,
    )
    next_state.update(
        {
            "params": params,
            "m": m,
            "v": v,
            "step": jnp.asarray(int(state["step"]) + 1, jnp.int32),
        }
    )
    info = {
        "loss_nats_per_token": float(loss),
        "learning_rate": rate,
        "step": int(next_state["step"]),
        "epoch": int(next_state["epoch"]),
        "cursor": int(next_state["cursor"]),
        "record_indices": indices.tolist(),
        "dropout_key": np.asarray(jax.random.key_data(dropout_key)).tolist(),
    }
    return next_state, info


def evaluate(params, examples, model_config, batch_size=8, text_only=False):
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    total_nll, token_count = 0.0, 0
    for start in range(0, len(examples["tokens"]), batch_size):
        end = min(start + batch_size, len(examples["tokens"]))
        tokens = jnp.asarray(examples["tokens"][start:end])
        segments = jnp.asarray(examples["segments"][start:end])
        inputs, targets, input_segments, eligible = model.causal_targets(
            tokens, segments
        )
        predictions = model.logits(
            params, inputs, model_config, inputs != 0, input_segments
        )
        log_probs = jax.nn.log_softmax(predictions, axis=-1)
        selected = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
        score_mask = (
            jnp.asarray(examples["text_mask"][start:end]) if text_only else eligible
        )
        total_nll += float(-jnp.sum(jnp.where(score_mask, selected, 0.0)))
        token_count += int(jnp.sum(score_mask))
    nll = total_nll / max(token_count, 1)
    return {
        "nll_nats_per_token": nll,
        "tokens": token_count,
        "nll_nats": total_nll,
        "perplexity": math.exp(min(80.0, nll)),
    }


def _flatten_state(state):
    leaves, structure = jax.tree.flatten(state)
    return leaves, structure


def save_checkpoint(folder, state, identity):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    leaves, _ = _flatten_state(state)
    arrays = {f"leaf_{i:04d}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    np.savez(folder / "state.npz", **arrays)
    state_bytes = (folder / "state.npz").read_bytes()
    manifest = {
        "schema": 1,
        "identity": identity,
        "state_sha256": _digest(state_bytes),
        "leaves": [
            {"shape": list(value.shape), "dtype": str(value.dtype)}
            for value in arrays.values()
        ],
        "step": int(state["step"]),
        "epoch": int(state["epoch"]),
        "cursor": int(state["cursor"]),
    }
    (folder / "checkpoint.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2)
    )
    return manifest


def load_checkpoint(folder, template_state, expected_identity):
    folder = Path(folder)
    manifest = json.loads((folder / "checkpoint.json").read_text())
    raw = (folder / "state.npz").read_bytes()
    if manifest.get("schema") != 1 or manifest.get("identity") != expected_identity:
        raise ValueError(
            "checkpoint source/config/tokenizer/data/code identity mismatch"
        )
    if _digest(raw) != manifest.get("state_sha256"):
        raise ValueError("checkpoint state checksum mismatch")
    template_leaves, structure = _flatten_state(template_state)
    restored = []
    with np.load(folder / "state.npz", allow_pickle=False) as data:
        expected_names = {f"leaf_{i:04d}" for i in range(len(template_leaves))}
        if set(data.files) != expected_names:
            raise ValueError("checkpoint numeric leaf set mismatch")
        if len(manifest.get("leaves", [])) != len(template_leaves):
            raise ValueError("checkpoint leaf manifest length mismatch")
        for index, template in enumerate(template_leaves):
            array = data[f"leaf_{index:04d}"]
            declared = manifest["leaves"][index]
            if array.shape != np.asarray(template).shape or str(array.dtype) != str(
                np.asarray(template).dtype
            ):
                raise ValueError(f"checkpoint leaf {index} shape or dtype mismatch")
            if declared != {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            } or not np.all(np.isfinite(array)):
                raise ValueError(
                    f"checkpoint leaf {index} manifest or finite-value check failed"
                )
            restored.append(jnp.asarray(array))
    state = jax.tree.unflatten(structure, restored)
    if (
        int(state["step"]) != manifest["step"]
        or int(state["epoch"]) != manifest["epoch"]
        or int(state["cursor"]) != manifest["cursor"]
    ):
        raise ValueError("checkpoint progress fields do not match serialized state")
    return state, manifest


def save_model_package(folder, state, model_config, processor_path, identity):
    """Write an inference-only params/processor package with independent checksums."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    processor_bytes = Path(processor_path).read_bytes()
    (folder / "tokenizer.json").write_bytes(processor_bytes)
    leaves, _ = jax.tree.flatten(state["params"])
    arrays = {f"param_{i:04d}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    np.savez(folder / "params.npz", **arrays)
    params_bytes = (folder / "params.npz").read_bytes()
    manifest = {
        "schema": 1,
        "model_config": model_config.as_dict(),
        "model_config_sha256": model_config.fingerprint(),
        "identity": identity,
        "params_sha256": _digest(params_bytes),
        "processor_sha256": _digest(processor_bytes),
        "step": int(state["step"]),
        "parameter_count": model.parameter_count(model_config),
    }
    (folder / "package.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))
    return manifest


def load_model_package(folder, expected_identity):
    folder = Path(folder)
    manifest = json.loads((folder / "package.json").read_text())
    cfg = ModelConfig(**manifest["model_config"])
    cfg.validate()
    if (
        manifest.get("schema") != 1
        or manifest.get("identity") != expected_identity
        or manifest.get("model_config_sha256") != cfg.fingerprint()
    ):
        raise ValueError("model package config or identity mismatch")
    processor_bytes = (folder / "tokenizer.json").read_bytes()
    params_bytes = (folder / "params.npz").read_bytes()
    if _digest(processor_bytes) != manifest.get("processor_sha256") or _digest(
        params_bytes
    ) != manifest.get("params_sha256"):
        raise ValueError("model package processor or parameter checksum mismatch")
    template = model.initialize(jax.random.key(0), cfg)
    leaves, structure = jax.tree.flatten(template)
    restored = []
    with np.load(folder / "params.npz", allow_pickle=False) as data:
        if set(data.files) != {f"param_{i:04d}" for i in range(len(leaves))}:
            raise ValueError("model package parameter tree mismatch")
        for index, leaf in enumerate(leaves):
            array = data[f"param_{index:04d}"]
            if (
                array.shape != np.asarray(leaf).shape
                or array.dtype != np.asarray(leaf).dtype
                or not np.all(np.isfinite(array))
            ):
                raise ValueError(f"model package parameter {index} is invalid")
            restored.append(jnp.asarray(array))
    params = jax.tree.unflatten(structure, restored)
    if (
        sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(params))
        != manifest["parameter_count"]
    ):
        raise ValueError("model package parameter count mismatch")
    return params, cfg, manifest
