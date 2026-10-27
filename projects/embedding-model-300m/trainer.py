"""Deterministic, resumable CPU pilot training with explicit stage lineage."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

try:
    from . import objectives
except ImportError:
    import objectives


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def initialize_state(
    model,
    config,
    count,
    *,
    seed,
    train_config,
    initial_params=None,
    initial_global_step=0,
):
    if count < 2:
        raise ValueError("training requires at least two examples")
    config.validate()
    train_config.validate()
    root = jax.random.key(seed)
    init_key, order_key, rng_key = jax.random.split(root, 3)
    params = (
        model.initialize(init_key, config) if initial_params is None else initial_params
    )
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {
        "params": params,
        "m": zeros,
        "v": zeros,
        "step": jnp.asarray(0, jnp.int32),
        "total_step": jnp.asarray(0, jnp.int32),
        "global_step": jnp.asarray(initial_global_step, jnp.int32),
        "rng": jax.random.key_data(rng_key),
        "permutation": jax.random.permutation(order_key, count).astype(jnp.int32),
        "epoch": jnp.asarray(0, jnp.int32),
        "cursor": jnp.asarray(0, jnp.int32),
        "schedule_total_steps": jnp.asarray(train_config.total_steps, jnp.int32),
    }


def identity_for(
    *,
    code_hashes,
    dataset_hash,
    processor_hash,
    config,
    train_config,
    objective,
    parent,
    branch,
):
    return {
        "code_sha256": code_hashes,
        "dataset_sha256": dataset_hash,
        "processor_sha256": processor_hash,
        "model_config_sha256": config.fingerprint(),
        "training_config_sha256": train_config.fingerprint(),
        "objective": objective,
        "parent_checkpoint_sha256": parent,
        "branch": branch,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
    }


def learning_rate(step, train_config):
    if train_config.warmup_steps and step < train_config.warmup_steps:
        return train_config.learning_rate * (step + 1) / train_config.warmup_steps
    decay = max(1, train_config.total_steps - train_config.warmup_steps)
    progress = min(1.0, max(0.0, (step - train_config.warmup_steps) / decay))
    return train_config.learning_rate * (1.0 - progress)


def next_indices(state, n, batch_size):
    """Take a partial tail batch at epoch end without skipping or duplicating rows."""
    cursor, epoch = int(state["cursor"]), int(state["epoch"])
    permutation = state["permutation"]
    key = jax.random.wrap_key_data(jnp.asarray(state["rng"], jnp.uint32))
    if cursor >= n:
        key, shuffle_key = jax.random.split(key)
        permutation = jax.random.permutation(shuffle_key, n).astype(jnp.int32)
        cursor, epoch = 0, epoch + 1
    end = min(n, cursor + batch_size)
    chosen = np.asarray(permutation[cursor:end], dtype=np.int32)
    key, sample_key = jax.random.split(key)
    next_state = {
        **state,
        "rng": jax.random.key_data(key),
        "permutation": permutation,
        "epoch": jnp.asarray(epoch, jnp.int32),
        "cursor": jnp.asarray(end, jnp.int32),
    }
    return next_state, chosen, sample_key


def _adamw(params, m, v, grads, *, step, rate, train_config):
    index = step + 1
    b1, b2 = train_config.beta1, train_config.beta2
    m = jax.tree.map(lambda old, grad: b1 * old + (1.0 - b1) * grad, m, grads)
    v = jax.tree.map(
        lambda old, grad: b2 * old + (1.0 - b2) * jnp.square(grad), v, grads
    )
    c1, c2 = 1.0 - b1**index, 1.0 - b2**index
    params = jax.tree.map(
        lambda param, first, second: param
        - rate
        * (
            (first / c1) / (jnp.sqrt(second / c2) + train_config.epsilon)
            + train_config.weight_decay * param
        ),
        params,
        m,
        v,
    )
    return params, m, v


def _masked_batch(ids, key, config, vocab_size, rate=0.15):
    ids = jnp.asarray(ids, jnp.int32)
    eligible = ids >= 5
    choose_key, action_key, random_key = jax.random.split(key, 3)
    selected = jax.random.bernoulli(choose_key, rate, ids.shape) & eligible
    has_selected = jnp.any(selected, axis=1)
    first_content = jnp.argmax(eligible, axis=1)
    fallback = jax.nn.one_hot(first_content, ids.shape[1], dtype=jnp.bool_) & eligible
    selected = selected | ((~has_selected)[:, None] & fallback)
    action = jax.random.uniform(action_key, ids.shape)
    random_token = jax.random.randint(random_key, ids.shape, 5, vocab_size)
    random_token = jnp.where(random_token < 5, 5, random_token)
    masked = jnp.where(selected & (action < 0.8), config.mask_id, ids)
    masked = jnp.where(
        selected & (action >= 0.8) & (action < 0.9), random_token, masked
    )
    labels = jnp.where(selected, ids, 0)
    return masked, labels, selected


def train_mlm(model, state, examples, config, train_config, *, steps, tag="phase1_mlm"):
    if len(examples) < train_config.batch_size:
        raise ValueError("MLM dataset is smaller than one requested batch")

    @jax.jit
    def loss_and_grad(params, ids, labels, mask, dropout_key):
        loss, _, _ = model.masked_token_loss(
            params, ids, labels, mask, config, dropout_key
        )
        return loss

    losses, trace = [], []
    for _ in range(steps):
        state, indices, sample_key = next_indices(
            state, len(examples), train_config.batch_size
        )
        ids = jnp.asarray(examples[indices])
        mask_key, dropout_key = jax.random.split(sample_key)
        masked, labels, selected = _masked_batch(
            ids, mask_key, config, config.vocab_size
        )
        loss, grads = jax.value_and_grad(loss_and_grad)(
            state["params"], masked, labels, selected, dropout_key
        )
        rate = learning_rate(int(state["step"]), train_config)
        params, m, v = _adamw(
            state["params"],
            state["m"],
            state["v"],
            grads,
            step=int(state["step"]),
            rate=rate,
            train_config=train_config,
        )
        state.update(
            {
                "params": params,
                "m": m,
                "v": v,
                "step": state["step"] + 1,
                "total_step": state["total_step"] + 1,
                "global_step": state["global_step"] + 1,
            }
        )
        losses.append(float(loss))
        trace.append(
            {
                "stage": tag,
                "step": int(state["step"]),
                "epoch": int(state["epoch"]),
                "cursor": int(state["cursor"]),
                "indices": indices.tolist(),
                "mask_key": np.asarray(jax.random.key_data(mask_key)).tolist(),
                "dropout_key": np.asarray(jax.random.key_data(dropout_key)).tolist(),
                "learning_rate": rate,
                "loss_nats_per_masked_token": float(loss),
                "input_tokens": int(np.count_nonzero(np.asarray(ids) != config.pad_id)),
                "selected_tokens": int(selected.sum()),
            }
        )
    return state, {"losses": losses, "trace": trace}


def train_weak_contrastive(
    model,
    state,
    left_ids,
    right_ids,
    group_ids,
    config,
    train_config,
    *,
    steps,
    retained_mlm_weight=0.0,
    mlm_examples=None,
    tag="phase2",
):
    if not (len(left_ids) == len(right_ids) == len(group_ids)):
        raise ValueError("weak pair arrays must have matching rows")
    if len(left_ids) < train_config.batch_size:
        raise ValueError("weak pair dataset is smaller than one requested batch")

    @jax.jit
    def contrastive(params, left, right, positive, dropout_key):
        left_key, right_key = jax.random.split(dropout_key)
        left_vectors = model.pool(
            params, left, config, left != config.pad_id, dropout_key=left_key
        )
        right_vectors = model.pool(
            params, right, config, right != config.pad_id, dropout_key=right_key
        )
        return objectives.multi_positive_contrastive(
            left_vectors, right_vectors, positive
        )

    losses, trace = [], []
    for _ in range(steps):
        state, indices, sample_key = next_indices(
            state, len(left_ids), train_config.batch_size
        )
        dropout_key, mlm_key = jax.random.split(sample_key)
        left, right = jnp.asarray(left_ids[indices]), jnp.asarray(right_ids[indices])
        groups = np.asarray(group_ids)[indices]
        positive = jnp.asarray(groups[:, None] == groups[None, :])
        mlm_idx = indices % len(mlm_examples) if mlm_examples is not None else []

        def objective(params):
            value = contrastive(params, left, right, positive, dropout_key)
            component = None
            selected_count = jnp.asarray(0, jnp.int32)
            if retained_mlm_weight and mlm_examples is not None:
                raw = jnp.asarray(mlm_examples[mlm_idx])
                corrupted, labels, selected = _masked_batch(
                    raw, mlm_key, config, config.vocab_size
                )
                selected_count = jnp.sum(selected)
                mlm, _, _ = model.masked_token_loss(
                    params, corrupted, labels, selected, config, dropout_key
                )
                component = mlm
                value = value + retained_mlm_weight * mlm
            return value, (component, selected_count)

        (loss, (mlm_loss, mlm_selected_tokens)), grads = jax.value_and_grad(
            objective, has_aux=True
        )(state["params"])
        rate = learning_rate(int(state["step"]), train_config)
        params, m, v = _adamw(
            state["params"],
            state["m"],
            state["v"],
            grads,
            step=int(state["step"]),
            rate=rate,
            train_config=train_config,
        )
        state.update(
            {
                "params": params,
                "m": m,
                "v": v,
                "step": state["step"] + 1,
                "total_step": state["total_step"] + 1,
                "global_step": state["global_step"] + 1,
            }
        )
        losses.append(float(loss))
        trace.append(
            {
                "stage": tag,
                "step": int(state["step"]),
                "epoch": int(state["epoch"]),
                "cursor": int(state["cursor"]),
                "indices": indices.tolist(),
                "dropout_key": np.asarray(jax.random.key_data(dropout_key)).tolist(),
                "learning_rate": rate,
                "contrastive_plus_weighted_mlm_loss": float(loss),
                "pair_input_tokens": int(
                    np.count_nonzero(np.asarray(left) != config.pad_id)
                    + np.count_nonzero(np.asarray(right) != config.pad_id)
                ),
                "mlm_indices": (
                    list(map(int, mlm_idx))
                    if retained_mlm_weight and mlm_examples is not None
                    else []
                ),
                "mlm_key": (
                    np.asarray(jax.random.key_data(mlm_key)).tolist()
                    if retained_mlm_weight and mlm_examples is not None
                    else None
                ),
                "mlm_input_tokens": (
                    int(np.count_nonzero(mlm_examples[mlm_idx] != config.pad_id))
                    if retained_mlm_weight and mlm_examples is not None
                    else 0
                ),
                "mlm_selected_tokens": int(mlm_selected_tokens),
                "weighted_mlm_component": (
                    None if mlm_loss is None else float(mlm_loss) * retained_mlm_weight
                ),
            }
        )
    return state, {"losses": losses, "trace": trace}


def train_human_contrastive(
    model,
    state,
    query_rows,
    document_rows,
    config,
    train_config,
    *,
    steps,
    tag="phase3_human_pairs",
    max_positive_per_query=4,
):
    if len(query_rows) < train_config.batch_size:
        raise ValueError("human pair dataset is smaller than one requested batch")
    documents = {int(row["id"]): row for row in document_rows}

    @jax.jit
    def loss_fn(params, qids, dids, positive, candidate_mask, dropout_key):
        qkey, dkey = jax.random.split(dropout_key)
        q = model.pool(params, qids, config, qids != config.pad_id, dropout_key=qkey)
        d = model.pool(params, dids, config, dids != config.pad_id, dropout_key=dkey)
        return objectives.multi_positive_contrastive(
            q, d, positive, candidate_mask=candidate_mask
        )

    losses, trace = [], []
    for _ in range(steps):
        state, indices, sample_key = next_indices(
            state, len(query_rows), train_config.batch_size
        )
        selected_rows = [query_rows[int(index)] for index in indices]
        qids = np.stack([row["ids"] for row in selected_rows])
        candidate_ids = sorted(
            set().union(*(set(row["positive_doc_ids"]) for row in selected_rows))
        )
        if not candidate_ids:
            raise ValueError("human training batch has no evidence positives")
        if any(
            len(row["positive_doc_ids"]) > max_positive_per_query
            for row in selected_rows
        ):
            raise ValueError(
                "human training query has more positives than the frozen candidate bound"
            )
        max_candidates = train_config.batch_size * max_positive_per_query
        if len(candidate_ids) > max_candidates:
            raise ValueError(
                "human minibatch positive union exceeds its frozen candidate bound"
            )
        dids = np.zeros((max_candidates, config.max_length), np.int32)
        for doc_index, doc_id in enumerate(candidate_ids):
            values = documents[doc_id]["ids"]
            dids[doc_index, : len(values)] = values
        candidate_mask = np.arange(max_candidates) < len(candidate_ids)
        positive = np.zeros((len(selected_rows), max_candidates), dtype=bool)
        for row_index, row in enumerate(selected_rows):
            positive[row_index, : len(candidate_ids)] = [
                doc_id in row["positive_doc_ids"] for doc_id in candidate_ids
            ]
        if not np.all(positive.any(axis=1)):
            raise AssertionError(
                "multi-positive training batch has a query without a positive"
            )
        (dropout_key,) = jax.random.split(sample_key, 1)
        loss, grads = jax.value_and_grad(loss_fn)(
            state["params"],
            jnp.asarray(qids),
            jnp.asarray(dids),
            jnp.asarray(positive),
            jnp.asarray(candidate_mask),
            dropout_key,
        )
        rate = learning_rate(int(state["step"]), train_config)
        params, m, v = _adamw(
            state["params"],
            state["m"],
            state["v"],
            grads,
            step=int(state["step"]),
            rate=rate,
            train_config=train_config,
        )
        state.update(
            {
                "params": params,
                "m": m,
                "v": v,
                "step": state["step"] + 1,
                "total_step": state["total_step"] + 1,
                "global_step": state["global_step"] + 1,
            }
        )
        losses.append(float(loss))
        trace.append(
            {
                "stage": tag,
                "step": int(state["step"]),
                "epoch": int(state["epoch"]),
                "cursor": int(state["cursor"]),
                "query_indices": indices.tolist(),
                "query_ids": [row["query_id"] for row in selected_rows],
                "candidate_doc_ids": candidate_ids,
                "positive_pairs": int(positive.sum()),
                "query_input_tokens": int(np.count_nonzero(qids != config.pad_id)),
                "candidate_document_input_tokens": int(
                    np.count_nonzero(dids[: len(candidate_ids)] != config.pad_id)
                ),
                "dropout_key": np.asarray(jax.random.key_data(dropout_key)).tolist(),
                "learning_rate": rate,
                "multi_positive_loss": float(loss),
            }
        )
    return state, {"losses": losses, "trace": trace}


def _flatten_state(state):
    leaves, tree = jax.tree.flatten(state)
    return leaves, tree


def save_checkpoint(folder, state, identity):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    leaves, _ = _flatten_state(state)
    arrays = {
        f"leaf_{index:04d}": np.asarray(leaf) for index, leaf in enumerate(leaves)
    }
    np.savez(folder / "state.npz", **arrays)
    raw = (folder / "state.npz").read_bytes()
    manifest = {
        "schema": 1,
        "identity": identity,
        "state_sha256": _digest(raw),
        "leaves": [
            {"shape": list(value.shape), "dtype": str(value.dtype)}
            for value in arrays.values()
        ],
        "step": int(state["step"]),
        "total_step": int(state["total_step"]),
        "global_step": int(state["global_step"]),
        "epoch": int(state["epoch"]),
        "cursor": int(state["cursor"]),
        "parent": identity.get("parent_checkpoint_sha256"),
        "state_keys": [
            "params",
            "m",
            "v",
            "step",
            "total_step",
            "global_step",
            "rng",
            "permutation",
            "epoch",
            "cursor",
            "schedule_total_steps",
        ],
    }
    (folder / "checkpoint.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return _digest((folder / "checkpoint.json").read_bytes())


def load_checkpoint(folder, template_state, expected_identity):
    folder = Path(folder)
    manifest = json.loads((folder / "checkpoint.json").read_text())
    raw = (folder / "state.npz").read_bytes()
    if manifest.get("schema") != 1 or manifest.get("identity") != expected_identity:
        raise ValueError(
            "checkpoint source/config/tokenizer/data/objective identity mismatch"
        )
    if _digest(raw) != manifest.get("state_sha256"):
        raise ValueError("checkpoint state checksum mismatch")
    template, tree = _flatten_state(template_state)
    with np.load(folder / "state.npz", allow_pickle=False) as archive:
        expected_names = [f"leaf_{index:04d}" for index in range(len(template))]
        if sorted(archive.files) != expected_names:
            raise ValueError("checkpoint tensor inventory differs from stage template")
        restored = [archive[f"leaf_{index:04d}"] for index in range(len(template))]
    for expected, value in zip(template, restored):
        if tuple(expected.shape) != tuple(value.shape):
            raise ValueError("checkpoint array shape does not match stage template")
        if np.dtype(expected.dtype) != value.dtype:
            raise ValueError("checkpoint array dtype does not match stage template")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError("checkpoint contains nonfinite floating-point state")
    restored_state = jax.tree.unflatten(
        tree, [jnp.asarray(value) for value in restored]
    )
    counters = {}
    for name in (
        "step",
        "total_step",
        "global_step",
        "epoch",
        "cursor",
        "schedule_total_steps",
    ):
        value = np.asarray(restored_state[name])
        if value.shape != () or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"checkpoint {name} must be an integer scalar")
        counters[name] = int(value)
    if (
        counters["step"] < 0
        or counters["total_step"] < counters["step"]
        or counters["global_step"] < counters["total_step"]
        or counters["epoch"] < 0
        or counters["schedule_total_steps"] < 1
        or counters["step"] > counters["schedule_total_steps"]
    ):
        raise ValueError("checkpoint counters violate stage progress invariants")
    if counters["schedule_total_steps"] != int(
        np.asarray(template_state["schedule_total_steps"])
    ):
        raise ValueError("checkpoint schedule length differs from configured stage")
    rng = np.asarray(restored_state["rng"])
    if rng.shape != (2,) or rng.dtype != np.dtype(np.uint32):
        raise ValueError("checkpoint PRNG state must be two uint32 key words")
    permutation = np.asarray(restored_state["permutation"])
    example_count = len(np.asarray(template_state["permutation"]))
    if (
        permutation.shape != (example_count,)
        or permutation.dtype != np.dtype(np.int32)
        or not np.array_equal(np.sort(permutation), np.arange(example_count))
    ):
        raise ValueError("checkpoint sampler permutation is not a valid bijection")
    if not 0 <= counters["cursor"] <= example_count:
        raise ValueError("checkpoint sampler cursor is outside the stage dataset")
    for key in ("step", "total_step", "global_step", "epoch", "cursor"):
        if manifest.get(key) != counters[key]:
            raise ValueError(f"checkpoint manifest {key} disagrees with saved state")
    expected_state_keys = [
        "params",
        "m",
        "v",
        "step",
        "total_step",
        "global_step",
        "rng",
        "permutation",
        "epoch",
        "cursor",
        "schedule_total_steps",
    ]
    if manifest.get("state_keys") != expected_state_keys:
        raise ValueError("checkpoint manifest omits required training state")
    return (
        restored_state,
        manifest,
    )
