"""Bounded C02 training, branch-local AdamW and portable full-state recovery."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

try:
    from . import model
    from .config import ModelConfig, TrainConfig
except ImportError:  # Direct script/checker loading.
    import model
    from config import ModelConfig, TrainConfig


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule_rate(
    base_rate: float,
    step: int,
    total: int,
    warmup: int,
    minimum_rate: float | None = None,
) -> float:
    """Warm up, then cosine-decay from the stage peak to its configured floor."""
    if total < 1 or not 0 <= step < total:
        raise ValueError("schedule step must be inside the declared stage")
    floor = base_rate * 0.1 if minimum_rate is None else minimum_rate
    if not math.isfinite(floor) or not 0 <= floor <= base_rate:
        raise ValueError("minimum learning rate must be finite and within the peak")
    warm = max(1, warmup)
    if step < warm:
        return base_rate * (step + 1) / warm
    decay_steps = max(1, total - warm)
    progress = min(1.0, (step - warm + 1) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (base_rate - floor) * cosine


def _tree_zeros(tree):
    return jax.tree.map(jnp.zeros_like, tree)


def _tree_norm(tree):
    leaves = jax.tree.leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in leaves))


def adamw_update(tree, gradients, first, second, step, learning_rate, train_config):
    """One global-norm-clipped FP32 AdamW update, returning complete moments."""
    cfg = (
        train_config
        if isinstance(train_config, TrainConfig)
        else TrainConfig(**train_config)
    )
    cfg.validate()
    norm = _tree_norm(gradients)
    clip = jnp.minimum(1.0, cfg.gradient_clip_norm / jnp.maximum(norm, 1e-12))
    gradients = jax.tree.map(lambda grad: grad * clip, gradients)
    beta1, beta2 = cfg.beta1, cfg.beta2
    first = jax.tree.map(
        lambda old, grad: beta1 * old + (1.0 - beta1) * grad, first, gradients
    )
    second = jax.tree.map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad),
        second,
        gradients,
    )
    correction1 = 1.0 - beta1 ** (step + 1)
    correction2 = 1.0 - beta2 ** (step + 1)
    updated = jax.tree.map(
        lambda value, m, v: value
        - learning_rate
        * (
            m / correction1 / (jnp.sqrt(v / correction2) + cfg.epsilon)
            + cfg.weight_decay * value
        ),
        tree,
        first,
        second,
    )
    return updated, first, second, float(norm)


def _encode_tree(tree, arrays, name):
    if isinstance(tree, dict):
        return {
            key: _encode_tree(value, arrays, f"{name}/{key}")
            for key, value in sorted(tree.items())
        }
    if isinstance(tree, (tuple, list)):
        return {
            "__sequence__": type(tree).__name__,
            "items": [
                _encode_tree(value, arrays, f"{name}/{index}")
                for index, value in enumerate(tree)
            ],
        }
    if isinstance(tree, (int, float, bool, str)) or tree is None:
        return {"__scalar__": tree}
    array = np.asarray(tree)
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot be written to a C02 checkpoint")
    arrays[name] = array
    return {"__array__": name, "shape": list(array.shape), "dtype": str(array.dtype)}


def _decode_tree(spec, archive):
    if "__array__" in spec:
        value = np.asarray(archive[spec["__array__"]])
        if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
            raise ValueError("checkpoint array does not match its declared shape/dtype")
        return jnp.asarray(value)
    if "__scalar__" in spec:
        return spec["__scalar__"]
    if "__sequence__" in spec:
        values = [_decode_tree(item, archive) for item in spec["items"]]
        return tuple(values) if spec["__sequence__"] == "tuple" else values
    return {key: _decode_tree(value, archive) for key, value in spec.items()}


def _array_names(spec):
    if "__array__" in spec:
        return [spec["__array__"]]
    if "__sequence__" in spec:
        return [name for child in spec["items"] for name in _array_names(child)]
    if "__scalar__" in spec:
        return []
    return [name for child in spec.values() for name in _array_names(child)]


def _validate_tree_against(tree, expected, name, nonnegative=False):
    if jax.tree.structure(tree) != jax.tree.structure(expected):
        raise ValueError(f"checkpoint {name} tree structure differs from model schema")
    for index, (value, reference) in enumerate(
        zip(jax.tree.leaves(tree), jax.tree.leaves(expected), strict=True)
    ):
        array, expected_array = np.asarray(value), np.asarray(reference)
        if array.shape != expected_array.shape or array.dtype != expected_array.dtype:
            raise ValueError(f"checkpoint {name} leaf {index} has wrong shape or dtype")
        if not np.isfinite(array).all():
            raise ValueError(f"checkpoint {name} leaf {index} is nonfinite")
        if nonnegative and np.any(array < 0):
            raise ValueError(f"checkpoint {name} leaf {index} must be nonnegative")


def save_checkpoint(path, state, identity):
    """Atomically save params/adapters, Adam moments, PRNG and data cursor."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    tree_specs = {
        key: _encode_tree(state[key], arrays, key)
        for key in ("params", "adapters", "first", "second")
    }
    rng = np.asarray(state["rng"], dtype=np.uint32)
    if rng.shape != (2,):
        raise ValueError("checkpoint PRNG must be an explicit two-word JAX key")
    arrays["rng"] = rng
    meta = {
        "format": "c02-full-state-v1",
        "identity": identity,
        "tree_specs": tree_specs,
        "step": int(state["step"]),
        "schedule_step": int(state["schedule_step"]),
        "epoch": int(state["epoch"]),
        "cursor": int(state["cursor"]),
        "permutation": [int(x) for x in state["permutation"]],
        "seen_ids": list(state.get("seen_ids", [])),
        "meta": dict(state.get("meta", {})),
    }
    arrays["__meta__"] = np.frombuffer(
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode(),
        dtype=np.uint8,
    )
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _hash_file(destination)


def load_checkpoint(
    path, expected_identity, expected_order=None, expected_seen_ids=None
):
    """Load only complete finite state whose experiment and data order match."""
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(np.asarray(archive["__meta__"], dtype=np.uint8).tobytes())
        if meta.get("format") != "c02-full-state-v1":
            raise ValueError("unsupported checkpoint format")
        if meta.get("identity") != expected_identity:
            raise ValueError(
                "checkpoint experiment identity mismatch: source, recipe, objective, parent or schedule"
            )
        permutation = meta.get("permutation")
        if not isinstance(permutation, list) or any(
            type(x) is not int for x in permutation
        ):
            raise ValueError("checkpoint permutation is malformed")
        if sorted(permutation) != list(range(len(permutation))):
            raise ValueError("checkpoint permutation is not a full permutation")
        if expected_order is not None and permutation != list(expected_order):
            raise ValueError("checkpoint data order changed")
        cursor, step = meta.get("cursor"), meta.get("step")
        if type(cursor) is not int or not 0 <= cursor <= len(permutation):
            raise ValueError("checkpoint cursor is outside the frozen order")
        if type(step) is not int or step < 0 or meta.get("schedule_step") != step:
            raise ValueError("checkpoint optimizer/schedule clocks are invalid")
        if not isinstance(meta.get("tree_specs"), dict) or set(meta["tree_specs"]) != {
            "params",
            "adapters",
            "first",
            "second",
        }:
            raise ValueError("checkpoint does not declare the complete state tree")
        config_payload = expected_identity.get("model_config")
        train_payload = expected_identity.get("train_config")
        if not isinstance(config_payload, dict) or not isinstance(train_payload, dict):
            raise ValueError(
                "expected experiment identity must declare model and train configs"
            )
        config = ModelConfig(**config_payload)
        train_config = TrainConfig(**train_payload)
        config.validate()
        train_config.validate()
        if step > train_config.total_steps:
            raise ValueError(
                "checkpoint optimizer clock exceeds frozen schedule horizon"
            )
        epoch = meta.get("epoch")
        if type(epoch) is not int or epoch != 0:
            raise ValueError("checkpoint epoch is outside the fixed stage schedule")
        seen_ids = meta.get("seen_ids")
        if not isinstance(seen_ids, list) or any(
            not isinstance(item, str) for item in seen_ids
        ):
            raise ValueError("checkpoint consumed-ID history is malformed")
        if len(seen_ids) != cursor or len(set(seen_ids)) != len(seen_ids):
            raise ValueError(
                "checkpoint cursor and unique consumed-ID history disagree"
            )
        frozen_ids = expected_identity.get("schedule_sample_ids")
        if not isinstance(frozen_ids, list) or any(
            not isinstance(item, str) for item in frozen_ids
        ):
            raise ValueError(
                "expected identity must include the full frozen sample order"
            )
        if len(permutation) != len(frozen_ids) or cursor > len(frozen_ids):
            raise ValueError(
                "checkpoint cursor/permutation differs from frozen sample order"
            )
        offsets = expected_identity.get("schedule_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != train_config.total_steps + 1
            or any(type(offset) is not int for offset in offsets)
            or not offsets
            or offsets[0] != 0
            or offsets[-1] != len(frozen_ids)
            or any(right <= left for left, right in zip(offsets, offsets[1:]))
        ):
            raise ValueError("expected identity has invalid per-update sample offsets")
        if step >= len(offsets) or cursor != offsets[step]:
            raise ValueError(
                "checkpoint cursor does not match the declared optimizer update boundary"
            )
        if seen_ids != frozen_ids[:cursor]:
            raise ValueError(
                "checkpoint consumed IDs do not match frozen schedule prefix"
            )
        if expected_seen_ids is not None and frozen_ids != list(expected_seen_ids):
            raise ValueError(
                "expected sample order differs from frozen schedule identity"
            )
        if not isinstance(meta.get("meta"), dict):
            raise ValueError("checkpoint stage metadata must be a mapping")
        expected_array_names = [
            name for spec in meta["tree_specs"].values() for name in _array_names(spec)
        ] + ["rng"]
        if len(set(expected_array_names)) != len(expected_array_names):
            raise ValueError("checkpoint declares duplicate array leaves")
        if set(archive.files) != set(expected_array_names) | {"__meta__"}:
            raise ValueError("checkpoint contains missing or unrecognized state arrays")
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
            if name != "__meta__"
        }
    rng = arrays.pop("rng", None)
    if rng is None or rng.shape != (2,) or rng.dtype != np.uint32:
        raise ValueError("checkpoint PRNG state is missing or malformed")
    for name, value in arrays.items():
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(
                f"checkpoint contains nonfinite or nonnumeric state: {name}"
            )

    class ArchiveView(dict):
        def __getitem__(self, key):
            return arrays[key]

    view = ArchiveView()
    state = {key: _decode_tree(spec, view) for key, spec in meta["tree_specs"].items()}
    state.update(
        {
            "rng": jnp.asarray(rng),
            "step": step,
            "schedule_step": int(meta["schedule_step"]),
            "epoch": epoch,
            "cursor": cursor,
            "permutation": permutation,
            "seen_ids": seen_ids,
            "meta": meta["meta"],
        }
    )
    expected_params = model.initialize(jax.random.PRNGKey(0), config)
    _validate_tree_against(state["params"], expected_params, "parameters")
    adapter_spec = expected_identity.get("adapter_spec")
    if adapter_spec is None:
        if state["adapters"] is not None:
            raise ValueError("checkpoint unexpectedly contains adapter leaves")
        trainable_schema = expected_params
    else:
        if not isinstance(adapter_spec, dict):
            raise ValueError("checkpoint adapter identity is malformed")
        rank = int(adapter_spec.get("rank", -1))
        if rank != model.LORA_RANK:
            raise ValueError("checkpoint adapter rank differs from frozen C02 schema")
        expected_adapters = model.initialize_lora(
            jax.random.PRNGKey(1), config, rank=rank
        )
        _validate_tree_against(state["adapters"], expected_adapters, "adapters")
        trainable_schema = expected_adapters
    _validate_tree_against(state["first"], trainable_schema, "first moment")
    _validate_tree_against(
        state["second"], trainable_schema, "second moment", nonnegative=True
    )
    if np.asarray(state["rng"]).dtype != np.uint32:
        raise ValueError("checkpoint PRNG must use uint32 key words")
    return state


def make_state(params, adapters, rng, permutation):
    trainable = adapters if adapters is not None else params
    return {
        "params": params,
        "adapters": adapters,
        "first": _tree_zeros(trainable),
        "second": _tree_zeros(trainable),
        "rng": jnp.asarray(rng, dtype=jnp.uint32),
        "step": 0,
        "schedule_step": 0,
        "epoch": 0,
        "cursor": 0,
        "permutation": [int(x) for x in permutation],
        "seen_ids": [],
        "meta": {},
    }


def next_batch(state, rows, batch_size):
    """Consume a no-drop permutation, carrying a partial batch across epochs."""
    if not rows or batch_size < 1:
        raise ValueError("training rows and batch size must be nonempty")
    picked = []
    while len(picked) < batch_size:
        if state["cursor"] >= len(state["permutation"]):
            state["epoch"] += 1
            key, state["rng"] = jax.random.split(state["rng"])
            order = np.asarray(jax.random.permutation(key, len(rows))).tolist()
            state["permutation"] = [int(x) for x in order]
            state["cursor"] = 0
        take = min(
            batch_size - len(picked), len(state["permutation"]) - state["cursor"]
        )
        indices = state["permutation"][state["cursor"] : state["cursor"] + take]
        picked.extend(rows[index] for index in indices)
        state["seen_ids"].extend(str(rows[index].get("id", index)) for index in indices)
        state["cursor"] += take
    return picked


def identity_payload(
    stage, parent_hash, model_config, train_config, prepared_dir, source_hash
):
    prepared = Path(prepared_dir)
    recipe = prepared / "frozen_recipe.json"
    tokenizer = prepared / "tokenizer.json"
    return {
        "stage": stage,
        "parent_checkpoint_sha256": parent_hash,
        "model_config": model_config.as_dict(),
        "train_config": train_config.as_dict(),
        "source_manifest_sha256": _hash_file(prepared / "source-manifest.json"),
        "recipe_sha256": _hash_file(recipe),
        "processor_sha256": _hash_file(tokenizer),
        "candidate_source_sha256": source_hash,
        "objective": (
            "aligned-token-cross-entropy"
            if not stage.startswith("sft")
            else "response-only-aligned-token-cross-entropy"
        ),
    }


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_blocks(prepared_dir, stem):
    prepared = Path(prepared_dir)
    index = read_jsonl(prepared / f"{stem}-block-index.jsonl")
    arrays = np.load(prepared / f"{stem}-blocks.npz")
    if len(index) != arrays["input_ids"].shape[0]:
        raise ValueError(f"{stem} index and arrays have different row counts")
    mapping = {row["block_id"]: offset for offset, row in enumerate(index)}
    return arrays, mapping


def _stack_block_arrays(arrays, indices):
    return {
        key: jnp.asarray(arrays[key][indices])
        for key in (
            "input_ids",
            "target_ids",
            "target_mask",
            "segment_ids",
            "position_ids",
        )
    }


def sft_batch(rows, tokenizer, config, prompt_builder):
    """Build response-only aligned arrays from frozen prompt/answer fields."""
    cfg = config if isinstance(config, ModelConfig) else ModelConfig(**config)
    inputs, targets, masks, segments, positions = [], [], [], [], []
    for row in rows:
        prompt = prompt_builder(row["question"], row["context"])
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        answer_ids = tokenizer.encode(
            row["target_answer"], add_special_tokens=False
        ).ids
        sequence = [1, *prompt_ids, *answer_ids, 2]
        if len(sequence) > cfg.context or not answer_ids:
            raise ValueError(
                f"SFT row {row['id']} violates the frozen input/response fit"
            )
        # Inputs predict the following token; the mask is on the target side.
        input_row = sequence[:-1]
        target_row = sequence[1:]
        response_start = 1 + len(prompt_ids) - 1
        target_mask = [False] * len(target_row)
        for index in range(response_start, len(target_row)):
            target_mask[index] = True
        padding = cfg.context - len(input_row)
        inputs.append(input_row + [0] * padding)
        targets.append(target_row + [0] * padding)
        masks.append(target_mask + [False] * padding)
        segments.append([1] * len(input_row) + [0] * padding)
        positions.append(list(range(len(input_row))) + [0] * padding)
        if sum(target_mask) != int(row["response_tokens_with_eos"]):
            raise AssertionError(
                f"response mask does not own all answer/EOS targets for {row['id']}"
            )
    return {
        "input_ids": jnp.asarray(inputs, dtype=jnp.int32),
        "target_ids": jnp.asarray(targets, dtype=jnp.int32),
        "target_mask": jnp.asarray(masks, dtype=jnp.bool_),
        "segment_ids": jnp.asarray(segments, dtype=jnp.int32),
        "position_ids": jnp.asarray(positions, dtype=jnp.int32),
    }


def _numerator(trainable, params, batch, config, dropout_key, is_lora):
    if is_lora:
        numerator, count = model.aligned_token_loss(
            params,
            batch["input_ids"],
            batch["target_ids"],
            batch["target_mask"],
            batch["segment_ids"],
            batch["position_ids"],
            config,
            key=dropout_key,
            adapters=trainable,
        )
    else:
        numerator, count = model.aligned_token_loss(
            trainable,
            batch["input_ids"],
            batch["target_ids"],
            batch["target_mask"],
            batch["segment_ids"],
            batch["position_ids"],
            config,
            key=dropout_key,
        )
    return numerator, count


def update_from_batches(state, batches, model_config, train_config, learning_rate):
    """Accumulate numerator gradients, then normalize once by valid targets."""
    if not batches:
        raise ValueError("an optimizer update needs at least one microbatch")
    is_lora = state["adapters"] is not None
    trainable = state["adapters"] if is_lora else state["params"]
    grad_sum = _tree_zeros(trainable)
    numerator_sum, count_sum = 0.0, 0.0
    next_rng = state["rng"]
    for batch in batches:
        dropout_key, next_rng = jax.random.split(next_rng)
        value_and_grad = jax.value_and_grad(
            lambda current: _numerator(
                current, state["params"], batch, model_config, dropout_key, is_lora
            )[0]
        )
        numerator, grad = value_and_grad(trainable)
        _, count = _numerator(
            trainable, state["params"], batch, model_config, dropout_key, is_lora
        )
        grad_sum = jax.tree.map(lambda left, right: left + right, grad_sum, grad)
        numerator_sum += float(numerator)
        count_sum += float(count)
    if count_sum <= 0 or not math.isfinite(numerator_sum):
        raise ValueError("optimizer update has no finite active targets")
    gradients = jax.tree.map(lambda value: value / count_sum, grad_sum)
    updated, first, second, gradient_norm = adamw_update(
        trainable,
        gradients,
        state["first"],
        state["second"],
        state["step"],
        learning_rate,
        train_config,
    )
    next_state = {
        **state,
        "params": state["params"] if is_lora else updated,
        "adapters": updated if is_lora else None,
        "first": first,
        "second": second,
        "rng": next_rng,
        "step": state["step"] + 1,
        "schedule_step": state["schedule_step"] + 1,
    }
    return next_state, {
        "loss_numerator": numerator_sum,
        "valid_target_count": int(count_sum),
        "loss": numerator_sum / count_sum,
        "learning_rate": learning_rate,
        "gradient_norm_before_clip": gradient_norm,
    }


def phase_step(
    state, data, model_config, train_config, base_rate, total_steps, warmup_steps
):
    cfg = (
        train_config
        if isinstance(train_config, TrainConfig)
        else TrainConfig(**train_config)
    )
    if not math.isclose(base_rate, cfg.learning_rate, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("phase peak learning rate differs from its train config")
    if total_steps != cfg.total_steps or warmup_steps != cfg.warmup_steps:
        raise ValueError("phase schedule differs from its train config")
    step = int(state["schedule_step"])
    rate = schedule_rate(
        base_rate, step, total_steps, warmup_steps, cfg.min_learning_rate
    )
    return update_from_batches(state, data, model_config, cfg, rate)
