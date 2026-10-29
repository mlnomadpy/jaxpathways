"""Small, deterministic training and crash-safe local checkpoint protocol.

The generation protocol demonstrates ordinary-process atomic publication on a
single POSIX local filesystem using `fcntl.flock`. It is not a claim about power
loss, Windows locking, network filesystems, or distributed workers.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2
STATE_ARRAYS = {
    "weight",
    "velocity",
    "rng_key",
    "cursor",
    "epoch",
    "epoch_order",
    "source_ids",
    "consumed_ids",
    "update_ends",
    "optimizer_step",
}
FAILPOINTS = {
    "during_temp_write",
    "after_temp_create",
    "after_payload",
    "after_manifest",
    "before_publish",
    "after_publish",
}


def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _array_digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        arr = np.asarray(arrays[name])
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        h.update(name.encode("utf-8") + b"\0")
        h.update(arr.dtype.str.encode("ascii") + b"\0")
        h.update(_json_bytes(list(arr.shape)) + b"\0")
        h.update(arr.tobytes())
    return h.hexdigest()


def _digest(step, config, arrays):
    """Compatibility digest for the original single-file public API."""
    h = hashlib.sha256(_json_bytes({"step": int(step), "config": config}))
    h.update(_array_digest(arrays).encode("ascii"))
    return h.hexdigest()


def _validate_config(config):
    required = {"lr", "momentum", "noise_scale"}
    if not isinstance(config, dict) or not required <= set(config):
        raise ValueError("invalid training config")
    if not all(np.isfinite(float(config[k])) for k in required):
        raise ValueError("nonfinite training config")
    if float(config["lr"]) <= 0 or not 0 <= float(config["momentum"]) < 1:
        raise ValueError("invalid learning rate or momentum")
    if float(config["noise_scale"]) < 0:
        raise ValueError("invalid noise scale")
    if "parameter_shape" in config:
        shape = config["parameter_shape"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
        ):
            raise ValueError(
                "parameter_shape must be a nonempty list of positive dimensions"
            )


def _validate_legacy_state(state, config):
    weight = np.asarray(state["weight"])
    velocity = np.asarray(state["velocity"])
    if weight.ndim == 0 or velocity.shape != weight.shape:
        raise ValueError("parameter and momentum shape mismatch")
    if "parameter_shape" in config and weight.shape != tuple(config["parameter_shape"]):
        raise ValueError("parameter shape differs from caller config")
    if np.asarray(state["rng_key"]).shape != (2,):
        raise ValueError("PRNG key shape mismatch")


def _key_words(key):
    a = np.asarray(key, dtype=np.uint32)
    if a.shape != (2,):
        raise ValueError("PRNG key must contain two uint32 words")
    x = int(a[0]) | (int(a[1]) << 32)
    if x == 0:
        raise ValueError("all-zero PRNG key is not supported")
    return x


def _random_vector(key, size):
    """Box-Muller noise using a small specified xorshift64* stream."""
    x = _key_words(key)
    mask = (1 << 64) - 1
    values = []
    while len(values) < size:
        pair = []
        for _ in range(2):
            x ^= x >> 12
            x ^= (x << 25) & mask
            x ^= x >> 27
            x &= mask
            z = (x * 2685821657736338717) & mask
            pair.append((z + 0.5) / float(1 << 64))
        radius = np.sqrt(-2.0 * np.log(pair[0]))
        angle = 2.0 * np.pi * pair[1]
        values.extend((radius * np.cos(angle), radius * np.sin(angle)))
    next_key = np.array([x & 0xFFFFFFFF, (x >> 32) & 0xFFFFFFFF], dtype=np.uint32)
    return np.asarray(values[:size], dtype=np.float32), next_key


def _order_for(source_ids, seed, epoch):
    ids = [int(x) for x in np.asarray(source_ids, dtype=np.int64)]
    return np.asarray(
        sorted(
            ids,
            key=lambda sample_id: hashlib.sha256(
                f"{seed}:{epoch}:{sample_id}".encode()
            ).hexdigest(),
        ),
        dtype=np.int64,
    )


def train_step(state, batch, config):
    """Original one-update API, retained for stages 1–4."""
    _validate_config(config)
    _validate_legacy_state(state, config)
    w = np.asarray(state["weight"], dtype=np.float32)
    velocity = np.asarray(state["velocity"], dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)
    if batch.shape != w.shape:
        raise ValueError("batch shape differs from parameter shape")
    noise, next_key = _random_vector(state["rng_key"], w.size)
    noise = noise.reshape(w.shape) * np.float32(config["noise_scale"])
    grad = (w - batch) + noise
    next_velocity = np.float32(config["momentum"]) * velocity + grad
    return {
        **state,
        "weight": w - np.float32(config["lr"]) * next_velocity,
        "velocity": next_velocity,
        "rng_key": next_key,
        "cursor": np.asarray(state["cursor"] + 1, dtype=np.int64),
    }


def stream_step(state, targets_by_id, config):
    """Consume one deterministic batch, including a short epoch-tail batch.

    A tail is an update of its own. The following call opens the next epoch,
    computes its order from the frozen source IDs/seed, and then consumes data.
    """
    _validate_config(config)
    _validate_structural_state(state, config)
    out = {
        k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
        for k, v in state.items()
    }
    source_ids = np.asarray(out["source_ids"], dtype=np.int64)
    epoch = int(out["epoch"])
    cursor = int(out["cursor"])
    if cursor == len(out["epoch_order"]):
        epoch += 1
        cursor = 0
        out["epoch"] = np.asarray(epoch, dtype=np.int64)
        out["cursor"] = np.asarray(cursor, dtype=np.int64)
        out["epoch_order"] = _order_for(source_ids, config["order_seed"], epoch)
    order = np.asarray(out["epoch_order"], dtype=np.int64)
    stop = min(cursor + int(config["batch_size"]), len(order))
    ids = order[cursor:stop]
    if len(ids) == 0:
        raise ValueError("empty training batch")
    targets = np.stack(
        [np.asarray(targets_by_id[int(i)], dtype=np.float32) for i in ids]
    )
    weight = np.asarray(out["weight"], dtype=np.float32)
    velocity = np.asarray(out["velocity"], dtype=np.float32)
    per_example = weight[None, :] - targets
    grad = np.mean(per_example, axis=0, dtype=np.float32)
    noise, next_key = _random_vector(out["rng_key"], weight.size)
    grad = grad + np.float32(config["noise_scale"]) * noise
    next_velocity = np.float32(config["momentum"]) * velocity + grad
    out["weight"] = weight - np.float32(config["lr"]) * next_velocity
    out["velocity"] = next_velocity
    out["rng_key"] = next_key
    out["cursor"] = np.asarray(stop, dtype=np.int64)
    out["consumed_ids"] = np.concatenate((out["consumed_ids"], ids)).astype(np.int64)
    out["update_ends"] = np.append(out["update_ends"], len(out["consumed_ids"])).astype(
        np.int64
    )
    out["optimizer_step"] = np.asarray(int(out["optimizer_step"]) + 1, dtype=np.int64)
    return out


def save_checkpoint(path, step, config, arrays):
    """Compatibility API that atomically replaces one integrity-bound file."""
    _validate_config(config)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    if {"weight", "velocity", "rng_key"} <= set(arrays):
        _validate_legacy_state(arrays, config)
    digest = _digest(step, config, arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(
                f,
                step=np.int64(step),
                config=np.array(_json_bytes(config).decode()),
                digest=np.array(digest),
                **arrays,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_checkpoint(path, expected_config):
    """Compatibility loader for the original three-function exercise."""
    _validate_config(expected_config)
    with np.load(path, allow_pickle=False) as z:
        step = int(z["step"])
        config = json.loads(str(z["config"]))
        digest = str(z["digest"])
        arrays = {k: z[k] for k in z.files if k not in {"step", "config", "digest"}}
    if config != expected_config:
        raise ValueError("checkpoint configuration mismatch")
    if digest != _digest(step, config, arrays):
        raise ValueError("checkpoint integrity failure")
    if {"weight", "velocity", "rng_key"} <= set(arrays):
        _validate_legacy_state(arrays, expected_config)
    return {"step": step, "config": config, "arrays": arrays}


def initial_stream_state(
    source_ids, config, shape=(2,), initial_weight=None, initial_velocity=None
):
    """Create a tiny, explicit full state for the instructional stream."""
    _validate_stream_config(config)
    ids = np.asarray(source_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("source IDs must be a nonempty unique vector")
    weight = (
        np.array([1.0, -1.0], np.float32)
        if initial_weight is None
        else np.asarray(initial_weight, dtype=np.float32)
    )
    velocity = (
        np.array([0.2, -0.3], np.float32)
        if initial_velocity is None
        else np.asarray(initial_velocity, dtype=np.float32)
    )
    expected_shape = tuple(int(x) for x in config["parameter_shape"])
    if (
        tuple(shape) != expected_shape
        or weight.shape != expected_shape
        or velocity.shape != expected_shape
    ):
        raise ValueError("initial parameter shapes do not match")
    key = np.asarray(config["initial_rng_key"], dtype=np.uint32)
    _key_words(key)
    return {
        "weight": weight,
        "velocity": velocity,
        "rng_key": key,
        "cursor": np.asarray(0, dtype=np.int64),
        "epoch": np.asarray(0, dtype=np.int64),
        "epoch_order": _order_for(ids, config["order_seed"], 0),
        "source_ids": ids,
        "consumed_ids": np.empty((0,), dtype=np.int64),
        "update_ends": np.empty((0,), dtype=np.int64),
        "optimizer_step": np.asarray(0, dtype=np.int64),
    }


def _validate_stream_config(config):
    _validate_config(config)
    required = {"batch_size", "order_seed", "initial_rng_key", "parameter_shape"}
    if not required <= set(config):
        raise ValueError("incomplete stream config")
    if int(config["batch_size"]) <= 0:
        raise ValueError("batch size must be positive")
    shape = config["parameter_shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
    ):
        raise ValueError(
            "parameter_shape must be a nonempty list of positive dimensions"
        )
    _key_words(config["initial_rng_key"])


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_sync(path, value):
    with open(path, "wb") as f:
        f.write(_json_bytes(value))
        f.flush()
        os.fsync(f.fileno())


def _safe_failpoint(failpoint, target):
    if failpoint == target:
        raise RuntimeError("injected crash at " + target)


def _read_current(root):
    p = Path(root) / "CURRENT.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save_generation(
    root, state, config, identities, expected_parent=None, failpoint=None
):
    """Prepare an immutable generation, then arbitrate atomic publication.

    Competing POSIX processes use an expected-parent compare-and-swap under
    ``fcntl.flock``. A conflict leaves the losing immutable generation
    unpublished. The legacy single-file APIs remain platform-compatible Python.
    """
    import fcntl

    _validate_stream_config(config)
    if failpoint not in (None, *FAILPOINTS):
        raise ValueError("unknown failpoint")
    if not isinstance(identities, dict) or not identities:
        raise ValueError("identities must be explicit")
    arrays = {}
    for k, value in state.items():
        arr = np.asarray(value)
        arrays[k] = arr if arr.flags.c_contiguous else np.ascontiguousarray(arr)
    if set(arrays) != STATE_ARRAYS:
        raise ValueError("state schema fields differ")
    _validate_state(arrays, config, identities)
    root = Path(root)
    generations = root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    generation = "g-" + uuid.uuid4().hex
    temp_dir = generations / (".tmp-" + generation)
    temp_dir.mkdir()
    _safe_failpoint(failpoint, "after_temp_create")
    payload = temp_dir / "state.npz"
    import io

    encoded = io.BytesIO()
    np.savez(encoded, **arrays)
    payload_bytes = encoded.getvalue()
    partial = temp_dir / "state.npz.tmp"
    with open(partial, "wb") as f:
        f.write(payload_bytes[: max(1, len(payload_bytes) // 2)])
        f.flush()
        os.fsync(f.fileno())
    _safe_failpoint(failpoint, "during_temp_write")
    with open(partial, "ab") as f:
        f.write(payload_bytes[max(1, len(payload_bytes) // 2) :])
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, payload)
    _fsync_dir(temp_dir)
    payload_hash = hashlib.sha256(payload.read_bytes()).hexdigest()
    _safe_failpoint(failpoint, "after_payload")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generation": generation,
        "parent_generation": expected_parent,
        "config": config,
        "config_sha256": hashlib.sha256(_json_bytes(config)).hexdigest(),
        "identities": identities,
        "state_sha256": payload_hash,
        "array_sha256": _array_digest(arrays),
        "array_schema": {
            name: {"dtype": arr.dtype.str, "shape": list(arr.shape)}
            for name, arr in sorted(arrays.items())
        },
    }
    _write_json_sync(temp_dir / "manifest.json", manifest)
    _fsync_dir(temp_dir)
    _safe_failpoint(failpoint, "after_manifest")
    final_dir = generations / generation
    os.replace(temp_dir, final_dir)
    _fsync_dir(generations)
    manifest_hash = hashlib.sha256(
        (final_dir / "manifest.json").read_bytes()
    ).hexdigest()
    pointer = {
        "schema_version": SCHEMA_VERSION,
        "generation": generation,
        "manifest_sha256": manifest_hash,
    }
    _safe_failpoint(failpoint, "before_publish")
    lock_path = root / ".publish.lock"
    with open(lock_path, "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = _read_current(root)
        actual_parent = None if current is None else current["generation"]
        if actual_parent != expected_parent:
            raise RuntimeError(
                "generation conflict: expected parent is no longer current"
            )
        tmp_pointer = root / (".CURRENT-" + uuid.uuid4().hex + ".tmp")
        _write_json_sync(tmp_pointer, pointer)
        os.replace(tmp_pointer, root / "CURRENT.json")
        _fsync_dir(root)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    _safe_failpoint(failpoint, "after_publish")
    return {
        "generation": generation,
        "parent_generation": expected_parent,
        "manifest_sha256": manifest_hash,
    }


def _validate_state(arrays, config, identities):
    _validate_structural_state(arrays, config)
    if set(arrays) != STATE_ARRAYS:
        raise ValueError("checkpoint state schema mismatch")
    source_ids = arrays["source_ids"]
    order = arrays["epoch_order"]
    consumed = arrays["consumed_ids"]
    update_ends = arrays["update_ends"]
    if (
        source_ids.ndim != 1
        or len(source_ids) == 0
        or len(np.unique(source_ids)) != len(source_ids)
    ):
        raise ValueError("checkpoint source IDs invalid")
    if len(arrays["epoch_order"]) != len(source_ids):
        raise ValueError("checkpoint epoch order length mismatch")
    if int(arrays["epoch"]) < 0 or not 0 <= int(arrays["cursor"]) <= len(
        arrays["epoch_order"]
    ):
        raise ValueError("checkpoint epoch/cursor range mismatch")
    if (
        identities.get("ordered_source_ids_sha256")
        != hashlib.sha256(source_ids.tobytes()).hexdigest()
    ):
        raise ValueError("checkpoint ordered source identity mismatch")
    step = int(arrays["optimizer_step"])
    if step != len(update_ends) or np.any(np.diff(update_ends) <= 0):
        raise ValueError("checkpoint update history invalid")
    if len(consumed) != (int(update_ends[-1]) if len(update_ends) else 0):
        raise ValueError("checkpoint consumed-ID history invalid")
    expected_ends = []
    n = len(source_ids)
    pos = 0
    while pos < len(consumed):
        epoch_remaining = n - (pos % n)
        pos += min(int(config["batch_size"]), epoch_remaining)
        expected_ends.append(pos)
    if not np.array_equal(update_ends, np.asarray(expected_ends, dtype=np.int64)):
        raise ValueError("checkpoint update boundaries mismatch")
    expected_ids = []
    for ep in range((len(consumed) + n - 1) // n):
        expected_ids.extend(_order_for(source_ids, config["order_seed"], ep).tolist())
    if not np.array_equal(
        consumed, np.asarray(expected_ids[: len(consumed)], dtype=np.int64)
    ):
        raise ValueError("checkpoint sample-order history mismatch")
    if len(consumed) == 0:
        expected_epoch, expected_cursor = 0, 0
    else:
        expected_epoch = (len(consumed) - 1) // n
        expected_cursor = (len(consumed) - 1) % n + 1
    if (
        int(arrays["epoch"]) != expected_epoch
        or int(arrays["cursor"]) != expected_cursor
    ):
        raise ValueError("checkpoint epoch cursor mismatch")
    if not np.array_equal(
        order, _order_for(source_ids, config["order_seed"], expected_epoch)
    ):
        raise ValueError("checkpoint current epoch order mismatch")
    if not np.all(np.isfinite(arrays["weight"])) or not np.all(
        np.isfinite(arrays["velocity"])
    ):
        raise ValueError("checkpoint parameter state is nonfinite")
    # Key advancement depends on consumed optimizer updates. Recompute from the
    # frozen initial key without trusting the saved key as its own oracle.
    key = np.asarray(config["initial_rng_key"], dtype=np.uint32)
    for _ in range(step):
        _, key = _random_vector(key, arrays["weight"].size)
    if not np.array_equal(arrays["rng_key"], key):
        raise ValueError("checkpoint PRNG history mismatch")


def _validate_parameter_shapes(arrays, config):
    expected = tuple(int(dim) for dim in config["parameter_shape"])
    weight = np.asarray(arrays["weight"])
    velocity = np.asarray(arrays["velocity"])
    if weight.dtype != np.dtype("float32") or velocity.dtype != np.dtype("float32"):
        raise ValueError("checkpoint parameter dtype mismatch")
    if weight.shape != expected:
        raise ValueError("checkpoint parameter shape differs from caller config")
    if velocity.shape != expected:
        raise ValueError("checkpoint optimizer shape differs from caller config")


def _validate_structural_state(arrays, config):
    if set(arrays) != STATE_ARRAYS:
        raise ValueError("checkpoint state schema mismatch")
    scalar_fields = ("cursor", "epoch", "optimizer_step")
    vector_fields = ("epoch_order", "source_ids", "consumed_ids", "update_ends")
    if any(np.asarray(arrays[k]).shape != () for k in scalar_fields):
        raise ValueError("checkpoint scalar clock rank mismatch")
    if any(np.asarray(arrays[k]).ndim != 1 for k in vector_fields):
        raise ValueError("checkpoint vector history rank mismatch")
    if any(
        np.asarray(arrays[k]).dtype != np.dtype("int64")
        for k in (*scalar_fields, *vector_fields)
    ):
        raise ValueError("checkpoint integer state dtype mismatch")
    _validate_parameter_shapes(arrays, config)
    if np.asarray(arrays["rng_key"]).dtype != np.dtype("uint32") or np.asarray(
        arrays["rng_key"]
    ).shape != (2,):
        raise ValueError("checkpoint PRNG schema mismatch")


def restore_generation(root, expected_config, expected_identities):
    """Load only the fully published generation matching caller expectations."""
    _validate_stream_config(expected_config)
    root = Path(root)
    pointer = _read_current(root)
    if pointer is None or pointer.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("no committed checkpoint generation")
    directory = root / "generations" / pointer["generation"]
    manifest_bytes = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != pointer.get("manifest_sha256"):
        raise ValueError("published manifest integrity failure")
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("generation") != pointer["generation"]
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("checkpoint generation schema mismatch")
    if (
        manifest.get("config") != expected_config
        or manifest.get("config_sha256")
        != hashlib.sha256(_json_bytes(expected_config)).hexdigest()
    ):
        raise ValueError("checkpoint configuration mismatch")
    if manifest.get("identities") != expected_identities:
        raise ValueError("checkpoint source identity mismatch")
    payload = directory / "state.npz"
    if hashlib.sha256(payload.read_bytes()).hexdigest() != manifest.get("state_sha256"):
        raise ValueError("checkpoint payload integrity failure")
    with np.load(payload, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    if _array_digest(arrays) != manifest.get("array_sha256"):
        raise ValueError("checkpoint array integrity failure")
    schema = {
        name: {"dtype": arr.dtype.str, "shape": list(arr.shape)}
        for name, arr in sorted(arrays.items())
    }
    if schema != manifest.get("array_schema"):
        raise ValueError("checkpoint array schema mismatch")
    _validate_state(arrays, expected_config, expected_identities)
    return {
        "state": arrays,
        "config": manifest["config"],
        "identities": manifest["identities"],
        "generation": manifest["generation"],
        "parent_generation": manifest["parent_generation"],
    }
