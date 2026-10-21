"""Training cursor checkpoint with optimizer momentum and explicit PRNG continuation."""

import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np


def _digest(step, config, arrays):
    h = hashlib.sha256(
        json.dumps({"step": step, "config": config}, sort_keys=True).encode()
    )
    for name in sorted(arrays):
        h.update(name.encode())
        h.update(np.asarray(arrays[name]).tobytes())
    return h.hexdigest()


def train_step(state, batch, config):
    w = np.asarray(state["weight"])
    velocity = np.asarray(state["velocity"])
    key = np.asarray(state["rng_key"], dtype=np.uint32)
    seed = int(key[0]) | (int(key[1]) << 32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=w.shape).astype(np.float32) * config["noise_scale"]
    grad = (w - batch) + noise
    next_velocity = config["momentum"] * velocity + grad
    return {
        "weight": w - config["lr"] * next_velocity,
        "velocity": next_velocity,
        "rng_key": rng.bit_generator.random_raw(2).astype(np.uint32),
        "cursor": np.asarray(state["cursor"] + 1, dtype=np.int64),
    }


def save_checkpoint(path, step, config, arrays):
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    digest = _digest(step, config, arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(
                f,
                step=np.int64(step),
                config=np.array(json.dumps(config, sort_keys=True)),
                digest=np.array(digest),
                **arrays
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_checkpoint(path, expected_config):
    with np.load(path, allow_pickle=False) as z:
        step = int(z["step"])
        config = json.loads(str(z["config"]))
        digest = str(z["digest"])
        arrays = {k: z[k] for k in z.files if k not in {"step", "config", "digest"}}
    if config != expected_config:
        raise ValueError("checkpoint configuration mismatch")
    if digest != _digest(step, config, arrays):
        raise ValueError("checkpoint integrity failure")
    return {"step": step, "config": config, "arrays": arrays}
