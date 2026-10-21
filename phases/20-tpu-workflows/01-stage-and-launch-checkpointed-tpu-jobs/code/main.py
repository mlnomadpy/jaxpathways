"""Stage and launch checkpointed TPU jobs: worked experiments and reference solutions. CPU checks."""

# Define deterministic hashing and atomic checkpoint writes
# Step 1 — Define deterministic hashing and atomic checkpoint writes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import jax
import jax.numpy as jnp
import numpy as np


def digest_obj(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()



def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Enter managed runtime/context scope for this block:
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

# Implement the TPU-ready job runner with warmup separation and resume
# Step 2 — Implement the TPU-ready job runner with warmup separation and resume
def run_tpu_ready_job(run_dir, steps=6, save_every=2, resume=False, seed=0, lr=0.05):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    ckpt_path = run_dir / "checkpoint-latest.json"

    def emit(event, **fields):
        # Check which hardware backend (`cpu`, `gpu`, or `tpu`) JAX selected for `record`.
        record = {"event": event, "backend": jax.default_backend(), "devices": jax.device_count(), **fields}
        # Enter managed runtime/context scope for this block:
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    x = jnp.linspace(-1.0, 1.0, 64, dtype=jnp.float32).reshape(64, 1)
    target_y = 3.0 * x - 0.5

    config = {"schema": 1, "seed": int(seed), "lr": float(lr), "data": "linspace64-target-3x-minus-0.5-v1", "batch_size": 16, "optimizer": "momentum-0.8-noise-key-v1"}
    config_hash = digest_obj(config)
    # A requested resume requires an existing, compatible, intact checkpoint.
    if resume:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume requested but checkpoint is missing: {ckpt_path}")
        saved = json.loads(ckpt_path.read_text(encoding="utf-8"))
        supplied_hash = saved.pop("state_hash", None)
        if supplied_hash != digest_obj(saved):
            raise ValueError("checkpoint state hash mismatch")
        if saved.get("config_hash") != config_hash or saved.get("config") != config:
            raise ValueError("checkpoint run configuration mismatch")
        step = int(saved["step"])
        w = jnp.array(saved["w"], dtype=jnp.float32)
        b = jnp.array(saved["b"], dtype=jnp.float32)
        m_w = jnp.array(saved["m_w"], dtype=jnp.float32)
        m_b = jnp.array(saved["m_b"], dtype=jnp.float32)
        key = jnp.array(saved["key"], dtype=jnp.uint32)
        state_payload = dict(saved)
        state_payload["state_hash"] = supplied_hash
        emit("restored", step=step, state_hash=supplied_hash)
    else:
        step = 0
        w = jnp.zeros((1, 1), dtype=jnp.float32)
        b = jnp.zeros((1,), dtype=jnp.float32)
        m_w = jnp.zeros_like(w)
        m_b = jnp.zeros_like(b)
        key = jax.random.PRNGKey(seed)
        emit("started", step=step)
        state_payload = {"step": 0, "w": np.asarray(w).tolist(), "b": np.asarray(b).tolist(), "m_w": np.asarray(m_w).tolist(), "m_b": np.asarray(m_b).tolist(), "key": np.asarray(key).tolist(), "config": config, "config_hash": config_hash}
        state_payload["state_hash"] = digest_obj(state_payload)
        atomic_write_json(ckpt_path, state_payload)
    if step > steps:
        raise ValueError(f"checkpoint step {step} exceeds requested final step {steps}")

    @jax.jit
    def train_step(w, b, m_w, m_b, key):
        # Split the PRNG key deterministically into independent subkeys (`(key, subkey)`).
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, 64, shape=(16,), replace=False)
        xb, yb = x[idx], target_y[idx]

        def loss_fn(w_in, b_in):
            # Perform matrix contraction / projection to compute `pred`.
            pred = xb @ w_in + b_in
            return jnp.mean((pred - yb) ** 2)

        loss, (gw, gb) = jax.value_and_grad(loss_fn, argnums=(0, 1))(w, b)
        m_w = 0.8 * m_w + gw
        m_b = 0.8 * m_b + gb
        w = w - lr * m_w
        b = b - lr * m_b
        return w, b, m_w, m_b, key, loss

    if step == steps:
        emit("completed", step=step, state_hash=state_payload["state_hash"])
        return {"final_step": step, "state_hash": state_payload["state_hash"], "w": float(np.asarray(w).squeeze()), "b": float(np.asarray(b).squeeze()), "warmup_ms": 0.0, "step_losses": [], "step_times_ms": []}

    t0 = time.perf_counter()
    warm = train_step(w, b, m_w, m_b, key)
    # Wait for device work to finish before timing it or comparing its result on the host.
    jax.block_until_ready(warm)
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    emit("warmup", warmup_ms=warmup_ms, step=step)

    step_losses = []
    step_times_ms = []
    while step < steps:
        t_step = time.perf_counter()
        w, b, m_w, m_b, key, loss = train_step(w, b, m_w, m_b, key)
        jax.block_until_ready((w, b, m_w, m_b, key, loss))
        elapsed_ms = (time.perf_counter() - t_step) * 1000.0
        step += 1
        state_payload = {
            "step": step,
            "w": np.asarray(w).tolist(),
            "b": np.asarray(b).tolist(),
            "m_w": np.asarray(m_w).tolist(),
            "m_b": np.asarray(m_b).tolist(),
            "key": np.asarray(key).tolist(),
            "config": config,
            "config_hash": config_hash,
        }
        state_payload["state_hash"] = digest_obj(state_payload)
        step_losses.append(float(loss))
        step_times_ms.append(float(elapsed_ms))
        emit("step", step=step, loss=float(loss), step_ms=elapsed_ms, state_hash=state_payload["state_hash"])
        if step % save_every == 0 or step == steps:
            atomic_write_json(ckpt_path, state_payload)
            emit("checkpoint", step=step, state_hash=state_payload["state_hash"])

    emit("completed", step=step, state_hash=state_payload["state_hash"])
    return {
        "final_step": step,
        "state_hash": state_payload["state_hash"],
        "w": float(np.asarray(w).squeeze()),
        "b": float(np.asarray(b).squeeze()),
        "warmup_ms": warmup_ms,
        "step_losses": step_losses,
        "step_times_ms": step_times_ms,
    }


# Compare a checkpoint-resumed run against an uninterrupted control run
# Step 3 — Compare a checkpoint-resumed run against an uninterrupted control run
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory(prefix="tpu-launch-") as tmp:
    root = Path(tmp)
    part1 = run_tpu_ready_job(root / "resumed", steps=3, save_every=3, resume=False)
    part2 = run_tpu_ready_job(root / "resumed", steps=6, save_every=3, resume=True)
    control = run_tpu_ready_job(root / "control", steps=6, save_every=3, resume=False)
    resumed_losses = part1["step_losses"] + part2["step_losses"]

# Assert invariant `part2["state_hash"] == control["state_hash"]` holds
assert part2["state_hash"] == control["state_hash"]
assert np.allclose(resumed_losses, control["step_losses"])
print("Resumed state_hash matches uninterrupted control:", part2["state_hash"])
print("Losses across 6 steps:", [round(v, 5) for v in control["step_losses"]])

# Step 1 — Define deterministic hashing and atomic checkpoint writes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import jax
import jax.numpy as jnp
import numpy as np


def digest_obj(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()



def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# Step 2 — Implement the TPU-ready job runner with warmup separation and resume
def run_tpu_ready_job(run_dir, steps=6, save_every=2, resume=False, seed=0, lr=0.05):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    ckpt_path = run_dir / "checkpoint-latest.json"

    def emit(event, **fields):
        # Record the backend selected by this JAX process in the event.
        record = {"event": event, "backend": jax.default_backend(), "devices": jax.device_count(), **fields}
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    x = jnp.linspace(-1.0, 1.0, 64, dtype=jnp.float32).reshape(64, 1)
    target_y = 3.0 * x - 0.5

    config = {"schema": 1, "seed": int(seed), "lr": float(lr), "data": "linspace64-target-3x-minus-0.5-v1", "batch_size": 16, "optimizer": "momentum-0.8-noise-key-v1"}
    config_hash = digest_obj(config)
    # A requested resume requires an existing, compatible, intact checkpoint.
    if resume:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume requested but checkpoint is missing: {ckpt_path}")
        saved = json.loads(ckpt_path.read_text(encoding="utf-8"))
        supplied_hash = saved.pop("state_hash", None)
        if supplied_hash != digest_obj(saved):
            raise ValueError("checkpoint state hash mismatch")
        if saved.get("config_hash") != config_hash or saved.get("config") != config:
            raise ValueError("checkpoint run configuration mismatch")
        step = int(saved["step"])
        w = jnp.array(saved["w"], dtype=jnp.float32)
        b = jnp.array(saved["b"], dtype=jnp.float32)
        m_w = jnp.array(saved["m_w"], dtype=jnp.float32)
        m_b = jnp.array(saved["m_b"], dtype=jnp.float32)
        key = jnp.array(saved["key"], dtype=jnp.uint32)
        state_payload = dict(saved)
        state_payload["state_hash"] = supplied_hash
        emit("restored", step=step, state_hash=supplied_hash)
    else:
        step = 0
        w = jnp.zeros((1, 1), dtype=jnp.float32)
        b = jnp.zeros((1,), dtype=jnp.float32)
        m_w = jnp.zeros_like(w)
        m_b = jnp.zeros_like(b)
        key = jax.random.PRNGKey(seed)
        emit("started", step=step)
        state_payload = {"step": 0, "w": np.asarray(w).tolist(), "b": np.asarray(b).tolist(), "m_w": np.asarray(m_w).tolist(), "m_b": np.asarray(m_b).tolist(), "key": np.asarray(key).tolist(), "config": config, "config_hash": config_hash}
        state_payload["state_hash"] = digest_obj(state_payload)
        atomic_write_json(ckpt_path, state_payload)
    if step > steps:
        raise ValueError(f"checkpoint step {step} exceeds requested final step {steps}")

    @jax.jit
    def train_step(w, b, m_w, m_b, key):
        # Split the PRNG key deterministically into independent subkeys (`(key, subkey)`).
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, 64, shape=(16,), replace=False)
        xb, yb = x[idx], target_y[idx]

        def loss_fn(w_in, b_in):
            pred = xb @ w_in + b_in
            return jnp.mean((pred - yb) ** 2)

        loss, (gw, gb) = jax.value_and_grad(loss_fn, argnums=(0, 1))(w, b)
        m_w = 0.8 * m_w + gw
        m_b = 0.8 * m_b + gb
        w = w - lr * m_w
        b = b - lr * m_b
        return w, b, m_w, m_b, key, loss

    if step == steps:
        emit("completed", step=step, state_hash=state_payload["state_hash"])
        return {"final_step": step, "state_hash": state_payload["state_hash"], "w": float(np.asarray(w).squeeze()), "b": float(np.asarray(b).squeeze()), "warmup_ms": 0.0, "step_losses": [], "step_times_ms": []}

    t0 = time.perf_counter()
    warm = train_step(w, b, m_w, m_b, key)
    # Wait for device work to finish before timing it or comparing its result on the host.
    jax.block_until_ready(warm)
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    emit("warmup", warmup_ms=warmup_ms, step=step)

    step_losses = []
    step_times_ms = []
    while step < steps:
        t_step = time.perf_counter()
        w, b, m_w, m_b, key, loss = train_step(w, b, m_w, m_b, key)
        jax.block_until_ready((w, b, m_w, m_b, key, loss))
        elapsed_ms = (time.perf_counter() - t_step) * 1000.0
        step += 1
        state_payload = {
            "step": step,
            "w": np.asarray(w).tolist(),
            "b": np.asarray(b).tolist(),
            "m_w": np.asarray(m_w).tolist(),
            "m_b": np.asarray(m_b).tolist(),
            "key": np.asarray(key).tolist(),
            "config": config,
            "config_hash": config_hash,
        }
        state_payload["state_hash"] = digest_obj(state_payload)
        step_losses.append(float(loss))
        step_times_ms.append(float(elapsed_ms))
        emit("step", step=step, loss=float(loss), step_ms=elapsed_ms, state_hash=state_payload["state_hash"])
        if step % save_every == 0 or step == steps:
            atomic_write_json(ckpt_path, state_payload)
            emit("checkpoint", step=step, state_hash=state_payload["state_hash"])

    emit("completed", step=step, state_hash=state_payload["state_hash"])
    return {
        "final_step": step,
        "state_hash": state_payload["state_hash"],
        "w": float(np.asarray(w).squeeze()),
        "b": float(np.asarray(b).squeeze()),
        "warmup_ms": warmup_ms,
        "step_losses": step_losses,
        "step_times_ms": step_times_ms,
    }



# Step 3 — Compare a checkpoint-resumed run against an uninterrupted control run
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory(prefix="tpu-launch-") as tmp:
    root = Path(tmp)
    part1 = run_tpu_ready_job(root / "resumed", steps=3, save_every=3, resume=False)
    part2 = run_tpu_ready_job(root / "resumed", steps=6, save_every=3, resume=True)
    control = run_tpu_ready_job(root / "control", steps=6, save_every=3, resume=False)
    resumed_losses = part1["step_losses"] + part2["step_losses"]

# Assert invariant `part2["state_hash"] == control["state_hash"]` holds
assert part2["state_hash"] == control["state_hash"]
assert np.allclose(resumed_losses, control["step_losses"])
print("Resumed state_hash matches uninterrupted control:", part2["state_hash"])
print("Losses across 6 steps:", [round(v, 5) for v in control["step_losses"]])

# Figure data experiment
visual_data = {
    'kind': 'line',
    'x': [1, 2, 3, 4, 5, 6],
    'xlabel': 'training step',
    'ylabel': 'minibatch mean-squared loss',
    'boundaries': [3],
    'series': [
        {'label': 'uninterrupted 6-step control', 'y': [float(v) for v in control['step_losses']]},
        {'label': 'stopped at step 3 + resumed (steps 4-6)', 'y': [float(v) for v in resumed_losses]},
    ],
}

# Experiment: Compare JIT warmup latency against steady-state step latency
# Experiment — Compare JIT warmup latency against steady-state step latency
median_step_ms = float(np.median(control["step_times_ms"]))
print("Warmup ms:", round(control["warmup_ms"], 3), "Median steady step ms:", round(median_step_ms, 4))
# Assert invariant `control["warmup_ms"] > median_step_ms` holds
assert control["warmup_ms"] > median_step_ms

# Experiment: Verify checkpoint integrity rejects omitted momentum
# Verify checkpoint integrity rejects a modified momentum buffer
with tempfile.TemporaryDirectory(prefix="tpu-bad-resume-") as tmp:
    bad_dir = Path(tmp) / "bad"
    run_tpu_ready_job(bad_dir, steps=3, save_every=3, resume=False)
    ckpt_path = bad_dir / "checkpoint-latest.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    ckpt["m_w"] = [[0.0]]
    ckpt["m_b"] = [0.0]
    ckpt_path.write_text(json.dumps(ckpt), encoding="utf-8")
    try:
        run_tpu_ready_job(bad_dir, steps=6, save_every=3, resume=True)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("modified checkpoint was accepted")
print("Resume rejection:", message)
assert message == "checkpoint state hash mismatch"

# Reference solution. Try the exercise before reading this.
# Exercise solution
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory(prefix="tpu-ex-") as tmp:
    ex_root = Path(tmp)
    r1 = run_tpu_ready_job(ex_root / "resumed", steps=4, save_every=2, resume=False)
    r2 = run_tpu_ready_job(ex_root / "resumed", steps=8, save_every=2, resume=True)
    ctrl8 = run_tpu_ready_job(ex_root / "control", steps=8, save_every=2, resume=False)
print("Step 8 resumed vs control hash:", r2["state_hash"], ctrl8["state_hash"])
assert r2["state_hash"] == ctrl8["state_hash"] and len(r1["step_losses"] + r2["step_losses"]) == 8

# Reference practice: Inspect the JSONL event sequence from a resumed run
# Inspect the JSONL event sequence from a resumed run (Foundations)
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory(prefix="tpu-events-") as tmp:
    ev_dir = Path(tmp) / "job"
    run_tpu_ready_job(ev_dir, steps=2, save_every=2, resume=False)
    run_tpu_ready_job(ev_dir, steps=4, save_every=2, resume=True)
    event_names = [json.loads(line)["event"] for line in (ev_dir / "events.jsonl").read_text().splitlines()]
print("Recorded event sequence:", event_names)
assert "started" in event_names and "restored" in event_names and event_names[-1] == "completed"

# Reference practice: Verify atomic checkpoint file replacement
# Verify atomic checkpoint file replacement (Transfer / diagnosis)
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory(prefix="tpu-atomic-") as tmp:
    target = Path(tmp) / "checkpoint-latest.json"
    atomic_write_json(target, {"step": 2})
    atomic_write_json(target, {"step": 4})
    leftovers = list(Path(tmp).glob("*.tmp"))
    loaded = json.loads(target.read_text(encoding="utf-8"))
print("Loaded step:", loaded["step"], "Leftover tmp files:", leftovers)
# Assert invariant `loaded["step"] == 4 and leftovers == []` holds
assert loaded["step"] == 4 and leftovers == []
print("PASS: tpu-02")
