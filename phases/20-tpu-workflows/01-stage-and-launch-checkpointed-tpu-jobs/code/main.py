"""Stage and launch checkpointed TPU jobs: worked experiments and reference solutions. CPU checks."""

# Define deterministic hashing and atomic checkpoint writes
# Step 1 — Define deterministic hashing and atomic checkpoint writes: Writing to a temporary sibling file, flushing with fsync, and...
# Import hashlib for this computation.
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import jax
import jax.numpy as jnp
import numpy as np


# Function `digest_obj(value)` implementing this stage's computation:
def digest_obj(value):
    # Return `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]` to the caller.
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


# Function `atomic_write_json(path, payload)` implementing this stage's computation:
def atomic_write_json(path, payload):
    # Read or serialize artifact data on disk (`path`).
    path = Path(path)
    # Run `path.parent.mkdir` to perform the next check or state transition.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Run `path.with_suffix` to compute `tmp`.
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Enter managed runtime/context scope for this block:
    with tmp.open("w", encoding="utf-8") as f:
        # Run `json.dump` to perform the next check or state transition.
        json.dump(payload, f, sort_keys=True)
        # Run `f.flush` to perform the next check or state transition.
        f.flush()
        # Run `os.fsync` to perform the next check or state transition.
        os.fsync(f.fileno())
    # Run `os.replace` to perform the next check or state transition.
    os.replace(tmp, path)

# Implement the TPU-ready job runner with warmup separation and resume
# Step 2 — Implement the TPU-ready job runner with warmup separation and resume: Warmup compiles the JIT function without mutating w, b, m_w, m_b,...
# Define `run_tpu_ready_job(run_dir, steps, save_every, resume...)` to evaluate the objective and its automatic derivatives:
def run_tpu_ready_job(run_dir, steps=6, save_every=2, resume=False, seed=0, lr=0.05):
    # Read or serialize artifact data on disk (`run_dir`).
    run_dir = Path(run_dir)
    # Run `run_dir.mkdir` to perform the next check or state transition.
    run_dir.mkdir(parents=True, exist_ok=True)
    # Compute `events_path` from `run_dir / "events.jsonl"`
    events_path = run_dir / "events.jsonl"
    # Compute `ckpt_path` from `run_dir / "checkpoint-latest.json"`
    ckpt_path = run_dir / "checkpoint-latest.json"

    # Function `emit(event)` implementing this stage's computation:
    def emit(event, **fields):
        # Check which hardware backend (`cpu`, `gpu`, or `tpu`) JAX selected for `record`.
        record = {"event": event, "backend": jax.default_backend(), "devices": jax.device_count(), **fields}
        # Enter managed runtime/context scope for this block:
        with events_path.open("a", encoding="utf-8") as f:
            # Read or serialize artifact data on disk (``).
            f.write(json.dumps(record, sort_keys=True) + "\n")
        # Return `record` to the caller.
        return record

    # Generate a uniform grid of points in `x`.
    x = jnp.linspace(-1.0, 1.0, 64, dtype=jnp.float32).reshape(64, 1)
    # Compute `target_y` from `3.0 * x - 0.5`
    target_y = 3.0 * x - 0.5

    # Branch on condition `resume and ckpt_path.exists()`:
    if resume and ckpt_path.exists():
        saved = json.loads(ckpt_path.read_text(encoding="utf-8"))
        step = int(saved["step"])
        w = jnp.array(saved["w"], dtype=jnp.float32)
        b = jnp.array(saved["b"], dtype=jnp.float32)
        m_w = jnp.array(saved["m_w"], dtype=jnp.float32)
        m_b = jnp.array(saved["m_b"], dtype=jnp.float32)
        key = jnp.array(saved["key"], dtype=jnp.uint32)
        emit("restored", step=step, state_hash=saved["state_hash"])
    else:
        step = 0
        w = jnp.zeros((1, 1), dtype=jnp.float32)
        b = jnp.zeros((1,), dtype=jnp.float32)
        m_w = jnp.zeros_like(w)
        m_b = jnp.zeros_like(b)
        key = jax.random.PRNGKey(seed)
        emit("started", step=step)

    @jax.jit
    # Function `train_step(w, b, m_w, m_b, ...)` implementing this stage's computation:
    def train_step(w, b, m_w, m_b, key):
        # Split the PRNG key deterministically into independent subkeys (`(key, subkey)`).
        key, subkey = jax.random.split(key)
        # Draw pseudorandom samples for `idx` using the explicit RNG state.
        idx = jax.random.choice(subkey, 64, shape=(16,), replace=False)
        # Compute `xb, yb` from `x[idx], target_y[idx]`
        xb, yb = x[idx], target_y[idx]

        # Function `loss_fn(w_in, b_in)` implementing this stage's computation:
        def loss_fn(w_in, b_in):
            # Perform matrix contraction / projection to compute `pred`.
            pred = xb @ w_in + b_in
            # Return `jnp.mean((pred - yb) ** 2)` to the caller.
            return jnp.mean((pred - yb) ** 2)

        # Evaluate both scalar loss and parameter gradients in one pass (`(loss, (gw, gb))`).
        loss, (gw, gb) = jax.value_and_grad(loss_fn, argnums=(0, 1))(w, b)
        # Compute `m_w` from `0.8 * m_w + gw`
        m_w = 0.8 * m_w + gw
        # Compute `m_b` from `0.8 * m_b + gb`
        m_b = 0.8 * m_b + gb
        # Compute `w` from `w - lr * m_w`
        w = w - lr * m_w
        # Compute `b` from `b - lr * m_b`
        b = b - lr * m_b
        # Return `(w, b, m_w, m_b, key, loss)` to the caller.
        return w, b, m_w, m_b, key, loss

    # Record execution timing or profiler trace in `t0`.
    t0 = time.perf_counter()
    # Run `train_step` to compute `warm`.
    warm = train_step(w, b, m_w, m_b, key)
    # Synchronize host execution until asynchronous device computation completes.
    jax.block_until_ready(warm)
    # Record execution timing or profiler trace in `warmup_ms`.
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    # Run `emit` to perform the next check or state transition.
    emit("warmup", warmup_ms=warmup_ms, step=step)

    # Compute `step_losses` from `[]`
    step_losses = []
    # Compute `step_times_ms` from `[]`
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
        }
        state_payload["state_hash"] = digest_obj(state_payload)
        step_losses.append(float(loss))
        step_times_ms.append(float(elapsed_ms))
        emit("step", step=step, loss=float(loss), step_ms=elapsed_ms, state_hash=state_payload["state_hash"])
        if step % save_every == 0 or step == steps:
            atomic_write_json(ckpt_path, state_payload)
            emit("checkpoint", step=step, state_hash=state_payload["state_hash"])

    # Run `emit` to perform the next check or state transition.
    emit("completed", step=step, state_hash=state_payload["state_hash"])
    # Return `{'final_step': step, 'state_hash': state_payload['state_hash'], 'w': float(np.asarray(w).squeeze()), 'b': float(np.asarray(b).squeeze()), 'warmup_ms': warmup_ms, 'step_losses': step_losses, 'step_times_ms': step_times_ms}` to the caller.
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
# Step 3 — Compare a checkpoint-resumed run against an uninterrupted control run: Because parameters, momentum buffers, and PRNG keys are all...
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix="tpu-launch-") as tmp:
    # Read or serialize artifact data on disk (`root`).
    root = Path(tmp)
    # Run `run_tpu_ready_job` to compute `part1`.
    part1 = run_tpu_ready_job(root / "resumed", steps=3, save_every=3, resume=False)
    # Run `run_tpu_ready_job` to compute `part2`.
    part2 = run_tpu_ready_job(root / "resumed", steps=6, save_every=3, resume=True)
    # Run `run_tpu_ready_job` to compute `control`.
    control = run_tpu_ready_job(root / "control", steps=6, save_every=3, resume=False)
    # Compute `resumed_losses` from `part1["step_losses"] + part2["step_losses"]`
    resumed_losses = part1["step_losses"] + part2["step_losses"]

# Assert invariant `part2["state_hash"] == control["state_hash"]` holds
assert part2["state_hash"] == control["state_hash"]
# Check numerical equivalence within tolerance: `np.allclose(resumed_losses`
assert np.allclose(resumed_losses, control["step_losses"])
# Print the observed values to compare against the expected result.
print("Resumed state_hash matches uninterrupted control:", part2["state_hash"])
# Print diagnostic summary of the computed outputs.
print("Losses across 6 steps:", [round(v, 5) for v in control["step_losses"]])

# Step 1 — Define deterministic hashing and atomic checkpoint writes: Writing to a temporary sibling file, flushing with fsync, and...
# Import hashlib for this computation.
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import jax
import jax.numpy as jnp
import numpy as np


# Function `digest_obj(value)` implementing this stage's computation:
def digest_obj(value):
    # Return `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]` to the caller.
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


# Function `atomic_write_json(path, payload)` implementing this stage's computation:
def atomic_write_json(path, payload):
    # Read or serialize artifact data on disk (`path`).
    path = Path(path)
    # Run `path.parent.mkdir` to perform the next check or state transition.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Run `path.with_suffix` to compute `tmp`.
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Enter managed runtime/context scope for this block:
    with tmp.open("w", encoding="utf-8") as f:
        # Run `json.dump` to perform the next check or state transition.
        json.dump(payload, f, sort_keys=True)
        # Run `f.flush` to perform the next check or state transition.
        f.flush()
        # Run `os.fsync` to perform the next check or state transition.
        os.fsync(f.fileno())
    # Run `os.replace` to perform the next check or state transition.
    os.replace(tmp, path)


# Step 2 — Implement the TPU-ready job runner with warmup separation and resume: Warmup compiles the JIT function without mutating w, b, m_w, m_b,...
# Define `run_tpu_ready_job(run_dir, steps, save_every, resume...)` to evaluate the objective and its automatic derivatives:
def run_tpu_ready_job(run_dir, steps=6, save_every=2, resume=False, seed=0, lr=0.05):
    # Read or serialize artifact data on disk (`run_dir`).
    run_dir = Path(run_dir)
    # Run `run_dir.mkdir` to perform the next check or state transition.
    run_dir.mkdir(parents=True, exist_ok=True)
    # Compute `events_path` from `run_dir / "events.jsonl"`
    events_path = run_dir / "events.jsonl"
    # Compute `ckpt_path` from `run_dir / "checkpoint-latest.json"`
    ckpt_path = run_dir / "checkpoint-latest.json"

    # Function `emit(event)` implementing this stage's computation:
    def emit(event, **fields):
        # Check which hardware backend (`cpu`, `gpu`, or `tpu`) JAX selected for `record`.
        record = {"event": event, "backend": jax.default_backend(), "devices": jax.device_count(), **fields}
        # Enter managed runtime/context scope for this block:
        with events_path.open("a", encoding="utf-8") as f:
            # Read or serialize artifact data on disk (``).
            f.write(json.dumps(record, sort_keys=True) + "\n")
        # Return `record` to the caller.
        return record

    # Generate a uniform grid of points in `x`.
    x = jnp.linspace(-1.0, 1.0, 64, dtype=jnp.float32).reshape(64, 1)
    # Compute `target_y` from `3.0 * x - 0.5`
    target_y = 3.0 * x - 0.5

    # Branch on condition `resume and ckpt_path.exists()`:
    if resume and ckpt_path.exists():
        saved = json.loads(ckpt_path.read_text(encoding="utf-8"))
        step = int(saved["step"])
        w = jnp.array(saved["w"], dtype=jnp.float32)
        b = jnp.array(saved["b"], dtype=jnp.float32)
        m_w = jnp.array(saved["m_w"], dtype=jnp.float32)
        m_b = jnp.array(saved["m_b"], dtype=jnp.float32)
        key = jnp.array(saved["key"], dtype=jnp.uint32)
        emit("restored", step=step, state_hash=saved["state_hash"])
    else:
        step = 0
        w = jnp.zeros((1, 1), dtype=jnp.float32)
        b = jnp.zeros((1,), dtype=jnp.float32)
        m_w = jnp.zeros_like(w)
        m_b = jnp.zeros_like(b)
        key = jax.random.PRNGKey(seed)
        emit("started", step=step)

    @jax.jit
    # Function `train_step(w, b, m_w, m_b, ...)` implementing this stage's computation:
    def train_step(w, b, m_w, m_b, key):
        # Split the PRNG key deterministically into independent subkeys (`(key, subkey)`).
        key, subkey = jax.random.split(key)
        # Draw pseudorandom samples for `idx` using the explicit RNG state.
        idx = jax.random.choice(subkey, 64, shape=(16,), replace=False)
        # Compute `xb, yb` from `x[idx], target_y[idx]`
        xb, yb = x[idx], target_y[idx]

        # Function `loss_fn(w_in, b_in)` implementing this stage's computation:
        def loss_fn(w_in, b_in):
            # Perform matrix contraction / projection to compute `pred`.
            pred = xb @ w_in + b_in
            # Return `jnp.mean((pred - yb) ** 2)` to the caller.
            return jnp.mean((pred - yb) ** 2)

        # Evaluate both scalar loss and parameter gradients in one pass (`(loss, (gw, gb))`).
        loss, (gw, gb) = jax.value_and_grad(loss_fn, argnums=(0, 1))(w, b)
        # Compute `m_w` from `0.8 * m_w + gw`
        m_w = 0.8 * m_w + gw
        # Compute `m_b` from `0.8 * m_b + gb`
        m_b = 0.8 * m_b + gb
        # Compute `w` from `w - lr * m_w`
        w = w - lr * m_w
        # Compute `b` from `b - lr * m_b`
        b = b - lr * m_b
        # Return `(w, b, m_w, m_b, key, loss)` to the caller.
        return w, b, m_w, m_b, key, loss

    # Record execution timing or profiler trace in `t0`.
    t0 = time.perf_counter()
    # Run `train_step` to compute `warm`.
    warm = train_step(w, b, m_w, m_b, key)
    # Synchronize host execution until asynchronous device computation completes.
    jax.block_until_ready(warm)
    # Record execution timing or profiler trace in `warmup_ms`.
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    # Run `emit` to perform the next check or state transition.
    emit("warmup", warmup_ms=warmup_ms, step=step)

    # Compute `step_losses` from `[]`
    step_losses = []
    # Compute `step_times_ms` from `[]`
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
        }
        state_payload["state_hash"] = digest_obj(state_payload)
        step_losses.append(float(loss))
        step_times_ms.append(float(elapsed_ms))
        emit("step", step=step, loss=float(loss), step_ms=elapsed_ms, state_hash=state_payload["state_hash"])
        if step % save_every == 0 or step == steps:
            atomic_write_json(ckpt_path, state_payload)
            emit("checkpoint", step=step, state_hash=state_payload["state_hash"])

    # Run `emit` to perform the next check or state transition.
    emit("completed", step=step, state_hash=state_payload["state_hash"])
    # Return `{'final_step': step, 'state_hash': state_payload['state_hash'], 'w': float(np.asarray(w).squeeze()), 'b': float(np.asarray(b).squeeze()), 'warmup_ms': warmup_ms, 'step_losses': step_losses, 'step_times_ms': step_times_ms}` to the caller.
    return {
        "final_step": step,
        "state_hash": state_payload["state_hash"],
        "w": float(np.asarray(w).squeeze()),
        "b": float(np.asarray(b).squeeze()),
        "warmup_ms": warmup_ms,
        "step_losses": step_losses,
        "step_times_ms": step_times_ms,
    }


# Step 3 — Compare a checkpoint-resumed run against an uninterrupted control run: Because parameters, momentum buffers, and PRNG keys are all...
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix="tpu-launch-") as tmp:
    # Read or serialize artifact data on disk (`root`).
    root = Path(tmp)
    # Run `run_tpu_ready_job` to compute `part1`.
    part1 = run_tpu_ready_job(root / "resumed", steps=3, save_every=3, resume=False)
    # Run `run_tpu_ready_job` to compute `part2`.
    part2 = run_tpu_ready_job(root / "resumed", steps=6, save_every=3, resume=True)
    # Run `run_tpu_ready_job` to compute `control`.
    control = run_tpu_ready_job(root / "control", steps=6, save_every=3, resume=False)
    # Compute `resumed_losses` from `part1["step_losses"] + part2["step_losses"]`
    resumed_losses = part1["step_losses"] + part2["step_losses"]

# Assert invariant `part2["state_hash"] == control["state_hash"]` holds
assert part2["state_hash"] == control["state_hash"]
# Check numerical equivalence within tolerance: `np.allclose(resumed_losses`
assert np.allclose(resumed_losses, control["step_losses"])
# Print the observed values to compare against the expected result.
print("Resumed state_hash matches uninterrupted control:", part2["state_hash"])
# Print diagnostic summary of the computed outputs.
print("Losses across 6 steps:", [round(v, 5) for v in control["step_losses"]])

# Figure data experiment
# Compute figure data for: Interrupted-and-resumed trajectory vs uninterrupted 6-step control run
# Compute `visual_data` from `{`
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
# Experiment — Compare JIT warmup latency against steady-state step latency: Logging warmup separately in events.jsonl prevents compilation...
median_step_ms = float(np.median(control["step_times_ms"]))
# Print the observed values to compare against the expected result.
print("Warmup ms:", round(control["warmup_ms"], 3), "Median steady step ms:", round(median_step_ms, 4))
# Assert invariant `control["warmup_ms"] > median_step_ms` holds
assert control["warmup_ms"] > median_step_ms

# Experiment: Verify what breaks if momentum state is omitted on resume
# Experiment — Verify what breaks if momentum state is omitted on resume: Restoring model weights alone is not a valid checkpoint resume;...
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix="tpu-bad-resume-") as tmp:
    # Read or serialize artifact data on disk (`bad_dir`).
    bad_dir = Path(tmp) / "bad"
    # Run `run_tpu_ready_job` to perform the next check or state transition.
    run_tpu_ready_job(bad_dir, steps=3, save_every=3, resume=False)
    # Read or serialize artifact data on disk (`ckpt`).
    ckpt = json.loads((bad_dir / "checkpoint-latest.json").read_text(encoding="utf-8"))
    # Compute `ckpt["m_w"]` from `[[0.0]]`
    ckpt["m_w"] = [[0.0]]
    # Compute `ckpt["m_b"]` from `[0.0]`
    ckpt["m_b"] = [0.0]
    # Read or serialize artifact data on disk (``).
    (bad_dir / "checkpoint-latest.json").write_text(json.dumps(ckpt), encoding="utf-8")
    # Run `run_tpu_ready_job` to compute `bad_part2`.
    bad_part2 = run_tpu_ready_job(bad_dir, steps=6, save_every=3, resume=True)
# Print the observed values to compare against the expected result.
print("Drifted state_hash vs control:", bad_part2["state_hash"], control["state_hash"])
# Assert invariant `bad_part2["state_hash"] != control["state_hash"]` holds
assert bad_part2["state_hash"] != control["state_hash"]

# Reference solution. Try the exercise before reading this.
# Exercise solution: Run run_tpu_ready_job for 4 steps (save_every=2), resume to 8 steps,...
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix="tpu-ex-") as tmp:
    # Read or serialize artifact data on disk (`ex_root`).
    ex_root = Path(tmp)
    # Run `run_tpu_ready_job` to compute `r1`.
    r1 = run_tpu_ready_job(ex_root / "resumed", steps=4, save_every=2, resume=False)
    # Run `run_tpu_ready_job` to compute `r2`.
    r2 = run_tpu_ready_job(ex_root / "resumed", steps=8, save_every=2, resume=True)
    # Run `run_tpu_ready_job` to compute `ctrl8`.
    ctrl8 = run_tpu_ready_job(ex_root / "control", steps=8, save_every=2, resume=False)
# Print the observed values to compare against the expected result.
print("Step 8 resumed vs control hash:", r2["state_hash"], ctrl8["state_hash"])
# Assert invariant `r2["state_hash"] == ctrl8["state_hash"] and len(r1["step_losses"]...` holds
assert r2["state_hash"] == ctrl8["state_hash"] and len(r1["step_losses"] + r2["step_losses"]) == 8

# Reference practice: Inspect the JSONL event sequence from a resumed run
# Inspect the JSONL event sequence from a resumed run (Foundations): The single append-only events.jsonl log captures both the...
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix="tpu-events-") as tmp:
    # Read or serialize artifact data on disk (`ev_dir`).
    ev_dir = Path(tmp) / "job"
    # Run `run_tpu_ready_job` to perform the next check or state transition.
    run_tpu_ready_job(ev_dir, steps=2, save_every=2, resume=False)
    # Run `run_tpu_ready_job` to perform the next check or state transition.
    run_tpu_ready_job(ev_dir, steps=4, save_every=2, resume=True)
    # Read or serialize artifact data on disk (`event_names`).
    event_names = [json.loads(line)["event"] for line in (ev_dir / "events.jsonl").read_text().splitlines()]
# Print the observed values to compare against the expected result.
print("Recorded event sequence:", event_names)
# Assert invariant `"started" in event_names and "restored" in event_names and event_...` holds
assert "started" in event_names and "restored" in event_names and event_names[-1] == "completed"

# Reference practice: Verify atomic checkpoint file replacement
# Verify atomic checkpoint file replacement (Transfer / diagnosis): os.replace atomically swaps the flushed .tmp file into...
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix="tpu-atomic-") as tmp:
    # Read or serialize artifact data on disk (`target`).
    target = Path(tmp) / "checkpoint-latest.json"
    # Run `atomic_write_json` to perform the next check or state transition.
    atomic_write_json(target, {"step": 2})
    # Run `atomic_write_json` to perform the next check or state transition.
    atomic_write_json(target, {"step": 4})
    # Read or serialize artifact data on disk (`leftovers`).
    leftovers = list(Path(tmp).glob("*.tmp"))
    # Read or serialize artifact data on disk (`loaded`).
    loaded = json.loads(target.read_text(encoding="utf-8"))
# Print the observed values to compare against the expected result.
print("Loaded step:", loaded["step"], "Leftover tmp files:", leftovers)
# Assert invariant `loaded["step"] == 4 and leftovers == []` holds
assert loaded["step"] == 4 and leftovers == []
print("PASS: tpu-02")
