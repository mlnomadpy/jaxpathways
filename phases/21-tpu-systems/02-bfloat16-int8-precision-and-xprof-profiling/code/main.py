"""bfloat16, int8 precision, and XProf profiling on TPU: worked experiments and reference solutions. CPU checks."""

# Compare BF16 (FP32 vs BF16 accumulation) and per-channel INT8 matmul errors
# Step 1 — Compare BF16 (FP32 vs BF16 accumulation) and per-channel INT8 matmul errors: Measuring both max and mean absolute error across accumulation...
# Import pathlib (Path) for this computation.
from pathlib import Path
import tempfile
import time
import jax
import jax.numpy as jnp
import numpy as np


# Function `evaluate_precision_policies(seed)` implementing this stage's computation:
def evaluate_precision_policies(seed=0):
    # Create or split explicit PRNG key(s) (`(k1, k2)`) for reproducible randomness.
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    # Sample deterministic random values into `x` using an explicit PRNG key.
    x = jax.random.normal(k1, (64, 256), dtype=jnp.float32)
    # Sample deterministic random values into `w` using an explicit PRNG key.
    w = jax.random.normal(k2, (256, 64), dtype=jnp.float32) * 0.25
    # Cast or evaluate `ref_fp32` in explicit floating-point precision.
    ref_fp32 = jnp.dot(x, w, preferred_element_type=jnp.float32)
    # Cast or evaluate `out_bf16_fp32acc` in explicit floating-point precision.
    out_bf16_fp32acc = jnp.dot(
        x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), preferred_element_type=jnp.float32
    )
    # Cast or evaluate `out_bf16_bf16acc` in explicit floating-point precision.
    out_bf16_bf16acc = jnp.dot(
        x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16
    ).astype(jnp.float32)
    # Reduce across the target axis to summarize `scale`.
    scale = jnp.max(jnp.abs(w), axis=0, keepdims=True) / 127.0
    # Combine or mask array elements to form `w_int8`.
    w_int8 = jnp.clip(jnp.round(w / scale), -127, 127).astype(jnp.int8)
    # Cast or evaluate `out_int8_fp32acc` in explicit floating-point precision.
    out_int8_fp32acc = jnp.dot(x, w_int8.astype(jnp.float32) * scale, preferred_element_type=jnp.float32)
    # Evaluate `policies` from the current inputs and state.
    policies = [
        ("bf16 (fp32 accum)", out_bf16_fp32acc),
        ("bf16 (bf16 accum)", out_bf16_bf16acc),
        ("int8 per-col (fp32 accum)", out_int8_fp32acc),
    ]
    # Evaluate `results` from the current inputs and state.
    results = []
    # Loop over `(label, tensor)` in `policies`:
    for label, tensor in policies:
        # Run `jnp.abs` to compute `diff`.
        diff = jnp.abs(tensor - ref_fp32)
        # Reduce across the target axis to summarize ``.
        results.append({
            "policy": label,
            "max_abs_err": round(float(jnp.max(diff)), 5),
            "mean_abs_err": round(float(jnp.mean(diff)), 5),
        })
    # Return `results` to the caller.
    return results

# Capture a warmed XProf trace and verify *.xplane.pb output
# Step 2 — Capture a warmed XProf trace and verify *.xplane.pb output: Warming up before entering jax.profiler.trace ensures the captured...
def capture_warmed_trace(steps=4):
    # Initialize array `x` with explicit values and shape.
    x = jnp.ones((256, 256), dtype=jnp.bfloat16)
    # Wrap with `jax.jit` (`step_fn`) so XLA traces and compiles the function.
    step_fn = jax.jit(lambda a: jnp.dot(a, a, preferred_element_type=jnp.float32).astype(jnp.bfloat16))
    # Record execution timing or profiler trace in `t0`.
    t0 = time.perf_counter()
    # Synchronize host execution until asynchronous device computation completes.
    jax.block_until_ready(step_fn(x))
    # Record execution timing or profiler trace in `warmup_ms`.
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    # Enter managed runtime/context scope for this block:
    with tempfile.TemporaryDirectory(prefix="tpu-course-trace-") as trace_dir:
        # Enter managed runtime/context scope for this block:
        with jax.profiler.trace(trace_dir):
            # Record execution timing or profiler trace in `t1`.
            t1 = time.perf_counter()
            # Repeat the update loop over `range(steps)` steps:
            for _ in range(steps):
                # Run `step_fn` to compute `x`.
                x = step_fn(x)
            # Synchronize host execution until asynchronous device computation completes.
            jax.block_until_ready(x)
            # Record execution timing or profiler trace in `steady_ms`.
            steady_ms = (time.perf_counter() - t1) * 1000.0 / steps
        # Read or serialize artifact data on disk (`xplane_files`).
        xplane_files = list(Path(trace_dir).rglob("*.xplane.pb"))
        # Return `{'warmup_ms': warmup_ms, 'steady_ms': steady_ms, 'xplane_count': len(xplane_files)}` to the caller.
        return {"warmup_ms": warmup_ms, "steady_ms": steady_ms, "xplane_count": len(xplane_files)}


# Run `evaluate_precision_policies` to compute `prec_table`.
prec_table = evaluate_precision_policies()
# Run `capture_warmed_trace` to compute `trace_info`.
trace_info = capture_warmed_trace()
# Verify contract: `trace_info['xplane_count'] >= 1`.
assert trace_info["xplane_count"] >= 1
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert prec_table[0]["max_abs_err"] < prec_table[1]["max_abs_err"]
# Print the observed values to compare against the expected result.
print("Precision comparison:", prec_table)
# Print diagnostic summary of the computed outputs.
print("Captured *.xplane.pb trace count:", trace_info["xplane_count"], "steady_ms:", round(trace_info["steady_ms"], 4))

# Step 1 — Compare BF16 (FP32 vs BF16 accumulation) and per-channel INT8 matmul errors: Measuring both max and mean absolute error across accumulation...
# Import pathlib (Path) for this computation.
from pathlib import Path
import tempfile
import time
import jax
import jax.numpy as jnp
import numpy as np


# Function `evaluate_precision_policies(seed)` implementing this stage's computation:
def evaluate_precision_policies(seed=0):
    # Create or split explicit PRNG key(s) (`(k1, k2)`) for reproducible randomness.
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    # Sample deterministic random values into `x` using an explicit PRNG key.
    x = jax.random.normal(k1, (64, 256), dtype=jnp.float32)
    # Sample deterministic random values into `w` using an explicit PRNG key.
    w = jax.random.normal(k2, (256, 64), dtype=jnp.float32) * 0.25
    # Cast or evaluate `ref_fp32` in explicit floating-point precision.
    ref_fp32 = jnp.dot(x, w, preferred_element_type=jnp.float32)
    # Cast or evaluate `out_bf16_fp32acc` in explicit floating-point precision.
    out_bf16_fp32acc = jnp.dot(
        x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), preferred_element_type=jnp.float32
    )
    # Cast or evaluate `out_bf16_bf16acc` in explicit floating-point precision.
    out_bf16_bf16acc = jnp.dot(
        x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16
    ).astype(jnp.float32)
    # Reduce across the target axis to summarize `scale`.
    scale = jnp.max(jnp.abs(w), axis=0, keepdims=True) / 127.0
    # Combine or mask array elements to form `w_int8`.
    w_int8 = jnp.clip(jnp.round(w / scale), -127, 127).astype(jnp.int8)
    # Cast or evaluate `out_int8_fp32acc` in explicit floating-point precision.
    out_int8_fp32acc = jnp.dot(x, w_int8.astype(jnp.float32) * scale, preferred_element_type=jnp.float32)
    # Evaluate `policies` from the current inputs and state.
    policies = [
        ("bf16 (fp32 accum)", out_bf16_fp32acc),
        ("bf16 (bf16 accum)", out_bf16_bf16acc),
        ("int8 per-col (fp32 accum)", out_int8_fp32acc),
    ]
    # Evaluate `results` from the current inputs and state.
    results = []
    # Loop over `(label, tensor)` in `policies`:
    for label, tensor in policies:
        # Run `jnp.abs` to compute `diff`.
        diff = jnp.abs(tensor - ref_fp32)
        # Reduce across the target axis to summarize ``.
        results.append({
            "policy": label,
            "max_abs_err": round(float(jnp.max(diff)), 5),
            "mean_abs_err": round(float(jnp.mean(diff)), 5),
        })
    # Return `results` to the caller.
    return results


# Step 2 — Capture a warmed XProf trace and verify *.xplane.pb output: Warming up before entering jax.profiler.trace ensures the captured...
def capture_warmed_trace(steps=4):
    # Initialize array `x` with explicit values and shape.
    x = jnp.ones((256, 256), dtype=jnp.bfloat16)
    # Wrap with `jax.jit` (`step_fn`) so XLA traces and compiles the function.
    step_fn = jax.jit(lambda a: jnp.dot(a, a, preferred_element_type=jnp.float32).astype(jnp.bfloat16))
    # Record execution timing or profiler trace in `t0`.
    t0 = time.perf_counter()
    # Synchronize host execution until asynchronous device computation completes.
    jax.block_until_ready(step_fn(x))
    # Record execution timing or profiler trace in `warmup_ms`.
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    # Enter managed runtime/context scope for this block:
    with tempfile.TemporaryDirectory(prefix="tpu-course-trace-") as trace_dir:
        # Enter managed runtime/context scope for this block:
        with jax.profiler.trace(trace_dir):
            # Record execution timing or profiler trace in `t1`.
            t1 = time.perf_counter()
            # Repeat the update loop over `range(steps)` steps:
            for _ in range(steps):
                # Run `step_fn` to compute `x`.
                x = step_fn(x)
            # Synchronize host execution until asynchronous device computation completes.
            jax.block_until_ready(x)
            # Record execution timing or profiler trace in `steady_ms`.
            steady_ms = (time.perf_counter() - t1) * 1000.0 / steps
        # Read or serialize artifact data on disk (`xplane_files`).
        xplane_files = list(Path(trace_dir).rglob("*.xplane.pb"))
        # Return `{'warmup_ms': warmup_ms, 'steady_ms': steady_ms, 'xplane_count': len(xplane_files)}` to the caller.
        return {"warmup_ms": warmup_ms, "steady_ms": steady_ms, "xplane_count": len(xplane_files)}


# Run `evaluate_precision_policies` to compute `prec_table`.
prec_table = evaluate_precision_policies()
# Run `capture_warmed_trace` to compute `trace_info`.
trace_info = capture_warmed_trace()
# Verify contract: `trace_info['xplane_count'] >= 1`.
assert trace_info["xplane_count"] >= 1
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert prec_table[0]["max_abs_err"] < prec_table[1]["max_abs_err"]
# Print the observed values to compare against the expected result.
print("Precision comparison:", prec_table)
# Print diagnostic summary of the computed outputs.
print("Captured *.xplane.pb trace count:", trace_info["xplane_count"], "steady_ms:", round(trace_info["steady_ms"], 4))

# Figure data experiment
# Compute figure data for: Maximum and mean absolute matmul error across BF16 and INT8 precision policies
# Perform matrix contraction / projection to compute `visual_data`.
visual_data = {
    'kind': 'bar',
    'labels': [r['policy'] for r in prec_table],
    'xlabel': 'matmul precision policy',
    'ylabel': 'absolute error vs fp32 reference',
    'series': [
        {'label': 'max abs error', 'y': [float(r['max_abs_err']) for r in prec_table]},
        {'label': 'mean abs error', 'y': [float(r['mean_abs_err']) for r in prec_table]},
    ],
}

# Experiment: Compare FP32 accumulation vs BF16 accumulation error ratio
# Experiment — Compare FP32 accumulation vs BF16 accumulation error ratio: Passing preferred_element_type=jnp.float32 to jnp.dot or...
fp32_acc_err = prec_table[0]["max_abs_err"]
# Evaluate `bf16_acc_err` from the current inputs and state.
bf16_acc_err = prec_table[1]["max_abs_err"]
# Run `round` to compute `ratio`.
ratio = round(bf16_acc_err / fp32_acc_err, 2)
# Print the observed values to compare against the expected result.
print("BF16-accum / FP32-accum max error ratio:", ratio)
# Verify contract: `ratio > 1.3`.
assert ratio > 1.3

# Experiment: Compare warmup duration vs traced steady-state step duration
# Experiment — Compare warmup duration vs traced steady-state step duration: Keeping warmup outside jax.profiler.trace prevents XLA...
# Print the observed values to compare against the expected result.
print("Warmup ms:", round(trace_info["warmup_ms"], 3), "Traced steady step ms:", round(trace_info["steady_ms"], 4))
# Verify contract: `trace_info['warmup_ms'] > trace_info['steady_ms']`.
assert trace_info["warmup_ms"] > trace_info["steady_ms"]

# Reference solution. Try the exercise before reading this.
# Exercise solution: Run evaluate_precision_policies(seed=42) and...
p42 = evaluate_precision_policies(seed=42)
# Run `capture_warmed_trace` to compute `t3`.
t3 = capture_warmed_trace(steps=3)
# Print the observed values to compare against the expected result.
print("Seed 42 mean errors:", {r["policy"]: r["mean_abs_err"] for r in p42}, "xplane files:", t3["xplane_count"])
# Verify contract: `p42[0]['mean_abs_err'] < p42[1]['mean_abs_err'] and t3['xplane_count...`.
assert p42[0]["mean_abs_err"] < p42[1]["mean_abs_err"] and t3["xplane_count"] >= 1

# Reference practice: Compare per-channel vs per-tensor INT8 quantization error
# Compare per-channel vs per-tensor INT8 quantization error (Foundations): One outlier channel stretches a per-tensor scale factor and...
# Create or split explicit PRNG key(s) (`w_skew`) for reproducible randomness.
w_skew = jax.random.normal(jax.random.PRNGKey(9), (128, 32), dtype=jnp.float32) * 0.2
# Evaluate `w_skew` from the current inputs and state.
w_skew = w_skew.at[:, 0].multiply(10.0)
# Create or split explicit PRNG key(s) (`x_in`) for reproducible randomness.
x_in = jax.random.normal(jax.random.PRNGKey(10), (32, 128), dtype=jnp.float32)
# Cast or evaluate `ref` in explicit floating-point precision.
ref = jnp.dot(x_in, w_skew, preferred_element_type=jnp.float32)
# Reduce along axis=0 to compute `s_col`.
s_col = jnp.max(jnp.abs(w_skew), axis=0, keepdims=True) / 127.0
# Aggregate array values to compute `s_glb`.
s_glb = jnp.max(jnp.abs(w_skew)) / 127.0
# Combine or mask array elements to form `q_col`.
q_col = jnp.clip(jnp.round(w_skew / s_col), -127, 127).astype(jnp.int8)
# Combine or mask array elements to form `q_glb`.
q_glb = jnp.clip(jnp.round(w_skew / s_glb), -127, 127).astype(jnp.int8)
# Aggregate array values to compute `err_col`.
err_col = float(jnp.mean(jnp.abs(jnp.dot(x_in, q_col.astype(jnp.float32) * s_col) - ref)))
# Aggregate array values to compute `err_glb`.
err_glb = float(jnp.mean(jnp.abs(jnp.dot(x_in, q_glb.astype(jnp.float32) * s_glb) - ref)))
# Print diagnostic summary of the computed outputs.
print("Mean error (per-channel vs per-tensor):", round(err_col, 5), round(err_glb, 5))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert err_col < err_glb

# Reference practice: Verify non-empty XProf *.xplane.pb byte size
# Verify non-empty XProf *.xplane.pb byte size (Transfer / diagnosis): Verifying a non-empty *.xplane.pb file confirms the profiler...
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix="tpu-xplane-size-") as d:
    # Initialize array `a` with explicit values and shape.
    a = jnp.ones((128, 128), dtype=jnp.bfloat16)
    # Wrap with `jax.jit` (`fn`) so XLA traces and compiles the function.
    fn = jax.jit(lambda z: z @ z)
    # Synchronize host execution until asynchronous device computation completes.
    jax.block_until_ready(fn(a))
    # Enter managed runtime/context scope for this block:
    with jax.profiler.trace(d):
        # Synchronize host execution until asynchronous device computation completes.
        jax.block_until_ready(fn(a))
    # Read or serialize artifact data on disk (`pb`).
    pb = list(Path(d).rglob("*.xplane.pb"))[0]
    # Run `pb.stat` to compute `size_bytes`.
    size_bytes = pb.stat().st_size
# Print the observed values to compare against the expected result.
print("xplane.pb bytes:", size_bytes)
# Verify contract: `size_bytes > 0`.
assert size_bytes > 0
print("PASS: tpu-06")
