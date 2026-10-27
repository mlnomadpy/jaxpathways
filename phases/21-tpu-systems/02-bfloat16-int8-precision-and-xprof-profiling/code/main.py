"""bfloat16, int8 precision, and XProf profiling on TPU: worked experiments and reference solutions. CPU checks."""

# Compare BF16 operands with requested FP32 or BF16 dot result dtypes, plus per-column INT8 reference error
# Step 1 — Compare BF16 operands with requested FP32 or BF16 dot result dtypes, plus per-column INT8 reference error
from pathlib import Path
import os
import tempfile
import tempfile
import time
import jax
import jax.numpy as jnp
import numpy as np


def evaluate_precision_policies(seed=0):
    # Create or split explicit PRNG key(s) (`(k1, k2)`) for reproducible randomness.
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    # Sample deterministic random values into `x` using an explicit PRNG key.
    x = jax.random.normal(k1, (64, 256), dtype=jnp.float32)
    # Sample deterministic random values into `w` using an explicit PRNG key.
    w = jax.random.normal(k2, (256, 64), dtype=jnp.float32) * 0.25
    # Cast or evaluate `ref_fp32` in explicit floating-point precision.
    ref_fp32 = jnp.dot(x, w, preferred_element_type=jnp.float32)
    # Cast or evaluate `out_bf16_requested_fp32` in explicit floating-point precision.
    out_bf16_requested_fp32 = jnp.dot(
        x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), preferred_element_type=jnp.float32
    )
    # Cast or evaluate `out_bf16_requested_bf16` in explicit floating-point precision.
    out_bf16_requested_bf16 = jnp.dot(
        x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16
    )
    # Reduce across the target axis to summarize `scale`.
    column_max = jnp.max(jnp.abs(w), axis=0, keepdims=True)
    scale = jnp.where(column_max == 0, 1.0, column_max / 127.0)
    # The scale of 1 for a zero column avoids division by zero; its quantized values remain zero.
    w_int8 = jnp.clip(jnp.round(w / scale), -127, 127).astype(jnp.int8)
    # Cast or evaluate `out_int8_dequant_requested_fp32` in explicit floating-point precision.
    out_int8_dequant_requested_fp32 = jnp.dot(x, w_int8.astype(jnp.float32) * scale, preferred_element_type=jnp.float32)
    result_dtype_cases = [
        ("BF16 operands / requested FP32 result", out_bf16_requested_fp32),
        ("BF16 operands / requested BF16 result", out_bf16_requested_bf16),
        ("INT8 per-column / FP32 dequantized dot result", out_int8_dequant_requested_fp32),
    ]
    results = []
    # Loop over each requested-result-dtype comparison:
    for label, tensor in result_dtype_cases:
        diff = jnp.abs(tensor - ref_fp32)
        # Append the computed value via `results.append({`.
        results.append({
            "result_dtype_setting": label,
            "returned_dtype": str(tensor.dtype),
            "max_abs_err": round(float(jnp.max(diff)), 5),
            "mean_abs_err": round(float(jnp.mean(diff)), 5),
            "input_shape": list(x.shape), "weight_shape": list(w.shape), "result_shape": list(tensor.shape),
            "stored_weight_bytes": int((w.astype(jnp.bfloat16).size * 2) if "BF16" in label else w_int8.size),
            "scale_metadata_bytes": int(0 if "BF16" in label else scale.size * 4),
            "runtime_input_bytes": int(x.size * (2 if "BF16" in label else 4)),
            "runtime_weight_operand_bytes": int(w.astype(jnp.bfloat16).size * 2 if "BF16" in label else w.size * 4),
            "dequantized_fp32_workspace_bytes": int(w.size * 4 if "INT8" in label else 0),
            "result_bytes": int(tensor.size * (2 if tensor.dtype == jnp.bfloat16 else 4)),
            "storage_note": "logical array sizes; not peak HBM or measured traffic",
        })
    return results


# Capture a bounded fixed-input XProf trace and measure unprofiled work
# Step 2 — Capture a warmed XProf trace and verify *.xplane.pb output
def capture_warmed_trace(steps=5, trace_dir=None):
    if not isinstance(steps, int) or not 1 <= steps <= 16:
        raise ValueError("steps must be an integer in [1, 16]")
    # Reuse the same bounded input each iteration. For this 128x128 matrix of 0.125,
    # every dot entry is 128 * 0.125 * 0.125 = 2.0, so repeated calls cannot grow the values.
    x = jnp.full((128, 128), 0.125, dtype=jnp.bfloat16)
    step_fn = jax.jit(lambda a: jnp.dot(a, a, preferred_element_type=jnp.float32).astype(jnp.bfloat16))

    warm = jax.block_until_ready(step_fn(x))
    if not bool(np.isfinite(np.asarray(warm)).all()):
        raise FloatingPointError("warmup output contains a non-finite value")

    trace_dir = Path(trace_dir or os.environ.get("TPU_TRACE_DIR", "./tpu-xprof-trace")).expanduser().resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_dir)):
        for _ in range(steps):
            profiled_result = jax.block_until_ready(step_fn(x))
            if not bool(np.isfinite(np.asarray(profiled_result)).all()):
                raise FloatingPointError("profiled output contains a non-finite value")

    # Measure the same fixed-signature work outside the profiler. This includes host dispatch
    # and synchronization, and is separate from trace duration or profiler event durations.
    t0 = time.perf_counter()
    for _ in range(steps):
        unprofiled_result = jax.block_until_ready(step_fn(x))
    unprofiled_ms = (time.perf_counter() - t0) * 1000.0
    if not bool(np.isfinite(np.asarray(unprofiled_result)).all()):
        raise FloatingPointError("unprofiled output contains a non-finite value")

    xplane_files = list(trace_dir.rglob("*.xplane.pb"))
    return {
        "backend": jax.default_backend(), "jax_version": jax.__version__, "input_shape": list(x.shape), "steps": steps,
        "warmup_finite": True, "all_finite": True, "profiled_xplane_count": len(xplane_files),
        "trace_dir": str(trace_dir), "unprofiled_total_ms": unprofiled_ms,
        "unprofiled_ms_per_step": unprofiled_ms / steps,
        "measurement_note": "unprofiled wall time includes host dispatch and synchronization; not a device-only duration",
        "workload_note": "fixed-input BF16 dot; no collective communication in this function",
    }



prec_table = evaluate_precision_policies()
trace_info = capture_warmed_trace(steps=5)
# Assert invariant `trace_info["profiled_xplane_count"] >= 1` holds
assert trace_info["profiled_xplane_count"] >= 1
assert prec_table[0]["returned_dtype"] == "float32"
assert prec_table[1]["returned_dtype"] == "bfloat16"
assert all(np.isfinite(row["max_abs_err"]) and np.isfinite(row["mean_abs_err"]) for row in prec_table)
print("Precision comparison:", prec_table)
print("Observed backend:", trace_info["backend"], "profiled steps:", trace_info["steps"], "XPlane files:", trace_info["profiled_xplane_count"], "unprofiled ms/step:", round(trace_info["unprofiled_ms_per_step"], 4))

# Step 3: Verify invariants on the completed state
assert trace_info["profiled_xplane_count"] >= 1
assert prec_table[0]["returned_dtype"] == "float32"
assert prec_table[1]["returned_dtype"] == "bfloat16"
assert all(np.isfinite(row["max_abs_err"]) and np.isfinite(row["mean_abs_err"]) for row in prec_table)

# Step 1 — Compare BF16 operands with requested FP32 or BF16 dot result dtypes, plus per-column INT8 reference error
from pathlib import Path
import os
import tempfile
import time
import jax
import jax.numpy as jnp
import numpy as np


def evaluate_precision_policies(seed=0):
    # Create or split explicit PRNG key(s) (`(k1, k2)`) for reproducible randomness.
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    # Sample deterministic random values into `x` using an explicit PRNG key.
    x = jax.random.normal(k1, (64, 256), dtype=jnp.float32)
    # Sample deterministic random values into `w` using an explicit PRNG key.
    w = jax.random.normal(k2, (256, 64), dtype=jnp.float32) * 0.25
    # Keep the reference matrix product in float32 for comparison.
    ref_fp32 = jnp.dot(x, w, preferred_element_type=jnp.float32)
    # Round both operands to BF16, then request a float32 result dtype from the dot.
    out_bf16_requested_fp32 = jnp.dot(
        x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), preferred_element_type=jnp.float32
    )
    # Request a BF16 dot result, then widen returned values to FP32 for error measurement.
    out_bf16_requested_bf16 = jnp.dot(
        x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16
    )
    # Take one maximum per output column so each column gets its own quantization scale.
    column_max = jnp.max(jnp.abs(w), axis=0, keepdims=True)
    scale = jnp.where(column_max == 0, 1.0, column_max / 127.0)
    # The scale of 1 for a zero column avoids division by zero; its quantized values remain zero.
    w_int8 = jnp.clip(jnp.round(w / scale), -127, 127).astype(jnp.int8)
    # Dequantize int8 weights to FP32, then request an FP32 result from the dot.
    out_int8_dequant_requested_fp32 = jnp.dot(x, w_int8.astype(jnp.float32) * scale, preferred_element_type=jnp.float32)
    result_dtype_cases = [
        ("BF16 operands / requested FP32 result", out_bf16_requested_fp32),
        ("BF16 operands / requested BF16 result", out_bf16_requested_bf16),
        ("INT8 per-column / FP32 dequantized dot result", out_int8_dequant_requested_fp32),
    ]
    results = []
    # Loop over each requested-result-dtype comparison:
    for label, tensor in result_dtype_cases:
        diff = jnp.abs(tensor - ref_fp32)
        # Append the computed value via `results.append({`.
        results.append({
            "result_dtype_setting": label,
            "returned_dtype": str(tensor.dtype),
            "max_abs_err": round(float(jnp.max(diff)), 5),
            "mean_abs_err": round(float(jnp.mean(diff)), 5),
            "input_shape": list(x.shape), "weight_shape": list(w.shape), "result_shape": list(tensor.shape),
            "stored_weight_bytes": int((w.astype(jnp.bfloat16).size * 2) if "BF16" in label else w_int8.size),
            "scale_metadata_bytes": int(0 if "BF16" in label else scale.size * 4),
            "runtime_input_bytes": int(x.size * (2 if "BF16" in label else 4)),
            "runtime_weight_operand_bytes": int(w.astype(jnp.bfloat16).size * 2 if "BF16" in label else w.size * 4),
            "dequantized_fp32_workspace_bytes": int(w.size * 4 if "INT8" in label else 0),
            "result_bytes": int(tensor.size * (2 if tensor.dtype == jnp.bfloat16 else 4)),
            "storage_note": "logical array sizes; not peak HBM or measured traffic",
        })
    return results



# Step 2 — Capture a warmed XProf trace and verify *.xplane.pb output
def capture_warmed_trace(steps=5, trace_dir=None):
    if not isinstance(steps, int) or not 1 <= steps <= 16:
        raise ValueError("steps must be an integer in [1, 16]")
    # Reuse the same bounded input each iteration. For this 128x128 matrix of 0.125,
    # every dot entry is 128 * 0.125 * 0.125 = 2.0, so repeated calls cannot grow the values.
    x = jnp.full((128, 128), 0.125, dtype=jnp.bfloat16)
    step_fn = jax.jit(lambda a: jnp.dot(a, a, preferred_element_type=jnp.float32).astype(jnp.bfloat16))

    warm = jax.block_until_ready(step_fn(x))
    if not bool(np.isfinite(np.asarray(warm)).all()):
        raise FloatingPointError("warmup output contains a non-finite value")

    trace_dir = Path(trace_dir or os.environ.get("TPU_TRACE_DIR", "./tpu-xprof-trace")).expanduser().resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_dir)):
        for _ in range(steps):
            profiled_result = jax.block_until_ready(step_fn(x))
            if not bool(np.isfinite(np.asarray(profiled_result)).all()):
                raise FloatingPointError("profiled output contains a non-finite value")

    # Measure the same fixed-signature work outside the profiler. This includes host dispatch
    # and synchronization, and is separate from trace duration or profiler event durations.
    t0 = time.perf_counter()
    for _ in range(steps):
        unprofiled_result = jax.block_until_ready(step_fn(x))
    unprofiled_ms = (time.perf_counter() - t0) * 1000.0
    if not bool(np.isfinite(np.asarray(unprofiled_result)).all()):
        raise FloatingPointError("unprofiled output contains a non-finite value")

    xplane_files = list(trace_dir.rglob("*.xplane.pb"))
    return {
        "backend": jax.default_backend(), "jax_version": jax.__version__, "input_shape": list(x.shape), "steps": steps,
        "warmup_finite": True, "all_finite": True, "profiled_xplane_count": len(xplane_files),
        "trace_dir": str(trace_dir), "unprofiled_total_ms": unprofiled_ms,
        "unprofiled_ms_per_step": unprofiled_ms / steps,
        "measurement_note": "unprofiled wall time includes host dispatch and synchronization; not a device-only duration",
        "workload_note": "fixed-input BF16 dot; no collective communication in this function",
    }



prec_table = evaluate_precision_policies()
trace_info = capture_warmed_trace(steps=5)
# Assert invariant `trace_info["profiled_xplane_count"] >= 1` holds
assert trace_info["profiled_xplane_count"] >= 1
assert prec_table[0]["returned_dtype"] == "float32"
assert prec_table[1]["returned_dtype"] == "bfloat16"
assert all(np.isfinite(row["max_abs_err"]) and np.isfinite(row["mean_abs_err"]) for row in prec_table)
print("Precision comparison:", prec_table)
print("Retained trace directory:", trace_info["trace_dir"])
print("Observed backend:", trace_info["backend"], "profiled steps:", trace_info["steps"], "XPlane files:", trace_info["profiled_xplane_count"], "unprofiled ms/step:", round(trace_info["unprofiled_ms_per_step"], 4))

# Figure data experiment
# Perform matrix contraction / projection to compute `visual_data`.
visual_data = {
    'kind': 'bar',
    'labels': ['BF16 inputs / FP32 result', 'BF16 inputs / BF16 result', 'INT8 weights / FP32 dequantized dot'],
    'xlabel': 'requested result dtype / operand path',
    'ylabel': 'absolute error vs fp32 reference',
    'series': [
        {'label': 'max abs error', 'y': [float(r['max_abs_err']) for r in prec_table]},
        {'label': 'mean abs error', 'y': [float(r['mean_abs_err']) for r in prec_table]},
    ],
}

# Experiment: Compare requested FP32 vs BF16 result dtype error ratio
# Experiment — Compare requested FP32 vs BF16 result dtype error ratio
requested_fp32_err = prec_table[0]["max_abs_err"]
requested_bf16_err = prec_table[1]["max_abs_err"]
ratio = round(requested_bf16_err / requested_fp32_err, 2)
print("Requested BF16 / FP32 result max-error ratio for seed 0:", ratio)
# This reports the measured ratio for the current backend and fixed seed-zero fixture.
assert np.isfinite(ratio)

# Experiment: Check finite output under a changed bounded trace count
import tempfile
with tempfile.TemporaryDirectory(prefix="tpu-trace-steps-") as tmp:
    five = capture_warmed_trace(steps=5, trace_dir=tmp)
print("Observed backend:", five["backend"], "steps:", five["steps"], "finite:", five["all_finite"], "XPlane files:", five["profiled_xplane_count"])
assert five["all_finite"] and five["steps"] == 5 and five["profiled_xplane_count"] >= 1
try:
    capture_warmed_trace(steps=17, trace_dir=tmp)
except ValueError:
    print("Rejected steps above the fixed maximum before tracing")
else:
    raise AssertionError("trace step count must be bounded")

# Experiment: Check shape-dependent byte accounting and a zero output column
# Shape ledger: BF16 weights are 128*32*2 B; INT8 codes are 128*32 B and 32 FP32 scales.
shape = (128, 32)
bf16_weight_bytes = shape[0] * shape[1] * 2
int8_code_bytes = shape[0] * shape[1]
scale_bytes = shape[1] * 4
zero_column = jnp.zeros((shape[0], 1), dtype=jnp.float32)
zero_scale = jnp.where(jnp.max(jnp.abs(zero_column), axis=0, keepdims=True) == 0, 1.0, jnp.max(jnp.abs(zero_column), axis=0, keepdims=True) / 127.0)
zero_codes = jnp.clip(jnp.round(zero_column / zero_scale), -127, 127).astype(jnp.int8)
print({"bf16_weight_bytes": bf16_weight_bytes, "int8_code_bytes": int8_code_bytes, "scale_bytes": scale_bytes, "zero_column_scale": float(zero_scale[0,0]), "zero_codes": int(jnp.count_nonzero(zero_codes))})
assert (bf16_weight_bytes, int8_code_bytes, scale_bytes) == (8192, 4096, 128)
assert float(zero_scale[0,0]) == 1.0 and int(jnp.count_nonzero(zero_codes)) == 0

# Reference solution. Try the exercise before reading this.
p42 = evaluate_precision_policies(seed=42)
t3 = capture_warmed_trace(steps=3)
print("Observed backend and JAX version:", t3["backend"], t3["jax_version"])
print("Three returned dtypes:", [r["returned_dtype"] for r in p42])
print("Three mean absolute errors:", [r["mean_abs_err"] for r in p42])
print("Trace steps/files:", t3["steps"], t3["profiled_xplane_count"], "unprofiled ms/step:", t3["unprofiled_ms_per_step"])
assert t3["warmup_finite"] and t3["all_finite"] and t3["profiled_xplane_count"] >= 1
assert [r["returned_dtype"] for r in p42] == ["float32", "bfloat16", "float32"]
# Changed-step-count check: five repeats remain finite because input is not fed back as output.
t5 = capture_warmed_trace(steps=5)
assert t5["all_finite"] and t5["steps"] == 5

# Reference practice: Compare per-channel vs per-tensor INT8 quantization error
# Compare per-channel vs per-tensor INT8 quantization error (Foundations)
# Create or split explicit PRNG key(s) (`w_skew`) for reproducible randomness.
w_skew = jax.random.normal(jax.random.PRNGKey(9), (128, 32), dtype=jnp.float32) * 0.2
w_skew = w_skew.at[:, 0].multiply(10.0)
# Create or split explicit PRNG key(s) (`x_in`) for reproducible randomness.
x_in = jax.random.normal(jax.random.PRNGKey(10), (32, 128), dtype=jnp.float32)
# Cast or evaluate `ref` in explicit floating-point precision.
ref = jnp.dot(x_in, w_skew, preferred_element_type=jnp.float32)
# Reduce along axis=0 to compute `s_col`.
s_col = jnp.max(jnp.abs(w_skew), axis=0, keepdims=True) / 127.0
s_glb = jnp.max(jnp.abs(w_skew)) / 127.0
q_col = jnp.clip(jnp.round(w_skew / s_col), -127, 127).astype(jnp.int8)
q_glb = jnp.clip(jnp.round(w_skew / s_glb), -127, 127).astype(jnp.int8)
err_col = float(jnp.mean(jnp.abs(jnp.dot(x_in, q_col.astype(jnp.float32) * s_col) - ref)))
err_glb = float(jnp.mean(jnp.abs(jnp.dot(x_in, q_glb.astype(jnp.float32) * s_glb) - ref)))
print("Mean error (per-channel vs per-tensor):", round(err_col, 5), round(err_glb, 5))
# Assert invariant `err_col < err_glb` holds
assert err_col < err_glb

# Reference practice: Verify non-empty XProf *.xplane.pb byte size
# Verify non-empty XProf *.xplane.pb byte size (Transfer / diagnosis)
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory(prefix="tpu-xplane-size-") as d:
    a = jnp.ones((128, 128), dtype=jnp.bfloat16)
    # Wrap with `jax.jit` (`fn`) so XLA traces and compiles the function.
    fn = jax.jit(lambda z: z @ z)
    # Wait for device work to finish before timing it or comparing its result on the host.
    jax.block_until_ready(fn(a))
    # Enter managed runtime/context scope for this block:
    with jax.profiler.trace(d):
        # Wait for device work to finish before timing it or comparing its result on the host.
        jax.block_until_ready(fn(a))
    pb = list(Path(d).rglob("*.xplane.pb"))[0]
    size_bytes = pb.stat().st_size
print("xplane.pb bytes:", size_bytes)
# Assert invariant `size_bytes > 0` holds
assert size_bytes > 0
print("PASS: tpu-06")
