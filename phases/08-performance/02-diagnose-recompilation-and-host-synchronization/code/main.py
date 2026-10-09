"""Diagnose recompilation and host synchronization: worked experiments and reference solutions. CPU checks."""

# Prepare a trace ledger
# Step 1 — Prepare a trace ledger: This CPU experiment deliberately keeps the trace ledger outside...
# Import json for this computation.
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
# Query the active JAX devices into `cpu`.
cpu = jax.devices("cpu")[0]
# Evaluate `trace_events` from the current inputs and state.
trace_events = []
# Function `score(x, gain)` implementing this stage's computation:
def score(x, gain):
    # Authoring diagnostic: this Python effect happens during tracing.
    # It is NOT model state and does not count executable compilations.
    trace_events.append({"shape": list(x.shape), "dtype": str(x.dtype)})
    # Return `jnp.sum((x * gain) ** 2)` to the caller.
    return jnp.sum((x * gain)**2)
# Wrap with `jax.jit` (`compiled_score`) so XLA traces and compiles the function.
compiled_score = jax.jit(score)
# Initialize array `x8` with explicit values and shape.
x8 = jax.device_put(np.arange(8, dtype=np.float32), cpu)
# Initialize array `x12` with explicit values and shape.
x12 = jax.device_put(np.arange(12, dtype=np.float32), cpu)
# Synchronize host execution until asynchronous device computation completes.
jax.block_until_ready((x8, x12))

# Record call signatures and observed traces
# Step 2 — Record call signatures and observed traces: The ledger relates each controlled signature change to the...
call_rows = []
# Function `observed_call(label, x, gain)` implementing this stage's computation:
def observed_call(label, x, gain):
    # Run `len` to compute `before`.
    before = len(trace_events)
    # Place `result` explicitly onto the target JAX device.
    result = compiled_score(x, jax.device_put(np.float32(gain), cpu))
    # Synchronize host execution until asynchronous device computation completes.
    result.block_until_ready()
    # Convert `expected` to a host NumPy array for inspection or verification.
    expected = np.sum((np.asarray(x) * np.float32(gain))**2)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-5, atol=2e-5)
    # Append the current step result to `call_rows`.
    call_rows.append({"label": label, "shape": list(x.shape), "dtype": str(x.dtype),
                      "observed_trace_delta": len(trace_events) - before,
                      "result": float(result)})
    # Return `result` to the caller.
    return result

# Run the controlled call sequence
# Step 3 — Run the controlled call sequence: Inspect the event deltas on your own environment.
observed_call("initial signature", x8, 1.)
# Run `observed_call` to perform the next check or state transition.
observed_call("same signature changed data", x8 + 1., 1.)
# Run `observed_call` to perform the next check or state transition.
observed_call("same signature changed dynamic gain", x8, 2.)
# Run `observed_call` to perform the next check or state transition.
observed_call("different input shape", x12, 1.)
# Print the observed values to compare against the expected result.
print(json.dumps({"jax": jax.__version__, "backend": str(cpu),
                  "calls": call_rows, "python_trace_events": trace_events}, indent=2))
# Verify contract: `call_rows[0]['result'] == 140.0`.
assert call_rows[0]["result"] == 140.
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert call_rows[2]["result"] == 560.
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert call_rows[3]["result"] == 506.

# Step 1 — Prepare a trace ledger: This CPU experiment deliberately keeps the trace ledger outside...
# Import json for this computation.
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
# Query the active JAX devices into `cpu`.
cpu = jax.devices("cpu")[0]
# Evaluate `trace_events` from the current inputs and state.
trace_events = []
# Function `score(x, gain)` implementing this stage's computation:
def score(x, gain):
    # Authoring diagnostic: this Python effect happens during tracing.
    # It is NOT model state and does not count executable compilations.
    trace_events.append({"shape": list(x.shape), "dtype": str(x.dtype)})
    # Return `jnp.sum((x * gain) ** 2)` to the caller.
    return jnp.sum((x * gain)**2)
# Wrap with `jax.jit` (`compiled_score`) so XLA traces and compiles the function.
compiled_score = jax.jit(score)
# Initialize array `x8` with explicit values and shape.
x8 = jax.device_put(np.arange(8, dtype=np.float32), cpu)
# Initialize array `x12` with explicit values and shape.
x12 = jax.device_put(np.arange(12, dtype=np.float32), cpu)
# Synchronize host execution until asynchronous device computation completes.
jax.block_until_ready((x8, x12))
# Step 2 — Record call signatures and observed traces: The ledger relates each controlled signature change to the...
call_rows = []
# Function `observed_call(label, x, gain)` implementing this stage's computation:
def observed_call(label, x, gain):
    # Run `len` to compute `before`.
    before = len(trace_events)
    # Place `result` explicitly onto the target JAX device.
    result = compiled_score(x, jax.device_put(np.float32(gain), cpu))
    # Synchronize host execution until asynchronous device computation completes.
    result.block_until_ready()
    # Convert `expected` to a host NumPy array for inspection or verification.
    expected = np.sum((np.asarray(x) * np.float32(gain))**2)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-5, atol=2e-5)
    # Append the current step result to `call_rows`.
    call_rows.append({"label": label, "shape": list(x.shape), "dtype": str(x.dtype),
                      "observed_trace_delta": len(trace_events) - before,
                      "result": float(result)})
    # Return `result` to the caller.
    return result
# Step 3 — Run the controlled call sequence: Inspect the event deltas on your own environment.
observed_call("initial signature", x8, 1.)
# Run `observed_call` to perform the next check or state transition.
observed_call("same signature changed data", x8 + 1., 1.)
# Run `observed_call` to perform the next check or state transition.
observed_call("same signature changed dynamic gain", x8, 2.)
# Run `observed_call` to perform the next check or state transition.
observed_call("different input shape", x12, 1.)
# Print the observed values to compare against the expected result.
print(json.dumps({"jax": jax.__version__, "backend": str(cpu),
                  "calls": call_rows, "python_trace_events": trace_events}, indent=2))
# Verify contract: `call_rows[0]['result'] == 140.0`.
assert call_rows[0]["result"] == 140.
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert call_rows[2]["result"] == 560.
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert call_rows[3]["result"] == 506.

# Figure data experiment
# Compute figure data for: A new value is not necessarily a new trace
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'bar', 'labels': ['initial', 'new values', 'new gain', 'new shape'], 'ylabel': 'observed Python trace events', 'series': [{'label': 'trace delta', 'y': [r['observed_trace_delta'] for r in call_rows]}]}

# Experiment: Contrast static branch metadata with a dynamic gain
# Experiment — Contrast static branch metadata with a dynamic gain: The static Boolean controls Python program structure.
branch_events = []
# Function `branch_score(x, square)` implementing this stage's computation:
def branch_score(x, square):
    # Append the current step result to `branch_events`.
    branch_events.append({"shape": list(x.shape), "square": square})
    # Return `jnp.sum(x ** 2) if square else jnp.sum(x)` to the caller.
    return jnp.sum(x**2) if square else jnp.sum(x)
# Wrap with `jax.jit` (`static_score`) so XLA traces and compiles the function.
static_score = jax.jit(branch_score, static_argnames=("square",))
# Iterate over `square` to step through the computation:
for square in [True, True, False]:
    # Run `static_score` to compute `out`.
    out = static_score(x8, square=square)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(out), 140. if square else 28.)
# Print the observed values to compare against the expected result.
print("Static branch trace observations:", branch_events)

# Experiment: Keep a metric on device until the reporting boundary
# Experiment — Keep a metric on device until the reporting boundary: The geometric-series result checks the computation independently.
# Wrap with `jax.jit` (`update`) so XLA traces and compiles the function.
update = jax.jit(lambda value: 0.9 * value + 1.)
# Place `initial` explicitly onto the target JAX device.
initial = jax.device_put(np.float32(0.), cpu)
# Synchronize host execution until asynchronous device computation completes.
update(initial).block_until_ready()
# Function `metric_run(read_each)` implementing this stage's computation:
def metric_run(read_each):
    # Evaluate `value` from the current inputs and state.
    value = initial
    # Evaluate `observed` from the current inputs and state.
    observed = []
    # Record execution timing or profiler trace in `start`.
    start = time.perf_counter()
    # Repeat the update loop over `range(6)` steps:
    for _ in range(6):
        # Run `update` to compute `value`.
        value = update(value)
        # Branch on condition `read_each`:
        if read_each:
            observed.append(float(value))
    # Synchronize host execution until asynchronous device computation completes.
    value.block_until_ready()
    # Record execution timing or profiler trace in `elapsed`.
    elapsed = time.perf_counter() - start
    # Return `(value, elapsed, observed)` to the caller.
    return value, elapsed, observed
# Run `metric_run` to compute `(read_result, read_seconds, metrics)`.
read_result, read_seconds, metrics = metric_run(True)
# Run `metric_run` to compute `(final_result, final_seconds, _)`.
final_result, final_seconds, _ = metric_run(False)
# Evaluate `reference_value` from the current inputs and state.
reference_value = 10. * (1. - 0.9**6)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(read_result), reference_value, rtol=2e-5)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(final_result), reference_value, rtol=2e-5)
# Verify contract: `len(metrics) == 6`.
assert len(metrics) == 6
# Print diagnostic summary of the computed outputs.
print("Per-step host reads seconds:", read_seconds)
# Print diagnostic summary of the computed outputs.
print("Final-only wait seconds:", final_seconds)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Add a call with sixteen float32 elements.
# Initialize array `x16` with explicit values and shape.
x16 = jax.device_put(np.arange(16, dtype=np.float32), cpu)
# Run `observed_call` to perform the next check or state transition.
observed_call("new length sixteen", x16, 1.)
# Verify contract: `call_rows[-1]['result'] == sum((i * i for i in range(16)))`.
assert call_rows[-1]["result"] == sum(i*i for i in range(16))
# Print the observed values to compare against the expected result.
print("Added ledger row:", call_rows[-1])

# Reference practice: Pad a final batch while preserving the objective
# Pad a final batch while preserving the objective (Transfer): The mask removes three synthetic rows.
# Wrap with `jax.jit` (`masked_score`) so XLA traces and compiles the function.
masked_score = jax.jit(lambda values, mask: jnp.sum(jnp.where(mask, (values + 1.)**2, 0.)))
# Initialize array `five` with explicit values and shape.
five = np.arange(5, dtype=np.float32)
# Place `padded` explicitly onto the target JAX device.
padded = jax.device_put(np.pad(five, (0, 3)), cpu)
# Initialize array `mask` with explicit values and shape.
mask = jax.device_put(np.arange(8) < 5, cpu)
# Combine or mask array elements to form `masked`.
masked = masked_score(padded, mask)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(masked), sum((i+1)**2 for i in range(5)))
# Aggregate array values to compute `wrong`.
wrong = jnp.sum((padded + 1.)**2)
# Verify contract: `float(wrong) == float(masked) + 3.0`.
assert float(wrong) == float(masked) + 3.
# Print the observed values to compare against the expected result.
print("Masked versus unmasked padded objective:", float(masked), float(wrong))

# Reference practice: Reproduce and repair runtime Python branching
# Reproduce and repair runtime Python branching (Diagnosis): The failure identifies a mismatch between Python control...
def broken_branch(values, gain):
    # Branch on condition `gain > 1.0`:
    if gain > 1.:
        return jnp.sum(values**2)
    # Return `jnp.sum(values)` to the caller.
    return jnp.sum(values)
# Run the boundary check and catch the expected exception:
try:
    jax.jit(broken_branch)(x8, jax.device_put(np.float32(2.), cpu))
except jax.errors.TracerBoolConversionError:
    print("Observed runtime-Python-branch tracing failure")
else:
    raise AssertionError("runtime branch unexpectedly accepted")
# Wrap with `jax.jit` (`repaired`) so XLA traces and compiles the function.
repaired = jax.jit(lambda values, gain: jax.lax.cond(gain > 1., lambda v:jnp.sum(v**2), lambda v:jnp.sum(v), values))
# Iterate over `(gain, expected)` to step through the computation:
for gain, expected in [(0.5, 28.), (2., 140.)]:
    # Place `result` explicitly onto the target JAX device.
    result = repaired(x8, jax.device_put(np.float32(gain), cpu))
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(result), expected)
print("PASS: performance-02")
