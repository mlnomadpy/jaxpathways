"""Benchmark asynchronous work correctly: worked experiments and reference solutions. CPU checks."""

# Prepare and place the workload
# Step 1 — Prepare and place the workload: CPU placement and waiting occur before timing.
# Import json for this computation.
import json
import platform
import time
import statistics
import numpy as np
import jax
import jax.numpy as jnp
# Query the active JAX devices into `cpu`.
cpu = jax.devices("cpu")[0]
# Draw pseudorandom samples for `rng` using the explicit RNG state.
rng = np.random.default_rng(4)
# Cast or evaluate `x_host` in explicit floating-point precision.
x_host = rng.normal(size=(64, 32)).astype(np.float32)
# Cast or evaluate `w_host` in explicit floating-point precision.
w_host = rng.normal(size=(32, 16)).astype(np.float32)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(x_host, cpu)
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(w_host, cpu)
# Synchronize host execution until asynchronous device computation completes.
jax.block_until_ready((x, w))

# Define the contract and timer
# Step 2 — Define the contract and timer: The timer accepts arbitrary result pytrees because...
def predict(x, w):
    # Return `jnp.tanh(x @ w)` to the caller.
    return jnp.tanh(x @ w)
# Wrap with `jax.jit` (`compiled_predict`) so XLA traces and compiles the function.
compiled_predict = jax.jit(predict)
# Function `synchronized_samples(fn, args, repeats)` implementing this stage's computation:
def synchronized_samples(fn, args, repeats=7):
    # Guard input contract (`repeats < 1`) and fail fast if violated.
    if repeats < 1:
        raise ValueError("at least one repeat is required")
    # Evaluate `samples` from the current inputs and state.
    samples = []
    # Repeat the update loop over `range(repeats)` steps:
    for _ in range(repeats):
        # Record execution timing or profiler trace in `start`.
        start = time.perf_counter()
        # Run `fn` to compute `result`.
        result = fn(*args)
        # Synchronize host execution until asynchronous device computation completes.
        jax.block_until_ready(result)
        # Record execution timing or profiler trace in ``.
        samples.append(time.perf_counter() - start)
    # Return `(result, samples)` to the caller.
    return result, samples

# Separate first call from warm calls
# Step 3 — Separate first call from warm calls: The report prints variable measured times.
start = time.perf_counter()
# Run `compiled_predict` to compute `first`.
first = compiled_predict(x, w)
# Synchronize host execution until asynchronous device computation completes.
first.block_until_ready()
# Record execution timing or profiler trace in `first_call_seconds`.
first_call_seconds = time.perf_counter() - start
# Run `synchronized_samples` to compute `(result, samples)`.
result, samples = synchronized_samples(compiled_predict, (x, w))
# Perform matrix contraction / projection to compute `reference`.
reference = np.tanh(x_host @ w_host)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(result), reference, atol=2e-5, rtol=2e-5)
# Aggregate array values to compute `report`.
report = {
    "backend": str(cpu), "jax": jax.__version__, "numpy": np.__version__,
    "python": platform.python_version(), "input_shape": list(x.shape),
    "weight_shape": list(w.shape), "dtype": str(x.dtype),
    "boundary": "input already placed; call plus output synchronization",
    "first_call_seconds": first_call_seconds,
    "warm_samples_seconds": samples, "warm_median_seconds": statistics.median(samples),
    "max_abs_error": float(np.max(np.abs(np.asarray(result) - reference))),
}
# Verify contract: `len(samples) == 7 and all((t >= 0 for t in samples))`.
assert len(samples) == 7 and all(t >= 0 for t in samples)
# Print the observed values to compare against the expected result.
print(json.dumps(report, indent=2))

# Step 1 — Prepare and place the workload: CPU placement and waiting occur before timing.
# Import json for this computation.
import json
import platform
import time
import statistics
import numpy as np
import jax
import jax.numpy as jnp
# Query the active JAX devices into `cpu`.
cpu = jax.devices("cpu")[0]
# Draw pseudorandom samples for `rng` using the explicit RNG state.
rng = np.random.default_rng(4)
# Cast or evaluate `x_host` in explicit floating-point precision.
x_host = rng.normal(size=(64, 32)).astype(np.float32)
# Cast or evaluate `w_host` in explicit floating-point precision.
w_host = rng.normal(size=(32, 16)).astype(np.float32)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(x_host, cpu)
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(w_host, cpu)
# Synchronize host execution until asynchronous device computation completes.
jax.block_until_ready((x, w))
# Step 2 — Define the contract and timer: The timer accepts arbitrary result pytrees because...
def predict(x, w):
    # Return `jnp.tanh(x @ w)` to the caller.
    return jnp.tanh(x @ w)
# Wrap with `jax.jit` (`compiled_predict`) so XLA traces and compiles the function.
compiled_predict = jax.jit(predict)
# Function `synchronized_samples(fn, args, repeats)` implementing this stage's computation:
def synchronized_samples(fn, args, repeats=7):
    # Guard input contract (`repeats < 1`) and fail fast if violated.
    if repeats < 1:
        raise ValueError("at least one repeat is required")
    # Evaluate `samples` from the current inputs and state.
    samples = []
    # Repeat the update loop over `range(repeats)` steps:
    for _ in range(repeats):
        # Record execution timing or profiler trace in `start`.
        start = time.perf_counter()
        # Run `fn` to compute `result`.
        result = fn(*args)
        # Synchronize host execution until asynchronous device computation completes.
        jax.block_until_ready(result)
        # Record execution timing or profiler trace in ``.
        samples.append(time.perf_counter() - start)
    # Return `(result, samples)` to the caller.
    return result, samples
# Step 3 — Separate first call from warm calls: The report prints variable measured times.
start = time.perf_counter()
# Run `compiled_predict` to compute `first`.
first = compiled_predict(x, w)
# Synchronize host execution until asynchronous device computation completes.
first.block_until_ready()
# Record execution timing or profiler trace in `first_call_seconds`.
first_call_seconds = time.perf_counter() - start
# Run `synchronized_samples` to compute `(result, samples)`.
result, samples = synchronized_samples(compiled_predict, (x, w))
# Perform matrix contraction / projection to compute `reference`.
reference = np.tanh(x_host @ w_host)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(result), reference, atol=2e-5, rtol=2e-5)
# Aggregate array values to compute `report`.
report = {
    "backend": str(cpu), "jax": jax.__version__, "numpy": np.__version__,
    "python": platform.python_version(), "input_shape": list(x.shape),
    "weight_shape": list(w.shape), "dtype": str(x.dtype),
    "boundary": "input already placed; call plus output synchronization",
    "first_call_seconds": first_call_seconds,
    "warm_samples_seconds": samples, "warm_median_seconds": statistics.median(samples),
    "max_abs_error": float(np.max(np.abs(np.asarray(result) - reference))),
}
# Verify contract: `len(samples) == 7 and all((t >= 0 for t in samples))`.
assert len(samples) == 7 and all(t >= 0 for t in samples)
# Print the observed values to compare against the expected result.
print(json.dumps(report, indent=2))

# Figure data experiment
# Compute figure data for: Separate the first call from warm measurements
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'panels', 'panels': [{'kind': 'bar', 'title': 'First call', 'labels': ['first'], 'ylabel': 'seconds', 'series': [{'label': 'synchronized call', 'y': [first_call_seconds]}]}, {'kind': 'line', 'title': 'Warm calls', 'x': list(range(1, len(samples) + 1)), 'xlabel': 'warm sample', 'ylabel': 'seconds', 'series': [{'label': 'synchronized call', 'y': samples}]}]}

# Experiment: Compare handle-only and synchronized boundaries
# Experiment — Compare handle-only and synchronized boundaries: The separate intervals have different measurement boundaries and...
start = time.perf_counter()
# Run `compiled_predict` to compute `handle`.
handle = compiled_predict(x, w)
# Record execution timing or profiler trace in `handle_seconds`.
handle_seconds = time.perf_counter() - start
handle.block_until_ready()  # drain this call before the separate comparison
# Run `synchronized_samples` to compute `(_, complete_samples)`.
_, complete_samples = synchronized_samples(compiled_predict, (x, w), repeats=3)
# Print the observed values to compare against the expected result.
print("Handle-only seconds:", handle_seconds)
# Print diagnostic summary of the computed outputs.
print("Complete-call samples:", complete_samples)

# Experiment: Return structured results without host copies
# Experiment — Return structured results without host copies: A tree-level wait makes the benchmark contract explicit for...
# Wrap with `jax.jit` (`structured`) so XLA traces and compiles the function.
structured = jax.jit(lambda a, b: {"prediction": jnp.tanh(a @ b), "mean": jnp.mean(jnp.tanh(a @ b))})
# Synchronize host execution until asynchronous device computation completes.
jax.block_until_ready(structured(x, w))
# Run `synchronized_samples` to compute `(structured_result, structured_times)`.
structured_result, structured_times = synchronized_samples(structured, (x, w), 3)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(structured_result["prediction"]), reference, atol=2e-5, rtol=2e-5)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(structured_result["mean"]), reference.mean(), atol=2e-5, rtol=2e-5)
# Print the observed values to compare against the expected result.
print("Structured samples:", structured_times)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Repeat the benchmark with 96 rows while preserving the feature and...
larger_host = rng.normal(size=(96, 32)).astype(np.float32)
# Place `larger` explicitly onto the target JAX device.
larger = jax.device_put(larger_host, cpu)
# Synchronize host execution until asynchronous device computation completes.
larger.block_until_ready()
# Synchronize host execution until asynchronous device computation completes.
compiled_predict(larger, w).block_until_ready()
# Run `synchronized_samples` to compute `(larger_result, larger_times)`.
larger_result, larger_times = synchronized_samples(compiled_predict, (larger, w), 5)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(larger_result), np.tanh(larger_host @ w_host), atol=2e-5, rtol=2e-5)
# Print the observed values to compare against the expected result.
print("Changed workload:", larger.shape, larger_times)

# Reference practice: Benchmark a different equivalent workload
# Benchmark a different equivalent workload (Transfer): Changing the computation changes the performance question.
# Wrap with `jax.jit` (`energy`) so XLA traces and compiles the function.
energy = jax.jit(lambda a, b: jnp.sum((a @ b)**2, axis=1))
# Synchronize host execution until asynchronous device computation completes.
energy(x, w).block_until_ready()
# Run `synchronized_samples` to compute `(energy_result, energy_times)`.
energy_result, energy_times = synchronized_samples(energy, (x, w), 5)
# Reduce along axis=1 to compute `energy_reference`.
energy_reference = np.sum((x_host @ w_host)**2, axis=1)
# Verify that the output tensor shape matches our prediction.
assert energy_result.shape == (64,)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(energy_result), energy_reference, atol=2e-3, rtol=2e-5)
# Print the observed values to compare against the expected result.
print("Energy shape and samples:", energy_result.shape, energy_times)

# Reference practice: Repair a timer that includes setup but omits completion
# Repair a timer that includes setup but omits completion (Diagnosis): The repaired boundary excludes setup, separates warmup and...
def repaired_benchmark(fn, args):
    # Synchronize host execution until asynchronous device computation completes.
    jax.block_until_ready(args)
    # Synchronize host execution until asynchronous device computation completes.
    jax.block_until_ready(fn(*args))
    # Run `synchronized_samples` to compute `(result, measured)`.
    result, measured = synchronized_samples(fn, args, 5)
    # Return `({'boundary': 'placed inputs; warm call plus output wait', 'samples': measured}, result)` to the caller.
    return {"boundary": "placed inputs; warm call plus output wait", "samples": measured}, result
# Run `repaired_benchmark` to compute `(fixed_report, fixed_result)`.
fixed_report, fixed_result = repaired_benchmark(compiled_predict, (x, w))
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(fixed_result), reference, atol=2e-5, rtol=2e-5)
# Verify contract: `len(fixed_report['samples']) == 5`.
assert len(fixed_report["samples"]) == 5
# Print the observed values to compare against the expected result.
print(fixed_report)
print("PASS: performance-01")
