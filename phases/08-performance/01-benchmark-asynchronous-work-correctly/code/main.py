"""Benchmark asynchronous work correctly: worked experiments and reference solutions. CPU checks."""

# Prepare and place the workload
# Step 1 — Prepare and place the workload: CPU placement and waiting occur before timing.
import json
import platform
import time
import statistics
import numpy as np
import jax
import jax.numpy as jnp
cpu = jax.devices("cpu")[0]
rng = np.random.default_rng(4)
x_host = rng.normal(size=(64, 32)).astype(np.float32)
w_host = rng.normal(size=(32, 16)).astype(np.float32)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(x_host, cpu)
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(w_host, cpu)
# Wait for device work to finish before timing it or comparing its result on the host.
jax.block_until_ready((x, w))

# Define the contract and timer
# Step 2 — Define the contract and timer
def predict(x, w):
    return jnp.tanh(x @ w)
# Wrap with `jax.jit` (`compiled_predict`) so XLA traces and compiles the function.
compiled_predict = jax.jit(predict)
def synchronized_samples(fn, args, repeats=7):
    if repeats < 1:
        raise ValueError("at least one repeat is required")
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args)
        # Wait for device work to finish before timing it or comparing its result on the host.
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return result, samples

# Separate first call from warm calls
# Step 3 — Separate first call from warm calls: The report prints variable measured times.
start = time.perf_counter()
first = compiled_predict(x, w)
# Wait for device work to finish before timing it or comparing its result on the host.
first.block_until_ready()
first_call_seconds = time.perf_counter() - start
result, samples = synchronized_samples(compiled_predict, (x, w))
reference = np.tanh(x_host @ w_host)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(result), reference, atol=2e-5, rtol=2e-5)
report = {
    "backend": str(cpu), "jax": jax.__version__, "numpy": np.__version__,
    "python": platform.python_version(), "input_shape": list(x.shape),
    "weight_shape": list(w.shape), "dtype": str(x.dtype),
    "boundary": "input already placed; call plus output synchronization",
    "first_call_seconds": first_call_seconds,
    "warm_samples_seconds": samples, "warm_median_seconds": statistics.median(samples),
    "max_abs_error": float(np.max(np.abs(np.asarray(result) - reference))),
}
assert len(samples) == 7 and all(t >= 0 for t in samples)
print(json.dumps(report, indent=2))

# Step 1 — Prepare and place the workload: CPU placement and waiting occur before timing.
import json
import platform
import time
import statistics
import numpy as np
import jax
import jax.numpy as jnp
cpu = jax.devices("cpu")[0]
rng = np.random.default_rng(4)
x_host = rng.normal(size=(64, 32)).astype(np.float32)
w_host = rng.normal(size=(32, 16)).astype(np.float32)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(x_host, cpu)
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(w_host, cpu)
# Wait for device work to finish before timing it or comparing its result on the host.
jax.block_until_ready((x, w))
# Step 2 — Define the contract and timer
def predict(x, w):
    return jnp.tanh(x @ w)
# Wrap with `jax.jit` (`compiled_predict`) so XLA traces and compiles the function.
compiled_predict = jax.jit(predict)
def synchronized_samples(fn, args, repeats=7):
    if repeats < 1:
        raise ValueError("at least one repeat is required")
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args)
        # Wait for device work to finish before timing it or comparing its result on the host.
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return result, samples
# Step 3 — Separate first call from warm calls: The report prints variable measured times.
start = time.perf_counter()
first = compiled_predict(x, w)
# Wait for device work to finish before timing it or comparing its result on the host.
first.block_until_ready()
first_call_seconds = time.perf_counter() - start
result, samples = synchronized_samples(compiled_predict, (x, w))
reference = np.tanh(x_host @ w_host)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(result), reference, atol=2e-5, rtol=2e-5)
report = {
    "backend": str(cpu), "jax": jax.__version__, "numpy": np.__version__,
    "python": platform.python_version(), "input_shape": list(x.shape),
    "weight_shape": list(w.shape), "dtype": str(x.dtype),
    "boundary": "input already placed; call plus output synchronization",
    "first_call_seconds": first_call_seconds,
    "warm_samples_seconds": samples, "warm_median_seconds": statistics.median(samples),
    "max_abs_error": float(np.max(np.abs(np.asarray(result) - reference))),
}
assert len(samples) == 7 and all(t >= 0 for t in samples)
print(json.dumps(report, indent=2))

# Figure data experiment
visual_data = {'kind': 'panels', 'panels': [{'kind': 'bar', 'title': 'First call', 'labels': ['first'], 'ylabel': 'seconds', 'series': [{'label': 'synchronized call', 'y': [first_call_seconds]}]}, {'kind': 'line', 'title': 'Warm calls', 'x': list(range(1, len(samples) + 1)), 'xlabel': 'warm sample', 'ylabel': 'seconds', 'series': [{'label': 'synchronized call', 'y': samples}]}]}

# Experiment: Compare handle-only and synchronized boundaries
# Experiment — Compare handle-only and synchronized boundaries
start = time.perf_counter()
handle = compiled_predict(x, w)
handle_seconds = time.perf_counter() - start
handle.block_until_ready()  # drain this call before the separate comparison
_, complete_samples = synchronized_samples(compiled_predict, (x, w), repeats=3)
print("Handle-only seconds:", handle_seconds)
print("Complete-call samples:", complete_samples)

# Experiment: Return structured results without host copies
# Experiment — Return structured results without host copies
# Wrap with `jax.jit` (`structured`) so XLA traces and compiles the function.
structured = jax.jit(lambda a, b: {"prediction": jnp.tanh(a @ b), "mean": jnp.mean(jnp.tanh(a @ b))})
# Wait for device work to finish before timing it or comparing its result on the host.
jax.block_until_ready(structured(x, w))
structured_result, structured_times = synchronized_samples(structured, (x, w), 3)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(structured_result["prediction"]), reference, atol=2e-5, rtol=2e-5)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(structured_result["mean"]), reference.mean(), atol=2e-5, rtol=2e-5)
print("Structured samples:", structured_times)

# Reference solution. Try the exercise before reading this.
# Exercise solution
larger_host = rng.normal(size=(96, 32)).astype(np.float32)
# Place `larger` explicitly onto the target JAX device.
larger = jax.device_put(larger_host, cpu)
# Wait for device work to finish before timing it or comparing its result on the host.
larger.block_until_ready()
# Wait for device work to finish before timing it or comparing its result on the host.
compiled_predict(larger, w).block_until_ready()
larger_result, larger_times = synchronized_samples(compiled_predict, (larger, w), 5)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(larger_result), np.tanh(larger_host @ w_host), atol=2e-5, rtol=2e-5)
print("Changed workload:", larger.shape, larger_times)

# Reference practice: Benchmark a different equivalent workload
# Benchmark a different equivalent workload (Transfer): Changing the computation changes the performance question.
# Wrap with `jax.jit` (`energy`) so XLA traces and compiles the function.
energy = jax.jit(lambda a, b: jnp.sum((a @ b)**2, axis=1))
# Wait for device work to finish before timing it or comparing its result on the host.
energy(x, w).block_until_ready()
energy_result, energy_times = synchronized_samples(energy, (x, w), 5)
# Reduce along axis=1 to compute `energy_reference`.
energy_reference = np.sum((x_host @ w_host)**2, axis=1)
# Check tensor shape invariant: `energy_result.shape == (64,)`
assert energy_result.shape == (64,)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(energy_result), energy_reference, atol=2e-3, rtol=2e-5)
print("Energy shape and samples:", energy_result.shape, energy_times)

# Reference practice: Repair a timer that includes setup but omits completion
# Repair a timer that includes setup but omits completion (Diagnosis)
def repaired_benchmark(fn, args):
    # Wait for device work to finish before timing it or comparing its result on the host.
    jax.block_until_ready(args)
    # Wait for device work to finish before timing it or comparing its result on the host.
    jax.block_until_ready(fn(*args))
    result, measured = synchronized_samples(fn, args, 5)
    return {"boundary": "placed inputs; warm call plus output wait", "samples": measured}, result
fixed_report, fixed_result = repaired_benchmark(compiled_predict, (x, w))
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(fixed_result), reference, atol=2e-5, rtol=2e-5)
assert len(fixed_report["samples"]) == 5
print(fixed_report)
print("PASS: performance-01")
