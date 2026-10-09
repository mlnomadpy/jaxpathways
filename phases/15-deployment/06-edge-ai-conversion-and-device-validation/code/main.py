"""Deploy at the edge: conversion, budgets and device checks: worked experiments and reference solutions. CPU checks."""

# Keep a known inference function
# Step 1 — Keep a known inference function: A row has three sensor features and the result has two scores.
# Import json for this computation.
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
# Construct `weights` via `jnp.array([[1., -2.], [.5, 1.], [-1., .25]], jnp.flo...`
weights = jnp.array([[1., -2.], [.5, 1.], [-1., .25]], jnp.float32)
# Construct `bias` via `jnp.array([.1, -.2], jnp.float32)`
bias = jnp.array([.1, -.2], jnp.float32)
# Define and JIT-compile `infer(features)` so XLA traces and fuses the operations:
@jax.jit
# Function `infer(features)` implementing this stage's computation:
def infer(features):
    # Return `features @ weights + bias` to the caller.
    return features @ weights + bias

# Check tensor shape invariant: `weights.shape == (3, 2) and bias.shape == (2,)`
assert weights.shape == (3, 2) and bias.shape == (2,)

# Build the raw-request boundary
# Step 2 — Build the raw-request boundary: This probe checks decoding and raw values without running inference.
payload = json.dumps({"features": [[255., 128., 0.]]})
# Function `validate_sensor_payload(payload)` implementing this stage's computation:
def validate_sensor_payload(payload):
    # Read or serialize artifact data on disk (`document`).
    document = json.loads(payload)
    # Guard input contract (`not isinstance(document, dict) or set(document) != {'features'}`) and fail fast if violated.
    if not isinstance(document, dict) or set(document) != {'features'}:
        raise ValueError('expected features object')
    # Compute `rows` from `document['features']`
    rows = document['features']
    # Guard input contract (`not isinstance(rows, list) or len(rows) != 1 or (not isinstance(rows[0], list)) or (len(rows[0]) != 3)`) and fail fast if violated.
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list) or len(rows[0]) != 3:
        raise ValueError('expected one row of three sensor values')
    # Guard input contract (`any((type(value) not in (int, float) or not 0 <= value <= 255 or (not np.isfinite(value)) for value in rows[0]))`) and fail fast if violated.
    if any(type(value) not in (int, float) or not 0 <= value <= 255 or not np.isfinite(value) for value in rows[0]):
        raise ValueError('expected numeric sensor values in the range 0 through 255')
    # Return `np.asarray(rows, dtype=np.float32)` to the caller.
    return np.asarray(rows, dtype=np.float32)
# Function `request(payload)` implementing this stage's computation:
def request(payload):
    # Run `validate_sensor_payload` to compute `decoded`.
    decoded = validate_sensor_payload(payload)
    # Compute `features` from `decoded / 255.`
    features = decoded / 255.
    # Synchronize host execution until asynchronous device computation completes.
    output = np.asarray(infer(jnp.asarray(features)).block_until_ready())
    # Return `json.dumps({'scores': output.tolist()})` to the caller.
    return json.dumps({"scores": output.tolist()})

# Execute `np.testing.assert_array_equal(validate_sensor_payload(payloa`
np.testing.assert_array_equal(validate_sensor_payload(payload), [[255., 128., 0.]])

# Measure cold and warm requests separately
# Step 3 — Measure cold and warm requests separately: The first request can include compilation and initialization.
start = time.perf_counter()
# Run `request` to compute `first_response`.
first_response = request(payload)
# Record execution timing or profiler trace in `first_ms`.
first_ms = (time.perf_counter() - start) * 1000
# Compute `samples` from `[]`
samples = []
# Repeat the update loop over `range(30)` steps:
for _ in range(30):
    # Record execution timing or profiler trace in `start`.
    start = time.perf_counter()
    # Run `request` to compute `response`.
    response = request(payload)
    # Record execution timing or profiler trace in ``.
    samples.append((time.perf_counter() - start) * 1000)

# Validate outputs and describe measurement scope
# Step 4 — Validate outputs and describe measurement scope: The report records local CPU timings and explicitly leaves...
# Compute `expected` from `(np.array([[255., 128., 0.]], np.float32) / 255.) @ ...`
expected = (np.array([[255., 128., 0.]], np.float32) / 255.) @ np.asarray(weights) + np.asarray(bias)
# Compute `np.testing.assert_allclose(json.loads(response)["scores"], expected, atol` as `1e-6)`.
np.testing.assert_allclose(json.loads(response)["scores"], expected, atol=1e-6)
# Compute `report` from `{"runtime": "JAX CPU instructional proxy", "jax": ja...`
report = {"runtime": "JAX CPU instructional proxy", "jax": jax.__version__,
          "first_request_ms": first_ms, "warm_samples_ms": samples,
          "p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95)),
          "boundary": "JSON decode + normalize + transfer + infer + wait + encode",
          "edge_device_validated": False}
# Assert invariant `len(samples) == 30 and all(t >= 0 for t in samples)` holds
assert len(samples) == 30 and all(t >= 0 for t in samples)
# Print the observed values to compare against the expected result.
print(json.dumps(report, indent=2))

# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(json.loads(request(json.dumps({'features':[[0,0,0]]})))['scores'], np.asarray(bias)[None,:], atol=1e-6)

# Deploy at the edge: conversion, budgets and device checks: An edge deployment includes input decoding, preprocessing,...
# Import json for this computation.
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
# Construct `weights` via `jnp.array([[1., -2.], [.5, 1.], [-1., .25]], jnp.flo...`
weights = jnp.array([[1., -2.], [.5, 1.], [-1., .25]], jnp.float32)
# Construct `bias` via `jnp.array([.1, -.2], jnp.float32)`
bias = jnp.array([.1, -.2], jnp.float32)
# Define and JIT-compile `infer(features)` so XLA traces and fuses the operations:
@jax.jit
# Function `infer(features)` implementing this stage's computation:
def infer(features):
    # Return `features @ weights + bias` to the caller.
    return features @ weights + bias
# Read or serialize artifact data on disk (`payload`).
payload = json.dumps({"features": [[255., 128., 0.]]})
# Function `validate_sensor_payload(payload)` implementing this stage's computation:
def validate_sensor_payload(payload):
    # Read or serialize artifact data on disk (`document`).
    document = json.loads(payload)
    # Guard input contract (`not isinstance(document, dict) or set(document) != {'features'}`) and fail fast if violated.
    if not isinstance(document, dict) or set(document) != {'features'}:
        raise ValueError('expected features object')
    # Compute `rows` from `document['features']`
    rows = document['features']
    # Guard input contract (`not isinstance(rows, list) or len(rows) != 1 or (not isinstance(rows[0], list)) or (len(rows[0]) != 3)`) and fail fast if violated.
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list) or len(rows[0]) != 3:
        raise ValueError('expected one row of three sensor values')
    # Guard input contract (`any((type(value) not in (int, float) or not 0 <= value <= 255 or (not np.isfinite(value)) for value in rows[0]))`) and fail fast if violated.
    if any(type(value) not in (int, float) or not 0 <= value <= 255 or not np.isfinite(value) for value in rows[0]):
        raise ValueError('expected numeric sensor values in the range 0 through 255')
    # Return `np.asarray(rows, dtype=np.float32)` to the caller.
    return np.asarray(rows, dtype=np.float32)
# Function `request(payload)` implementing this stage's computation:
def request(payload):
    # Run `validate_sensor_payload` to compute `decoded`.
    decoded = validate_sensor_payload(payload)
    # Compute `features` from `decoded / 255.`
    features = decoded / 255.
    # Synchronize host execution until asynchronous device computation completes.
    output = np.asarray(infer(jnp.asarray(features)).block_until_ready())
    # Return `json.dumps({'scores': output.tolist()})` to the caller.
    return json.dumps({"scores": output.tolist()})
# Record execution timing or profiler trace in `start`.
start = time.perf_counter()
# Run `request` to compute `first_response`.
first_response = request(payload)
# Record execution timing or profiler trace in `first_ms`.
first_ms = (time.perf_counter() - start) * 1000
# Compute `samples` from `[]`
samples = []
# Repeat the update loop over `range(30)` steps:
for _ in range(30):
    # Record execution timing or profiler trace in `start`.
    start = time.perf_counter()
    # Run `request` to compute `response`.
    response = request(payload)
    # Record execution timing or profiler trace in ``.
    samples.append((time.perf_counter() - start) * 1000)
# Convert `expected` to a host NumPy array for inspection or verification.
expected = (np.array([[255., 128., 0.]], np.float32) / 255.) @ np.asarray(weights) + np.asarray(bias)
# Compute `np.testing.assert_allclose(json.loads(response)["scores"], expected, atol` as `1e-6)`.
np.testing.assert_allclose(json.loads(response)["scores"], expected, atol=1e-6)
# Compute `report` from `{"runtime": "JAX CPU instructional proxy", "jax": ja...`
report = {"runtime": "JAX CPU instructional proxy", "jax": jax.__version__,
          "first_request_ms": first_ms, "warm_samples_ms": samples,
          "p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95)),
          "boundary": "JSON decode + normalize + transfer + infer + wait + encode",
          "edge_device_validated": False}
# Assert invariant `len(samples) == 30 and all(t >= 0 for t in samples)` holds
assert len(samples) == 30 and all(t >= 0 for t in samples)
# Print diagnostic summary of the computed outputs.
print(json.dumps(report, indent=2))

# Figure data experiment
# Compute figure data for: An end-to-end boundary includes more than inference
# Compute `visual_data` from `{'kind': 'line', 'x': list(range(1, len(samples) + 1...`
visual_data = {'kind': 'line', 'x': list(range(1, len(samples) + 1)), 'xlabel': 'warm request number', 'ylabel': 'end-to-end milliseconds', 'series': [{'label': 'local CPU request', 'y': samples}, {'label': 'recorded p95', 'y': [report['p95_ms']] * len(samples)}]}

# Experiment: A normalization mismatch survives conversion
# Experiment — A normalization mismatch survives conversion: Conversion checks must start at the raw input boundary when...
# Compute `raw` from `np.array([[255., 128., 0.]], np.float32)`
raw = np.array([[255., 128., 0.]], np.float32)
# Convert `wrong` to a host NumPy array for inspection or verification.
wrong = np.asarray(infer(raw))
# Convert `right` to a host NumPy array for inspection or verification.
right = np.asarray(infer(raw / 255.))
# Assert that `not np.allclose(wrong, right)`.
assert not np.allclose(wrong, right)
# Print the observed values to compare against the expected result.
print("Preprocessing mismatch max error", float(np.max(np.abs(wrong-right))))

# Experiment: Reject raw inputs before normalization hides their meaning
# Experiment — Reject raw inputs before normalization hides their meaning: The zero-input oracle checks that preprocessing does not add a...
invalid_payloads=[{'features':[['255',128,0]]},{'features':[[True,128,0]]},{'features':[[256,128,0]]},{'features':[[-1,128,0]]},{'features':[[0,1]]},{'features':[[0,1,float('nan')]]},{'features':[[0,1,2]],'extra':1}]
# Iterate over `invalid` to step through the computation:
for invalid in invalid_payloads:
    try:
        request(json.dumps(invalid))
    except ValueError:
        pass
    else:
        raise AssertionError('invalid sensor payload accepted')
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(json.loads(request(json.dumps({'features':[[0,0,0]]})))['scores'],np.asarray(bias)[None,:],atol=1e-6)
# Print the observed values to compare against the expected result.
print('Seven malformed or out-of-domain requests rejected; zero input returns the bias.')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Send a second raw input [0, 255, 128] through the full request, and...
# Compute `changed_raw` from `np.array([[0., 255., 128.]], np.float32)`
changed_raw = np.array([[0., 255., 128.]], np.float32)
# Read or serialize artifact data on disk (`changed_response`).
changed_response = json.loads(request(json.dumps({"features": changed_raw.tolist()})))
# Convert `expected_changed` to a host NumPy array for inspection or verification.
expected_changed = (changed_raw / 255.) @ np.asarray(weights) + np.asarray(bias)
# Compute `np.testing.assert_allclose(changed_response["scores"], expected_changed, atol` as `1e-6)`.
np.testing.assert_allclose(changed_response["scores"], expected_changed, atol=1e-6)
# Print the observed values to compare against the expected result.
print("Changed end-to-end request verified")

# Reference practice: Make incomplete deployment evidence explicit
# Make incomplete deployment evidence explicit (Transfer): Empty evidence stays empty.
device_report = {"device": None, "runtime_version": None, "delegate": None,
                 "quality_metric": None, "p95_ms": None, "peak_memory_bytes": None,
                 "cold_start_ms": None, "fallback_operators": None}
# Compute `missing` from `[key for key, value in device_report.items() if valu...`
missing = [key for key, value in device_report.items() if value is None]
# Assert invariant `missing and "device" in missing` holds
assert missing and "device" in missing
# Print the observed values to compare against the expected result.
print("Not device-validated; missing:", ", ".join(missing))

# Reference practice: Decide whether a model speedup meets the request budget
# Decide whether a model speedup meets the request budget (Transfer / diagnosis): Halving model time leaves a 20-millisecond request, so it...
fixed_ms=12.+4.
model_ms=8.
deadline_ms=19.
# Compute `before_ms` from `fixed_ms+model_ms`
before_ms=fixed_ms+model_ms
# Compute `after_ms` from `fixed_ms+model_ms/2`
after_ms=fixed_ms+model_ms/2
# Execute `np.testing.assert_allclose([before_ms,after_ms,before_ms/aft`
np.testing.assert_allclose([before_ms,after_ms,before_ms/after_ms],[24.,20.,1.2])
# Assert invariant `after_ms>deadline_ms and fixed_ms==16.` holds
assert after_ms>deadline_ms and fixed_ms==16.
# Print the observed values to compare against the expected result.
print('Illustrative request before/after/lower-bound (ms):',before_ms,after_ms,fixed_ms)
print("PASS: deployment-06")
