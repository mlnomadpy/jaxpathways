"""Deploy at the edge: conversion, budgets and device checks: worked experiments and reference solutions. CPU checks."""

# Keep a known inference function
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
weights = jnp.array([[1., -2.], [.5, 1.], [-1., .25]], jnp.float32)
bias = jnp.array([.1, -.2], jnp.float32)
@jax.jit
def infer(features):
    return features @ weights + bias

assert weights.shape == (3, 2) and bias.shape == (2,)


# Build the raw-request boundary
payload = json.dumps({"features": [[255., 128., 0.]]})
def validate_sensor_payload(payload):
    document = json.loads(payload)
    if not isinstance(document, dict) or set(document) != {'features'}:
        raise ValueError('expected features object')
    rows = document['features']
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list) or len(rows[0]) != 3:
        raise ValueError('expected one row of three sensor values')
    if any(type(value) not in (int, float) or not 0 <= value <= 255 or not np.isfinite(value) for value in rows[0]):
        raise ValueError('expected numeric sensor values in the range 0 through 255')
    return np.asarray(rows, dtype=np.float32)
def request(payload):
    decoded = validate_sensor_payload(payload)
    features = decoded / 255.
    output = np.asarray(infer(jnp.asarray(features)).block_until_ready())
    return json.dumps({"scores": output.tolist()})

np.testing.assert_array_equal(validate_sensor_payload(payload), [[255., 128., 0.]])


# Measure cold and warm requests separately
start = time.perf_counter()
first_response = request(payload)
first_ms = (time.perf_counter() - start) * 1000
samples = []
for _ in range(30):
    start = time.perf_counter()
    response = request(payload)
    samples.append((time.perf_counter() - start) * 1000)


# Validate outputs and describe measurement scope
expected = (np.array([[255., 128., 0.]], np.float32) / 255.) @ np.asarray(weights) + np.asarray(bias)
np.testing.assert_allclose(json.loads(response)["scores"], expected, atol=1e-6)
report = {"runtime": "JAX CPU instructional proxy", "jax": jax.__version__,
          "first_request_ms": first_ms, "warm_samples_ms": samples,
          "p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95)),
          "boundary": "JSON decode + normalize + transfer + infer + wait + encode",
          "edge_device_validated": False}
assert len(samples) == 30 and all(t >= 0 for t in samples)
print(json.dumps(report, indent=2))

np.testing.assert_allclose(json.loads(request(json.dumps({'features':[[0,0,0]]})))['scores'], np.asarray(bias)[None,:], atol=1e-6)


import json
import time
import numpy as np
import jax
import jax.numpy as jnp
weights = jnp.array([[1., -2.], [.5, 1.], [-1., .25]], jnp.float32)
bias = jnp.array([.1, -.2], jnp.float32)
@jax.jit
def infer(features):
    return features @ weights + bias
payload = json.dumps({"features": [[255., 128., 0.]]})
def validate_sensor_payload(payload):
    document = json.loads(payload)
    if not isinstance(document, dict) or set(document) != {'features'}:
        raise ValueError('expected features object')
    rows = document['features']
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list) or len(rows[0]) != 3:
        raise ValueError('expected one row of three sensor values')
    if any(type(value) not in (int, float) or not 0 <= value <= 255 or not np.isfinite(value) for value in rows[0]):
        raise ValueError('expected numeric sensor values in the range 0 through 255')
    return np.asarray(rows, dtype=np.float32)
def request(payload):
    decoded = validate_sensor_payload(payload)
    features = decoded / 255.
    output = np.asarray(infer(jnp.asarray(features)).block_until_ready())
    return json.dumps({"scores": output.tolist()})
start = time.perf_counter()
first_response = request(payload)
first_ms = (time.perf_counter() - start) * 1000
samples = []
for _ in range(30):
    start = time.perf_counter()
    response = request(payload)
    samples.append((time.perf_counter() - start) * 1000)
expected = (np.array([[255., 128., 0.]], np.float32) / 255.) @ np.asarray(weights) + np.asarray(bias)
np.testing.assert_allclose(json.loads(response)["scores"], expected, atol=1e-6)
report = {"runtime": "JAX CPU instructional proxy", "jax": jax.__version__,
          "first_request_ms": first_ms, "warm_samples_ms": samples,
          "p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95)),
          "boundary": "JSON decode + normalize + transfer + infer + wait + encode",
          "edge_device_validated": False}
assert len(samples) == 30 and all(t >= 0 for t in samples)
print(json.dumps(report, indent=2))


# Figure data experiment
visual_data = {'kind': 'line', 'x': list(range(1, len(samples) + 1)), 'xlabel': 'warm request number', 'ylabel': 'end-to-end milliseconds', 'series': [{'label': 'local CPU request', 'y': samples}, {'label': 'recorded p95', 'y': [report['p95_ms']] * len(samples)}]}

# Experiment: A normalization mismatch survives conversion
raw = np.array([[255., 128., 0.]], np.float32)
wrong = np.asarray(infer(raw))
right = np.asarray(infer(raw / 255.))
assert not np.allclose(wrong, right)
print("Preprocessing mismatch max error", float(np.max(np.abs(wrong-right))))


# Experiment: Reject raw inputs before normalization hides their meaning
invalid_payloads=[{'features':[['255',128,0]]},{'features':[[True,128,0]]},{'features':[[256,128,0]]},{'features':[[-1,128,0]]},{'features':[[0,1]]},{'features':[[0,1,float('nan')]]},{'features':[[0,1,2]],'extra':1}]
for invalid in invalid_payloads:
    try: request(json.dumps(invalid))
    except ValueError: pass
    else: raise AssertionError('invalid sensor payload accepted')
np.testing.assert_allclose(json.loads(request(json.dumps({'features':[[0,0,0]]})))['scores'],np.asarray(bias)[None,:],atol=1e-6)
print('Seven malformed or out-of-domain requests rejected; zero input returns the bias.')


# Reference solution. Try the exercise before reading this.
changed_raw = np.array([[0., 255., 128.]], np.float32)
changed_response = json.loads(request(json.dumps({"features": changed_raw.tolist()})))
expected_changed = (changed_raw / 255.) @ np.asarray(weights) + np.asarray(bias)
np.testing.assert_allclose(changed_response["scores"], expected_changed, atol=1e-6)
print("Changed end-to-end request verified")

# Reference practice: Make incomplete deployment evidence explicit
device_report = {"device": None, "runtime_version": None, "delegate": None,
                 "quality_metric": None, "p95_ms": None, "peak_memory_bytes": None,
                 "cold_start_ms": None, "fallback_operators": None}
missing = [key for key, value in device_report.items() if value is None]
assert missing and "device" in missing
print("Not device-validated; missing:", ", ".join(missing))


# Reference practice: Decide whether a model speedup meets the request budget
fixed_ms=12.+4.;model_ms=8.;deadline_ms=19.
before_ms=fixed_ms+model_ms
after_ms=fixed_ms+model_ms/2
np.testing.assert_allclose([before_ms,after_ms,before_ms/after_ms],[24.,20.,1.2])
assert after_ms>deadline_ms and fixed_ms==16.
print('Illustrative request before/after/lower-bound (ms):',before_ms,after_ms,fixed_ms)

print("PASS: deployment-06")
