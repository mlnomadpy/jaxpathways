"""Deploy at the edge: conversion, budgets and device checks: worked experiments and reference solutions. CPU checks."""



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
def request(payload):
    decoded = np.asarray(json.loads(payload)["features"], dtype=np.float32)
    if decoded.shape != (1, 3) or not np.isfinite(decoded).all():
        raise ValueError("expected finite [1, 3] sensor features")
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

print("PASS: deployment-06")
