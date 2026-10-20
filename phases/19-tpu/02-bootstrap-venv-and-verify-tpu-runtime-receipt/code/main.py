"""Bootstrap .venv and verify a TPU runtime receipt: worked experiments and reference solutions. CPU checks."""

# Define the closed-form sum-of-squares oracle and pmap receipt builder
# Step 1 — Define the closed-form sum-of-squares oracle and pmap receipt builder
import hashlib
import json
import platform
import jax
import jax.numpy as jnp
import numpy as np


def closed_form_sum_of_squares(num_devices, width=8):
    n = int(num_devices) * int(width)
    return float((n - 1) * n * (2 * n - 1) // 6)


def build_runtime_receipt():
    devices = jax.local_devices()
    n_dev = len(devices)
    x = jnp.arange(n_dev * 8, dtype=jnp.float32).reshape(n_dev, 8)
    # Reduce across the target axis to summarize `per_device`.
    per_device = jax.pmap(lambda row: jnp.sum(row * row))(x)
    # Wait for device work to finish before timing it or comparing its result on the host.
    observed_total = float(jax.block_until_ready(jnp.sum(per_device)))
    expected_total = closed_form_sum_of_squares(n_dev, 8)
    # Check which hardware backend (`cpu`, `gpu`, or `tpu`) JAX selected for `payload`.
    payload = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "local_device_count": n_dev,
        "per_device_sums": [float(v) for v in np.asarray(per_device)],
        "observed_total": observed_total,
        "expected_total": expected_total,
        "verified": bool(np.isclose(observed_total, expected_total)),
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return payload

# Verify the active runtime receipt and tabulate multi-chip expectations
# Step 2 — Verify the active runtime receipt and tabulate multi-chip expectations
slice_expectations = [
    {"slice": "1-device (CPU / v5e-1)", "devices": 1, "expected_sum": closed_form_sum_of_squares(1)},
    {"slice": "4-device (v5litepod-4 / v6e-4)", "devices": 4, "expected_sum": closed_form_sum_of_squares(4)},
    {"slice": "8-device (v5litepod-8 / v6e-8)", "devices": 8, "expected_sum": closed_form_sum_of_squares(8)},
]
receipt = build_runtime_receipt()
# Assert invariant `receipt["verified"] is True` holds
assert receipt["verified"] is True
print("Runtime receipt:", json.dumps(receipt, sort_keys=True))
print("Expected pmap sums by slice:", {s["slice"]: s["expected_sum"] for s in slice_expectations})

# Step 3: Verify invariants on the completed state
receipt = build_runtime_receipt()
assert receipt["verified"] is True

# Step 1 — Define the closed-form sum-of-squares oracle and pmap receipt builder
import hashlib
import json
import platform
import jax
import jax.numpy as jnp
import numpy as np


def closed_form_sum_of_squares(num_devices, width=8):
    n = int(num_devices) * int(width)
    return float((n - 1) * n * (2 * n - 1) // 6)


def build_runtime_receipt():
    devices = jax.local_devices()
    n_dev = len(devices)
    x = jnp.arange(n_dev * 8, dtype=jnp.float32).reshape(n_dev, 8)
    # Each device reduces its eight values to one local sum of squares.
    per_device = jax.pmap(lambda row: jnp.sum(row * row))(x)
    # Wait for device work to finish before timing it or comparing its result on the host.
    observed_total = float(jax.block_until_ready(jnp.sum(per_device)))
    expected_total = closed_form_sum_of_squares(n_dev, 8)
    # Record the backend selected by this JAX process in the payload.
    payload = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "local_device_count": n_dev,
        "per_device_sums": [float(v) for v in np.asarray(per_device)],
        "observed_total": observed_total,
        "expected_total": expected_total,
        "verified": bool(np.isclose(observed_total, expected_total)),
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return payload


# Step 2 — Verify the active runtime receipt and tabulate multi-chip expectations
slice_expectations = [
    {"slice": "1-device (CPU / v5e-1)", "devices": 1, "expected_sum": closed_form_sum_of_squares(1)},
    {"slice": "4-device (v5litepod-4 / v6e-4)", "devices": 4, "expected_sum": closed_form_sum_of_squares(4)},
    {"slice": "8-device (v5litepod-8 / v6e-8)", "devices": 8, "expected_sum": closed_form_sum_of_squares(8)},
]
receipt = build_runtime_receipt()
# Assert invariant `receipt["verified"] is True` holds
assert receipt["verified"] is True
print("Runtime receipt:", json.dumps(receipt, sort_keys=True))
print("Expected pmap sums by slice:", {s["slice"]: s["expected_sum"] for s in slice_expectations})

# Figure data experiment
per_chip = [float(np.sum(np.arange(i * 8, (i + 1) * 8, dtype=np.float64) ** 2)) for i in range(4)]
cumulative = [float(sum(per_chip[: i + 1])) for i in range(4)]
visual_data = {
    'kind': 'bar',
    'labels': ['chip 0 (0..7)', 'chip 1 (8..15)', 'chip 2 (16..23)', 'chip 3 (24..31)'],
    'xlabel': '4-chip TPU VM local device shard',
    'ylabel': 'sum of squares',
    'series': [
        {'label': 'per-chip row sum', 'y': per_chip},
        {'label': 'cumulative receipt sum', 'y': cumulative},
    ],
}

# Experiment: Compare per-device row sums on a 4-chip TPU slice
# Experiment — Compare per-device row sums on a 4-chip TPU slice
x4 = jnp.arange(4 * 8, dtype=jnp.float32).reshape(4, 8)
# Reduce along axis=1 to compute `row_sums_4`.
row_sums_4 = [float(v) for v in jnp.sum(x4 * x4, axis=1)]
print("Per-chip sums on a 4-chip slice:", row_sums_4, "Total:", sum(row_sums_4))
assert row_sums_4 == [140.0, 1100.0, 3084.0, 6092.0] and sum(row_sums_4) == 10416.0

# Experiment: Verify teardown confirmation logic
# Experiment — Verify teardown confirmation logic
def is_tpu_zone_clean(gcloud_list_json):
    items = json.loads(gcloud_list_json)
    return len(items) == 0

# Assert invariant `is_tpu_zone_clean("[]") is True` holds
assert is_tpu_zone_clean("[]") is True
# Assert invariant `is_tpu_zone_clean('[{"name": "jax-tpu-course"` holds
assert is_tpu_zone_clean('[{"name": "jax-tpu-course", "state": "READY"}]') is False
print("Teardown verifier passed for empty and non-empty gcloud list outputs.")

# Reference solution. Try the exercise before reading this.
# Exercise solution
s4 = closed_form_sum_of_squares(4)
s8 = closed_form_sum_of_squares(8)
print("4-chip expected sum:", s4, "8-chip expected sum:", s8, "receipt hash:", receipt["receipt_sha256"])
# Assert invariant `s4 == 10416.0 and s8 == 85344.0 and receipt["verified"] is True` holds
assert s4 == 10416.0 and s8 == 85344.0 and receipt["verified"] is True

# Reference practice: Detect a silent fallback from 4 TPU chips to 1 CPU device
# Detect a silent fallback from 4 TPU chips to 1 CPU device (Foundations)
def is_four_chip_tpu_receipt(r):
    return bool(r["backend"] == "tpu" and r["local_device_count"] == 4 and np.isclose(r["observed_total"], 10416.0))

simulated_tpu = {"backend": "tpu", "local_device_count": 4, "observed_total": 10416.0}
print("Simulated 4-chip TPU check:", is_four_chip_tpu_receipt(simulated_tpu), "Local CPU check:", is_four_chip_tpu_receipt(receipt))
# Assert invariant `is_four_chip_tpu_receipt(simulated_tpu) is True` holds
assert is_four_chip_tpu_receipt(simulated_tpu) is True

# Reference practice: Estimate hourly billing savings from immediate teardown
# Estimate hourly billing savings from immediate teardown (Transfer / diagnosis)
hourly_rate = 4.80
saved_usd = round((16.0 - 0.5) * hourly_rate, 2)
print("Overnight idle cost avoided (USD):", saved_usd)
# Assert invariant `saved_usd == 74.4` holds
assert saved_usd == 74.4
print("PASS: tpu-04")
