"""Bootstrap .venv and verify a TPU runtime receipt: worked experiments and reference solutions. CPU checks."""

# Define the closed-form sum-of-squares oracle and pmap receipt builder
# Step 1 — Define the closed-form sum-of-squares oracle and pmap receipt builder: Each local device computes the sum of squares of its 8-element row...
# Import hashlib for this computation.
import hashlib
import json
import platform
import jax
import jax.numpy as jnp
import numpy as np


# Function `closed_form_sum_of_squares(num_devices, width)` implementing this stage's computation:
def closed_form_sum_of_squares(num_devices, width=8):
    # Evaluate `num_devices) * int(width` and convert the result into Python scalar/collection `n`.
    n = int(num_devices) * int(width)
    # Return `float((n - 1) * n * (2 * n - 1) // 6)` to the caller.
    return float((n - 1) * n * (2 * n - 1) // 6)


# Function `build_runtime_receipt()` implementing this stage's computation:
def build_runtime_receipt():
    # Query the active JAX devices into `devices`.
    devices = jax.local_devices()
    # Run `len` to compute `n_dev`.
    n_dev = len(devices)
    # Construct and reshape `x` into the target tensor dimensions.
    x = jnp.arange(n_dev * 8, dtype=jnp.float32).reshape(n_dev, 8)
    # Reduce across the target axis to summarize `per_device`.
    per_device = jax.pmap(lambda row: jnp.sum(row * row))(x)
    # Synchronize host execution until asynchronous device computation completes.
    observed_total = float(jax.block_until_ready(jnp.sum(per_device)))
    # Run `closed_form_sum_of_squares` to compute `expected_total`.
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
    # Read or serialize artifact data on disk (`payload['receipt_sha256']`).
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    # Return `payload` to the caller.
    return payload

# Verify the active runtime receipt and tabulate multi-chip expectations
# Step 2 — Verify the active runtime receipt and tabulate multi-chip expectations: Knowing the exact expected sum for 1, 4, and 8 devices lets you...
slice_expectations = [
    {"slice": "1-device (CPU / v5e-1)", "devices": 1, "expected_sum": closed_form_sum_of_squares(1)},
    {"slice": "4-device (v5litepod-4 / v6e-4)", "devices": 4, "expected_sum": closed_form_sum_of_squares(4)},
    {"slice": "8-device (v5litepod-8 / v6e-8)", "devices": 8, "expected_sum": closed_form_sum_of_squares(8)},
]
# Run `build_runtime_receipt` to compute `receipt`.
receipt = build_runtime_receipt()
# Assert invariant `receipt["verified"] is True` holds
assert receipt["verified"] is True
# Print the observed values to compare against the expected result.
print("Runtime receipt:", json.dumps(receipt, sort_keys=True))
# Print diagnostic summary of the computed outputs.
print("Expected pmap sums by slice:", {s["slice"]: s["expected_sum"] for s in slice_expectations})

# Step 3: Verify invariants on the completed state
receipt = build_runtime_receipt()
assert receipt["verified"] is True

# Step 1 — Define the closed-form sum-of-squares oracle and pmap receipt builder: Each local device computes the sum of squares of its 8-element row...
# Import hashlib for this computation.
import hashlib
import json
import platform
import jax
import jax.numpy as jnp
import numpy as np


# Function `closed_form_sum_of_squares(num_devices, width)` implementing this stage's computation:
def closed_form_sum_of_squares(num_devices, width=8):
    # Evaluate `num_devices) * int(width` and convert the result into Python scalar/collection `n`.
    n = int(num_devices) * int(width)
    # Return `float((n - 1) * n * (2 * n - 1) // 6)` to the caller.
    return float((n - 1) * n * (2 * n - 1) // 6)


# Function `build_runtime_receipt()` implementing this stage's computation:
def build_runtime_receipt():
    # Query the active JAX devices into `devices`.
    devices = jax.local_devices()
    # Run `len` to compute `n_dev`.
    n_dev = len(devices)
    # Construct and reshape `x` into the target tensor dimensions.
    x = jnp.arange(n_dev * 8, dtype=jnp.float32).reshape(n_dev, 8)
    # Reduce across the target axis to summarize `per_device`.
    per_device = jax.pmap(lambda row: jnp.sum(row * row))(x)
    # Synchronize host execution until asynchronous device computation completes.
    observed_total = float(jax.block_until_ready(jnp.sum(per_device)))
    # Run `closed_form_sum_of_squares` to compute `expected_total`.
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
    # Read or serialize artifact data on disk (`payload['receipt_sha256']`).
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    # Return `payload` to the caller.
    return payload


# Step 2 — Verify the active runtime receipt and tabulate multi-chip expectations: Knowing the exact expected sum for 1, 4, and 8 devices lets you...
slice_expectations = [
    {"slice": "1-device (CPU / v5e-1)", "devices": 1, "expected_sum": closed_form_sum_of_squares(1)},
    {"slice": "4-device (v5litepod-4 / v6e-4)", "devices": 4, "expected_sum": closed_form_sum_of_squares(4)},
    {"slice": "8-device (v5litepod-8 / v6e-8)", "devices": 8, "expected_sum": closed_form_sum_of_squares(8)},
]
# Run `build_runtime_receipt` to compute `receipt`.
receipt = build_runtime_receipt()
# Assert invariant `receipt["verified"] is True` holds
assert receipt["verified"] is True
# Print the observed values to compare against the expected result.
print("Runtime receipt:", json.dumps(receipt, sort_keys=True))
# Print diagnostic summary of the computed outputs.
print("Expected pmap sums by slice:", {s["slice"]: s["expected_sum"] for s in slice_expectations})

# Figure data experiment
# Compute figure data for: Per-chip contribution to the 4-device synchronized pmap verification receipt
# Create evenly spaced index values in `per_chip`.
per_chip = [float(np.sum(np.arange(i * 8, (i + 1) * 8, dtype=np.float64) ** 2)) for i in range(4)]
# Compute `cumulative` from `[float(sum(per_chip[: i + 1])) for i in range(4)]`
cumulative = [float(sum(per_chip[: i + 1])) for i in range(4)]
# Construct dictionary `visual_data` with the structured fields for this stage.
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
# Experiment — Compare per-device row sums on a 4-chip TPU slice: Inspecting per_device_sums confirms that distinct shards of x...
# Construct and reshape `x4` into the target tensor dimensions.
x4 = jnp.arange(4 * 8, dtype=jnp.float32).reshape(4, 8)
# Reduce along axis=1 to compute `row_sums_4`.
row_sums_4 = [float(v) for v in jnp.sum(x4 * x4, axis=1)]
# Print the observed values to compare against the expected result.
print("Per-chip sums on a 4-chip slice:", row_sums_4, "Total:", sum(row_sums_4))
# Assert that `row_sums_4 == [140.0, 1100.0, 3084.0, 6092.0] and sum(row_sums_4) == 10416.0`.
assert row_sums_4 == [140.0, 1100.0, 3084.0, 6092.0] and sum(row_sums_4) == 10416.0

# Experiment: Verify teardown confirmation logic
# Experiment — Verify teardown confirmation logic: Checking gcloud compute tpus tpu-vm list --zone=$ZONE...
def is_tpu_zone_clean(gcloud_list_json):
    # Read or serialize artifact data on disk (`items`).
    items = json.loads(gcloud_list_json)
    # Return `len(items) == 0` to the caller.
    return len(items) == 0

# Assert invariant `is_tpu_zone_clean("[]") is True` holds
assert is_tpu_zone_clean("[]") is True
# Assert invariant `is_tpu_zone_clean('[{"name": "jax-tpu-course"` holds
assert is_tpu_zone_clean('[{"name": "jax-tpu-course", "state": "READY"}]') is False
# Print the observed values to compare against the expected result.
print("Teardown verifier passed for empty and non-empty gcloud list outputs.")

# Reference solution. Try the exercise before reading this.
# Exercise solution: Compute closed_form_sum_of_squares(4) and...
s4 = closed_form_sum_of_squares(4)
# Run `closed_form_sum_of_squares` to compute `s8`.
s8 = closed_form_sum_of_squares(8)
# Print the observed values to compare against the expected result.
print("4-chip expected sum:", s4, "8-chip expected sum:", s8, "receipt hash:", receipt["receipt_sha256"])
# Assert invariant `s4 == 10416.0 and s8 == 85344.0 and receipt["verified"] is True` holds
assert s4 == 10416.0 and s8 == 85344.0 and receipt["verified"] is True

# Reference practice: Detect a silent fallback from 4 TPU chips to 1 CPU device
# Detect a silent fallback from 4 TPU chips to 1 CPU device (Foundations): Requiring backend == 'tpu', local_device_count == 4, and the...
def is_four_chip_tpu_receipt(r):
    # Return `bool(r['backend'] == 'tpu' and r['local_device_count'] == 4 and np.isclose(r['observed_total'], 10416.0))` to the caller.
    return bool(r["backend"] == "tpu" and r["local_device_count"] == 4 and np.isclose(r["observed_total"], 10416.0))

# Compute `simulated_tpu` from `{"backend": "tpu", "local_device_count": 4, "observe...`
simulated_tpu = {"backend": "tpu", "local_device_count": 4, "observed_total": 10416.0}
# Print the observed values to compare against the expected result.
print("Simulated 4-chip TPU check:", is_four_chip_tpu_receipt(simulated_tpu), "Local CPU check:", is_four_chip_tpu_receipt(receipt))
# Assert invariant `is_four_chip_tpu_receipt(simulated_tpu) is True` holds
assert is_four_chip_tpu_receipt(simulated_tpu) is True

# Reference practice: Estimate hourly billing savings from immediate teardown
# Estimate hourly billing savings from immediate teardown (Transfer / diagnosis): Deleting the Cloud TPU VM immediately after copying...
hourly_rate = 4.80
# Run `round` to compute `saved_usd`.
saved_usd = round((16.0 - 0.5) * hourly_rate, 2)
# Print the observed values to compare against the expected result.
print("Overnight idle cost avoided (USD):", saved_usd)
# Assert invariant `saved_usd == 74.4` holds
assert saved_usd == 74.4
print("PASS: tpu-04")
