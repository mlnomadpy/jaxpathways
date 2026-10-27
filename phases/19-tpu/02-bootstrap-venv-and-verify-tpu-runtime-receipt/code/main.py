"""Bootstrap .venv and verify a TPU runtime receipt: worked experiments and reference solutions. CPU checks."""

# Define the closed-form sum-of-squares oracle and pmap receipt builder
# Step 1 — Define the closed-form sum-of-squares oracle and pmap receipt builder
import hashlib
import json
import platform
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np


def closed_form_sum_of_squares(num_devices, width=8):
    n = int(num_devices) * int(width)
    return float((n - 1) * n * (2 * n - 1) // 6)


def build_runtime_receipt(expected_backend="tpu", expected_device_count=4):
    devices = jax.local_devices()
    n_dev = len(devices)
    x = jnp.arange(n_dev * 8, dtype=jnp.float32).reshape(n_dev, 8)
    # Reduce across the target axis to summarize `per_device`.
    per_device = jax.pmap(lambda row: jnp.sum(row * row))(x)
    # Wait for device work to finish before timing it or comparing its result on the host.
    observed_total = float(jax.block_until_ready(jnp.sum(per_device)))
    expected_total = closed_form_sum_of_squares(n_dev, 8)
    backend = jax.default_backend()
    arithmetic_ok = bool(np.isclose(observed_total, expected_total))
    expected_backend_ok = backend == expected_backend
    expected_topology_ok = jax.device_count() == expected_device_count
    source_value = globals().get("__file__")
    source_path = Path(source_value) if isinstance(source_value, str) and source_value else None
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path is not None and source_path.is_file() else None
    source_identity = str(source_path.resolve()) if source_sha256 is not None else "unavailable: interactive execution has no readable __file__"
    payload = {
        "source_sha256": source_sha256,
        "source_identity": source_identity,
        "environment_lock": "projects/reproducible-experiments/requirements.txt; jax[tpu]==0.9.2",
        "python": platform.python_version(), "jax": jax.__version__,
        "expected_backend": expected_backend, "observed_backend": backend,
        "expected_device_count": expected_device_count, "device_count": jax.device_count(),
        "process_count": jax.process_count(),
        "local_device_count": n_dev, "per_device_sums": [float(v) for v in np.asarray(per_device)],
        "observed_total": observed_total, "expected_total": expected_total,
        "arithmetic_ok": arithmetic_ok, "expected_backend_ok": expected_backend_ok,
        "expected_topology_ok": expected_topology_ok,
        "requested_target_ready": bool(arithmetic_ok and expected_backend_ok and expected_topology_ok),
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return payload

# Verify the active runtime receipt and tabulate multi-chip expectations
# Step 2 — Verify the active runtime receipt and tabulate device-count expectations
slice_expectations = [
    {"slice": "1-device (CPU / v5e-1)", "devices": 1, "expected_sum": closed_form_sum_of_squares(1)},
    {"slice": "4-device (v5litepod-4 / v6e-4)", "devices": 4, "expected_sum": closed_form_sum_of_squares(4)},
    {"slice": "8-device (v5litepod-8 / v6e-8)", "devices": 8, "expected_sum": closed_form_sum_of_squares(8)},
]
receipt = build_runtime_receipt(expected_backend="tpu", expected_device_count=4)
# CPU rehearsal validates arithmetic while explicitly showing the TPU target is not ready.
assert receipt["arithmetic_ok"] is True
if receipt["observed_backend"] == "cpu":
    assert receipt["requested_target_ready"] is False
# Actual TPU launch/preflight commands set TPU_REQUIRE_TARGET=1 and require exact backend/count.
if __import__("os").environ.get("TPU_REQUIRE_TARGET") == "1":
    if receipt["source_sha256"] is None:
        raise SystemExit("TPU_REQUIRE_TARGET=1 requires a saved verifier script with a readable source identity")
    if not receipt["requested_target_ready"]:
        raise SystemExit("Requested TPU target is not ready: expected TPU backend and exactly 4 JAX devices")
print("Runtime receipt:", json.dumps(receipt, sort_keys=True))
print("Expected pmap sums by slice:", {s["slice"]: s["expected_sum"] for s in slice_expectations})

# Step 3: Verify invariants on the completed state
receipt = build_runtime_receipt(expected_backend="tpu", expected_device_count=4)
assert receipt["arithmetic_ok"] is True

# Step 1 — Define the closed-form sum-of-squares oracle and pmap receipt builder
import hashlib
import json
import platform
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np


def closed_form_sum_of_squares(num_devices, width=8):
    n = int(num_devices) * int(width)
    return float((n - 1) * n * (2 * n - 1) // 6)


def build_runtime_receipt(expected_backend="tpu", expected_device_count=4):
    devices = jax.local_devices()
    n_dev = len(devices)
    x = jnp.arange(n_dev * 8, dtype=jnp.float32).reshape(n_dev, 8)
    # Each device reduces its eight values to one local sum of squares.
    per_device = jax.pmap(lambda row: jnp.sum(row * row))(x)
    # Wait for device work to finish before timing it or comparing its result on the host.
    observed_total = float(jax.block_until_ready(jnp.sum(per_device)))
    expected_total = closed_form_sum_of_squares(n_dev, 8)
    backend = jax.default_backend()
    arithmetic_ok = bool(np.isclose(observed_total, expected_total))
    expected_backend_ok = backend == expected_backend
    expected_topology_ok = jax.device_count() == expected_device_count
    source_value = globals().get("__file__")
    source_path = Path(source_value) if isinstance(source_value, str) and source_value else None
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path is not None and source_path.is_file() else None
    source_identity = str(source_path.resolve()) if source_sha256 is not None else "unavailable: interactive execution has no readable __file__"
    payload = {
        "source_sha256": source_sha256,
        "source_identity": source_identity,
        "environment_lock": "projects/reproducible-experiments/requirements.txt; jax[tpu]==0.9.2",
        "python": platform.python_version(), "jax": jax.__version__,
        "expected_backend": expected_backend, "observed_backend": backend,
        "expected_device_count": expected_device_count, "device_count": jax.device_count(),
        "process_count": jax.process_count(),
        "local_device_count": n_dev, "per_device_sums": [float(v) for v in np.asarray(per_device)],
        "observed_total": observed_total, "expected_total": expected_total,
        "arithmetic_ok": arithmetic_ok, "expected_backend_ok": expected_backend_ok,
        "expected_topology_ok": expected_topology_ok,
        "requested_target_ready": bool(arithmetic_ok and expected_backend_ok and expected_topology_ok),
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return payload


# Step 2 — Verify the active runtime receipt and tabulate device-count expectations
slice_expectations = [
    {"slice": "1-device (CPU / v5e-1)", "devices": 1, "expected_sum": closed_form_sum_of_squares(1)},
    {"slice": "4-device (v5litepod-4 / v6e-4)", "devices": 4, "expected_sum": closed_form_sum_of_squares(4)},
    {"slice": "8-device (v5litepod-8 / v6e-8)", "devices": 8, "expected_sum": closed_form_sum_of_squares(8)},
]
receipt = build_runtime_receipt(expected_backend="tpu", expected_device_count=4)
# CPU rehearsal validates arithmetic while explicitly showing the TPU target is not ready.
assert receipt["arithmetic_ok"] is True
if receipt["observed_backend"] == "cpu":
    assert receipt["requested_target_ready"] is False
# Actual TPU launch/preflight commands set TPU_REQUIRE_TARGET=1 and require exact backend/count.
if __import__("os").environ.get("TPU_REQUIRE_TARGET") == "1":
    if receipt["source_sha256"] is None:
        raise SystemExit("TPU_REQUIRE_TARGET=1 requires a saved verifier script with a readable source identity")
    if not receipt["requested_target_ready"]:
        raise SystemExit("Requested TPU target is not ready: expected TPU backend and exactly 4 JAX devices")
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

# Experiment: Verify absence of named route-owned resources
import json
def named_resources_absent(gcloud_nodes_json, target_vm_name, queued_requests_json="[]", queued_resource_id=None):
    nodes = json.loads(gcloud_nodes_json)
    requests = json.loads(queued_requests_json)
    vm_absent = all(row.get("name") != target_vm_name for row in nodes)
    request_absent = queued_resource_id is None or all(row.get("name") != queued_resource_id for row in requests)
    return vm_absent and request_absent

# An unrelated VM can remain; the named VM and owning request are absent.
assert named_resources_absent('[{"name":"other-course-vm","state":"READY"}]', "jax-tpu-course", "[]", "jax-tpu-course-request") is True
assert named_resources_absent('[{"name":"jax-tpu-course","state":"READY"}]', "jax-tpu-course") is False
print("Named-resource cleanup checks passed; unrelated nodes may remain.")

# Reference solution. Try the exercise before reading this.
# Exercise solution
s4 = closed_form_sum_of_squares(4)
s8 = closed_form_sum_of_squares(8)
print("4-device expected sum:", s4, "8-device expected sum:", s8, "receipt hash:", receipt["receipt_sha256"])
# Assert invariant `s4 == 10416.0 and s8 == 85344.0 and receipt["arithmetic_ok"] is True` holds
assert s4 == 10416.0 and s8 == 85344.0 and receipt["arithmetic_ok"] is True
if receipt["observed_backend"] == "cpu":
    assert receipt["requested_target_ready"] is False

# Readiness is a requested contract, not a synonym for successful arithmetic.
expected_ready = bool(receipt['arithmetic_ok'] and receipt['observed_backend'] == 'tpu'
                      and receipt['device_count'] == 4)
assert receipt['requested_target_ready'] is expected_ready
# A schema-only negative control is not hardware evidence.
wrong_count = dict(receipt, observed_backend='tpu', device_count=8)
wrong_count_ready = bool(wrong_count['arithmetic_ok'] and wrong_count['observed_backend'] == 'tpu'
                         and wrong_count['device_count'] == 4)
assert wrong_count_ready is False
print('Requested four-device TPU readiness:', expected_ready,
      '; simulated wrong-count contract rejected:', not wrong_count_ready)


# Reference practice: Detect a silent fallback from 4 TPU chips to 1 CPU device
# Detect a silent fallback from 4 TPU chips to 1 CPU device (Foundations)
def is_four_chip_tpu_receipt(r):
    try:
        return bool(
            r["observed_backend"] == "tpu"
            and r["local_device_count"] == 4
            and r["device_count"] == 4
            and np.isclose(r["observed_total"], 10416.0)
        )
    except (KeyError, TypeError):
        return False

cpu_result = is_four_chip_tpu_receipt(receipt)
print("Real runtime receipt qualifies as four-device TPU:", cpu_result)
assert cpu_result is (receipt["observed_backend"] == "tpu" and receipt["device_count"] == 4)


# Reference practice: Estimate hourly billing savings from immediate teardown
# Estimate hourly billing savings from immediate teardown (Transfer / diagnosis)
hourly_rate = 4.80
saved_usd = round((16.0 - 0.5) * hourly_rate, 2)
print("Overnight idle cost avoided (USD):", saved_usd)
# Assert invariant `saved_usd == 74.4` holds
assert saved_usd == 74.4
print("PASS: tpu-04")
