"""Select a TPU slice and provision your Cloud TPU VM: worked experiments and reference solutions. CPU checks."""

# Write the runtime contract and synchronized pmap verifier
# Step 1 — Write the runtime contract and synchronized pmap verifier
import hashlib
import json
import os
import platform
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np


def verify_runtime_contract(expected_backend="tpu", expected_device_count=4):
    backend = jax.default_backend()
    devices = jax.devices()
    local_devices = jax.local_devices()
    x = jnp.arange(len(local_devices) * 8, dtype=jnp.float32).reshape(len(local_devices), 8)
    per_device = jax.pmap(lambda row: jnp.sum(row * row))(x)
    total = float(jax.block_until_ready(jnp.sum(per_device)))
    expected_total = float(np.sum(np.arange(len(local_devices) * 8, dtype=np.float64) ** 2))
    arithmetic_ok = bool(np.isclose(total, expected_total))
    expected_backend_ok = backend == expected_backend
    expected_topology_ok = len(devices) == expected_device_count
    source_value = globals().get("__file__")
    source_path = Path(source_value) if isinstance(source_value, str) and source_value else None
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path is not None and source_path.is_file() else None
    source_identity = str(source_path.resolve()) if source_sha256 is not None else "unavailable: interactive execution has no readable __file__"
    receipt = {
        "source_sha256": source_sha256,
        "source_identity": source_identity,
        "environment_lock": "projects/reproducible-experiments/requirements.txt; jax[tpu]==0.9.2",
        "python": platform.python_version(), "jax": jax.__version__,
        "expected_backend": expected_backend, "observed_backend": backend,
        "expected_device_count": expected_device_count, "device_count": len(devices),
        "process_count": jax.process_count(),
        "target_topology": {"api": "Cloud TPU API direct VM/Flex-start", "machine_type": "v5litepod-4", "physical_chips_expected": 4, "hosts_expected": 1, "topology_source": "requested Cloud TPU resource contract; physical chips are not inferred from JAX devices"},
        "local_device_count": len(local_devices), "device_kinds": [str(d.device_kind) for d in devices],
        "pmap_sum_of_squares": total, "expected_sum_of_squares": expected_total,
        "arithmetic_ok": arithmetic_ok, "expected_backend_ok": expected_backend_ok,
        "expected_topology_ok": expected_topology_ok,
        "requested_target_ready": bool(arithmetic_ok and expected_backend_ok and expected_topology_ok),
    }
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return receipt

# Compare single-host TPU slice topologies and emit the receipt
# Step 2 — Compare single-host TPU slice topologies and emit the receipt
topologies = [
    {"slice": "v5litepod-1", "api": "Cloud TPU API direct VM/Flex-start", "hosts": 1, "chips_per_host": 1, "jax_devices_per_chip_expected": 1, "hbm_gib_per_chip": 16},
    {"slice": "v5litepod-4", "api": "Cloud TPU API direct VM/Flex-start", "hosts": 1, "chips_per_host": 4, "jax_devices_per_chip_expected": 1, "hbm_gib_per_chip": 16},
    {"slice": "v5litepod-8", "api": "Cloud TPU API direct VM/Flex-start", "hosts": 1, "chips_per_host": 8, "jax_devices_per_chip_expected": 1, "hbm_gib_per_chip": 16},
    {"slice": "v6e-4", "api": "Compute Engine Cloud TPU VM", "hosts": 1, "chips_per_host": 4, "jax_devices_per_chip_expected": 1, "hbm_gib_per_chip": 32},
]
for spec in topologies:
    spec["total_chips"] = spec["hosts"] * spec["chips_per_host"]
    spec["expected_jax_devices"] = spec["total_chips"] * spec["jax_devices_per_chip_expected"]
    spec["processes"] = spec["hosts"]
    spec["total_hbm_gib"] = spec["total_chips"] * spec["hbm_gib_per_chip"]

# Check which hardware backend (`cpu`, `gpu`, or `tpu`) JAX selected for `receipt`.
receipt = verify_runtime_contract(expected_backend="tpu", expected_device_count=4)
# The CPU rehearsal must complete arithmetic while clearly reporting the TPU target as not ready.
assert receipt["arithmetic_ok"] is True
if receipt["observed_backend"] == "cpu":
    assert receipt["requested_target_ready"] is False
# Actual TPU launch/preflight commands set TPU_REQUIRE_TARGET=1 and require an exact backend/count match.
if os.environ.get("TPU_REQUIRE_TARGET") == "1":
    if receipt["source_sha256"] is None:
        raise SystemExit("TPU_REQUIRE_TARGET=1 requires a saved verifier script with a readable source identity")
    if not receipt["requested_target_ready"]:
        raise SystemExit("Requested TPU target is not ready: expected TPU backend and exactly 4 JAX devices")
print("Verified runtime receipt:", json.dumps(receipt, sort_keys=True))
print("Single-host TPU slice HBM totals (GiB):", {t["slice"]: t["total_hbm_gib"] for t in topologies})

# Step 3: Verify invariants on the completed state
receipt = verify_runtime_contract(expected_backend="tpu", expected_device_count=4)
assert receipt["arithmetic_ok"] is True

# Step 1 — Write the runtime contract and synchronized pmap verifier
import hashlib
import json
import os
import platform
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np


def verify_runtime_contract(expected_backend="tpu", expected_device_count=4):
    backend = jax.default_backend()
    devices = jax.devices()
    local_devices = jax.local_devices()
    x = jnp.arange(len(local_devices) * 8, dtype=jnp.float32).reshape(len(local_devices), 8)
    per_device = jax.pmap(lambda row: jnp.sum(row * row))(x)
    total = float(jax.block_until_ready(jnp.sum(per_device)))
    expected_total = float(np.sum(np.arange(len(local_devices) * 8, dtype=np.float64) ** 2))
    arithmetic_ok = bool(np.isclose(total, expected_total))
    expected_backend_ok = backend == expected_backend
    expected_topology_ok = len(devices) == expected_device_count
    source_value = globals().get("__file__")
    source_path = Path(source_value) if isinstance(source_value, str) and source_value else None
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path is not None and source_path.is_file() else None
    source_identity = str(source_path.resolve()) if source_sha256 is not None else "unavailable: interactive execution has no readable __file__"
    receipt = {
        "source_sha256": source_sha256,
        "source_identity": source_identity,
        "environment_lock": "projects/reproducible-experiments/requirements.txt; jax[tpu]==0.9.2",
        "python": platform.python_version(), "jax": jax.__version__,
        "expected_backend": expected_backend, "observed_backend": backend,
        "expected_device_count": expected_device_count, "device_count": len(devices),
        "process_count": jax.process_count(),
        "target_topology": {"api": "Cloud TPU API direct VM/Flex-start", "machine_type": "v5litepod-4", "physical_chips_expected": 4, "hosts_expected": 1, "topology_source": "requested Cloud TPU resource contract; physical chips are not inferred from JAX devices"},
        "local_device_count": len(local_devices), "device_kinds": [str(d.device_kind) for d in devices],
        "pmap_sum_of_squares": total, "expected_sum_of_squares": expected_total,
        "arithmetic_ok": arithmetic_ok, "expected_backend_ok": expected_backend_ok,
        "expected_topology_ok": expected_topology_ok,
        "requested_target_ready": bool(arithmetic_ok and expected_backend_ok and expected_topology_ok),
    }
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return receipt


# Step 2 — Compare single-host TPU slice topologies and emit the receipt
topologies = [
    {"slice": "v5litepod-1", "api": "Cloud TPU API direct VM/Flex-start", "hosts": 1, "chips_per_host": 1, "jax_devices_per_chip_expected": 1, "hbm_gib_per_chip": 16},
    {"slice": "v5litepod-4", "api": "Cloud TPU API direct VM/Flex-start", "hosts": 1, "chips_per_host": 4, "jax_devices_per_chip_expected": 1, "hbm_gib_per_chip": 16},
    {"slice": "v5litepod-8", "api": "Cloud TPU API direct VM/Flex-start", "hosts": 1, "chips_per_host": 8, "jax_devices_per_chip_expected": 1, "hbm_gib_per_chip": 16},
    {"slice": "v6e-4", "api": "Compute Engine Cloud TPU VM", "hosts": 1, "chips_per_host": 4, "jax_devices_per_chip_expected": 1, "hbm_gib_per_chip": 32},
]
for spec in topologies:
    spec["total_chips"] = spec["hosts"] * spec["chips_per_host"]
    spec["expected_jax_devices"] = spec["total_chips"] * spec["jax_devices_per_chip_expected"]
    spec["processes"] = spec["hosts"]
    spec["total_hbm_gib"] = spec["total_chips"] * spec["hbm_gib_per_chip"]

# Record the backend selected by this JAX process in the receipt.
receipt = verify_runtime_contract(expected_backend="tpu", expected_device_count=4)
# The CPU rehearsal must complete arithmetic while clearly reporting the TPU target as not ready.
assert receipt["arithmetic_ok"] is True
if receipt["observed_backend"] == "cpu":
    assert receipt["requested_target_ready"] is False
# Actual TPU launch/preflight commands set TPU_REQUIRE_TARGET=1 and require an exact backend/count match.
if os.environ.get("TPU_REQUIRE_TARGET") == "1":
    if receipt["source_sha256"] is None:
        raise SystemExit("TPU_REQUIRE_TARGET=1 requires a saved verifier script with a readable source identity")
    if not receipt["requested_target_ready"]:
        raise SystemExit("Requested TPU target is not ready: expected TPU backend and exactly 4 JAX devices")
print("Verified runtime receipt:", json.dumps(receipt, sort_keys=True))
print("Single-host TPU slice HBM totals (GiB):", {t["slice"]: t["total_hbm_gib"] for t in topologies})

# Figure data experiment
visual_data = {
    'kind': 'bar',
    'panels': [
        {'kind': 'bar', 'labels': [t['slice'] for t in topologies],
         'xlabel': 'selected Cloud TPU API topology', 'ylabel': 'physical chips (count)',
         'series': [{'label': 'physical TPU chips', 'y': [float(t['total_chips']) for t in topologies]}]},
        {'kind': 'bar', 'labels': [t['slice'] for t in topologies],
         'xlabel': 'selected Cloud TPU API topology', 'ylabel': 'total HBM (GiB)',
         'series': [{'label': 'HBM capacity', 'y': [float(t['total_hbm_gib']) for t in topologies]}]},
    ],
}
assert [t['total_chips'] for t in topologies] == [1, 4, 8, 4]
assert [t['total_hbm_gib'] for t in topologies] == [16, 64, 128, 128]
assert all(t['expected_jax_devices'] == t['total_chips'] for t in topologies)

# Experiment: Reject backend mismatch and insufficient topology before launch
# Backend mismatch: CPU runtime, TPU requested.
backend_mismatch = verify_runtime_contract(expected_backend="tpu", expected_device_count=4)
# Isolate topology mismatch: CPU backend matches, but only one CPU device is available.
topology_mismatch = verify_runtime_contract(expected_backend="cpu", expected_device_count=4)
print(backend_mismatch["expected_backend_ok"], backend_mismatch["expected_topology_ok"], topology_mismatch["expected_backend_ok"], topology_mismatch["expected_topology_ok"])
assert backend_mismatch["expected_backend_ok"] is False
assert topology_mismatch["expected_backend_ok"] is True and topology_mismatch["expected_topology_ok"] is False
assert backend_mismatch["requested_target_ready"] is False and topology_mismatch["requested_target_ready"] is False

# Experiment: Reject a mismatched backend or missing TPU devices immediately
# Experiment — Reject a mismatched backend or missing TPU devices immediately
mismatch = verify_runtime_contract(expected_backend="tpu", expected_device_count=4)
print("Mismatch requested_target_ready:", mismatch["requested_target_ready"], "observed:", mismatch["observed_backend"], "devices:", mismatch["device_count"])
assert mismatch["requested_target_ready"] == (jax.default_backend() == "tpu" and jax.device_count() == 4)

# Experiment: Verify single-host vs multi-host process counts
# Experiment — Verify single-host vs multi-host process counts
slice_hosts = {"v5litepod-1": 1, "v5litepod-4": 1, "v5litepod-8": 1, "v5litepod-16": 4}
needs_multi_host = {k: (v > 1) for k, v in slice_hosts.items()}
print("Requires --worker=all and jax.distributed.initialize():", needs_multi_host)
assert needs_multi_host["v5litepod-4"] is False and needs_multi_host["v5litepod-16"] is True

# Reference solution. Try the exercise before reading this.
# Exercise solution

def inspect_runtime_and_slice(verify_runtime_contract, expected_backend, expected_device_count, topologies):
    receipt = verify_runtime_contract(expected_backend=expected_backend, expected_device_count=expected_device_count)
    if not receipt["arithmetic_ok"]:
        raise AssertionError("Synchronized arithmetic rehearsal failed")
    chips = 4
    total_state_gib = 48.0
    usable_per_chip_gib = 16.0 * 0.8
    replicated_per_chip_gib = total_state_gib
    ideal_shard_per_chip_gib = total_state_gib / chips
    activation_per_chip_gib = 1.0
    print("48 GiB state/device replicated:", replicated_per_chip_gib)
    print("48 GiB state/device ideal even shard:", ideal_shard_per_chip_gib)
    print("Usable HBM/device:", usable_per_chip_gib)
    print("Ideal shard plus assumed activation:", ideal_shard_per_chip_gib + activation_per_chip_gib)
    assert replicated_per_chip_gib > usable_per_chip_gib
    assert ideal_shard_per_chip_gib <= usable_per_chip_gib
    assert ideal_shard_per_chip_gib + activation_per_chip_gib > usable_per_chip_gib
    target = next(row for row in topologies if row["slice"] == "v5litepod-4")
    selected = {
        "api": target["api"], "machine_type": target["slice"],
        "requested_processes": target["processes"],
        "observed_process_count": receipt["process_count"],
        "requested_physical_chips": target["total_chips"],
        "observed_jax_device_count": receipt["device_count"],
        "observed_backend": receipt["observed_backend"],
        "requested_target_ready": receipt["requested_target_ready"],
    }
    print("Selected target and runtime receipt:", selected)
    assert selected["requested_physical_chips"] == 4
    assert selected["requested_processes"] == 1
    assert selected["api"] == "Cloud TPU API direct VM/Flex-start"
    if expected_backend == "tpu" and receipt["observed_backend"] == "cpu":
        assert selected["requested_target_ready"] is False
    return {"receipt": receipt, "selected_target": selected}

result = inspect_runtime_and_slice(verify_runtime_contract, "tpu", 4, topologies)
assert result["receipt"]["arithmetic_ok"] is True


# Reference practice: Compare replicated and ideally sharded per-device HBM
total_state_gib = 48.0
chips = 4
usable_per_chip = 16.0 * 0.8
replicated_per_device = total_state_gib
sharded_per_device = total_state_gib / chips
activation_gib_per_device = 1.0
print("48 GiB state per-device, replicated/sharded:", replicated_per_device, sharded_per_device)
print("Usable HBM/device:", usable_per_chip, "sharded plus activation:", sharded_per_device + activation_gib_per_device)
assert replicated_per_device > usable_per_chip
assert sharded_per_device <= usable_per_chip
assert sharded_per_device + activation_gib_per_device > usable_per_chip
changed_state_gib = 52.0
changed_shard = changed_state_gib / chips
print("Changed 52 GiB ideal shard/device:", changed_shard)
assert changed_shard > usable_per_chip

# Reference practice: Verify the pmap sum-of-squares formula for 4 TPU devices
# Verify the pmap sum-of-squares formula for 4 TPU devices (Transfer / diagnosis)
four_device_expected = float(np.sum(np.arange(4 * 8, dtype=np.float64) ** 2))
print("Expected 4-device pmap sum of squares:", four_device_expected)
# Assert invariant `four_device_expected == 10416.0` holds
assert four_device_expected == 10416.0
print("PASS: tpu-01")
