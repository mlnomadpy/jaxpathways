"""Select a TPU slice and provision your Cloud TPU VM: worked experiments and reference solutions. CPU checks."""

# Write the runtime contract and synchronized pmap verifier
# Step 1 — Write the runtime contract and synchronized pmap verifier: The verifier checks both device discovery and a synchronized pmap...
# Import hashlib for this computation.
import hashlib
import json
import os
import platform
import jax
import jax.numpy as jnp
import numpy as np


# Function `verify_runtime_contract(expected_backend, min_devices)` implementing this stage's computation:
def verify_runtime_contract(expected_backend=None, min_devices=1):
    # Check which hardware backend (`cpu`, `gpu`, or `tpu`) JAX selected for `backend`.
    backend = jax.default_backend()
    # Query the active JAX devices into `devices`.
    devices = jax.devices()
    # Query the active JAX devices into `local_devices`.
    local_devices = jax.local_devices()
    # Compute `target` from `expected_backend or backend`
    target = expected_backend or backend
    # Create evenly spaced index values in `x`.
    x = jnp.arange(len(local_devices) * 8, dtype=jnp.float32).reshape(len(local_devices), 8)
    # Reduce across the target axis to summarize `per_device`.
    per_device = jax.pmap(lambda row: jnp.sum(row * row))(x)
    # Synchronize host execution until asynchronous device computation completes.
    total = float(jax.block_until_ready(jnp.sum(per_device)))
    # Create evenly spaced index values in `expected_total`.
    expected_total = float(np.sum(np.arange(len(local_devices) * 8, dtype=np.float64) ** 2))
    # Evaluate the compound expression for `contract_ok`.
    contract_ok = (
        backend == target
        and len(devices) >= min_devices
        and np.isclose(total, expected_total)
    )
    # Construct dictionary `receipt` with the structured fields for this stage.
    receipt = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "expected_backend": target,
        "observed_backend": backend,
        "device_count": len(devices),
        "local_device_count": len(local_devices),
        "device_kinds": [str(d.device_kind) for d in devices],
        "pmap_sum_of_squares": total,
        "expected_sum_of_squares": expected_total,
        "contract_ok": bool(contract_ok),
    }
    # Read or serialize artifact data on disk (`receipt['receipt_sha256']`).
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    # Return `receipt` to the caller.
    return receipt

# Compare single-host TPU slice topologies and emit the receipt
# Step 2 — Compare single-host TPU slice topologies and emit the receipt: Calculating total chips and HBM per slice makes the hardware...
topologies = [
    {"slice": "v5litepod-1", "hosts": 1, "chips_per_host": 1, "hbm_gib_per_chip": 16},
    {"slice": "v5litepod-4", "hosts": 1, "chips_per_host": 4, "hbm_gib_per_chip": 16},
    {"slice": "v5litepod-8", "hosts": 1, "chips_per_host": 8, "hbm_gib_per_chip": 16},
    {"slice": "v6e-4", "hosts": 1, "chips_per_host": 4, "hbm_gib_per_chip": 32},
]
# Iterate over `spec` to step through the computation:
for spec in topologies:
    # Compute `spec["total_chips"]` from `spec["hosts"] * spec["chips_per_host"]`
    spec["total_chips"] = spec["hosts"] * spec["chips_per_host"]
    # Compute `spec["total_hbm_gib"]` from `spec["total_chips"] * spec["hbm_gib_per_chip"]`
    spec["total_hbm_gib"] = spec["total_chips"] * spec["hbm_gib_per_chip"]

# Check which hardware backend (`cpu`, `gpu`, or `tpu`) JAX selected for `receipt`.
receipt = verify_runtime_contract(expected_backend=jax.default_backend(), min_devices=1)
# Assert invariant `receipt["contract_ok"] is True` holds
assert receipt["contract_ok"] is True
# Print the observed values to compare against the expected result.
print("Verified runtime receipt:", json.dumps(receipt, sort_keys=True))
# Print diagnostic summary of the computed outputs.
print("Single-host TPU slice HBM totals (GiB):", {t["slice"]: t["total_hbm_gib"] for t in topologies})

# Step 3: Verify invariants on the completed state
receipt = verify_runtime_contract(expected_backend=jax.default_backend(), min_devices=1)
assert receipt["contract_ok"] is True

# Step 1 — Write the runtime contract and synchronized pmap verifier: The verifier checks both device discovery and a synchronized pmap...
# Import hashlib for this computation.
import hashlib
import json
import os
import platform
import jax
import jax.numpy as jnp
import numpy as np


# Function `verify_runtime_contract(expected_backend, min_devices)` implementing this stage's computation:
def verify_runtime_contract(expected_backend=None, min_devices=1):
    # Check which hardware backend (`cpu`, `gpu`, or `tpu`) JAX selected for `backend`.
    backend = jax.default_backend()
    # Query the active JAX devices into `devices`.
    devices = jax.devices()
    # Query the active JAX devices into `local_devices`.
    local_devices = jax.local_devices()
    # Compute `target` from `expected_backend or backend`
    target = expected_backend or backend
    # Create evenly spaced index values in `x`.
    x = jnp.arange(len(local_devices) * 8, dtype=jnp.float32).reshape(len(local_devices), 8)
    # Reduce across the target axis to summarize `per_device`.
    per_device = jax.pmap(lambda row: jnp.sum(row * row))(x)
    # Synchronize host execution until asynchronous device computation completes.
    total = float(jax.block_until_ready(jnp.sum(per_device)))
    # Create evenly spaced index values in `expected_total`.
    expected_total = float(np.sum(np.arange(len(local_devices) * 8, dtype=np.float64) ** 2))
    # Evaluate the compound expression for `contract_ok`.
    contract_ok = (
        backend == target
        and len(devices) >= min_devices
        and np.isclose(total, expected_total)
    )
    # Construct dictionary `receipt` with the structured fields for this stage.
    receipt = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "expected_backend": target,
        "observed_backend": backend,
        "device_count": len(devices),
        "local_device_count": len(local_devices),
        "device_kinds": [str(d.device_kind) for d in devices],
        "pmap_sum_of_squares": total,
        "expected_sum_of_squares": expected_total,
        "contract_ok": bool(contract_ok),
    }
    # Read or serialize artifact data on disk (`receipt['receipt_sha256']`).
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    # Return `receipt` to the caller.
    return receipt


# Step 2 — Compare single-host TPU slice topologies and emit the receipt: Calculating total chips and HBM per slice makes the hardware...
topologies = [
    {"slice": "v5litepod-1", "hosts": 1, "chips_per_host": 1, "hbm_gib_per_chip": 16},
    {"slice": "v5litepod-4", "hosts": 1, "chips_per_host": 4, "hbm_gib_per_chip": 16},
    {"slice": "v5litepod-8", "hosts": 1, "chips_per_host": 8, "hbm_gib_per_chip": 16},
    {"slice": "v6e-4", "hosts": 1, "chips_per_host": 4, "hbm_gib_per_chip": 32},
]
# Iterate over `spec` to step through the computation:
for spec in topologies:
    # Compute `spec["total_chips"]` from `spec["hosts"] * spec["chips_per_host"]`
    spec["total_chips"] = spec["hosts"] * spec["chips_per_host"]
    # Compute `spec["total_hbm_gib"]` from `spec["total_chips"] * spec["hbm_gib_per_chip"]`
    spec["total_hbm_gib"] = spec["total_chips"] * spec["hbm_gib_per_chip"]

# Check which hardware backend (`cpu`, `gpu`, or `tpu`) JAX selected for `receipt`.
receipt = verify_runtime_contract(expected_backend=jax.default_backend(), min_devices=1)
# Assert invariant `receipt["contract_ok"] is True` holds
assert receipt["contract_ok"] is True
# Print the observed values to compare against the expected result.
print("Verified runtime receipt:", json.dumps(receipt, sort_keys=True))
# Print diagnostic summary of the computed outputs.
print("Single-host TPU slice HBM totals (GiB):", {t["slice"]: t["total_hbm_gib"] for t in topologies})

# Figure data experiment
# Compute figure data for: Single-host Cloud TPU slice chip counts and total HBM capacity
# Construct dictionary `visual_data` with the structured fields for this stage.
visual_data = {
    'kind': 'bar',
    'labels': [t['slice'] for t in topologies],
    'xlabel': 'single-host Cloud TPU slice',
    'ylabel': 'count / capacity',
    'series': [
        {'label': 'TPU chips on host', 'y': [float(t['total_chips']) for t in topologies]},
        {'label': 'total HBM (GiB)', 'y': [float(t['total_hbm_gib']) for t in topologies]},
    ],
}

# Experiment: Reject a mismatched backend or missing TPU devices immediately
# Experiment — Reject a mismatched backend or missing TPU devices immediately: Failing the contract check explicitly prevents a script from...
mismatch = verify_runtime_contract(expected_backend="tpu", min_devices=4)
# Print the observed values to compare against the expected result.
print("Mismatch contract_ok:", mismatch["contract_ok"], "observed:", mismatch["observed_backend"], "devices:", mismatch["device_count"])
# Assert that `mismatch["contract_ok"] == (jax.default_backend() == "tpu" and jax.device_count() >= 4)`.
assert mismatch["contract_ok"] == (jax.default_backend() == "tpu" and jax.device_count() >= 4)

# Experiment: Verify single-host vs multi-host process counts
# Experiment — Verify single-host vs multi-host process counts: Single-host TPU VMs attach all chips to one Linux VM; multi-host...
slice_hosts = {"v5litepod-1": 1, "v5litepod-4": 1, "v5litepod-8": 1, "v5litepod-16": 4}
# Compute `needs_multi_host` from `{k: (v > 1) for k, v in slice_hosts.items()}`
needs_multi_host = {k: (v > 1) for k, v in slice_hosts.items()}
# Print the observed values to compare against the expected result.
print("Requires --worker=all and jax.distributed.initialize():", needs_multi_host)
# Assert that `needs_multi_host["v5litepod-4"] is False and needs_multi_host["v5litepod-16"] is True`.
assert needs_multi_host["v5litepod-4"] is False and needs_multi_host["v5litepod-16"] is True

# Reference solution. Try the exercise before reading this.
# Exercise solution: Call verify_runtime_contract for the active backend with...
checked = verify_runtime_contract(expected_backend=jax.default_backend(), min_devices=1)
# Run `next` to compute `v5e4`.
v5e4 = next(t for t in topologies if t["slice"] == "v5litepod-4")
# Print the observed values to compare against the expected result.
print("Checked receipt sha256:", checked["receipt_sha256"], "v5litepod-4 chips:", v5e4["total_chips"], "HBM GiB:", v5e4["total_hbm_gib"])
# Assert that `checked["contract_ok"] is True and v5e4["total_chips"] == 4 and v5e4["total_hbm_gib"] == 64`.
assert checked["contract_ok"] is True and v5e4["total_chips"] == 4 and v5e4["total_hbm_gib"] == 64

# Reference practice: Check whether a model state fits on a single-host TPU slice
# Check whether a model state fits on a single-host TPU slice (Foundations): A single v5litepod-1 chip has 16 GiB HBM, whereas...
fitting_slices = [t["slice"] for t in topologies if t["total_hbm_gib"] >= 48]
# Print the observed values to compare against the expected result.
print("Slices fitting 48 GiB:", fitting_slices)
# Assert invariant `fitting_slices == ["v5litepod-4"` holds
assert fitting_slices == ["v5litepod-4", "v5litepod-8", "v6e-4"]

# Reference practice: Verify the pmap sum-of-squares formula for 4 TPU devices
# Verify the pmap sum-of-squares formula for 4 TPU devices (Transfer / diagnosis): On a 4-chip TPU VM (v5litepod-4 or v6e-4), x has shape (4,...
# Aggregate array values to compute `four_device_expected`.
four_device_expected = float(np.sum(np.arange(4 * 8, dtype=np.float64) ** 2))
# Print the observed values to compare against the expected result.
print("Expected 4-device pmap sum of squares:", four_device_expected)
# Assert invariant `four_device_expected == 10416.0` holds
assert four_device_expected == 10416.0
print("PASS: tpu-01")
