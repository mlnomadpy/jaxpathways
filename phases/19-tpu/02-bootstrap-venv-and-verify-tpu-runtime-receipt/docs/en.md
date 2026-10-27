# Bootstrap .venv and verify a TPU runtime receipt

Phase 19: TPU Setup: Provisioning & Runtime Verification · about 20 minutes · CPU

This worked reference is validated on CPU. Accelerator compatibility requires a separate source review and target execution receipt. A separate runtime receipt collector is implemented in the target instructions. Bounded hardware receipts cover v5e1 launch, resume and precision checks; four single-device Pallas cases; and a four-device v5e collective fixture. Each receipt applies only to its recorded source, environment, cases and hardware. Other lessons and TPU configurations require separate qualification.

**Before this lesson:** [Select a TPU slice and provision your Cloud TPU VM](https://www.tahabouhsine.com/jaxpathways/lesson-tpu-01.html).

## What you will be able to do

- Bootstrap `~/jax-tpu-lab/.venv` on a Cloud TPU VM and install `jax[tpu]` from the official `libtpu` wheel index.
- Verify `jax.default_backend()`, `jax.device_count()`, and a synchronized `jax.pmap` reduction across 1, 4, or 8 chips.
- Copy the JSON runtime receipt back to your workstation with `gcloud compute tpus tpu-vm scp`.
- Delete the named TPU VM or its owning queued resource and verify those named resources are absent; unrelated TPU resources may remain.

## The problem

After `gcloud compute tpus tpu-vm create` reports `READY`, you still cannot trust a benchmark or training job until Python inside your TPU VM's `.venv` proves it can load `libtpu`, see all attached TPU chips, execute a synchronized collective across them, and tear down cleanly so nothing keeps billing.

## The idea

Bootstrap an isolated `~/jax-tpu-lab/.venv` on the TPU VM Linux host with `jax[tpu]`, run a deterministic `jax.pmap` runtime receipt across all local chips, copy the receipt back to your laptop with `gcloud compute tpus tpu-vm scp`, and delete the TPU VM (`tpu-vm delete` + `tpu-vm list`) whenever your session ends.

## Why a synchronized `pmap` receipt is stronger than `import jax`

Importing `jax` on a Cloud TPU VM only proves that the Python package is installed on the Linux host. To prove that `libtpu` can initialize the attached TPU chips, allocate High Bandwidth Memory (HBM), compile an XLA HLO computation, and synchronize results back to the host, you should run a small `jax.pmap` reduction across `jax.local_devices()` and call `jax.block_until_ready`.

If each device $d \in \{0, \dots, D-1\}$ receives a slice of $8$ consecutive integers starting at $8d$, then across $D$ devices the global vector contains $0, 1, \dots, 8D - 1$, and the exact sum of squares is:
$$

S(D) = \sum_{k=0}^{8D - 1} k^2 = \frac{(8D - 1)(8D)(16D - 1)}{6}

$$
The formula gives an expected value for a chosen device count. Only a receipt that records `observed_backend`, `device_count`, and the synchronized result can support a runtime claim. The four-row figure below is a NumPy calculation, not evidence that four TPU devices ran.

$$
S(D) = \sum_{k=0}^{8D-1} k^2 = \frac{(8D-1)(8D)(16D-1)}{6}, \quad D=\text{observed devices}
$$

### Pause and reason

What happens if you close your SSH terminal without running `gcloud compute tpus tpu-vm delete`?

<details><summary>Compare your reasoning</summary>

The Cloud TPU VM stays allocated in the `READY` state and continues billing every hour even when no Python process or SSH session is active. Always delete the TPU VM and verify with `gcloud compute tpus tpu-vm list`.

</details>

## 1. Bootstrap `~/jax-tpu-lab/.venv` and install `jax[tpu]` over SSH

Install the pinned course environment from the reviewed requirements file. On TPU, select the matching TPU extra version explicitly; do not upgrade package versions during qualification.

**Create ~/jax-tpu-lab/.venv and install jax[tpu] on the Cloud TPU VM**

```bash
# Install the pinned environment on the Cloud TPU VM
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command="mkdir -p ~/jax-tpu-lab"
gcloud compute tpus tpu-vm scp projects/reproducible-experiments/requirements.txt "$TPU_NAME":~/jax-tpu-lab/ --zone="$ZONE"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command='set -euo pipefail; mkdir -p ~/jax-tpu-lab; python3 -m venv ~/jax-tpu-lab/.venv; ~/jax-tpu-lab/.venv/bin/python -m pip install -r ~/jax-tpu-lab/requirements.txt; ~/jax-tpu-lab/.venv/bin/python -m pip install \"jax[tpu]==0.9.2\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'
```

**Expected:** Uses the pinned requirements file and jax[tpu]==0.9.2; record both source and environment identity.

## 2. Run the runtime receipt script, copy the JSON receipt back, and delete the TPU VM

Run on the requested target and evaluate arithmetic, expected backend, and exact device-count checks separately. `verified` is not a single arithmetic-only flag. This fixture requests the TPU backend and 4 devices; on CPU, arithmetic can pass while target readiness is false.

**Run the runtime receipt verifier on the TPU VM and copy the receipt home**

```bash
# Run the runtime receipt verifier on the TPU VM and copy the receipt home
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command="mkdir -p ~/jax-tpu-lab"
gcloud compute tpus tpu-vm scp public/exercises/tpu-04.py "$TPU_NAME":~/jax-tpu-lab/ --zone="$ZONE"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
  --command="set -euo pipefail; TPU_REQUIRE_TARGET=1 JAX_PLATFORMS=tpu ~/jax-tpu-lab/.venv/bin/python ~/jax-tpu-lab/tpu-04.py | tee ~/jax-tpu-lab/tpu-receipt.txt"
gcloud compute tpus tpu-vm scp "$TPU_NAME":~/jax-tpu-lab/tpu-receipt.txt ./ --zone="$ZONE"
```

**Expected:** The receipt records source SHA256, pinned environment identity, expected and observed backend, expected and observed device count, arithmetic_ok, expected_backend_ok, expected_topology_ok, and requested_target_ready. On CPU, arithmetic may pass but target readiness must be false.

**Delete this route’s TPU resources and verify their named absence**

```bash
# Flex-start: deleting the owning request with --force deletes its allocated VM too.
if [ -n "${QUEUED_RESOURCE_ID:-}" ]; then
  gcloud alpha compute tpus queued-resources delete "$QUEUED_RESOURCE_ID" --project="$PROJECT_ID" --zone="$ZONE" --force --quiet
else
  gcloud compute tpus tpu-vm delete "$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE" --quiet
fi
remaining_vm=$(gcloud compute tpus tpu-vm list --project="$PROJECT_ID" --zone="$ZONE" --filter="name=$TPU_NAME" --format="value(name)")
[ -z "$remaining_vm" ] || { echo "Named TPU VM still exists: $remaining_vm"; exit 1; }
if [ -n "${QUEUED_RESOURCE_ID:-}" ]; then
  remaining_request=$(gcloud alpha compute tpus queued-resources list --project="$PROJECT_ID" --zone="$ZONE" --filter="name=$QUEUED_RESOURCE_ID" --format="value(name)")
  [ -z "$remaining_request" ] || { echo "Owning queued resource still exists: $remaining_request"; exit 1; }
fi
printf "Named route resources are absent; unrelated resources may remain.\n"
```

**Expected:** The named VM and, when used, its owning queued resource are absent. The filtered checks allow unrelated resources to remain.

## Define the closed-form sum-of-squares oracle and pmap receipt builder

Create main.py with closed_form_sum_of_squares and build_runtime_receipt.

```python
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
```

Each local device computes the sum of squares of its 8-element row via jax.pmap, and the host verifies the synchronized total against the exact polynomial identity.

## Verify the active runtime receipt and tabulate multi-chip expectations

Append slice_expectations and the receipt verification assertions to main.py.

```python
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
```

Knowing the exact expected sum for 1, 4, and 8 devices lets you immediately spot if a 4-chip or 8-chip TPU VM only initialized a single device.

## Step 3: Verify invariants on the completed state

Run the final shape and numerical assertions to confirm the state built in Steps 1 and 2.

```python
receipt = build_runtime_receipt(expected_backend="tpu", expected_device_count=4)
assert receipt["arithmetic_ok"] is True
```

Checking these invariants confirms the computation is ready for the full worked experiment.

## Run the example

```python
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
```

Expected: Prints the expected sum table and a receipt with separate arithmetic and requested-target status. The CPU rehearsal can verify the arithmetic but is not TPU readiness.

## Calculated expectation for four devices (not an observed runtime receipt)

**Predict:** If four devices each sum a distinct block of eight integers, what cumulative sum of squares should the verifier report?

![Calculated expectation for four devices (not an observed runtime receipt)](../outputs/figure.svg)

**Analytic calculation; no device result recorded**

### Read the figure

This analytic illustration computes four eight-value row sums with NumPy. It does not read a TPU execution receipt and does not show observed device participation.

### Connect it to the computation

The calculated total is 10416. A runtime claim must instead use the device count and per-device values in a saved receipt; the current local receipt records one CPU device.

```python
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
```

## Recorded reference execution

CPU run: 2026-10-27T15:46:40.950880+00:00. JAX 0.9.2.

```text
Runtime receipt: {"arithmetic_ok": true, "device_count": 1, "environment_lock": "projects/reproducible-experiments/requirements.txt; jax[tpu]==0.9.2", "expected_backend": "tpu", "expected_backend_ok": false, "expected_device_count": 4, "expected_topology_ok": false, "expected_total": 140.0, "jax": "0.9.2", "local_device_count": 1, "observed_backend": "cpu", "observed_total": 140.0, "per_device_sums": [140.0], "process_count": 1, "python": "3.14.3", "receipt_sha256": "64561213664a634b", "requested_target_ready": false, "source_identity": "unavailable: interactive execution has no readable __file__", "source_sha256": null}
Expected pmap sums by slice: {'1-device (CPU / v5e-1)': 140.0, '4-device (v5litepod-4 / v6e-4)': 10416.0, '8-device (v5litepod-8 / v6e-8)': 85344.0}
Runtime receipt: {"arithmetic_ok": true, "device_count": 1, "environment_lock": "projects/reproducible-experiments/requirements.txt; jax[tpu]==0.9.2", "expected_backend": "tpu", "expected_backend_ok": false, "expected_device_count": 4, "expected_topology_ok": false, "expected_total": 140.0, "jax": "0.9.2", "local_device_count": 1, "observed_backend": "cpu", "observed_total": 140.0, "per_device_sums": [140.0], "process_count": 1, "python": "3.14.3", "receipt_sha256": "64561213664a634b", "requested_target_ready": false, "source_identity": "unavailable: interactive execution has no readable __file__", "source_sha256": null}
Expected pmap sums by slice: {'1-device (CPU / v5e-1)': 140.0, '4-device (v5litepod-4 / v6e-4)': 10416.0, '8-device (v5litepod-8 / v6e-8)': 85344.0}
Per-chip sums on a 4-chip slice: [140.0, 1100.0, 3084.0, 6092.0] Total: 10416.0
Named-resource cleanup checks passed; unrelated nodes may remain.
4-device expected sum: 10416.0 8-device expected sum: 85344.0 receipt hash: 64561213664a634b
Requested four-device TPU readiness: False ; simulated wrong-count contract rejected: True
Real runtime receipt qualifies as four-device TPU: False
Overnight idle cost avoided (USD): 74.4
PASS: tpu-04

```

## Compare per-device row sums on a 4-chip TPU slice

**Predict before running:** On a 4-chip TPU VM (`v5litepod-4`), why does chip 3 produce a much larger row sum than chip 0?

```python
# Experiment — Compare per-device row sums on a 4-chip TPU slice
x4 = jnp.arange(4 * 8, dtype=jnp.float32).reshape(4, 8)
# Reduce along axis=1 to compute `row_sums_4`.
row_sums_4 = [float(v) for v in jnp.sum(x4 * x4, axis=1)]
print("Per-chip sums on a 4-chip slice:", row_sums_4, "Total:", sum(row_sums_4))
assert row_sums_4 == [140.0, 1100.0, 3084.0, 6092.0] and sum(row_sums_4) == 10416.0
```

**Expected:** Chip 0 sums 0^2..7^2 = 140.0, chip 1 sums 8^2..15^2 = 1100.0, chip 2 sums 16^2..23^2 = 3084.0, and chip 3 sums 24^2..31^2 = 6092.0 (total 10416.0).

Inspecting `per_device_sums` confirms that distinct shards of `x` were placed and squared on each of the 4 chips.

## Verify absence of named route-owned resources

**Predict before running:** Can unrelated TPU nodes remain while this lesson’s route-owned VM and queued request have both been removed?

```python
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
```

**Expected:** The cleanup verifier accepts remaining unrelated TPU VMs, rejects a still-present target VM, and checks the owning queued request when one was used.

Cleanup must prove absence of the resources created by this route. Requiring an empty zone could encourage deleting unrelated user resources.

## Make it yours

Compute the 4-device and 8-device oracle values. On the available runtime, assert arithmetic_ok; assert requested_target_ready only when observed_backend is tpu and device_count is exactly 4.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Inputs:**
- The closed-form sum helper and the active CPU/TPU receipt with expected and observed backend/device counts.

**Outputs:**
- Calculated four-/eight-device oracle values, actual arithmetic result and requested-target readiness.

**Invariants to check:**
- arithmetic_ok is separate from expected_backend_ok and expected_topology_ok.
- requested_target_ready requires the requested TPU backend and exact expected count; a CPU rehearsal does not qualify TPU hardware.

**Hints:**
- Use observed_backend and device_count from the actual receipt.
- Test a schema-only wrong-count control separately from real hardware observations.

**Starter code scaffold (fill in the TODOs):**

```python
def audit_runtime_sum_receipt(receipt, closed_form_sum_of_squares):
    # TODO: compare both theoretical totals and assert readiness matches the requested backend/count contract.
    raise NotImplementedError
```

<details><summary>Reference solution</summary>

```python
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

```

</details>

## Detect a silent fallback from 4 TPU chips to 1 CPU device

**Foundations**

Write `is_four_chip_tpu_receipt(r)` using the real receipt schema: return True only when `r['observed_backend'] == 'tpu'`, `r['local_device_count'] == 4`, `r['device_count'] == 4`, and `r['observed_total']` is 10416.0. Missing keys should return False.

<details><summary>Hint</summary>

Read backend from `observed_backend`, not `backend`; test the local count, global JAX device count, and synchronized sum independently. Handle missing fields as a failed check.

</details>

### How to write: Detect a silent fallback from 4 TPU chips to 1 CPU device — Step-by-step recipe & starter scaffold

**Inputs:**
- A candidate runtime receipt mapping with backend, local device count, and observed total fields.

**Outputs:**
- A Boolean predicate for the exact four-chip TPU receipt contract.

**Invariants to check:**
- Use observed_backend from the emitted receipt.
- Missing keys, CPU backend, wrong local/global device count, or wrong synchronized total return False.

**Hints:**
- Validate backend, count, and total independently.
- Use a tolerance only for the floating representation of the exact integer total.

**Starter code scaffold (fill in the TODOs):**

```python
import numpy as np

def is_four_chip_tpu_receipt(receipt):
    try:
        # Use the schema returned by build_runtime_receipt.
        return bool(
            receipt["observed_backend"] == "tpu"
            and receipt["local_device_count"] == 4
            and receipt["device_count"] == 4
            and np.isclose(receipt["observed_total"], 10416.0)
        )
    except (KeyError, TypeError):
        return False
```

<details><summary>Reference solution and reasoning</summary>

```python
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

```

The predicate uses fields actually emitted by build_runtime_receipt. It rejects missing fields, CPU fallback, and either local or global device-count mismatch.

</details>

## Estimate hourly billing savings from immediate teardown

**Transfer / diagnosis**

If a 4-chip TPU slice costs $4.80$ USD/hour and verification plus experiment takes $0.5$ hours, compute how much you save by deleting the VM immediately instead of keeping it running for $16$ total hours, including the active time.

<details><summary>Hint</summary>

Compare `16.0 * 4.80` against `0.5 * 4.80`.

</details>

### How to write: Estimate hourly billing savings from immediate teardown — Step-by-step recipe & starter scaffold

**Inputs:**
- Hourly slice cost, active verification duration of 0.5 hours, and total elapsed VM time of 16 hours.

**Outputs:**
- Avoided idle duration and the corresponding teardown savings.

**Invariants to check:**
- Total elapsed time includes the active work, so idle time is total time minus active time; immediate teardown and the overnight choice share the active-time cost.

**Hints:**
- Subtract active hours from total hours to get avoidable idle hours.
- Multiply only those idle hours by the hourly rate; 16 idle hours would be $76.80, while 16 total hours including 0.5 active hour yields $74.40.

**Starter code scaffold (fill in the TODOs):**

```python
def idle_teardown_savings(hourly_cost, active_hours, total_hours):
    # TODO: compute avoidable idle hours and savings when total_hours includes active work.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Estimate hourly billing savings from immediate teardown (Transfer / diagnosis)
hourly_rate = 4.80
saved_usd = round((16.0 - 0.5) * hourly_rate, 2)
print("Overnight idle cost avoided (USD):", saved_usd)
# Assert invariant `saved_usd == 74.4` holds
assert saved_usd == 74.4
```

Deleting the Cloud TPU VM immediately after copying artifacts back avoids paying for 15.5 idle hours.

</details>

## Check your understanding

Why does `build_runtime_receipt()` run `jax.pmap` across `jax.local_devices()` and call `jax.block_until_ready` instead of only checking `jax.default_backend()`?

1. Because `jax.default_backend()` deletes the virtual environment
2. To exercise and synchronize a small computation on the visible local devices, beyond merely enumerating them
3. Because `jax.pmap` automatically deletes the Cloud TPU VM

<details><summary>Answer and explanation</summary>

**1. Because `jax.default_backend()` deletes the virtual environment**

Backend discovery does not alter the environment. The extra computation tests a small operation on the visible local devices.

**2. To exercise and synchronize a small computation on the visible local devices, beyond merely enumerating them (correct)**

With the backend explicitly checked, mapped arithmetic and an independent expected sum exercise this small local-device contract. This does not qualify all chips on other hosts, the full model, precision paths or performance.

**3. Because `jax.pmap` automatically deletes the Cloud TPU VM**

pmap executes a mapped function; it does not manage cloud resources. Cleanup is a separate control-plane operation.

</details>

## Diagnose the result

If `pip install 'jax[tpu]'` fails to find `libtpu`, verify that `-f https://storage.googleapis.com/jax-releases/libtpu_releases.html` was passed to `pip install` inside the activated `~/jax-tpu-lab/.venv` on the Linux TPU VM.

## Carry forward

- Always install `jax[tpu]` inside an isolated `~/jax-tpu-lab/.venv` on the TPU VM host.
- Verify `backend`, `local_device_count`, and a synchronized `jax.pmap` sum-of-squares receipt before starting longer experiments.
- Copy your receipt back with `gcloud compute tpus tpu-vm scp`, run `tpu-vm delete`, and confirm `tpu-vm list` is empty.

## Keep your evidence

Keep arithmetic_ok, expected_backend_ok, expected_topology_ok, requested_target_ready, expected/observed backend and counts, source identity or its explicit absence, observed/expected reduction and receipt digest. Separate calculated 4-/8-device expectations from observed execution and require the gated target preflight to fail closed.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX installation for Cloud TPU](https://docs.jax.dev/en/latest/installation.html)
- [Manage Cloud TPU VMs with gcloud](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm)

