# Bootstrap .venv and verify a TPU runtime receipt

Phase 19: TPU Setup: Provisioning & Runtime Verification · about 20 minutes · CPU

This worked reference is validated on CPU. Accelerator compatibility requires a separate source review and target execution receipt. A separate runtime receipt collector is implemented in the target instructions. Bounded hardware receipts cover v5e1 launch, resume and precision checks; four single-device Pallas cases; and a four-device v5e collective fixture. Each receipt applies only to its recorded source, environment, cases and hardware. Other lessons and TPU configurations require separate qualification.

## What you will be able to do

- Bootstrap `~/jax-tpu-lab/.venv` on a Cloud TPU VM and install `jax[tpu]` from the official `libtpu` wheel index.
- Verify `jax.default_backend()`, `jax.device_count()`, and a synchronized `jax.pmap` reduction across 1, 4, or 8 chips.
- Copy the JSON runtime receipt back to your workstation with `gcloud compute tpus tpu-vm scp`.
- Delete the Cloud TPU VM and confirm with `gcloud compute tpus tpu-vm list` that zero billable TPU nodes remain.

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
M_{\text{state}} = N_{\text{params}} \times (\text{bytes}_{\text{param}} + \text{bytes}_{\text{grad}} + 2 \cdot \text{bytes}_{\text{moment}})
$$

### Pause and reason

What happens if you close your SSH terminal without running `gcloud compute tpus tpu-vm delete`?

<details><summary>Compare your reasoning</summary>

The Cloud TPU VM stays allocated in the `READY` state and continues billing every hour even when no Python process or SSH session is active. Always delete the TPU VM and verify with `gcloud compute tpus tpu-vm list`.

</details>

## 1. Bootstrap `~/jax-tpu-lab/.venv` and install `jax[tpu]` over SSH

Use `gcloud compute tpus tpu-vm ssh` to create a dedicated virtual environment on the TPU VM host and install `jax[tpu]` plus the course libraries (`flax`, `optax`, `orbax-checkpoint`, `numpy`, `matplotlib`).

**Create ~/jax-tpu-lab/.venv and install jax[tpu] on the Cloud TPU VM**

```bash
# Create ~/jax-tpu-lab/.venv and install jax[tpu] on the Cloud TPU VM
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command="
  set -euo pipefail
  mkdir -p ~/jax-tpu-lab
  python3 -m venv ~/jax-tpu-lab/.venv
  source ~/jax-tpu-lab/.venv/bin/activate
  pip install -U pip
  pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  pip install flax optax orbax-checkpoint numpy matplotlib
"
```

**Expected:** Installs jax[tpu], libtpu, and the course dependencies inside ~/jax-tpu-lab/.venv on the TPU VM.

## 2. Run the runtime receipt script, copy the JSON receipt back, and delete the TPU VM

Stage `public/exercises/tpu-04.py` to the TPU VM, run it inside `~/jax-tpu-lab/.venv`, copy the saved receipt back to your workstation, and delete the TPU VM when finished.

**Run the runtime receipt verifier on the TPU VM and copy the receipt home**

```bash
# Run the runtime receipt verifier on the TPU VM and copy the receipt home
gcloud compute tpus tpu-vm scp public/exercises/tpu-04.py "$TPU_NAME":~/jax-tpu-lab/ --zone="$ZONE"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
  --command="JAX_PLATFORMS=tpu ~/jax-tpu-lab/.venv/bin/python ~/jax-tpu-lab/tpu-04.py | tee ~/jax-tpu-lab/tpu-receipt.txt"
gcloud compute tpus tpu-vm scp "$TPU_NAME":~/jax-tpu-lab/tpu-receipt.txt ./ --zone="$ZONE"
```

**Expected:** Runs tpu-04.py on the TPU VM, writes tpu-receipt.txt, and downloads it to your current directory.

**Delete the Cloud TPU VM and verify zero active TPU resources remain**

```bash
# Delete the Cloud TPU VM and verify zero active TPU resources remain
gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$ZONE" --quiet
gcloud compute tpus tpu-vm list --zone="$ZONE"
```

**Expected:** Deletes the TPU VM and prints Listed 0 items.

## Define the closed-form sum-of-squares oracle and pmap receipt builder

Create main.py with closed_form_sum_of_squares and build_runtime_receipt.

```python
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
```

Each local device computes the sum of squares of its 8-element row via jax.pmap, and the host verifies the synchronized total against the exact polynomial identity.

## Verify the active runtime receipt and tabulate multi-chip expectations

Append slice_expectations and the receipt verification assertions to main.py.

```python
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
```

Knowing the exact expected sum for 1, 4, and 8 devices lets you immediately spot if a 4-chip or 8-chip TPU VM only initialized a single device.

## Step 3: Verify invariants on the completed state

Run the final shape and numerical assertions to confirm the state built in Steps 1 and 2.

```python
receipt = build_runtime_receipt()
assert receipt["verified"] is True
```

Checking these invariants confirms the computation is ready for the full worked experiment.

## Run the example

```python
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
```

Expected: Prints the verified JSON runtime receipt (observed_total: 140.0 on 1 device) and the expected pmap sums for 1-device (140.0), 4-device (10416.0), and 8-device (85344.0) slices.

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

CPU run: 2026-10-21T16:11:43.731071+00:00. JAX 0.9.2.

```text
Runtime receipt: {"backend": "cpu", "expected_total": 140.0, "jax": "0.9.2", "local_device_count": 1, "observed_total": 140.0, "per_device_sums": [140.0], "python": "3.14.3", "receipt_sha256": "8c6bd0cbe29edc78", "verified": true}
Expected pmap sums by slice: {'1-device (CPU / v5e-1)': 140.0, '4-device (v5litepod-4 / v6e-4)': 10416.0, '8-device (v5litepod-8 / v6e-8)': 85344.0}
Runtime receipt: {"backend": "cpu", "expected_total": 140.0, "jax": "0.9.2", "local_device_count": 1, "observed_total": 140.0, "per_device_sums": [140.0], "python": "3.14.3", "receipt_sha256": "8c6bd0cbe29edc78", "verified": true}
Expected pmap sums by slice: {'1-device (CPU / v5e-1)': 140.0, '4-device (v5litepod-4 / v6e-4)': 10416.0, '8-device (v5litepod-8 / v6e-8)': 85344.0}
Per-chip sums on a 4-chip slice: [140.0, 1100.0, 3084.0, 6092.0] Total: 10416.0
Teardown verifier passed for empty and non-empty gcloud list outputs.
4-chip expected sum: 10416.0 8-chip expected sum: 85344.0 receipt hash: 8c6bd0cbe29edc78
Simulated 4-chip TPU check: True Local CPU check: False
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

## Verify teardown confirmation logic

**Predict before running:** How can a helper script check that `gcloud compute tpus tpu-vm list` returned zero active VMs?

```python
# Experiment — Verify teardown confirmation logic
def is_tpu_zone_clean(gcloud_list_json):
    items = json.loads(gcloud_list_json)
    return len(items) == 0

# Assert invariant `is_tpu_zone_clean("[]") is True` holds
assert is_tpu_zone_clean("[]") is True
# Assert invariant `is_tpu_zone_clean('[{"name": "jax-tpu-course"` holds
assert is_tpu_zone_clean('[{"name": "jax-tpu-course", "state": "READY"}]') is False
print("Teardown verifier passed for empty and non-empty gcloud list outputs.")
```

**Expected:** Returns True only when the JSON array from `gcloud compute tpus tpu-vm list --format=json` is empty (`[]`).

Checking `gcloud compute tpus tpu-vm list --zone=$ZONE --format=json` makes teardown verification machine-checkable.

## Make it yours

Compute `closed_form_sum_of_squares(4)` and `closed_form_sum_of_squares(8)`, verify that `closed_form_sum_of_squares(4) == 10416.0`, and assert that `receipt['verified']` is `True`.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Inputs:**
- The closed-form sum helper and a runtime receipt produced by the active CPU or TPU backend.

**Outputs:**
- Theoretical sums for the requested four- and eight-chip mappings, plus the receipt verification flag, observed backend, and actual device counts.

**Invariants to check:**
- The four-chip theoretical total matches the exercise value and the active receipt reports its actual backend and counts. A CPU receipt may pass this local exercise but does not qualify as physical TPU evidence.

**Hints:**
- Evaluate the helper for both requested chip counts.
- Keep theoretical chip-mapping totals separate from observed device arithmetic; read verified status, backend, and device counts from the receipt.

**Starter code scaffold (fill in the TODOs):**

```python
def audit_runtime_sum_receipt(receipt, closed_form_sum_of_squares):
    # TODO: check both theoretical totals and report the active receipt fields as observed.
    raise NotImplementedError
```

<details><summary>Reference solution</summary>

```python
# Exercise solution
s4 = closed_form_sum_of_squares(4)
s8 = closed_form_sum_of_squares(8)
print("4-chip expected sum:", s4, "8-chip expected sum:", s8, "receipt hash:", receipt["receipt_sha256"])
# Assert invariant `s4 == 10416.0 and s8 == 85344.0 and receipt["verified"] is True` holds
assert s4 == 10416.0 and s8 == 85344.0 and receipt["verified"] is True
```

</details>

## Detect a silent fallback from 4 TPU chips to 1 CPU device

**Foundations**

Write a check `is_four_chip_tpu_receipt(r)` that returns `True` only when `r['backend'] == 'tpu'`, `r['local_device_count'] == 4`, and `r['observed_total'] == 10416.0`.

<details><summary>Hint</summary>

Check all three fields in the receipt dictionary.

</details>

### How to write: Detect a silent fallback from 4 TPU chips to 1 CPU device — Step-by-step recipe & starter scaffold

**Inputs:**
- A candidate runtime receipt mapping with backend, local device count, and observed total fields.

**Outputs:**
- A Boolean predicate for the exact four-chip TPU receipt contract.

**Invariants to check:**
- All required fields must match; missing fields or CPU backend return false.

**Hints:**
- Validate backend, count, and total independently.
- Use a tolerance only for the floating representation of the exact integer total.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp
import numpy as np

def is_four_chip_tpu_receipt(receipt):
    # TODO: return whether all required four-chip TPU receipt fields are valid.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Detect a silent fallback from 4 TPU chips to 1 CPU device (Foundations)
def is_four_chip_tpu_receipt(r):
    return bool(r["backend"] == "tpu" and r["local_device_count"] == 4 and np.isclose(r["observed_total"], 10416.0))

simulated_tpu = {"backend": "tpu", "local_device_count": 4, "observed_total": 10416.0}
print("Simulated 4-chip TPU check:", is_four_chip_tpu_receipt(simulated_tpu), "Local CPU check:", is_four_chip_tpu_receipt(receipt))
# Assert invariant `is_four_chip_tpu_receipt(simulated_tpu) is True` holds
assert is_four_chip_tpu_receipt(simulated_tpu) is True
```

Requiring `backend == 'tpu'`, `local_device_count == 4`, and the 4-chip `pmap` total prevents a misconfigured environment from passing as a 4-chip TPU run.

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

Keep the JSON runtime receipt (observed_backend, device_count, pmap_sum_of_squares, receipt_sha256) and the 1-chip vs 4-chip vs 8-chip pmap reference values.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX installation for Cloud TPU](https://docs.jax.dev/en/latest/installation.html)
- [Manage Cloud TPU VMs with gcloud](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm)

