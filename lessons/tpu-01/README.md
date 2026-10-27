# Select a TPU slice and provision your Cloud TPU VM

Phase 19: TPU Setup: Provisioning & Runtime Verification · about 25 minutes · CPU

This worked reference is validated on CPU. Accelerator compatibility requires a separate source review and target execution receipt. Use the execution matrix to review device selection and data ownership before adapting a separate target program.

## What you will be able to do

- Choose between a free notebook TPU (Colab/Kaggle) and a single-host Google Cloud TPU VM (`v5litepod-4` or `v6e-4`).
- Provision a Cloud TPU VM using `gcloud compute tpus tpu-vm create` or a Flex-start instance group, then connect over SSH.
- Create an isolated `.venv` from the reviewed dependency lock, then require the expected TPU backend, exact requested device count, and synchronized arithmetic before marking the requested target ready.
- Delete the named TPU VM or owning queued resource and verify those named resources are absent; unrelated resources may remain.

## The problem

Before you run a Transformer, sharded step, Pallas kernel, or distributed workload on a TPU, you need a working TPU runtime and a repeatable receipt that proves JAX actually discovered TPU chips—and you need a teardown habit so an idle Cloud TPU VM never keeps billing.

## The idea

Getting a TPU running separates into four explicit boundaries: choosing an access route (free Colab/Kaggle TPU vs single-host Google Cloud TPU VM via on-demand or Flex-start), creating the TPU VM in a supported zone, installing the reviewed pinned environment inside an isolated `.venv` on the TPU VM host, and recording a machine-checkable runtime receipt before deleting the VM.

## Why a TPU VM is a host computer attached to accelerator chips

When you create a Cloud TPU VM, Google Cloud allocates a Linux host machine directly attached over PCIe/ICI to the TPU chips on that board—for example, a `v5litepod-4` VM is one Linux host attached to $4$ TPU v5e chips (`16` GiB HBM each). Your SSH session lands on that Linux host, not inside a chip.

For the selected direct Cloud TPU API v5litepod-4 VM route, the request describes one host with four v5e chips (16 GiB per chip). That mapping is specific to the named generation, machine type and API. jax.device_count() counts JAX devices, not physical chips; record the API resource description and JAX device inventory separately. Newer TPU generations or APIs can map multiple JAX devices to one physical chip.

$$
N_{\text{JAX devices}} = \texttt{jax.device\_count()}, \qquad N_{\text{physical chips}} = f(\text{generation},\text{machine type},\text{API topology})
$$

### Process, JAX-device and physical-chip ownership for a selected Cloud TPU topology

**Predict:** Which counts come from the host launcher, which from JAX, and which from the resource topology?

![Process, JAX-device and physical-chip ownership for a selected Cloud TPU topology](../../phases/19-tpu/01-get-a-tpu-running-and-verify-the-runtime/outputs/mechanism.svg)

*Topology schematic for the named Cloud TPU API routes; check the selected resource and runtime receipt.*

For the listed single-host Cloud TPU API examples, one Linux host runs one JAX process. The process reports JAX devices. The selected v5e/v6e topology maps one JAX device to each physical chip; that mapping is API and generation specific. The TPU7x transfer case exposes two JAX devices per physical chip. HBM is counted per physical chip, not per process.

### Pause and reason

Why does creating a Cloud TPU VM and seeing `STATUS: READY` in `gcloud` not prove that your JAX experiment can run on TPU?

<details><summary>Compare your reasoning</summary>

`STATUS: READY` only confirms that the Linux VM and attached TPU hardware were allocated. You still must verify inside Python on that VM that `libtpu` loads, `jax.default_backend() == 'tpu'`, `jax.device_count()` matches the topology, and a synchronized computation completes.

</details>

## 1. Choose how to get a TPU: Colab/Kaggle vs Google Cloud TPU VM

Choose the runtime that matches the task. A hosted notebook can support a small single-process backend check if TPU access is available. It does not provide every topology, shell permission or multi-host workflow in this course. Availability and quotas depend on the provider.

For the v5e lab below, use the Cloud TPU API: direct TPU VM creation or a bounded Flex-start queued resource. Keep creation, status, SSH and deletion commands within that API. Google now recommends Compute Engine or GKE for newer features. Its Compute Engine TPU Flex-start workflow uses supported TPU machine types and MIGs, but does not support GPU-style resize requests. Do not mix that workflow with this v5e queued-resource example. See both linked API guides before adapting the lab.

## 2. Authenticate `gcloud`, enable the TPU API, and check quota

On your local workstation (or inside Google Cloud Shell), set your project ID, zone, and TPU name, enable `tpu.googleapis.com`, and inspect the available accelerator types in your chosen zone before requesting hardware. Using a single-host slice such as `v5litepod-4` (`4` chips on `1` host) avoids multi-host coordination while still letting you run real `jax.pmap` and mesh sharding across $4$ TPU devices.

The example zone is a starting choice for v5e Flex-start, not a capacity promise. Check supported zones, quota and current pricing before creating a request. Set both a queue expiry and a run duration; a budget alert is not a spending cap.

For the selected Cloud TPU API route, v5litepod-4 requests 4 physical v5e chips on 1 VM host; v5e has one TensorCore per chip, and the expected 4 JAX devices must still be checked at runtime. Other APIs can choose different host layouts for a similarly named topology. Do not derive physical chip count from jax.device_count() alone.

**Configure project, zone, and enable the Cloud TPU API**

```bash
# Configure project, zone, and enable the Cloud TPU API
export PROJECT_ID="your-gcp-project-id"
export ZONE="us-west4-a"
export TPU_NAME="jax-tpu-course"
export ACCELERATOR_TYPE="v5litepod-4"
export RUNTIME_VERSION="v2-alpha-tpuv5-lite"

gcloud auth login
gcloud config set project "$PROJECT_ID"
gcloud services enable tpu.googleapis.com
gcloud compute tpus accelerator-types list --zone="$ZONE"
```

**Expected:** Lists the accelerator types supported in the selected zone. Confirm the requested type is listed; quota and free capacity are separate checks.

## 3. Select one Cloud TPU API provisioning route

Choose one route, not both. A direct VM must reach READY. A Flex-start request must reach ACTIVE and its named VM must become READY before SSH. A queued request is not proof that capacity has been allocated. Inspect failed or expired requests and delete them instead of leaving a future allocation behind.

**Route 2A: Create a single-host Cloud TPU v5e-4 VM directly**

```bash
# Route 2A: direct Cloud TPU VM create
gcloud compute tpus tpu-vm create "$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE" --accelerator-type="$ACCELERATOR_TYPE" --version="$RUNTIME_VERSION"
gcloud compute tpus tpu-vm describe "$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE" --format="value(state,acceleratorType,networkEndpoints[0].ipAddress)"
```

**Expected:** Continue only when the named VM is READY and its accelerator type matches the requested v5litepod-4 resource.

**Route 2B: Bounded Cloud TPU API Flex-start request for v5e**

```bash
export QUEUED_RESOURCE_ID="$TPU_NAME-request"
gcloud alpha compute tpus queued-resources create "$QUEUED_RESOURCE_ID" --project="$PROJECT_ID" --zone="$ZONE" --node-id="$TPU_NAME" --accelerator-type="$ACCELERATOR_TYPE" --runtime-version="$RUNTIME_VERSION" --provisioning-model=flex-start --max-run-duration=1h --valid-until-duration=30m
gcloud alpha compute tpus queued-resources describe "$QUEUED_RESOURCE_ID" --project="$PROJECT_ID" --zone="$ZONE" --format="value(state.state)"
# Continue only after ACTIVE and READY.
gcloud compute tpus tpu-vm describe "$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE" --format="value(state,acceleratorType)"
```

**Expected:** The queue may wait or fail. Proceed only after ACTIVE and READY; verify the named accelerator type and API route. No capacity or cost-free execution is promised.

## 4. SSH into the TPU VM and install the pinned runtime

After the selected TPU VM is READY, bootstrap its isolated environment from the reviewed pinned requirements file. Install the matching TPU JAX extra at the same pinned version. This is environment setup; backend and topology readiness are established separately by the synchronized receipt.

**Bootstrap the pinned runtime on the TPU VM**

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command="mkdir -p ~/jax-tpu-lab"
gcloud compute tpus tpu-vm scp projects/reproducible-experiments/requirements.txt "$TPU_NAME":~/jax-tpu-lab/ --zone="$ZONE"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command='set -euo pipefail; mkdir -p ~/jax-tpu-lab; python3 -m venv ~/jax-tpu-lab/.venv; ~/jax-tpu-lab/.venv/bin/python -m pip install -r ~/jax-tpu-lab/requirements.txt; ~/jax-tpu-lab/.venv/bin/python -m pip install "jax[tpu]==0.9.2" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'
```

**Expected:** Installs pinned JAX ecosystem packages and the matching jax[tpu]==0.9.2 extra. Record this dependency identity with source and runtime receipts.

## 5. Run the requested-target preflight, retrieve its receipt, and clean up route-owned resources

Before any training launch, run the runtime verifier on the requested target and retrieve its receipt. Arithmetic, backend, device-count and topology/API identity remain separate. A CPU rehearsal may pass arithmetic without establishing TPU readiness. After saving evidence, delete only the direct VM or the owning queued request and verify the named resources are absent.

**Run the target verifier and copy its receipt to the workstation**

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command="mkdir -p ~/jax-tpu-lab"
gcloud compute tpus tpu-vm scp public/exercises/tpu-01.py "$TPU_NAME":~/jax-tpu-lab/ --zone="$ZONE"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command='set -euo pipefail; TPU_REQUIRE_TARGET=1 JAX_PLATFORMS=tpu ~/jax-tpu-lab/.venv/bin/python ~/jax-tpu-lab/tpu-01.py | tee ~/jax-tpu-lab/tpu-01-receipt.jsonl'
gcloud compute tpus tpu-vm scp "$TPU_NAME":~/jax-tpu-lab/tpu-01-receipt.jsonl ./ --zone="$ZONE"
```

**Expected:** The copied receipt identifies source and environment, requested and observed backend/device count, arithmetic status and target readiness. On CPU, arithmetic can pass while TPU readiness is false.

**Delete the named direct VM or owning Flex-start request and verify absence**

```bash
if [ -n "$QUEUED_RESOURCE_ID" ]; then
  gcloud alpha compute tpus queued-resources delete "$QUEUED_RESOURCE_ID" --project="$PROJECT_ID" --zone="$ZONE" --force --quiet
else
  gcloud compute tpus tpu-vm delete "$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE" --quiet
fi
remaining_vm=$(gcloud compute tpus tpu-vm list --project="$PROJECT_ID" --zone="$ZONE" --filter="name=$TPU_NAME" --format="value(name)")
[ -z "$remaining_vm" ] || { echo "Named TPU VM still exists: $remaining_vm"; exit 1; }
if [ -n "$QUEUED_RESOURCE_ID" ]; then
  remaining_request=$(gcloud alpha compute tpus queued-resources list --project="$PROJECT_ID" --zone="$ZONE" --filter="name=$QUEUED_RESOURCE_ID" --format="value(name)")
  [ -z "$remaining_request" ] || { echo "Owning queued resource still exists: $remaining_request"; exit 1; }
fi
printf "Named route resources are absent; unrelated resources may remain.\n"
```

**Expected:** The named VM and, if used, its owning queued request are absent. Other resources may remain.

## Write the runtime contract and synchronized pmap verifier

Create main.py with verify_runtime_contract so the same function validates CPU locally and TPU on your Cloud TPU VM.

```python
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
```

The verifier checks both device discovery and a synchronized pmap reduction across all local devices, then hashes the receipt.

## Compare single-host TPU slice topologies and emit the receipt

Append the slice topology table and runtime receipt check to main.py.

```python
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
```

Calculating total chips and HBM per slice makes the hardware capacity concrete before you provision a Cloud TPU VM.

## Step 3: Verify invariants on the completed state

Run the final shape and numerical assertions to confirm the state built in Steps 1 and 2.

```python
receipt = verify_runtime_contract(expected_backend="tpu", expected_device_count=4)
assert receipt["arithmetic_ok"] is True
```

Checking these invariants confirms the computation is ready for the full worked experiment.

## Run the example

```python
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
```

Expected: The topology catalog distinguishes VM hosts, expected one-process-per-host ownership, expected JAX device count for the listed v5e/v6e APIs, physical chips, and HBM GiB. These are planning expectations; compare the selected API resource description and live JAX devices before interpreting a target receipt. TPU7x can expose two JAX devices per physical chip.

## Cloud TPU chip counts and HBM capacity in separate unit panels

**Predict:** How do physical chips and total HBM change across the listed single-host slice requests, and which values use different units?

![Cloud TPU chip counts and HBM capacity in separate unit panels](../../phases/19-tpu/01-get-a-tpu-running-and-verify-the-runtime/outputs/figure.svg)

**Analytic calculation; no device result recorded**

### Read the figure

Analytic catalog calculation; not an observed device result. Panel one reports physical chip count per requested API topology. Panel two reports total HBM in GiB. The axes are independent and must not be numerically compared. These are catalog expectations, not observed device receipts.

### Connect it to the computation

The chip panel is derived from hosts times chips per host. The HBM panel multiplies physical chips by published per-chip capacity. The topology diagram separates process, JAX device and chip ownership.

```python
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
```

## Recorded reference execution

CPU run: 2026-10-27T15:46:39.416210+00:00. JAX 0.9.2.

```text
Verified runtime receipt: {"arithmetic_ok": true, "device_count": 1, "device_kinds": ["cpu"], "environment_lock": "projects/reproducible-experiments/requirements.txt; jax[tpu]==0.9.2", "expected_backend": "tpu", "expected_backend_ok": false, "expected_device_count": 4, "expected_sum_of_squares": 140.0, "expected_topology_ok": false, "jax": "0.9.2", "local_device_count": 1, "observed_backend": "cpu", "pmap_sum_of_squares": 140.0, "process_count": 1, "python": "3.14.3", "receipt_sha256": "71ec1d62d8092ac592002e0ca6cd5d79c77fb8ccfee100fb33a747381d7baec8", "requested_target_ready": false, "source_identity": "unavailable: interactive execution has no readable __file__", "source_sha256": null, "target_topology": {"api": "Cloud TPU API direct VM/Flex-start", "hosts_expected": 1, "machine_type": "v5litepod-4", "physical_chips_expected": 4, "topology_source": "requested Cloud TPU resource contract; physical chips are not inferred from JAX devices"}}
Single-host TPU slice HBM totals (GiB): {'v5litepod-1': 16, 'v5litepod-4': 64, 'v5litepod-8': 128, 'v6e-4': 128}
Verified runtime receipt: {"arithmetic_ok": true, "device_count": 1, "device_kinds": ["cpu"], "environment_lock": "projects/reproducible-experiments/requirements.txt; jax[tpu]==0.9.2", "expected_backend": "tpu", "expected_backend_ok": false, "expected_device_count": 4, "expected_sum_of_squares": 140.0, "expected_topology_ok": false, "jax": "0.9.2", "local_device_count": 1, "observed_backend": "cpu", "pmap_sum_of_squares": 140.0, "process_count": 1, "python": "3.14.3", "receipt_sha256": "71ec1d62d8092ac592002e0ca6cd5d79c77fb8ccfee100fb33a747381d7baec8", "requested_target_ready": false, "source_identity": "unavailable: interactive execution has no readable __file__", "source_sha256": null, "target_topology": {"api": "Cloud TPU API direct VM/Flex-start", "hosts_expected": 1, "machine_type": "v5litepod-4", "physical_chips_expected": 4, "topology_source": "requested Cloud TPU resource contract; physical chips are not inferred from JAX devices"}}
Single-host TPU slice HBM totals (GiB): {'v5litepod-1': 16, 'v5litepod-4': 64, 'v5litepod-8': 128, 'v6e-4': 128}
False False True False
Mismatch requested_target_ready: False observed: cpu devices: 1
Requires --worker=all and jax.distributed.initialize(): {'v5litepod-1': False, 'v5litepod-4': False, 'v5litepod-8': False, 'v5litepod-16': True}
48 GiB state/device replicated: 48.0
48 GiB state/device ideal even shard: 12.0
Usable HBM/device: 12.8
Ideal shard plus assumed activation: 13.0
Selected target and runtime receipt: {'api': 'Cloud TPU API direct VM/Flex-start', 'machine_type': 'v5litepod-4', 'requested_processes': 1, 'observed_process_count': 1, 'requested_physical_chips': 4, 'observed_jax_device_count': 1, 'observed_backend': 'cpu', 'requested_target_ready': False}
48 GiB state per-device, replicated/sharded: 48.0 12.0
Usable HBM/device: 12.8 sharded plus activation: 13.0
Changed 52 GiB ideal shard/device: 13.0
Expected 4-device pmap sum of squares: 10416.0
PASS: tpu-01

```

## Reject backend mismatch and insufficient topology before launch

**Predict before running:** On CPU, what fails when the requested target is TPU/4 devices, and what fails if CPU/4 devices is requested?

```python
# Backend mismatch: CPU runtime, TPU requested.
backend_mismatch = verify_runtime_contract(expected_backend="tpu", expected_device_count=4)
# Isolate topology mismatch: CPU backend matches, but only one CPU device is available.
topology_mismatch = verify_runtime_contract(expected_backend="cpu", expected_device_count=4)
print(backend_mismatch["expected_backend_ok"], backend_mismatch["expected_topology_ok"], topology_mismatch["expected_backend_ok"], topology_mismatch["expected_topology_ok"])
assert backend_mismatch["expected_backend_ok"] is False
assert topology_mismatch["expected_backend_ok"] is True and topology_mismatch["expected_topology_ok"] is False
assert backend_mismatch["requested_target_ready"] is False and topology_mismatch["requested_target_ready"] is False
```

**Expected:** A CPU process fails the TPU backend check. A CPU/4 request matches its backend but fails the exact device-count check. Both are rejected before a training launch.

The checks isolate backend selection from topology. Arithmetic success alone does not qualify either requested target.

## Reject a mismatched backend or missing TPU devices immediately

**Predict before running:** What should verify_runtime_contract return if you require 4 TPU devices on a single-CPU host?

```python
# Experiment — Reject a mismatched backend or missing TPU devices immediately
mismatch = verify_runtime_contract(expected_backend="tpu", expected_device_count=4)
print("Mismatch requested_target_ready:", mismatch["requested_target_ready"], "observed:", mismatch["observed_backend"], "devices:", mismatch["device_count"])
assert mismatch["requested_target_ready"] == (jax.default_backend() == "tpu" and jax.device_count() == 4)
```

**Expected:** On CPU, mismatch['requested_target_ready'] is False while still recording the observed backend and device count.

Failing the contract check explicitly prevents a script from silently running on CPU when you intended to benchmark or train on a TPU VM.

## Verify single-host vs multi-host process counts

**Predict before running:** Why does v5litepod-4 need only one SSH command while v5litepod-16 requires --worker=all?

```python
# Experiment — Verify single-host vs multi-host process counts
slice_hosts = {"v5litepod-1": 1, "v5litepod-4": 1, "v5litepod-8": 1, "v5litepod-16": 4}
needs_multi_host = {k: (v > 1) for k, v in slice_hosts.items()}
print("Requires --worker=all and jax.distributed.initialize():", needs_multi_host)
assert needs_multi_host["v5litepod-4"] is False and needs_multi_host["v5litepod-16"] is True
```

**Expected:** v5litepod-1, v5litepod-4, and v5litepod-8 run on 1 host (False); v5litepod-16 spans 4 hosts (True).

Single-host TPU VMs attach all chips to one Linux VM; multi-host slices attach chips across multiple Linux VMs that must all launch the Python process.

## Make it yours

Use the Foundations ledger for a 48 GiB state on four 16 GiB v5e chips. Report replicated and ideal-sharded state per chip, the 12.8 GiB usable budget, and the result with a 1 GiB activation allowance. Then inspect the selected v5litepod-4 receipt: report observed process count and JAX device count separately from the requested physical chip count, plus the API and machine type. On CPU, arithmetic should pass while requested_target_ready remains false.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Inputs:**
- The runtime verifier with expected_backend and expected_device_count, plus the named topology catalog.
- A 48 GiB state, four 16 GiB chips, 80% usable capacity and a 1 GiB per-chip activation assumption.

**Outputs:**
- Replicated/ideal-sharded per-device memory decisions with and without activation overhead.
- Separate requested API/machine/chip/process values and observed JAX backend/process/device values.

**Invariants to check:**
- Aggregate HBM alone does not establish per-device fit.
- CPU arithmetic may pass while requested TPU readiness is false.
- Observed JAX device count does not by itself establish a physical chip count.

**Hints:**
- Compute 48/4 and compare 12 plus the activation allowance with 16 times 0.8.
- Use expected_device_count, not the obsolete minimum-device argument.

**Starter code scaffold (fill in the TODOs):**

```python
def inspect_runtime_and_slice(verify_runtime_contract, expected_backend, expected_device_count, topologies):
    # TODO: check arithmetic, requested readiness, per-device fit and separately identified topology.
    raise NotImplementedError
```

<details><summary>Reference solution</summary>

```python
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

```

</details>

## Compare replicated and ideally sharded per-device HBM

**Foundations**

A 48 GiB training state runs on four v5e chips, each with 16 GiB HBM. At an 80% usable-HBM budget, calculate per-device state for replication and ideal even sharding. Which placement meets the state-only limit? How does an added 1 GiB/device of activations change the result? Repeat for a 52 GiB state.

<details><summary>Hint</summary>

Replication stores the whole state on every device. Ideal even sharding divides by four. The 80% state budget is 12.8 GiB/device; separately subtract assumed activations.

</details>

### How to write: Compare replicated and ideally sharded per-device HBM — Step-by-step recipe & starter scaffold

**Inputs:**
- A 48 GiB state, four physical v5e chips with 16 GiB HBM each, 80% usable-HBM fraction and 1 GiB/device activation allowance.

**Outputs:**
- Per-device replicated and ideal-sharded state requirements, plus fit decisions before and after activation allowance.
- A 52 GiB changed-input result that shows the ideal shard exceeds the usable budget.

**Invariants to check:**
- Replication stores the whole state on each chip; ideal even sharding divides by chip count; report this as an assumption, not a measured placement.

**Hints:**
- Compute the per-chip usable budget first.
- Repeat with 52 GiB to see the changed input cross the ideal-shard capacity limit.

**Starter code scaffold (fill in the TODOs):**

```python
def placement_hbm_screen(total_state_gib, chips, chip_hbm_gib, usable_fraction, activation_gib_per_chip):
    # TODO: return per-chip state for replication and ideal even sharding,
    # the usable budget after activation allowance, and fit flags.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
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
```

The 48 GiB total is not a placement claim. Replication needs 48 GiB on each 16 GiB chip and cannot fit. Ideal even sharding needs 12 GiB/device, leaving only 0.8 GiB below the 12.8 GiB state budget, so an assumed 1 GiB activation exceeds it. At 52 GiB, even the ideal shard is 13 GiB/device and fails the 80% budget. Real fit additionally requires verifying actual state sharding and peak live memory.

</details>

## Verify the pmap sum-of-squares formula for 4 TPU devices

**Transfer / diagnosis**

Compute the exact sum of squares that `verify_runtime_contract` expects when `len(local_devices) == 4` (`32` elements `0..31`).

<details><summary>Hint</summary>

Sum `np.arange(32, dtype=np.float64) ** 2`.

</details>

### How to write: Verify the pmap sum-of-squares formula for 4 TPU devices — Step-by-step recipe & starter scaffold

**Inputs:**
- Four local TPU devices and the mapped range of 32 integers from zero through 31.

**Outputs:**
- Expected exact sum of squares for the four-device pmap fixture.

**Invariants to check:**
- The expected value sums each integer exactly once; it is an oracle for the four-device test only.

**Hints:**
- Use the closed-form sum-of-squares identity or an integer reference sum.
- Run the pmap assertion on a real four-chip TPU to qualify runtime behavior.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp
import numpy as np

def expected_four_chip_sum():
    # TODO: compute the integer oracle for the range consumed by the four-device pmap.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Verify the pmap sum-of-squares formula for 4 TPU devices (Transfer / diagnosis)
four_device_expected = float(np.sum(np.arange(4 * 8, dtype=np.float64) ** 2))
print("Expected 4-device pmap sum of squares:", four_device_expected)
# Assert invariant `four_device_expected == 10416.0` holds
assert four_device_expected == 10416.0
```

On a 4-chip TPU VM (`v5litepod-4` or `v6e-4`), `x` has shape `(4, 8)` and the synchronized sum of squares across the 4 chips is `10416.0`.

</details>

## Check your understanding

Which workflow records a small synchronized TPU computation and checks that the lab’s cloud resources were removed?

1. Running `gcloud compute tpus tpu-vm create` and closing your laptop lid when done
2. Assert the remote TPU backend, run and synchronize the verifier, retrieve evidence, delete the VM and any owning request, then check their absence
3. Checking `pip list` on your local laptop

<details><summary>Answer and explanation</summary>

**1. Running `gcloud compute tpus tpu-vm create` and closing your laptop lid when done**

Closing a laptop does not stop a remote job or release capacity. Retrieve the receipt, then inspect both the VM and any owning request.

**2. Assert the remote TPU backend, run and synchronize the verifier, retrieve evidence, delete the VM and any owning request, then check their absence (correct)**

The computation receipt identifies a bounded tested runtime. The separate cleanup checks cover the named VM and request. Neither a small arithmetic pass nor an empty resource listing proves all workload behavior or a final bill.

**3. Checking `pip list` on your local laptop**

Local packages describe the workstation, not the remote backend. Execute and synchronize the verifier on the TPU host.

</details>

## Diagnose the result

If `gcloud compute tpus tpu-vm create` fails with `RESOURCE_EXHAUSTED`, switch zones or use Flex-start (`Route 2B`). If `jax.default_backend()` prints `cpu` on the TPU VM, confirm that `jax[tpu]` was installed from `libtpu_releases.html` inside the active `.venv` and that no other process holds `/dev/accel*` or `/dev/vfio/*`.

## Carry forward

- A Cloud TPU VM is a Linux host directly attached to TPU chips; you SSH into the host and run Python inside a `.venv` with `jax[tpu]`.
- Single-host slices (`v5litepod-1`, `v5litepod-4`, `v5litepod-8`, `v6e-4`) let you run 1 to 8 TPU chips from a single SSH command without multi-host coordination.
- Pair the selected provisioning route with a saved receipt, then delete only its named TPU VM or owning queued resource and verify those names are absent.

## Keep your evidence

Keep the 48 GiB replicated versus ideal-sharded ledger with headroom/activation assumptions, requested API/machine/chip/process identity, observed process/device/backend receipt and source/environment identity where available. Retain absence checks only for the named owned VM or queued resource.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [Google Cloud TPU VM user guide and architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)
- [Manage TPU VMs with gcloud](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm)
- [JAX installation for Cloud TPU](https://docs.jax.dev/en/latest/installation.html)
- [Cloud TPU API Flex-start requests (checked 2026-10-21)](https://docs.cloud.google.com/tpu/docs/request-using-flex-start)
- [Compute Engine TPU Flex-start and MIG limitations](https://docs.cloud.google.com/tpu/docs/create-flex-start-compute)
- [Cloud TPU pricing and READY-state charges](https://cloud.google.com/tpu/pricing)
- [JAX TPU hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html)
- [TPU v5e architecture and supported configurations](https://docs.cloud.google.com/tpu/docs/v5e)
- [TPU machine types and physical chips per VM](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines)
- [JAX TPU hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html)
- [TPU v5e topology and supported configurations](https://docs.cloud.google.com/tpu/docs/v5e)
- [Compute Engine TPU machine types](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines)

