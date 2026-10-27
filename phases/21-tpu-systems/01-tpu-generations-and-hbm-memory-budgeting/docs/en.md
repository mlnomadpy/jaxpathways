# TPU generations and HBM memory budgeting

Phase 21: TPU Systems: Generations, Memory, Precision & XProf · about 25 minutes · CPU

This worked reference is validated on CPU. Accelerator compatibility requires a separate source review and target execution receipt. Use the execution matrix to review device selection and data ownership before adapting a separate target program.

## What you will be able to do

- Compare Google Cloud TPU generations (`v5e`, `v5p`, `v6e` Trillium, and Ironwood) by HBM capacity, HBM bandwidth, and MXU BF16 throughput.
- Calculate the Roofline ridge point (`bf16_tflops / bw_gbs`) in FLOPs per byte for each TPU generation.
- Compute training-state and KV-cache HBM from explicit tensor byte counts, including master weights, compute weights, gradients, and Adam moments.
- Make per-device training-state sizing decisions for replicated or explicitly sharded placement, then size inference KV cache separately with headroom.

## The problem

Provisioning and launching a TPU VM is only half the engineering job: if you pick the wrong TPU generation or miscalculate parameter + optimizer + KV-cache HBM bytes, your TPU experiment will either fail with `RESOURCE_EXHAUSTED: Out of memory` or under-utilize the Matrix Multiply Unit (MXU).

## The idea

Budget capacity with a byte ledger before selecting a TPU slice, and treat the roofline ridge point as a theoretical screening estimate. GiB capacity and FLOP/byte ridge points are different quantities and must be read in separate panels. The worked training-state estimates state their tensor dtypes and exclusions; they are not observed runtime memory.

## Connect MXU arithmetic intensity and HBM capacity to TPU slice sizing

Use a per-chip ledger before choosing a slice. In the exercise, 7B parameters with the stated mixed-BF16 accounting require 117.347 GiB of training state. Under ideal even sharding, four v6e chips need 29.33675 GiB per chip, above the fixture's 25.6 GiB usable budget at 80% of 32 GiB, so this plan fails before activations. Eight chips need 14.668375 GiB per chip, leaving 10.931625 GiB for activations and temporaries; with the assumed 8 GiB allowance it fits this arithmetic screen. This is a placement assumption, not proof that the implementation shards optimizer state evenly. Verify actual sharding and peak memory before reserving hardware.

Keep inference separate: the fixture's 7B BF16 KV cache at 8192 tokens is 1 GiB. Do not add it to the training-state total unless the workload actually runs training and inference together. The ridge point is peak BF16 throughput divided by HBM bandwidth, a theoretical roofline balance in FLOP/byte, not observed speed. Use it to form a profile hypothesis, then decide from a target trace whether compute, memory traffic or communication needs attention.

$$
B_{\text{state/device}} = \begin{cases} B_{\text{state}}, & \text{replicated} \\ B_{\text{state}}/N_{\text{chips}}, & \text{ideal even sharding} \end{cases}; \quad B_{\text{usable/device}} = f_{\text{usable}} B_{\text{HBM/chip}} - B_{\text{activation/device}}
$$

### TPU state placement, HBM budget and compute path

**Predict:** What remains on each physical chip under replication versus explicit sharding, and where do BF16 operands enter?

![TPU state placement, HBM budget and compute path](../outputs/mechanism.svg)

*Architecture and dataflow mechanism diagram.*

Follow the arrows from the per-device HBM ledger, through operand traffic, into the chosen compute path, and finally to a target trace. The ledger must name placement: for 48 GiB of state on four 16 GiB v5e chips, replication needs 48 GiB/device; ideal even sharding needs 12 GiB/device. With the fixture's 1 GiB activation allowance, that shard needs 13 GiB against 12.8 GiB usable, so it misses by 0.2 GiB. The bandwidth node carries GB/s figures, while the compute comparison is in FLOP/byte; inspect each unit on its own. Inference KV cache is a separate ledger. The next step is to confirm actual sharding and inspect a target trace for memory, compute and communication evidence.

### Pause and reason

For 7 billion parameters, how many GiB does the stated 18-byte mixed-BF16 ledger require before activations, and why is a ridge point not a measured bottleneck?

<details><summary>Compare your reasoning</summary>

The estimate is 7e9 × 18 / 2^30 ≈ 117.4 GiB. It excludes activations, communication buffers, allocator overhead and temporary copies, so it is only a capacity lower bound. The ridge point compares published peak compute and bandwidth; only a workload measurement and profile can identify an observed bottleneck.

</details>

## 1. Compare TPU generations (`v5e`, `v5p`, `v6e`, Ironwood) and calculate HBM budgets

The specification table provides per-chip capacity and bandwidth inputs for a first estimate; it does not promise availability, fit or workload speed. For the named v5e example, four chips have 64 GiB aggregate HBM, but the 48 GiB state still needs either 48 GiB on every chip if replicated or 12 GiB per chip under ideal even sharding. At the fixture's 80% usable allowance, each 16 GiB chip offers 12.8 GiB; adding 1 GiB of assumed activations makes the ideal shard 13 GiB and misses by 0.2 GiB. This is why aggregate slice capacity is not the fit test.

For each candidate, state the physical chip count, per-chip HBM budget, placement assumption and activation/temporary allowance. For the later v6e exercise, four chips fail the 7B mixed-BF16 estimate before activations, while eight leave room for the stated 8 GiB allowance. Keep inference KV cache as a separate workload budget. Select a topology only after checking real per-device placement and a target memory trace.

**Run the TPU generation & HBM budgeting verifier on your TPU VM or workstation**

```bash
# Run the TPU generation & HBM budgeting verifier on your TPU VM or workstation
gcloud compute tpus tpu-vm scp public/exercises/tpu-03.py "$TPU_NAME":~/jax-tpu-lab/ --zone="$ZONE"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
  --command="JAX_PLATFORMS=tpu ~/jax-tpu-lab/.venv/bin/python ~/jax-tpu-lab/tpu-03.py"
```

**Expected:** Prints the TPU ridge points (FLOP/byte) and the 7B and 1.5B training + KV-cache HBM budgets in GiB.

## Use chip specifications without confusing them with a runtime contract

Treat physical chips, JAX devices and host processes as separate counts. Specifications describe per-chip resources; the selected machine type and API determine hosts and physical chips, while the runtime receipt reports observed processes and JAX devices. The TPU7x example in this lesson maps two JAX devices to one chip, so multiplying every JAX device count by per-chip HBM would overstate the physical capacity. Record API, machine type, requested topology and observed inventory together, and use the matching mapping when translating a device plan into a per-chip ledger. Peak throughput is a hardware ceiling, not evidence that a JAX expression uses that path.

## Define TPU generation specs and compute Roofline ridge points

Create main.py with TPU_SPECS and calculate ridge_flops_per_byte for v5e, v5p, and v6e.

```python
# Step 1 — Define TPU generation specs and compute Roofline ridge points
import jax
import jax.numpy as jnp
import numpy as np

TPU_SPECS = [
    {"gen": "v5e", "hbm_gib": 16.0, "bw_gbs": 819.0, "bf16_tflops": 197.0},
    {"gen": "v5p", "hbm_gib": 95.0, "bw_gbs": 2765.0, "bf16_tflops": 459.0},
    {"gen": "v6e", "hbm_gib": 32.0, "bw_gbs": 1638.0, "bf16_tflops": 918.0},
]
for s in TPU_SPECS:
    s["ridge_flops_per_byte"] = round((s["bf16_tflops"] * 1e12) / (s["bw_gbs"] * 1e9), 1)
```

Converting TFLOP/s to FLOP/s and GB/s to byte/s introduces a factor of 1000. Their ratio is the ideal Roofline intersection, not a guarantee that a kernel saturates the MXU.

## Estimate training-state and KV-cache HBM budgets across precisions

Append estimate_memory_gib and compare 7B vs 1.5B model memory footprints.

```python
# Step 2 — Estimate training-state and KV-cache HBM budgets across precisions
def estimate_memory_gib(params_billions, kv_tokens, layers=32, kv_heads=8, head_dim=128):
    n_params = params_billions * 1e9
    kv_elements = 2.0 * layers * kv_tokens * kv_heads * head_dim
    gib = float(1024 ** 3)
    return {
        "fp32_train_state_gib": round((n_params * 16.0) / gib, 3),
        "mixed_bf16_train_state_gib": round((n_params * 18.0) / gib, 3),
        "kv_fp32_gib": round((kv_elements * 4.0) / gib, 3),
        "kv_bf16_gib": round((kv_elements * 2.0) / gib, 3),
        "kv_int8_gib": round((kv_elements * 1.0) / gib, 3),
    }


def per_device_hbm_ledger(total_state_gib, chips, chip_hbm_gib, placement, usable_fraction=0.8, activation_gib_per_chip=0.0):
    if chips < 1 or not 0 < usable_fraction <= 1 or activation_gib_per_chip < 0:
        raise ValueError("chips and utilization must be positive; activation allowance cannot be negative")
    if placement == "replicated":
        state_per_chip = float(total_state_gib)
    elif placement == "evenly_sharded_ideal":
        state_per_chip = float(total_state_gib) / chips
    else:
        raise ValueError("placement must be replicated or evenly_sharded_ideal")
    usable_hbm_per_chip = float(chip_hbm_gib) * usable_fraction
    available_for_state = usable_hbm_per_chip - activation_gib_per_chip
    return {"placement_assumption": placement, "chips": chips, "state_per_chip_gib": state_per_chip,
            "activation_allowance_per_chip_gib": activation_gib_per_chip,
            "usable_hbm_per_chip_gib": usable_hbm_per_chip,
            "fits_under_assumptions": state_per_chip <= available_for_state,
            "remaining_after_state_before_activation_gib": usable_hbm_per_chip - state_per_chip}



budget_7b = estimate_memory_gib(params_billions=7.0, kv_tokens=8192)
budget_1p5b = estimate_memory_gib(params_billions=1.5, kv_tokens=4096)
assert budget_7b["mixed_bf16_train_state_gib"] > 64.0 and budget_7b["mixed_bf16_train_state_gib"] < 128.0
# Assert invariant `budget_1p5b["mixed_bf16_train_state_gib"] < 64.0` holds
assert budget_1p5b["mixed_bf16_train_state_gib"] < 64.0
print("TPU ridge points (FLOP/byte):", {s["gen"]: s["ridge_flops_per_byte"] for s in TPU_SPECS})
print("7B memory budget (GiB):", budget_7b)
print("1.5B memory budget (GiB):", budget_1p5b)
```

Computing the named-tensor ledger identifies whether a topology has enough total HBM, but usable capacity must also reserve room for activations, communication buffers, temporary copies and allocator overhead. The 7B estimate exceeds the 80%-usable budget of a 4-chip v6e slice; it is an estimate, not a fit guarantee.

## Step 3: Verify invariants on the completed state

Run the final shape and numerical assertions to confirm the state built in Steps 1 and 2.

```python
assert budget_7b["mixed_bf16_train_state_gib"] > 64.0 and budget_7b["mixed_bf16_train_state_gib"] < 128.0
assert budget_1p5b["mixed_bf16_train_state_gib"] < 64.0
```

Checking these invariants confirms the computation is ready for the full worked experiment.

## Compare replicated and sharded state per device

Add per_device_hbm_ledger and calculate 48 GiB on four 16 GiB chips with a 1 GiB/device activation allowance. Change the total to 52 GiB and recompute.

```python


def per_device_hbm_ledger(total_state_gib, chips, chip_hbm_gib, placement, usable_fraction=0.8, activation_gib_per_chip=0.0):
    if chips < 1 or not 0 < usable_fraction <= 1 or activation_gib_per_chip < 0:
        raise ValueError("chips and utilization must be positive; activation allowance cannot be negative")
    if placement == "replicated":
        state_per_chip = float(total_state_gib)
    elif placement == "evenly_sharded_ideal":
        state_per_chip = float(total_state_gib) / chips
    else:
        raise ValueError("placement must be replicated or evenly_sharded_ideal")
    usable_hbm_per_chip = float(chip_hbm_gib) * usable_fraction
    available_for_state = usable_hbm_per_chip - activation_gib_per_chip
    return {"placement_assumption": placement, "chips": chips, "state_per_chip_gib": state_per_chip,
            "activation_allowance_per_chip_gib": activation_gib_per_chip,
            "usable_hbm_per_chip_gib": usable_hbm_per_chip,
            "fits_under_assumptions": state_per_chip <= available_for_state,
            "remaining_after_state_before_activation_gib": usable_hbm_per_chip - state_per_chip}

rep = per_device_hbm_ledger(48, 4, 16, "replicated", activation_gib_per_chip=1)
shard = per_device_hbm_ledger(48, 4, 16, "evenly_sharded_ideal", activation_gib_per_chip=1)
changed = per_device_hbm_ledger(52, 4, 16, "evenly_sharded_ideal", activation_gib_per_chip=1)
print("state/device, replicated/sharded/changed:", rep["state_per_chip_gib"], shard["state_per_chip_gib"], changed["state_per_chip_gib"])
assert rep["fits_under_assumptions"] is False and shard["fits_under_assumptions"] is False and changed["fits_under_assumptions"] is False
assert round(shard["remaining_after_state_before_activation_gib"], 1) == 0.8 and changed["state_per_chip_gib"] == 13.0
```

A 48 GiB state needs 48 GiB/device if replicated or 12 GiB/device under ideal even sharding. At 80% HBM the per-device budget is 12.8 GiB; allowing 1 GiB for activations makes the ideal shard fail. A changed total of 52 GiB needs 13 GiB/device before activations.

## Run the example

```python
# Step 1 — Define TPU generation specs and compute Roofline ridge points
import jax
import jax.numpy as jnp
import numpy as np

TPU_SPECS = [
    {"gen": "v5e", "hbm_gib": 16.0, "bw_gbs": 819.0, "bf16_tflops": 197.0},
    {"gen": "v5p", "hbm_gib": 95.0, "bw_gbs": 2765.0, "bf16_tflops": 459.0},
    {"gen": "v6e", "hbm_gib": 32.0, "bw_gbs": 1638.0, "bf16_tflops": 918.0},
]
for s in TPU_SPECS:
    s["ridge_flops_per_byte"] = round((s["bf16_tflops"] * 1e12) / (s["bw_gbs"] * 1e9), 1)


# Step 2 — Estimate training-state and KV-cache HBM budgets across precisions
def estimate_memory_gib(params_billions, kv_tokens, layers=32, kv_heads=8, head_dim=128):
    n_params = params_billions * 1e9
    kv_elements = 2.0 * layers * kv_tokens * kv_heads * head_dim
    gib = float(1024 ** 3)
    return {
        "fp32_train_state_gib": round((n_params * 16.0) / gib, 3),
        "mixed_bf16_train_state_gib": round((n_params * 18.0) / gib, 3),
        "kv_fp32_gib": round((kv_elements * 4.0) / gib, 3),
        "kv_bf16_gib": round((kv_elements * 2.0) / gib, 3),
        "kv_int8_gib": round((kv_elements * 1.0) / gib, 3),
    }


def per_device_hbm_ledger(total_state_gib, chips, chip_hbm_gib, placement, usable_fraction=0.8, activation_gib_per_chip=0.0):
    if chips < 1 or not 0 < usable_fraction <= 1 or activation_gib_per_chip < 0:
        raise ValueError("chips and utilization must be positive; activation allowance cannot be negative")
    if placement == "replicated":
        state_per_chip = float(total_state_gib)
    elif placement == "evenly_sharded_ideal":
        state_per_chip = float(total_state_gib) / chips
    else:
        raise ValueError("placement must be replicated or evenly_sharded_ideal")
    usable_hbm_per_chip = float(chip_hbm_gib) * usable_fraction
    available_for_state = usable_hbm_per_chip - activation_gib_per_chip
    return {"placement_assumption": placement, "chips": chips, "state_per_chip_gib": state_per_chip,
            "activation_allowance_per_chip_gib": activation_gib_per_chip,
            "usable_hbm_per_chip_gib": usable_hbm_per_chip,
            "fits_under_assumptions": state_per_chip <= available_for_state,
            "remaining_after_state_before_activation_gib": usable_hbm_per_chip - state_per_chip}



budget_7b = estimate_memory_gib(params_billions=7.0, kv_tokens=8192)
budget_1p5b = estimate_memory_gib(params_billions=1.5, kv_tokens=4096)
assert budget_7b["mixed_bf16_train_state_gib"] > 64.0 and budget_7b["mixed_bf16_train_state_gib"] < 128.0
# Assert invariant `budget_1p5b["mixed_bf16_train_state_gib"] < 64.0` holds
assert budget_1p5b["mixed_bf16_train_state_gib"] < 64.0
print("TPU ridge points (FLOP/byte):", {s["gen"]: s["ridge_flops_per_byte"] for s in TPU_SPECS})
print("7B memory budget (GiB):", budget_7b)
print("1.5B memory budget (GiB):", budget_1p5b)
```

Expected: Prints the ridge-point FLOP/byte ratios for v5e (240.5), v5p (166.0), and v6e (560.4) along with the 7B and 1.5B memory budgets in GiB. Placement-aware sizing reports per-chip state for replication and ideal even sharding, then subtracts explicit activation allowance from the 80% usable-HBM budget. These are planning bounds; runtime array shardings and peak memory must be checked on the selected target.

## TPU capacity, ridge point and per-device placement bounds

**Predict:** How do per-chip HBM capacity (GiB) and the compute-to-memory ridge point (FLOP/byte) change across TPU `v5e`, `v5p`, and `v6e`?

![TPU capacity, ridge point and per-device placement bounds](../outputs/figure.svg)

**Analytic calculation; no device result recorded**

### Read the figure

The first panel's vertical axis is HBM per physical chip in GiB: v5e 16, v5p 95 and v6e 32. The second is theoretical BF16 ridge point in FLOP/byte: 240.5, 166.0 and 560.4. v5p has more per-chip HBM than v6e but a lower ridge point; a larger ridge value means more arithmetic per HBM byte is needed to reach the compute ceiling, not that the workload is faster. In the third panel, the 48 GiB state on four v5e chips is 48 GiB/device if replicated and 12 GiB/device if ideally sharded, against 12.8 GiB/device usable. Adding the fixture's 1 GiB activation allowance makes the shard 13 GiB and exceeds budget by 0.2 GiB. Choose smaller state, more chips or a validated placement, then check measured peak memory.

These are calculated capacity scenarios, not device measurements.

### Connect it to the computation

These are specification and fixture calculations, not measured target behavior. The first two panels compare hardware capacity and roofline balance; the third makes the placement decision explicit. The later 7B mixed-BF16 exercise estimates 29.33675 GiB/device on four v6e chips versus a 25.6 GiB usable budget, while eight chips need 14.668375 GiB/device and leave 10.931625 GiB before activations. Its assumed 8 GiB allowance fits arithmetically. Confirm actual sharding and live peak memory before selecting a slice; assess the separate 1 GiB BF16 inference KV cache only for inference.

```python
visual_data = {
    'kind': 'bar',
    'panels': [
        {'kind': 'bar', 'labels': [s['gen'] for s in TPU_SPECS], 'xlabel': 'Cloud TPU generation', 'ylabel': 'HBM per chip (GiB)',
         'series': [{'label': 'HBM capacity', 'y': [float(s['hbm_gib']) for s in TPU_SPECS]}]},
        {'kind': 'bar', 'labels': [s['gen'] for s in TPU_SPECS], 'xlabel': 'Cloud TPU generation', 'ylabel': 'theoretical BF16 ridge point (FLOP/byte)',
         'series': [{'label': 'peak BF16 FLOP/s ÷ HBM bytes/s', 'y': [float(s['ridge_flops_per_byte']) for s in TPU_SPECS]}]},
        {'kind': 'bar', 'labels': ['replicated', 'ideal even shard', '80% usable HBM'],
         'xlabel': '48 GiB state on four v5e chips', 'ylabel': 'GiB per device',
         'series': [{'label': 'per-device requirement or budget', 'y': [48.0, 12.0, 12.8]}]},
    ],
}
assert 48.0 / 4 == 12.0 and 16.0 * 0.8 == 12.8
```

## Recorded reference execution

CPU run: 2026-10-27T17:21:17.931462+00:00. JAX 0.9.2.

```text
TPU ridge points (FLOP/byte): {'v5e': 240.5, 'v5p': 166.0, 'v6e': 560.4}
7B memory budget (GiB): {'fp32_train_state_gib': 104.308, 'mixed_bf16_train_state_gib': 117.347, 'kv_fp32_gib': 2.0, 'kv_bf16_gib': 1.0, 'kv_int8_gib': 0.5}
1.5B memory budget (GiB): {'fp32_train_state_gib': 22.352, 'mixed_bf16_train_state_gib': 25.146, 'kv_fp32_gib': 1.0, 'kv_bf16_gib': 0.5, 'kv_int8_gib': 0.25}
state/device, replicated/sharded/changed: 48.0 12.0 13.0
TPU ridge points (FLOP/byte): {'v5e': 240.5, 'v5p': 166.0, 'v6e': 560.4}
7B memory budget (GiB): {'fp32_train_state_gib': 104.308, 'mixed_bf16_train_state_gib': 117.347, 'kv_fp32_gib': 2.0, 'kv_bf16_gib': 1.0, 'kv_int8_gib': 0.5}
1.5B memory budget (GiB): {'fp32_train_state_gib': 22.352, 'mixed_bf16_train_state_gib': 25.146, 'kv_fp32_gib': 1.0, 'kv_bf16_gib': 0.5, 'kv_int8_gib': 0.25}
KV cache GiB (FP32 / BF16 / INT8): 2.0 1.0 0.5
Ideal-sharding arithmetic minimum: 10
At 16 chips, state/device GiB: 7.3341875 remaining before activations: 5.465812500000001
Per-chip ledgers: {'placement_assumption': 'replicated', 'chips': 4, 'state_per_chip_gib': 48.0, 'activation_allowance_per_chip_gib': 1, 'usable_hbm_per_chip_gib': 12.8, 'fits_under_assumptions': False, 'remaining_after_state_before_activation_gib': -35.2} {'placement_assumption': 'evenly_sharded_ideal', 'chips': 4, 'state_per_chip_gib': 12.0, 'activation_allowance_per_chip_gib': 1, 'usable_hbm_per_chip_gib': 12.8, 'fits_under_assumptions': False, 'remaining_after_state_before_activation_gib': 0.8000000000000007} {'placement_assumption': 'evenly_sharded_ideal', 'chips': 4, 'state_per_chip_gib': 13.0, 'activation_allowance_per_chip_gib': 1, 'usable_hbm_per_chip_gib': 12.8, 'fits_under_assumptions': False, 'remaining_after_state_before_activation_gib': -0.1999999999999993}
v6e chips: 4 state/device GiB: 29.33675 usable GiB: 25.6 fits with assumed activation: False
v6e chips: 8 state/device GiB: 14.668375 usable GiB: 25.6 fits with assumed activation: True
Separate inference BF16 KV cache GiB: 1.0
Arithmetic intensity: 341.3333333333333 FLOPs/byte
4 chips: {'placement_assumption': 'evenly_sharded_ideal', 'chips': 4, 'state_per_chip_gib': 29.33675, 'activation_allowance_per_chip_gib': 8.0, 'usable_hbm_per_chip_gib': 25.6, 'fits_under_assumptions': False, 'remaining_after_state_before_activation_gib': -3.736749999999997}
8 chips: {'placement_assumption': 'evenly_sharded_ideal', 'chips': 8, 'state_per_chip_gib': 14.668375, 'activation_allowance_per_chip_gib': 8.0, 'usable_hbm_per_chip_gib': 25.6, 'fits_under_assumptions': True, 'remaining_after_state_before_activation_gib': 10.931625000000002}
Separate inference KV GiB: 1.0
PASS: tpu-03

```

## Compare KV-cache memory across FP32, BF16, and INT8

**Predict before running:** By what factor does KV-cache memory drop when moving from FP32 to BF16 and INT8?

```python
# Experiment — Compare KV-cache memory across FP32, BF16, and INT8
assert np.isclose(budget_7b["kv_fp32_gib"], 2.0 * budget_7b["kv_bf16_gib"])
assert np.isclose(budget_7b["kv_bf16_gib"], 2.0 * budget_7b["kv_int8_gib"])
print("KV cache GiB (FP32 / BF16 / INT8):", budget_7b["kv_fp32_gib"], budget_7b["kv_bf16_gib"], budget_7b["kv_int8_gib"])
```

**Expected:** For 8192 tokens, KV cache drops from 2.0 GiB (FP32) to 1.0 GiB (BF16) and 0.5 GiB (INT8).

This tensor-element ledger halves storage at each step. A real INT8 cache also needs scales and supported kernels; actual decode traffic and latency depend on reuse and the implementation. This does not measure quantized inference.

## Estimate minimum v5e chips under ideal even sharding

**Predict before running:** Under even state sharding and an 80% per-chip HBM budget, what is the arithmetic minimum number of 16 GiB v5e chips? What would a supported 16-chip request leave per chip before activations?

```python
state_gib = budget_7b["mixed_bf16_train_state_gib"]
usable_gib_per_chip = 16.0 * 0.8
minimum_ideal_chips = int(np.ceil(state_gib / usable_gib_per_chip))
state_per_chip_on_16 = state_gib / 16
remaining_before_activations = usable_gib_per_chip - state_per_chip_on_16
print("Ideal-sharding arithmetic minimum:", minimum_ideal_chips)
print("At 16 chips, state/device GiB:", state_per_chip_on_16, "remaining before activations:", remaining_before_activations)
assert minimum_ideal_chips == 10
assert state_per_chip_on_16 < usable_gib_per_chip
assert remaining_before_activations > 0
```

**Expected:** The arithmetic lower bound is 10 chips if the entire 117.347 GiB state is evenly sharded. The illustrative 16-chip v5e layout needs about 7.33 GiB/device, leaving about 5.47 GiB/device below the 80% budget for all activations and temporary buffers.

This is only a capacity bound under ideal even sharding, not a proof of a 10-chip supported slice or of actual placement. The selected Cloud TPU API/GKE topology determines which slice shapes exist. Verify parameter and optimizer-state shardings, plus peak activations and temporary memory, before launch.

## Decide between replicated and sharded placement with headroom

**Predict before running:** Does a 48 GiB state fit per 16 GiB v5e chip when replicated, ideally sharded over four chips, and when each chip also needs 1 GiB for activations? What changes at 52 GiB?

```python
replicated = per_device_hbm_ledger(48, 4, 16, "replicated", activation_gib_per_chip=1)
ideal_shard = per_device_hbm_ledger(48, 4, 16, "evenly_sharded_ideal", activation_gib_per_chip=1)
changed = per_device_hbm_ledger(52, 4, 16, "evenly_sharded_ideal", activation_gib_per_chip=1)
print("Per-chip ledgers:", replicated, ideal_shard, changed)
assert replicated["state_per_chip_gib"] == 48.0 and not replicated["fits_under_assumptions"]
assert ideal_shard["state_per_chip_gib"] == 12.0 and not ideal_shard["fits_under_assumptions"]
assert changed["state_per_chip_gib"] == 13.0 and not changed["fits_under_assumptions"]
```

**Expected:** Replication requires 48 GiB/device and fails. Ideal sharding requires 12 GiB/device, but with a 1 GiB activation allowance it exceeds the 12.8 GiB usable budget; 52 GiB sharded requires 13 GiB/device and fails before activations.

This is conditional on placement. Confirm array shardings and optimizer-state placement before treating the ideal-sharded estimate as a fit.

## Make it yours

For a 7B mixed-BF16 training state, compare ideal evenly sharded v6e-4 and v6e-8 plans at 80% HBM per chip. Report per-chip state and remaining HBM before activations, then decide whether each can fit under an additional 8 GiB/device activation-and-temporary allowance. Separately report the BF16 inference KV cache at 8192 tokens. State the placement assumption and do not add inference KV cache to training state.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Inputs:**
- The public memory estimator, a 1.5-billion parameter training case, 4096 KV tokens, and an inference cache scenario.

**Outputs:**
- Training state estimate, slice-budget comparisons, and a separate BF16 KV-cache estimate.

**Invariants to check:**
- Training state and inference KV cache are distinct ledgers; aggregate slice HBM does not ensure each sharded device fits.

**Hints:**
- Read and preserve the estimator's dtype and exclusion assumptions.
- Report cache memory separately from training-state memory.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp
import numpy as np

def audit_training_and_inference_memory(estimate_memory_gib):
    # TODO: evaluate the training case and compute a separately labeled inference-cache estimate.
    raise NotImplementedError
```

<details><summary>Reference solution</summary>

```python
state = budget_7b["mixed_bf16_train_state_gib"]
for chips in (4, 8):
    ledger = per_device_hbm_ledger(state, chips, 32.0, "evenly_sharded_ideal", activation_gib_per_chip=8.0)
    print("v6e chips:", chips, "state/device GiB:", ledger["state_per_chip_gib"], "usable GiB:", ledger["usable_hbm_per_chip_gib"], "fits with assumed activation:", ledger["fits_under_assumptions"])
    assert ledger["fits_under_assumptions"] is (chips == 8)
print("Separate inference BF16 KV cache GiB:", budget_7b["kv_bf16_gib"])
assert np.isclose(budget_7b["kv_bf16_gib"], 1.0)
```

</details>

## Compare arithmetic intensity against the TPU v5e and v6e ridge points

**Foundations**

Compute the arithmetic intensity (FLOPs/byte) of a `bfloat16` square matrix multiply of size $N = 1024$ (`2 * N^3` FLOPs divided by `3 * 2 * N^2` bytes) and check whether it exceeds the `v5e` ridge point.

<details><summary>Hint</summary>

For two `N x N` BF16 inputs and one BF16 output, arithmetic intensity simplifies to `N / 3.0` FLOPs/byte.

</details>

### How to write: Compare arithmetic intensity against the TPU v5e and v6e ridge points — Step-by-step recipe & starter scaffold

**Inputs:**
- Square BF16 matrix dimension 1024 and the documented v5e and v6e ridge points.

**Outputs:**
- Arithmetic intensity in FLOPs per byte and a comparison with each ridge point.

**Invariants to check:**
- FLOPs and bytes use the same operation count as the lesson; intensity is not device memory capacity.

**Hints:**
- Simplify the given expression before evaluating it.
- Compare with ridge points in matching FLOPs-per-byte units.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp
import numpy as np

def compare_matmul_intensity(size, ridge_points):
    # TODO: compute the stated BF16 intensity and report which ridge points it exceeds.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
n = 1024
intensity = (2 * n**3) / (3 * 2 * n**2)
print("Arithmetic intensity:", intensity, "FLOPs/byte")
assert intensity == n / 3.0
assert intensity > TPU_SPECS[0]["ridge_flops_per_byte"]
```

The idealized operation has about 341.3 FLOP/byte. It is above v5e’s 240.5 ridge and below v6e’s 560.4 ridge, assuming exactly the stated two BF16 input reads and one BF16 output write. That assigns different ideal ceilings; padding, actual output dtype, cache reuse and measured traffic can change the comparison.

</details>

## Compare v6e-4 and v6e-8 per-device training placement

**Transfer / diagnosis**

Using the ledger helper, compare a 7B mixed-BF16 state under ideal sharding on v6e-4 and v6e-8, with an assumed 8 GiB/device activation and temporary allowance. Keep the 8192-token inference KV cache separate.

<details><summary>Hint</summary>

Use 32 GiB physical HBM per v6e chip and the 80% usable fraction. Divide state only under the explicit ideal-sharding assumption.

</details>

### How to write: Compare v6e-4 and v6e-8 per-device training placement — Step-by-step recipe & starter scaffold

**Inputs:**
- 7B mixed-BF16 training-state estimate, 4 or 8 v6e chips at 32 GiB/chip, 80% usable-HBM fraction, and 8 GiB/device activations plus temporary allowance.

**Outputs:**
- Per-device state and remaining capacity for v6e-4 and v6e-8 under ideal even sharding, plus separate BF16 inference KV bytes.

**Invariants to check:**
- Divide training state by chip count only under explicit ideal-sharding assumption. Do not add inference KV cache to training state.

**Hints:**
- Use the per-device ledger helper and report state/device separately from usable HBM/device.
- Try a 48 GiB versus 52 GiB state in the helper to observe an input change.

**Starter code scaffold (fill in the TODOs):**

```python
def compare_v6e_placement(training_state_gib, kv_cache_gib, chips_options, activation_gib_per_chip):
    # TODO: return explicit ideal-sharding per-device ledgers for each topology
    # and keep inference KV cache in a separate result.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
state = budget_7b["mixed_bf16_train_state_gib"]
ledgers = {chips: per_device_hbm_ledger(state, chips, 32.0, "evenly_sharded_ideal", activation_gib_per_chip=8.0) for chips in (4, 8)}
for chips, row in ledgers.items():
    print(chips, "chips:", row)
assert ledgers[4]["fits_under_assumptions"] is False
assert ledgers[8]["fits_under_assumptions"] is True
print("Separate inference KV GiB:", budget_7b["kv_bf16_gib"])
assert np.isclose(budget_7b["kv_bf16_gib"], 1.0)
```

The 117 GiB state needs about 29.3 GiB/device on four chips, exceeding 25.6 GiB usable HBM before activations. Eight-chip ideal sharding needs about 14.7 GiB/device; after the assumed 8 GiB activation/temporary allowance it remains below 25.6 GiB. Inference KV is separate.

</details>

## Check your understanding

What does the theoretical BF16 Roofline ridge point, $1000 \times \text{TFLOP/s} / \text{GB/s}$ in FLOP/byte, describe?

1. How many SSH keys can connect to the TPU VM
2. The arithmetic intensity where the ideal HBM-bandwidth ceiling meets the ideal BF16-compute ceiling
3. How long `gcloud compute tpus tpu-vm create` takes to provision

<details><summary>Answer and explanation</summary>

**1. How many SSH keys can connect to the TPU VM**

A roofline ratio describes compute and memory rates. Authentication and SSH limits belong to a different control-plane contract.

**2. The arithmetic intensity where the ideal HBM-bandwidth ceiling meets the ideal BF16-compute ceiling (correct)**

Convert both rates to FLOP/s and byte/s first. The ratio locates the intersection of two ideal ceilings. Below it the bandwidth ceiling is lower; above it the compute ceiling is lower. Actual performance can fall below either ceiling because of tiling, communication, vector work or launch overhead.

**3. How long `gcloud compute tpus tpu-vm create` takes to provision**

Provisioning wait depends on capacity and scheduling, not this compute-to-bandwidth ratio. Keep capacity planning separate from kernel analysis.

</details>

## Diagnose the result

If your TPU job fails with `RESOURCE_EXHAUSTED: Out of memory while trying to allocate...`, compare your parameter + optimizer + activation + KV-cache byte estimate against per-chip HBM (`16` GiB on `v5e`, `32` GiB on `v6e`) and shard the state across chips (`NamedSharding` / FSDP) or reduce microbatch size.

## Carry forward

- Size your TPU slice (`v5e`, `v5p`, `v6e`) from explicit HBM byte budgets for training state and KV caches before launching.
- Use the Roofline comparison to propose a bottleneck hypothesis, then check actual target traces, traffic, precision and synchronized timing before choosing fusion, quantization or a tile.

## Keep your evidence

Keep aggregate and per-device state ledgers, replication/sharding/headroom assumptions, invalid aggregate-fit counterexamples and separate training-versus-inference capacity decisions.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [Cloud TPU v5e, v5p, and v6e (Trillium) specifications](https://cloud.google.com/tpu/docs/v6e)
- [Cloud TPU performance guide](https://cloud.google.com/tpu/docs/performance-guide)
- [TPU7x cross-generation specifications and chiplet exposure (checked 2026-10-21)](https://docs.cloud.google.com/tpu/docs/tpu7x)
- [JAX TPU hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html)
- [JAX TPU info API](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.TpuInfo.html)
- [Cloud TPU machine types and physical chips per VM](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines)
- [Cloud TPU v5e topologies](https://docs.cloud.google.com/tpu/docs/v5e)

