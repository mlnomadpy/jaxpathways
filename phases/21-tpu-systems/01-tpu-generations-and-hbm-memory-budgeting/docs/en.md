# TPU generations and HBM memory budgeting

Phase 21: TPU Systems: Generations, Memory, Precision & XProf · about 25 minutes · CPU

## What you will be able to do

- Compare Google Cloud TPU generations (`v5e`, `v5p`, `v6e` Trillium, and Ironwood) by HBM capacity, HBM bandwidth, and MXU BF16 throughput.
- Calculate the Roofline ridge point (`bf16_tflops / bw_gbs`) in FLOPs per byte for each TPU generation.
- Compute training HBM (weights + gradients + Adam moments) and inference KV-cache HBM across `fp32`, `bfloat16`, and `int8`.
- Select the minimum single-host or Pod TPU slice that fits a given model's training state with activation headroom.

## The problem

Provisioning and launching a TPU VM is only half the engineering job: if you pick the wrong TPU generation or miscalculate parameter + optimizer + KV-cache HBM bytes, your TPU experiment will either fail with `RESOURCE_EXHAUSTED: Out of memory` or under-utilize the Matrix Multiply Unit (MXU).

## The idea

Before scaling a Transformer or distributed job on TPU, compare per-chip HBM capacity and the BF16 compute-to-memory ridge point (`FLOP/byte`) across TPU generations (`v5e`, `v5p`, `v6e` Trillium, and Ironwood), and calculate the exact GiB budget for training state (weights, gradients, Adam moments) and inference KV cache across `fp32`, `bfloat16`, and `int8`.

## Connect MXU arithmetic intensity and HBM capacity to TPU slice sizing

Each TPU chip pairs a Matrix Multiply Unit (MXU) designed for high-throughput `bfloat16` / `int8` matrix multiplications with High Bandwidth Memory (HBM). A TPU v5e chip provides $16$ GiB of HBM at $819$ GB/s and $197$ TFLOP/s of `bfloat16` compute, giving a ridge-point balance of:
$$

\text{Ridge}_{\text{v5e}} = \frac{197 \times 10^{12}\text{ FLOP/s}}{819 \times 10^9\text{ B/s}} \approx 240.5\text{ FLOP/byte}

$$
Operations whose arithmetic intensity is below this ridge point (such as batch-$1$ autoregressive token decoding) are HBM-bandwidth bound; operations above it (such as large batched matmuls) are MXU-compute bound.

Before launching a run, you also need to verify HBM capacity. In mixed-precision Adam training, each parameter typically requires $12$ to $16$ bytes of persistent state (master `fp32` weights + `fp32` first and second Adam moments + `bf16` or `fp32` gradients), plus activation memory and any KV cache.

$$
\text{Throughput}_{\text{tok/s}} = \frac{B_{\text{global}} \times L_{\text{seq}}}{T_{\text{step}}}
$$

### TPU HBM-to-MXU dataflow and mixed-precision state budget

**Predict:** Which tensors stay in `float32` in HBM and which operands stream as `bfloat16` into the MXU?

![TPU HBM-to-MXU dataflow and mixed-precision state budget](../outputs/mechanism.svg)

*Conceptual / analytic teaching diagram; not a recorded benchmark.*

Read top to bottom: HBM stores master `fp32` parameters and Adam moments (`m`, `v`) alongside `bf16` activations and KV cache; operands are cast to `bf16` (or `int8`) to stream across the HBM bandwidth bottleneck into the MXU, which accumulates dot products in `fp32` before writing updated state back.

### Pause and reason

Why does a $7$B-parameter model's mixed-precision Adam training state (`~78.2` GiB before activations) fail with OOM on a single-host `v5litepod-4` (`64` GiB total HBM)?

<details><summary>Compare your reasoning</summary>

`4` TPU v5e chips provide `4 * 16 = 64` GiB of total HBM, which is smaller than the `~78.2` GiB needed just for parameters, gradients, and Adam moments. You need at least `v5litepod-8` (`128` GiB) or `v6e-4` (`128` GiB) with sharding across chips.

</details>

## 1. Compare TPU generations (`v5e`, `v5p`, `v6e`, Ironwood) and calculate HBM budgets

Published per-chip TPU specifications determine which slice fits your model:

- **TPU v5e (`v5litepod`)**: $16$ GiB HBM per chip, $819$ GB/s HBM bandwidth, $197$ BF16 TFLOP/s (`240.5` FLOP/byte ridge point). Single-host slices: `1`, `4` (`64` GiB), or `8` (`128` GiB) chips.
- **TPU v5p**: $95$ GiB HBM per chip, $2765$ GB/s HBM bandwidth, $459$ BF16 TFLOP/s (`166.0` FLOP/byte ridge point).
- **TPU v6e (Trillium)**: $32$ GiB HBM per chip, $1600$ GB/s HBM bandwidth, $918$ BF16 TFLOP/s (`573.8` FLOP/byte ridge point). Single-host slices: `1`, `4` (`128` GiB), or `8` (`256` GiB) chips.
- **TPU Ironwood (7th-gen scale)**: $192$ GiB HBM per chip, $7370$ GB/s HBM bandwidth.

**Run the TPU generation & HBM budgeting verifier on your TPU VM or workstation**

```bash
# Run the TPU generation & HBM budgeting verifier on your TPU VM or workstation
gcloud compute tpus tpu-vm scp public/exercises/tpu-03.py "$TPU_NAME":~/jax-tpu-lab/ --zone="$ZONE"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
  --command="JAX_PLATFORMS=tpu ~/jax-tpu-lab/.venv/bin/python ~/jax-tpu-lab/tpu-03.py"
```

**Expected:** Prints the TPU ridge points (FLOP/byte) and the 7B and 1.5B training + KV-cache HBM budgets in GiB.

## Define TPU generation specs and compute Roofline ridge points

Create main.py with TPU_SPECS and calculate ridge_flops_per_byte for v5e, v5p, and v6e.

```python
# Step 1 — Define TPU generation specs and compute Roofline ridge points: Dividing peak BF16 FLOP/s by HBM bandwidth (B/s) gives the minimum...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Compute `TPU_SPECS` from `[`
TPU_SPECS = [
    {"gen": "v5e", "hbm_gib": 16.0, "bw_gbs": 819.0, "bf16_tflops": 197.0},
    {"gen": "v5p", "hbm_gib": 95.0, "bw_gbs": 2765.0, "bf16_tflops": 459.0},
    {"gen": "v6e", "hbm_gib": 32.0, "bw_gbs": 1600.0, "bf16_tflops": 918.0},
]
# Iterate over `s` to step through the computation:
for s in TPU_SPECS:
    # Run `round` to compute `s['ridge_flops_per_byte']`.
    s["ridge_flops_per_byte"] = round((s["bf16_tflops"] * 1e12) / (s["bw_gbs"] * 1e9), 1)
```

Dividing peak BF16 FLOP/s by HBM bandwidth (B/s) gives the minimum FLOPs per byte needed to saturate the MXU.

## Estimate training-state and KV-cache HBM budgets across precisions

Append estimate_memory_gib and compare 7B vs 1.5B model memory footprints.

```python
# Step 2 — Estimate training-state and KV-cache HBM budgets across precisions: Computing both training state and KV-cache GiB upfront prevents...
def estimate_memory_gib(params_billions, kv_tokens, layers=32, kv_heads=8, head_dim=128):
    # Compute `n_params` from `params_billions * 1e9`
    n_params = params_billions * 1e9
    # Compute `kv_elements` from `2.0 * layers * kv_tokens * kv_heads * head_dim`
    kv_elements = 2.0 * layers * kv_tokens * kv_heads * head_dim
    # Evaluate `1024 ** 3` and convert the result into Python scalar/collection `gib`.
    gib = float(1024 ** 3)
    # Return `{'fp32_train_state_gib': round(n_params * 16.0 / gib, 3), 'mixed_bf16_train_state_gib': round(n_params * 12.0 / gib, 3), 'kv_fp32_gib': round(kv_elements * 4.0 / gib, 3), 'kv_bf16_gib': round(kv_elements * 2.0 / gib, 3), 'kv_int8_gib': round(kv_elements * 1.0 / gib, 3)}` to the caller.
    return {
        "fp32_train_state_gib": round((n_params * 16.0) / gib, 3),
        "mixed_bf16_train_state_gib": round((n_params * 12.0) / gib, 3),
        "kv_fp32_gib": round((kv_elements * 4.0) / gib, 3),
        "kv_bf16_gib": round((kv_elements * 2.0) / gib, 3),
        "kv_int8_gib": round((kv_elements * 1.0) / gib, 3),
    }


# Run `estimate_memory_gib` to compute `budget_7b`.
budget_7b = estimate_memory_gib(params_billions=7.0, kv_tokens=8192)
# Run `estimate_memory_gib` to compute `budget_1p5b`.
budget_1p5b = estimate_memory_gib(params_billions=1.5, kv_tokens=4096)
# Assert invariant `budget_7b["mixed_bf16_train_state_gib"] > 64.0 and budget_7b["mix...` holds
assert budget_7b["mixed_bf16_train_state_gib"] > 64.0 and budget_7b["mixed_bf16_train_state_gib"] < 128.0
# Assert invariant `budget_1p5b["mixed_bf16_train_state_gib"] < 64.0` holds
assert budget_1p5b["mixed_bf16_train_state_gib"] < 64.0
# Print the observed values to compare against the expected result.
print("TPU ridge points (FLOP/byte):", {s["gen"]: s["ridge_flops_per_byte"] for s in TPU_SPECS})
# Print diagnostic summary of the computed outputs.
print("7B memory budget (GiB):", budget_7b)
# Print diagnostic summary of the computed outputs.
print("1.5B memory budget (GiB):", budget_1p5b)
```

Computing both training state and KV-cache GiB upfront prevents OOM surprises when selecting between `v5litepod-4` (`64` GiB) and `v5litepod-8` / `v6e-4` (`128` GiB).

## Step 3: Verify invariants on the completed state

Run the final shape and numerical assertions to confirm the state built in Steps 1 and 2.

```python
assert budget_7b["mixed_bf16_train_state_gib"] > 64.0 and budget_7b["mixed_bf16_train_state_gib"] < 128.0
assert budget_1p5b["mixed_bf16_train_state_gib"] < 64.0
```

Checking these invariants confirms the computation is ready for the full worked experiment.

## Run the example

```python
# Step 1 — Define TPU generation specs and compute Roofline ridge points: Dividing peak BF16 FLOP/s by HBM bandwidth (B/s) gives the minimum...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Compute `TPU_SPECS` from `[`
TPU_SPECS = [
    {"gen": "v5e", "hbm_gib": 16.0, "bw_gbs": 819.0, "bf16_tflops": 197.0},
    {"gen": "v5p", "hbm_gib": 95.0, "bw_gbs": 2765.0, "bf16_tflops": 459.0},
    {"gen": "v6e", "hbm_gib": 32.0, "bw_gbs": 1600.0, "bf16_tflops": 918.0},
]
# Iterate over `s` to step through the computation:
for s in TPU_SPECS:
    # Run `round` to compute `s['ridge_flops_per_byte']`.
    s["ridge_flops_per_byte"] = round((s["bf16_tflops"] * 1e12) / (s["bw_gbs"] * 1e9), 1)


# Step 2 — Estimate training-state and KV-cache HBM budgets across precisions: Computing both training state and KV-cache GiB upfront prevents...
def estimate_memory_gib(params_billions, kv_tokens, layers=32, kv_heads=8, head_dim=128):
    # Compute `n_params` from `params_billions * 1e9`
    n_params = params_billions * 1e9
    # Compute `kv_elements` from `2.0 * layers * kv_tokens * kv_heads * head_dim`
    kv_elements = 2.0 * layers * kv_tokens * kv_heads * head_dim
    # Evaluate `1024 ** 3` and convert the result into Python scalar/collection `gib`.
    gib = float(1024 ** 3)
    # Return `{'fp32_train_state_gib': round(n_params * 16.0 / gib, 3), 'mixed_bf16_train_state_gib': round(n_params * 12.0 / gib, 3), 'kv_fp32_gib': round(kv_elements * 4.0 / gib, 3), 'kv_bf16_gib': round(kv_elements * 2.0 / gib, 3), 'kv_int8_gib': round(kv_elements * 1.0 / gib, 3)}` to the caller.
    return {
        "fp32_train_state_gib": round((n_params * 16.0) / gib, 3),
        "mixed_bf16_train_state_gib": round((n_params * 12.0) / gib, 3),
        "kv_fp32_gib": round((kv_elements * 4.0) / gib, 3),
        "kv_bf16_gib": round((kv_elements * 2.0) / gib, 3),
        "kv_int8_gib": round((kv_elements * 1.0) / gib, 3),
    }


# Run `estimate_memory_gib` to compute `budget_7b`.
budget_7b = estimate_memory_gib(params_billions=7.0, kv_tokens=8192)
# Run `estimate_memory_gib` to compute `budget_1p5b`.
budget_1p5b = estimate_memory_gib(params_billions=1.5, kv_tokens=4096)
# Assert invariant `budget_7b["mixed_bf16_train_state_gib"] > 64.0 and budget_7b["mix...` holds
assert budget_7b["mixed_bf16_train_state_gib"] > 64.0 and budget_7b["mixed_bf16_train_state_gib"] < 128.0
# Assert invariant `budget_1p5b["mixed_bf16_train_state_gib"] < 64.0` holds
assert budget_1p5b["mixed_bf16_train_state_gib"] < 64.0
# Print the observed values to compare against the expected result.
print("TPU ridge points (FLOP/byte):", {s["gen"]: s["ridge_flops_per_byte"] for s in TPU_SPECS})
# Print diagnostic summary of the computed outputs.
print("7B memory budget (GiB):", budget_7b)
# Print diagnostic summary of the computed outputs.
print("1.5B memory budget (GiB):", budget_1p5b)
```

Expected: Prints the ridge-point FLOP/byte ratios for v5e (240.5), v5p (166.0), and v6e (573.8) along with the 7B and 1.5B memory budgets in GiB.

## Per-chip HBM capacity and ridge-point arithmetic intensity across TPU generations

**Predict:** How do per-chip HBM capacity (GiB) and the compute-to-memory ridge point (FLOP/byte) change across TPU `v5e`, `v5p`, and `v6e`?

![Per-chip HBM capacity and ridge-point arithmetic intensity across TPU generations](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis compares three Cloud TPU generations (`v5e`, `v5p`, and `v6e`). For each generation, the two bars show per-chip HBM capacity in GiB and the BF16 ridge point in FLOPs per byte (`bf16_tflops / bw_gbs`).

### Connect it to the computation

`v5p` maximizes HBM capacity per chip (`95` GiB) and lowers the ridge point (`166` FLOP/byte), while `v6e` doubles `v5e`'s HBM (`32` GiB vs `16` GiB) and reaches `918` BF16 TFLOP/s (`573.8` FLOP/byte ridge point), rewarding fused kernels and batch sizes that keep the MXU fed.

```python
# Compute figure data for: Per-chip HBM capacity and ridge-point arithmetic intensity across TPU generations
# Compute `visual_data` from `{`
visual_data = {
    'kind': 'bar',
    'labels': [s['gen'] for s in TPU_SPECS],
    'xlabel': 'Cloud TPU generation',
    'ylabel': 'GiB / (FLOP per byte)',
    'series': [
        {'label': 'HBM per chip (GiB)', 'y': [float(s['hbm_gib']) for s in TPU_SPECS]},
        {'label': 'BF16 ridge point (FLOP/byte)', 'y': [float(s['ridge_flops_per_byte']) for s in TPU_SPECS]},
    ],
}
```

## Recorded reference execution

CPU run: 2026-10-08T14:08:21.911895+00:00. JAX 0.9.2.

```text
TPU ridge points (FLOP/byte): {'v5e': 240.5, 'v5p': 166.0, 'v6e': 573.8}
7B memory budget (GiB): {'fp32_train_state_gib': 104.308, 'mixed_bf16_train_state_gib': 78.231, 'kv_fp32_gib': 2.0, 'kv_bf16_gib': 1.0, 'kv_int8_gib': 0.5}
1.5B memory budget (GiB): {'fp32_train_state_gib': 22.352, 'mixed_bf16_train_state_gib': 16.764, 'kv_fp32_gib': 1.0, 'kv_bf16_gib': 0.5, 'kv_int8_gib': 0.25}
TPU ridge points (FLOP/byte): {'v5e': 240.5, 'v5p': 166.0, 'v6e': 573.8}
7B memory budget (GiB): {'fp32_train_state_gib': 104.308, 'mixed_bf16_train_state_gib': 78.231, 'kv_fp32_gib': 2.0, 'kv_bf16_gib': 1.0, 'kv_int8_gib': 0.5}
1.5B memory budget (GiB): {'fp32_train_state_gib': 22.352, 'mixed_bf16_train_state_gib': 16.764, 'kv_fp32_gib': 1.0, 'kv_bf16_gib': 0.5, 'kv_int8_gib': 0.25}
KV cache GiB (FP32 / BF16 / INT8): 2.0 1.0 0.5
7B mixed train state GiB: 78.231 Min v5e chips (at 80% HBM budget): 7
1.5B budget (GiB): {'fp32_train_state_gib': 22.352, 'mixed_bf16_train_state_gib': 16.764, 'kv_fp32_gib': 1.0, 'kv_bf16_gib': 0.5, 'kv_int8_gib': 0.25}
Matmul N=1024 intensity: 341.3 v5e ridge: 240.5
v6e-4 80% usable GiB: 102.4 Fits 7B state: True
PASS: tpu-03

```

## Compare KV-cache memory across FP32, BF16, and INT8

**Predict before running:** By what factor does KV-cache memory drop when moving from FP32 to BF16 and INT8?

```python
# Experiment — Compare KV-cache memory across FP32, BF16, and INT8: Halving bytes per element halves both the HBM capacity consumed...
# Verify that the numerical values match the expected reference within tolerance.
assert np.isclose(budget_7b["kv_fp32_gib"], 2.0 * budget_7b["kv_bf16_gib"])
# Check numerical equivalence within tolerance: `np.isclose(budget_7b["kv_bf16_gib"]`
assert np.isclose(budget_7b["kv_bf16_gib"], 2.0 * budget_7b["kv_int8_gib"])
# Print the observed values to compare against the expected result.
print("KV cache GiB (FP32 / BF16 / INT8):", budget_7b["kv_fp32_gib"], budget_7b["kv_bf16_gib"], budget_7b["kv_int8_gib"])
```

**Expected:** For 8192 tokens, KV cache drops from 2.0 GiB (FP32) to 1.0 GiB (BF16) and 0.5 GiB (INT8).

Halving bytes per element halves both the HBM capacity consumed by the KV cache and the HBM bytes read on every decode step.

## Verify how many v5e chips are needed to hold a 7B mixed-precision training state

**Predict before running:** Can a 4-chip TPU v5e slice (`64` GiB) hold a 7B parameter model's mixed-precision training state?

```python
# Experiment — Verify how many v5e chips are needed to hold a 7B mixed-precision training state: Calculating state GiB upfront tells you immediately whether you...
v5e_chip_gib = 16.0
# Evaluate `np.ceil(budget_7b['mixed_bf16_train_state_gib'] / (v5e_chip_gib * 0.8))` and convert the result into Python scalar/collection `min_v5e_chips`.
min_v5e_chips = int(np.ceil(budget_7b["mixed_bf16_train_state_gib"] / (v5e_chip_gib * 0.8)))
# Print the observed values to compare against the expected result.
print("7B mixed train state GiB:", budget_7b["mixed_bf16_train_state_gib"], "Min v5e chips (at 80% HBM budget):", min_v5e_chips)
# Assert invariant `min_v5e_chips > 4 and min_v5e_chips <= 8` holds
assert min_v5e_chips > 4 and min_v5e_chips <= 8
```

**Expected:** A 7B mixed-precision state takes ~78.23 GiB, requiring at least an 8-chip v5litepod-8 slice (128 GiB HBM) when leaving 20% headroom for activations.

Calculating state GiB upfront tells you immediately whether you need `v5litepod-4` (`64` GiB) or `v5litepod-8` (`128` GiB) and FSDP sharding.

## Make it yours

Compute `estimate_memory_gib` for a `1.5`B parameter model with `4096` KV tokens and verify that its mixed-precision training state fits comfortably inside a single-host `v5litepod-4` (`64` GiB total HBM).

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `estimate_memory_gib(...)` — Call `estimate_memory_gib` with your updated parameters or inputs from this lesson's workspace.
- `budget(...)` — Call `budget` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Print the observed values to compare against the expected result.
2. Assert invariant `budget_check["mixed_bf16_train_state_gib"] < 20.0 and (budget_che...` holds

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Compute estimate_memory_gib for a 1.5B parameter model with 4096 KV...
budget_check = estimate_memory_gib(...)  # TODO: compute budget_check
# Print the observed values to compare against the expected result.
print("1.5B budget (GiB):", budget_check)
# Assert invariant `budget_check["mixed_bf16_train_state_gib"] < 20.0 and (budget_che...` holds
assert budget_check["mixed_bf16_train_state_gib"]  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Compute estimate_memory_gib for a 1.5B parameter model with 4096 KV...
budget_check = estimate_memory_gib(params_billions=1.5, kv_tokens=4096)
# Print the observed values to compare against the expected result.
print("1.5B budget (GiB):", budget_check)
# Assert invariant `budget_check["mixed_bf16_train_state_gib"] < 20.0 and (budget_che...` holds
assert budget_check["mixed_bf16_train_state_gib"] < 20.0 and (budget_check["mixed_bf16_train_state_gib"] + budget_check["kv_bf16_gib"]) < 64.0
```

</details>

## Compare arithmetic intensity against the TPU v5e and v6e ridge points

**Foundations**

Compute the arithmetic intensity (FLOPs/byte) of a `bfloat16` square matrix multiply of size $N = 1024$ (`2 * N^3` FLOPs divided by `3 * 2 * N^2` bytes) and check whether it exceeds the `v5e` ridge point.

<details><summary>Hint</summary>

For two `N x N` BF16 inputs and one BF16 output, arithmetic intensity simplifies to `N / 3.0` FLOPs/byte.

</details>

### How to write: Compare arithmetic intensity against the TPU v5e and v6e ridge points — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `points(...)` — Call `points` with your updated parameters or inputs from this lesson's workspace.
- `next(...)` — Call `next` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Compare arithmetic intensity against the TPU v5e and v6e ridge points (Foundations): At N=1024, arithmetic intensity is ~341.3 FLOP/byte, which...
2. Compute `arith_intensity` from `(2.0 * (n ** 3)) / (3.0 * 2.0 * (n ** 2))`
3. Run `next` to compute `v5e_ridge`.
4. Print the observed values to compare against the expected result.
5. Assert invariant `arith_intensity > v5e_ridge` holds

**Starter code scaffold (fill in the TODOs):**

```python
# Compare arithmetic intensity against the TPU v5e and v6e ridge points (Foundations): At N=1024, arithmetic intensity is ~341.3 FLOP/byte, which...
n = ...  # TODO: compute n
# Compute `arith_intensity` from `(2.0 * (n ** 3)) / (3.0 * 2.0 * (n ** 2))`
arith_intensity = ...  # TODO: compute arith_intensity
# Run `next` to compute `v5e_ridge`.
v5e_ridge = next(...)  # TODO: compute v5e_ridge
# Print the observed values to compare against the expected result.
print("Matmul N = ...  # TODO: compute print("Matmul N
# Assert invariant `arith_intensity > v5e_ridge` holds
assert arith_intensity  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Compare arithmetic intensity against the TPU v5e and v6e ridge points (Foundations): At N=1024, arithmetic intensity is ~341.3 FLOP/byte, which...
n = 1024
# Compute `arith_intensity` from `(2.0 * (n ** 3)) / (3.0 * 2.0 * (n ** 2))`
arith_intensity = (2.0 * (n ** 3)) / (3.0 * 2.0 * (n ** 2))
# Run `next` to compute `v5e_ridge`.
v5e_ridge = next(s["ridge_flops_per_byte"] for s in TPU_SPECS if s["gen"] == "v5e")
# Print the observed values to compare against the expected result.
print("Matmul N=1024 intensity:", round(arith_intensity, 1), "v5e ridge:", v5e_ridge)
# Assert invariant `arith_intensity > v5e_ridge` holds
assert arith_intensity > v5e_ridge
```

At N=1024, arithmetic intensity is ~341.3 FLOP/byte, which is above TPU v5e's ~240.5 FLOP/byte ridge point (compute-bound).

</details>

## Size a Trillium v6e-4 slice for a 7B training state

**Transfer / diagnosis**

Compute the total HBM (in GiB) of a 4-chip `v6e-4` slice (`32` GiB/chip) and check whether `budget_7b['mixed_bf16_train_state_gib']` fits inside `80%` of that slice's HBM.

<details><summary>Hint</summary>

Compare `budget_7b['mixed_bf16_train_state_gib']` with `4 * 32.0 * 0.8`.

</details>

### How to write: Size a Trillium v6e-4 slice for a 7B training state — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `state(...)` — Call `state` with your updated parameters or inputs from this lesson's workspace.
- `GiB(...)` — Call `GiB` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Compute `fits_v6e4` from `budget_7b["mixed_bf16_train_state_gib"] <= v6e4_usab...`
2. Print the observed values to compare against the expected result.
3. Assert invariant `fits_v6e4 is True` holds

**Starter code scaffold (fill in the TODOs):**

```python
# Size a Trillium v6e-4 slice for a 7B training state (Transfer / diagnosis): Because v6e doubles per-chip HBM to 32 GiB (128 GiB across 4...
v6e4_usable_gib = ...  # TODO: compute v6e4_usable_gib
# Compute `fits_v6e4` from `budget_7b["mixed_bf16_train_state_gib"] <= v6e4_usab...`
fits_v6e4 = ...  # TODO: compute fits_v6e4
# Print the observed values to compare against the expected result.
print("v6e-4 80% usable GiB:", v6e4_usable_gib, "Fits 7B state:", fits_v6e4)
# Assert invariant `fits_v6e4 is True` holds
assert fits_v6e4 is True  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Size a Trillium v6e-4 slice for a 7B training state (Transfer / diagnosis): Because v6e doubles per-chip HBM to 32 GiB (128 GiB across 4...
v6e4_usable_gib = 4 * 32.0 * 0.8
# Compute `fits_v6e4` from `budget_7b["mixed_bf16_train_state_gib"] <= v6e4_usab...`
fits_v6e4 = budget_7b["mixed_bf16_train_state_gib"] <= v6e4_usable_gib
# Print the observed values to compare against the expected result.
print("v6e-4 80% usable GiB:", v6e4_usable_gib, "Fits 7B state:", fits_v6e4)
# Assert invariant `fits_v6e4 is True` holds
assert fits_v6e4 is True
```

Because `v6e` doubles per-chip HBM to `32` GiB (`128` GiB across 4 chips, `102.4` GiB at 80% budget), a 7B sharded training state fits on a 4-chip `v6e-4` slice.

</details>

## Check your understanding

What does a TPU generation's Roofline ridge point (`bf16_tflops / bw_gbs` in FLOPs/byte) tell you about a kernel?

1. How many SSH keys can connect to the TPU VM
2. The minimum FLOPs performed per byte transferred from HBM required to become MXU compute-bound rather than HBM bandwidth-bound
3. How long `gcloud compute tpus tpu-vm create` takes to provision

<details><summary>Answer and explanation</summary>

The minimum FLOPs performed per byte transferred from HBM required to become MXU compute-bound rather than HBM bandwidth-bound

Below the ridge point, execution speed is limited by HBM memory bandwidth; above the ridge point, execution is limited by MXU matrix compute throughput.

</details>

## Diagnose the result

If your TPU job fails with `RESOURCE_EXHAUSTED: Out of memory while trying to allocate...`, compare your parameter + optimizer + activation + KV-cache byte estimate against per-chip HBM (`16` GiB on `v5e`, `32` GiB on `v6e`) and shard the state across chips (`NamedSharding` / FSDP) or reduce microbatch size.

## Carry forward

- Size your TPU slice (`v5e`, `v5p`, `v6e`) from explicit HBM byte budgets for training state and KV caches before launching.
- Compare your workload's FLOPs/byte against the chip's ridge point to know whether fusion/quantization (bandwidth) or MXU tiling (compute) will speed it up.

## Keep your evidence

Keep the TPU ridge-point table, 7B and 1.5B training/KV-cache HBM budgets, and minimum chip count calculation at an 80% HBM ceiling.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [Cloud TPU v5e, v5p, and v6e (Trillium) specifications](https://cloud.google.com/tpu/docs/v6e)
- [Cloud TPU performance guide](https://cloud.google.com/tpu/docs/performance-guide)

