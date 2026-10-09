"""TPU generations and HBM memory budgeting: worked experiments and reference solutions. CPU checks."""

# Define TPU generation specs and compute Roofline ridge points
# Step 1 — Define TPU generation specs and compute Roofline ridge points: Dividing peak BF16 FLOP/s by HBM bandwidth (B/s) gives the minimum...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Initialize list `TPU_SPECS` for the stage values.
TPU_SPECS = [
    {"gen": "v5e", "hbm_gib": 16.0, "bw_gbs": 819.0, "bf16_tflops": 197.0},
    {"gen": "v5p", "hbm_gib": 95.0, "bw_gbs": 2765.0, "bf16_tflops": 459.0},
    {"gen": "v6e", "hbm_gib": 32.0, "bw_gbs": 1600.0, "bf16_tflops": 918.0},
]
# Iterate over `s` to step through the computation:
for s in TPU_SPECS:
    # Run `round` to compute `s['ridge_flops_per_byte']`.
    s["ridge_flops_per_byte"] = round((s["bf16_tflops"] * 1e12) / (s["bw_gbs"] * 1e9), 1)

# Estimate training-state and KV-cache HBM budgets across precisions
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
# Assert that `budget_7b["mixed_bf16_train_state_gib"] > 64.0 and budget_7b["mixed_bf16_train_state_gib"] <`.
assert budget_7b["mixed_bf16_train_state_gib"] > 64.0 and budget_7b["mixed_bf16_train_state_gib"] < 128.0
# Assert invariant `budget_1p5b["mixed_bf16_train_state_gib"] < 64.0` holds
assert budget_1p5b["mixed_bf16_train_state_gib"] < 64.0
# Print the observed values to compare against the expected result.
print("TPU ridge points (FLOP/byte):", {s["gen"]: s["ridge_flops_per_byte"] for s in TPU_SPECS})
# Print diagnostic summary of the computed outputs.
print("7B memory budget (GiB):", budget_7b)
# Print diagnostic summary of the computed outputs.
print("1.5B memory budget (GiB):", budget_1p5b)

# Step 3: Verify invariants on the completed state
assert budget_7b["mixed_bf16_train_state_gib"] > 64.0 and budget_7b["mixed_bf16_train_state_gib"] < 128.0
assert budget_1p5b["mixed_bf16_train_state_gib"] < 64.0

# Step 1 — Define TPU generation specs and compute Roofline ridge points: Dividing peak BF16 FLOP/s by HBM bandwidth (B/s) gives the minimum...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Initialize list `TPU_SPECS` for the stage values.
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
# Assert that `budget_7b["mixed_bf16_train_state_gib"] > 64.0 and budget_7b["mixed_bf16_train_state_gib"] <`.
assert budget_7b["mixed_bf16_train_state_gib"] > 64.0 and budget_7b["mixed_bf16_train_state_gib"] < 128.0
# Assert invariant `budget_1p5b["mixed_bf16_train_state_gib"] < 64.0` holds
assert budget_1p5b["mixed_bf16_train_state_gib"] < 64.0
# Print the observed values to compare against the expected result.
print("TPU ridge points (FLOP/byte):", {s["gen"]: s["ridge_flops_per_byte"] for s in TPU_SPECS})
# Print diagnostic summary of the computed outputs.
print("7B memory budget (GiB):", budget_7b)
# Print diagnostic summary of the computed outputs.
print("1.5B memory budget (GiB):", budget_1p5b)

# Figure data experiment
# Compute figure data for: Per-chip HBM capacity and ridge-point arithmetic intensity across TPU generations
# Construct dictionary `visual_data` with the structured fields for this stage.
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

# Experiment: Compare KV-cache memory across FP32, BF16, and INT8
# Experiment — Compare KV-cache memory across FP32, BF16, and INT8: Halving bytes per element halves both the HBM capacity consumed...
# Assert that `np.isclose(budget_7b["kv_fp32_gib"], 2.0 * budget_7b["kv_bf16_gib"])`.
assert np.isclose(budget_7b["kv_fp32_gib"], 2.0 * budget_7b["kv_bf16_gib"])
# Assert that `np.isclose(budget_7b["kv_bf16_gib"], 2.0 * budget_7b["kv_int8_gib"])`.
assert np.isclose(budget_7b["kv_bf16_gib"], 2.0 * budget_7b["kv_int8_gib"])
# Print the observed values to compare against the expected result.
print("KV cache GiB (FP32 / BF16 / INT8):", budget_7b["kv_fp32_gib"], budget_7b["kv_bf16_gib"], budget_7b["kv_int8_gib"])

# Experiment: Verify how many v5e chips are needed to hold a 7B mixed-precision training state
# Experiment — Verify how many v5e chips are needed to hold a 7B mixed-precision training state: Calculating state GiB upfront tells you immediately whether you...
v5e_chip_gib = 16.0
# Evaluate `np.ceil(budget_7b['mixed_bf16_train_state_gib'] / (v5e_chip_gib * 0.8))` and convert the result into Python scalar/collection `min_v5e_chips`.
min_v5e_chips = int(np.ceil(budget_7b["mixed_bf16_train_state_gib"] / (v5e_chip_gib * 0.8)))
# Print the observed values to compare against the expected result.
print("7B mixed train state GiB:", budget_7b["mixed_bf16_train_state_gib"], "Min v5e chips (at 80% HBM budget):", min_v5e_chips)
# Assert invariant `min_v5e_chips > 4 and min_v5e_chips <= 8` holds
assert min_v5e_chips > 4 and min_v5e_chips <= 8

# Reference solution. Try the exercise before reading this.
# Exercise solution: Compute estimate_memory_gib for a 1.5B parameter model with 4096 KV...
budget_check = estimate_memory_gib(params_billions=1.5, kv_tokens=4096)
# Print the observed values to compare against the expected result.
print("1.5B budget (GiB):", budget_check)
# Assert that `budget_check["mixed_bf16_train_state_gib"] < 20.0 and (budget_check["mixed_bf16_train_state_`.
assert budget_check["mixed_bf16_train_state_gib"] < 20.0 and (budget_check["mixed_bf16_train_state_gib"] + budget_check["kv_bf16_gib"]) < 64.0

# Reference practice: Compare arithmetic intensity against the TPU v5e and v6e ridge points
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

# Reference practice: Size a Trillium v6e-4 slice for a 7B training state
# Size a Trillium v6e-4 slice for a 7B training state (Transfer / diagnosis): Because v6e doubles per-chip HBM to 32 GiB (128 GiB across 4...
v6e4_usable_gib = 4 * 32.0 * 0.8
# Compute `fits_v6e4` from `budget_7b["mixed_bf16_train_state_gib"] <= v6e4_usab...`
fits_v6e4 = budget_7b["mixed_bf16_train_state_gib"] <= v6e4_usable_gib
# Print the observed values to compare against the expected result.
print("v6e-4 80% usable GiB:", v6e4_usable_gib, "Fits 7B state:", fits_v6e4)
# Assert invariant `fits_v6e4 is True` holds
assert fits_v6e4 is True
print("PASS: tpu-03")
