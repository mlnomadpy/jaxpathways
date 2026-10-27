"""TPU generations and HBM memory budgeting: worked experiments and reference solutions. CPU checks."""

# Define TPU generation specs and compute Roofline ridge points
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

# Estimate training-state and KV-cache HBM budgets across precisions
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

# Step 3: Verify invariants on the completed state
assert budget_7b["mixed_bf16_train_state_gib"] > 64.0 and budget_7b["mixed_bf16_train_state_gib"] < 128.0
assert budget_1p5b["mixed_bf16_train_state_gib"] < 64.0

# Compare replicated and sharded state per device


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

# Figure data experiment
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

# Experiment: Compare KV-cache memory across FP32, BF16, and INT8
# Experiment — Compare KV-cache memory across FP32, BF16, and INT8
assert np.isclose(budget_7b["kv_fp32_gib"], 2.0 * budget_7b["kv_bf16_gib"])
assert np.isclose(budget_7b["kv_bf16_gib"], 2.0 * budget_7b["kv_int8_gib"])
print("KV cache GiB (FP32 / BF16 / INT8):", budget_7b["kv_fp32_gib"], budget_7b["kv_bf16_gib"], budget_7b["kv_int8_gib"])

# Experiment: Estimate minimum v5e chips under ideal even sharding
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

# Experiment: Decide between replicated and sharded placement with headroom
replicated = per_device_hbm_ledger(48, 4, 16, "replicated", activation_gib_per_chip=1)
ideal_shard = per_device_hbm_ledger(48, 4, 16, "evenly_sharded_ideal", activation_gib_per_chip=1)
changed = per_device_hbm_ledger(52, 4, 16, "evenly_sharded_ideal", activation_gib_per_chip=1)
print("Per-chip ledgers:", replicated, ideal_shard, changed)
assert replicated["state_per_chip_gib"] == 48.0 and not replicated["fits_under_assumptions"]
assert ideal_shard["state_per_chip_gib"] == 12.0 and not ideal_shard["fits_under_assumptions"]
assert changed["state_per_chip_gib"] == 13.0 and not changed["fits_under_assumptions"]

# Reference solution. Try the exercise before reading this.
state = budget_7b["mixed_bf16_train_state_gib"]
for chips in (4, 8):
    ledger = per_device_hbm_ledger(state, chips, 32.0, "evenly_sharded_ideal", activation_gib_per_chip=8.0)
    print("v6e chips:", chips, "state/device GiB:", ledger["state_per_chip_gib"], "usable GiB:", ledger["usable_hbm_per_chip_gib"], "fits with assumed activation:", ledger["fits_under_assumptions"])
    assert ledger["fits_under_assumptions"] is (chips == 8)
print("Separate inference BF16 KV cache GiB:", budget_7b["kv_bf16_gib"])
assert np.isclose(budget_7b["kv_bf16_gib"], 1.0)

# Reference practice: Compare arithmetic intensity against the TPU v5e and v6e ridge points
n = 1024
intensity = (2 * n**3) / (3 * 2 * n**2)
print("Arithmetic intensity:", intensity, "FLOPs/byte")
assert intensity == n / 3.0
assert intensity > TPU_SPECS[0]["ridge_flops_per_byte"]

# Reference practice: Compare v6e-4 and v6e-8 per-device training placement
state = budget_7b["mixed_bf16_train_state_gib"]
ledgers = {chips: per_device_hbm_ledger(state, chips, 32.0, "evenly_sharded_ideal", activation_gib_per_chip=8.0) for chips in (4, 8)}
for chips, row in ledgers.items():
    print(chips, "chips:", row)
assert ledgers[4]["fits_under_assumptions"] is False
assert ledgers[8]["fits_under_assumptions"] is True
print("Separate inference KV GiB:", budget_7b["kv_bf16_gib"])
assert np.isclose(budget_7b["kv_bf16_gib"], 1.0)
print("PASS: tpu-03")
