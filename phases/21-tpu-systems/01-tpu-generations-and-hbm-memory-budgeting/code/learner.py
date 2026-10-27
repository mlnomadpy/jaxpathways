"""TPU generations and HBM memory budgeting · learner-only exercises (3/3 contracts ready)"""

# Exercise: For a 7B mixed-BF16 training state, compare ideal evenly sharded v6e-4 and v6e-8 plans at 80% HBM per chip. Report per-chip state and remaining HBM before activations, then decide whether each can fit under an additional 8 GiB/device activation-and-temporary allowance. Separately report the BF16 inference KV cache at 8192 tokens. State the placement assumption and do not add inference KV cache to training state.

# Inputs:
# - The public memory estimator, a 1.5-billion parameter training case, 4096 KV tokens, and an inference cache scenario.

# Outputs:
# - Training state estimate, slice-budget comparisons, and a separate BF16 KV-cache estimate.

# Invariants to check:
# - Training state and inference KV cache are distinct ledgers; aggregate slice HBM does not ensure each sharded device fits.

# Hints:
# - Read and preserve the estimator's dtype and exclusion assumptions.
# - Report cache memory separately from training-state memory.

import jax.numpy as jnp
import numpy as np

def audit_training_and_inference_memory(estimate_memory_gib):
    # TODO: evaluate the training case and compute a separately labeled inference-cache estimate.
    raise NotImplementedError



# Compare arithmetic intensity against the TPU v5e and v6e ridge points (Foundations):
# Compute the arithmetic intensity (FLOPs/byte) of a `bfloat16` square matrix multiply of size \(N = 1024\) (`2 * N^3` FLOPs divided by `3 * 2 * N^2` bytes) and check whether it exceeds the `v5e` ridge point.

# Inputs:
# - Square BF16 matrix dimension 1024 and the documented v5e and v6e ridge points.

# Outputs:
# - Arithmetic intensity in FLOPs per byte and a comparison with each ridge point.

# Invariants to check:
# - FLOPs and bytes use the same operation count as the lesson; intensity is not device memory capacity.

# Hints:
# - Simplify the given expression before evaluating it.
# - Compare with ridge points in matching FLOPs-per-byte units.

import jax.numpy as jnp
import numpy as np

def compare_matmul_intensity(size, ridge_points):
    # TODO: compute the stated BF16 intensity and report which ridge points it exceeds.
    raise NotImplementedError



# Compare v6e-4 and v6e-8 per-device training placement (Transfer / diagnosis):
# Using the ledger helper, compare a 7B mixed-BF16 state under ideal sharding on v6e-4 and v6e-8, with an assumed 8 GiB/device activation and temporary allowance. Keep the 8192-token inference KV cache separate.

# Inputs:
# - 7B mixed-BF16 training-state estimate, 4 or 8 v6e chips at 32 GiB/chip, 80% usable-HBM fraction, and 8 GiB/device activations plus temporary allowance.

# Outputs:
# - Per-device state and remaining capacity for v6e-4 and v6e-8 under ideal even sharding, plus separate BF16 inference KV bytes.

# Invariants to check:
# - Divide training state by chip count only under explicit ideal-sharding assumption. Do not add inference KV cache to training state.

# Hints:
# - Use the per-device ledger helper and report state/device separately from usable HBM/device.
# - Try a 48 GiB versus 52 GiB state in the helper to observe an input change.

def compare_v6e_placement(training_state_gib, kv_cache_gib, chips_options, activation_gib_per_chip):
    # TODO: return explicit ideal-sharding per-device ledgers for each topology
    # and keep inference KV cache in a separate result.
    raise NotImplementedError
