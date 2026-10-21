"""TPU generations and HBM memory budgeting · learner-only exercises (3/3 contracts ready)"""

# Exercise: Compute estimate_memory_gib for a 1.5B parameter model with 4096 KV tokens. Check that the assumed mixed-precision training state is below 30 GiB and below 80% of a four-chip v5e slice. Separately report the BF16 KV-cache estimate for an inference scenario. Do not assume training normally retains an inference KV cache, or that aggregate HBM guarantees per-device fit.

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



# Size a Trillium v6e-4 slice for a 7B training state (Transfer / diagnosis):
# Compute the total HBM (in GiB) of a 4-chip `v6e-4` slice (`32` GiB/chip) and check whether `budget_7b['mixed_bf16_train_state_gib']` fits inside `80%` of that slice's HBM.

# Inputs:
# - Four v6e chips at 32 GiB each and the 7B mixed-BF16 training estimate.

# Outputs:
# - Aggregate HBM and a check against 80 percent of the slice capacity.

# Invariants to check:
# - The calculation is a catalog budget screen and does not guarantee per-device placement or runtime success.

# Hints:
# - Multiply chips by HBM per chip.
# - Compare the estimate with the reserved-capacity threshold.

import jax.numpy as jnp
import numpy as np

def check_v6e4_budget(training_state_gib, chips, hbm_per_chip_gib):
    # TODO: compare the training estimate with the specified fraction of aggregate HBM.
    raise NotImplementedError
