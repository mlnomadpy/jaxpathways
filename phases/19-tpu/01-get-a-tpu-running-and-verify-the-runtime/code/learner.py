"""Select a TPU slice and provision your Cloud TPU VM · learner-only exercises (3/3 contracts ready)"""

# Exercise: Use the Foundations ledger for a 48 GiB state on four 16 GiB v5e chips. Report replicated and ideal-sharded state per chip, the 12.8 GiB usable budget, and the result with a 1 GiB activation allowance. Then inspect the selected v5litepod-4 receipt: report observed process count and JAX device count separately from the requested physical chip count, plus the API and machine type. On CPU, arithmetic should pass while requested_target_ready remains false.

# Inputs:
# - The runtime verifier with expected_backend and expected_device_count, plus the named topology catalog.
# - A 48 GiB state, four 16 GiB chips, 80% usable capacity and a 1 GiB per-chip activation assumption.

# Outputs:
# - Replicated/ideal-sharded per-device memory decisions with and without activation overhead.
# - Separate requested API/machine/chip/process values and observed JAX backend/process/device values.

# Invariants to check:
# - Aggregate HBM alone does not establish per-device fit.
# - CPU arithmetic may pass while requested TPU readiness is false.
# - Observed JAX device count does not by itself establish a physical chip count.

# Hints:
# - Compute 48/4 and compare 12 plus the activation allowance with 16 times 0.8.
# - Use expected_device_count, not the obsolete minimum-device argument.

def inspect_runtime_and_slice(verify_runtime_contract, expected_backend, expected_device_count, topologies):
    # TODO: check arithmetic, requested readiness, per-device fit and separately identified topology.
    raise NotImplementedError




# Compare replicated and ideally sharded per-device HBM (Foundations):
# A 48 GiB training state runs on four v5e chips, each with 16 GiB HBM. At an 80% usable-HBM budget, calculate per-device state for replication and ideal even sharding. Which placement meets the state-only limit? How does an added 1 GiB/device of activations change the result? Repeat for a 52 GiB state.

# Inputs:
# - A 48 GiB state, four physical v5e chips with 16 GiB HBM each, 80% usable-HBM fraction and 1 GiB/device activation allowance.

# Outputs:
# - Per-device replicated and ideal-sharded state requirements, plus fit decisions before and after activation allowance.
# - A 52 GiB changed-input result that shows the ideal shard exceeds the usable budget.

# Invariants to check:
# - Replication stores the whole state on each chip; ideal even sharding divides by chip count; report this as an assumption, not a measured placement.

# Hints:
# - Compute the per-chip usable budget first.
# - Repeat with 52 GiB to see the changed input cross the ideal-shard capacity limit.

def placement_hbm_screen(total_state_gib, chips, chip_hbm_gib, usable_fraction, activation_gib_per_chip):
    # TODO: return per-chip state for replication and ideal even sharding,
    # the usable budget after activation allowance, and fit flags.
    raise NotImplementedError



# Verify the pmap sum-of-squares formula for 4 TPU devices (Transfer / diagnosis):
# Compute the exact sum of squares that `verify_runtime_contract` expects when `len(local_devices) == 4` (`32` elements `0..31`).

# Inputs:
# - Four local TPU devices and the mapped range of 32 integers from zero through 31.

# Outputs:
# - Expected exact sum of squares for the four-device pmap fixture.

# Invariants to check:
# - The expected value sums each integer exactly once; it is an oracle for the four-device test only.

# Hints:
# - Use the closed-form sum-of-squares identity or an integer reference sum.
# - Run the pmap assertion on a real four-chip TPU to qualify runtime behavior.

import jax.numpy as jnp
import numpy as np

def expected_four_chip_sum():
    # TODO: compute the integer oracle for the range consumed by the four-device pmap.
    raise NotImplementedError
