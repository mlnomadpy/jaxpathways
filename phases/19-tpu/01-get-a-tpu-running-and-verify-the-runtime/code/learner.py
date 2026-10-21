"""Select a TPU slice and provision your Cloud TPU VM · learner-only exercises (3/3 contracts ready)"""

# Exercise: Call `verify_runtime_contract` for the active backend with `min_devices=1`, then compute how many `v5litepod-4` chips and total GiB of HBM are available on that single-host slice and assert both values.

# Inputs:
# - The verify_runtime_contract callable, active backend expectation, minimum local device count, and named TPU slice catalog.

# Outputs:
# - A runtime receipt plus chip and aggregate HBM counts for v5litepod-4.

# Invariants to check:
# - Runtime acceptance describes devices actually discovered by JAX; a CPU backend must not be reported as TPU qualification.

# Hints:
# - Call the verifier for the active backend with a minimum of one local device.
# - Treat catalog capacity as descriptive and runtime discovery as observed evidence.

import jax

def inspect_runtime_and_slice(verify_runtime_contract, expected_backend, min_devices, topologies):
    # TODO: verify the active runtime and look up v5litepod-4 chip and HBM metadata.
    raise NotImplementedError



# Check whether a model state fits on a single-host TPU slice (Foundations):
# Given a training state requiring \(48\) GiB of HBM, determine which slices in `topologies` have at least \(48\) GiB of total HBM.

# Inputs:
# - A slice catalog with total HBM values and a required state size of 48 GiB.

# Outputs:
# - The names of slices with enough catalogued aggregate HBM.

# Invariants to check:
# - Capacity filtering uses the provided requirement and catalog units consistently; it does not prove per-device fit.

# Hints:
# - Filter the catalog by total_hbm_gib.
# - Keep memory capacity claims separate from runtime verification.

import jax.numpy as jnp
import numpy as np

def slices_meeting_hbm_requirement(topologies, required_gib):
    # TODO: return the catalog slices whose aggregate HBM meets the requirement.
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
