"""Convert PyTorch weights to Flax and locate numerical errors · learner-only exercises (3/3 contracts ready)"""

# Exercise: Compare a near-zero reference against two candidate errors. Verify that the absolute tolerance accepts rounding noise but rejects a meaningful discrepancy.

# Inputs:
# - A near-zero reference value, the two candidate values, and the declared absolute/relative tolerances.

# Outputs:
# - Whether rounding noise is accepted and whether the meaningful discrepancy is rejected.

# Invariants to check:
# - The absolute tolerance handles the near-zero rounding case; the larger discrepancy remains outside tolerance.

# Hints:
# - Evaluate each candidate against the same near-zero reference.
# - Include the absolute and relative tolerance terms in the comparison rule.

import numpy as np

def check_near_zero_tolerance(reference, candidates, atol, rtol):
    # TODO: Apply the declared tolerance to both candidate errors and report each acceptance result.
    raise NotImplementedError




# Reject a plausible but wrong tensor layout (Transfer):
# Transpose the first PyTorch weight before conversion and prove the boundary rejects it instead of transposing it a second time.

# Inputs:
# - Transpose the first PyTorch weight before conversion and prove the boundary rejects it instead of transposing it a second time.

# Outputs:
# - A rejection for the transposed source tensor and parity for the correct mapping.

# Invariants to check:
# - The converter performs one documented layout transform.

# Hints:
# - The first weight is rectangular, so validate its source shape before applying the mapping.
# - Compare asymmetric matrix dimensions so a mistaken second transpose is visible.

import numpy as np
import jax
import jax.numpy as jnp
import torch
from flax import nnx

def practice_reject_a_plausible_but_wrong_tensor_layout(source_weight, converted_weight, tolerance):
    """Complete the reject a plausible but wrong tensor layout check."""
    # TODO: ('Transpose the source first-layer weight before mapping and check that the parity boundary rejects it without a second transpose.', 'A failed wrong-layout parity check and an accepted correctly mapped matrix.', 'Each framework layout conversion is applied exactly once.')
    raise NotImplementedError




# Check a nonuniform output sensitivity (Transfer / diagnosis):
# Compare the source and converted input gradients for an output cotangent that contains unequal positive and negative entries. Explain what this tests beyond summing the outputs.

# Inputs:
# - Compare the source and converted input gradients for an output cotangent that contains unequal positive and negative entries. Explain what this tests beyond summing the outputs.

# Outputs:
# - Source and converted input-gradient vectors.

# Invariants to check:
# - Both paths receive the same mixed-sign output cotangent.

# Hints:
# - Form the same scalar weighted output in each framework. Create a fresh PyTorch input so earlier gradients do not accumulate.
# - Compare gradient entries with declared numerical tolerances.

import numpy as np
import jax
import jax.numpy as jnp
import torch
from flax import nnx

def practice_check_a_nonuniform_output_sensitivity(source_fn, converted_fn, inputs, cotangent):
    """Complete the check a nonuniform output sensitivity check."""
    # TODO: ('Use a mixed-sign output cotangent to compare source and converted input gradients.', 'Two input-gradient vectors with tolerance-based parity.', 'The same nonuniform cotangent is applied to both implementations.')
    raise NotImplementedError

