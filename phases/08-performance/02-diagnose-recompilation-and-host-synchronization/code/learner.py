"""Diagnose recompilation and host synchronization · learner-only exercises (3/3 contracts ready)"""

# Exercise: Add a call with sixteen float32 elements. Calculate its expected score independently and extend the signature ledger without labeling a trace event as a compilation.

# Inputs:
# - The existing specialization ledger, a 16-element float32 input, and the observed-call helper.

# Outputs:
# - An independently checked score for the 16-element call and a new ledger row with its signature and trace observation.

# Invariants to check:
# - The changed input shape adds a signature entry; a Python trace event is not labeled as proof of compilation.

# Hints:
# - Calculate the sum of squared scaled elements on the host independently.
# - Record shape, dtype, and trace delta as separate observations.

import numpy as np

def add_sixteen_element_call(ledger, values, gain, observed_call):
    # TODO: Run the new 16-element signature, verify its score independently, and append its observed ledger row.
    raise NotImplementedError




# Pad a final batch while preserving the objective (Transfer):
# A final batch has five values while other batches have eight. Pad to eight and use an explicit validity mask in a dynamic array. Compare the masked sum of squares to an ordinary five-value calculation. Explain why padding without masking changes objectives with offsets.

# Inputs:
# - A final batch has five values while other batches have eight. Pad to eight and use an explicit validity mask in a dynamic array. Compare the masked sum of squares to an ordinary five-value calculation. Explain why padding without masking changes objectives with offsets.

# Outputs:
# - Equal masked/unpadded objectives plus a counterexample without masking.

# Invariants to check:
# - Padded entries contribute neither residual nor offset terms.

# Hints:
# - Use a fixed-size input and mask; compute the offset transformation before masking.

import jax.numpy as jnp
import time

def practice_pad_a_final_batch_while_preserving_the_objective(values, valid_mask):
    """Implement the pad a final batch while preserving the objective check."""
    # TODO: Pad five values to eight with a validity mask and compare masked squared-error sum with the unpadded calculation.
    raise NotImplementedError




# Reproduce and repair runtime Python branching (Diagnosis):
# Write a jitted function using if gain > \(1\) on a scalar array. Capture its tracing failure and replace the branch with lax.cond. Verify both branches against independent arithmetic.

# Inputs:
# - Write a jitted function using if gain > \(1\) on a scalar array. Capture its tracing failure and replace the branch with lax.cond. Verify both branches against independent arithmetic.

# Outputs:
# - Captured tracing failure and correct outputs for both predicate outcomes.

# Invariants to check:
# - The branch predicate is a scalar JAX value and both branches return matching result types.

# Hints:
# - A runtime scalar predicate is a tracer, so Python cannot decide its truth value during tracing.

import jax
import jax.numpy as jnp
import time

def practice_reproduce_and_repair_runtime_python_branching(value, threshold):
    """Implement the reproduce and repair runtime python branching check."""
    # TODO: Trigger the traced scalar-array branch failure, then replace it with lax.cond and verify both branches against arithmetic.
    raise NotImplementedError

