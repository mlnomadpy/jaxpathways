"""Read your first jaxpr · learner-only exercises (3/3 contracts ready)"""

# Exercise: Change the scalar offset from \(1\) to \(2\). Draw the new value flow for \([1, 2, 3]\), calculate the sum and gradient by hand, then inspect both jaxprs. Explain which structural features remain and which values change.

# Inputs:
# - Initial vector [1,2,3] and the original scalar offset 1.
# - Changed scalar offset 2 and the original/modified functions to compare.

# Outputs:
# - Hand-derived changed function value and input gradient.
# - jaxprs for both the original and changed-offset functions.

# Invariants to check:
# - Changing the offset changes constants/values while preserving the relevant traced operation structure.
# - The gradient is with respect to the input vector, not the scalar offset.

# Hints:
# - Trace both offset values at the same input shape.
# - Differentiate the changed function and compare its value flow with the original.

import jax
import jax.numpy as jnp

def inspect_offset_function(values, original_offset=1.0, changed_offset=2.0):
    # TODO: compare hand-derived value flow, input gradient, and both jaxprs.
    raise NotImplementedError("Compare the original and changed scalar-offset traces")



# Compare unrolled and scanned recurrences (Transfer / diagnosis):
# Implement four repetitions of \(z\leftarrow0.5z+1\) using a fixed Python loop and scan. Print both representations and check the final result from \(z=0\).

# Inputs:
# - Initial scalar state 0 and an affine recurrence applied four times.

# Outputs:
# - Unrolled and lax.scan representations and equal final states.

# Invariants to check:
# - Both implementations use the same number of recurrence steps.
# - The final value agrees with an independent scalar calculation.

# Hints:
# - Keep the unrolled Python loop static.
# - Return scan carry explicitly and inspect the jaxpr structures.

import jax
import jax.numpy as jnp

def compare_recurrences(initial):
    # TODO: express the same four updates with a static loop and lax.scan.
    raise NotImplementedError("Compare unrolled and scanned recurrence representations")



# Reproduce and repair a traced boolean failure (Transfer / diagnosis):
# Write a Python branch that chooses `sum(z)` or −`sum(z)` based on \(z\)\([0]\)>\(0\). Reproduce the make_jaxpr error, replace the branch with cond, and verify positive and negative cases.

# Inputs:
# - A traced vector with positive and negative first elements.

# Outputs:
# - Captured-boolean tracing failure and a lax.cond implementation for both signs.

# Invariants to check:
# - The condition is a scalar traced predicate.
# - Both branches return compatible scalar shapes.

# Hints:
# - First reproduce Python if on a traced boolean.
# - Replace it with lax.cond without converting the predicate to Python.

import jax
import jax.numpy as jnp

def repair_traced_branch(vector):
    # TODO: demonstrate the Python-branch trace error, then use lax.cond.
    raise NotImplementedError("Replace data-dependent Python control flow with lax.cond")
