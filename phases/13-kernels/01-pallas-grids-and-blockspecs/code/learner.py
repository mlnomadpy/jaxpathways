"""Pallas grids and BlockSpecs · learner-only exercises (3/3 contracts ready)"""

# Exercise: Test a singleton row and a changed nondivisible matrix with negative values. Verify full NumPy agreement and exact output shapes for two block choices.

# Inputs:
# - Two same-shaped float32 matrices for each fixture: singleton-row shape (1,7) and changed tail shape (7,9).
# - Two block choices, (2,4) and (3,5), with the lesson kernel wrapper.

# Outputs:
# - Exact logical output shape for each fixture/block pair.
# - Full elementwise agreement with the independent NumPy expression.

# Invariants to check:
# - Each input pair has identical shape and the result preserves that shape.
# - Masked partial blocks cover every logical element, including negative values.

# Hints:
# - Test the one-row case separately from a partial final tile.
# - Compare every output element with the NumPy expression.

import numpy as np
import jax.numpy as jnp

def verify_changed_kernel(wrapper, singleton_pair, changed_pair, block_choices):
    # TODO: run both same-shaped input pairs under each block and check NumPy values.
    raise NotImplementedError("Exercise singleton rows and nondivisible negative-valued tails")



# Reject a broadcast that changes the contract (Practice):
# Pass a column-shaped second input and confirm the wrapper rejects it before launching the kernel. Explain why silent broadcasting would hide an input-contract error.

# Inputs:
# - A kernel wrapper and a (n,1) second input where the contract requires matching vector shape.

# Outputs:
# - A pre-launch validation error and its message.

# Invariants to check:
# - Shape validation happens before kernel execution.
# - Broadcast compatibility does not imply the documented input contract.

# Hints:
# - Check exact ranks and dimensions before creating a launch grid.
# - Make the failure distinguishable from arithmetic errors.

def reject_broadcast_input(kernel_wrapper, values, column_values):
    # TODO: call the wrapper and require its documented shape error.
    raise NotImplementedError("Verify invalid broadcast shapes are rejected before launch")



# Measure padding overhead without claiming runtime (Challenge):
# For shapes (5,11) and (33,129), compare logical elements, padded elements and program counts for three block choices. Explain why fewer programs may still mean more work.

# Inputs:
# - Logical shapes (5,11) and (33,129), plus three valid block shapes.

# Outputs:
# - Program count and padded elements for each shape/block choice.

# Invariants to check:
# - Logical element count is invariant across tile choices.
# - Padded element count can rise even when program count falls.

# Hints:
# - Compute ceil-divided grid dimensions.
# - Report logical work and padded launch work as separate columns.

import math

def padding_report(shapes, blocks):
    # TODO: calculate grids, program counts, and padded elements.
    raise NotImplementedError("Quantify launch padding without making a timing claim")
