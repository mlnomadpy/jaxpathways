"""Immutable updates and indexing · learner-only exercises (3/3 contracts ready)"""

# Exercise: Replace the last two elements with zero in a new array. Keep \(x\) unchanged.

# Inputs:
# - A one-dimensional four-value array `x`.
# - A request to replace its final two positions in a new value.

# Outputs:
# - A changed array and the unchanged original input.

# Invariants to check:
# - The source array remains unchanged.
# - Output shape matches input shape.
# - Only requested positions change.

# Hints:
# - Use a functional indexed update and keep its returned array.
# - Compare the input after the update.

import jax.numpy as jnp

def replace_suffix(x):
    # TODO: make the requested functional indexed update
    raise NotImplementedError




# Replace a region of a matrix (Practice):
# Create a \((3, 3)\) matrix containing \(0\) through \(8\). Set its final column to \(-1\) in a returned value. Verify the whole result and original matrix.

# Inputs:
# - A three-by-three matrix of consecutive values.

# Outputs:
# - A new matrix with its final column replaced.

# Invariants to check:
# - Nonselected entries match the input.
# - The original matrix is unchanged.
# - The returned shape remains `(3, 3)`.

# Hints:
# - Select the last column in every row.
# - Check the entire result and original.

import jax.numpy as jnp

def replace_final_column(matrix):
    # TODO: update the selected region in a returned array
    raise NotImplementedError




# Keep shape while masking values (Challenge):
# Keep only values greater than \(2\) from \(x\), replacing other positions with zero. Compare with compact selection and explain the shape difference.

# Inputs:
# - A one-dimensional input `x` and threshold two.

# Outputs:
# - A fixed-shape masked value and a compact selected value.

# Invariants to check:
# - The masked result keeps input shape.
# - The compact result contains only selected entries.

# Hints:
# - Use elementwise selection for fixed shape.
# - Inspect both output shapes.

import jax.numpy as jnp

def compare_mask_and_selection(x, threshold=2.):
    # TODO: construct both fixed-shape mask and compact selection
    raise NotImplementedError

