"""Attention from small pieces · learner-only exercises (3/3 contracts ready)"""

# Exercise: Permute the three key/value pairs together, then permute only values. Verify invariance for the paired permutation and a changed result for the mismatched pairing.

# Inputs:
# - Q/K/V arrays with explicit batch, query, key, and feature axes.

# Outputs:
# - Attention outputs with one result per query and value feature.

# Invariants to check:
# - Softmax normalizes over keys; paired key/value permutations preserve the function.

# Hints:
# - Compute score scaling from the key feature width.
# - Check zero queries against the mean of the values.

import jax.numpy as jnp

def scaled_dot_product_attention(queries, keys, values):
    """Attention outputs with one result per query and value feature."""
    # TODO: Compute scaled query-key scores, normalize along keys, and combine the paired value vectors.
    raise NotImplementedError




# Use a different value dimension (Transfer / diagnosis):
# Make each value a single scalar \([3, 6, 9]\). Check the output shape and zero-query mean.

# Inputs:
# - Make each value a single scalar \([3, 6, 9]\). Check the output shape and zero-query mean.

# Outputs:
# - One scalar attention output per query and a zero-query mean check.

# Invariants to check:
# - The key/value row pairing remains aligned and the query axis is preserved.

# Hints:
# - The score matrix does not depend on Dv.

import jax.numpy as jnp

def practice_use_a_different_value_dimension(values):
    """Implement the use a different value dimension check."""
    # TODO: Construct scalar values with shape (3,1), attend with a zero query, and assert the output shape and arithmetic mean.
    raise NotImplementedError




# Expose a wrong softmax axis (Transfer / diagnosis):
# Normalize zero scores along axis \(0\) and compare row sums with the intended contract. Repair the axis.

# Inputs:
# - Normalize zero scores along axis \(0\) and compare row sums with the intended contract. Repair the axis.

# Outputs:
# - Both row-sum diagnostics and a corrected attention-weight matrix.

# Invariants to check:
# - For each query, probabilities sum across key positions.

# Hints:
# - Two queries and three keys make the incorrect result visible.

import jax.numpy as jnp

def practice_expose_a_wrong_softmax_axis(scores):
    """Implement the expose a wrong softmax axis check."""
    # TODO: Normalize a zero score matrix across axis zero, inspect column sums, and implement the key-axis normalization.
    raise NotImplementedError

