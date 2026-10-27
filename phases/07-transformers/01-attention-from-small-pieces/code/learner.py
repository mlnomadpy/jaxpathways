"""Attention from small pieces · learner-only exercises (3/3 contracts ready)"""

# Exercise: Start from the explicit independent fixture below: q_ex = [[2, 0]], k_ex = [[1, 0], [0, 1], [1, 1]], and v_ex = [[2, 0], [0, 4], [4, 2]]. Implement `attend` using the declared starter. Show algebraically why permuting keys and their paired values together preserves the weighted sum. Then test order [2, 0, 1] for both the paired permutation and a value-only permutation. Compare each result to the original output with NumPy. State why the separate zero-query worked example cannot show a value-only change.

# Inputs:
# - Independent nonuniform fixture: q_ex=[[2,0]], k_ex=[[1,0],[0,1],[1,1]], v_ex=[[2,0],[0,4],[4,2]], and permutation order [2,0,1].

# Outputs:
# - Original attention output, paired-permutation output, and value-only-permutation output compared with an independent NumPy reference.

# Invariants to check:
# - Permuting keys and their paired values together preserves the weighted sum.
# - A value-only permutation can change the result for this nonuniform fixture; this is not guaranteed for every query/value set.
# - Zero queries produce a uniform-weight mean that is invariant under value permutations.

# Hints:
# - Implement scaled scores, softmax over keys, then weights @ values.
# - Write the paired permutation as a reindexing of the same sum.
# - Run the full solution using only the fixture declared in this starter.

import jax
import jax.numpy as jnp

def attend(q, k, v):
    """Return attention outputs and normalized key weights."""
    # TODO: compute scaled query-key scores, normalize over keys, and mix values.
    raise NotImplementedError

q_ex = jnp.array([[2., 0.]])
k_ex = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
v_ex = jnp.array([[2., 0.], [0., 4.], [4., 2.]])
order = jnp.array([2, 0, 1])



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

