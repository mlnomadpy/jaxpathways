"""Immutable updates and indexing: worked experiments and reference solutions. CPU checks."""

# Step 1: Set up imports and input tensors
import jax.numpy as jnp
# Construct `x` via `jnp.array([1., 2., 3., 4.])`
x = jnp.array([1., 2., 3., 4.])

# Step 2: Apply the core JAX transformation
y = x.at[1].set(20.)
# Construct `z` via `x.at[jnp.array([0, 2])].add(5.)`
z = x.at[jnp.array([0, 2])].add(5.)

# Step 3: Verify shapes and numerical invariants
assert jnp.allclose(x, jnp.array([1.,2.,3.,4.]))
# Assert that `jnp.allclose(y, jnp.array([1.,20.,3.,4.]))`.
assert jnp.allclose(y, jnp.array([1.,20.,3.,4.]))
# Assert that `jnp.allclose(z, jnp.array([6.,2.,8.,4.]))`.
assert jnp.allclose(z, jnp.array([6.,2.,8.,4.]))

# Immutable updates and indexing: An indexed update in JAX returns an array value.
# Import jax.numpy for this computation.
import jax.numpy as jnp
# Construct `x` via `jnp.array([1., 2., 3., 4.])`
x = jnp.array([1., 2., 3., 4.])
# Compute `y` from `x.at[1].set(20.)`
y = x.at[1].set(20.)
# Construct `z` via `x.at[jnp.array([0, 2])].add(5.)`
z = x.at[jnp.array([0, 2])].add(5.)
# Print the observed values to compare against the expected result.
print("Original:", x)
# Print diagnostic summary of the computed outputs.
print("Set:", y)
# Print diagnostic summary of the computed outputs.
print("Add:", z)
# Assert that `jnp.allclose(x, jnp.array([1.,2.,3.,4.]))`.
assert jnp.allclose(x, jnp.array([1.,2.,3.,4.]))
# Assert that `jnp.allclose(y, jnp.array([1.,20.,3.,4.]))`.
assert jnp.allclose(y, jnp.array([1.,20.,3.,4.]))
# Assert that `jnp.allclose(z, jnp.array([6.,2.,8.,4.]))`.
assert jnp.allclose(z, jnp.array([6.,2.,8.,4.]))

# Figure data experiment
# Compute figure data for: An update returns a new array
# Combine or mask array elements to form `visual_data`.
visual_data = {'kind': 'heatmap', 'values': jnp.stack([x, y, z]).tolist(), 'rows': ['original', 'set index 1', 'add at 0 and 2'], 'columns': ['0', '1', '2', '3'], 'unit': 'stored value'}

# Experiment: See name rebinding without old-value mutation
# Experiment — See name rebinding without old-value mutation: Assigning a result changes the binding, while ignoring a result...
original = x
# Compute `rebound` from `x`
rebound = x
# Compute `rebound` from `rebound.at[1].set(20.)`
rebound = rebound.at[1].set(20.)
# Assert that `jnp.allclose(original, jnp.array([1., 2., 3., 4.]))`.
assert jnp.allclose(original, jnp.array([1., 2., 3., 4.]))
# Assert that `jnp.allclose(rebound, jnp.array([1., 20., 3., 4.]))`.
assert jnp.allclose(rebound, jnp.array([1., 20., 3., 4.]))
# Compute `ignored` from `x.at[0].set(99.)`
ignored = x.at[0].set(99.)
# Assert that `jnp.allclose(x, original)`.
assert jnp.allclose(x, original)
# Assert invariant `float(ignored[0]) == 99.` holds
assert float(ignored[0]) == 99.

# Experiment: Accumulate repeated integer contributions
# Experiment — Accumulate repeated integer contributions: The integer loop reference makes accumulation explicit.
# Import numpy for this computation.
import numpy as np
# Construct `indices` via `jnp.array([1, 1, 3])`
indices = jnp.array([1, 1, 3])
# Construct `contributions` via `jnp.array([2, 3, 7], dtype=jnp.int32)`
contributions = jnp.array([2, 3, 7], dtype=jnp.int32)
# Construct `base_counts` via `jnp.zeros(4, dtype=jnp.int32)`
base_counts = jnp.zeros(4, dtype=jnp.int32)
# Compute `counts` from `base_counts.at[indices].add(contributions)`
counts = base_counts.at[indices].add(contributions)
# Allocate initialized array `reference_counts` with the specified shape and dtype.
reference_counts = np.zeros(4, dtype=np.int32)
# Iterate over `(index, contribution)` to step through the computation:
for index, contribution in zip([1, 1, 3], [2, 3, 7]):
    # Accumulate the next contribution into `reference_counts[index]`.
    reference_counts[index] += contribution
# Assert invariant `jnp.array_equal(counts, reference_counts)` holds
assert jnp.array_equal(counts, reference_counts)
# Assert invariant `jnp.array_equal(counts, jnp.array([0, 5, 0, 7]))` holds
assert jnp.array_equal(counts, jnp.array([0, 5, 0, 7]))
# Assert invariant `jnp.array_equal(base_counts, jnp.zeros(4, dtype=jnp.int32))` holds
assert jnp.array_equal(base_counts, jnp.zeros(4, dtype=jnp.int32))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Replace the last two elements with zero in a new array.
masked = x.at[2:].set(0.)
# Assert that `jnp.allclose(masked, jnp.array([1.,2.,0.,0.]))`.
assert jnp.allclose(masked, jnp.array([1.,2.,0.,0.]))
# Assert that `jnp.allclose(x, jnp.array([1.,2.,3.,4.]))`.
assert jnp.allclose(x, jnp.array([1.,2.,3.,4.]))

# Reference practice: Replace a region of a matrix
# Replace a region of a matrix (Practice): Checking the entire matrix verifies both changed and...
# Construct and reshape `matrix` into the target tensor dimensions.
matrix = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
# Compute `updated_matrix` from `matrix.at[:, -1].set(-1.)`
updated_matrix = matrix.at[:, -1].set(-1.)
# Assert that `jnp.allclose(updated_matrix, jnp.array([[0., 1., -1.], [3., 4., -1.], [6., 7., -1.]]))`.
assert jnp.allclose(updated_matrix, jnp.array([[0., 1., -1.], [3., 4., -1.], [6., 7., -1.]]))
# Assert that `jnp.allclose(matrix, jnp.arange(9).reshape(3, 3))`.
assert jnp.allclose(matrix, jnp.arange(9).reshape(3, 3))

# Reference practice: Keep shape while masking values
# Keep shape while masking values (Challenge): Both are useful representations, but they are not...
fixed_shape = jnp.where(x > 2., x, 0.)
# Compute `selected` from `x[x > 2.]`
selected = x[x > 2.]
# Check tensor shape invariant: `fixed_shape.shape == (4,)`
assert fixed_shape.shape == (4,)
# Check tensor shape invariant: `selected.shape == (2,)`
assert selected.shape == (2,)
# Assert that `jnp.allclose(fixed_shape, jnp.array([0., 0., 3., 4.]))`.
assert jnp.allclose(fixed_shape, jnp.array([0., 0., 3., 4.]))
# Assert that `jnp.allclose(selected, jnp.array([3., 4.]))`.
assert jnp.allclose(selected, jnp.array([3., 4.]))
print("PASS: arrays-04")
