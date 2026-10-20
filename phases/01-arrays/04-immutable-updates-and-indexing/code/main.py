"""Immutable updates and indexing: worked experiments and reference solutions. CPU checks."""

# Create the original four-element array
import jax.numpy as jnp
x = jnp.array([1., 2., 3., 4.])

# Set one index and add to two others
y = x.at[1].set(20.)
z = x.at[jnp.array([0, 2])].add(5.)

# Check both updated values and the unchanged input
assert jnp.allclose(x, jnp.array([1.,2.,3.,4.]))
assert jnp.allclose(y, jnp.array([1.,20.,3.,4.]))
assert jnp.allclose(z, jnp.array([6.,2.,8.,4.]))

# Immutable updates and indexing: An indexed update in JAX returns an array value.
import jax.numpy as jnp
x = jnp.array([1., 2., 3., 4.])
y = x.at[1].set(20.)
z = x.at[jnp.array([0, 2])].add(5.)
print("Original:", x)
print("Set:", y)
print("Add:", z)
assert jnp.allclose(x, jnp.array([1.,2.,3.,4.]))
assert jnp.allclose(y, jnp.array([1.,20.,3.,4.]))
assert jnp.allclose(z, jnp.array([6.,2.,8.,4.]))

# Figure data experiment
visual_data = {'kind': 'heatmap', 'values': jnp.stack([x, y, z]).tolist(), 'rows': ['original', 'set index 1', 'add at 0 and 2'], 'columns': ['0', '1', '2', '3'], 'unit': 'stored value'}

# Experiment: See name rebinding without old-value mutation
# Experiment — See name rebinding without old-value mutation
original = x
rebound = x
rebound = rebound.at[1].set(20.)
assert jnp.allclose(original, jnp.array([1., 2., 3., 4.]))
assert jnp.allclose(rebound, jnp.array([1., 20., 3., 4.]))
ignored = x.at[0].set(99.)
assert jnp.allclose(x, original)
assert float(ignored[0]) == 99.

# Experiment: Accumulate repeated integer contributions
# Experiment — Accumulate repeated integer contributions: The integer loop reference makes accumulation explicit.
import numpy as np
indices = jnp.array([1, 1, 3])
contributions = jnp.array([2, 3, 7], dtype=jnp.int32)
base_counts = jnp.zeros(4, dtype=jnp.int32)
counts = base_counts.at[indices].add(contributions)
reference_counts = np.zeros(4, dtype=np.int32)
for index, contribution in zip([1, 1, 3], [2, 3, 7]):
    # Accumulate the next contribution into `reference_counts[index]`.
    reference_counts[index] += contribution
assert jnp.array_equal(counts, reference_counts)
assert jnp.array_equal(counts, jnp.array([0, 5, 0, 7]))
assert jnp.array_equal(base_counts, jnp.zeros(4, dtype=jnp.int32))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Replace the last two elements with zero in a new array.
masked = x.at[2:].set(0.)
assert jnp.allclose(masked, jnp.array([1.,2.,0.,0.]))
assert jnp.allclose(x, jnp.array([1.,2.,3.,4.]))

# Reference practice: Replace a region of a matrix
# Replace a region of a matrix (Practice)
matrix = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
updated_matrix = matrix.at[:, -1].set(-1.)
assert jnp.allclose(updated_matrix, jnp.array([[0., 1., -1.], [3., 4., -1.], [6., 7., -1.]]))
assert jnp.allclose(matrix, jnp.arange(9).reshape(3, 3))

# Reference practice: Keep shape while masking values
# Keep shape while masking values (Challenge)
fixed_shape = jnp.where(x > 2., x, 0.)
selected = x[x > 2.]
# Check tensor shape invariant: `fixed_shape.shape == (4,)`
assert fixed_shape.shape == (4,)
# Check tensor shape invariant: `selected.shape == (2,)`
assert selected.shape == (2,)
assert jnp.allclose(fixed_shape, jnp.array([0., 0., 3., 4.]))
assert jnp.allclose(selected, jnp.array([3., 4.]))
print("PASS: arrays-04")
