"""Immutable updates and indexing: worked experiments and reference solutions. CPU checks."""



# Immutable updates and indexing: An indexed update in JAX returns an array value.
# Import jax.numpy for this computation.
import jax.numpy as jnp
# Initialize array `x` with explicit values and shape.
x = jnp.array([1., 2., 3., 4.])
# Evaluate `y` from the current inputs and state.
y = x.at[1].set(20.)
# Initialize array `z` with explicit values and shape.
z = x.at[jnp.array([0, 2])].add(5.)
# Print the observed values to compare against the expected result.
print("Original:", x)
# Print diagnostic summary of the computed outputs.
print("Set:", y)
# Print diagnostic summary of the computed outputs.
print("Add:", z)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(x, jnp.array([1.,2.,3.,4.]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(y, jnp.array([1.,20.,3.,4.]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(z, jnp.array([6.,2.,8.,4.]))

# Figure data experiment
# Compute figure data for: An update returns a new array
# Combine or mask array elements to form `visual_data`.
visual_data = {'kind': 'heatmap', 'values': jnp.stack([x, y, z]).tolist(), 'rows': ['original', 'set index 1', 'add at 0 and 2'], 'columns': ['0', '1', '2', '3'], 'unit': 'stored value'}

# Experiment: See name rebinding without old-value mutation
# Experiment — See name rebinding without old-value mutation: Assigning a result changes the binding, while ignoring a result...
original = x
# Evaluate `rebound` from the current inputs and state.
rebound = x
# Evaluate `rebound` from the current inputs and state.
rebound = rebound.at[1].set(20.)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(original, jnp.array([1., 2., 3., 4.]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(rebound, jnp.array([1., 20., 3., 4.]))
# Evaluate `ignored` from the current inputs and state.
ignored = x.at[0].set(99.)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(x, original)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert float(ignored[0]) == 99.

# Experiment: Accumulate repeated integer contributions
# Experiment — Accumulate repeated integer contributions: The integer loop reference makes accumulation explicit.
# Import numpy for this computation.
import numpy as np
# Initialize array `indices` with explicit values and shape.
indices = jnp.array([1, 1, 3])
# Initialize array `contributions` with explicit values and shape.
contributions = jnp.array([2, 3, 7], dtype=jnp.int32)
# Initialize array `base_counts` with explicit values and shape.
base_counts = jnp.zeros(4, dtype=jnp.int32)
# Evaluate `counts` from the current inputs and state.
counts = base_counts.at[indices].add(contributions)
# Allocate initialized array `reference_counts` with the specified shape and dtype.
reference_counts = np.zeros(4, dtype=np.int32)
# Iterate over `(index, contribution)` to step through the computation:
for index, contribution in zip([1, 1, 3], [2, 3, 7]):
    # Accumulate the next contribution into `reference_counts[index]`.
    reference_counts[index] += contribution
# Verify contract: `jnp.array_equal(counts, reference_counts)`.
assert jnp.array_equal(counts, reference_counts)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.array_equal(counts, jnp.array([0, 5, 0, 7]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.array_equal(base_counts, jnp.zeros(4, dtype=jnp.int32))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Replace the last two elements with zero in a new array.
masked = x.at[2:].set(0.)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(masked, jnp.array([1.,2.,0.,0.]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(x, jnp.array([1.,2.,3.,4.]))

# Reference practice: Replace a region of a matrix
# Replace a region of a matrix (Practice): Checking the entire matrix verifies both changed and...
# Construct and reshape `matrix` into the target tensor dimensions.
matrix = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
# Evaluate `updated_matrix` from the current inputs and state.
updated_matrix = matrix.at[:, -1].set(-1.)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(updated_matrix, jnp.array([[0., 1., -1.], [3., 4., -1.], [6., 7., -1.]]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(matrix, jnp.arange(9).reshape(3, 3))

# Reference practice: Keep shape while masking values
# Keep shape while masking values (Challenge): Both are useful representations, but they are not...
fixed_shape = jnp.where(x > 2., x, 0.)
# Evaluate `selected` from the current inputs and state.
selected = x[x > 2.]
# Verify that the output tensor shape matches our prediction.
assert fixed_shape.shape == (4,)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert selected.shape == (2,)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(fixed_shape, jnp.array([0., 0., 3., 4.]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(selected, jnp.array([3., 4.]))
print("PASS: arrays-04")
