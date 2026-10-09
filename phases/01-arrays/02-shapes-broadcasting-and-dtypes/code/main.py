"""Shapes, broadcasting, and dtypes: worked experiments and reference solutions. CPU checks."""



# Shapes, broadcasting, and dtypes: Before adding or subtracting arrays, name what each axis represents.
# Import jax.numpy for this computation.
import jax.numpy as jnp
# Initialize array `batch` with explicit values and shape.
batch = jnp.array([[1., 2., 3.], [4., 5., 6.]], dtype=jnp.float32)
# Initialize array `bias` with explicit values and shape.
bias = jnp.array([10., 20., 30.], dtype=jnp.float32)
# Evaluate `y` from the current inputs and state.
y = batch + bias
# Print the observed values to compare against the expected result.
print(y)
# Print diagnostic summary of the computed outputs.
print("Shape:", y.shape, "dtype:", y.dtype)
# Verify that the output tensor shape matches our prediction.
assert y.shape == (2, 3)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(y[1], jnp.array([14., 25., 36.]))

# Figure data experiment
# Compute figure data for: Broadcasting repeats the bias across rows
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'heatmap', 'values': (y - batch).tolist(), 'rows': ['observation 0', 'observation 1'], 'columns': ['feature 0', 'feature 1', 'feature 2'], 'unit': 'added bias'}

# Experiment: Make the pairwise bug visible
# Experiment — Make the pairwise bug visible: The final result is scalar in both cases.
# Initialize array `predictions` with explicit values and shape.
predictions = jnp.array([1., 3., 5.])
# Initialize array `targets` with explicit values and shape.
targets = jnp.array([1., 3., 5.])
# Evaluate `correct_residuals` from the current inputs and state.
correct_residuals = predictions - targets
# Evaluate `pairwise_residuals` from the current inputs and state.
pairwise_residuals = predictions - targets[:, None]
# Print the observed values to compare against the expected result.
print("aligned residuals:", correct_residuals)
# Print diagnostic summary of the computed outputs.
print("pairwise residuals:", pairwise_residuals)
# Verify that the output tensor shape matches our prediction.
assert correct_residuals.shape == (3,)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert pairwise_residuals.shape == (3, 3)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.mean(correct_residuals ** 2), 0.)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.mean(pairwise_residuals ** 2), 16. / 3.)

# Experiment: Observe a precision limit separately
# Experiment — Observe a precision limit separately: This scalar example isolates precision from broadcasting.
# Initialize array `large` with explicit values and shape.
large = jnp.array(100_000_000., dtype=jnp.float32)
# Print the observed values to compare against the expected result.
print("float32 large + 1 − large:", float((large + 1.) - large))
# Verify contract: `float(large + 1.0 - large) == 0.0`.
assert float((large + 1.) - large) == 0.
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert large.shape == ()

# Reference solution. Try the exercise before reading this.
# Exercise solution: Add a different scalar offset to each row using offsets [100., 200.].
# Initialize array `offsets` with explicit values and shape.
offsets = jnp.array([100., 200.])[:, None]
# Evaluate `z` from the current inputs and state.
z = batch + offsets
# Verify that the output tensor shape matches our prediction.
assert offsets.shape == (2, 1)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(z, jnp.array([[101.,102.,103.],[204.,205.,206.]]))

# Reference practice: Use both kinds of offsets together
# Use both kinds of offsets together (Practice): The first row receives 100 in every feature and the second...
# Initialize array `combined` with explicit values and shape.
combined = batch + bias + jnp.array([100., 200.])[:, None]
# Verify that the output tensor shape matches our prediction.
assert combined.shape == (2, 3)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(combined, jnp.array([[111., 122., 133.], [214., 225., 236.]]))

# Reference practice: Reject an aligned-loss contract violation
# Reject an aligned-loss contract violation (Challenge): Fix shapes at the data/model interface according to the...
def aligned_mse(prediction, target):
    # Guard input contract (`prediction.shape != target.shape`) and fail fast if violated.
    if prediction.shape != target.shape:
        raise ValueError("aligned prediction and target shapes must match")
    # Return `jnp.mean((prediction - target) ** 2)` to the caller.
    return jnp.mean((prediction - target) ** 2)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(aligned_mse(predictions, targets), 0.)
# Run the boundary check and catch the expected exception:
try:
    aligned_mse(predictions, targets[:, None])
except ValueError:
    pass
else:
    raise AssertionError("Column targets must be rejected")
print("PASS: arrays-02")
