"""Shapes, broadcasting, and dtypes: worked experiments and reference solutions. CPU checks."""

# Step 1: Set up imports and input tensors
import jax.numpy as jnp
# Construct `batch` via `jnp.array([[1., 2., 3.], [4., 5., 6.]], dtype=jnp.fl...`
batch = jnp.array([[1., 2., 3.], [4., 5., 6.]], dtype=jnp.float32)

# Step 2: Apply the core JAX transformation
bias = jnp.array([10., 20., 30.], dtype=jnp.float32)
# Compute `y` from `batch + bias`
y = batch + bias

# Step 3: Verify shapes and numerical invariants
assert y.shape == (2, 3)
# Check numerical equivalence within tolerance: `jnp.allclose(y[1], jnp.array([14., 25., 36.]))`
assert jnp.allclose(y[1], jnp.array([14., 25., 36.]))

# Shapes, broadcasting, and dtypes: Before adding or subtracting arrays, name what each axis represents.
# Import jax.numpy for this computation.
import jax.numpy as jnp
# Construct `batch` via `jnp.array([[1., 2., 3.], [4., 5., 6.]], dtype=jnp.fl...`
batch = jnp.array([[1., 2., 3.], [4., 5., 6.]], dtype=jnp.float32)
# Construct `bias` via `jnp.array([10., 20., 30.], dtype=jnp.float32)`
bias = jnp.array([10., 20., 30.], dtype=jnp.float32)
# Compute `y` from `batch + bias`
y = batch + bias
# Print the observed values to compare against the expected result.
print(y)
# Print diagnostic summary of the computed outputs.
print("Shape:", y.shape, "dtype:", y.dtype)
# Check tensor shape invariant: `y.shape == (2, 3)`
assert y.shape == (2, 3)
# Check numerical equivalence within tolerance: `jnp.allclose(y[1], jnp.array([14., 25., 36.]))`
assert jnp.allclose(y[1], jnp.array([14., 25., 36.]))

# Figure data experiment
# Compute figure data for: Broadcasting repeats the bias across rows
# Compute `visual_data` from `{'kind': 'heatmap', 'values': (y - batch).tolist(), ...`
visual_data = {'kind': 'heatmap', 'values': (y - batch).tolist(), 'rows': ['observation 0', 'observation 1'], 'columns': ['feature 0', 'feature 1', 'feature 2'], 'unit': 'added bias'}

# Experiment: Make the pairwise bug visible
# Experiment — Make the pairwise bug visible: The final result is scalar in both cases.
# Construct `predictions` via `jnp.array([1., 3., 5.])`
predictions = jnp.array([1., 3., 5.])
# Construct `targets` via `jnp.array([1., 3., 5.])`
targets = jnp.array([1., 3., 5.])
# Compute `correct_residuals` from `predictions - targets`
correct_residuals = predictions - targets
# Compute `pairwise_residuals` from `predictions - targets[:, None]`
pairwise_residuals = predictions - targets[:, None]
# Print the observed values to compare against the expected result.
print("aligned residuals:", correct_residuals)
# Print diagnostic summary of the computed outputs.
print("pairwise residuals:", pairwise_residuals)
# Check tensor shape invariant: `correct_residuals.shape == (3,)`
assert correct_residuals.shape == (3,)
# Check tensor shape invariant: `pairwise_residuals.shape == (3, 3)`
assert pairwise_residuals.shape == (3, 3)
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.mean(correct_residuals ** 2), 0.)`
assert jnp.allclose(jnp.mean(correct_residuals ** 2), 0.)
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.mean(pairwise_residuals ** 2), 16. / 3.)`
assert jnp.allclose(jnp.mean(pairwise_residuals ** 2), 16. / 3.)

# Experiment: Observe a precision limit separately
# Experiment — Observe a precision limit separately: This scalar example isolates precision from broadcasting.
# Construct `large` via `jnp.array(100_000_000., dtype=jnp.float32)`
large = jnp.array(100_000_000., dtype=jnp.float32)
# Print the observed values to compare against the expected result.
print("float32 large + 1 − large:", float((large + 1.) - large))
# Assert invariant `float((large + 1.) - large) == 0.` holds
assert float((large + 1.) - large) == 0.
# Check tensor shape invariant: `large.shape == ()`
assert large.shape == ()

# Reference solution. Try the exercise before reading this.
# Exercise solution: Add a different scalar offset to each row using offsets [100., 200.].
# Construct `offsets` via `jnp.array([100., 200.])[:, None]`
offsets = jnp.array([100., 200.])[:, None]
# Compute `z` from `batch + offsets`
z = batch + offsets
# Check tensor shape invariant: `offsets.shape == (2, 1)`
assert offsets.shape == (2, 1)
# Check numerical equivalence within tolerance: `jnp.allclose(z, jnp.array([[101.,102.,103.],[204.,205.,206.]]))`
assert jnp.allclose(z, jnp.array([[101.,102.,103.],[204.,205.,206.]]))

# Reference practice: Use both kinds of offsets together
# Use both kinds of offsets together (Practice): The first row receives 100 in every feature and the second...
# Construct `combined` via `batch + bias + jnp.array([100., 200.])[:, None]`
combined = batch + bias + jnp.array([100., 200.])[:, None]
# Check tensor shape invariant: `combined.shape == (2, 3)`
assert combined.shape == (2, 3)
# Check numerical equivalence within tolerance: `jnp.allclose(combined, jnp.array([[111., 122., 133.], [214., 225....`
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
