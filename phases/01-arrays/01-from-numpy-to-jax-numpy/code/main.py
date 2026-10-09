"""From NumPy to jax.numpy: worked experiments and reference solutions. CPU checks."""

# Step 1: Set up imports and input tensors
import jax.numpy as jnp
# Construct `x` via `jnp.array([[1., 10.], [3., 14.], [5., 18.]])`
x = jnp.array([[1., 10.], [3., 14.], [5., 18.]])

# Step 2: Apply the core JAX transformation
mean = x.mean(axis=0)
# Compute `centered` from `x - mean`
centered = x - mean

# Step 3: Verify shapes and numerical invariants
assert jnp.allclose(mean, jnp.array([3., 14.]))
# Assert that `jnp.allclose(centered.mean(axis=0), jnp.zeros(2))`.
assert jnp.allclose(centered.mean(axis=0), jnp.zeros(2))

# From NumPy to jax.numpy: Feature preprocessing answers a specific question: how does each...
# Import jax.numpy for this computation.
import jax.numpy as jnp
# Construct `x` via `jnp.array([[1., 10.], [3., 14.], [5., 18.]])`
x = jnp.array([[1., 10.], [3., 14.], [5., 18.]])
# Reduce along axis=0 to compute `mean`.
mean = x.mean(axis=0)
# Compute `centered` from `x - mean`
centered = x - mean
# Print the observed values to compare against the expected result.
print("Mean:", mean)
# Print diagnostic summary of the computed outputs.
print("Centered:\n", centered)
# Assert that `jnp.allclose(mean, jnp.array([3., 14.]))`.
assert jnp.allclose(mean, jnp.array([3., 14.]))
# Assert that `jnp.allclose(centered.mean(axis=0), jnp.zeros(2))`.
assert jnp.allclose(centered.mean(axis=0), jnp.zeros(2))

# Figure data experiment
# Compute figure data for: Center each feature around its own mean
# Compute `visual_data` from `{'kind': 'line', 'x': [0, 1, 2], 'xlabel': 'observat...`
visual_data = {'kind': 'line', 'x': [0, 1, 2], 'xlabel': 'observation index', 'ylabel': 'feature value', 'series': [{'label': 'original feature 0', 'y': x[:, 0].tolist()}, {'label': 'centered feature 0', 'y': centered[:, 0].tolist()}, {'label': 'original feature 1', 'y': x[:, 1].tolist()}, {'label': 'centered feature 1', 'y': centered[:, 1].tolist()}]}

# Experiment: Check the normalization arithmetic
# Experiment — Check the normalization arithmetic: Matching dtype and statistic definition makes this a useful...
# Import numpy for this computation.
import numpy as np
# Compute `reference_data` from `np.array([[1., 10.], [3., 14.], [5., 18.]], dtype=np...`
reference_data = np.array([[1., 10.], [3., 14.], [5., 18.]], dtype=np.float32)
# Reduce along axis=0 to compute `reference_mean`.
reference_mean = reference_data.mean(axis=0)
# Reduce along axis=0 to compute `reference_scale`.
reference_scale = reference_data.std(axis=0)
# Assert that `jnp.allclose(mean, reference_mean)`.
assert jnp.allclose(mean, reference_mean)
# Assert that `jnp.allclose(x.std(axis=0), reference_scale)`.
assert jnp.allclose(x.std(axis=0), reference_scale)
# Assert that `jnp.allclose(x.std(axis=0), jnp.sqrt(jnp.array([8. / 3., 32. / 3.])))`.
assert jnp.allclose(x.std(axis=0), jnp.sqrt(jnp.array([8. / 3., 32. / 3.])))

# Experiment: Observe a new batch in training coordinates
# Experiment — Observe a new batch in training coordinates: The new observation is measured relative to training data.
def fit_standardizer(training_data):
    # Reduce along axis=0 to compute `fitted_mean`.
    fitted_mean = training_data.mean(axis=0)
    # Reduce along axis=0 to compute `fitted_std`.
    fitted_std = training_data.std(axis=0)
    # Combine or mask array elements to form `fitted_scale`.
    fitted_scale = jnp.where(fitted_std == 0., 1., fitted_std)
    # Return `(fitted_mean, fitted_scale)` to the caller.
    return fitted_mean, fitted_scale
# Function `apply_standardizer(data, statistics)` implementing this stage's computation:
def apply_standardizer(data, statistics):
    # Compute `fitted_mean, fitted_scale` from `statistics`
    fitted_mean, fitted_scale = statistics
    # Return `(data - fitted_mean) / fitted_scale` to the caller.
    return (data - fitted_mean) / fitted_scale
# Run `fit_standardizer` to compute `statistics`.
statistics = fit_standardizer(x)
# Construct `new_batch` via `jnp.array([[7., 22.]])`
new_batch = jnp.array([[7., 22.]])
# Run `apply_standardizer` to compute `new_values`.
new_values = apply_standardizer(new_batch, statistics)
# Print the observed values to compare against the expected result.
print("new batch transformed:", new_values)
# Assert that `jnp.allclose(new_values, jnp.full((1, 2), jnp.sqrt(6.)), atol=1e-6)`.
assert jnp.allclose(new_values, jnp.full((1, 2), jnp.sqrt(6.)), atol=1e-6)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Foundation · Standardize the original data using its standard...
# Reduce along axis=0 to compute `standardized`.
standardized = centered / (x.std(axis=0) + 1e-6)
# Assert that `jnp.allclose(standardized.mean(axis=0), 0., atol=1e-6)`.
assert jnp.allclose(standardized.mean(axis=0), 0., atol=1e-6)
# Assert that `jnp.allclose(standardized.std(axis=0), 1., atol=1e-5)`.
assert jnp.allclose(standardized.std(axis=0), 1., atol=1e-5)

# Reference practice: Test the constant-feature policy
# Test the constant-feature policy (Practice): The training feature is constant but the new value differs...
# Construct `constant_data` via `jnp.array([[1., 5.], [3., 5.], [5., 5.]])`
constant_data = jnp.array([[1., 5.], [3., 5.], [5., 5.]])
# Run `fit_standardizer` to compute `constant_statistics`.
constant_statistics = fit_standardizer(constant_data)
# Run `apply_standardizer` to compute `constant_transformed`.
constant_transformed = apply_standardizer(constant_data, constant_statistics)
# Confirm that all computed values remain finite (no NaN or Inf).
assert jnp.all(jnp.isfinite(constant_transformed))
# Assert that `jnp.allclose(constant_transformed[:, 1], 0.)`.
assert jnp.allclose(constant_transformed[:, 1], 0.)
# Assert that `jnp.allclose(apply_standardizer(jnp.array([[7., 6.]]), constant_statistics)[0, 1], 1.)`.
assert jnp.allclose(apply_standardizer(jnp.array([[7., 6.]]), constant_statistics)[0, 1], 1.)

# Reference practice: Detect accidental batch-dependent preprocessing
# Detect accidental batch-dependent preprocessing (Challenge): This consistency test expresses an inference boundary: a...
# Construct `new_rows` via `jnp.array([[7., 22.], [9., 26.]])`
new_rows = jnp.array([[7., 22.], [9., 26.]])
# Run `apply_standardizer` to compute `together`.
together = apply_standardizer(new_rows, statistics)
# Combine or mask array elements to form `separately`.
separately = jnp.concatenate([apply_standardizer(new_rows[i:i+1], statistics) for i in range(2)])
# Assert that `jnp.allclose(together, separately)`.
assert jnp.allclose(together, separately)
# Run `apply_standardizer` to compute `refitted`.
refitted = apply_standardizer(new_rows, fit_standardizer(new_rows))
# Assert that `not jnp.allclose(together, refitted)`.
assert not jnp.allclose(together, refitted)
print("PASS: arrays-01")
