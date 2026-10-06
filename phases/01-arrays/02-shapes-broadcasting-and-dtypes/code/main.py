"""Shapes, broadcasting, and dtypes: worked experiments and reference solutions. CPU checks."""



import jax.numpy as jnp
batch = jnp.array([[1., 2., 3.], [4., 5., 6.]], dtype=jnp.float32)
bias = jnp.array([10., 20., 30.], dtype=jnp.float32)
y = batch + bias
print(y)
print("Shape:", y.shape, "dtype:", y.dtype)
assert y.shape == (2, 3)
assert jnp.allclose(y[1], jnp.array([14., 25., 36.]))

# Figure data experiment
visual_data = {'kind': 'heatmap', 'values': (y - batch).tolist(), 'rows': ['observation 0', 'observation 1'], 'columns': ['feature 0', 'feature 1', 'feature 2'], 'unit': 'added bias'}

# Experiment: Make the pairwise bug visible
predictions = jnp.array([1., 3., 5.])
targets = jnp.array([1., 3., 5.])
correct_residuals = predictions - targets
pairwise_residuals = predictions - targets[:, None]
print("aligned residuals:", correct_residuals)
print("pairwise residuals:", pairwise_residuals)
assert correct_residuals.shape == (3,)
assert pairwise_residuals.shape == (3, 3)
assert jnp.allclose(jnp.mean(correct_residuals ** 2), 0.)
assert jnp.allclose(jnp.mean(pairwise_residuals ** 2), 16. / 3.)

# Experiment: Observe a precision limit separately
large = jnp.array(100_000_000., dtype=jnp.float32)
print("float32 large + 1 − large:", float((large + 1.) - large))
assert float((large + 1.) - large) == 0.
assert large.shape == ()

# Reference solution. Try the exercise before reading this.
offsets = jnp.array([100., 200.])[:, None]
z = batch + offsets
assert offsets.shape == (2, 1)
assert jnp.allclose(z, jnp.array([[101.,102.,103.],[204.,205.,206.]]))

# Reference practice: Use both kinds of offsets together
combined = batch + bias + jnp.array([100., 200.])[:, None]
assert combined.shape == (2, 3)
assert jnp.allclose(combined, jnp.array([[111., 122., 133.], [214., 225., 236.]]))

# Reference practice: Reject an aligned-loss contract violation
def aligned_mse(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("aligned prediction and target shapes must match")
    return jnp.mean((prediction - target) ** 2)
assert jnp.allclose(aligned_mse(predictions, targets), 0.)
try:
    aligned_mse(predictions, targets[:, None])
except ValueError:
    pass
else:
    raise AssertionError("Column targets must be rejected")
print("PASS: arrays-02")
