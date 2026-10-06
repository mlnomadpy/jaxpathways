"""From NumPy to jax.numpy: worked experiments and reference solutions. CPU checks."""



import jax.numpy as jnp
x = jnp.array([[1., 10.], [3., 14.], [5., 18.]])
mean = x.mean(axis=0)
centered = x - mean
print("Mean:", mean)
print("Centered:\n", centered)
assert jnp.allclose(mean, jnp.array([3., 14.]))
assert jnp.allclose(centered.mean(axis=0), jnp.zeros(2))

# Figure data experiment
visual_data = {'kind': 'line', 'x': [0, 1, 2], 'xlabel': 'observation index', 'ylabel': 'feature value', 'series': [{'label': 'original feature 0', 'y': x[:, 0].tolist()}, {'label': 'centered feature 0', 'y': centered[:, 0].tolist()}, {'label': 'original feature 1', 'y': x[:, 1].tolist()}, {'label': 'centered feature 1', 'y': centered[:, 1].tolist()}]}

# Experiment: Check the normalization arithmetic
import numpy as np
reference_data = np.array([[1., 10.], [3., 14.], [5., 18.]], dtype=np.float32)
reference_mean = reference_data.mean(axis=0)
reference_scale = reference_data.std(axis=0)
assert jnp.allclose(mean, reference_mean)
assert jnp.allclose(x.std(axis=0), reference_scale)
assert jnp.allclose(x.std(axis=0), jnp.sqrt(jnp.array([8. / 3., 32. / 3.])))

# Experiment: Observe a new batch in training coordinates
def fit_standardizer(training_data):
    fitted_mean = training_data.mean(axis=0)
    fitted_std = training_data.std(axis=0)
    fitted_scale = jnp.where(fitted_std == 0., 1., fitted_std)
    return fitted_mean, fitted_scale
def apply_standardizer(data, statistics):
    fitted_mean, fitted_scale = statistics
    return (data - fitted_mean) / fitted_scale
statistics = fit_standardizer(x)
new_batch = jnp.array([[7., 22.]])
new_values = apply_standardizer(new_batch, statistics)
print("new batch transformed:", new_values)
assert jnp.allclose(new_values, jnp.full((1, 2), jnp.sqrt(6.)), atol=1e-6)

# Reference solution. Try the exercise before reading this.
standardized = centered / (x.std(axis=0) + 1e-6)
assert jnp.allclose(standardized.mean(axis=0), 0., atol=1e-6)
assert jnp.allclose(standardized.std(axis=0), 1., atol=1e-5)

# Reference practice: Test the constant-feature policy
constant_data = jnp.array([[1., 5.], [3., 5.], [5., 5.]])
constant_statistics = fit_standardizer(constant_data)
constant_transformed = apply_standardizer(constant_data, constant_statistics)
assert jnp.all(jnp.isfinite(constant_transformed))
assert jnp.allclose(constant_transformed[:, 1], 0.)
assert jnp.allclose(apply_standardizer(jnp.array([[7., 6.]]), constant_statistics)[0, 1], 1.)

# Reference practice: Detect accidental batch-dependent preprocessing
new_rows = jnp.array([[7., 22.], [9., 26.]])
together = apply_standardizer(new_rows, statistics)
separately = jnp.concatenate([apply_standardizer(new_rows[i:i+1], statistics) for i in range(2)])
assert jnp.allclose(together, separately)
refitted = apply_standardizer(new_rows, fit_standardizer(new_rows))
assert not jnp.allclose(together, refitted)
print("PASS: arrays-01")
