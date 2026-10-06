"""Linear algebra and loss intuition: worked experiments and reference solutions. CPU checks."""

# 1. Prepare the inputs
import jax
import jax.numpy as jnp

# 2. Define the computation
X = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
w = jnp.array([2., -1.])
targets = jnp.array([3., 0., 2.])
def predict(w, bias, X):
    return X @ w + bias
def mse(w, bias):
    prediction = predict(w, bias, X)
    assert prediction.shape == targets.shape
    return jnp.mean((prediction - targets) ** 2)

# 3. Measure and verify
print("Predictions:", predict(w, 1., X))
print("Loss:", float(mse(w, 1.)))
assert jnp.allclose(predict(w, 1., X), targets)
assert jnp.allclose(mse(w, 1.), 0.)

import jax
import jax.numpy as jnp
X = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
w = jnp.array([2., -1.])
targets = jnp.array([3., 0., 2.])
def predict(w, bias, X):
    return X @ w + bias
def mse(w, bias):
    prediction = predict(w, bias, X)
    assert prediction.shape == targets.shape
    return jnp.mean((prediction - targets) ** 2)
print("Predictions:", predict(w, 1., X))
print("Loss:", float(mse(w, 1.)))
assert jnp.allclose(predict(w, 1., X), targets)
assert jnp.allclose(mse(w, 1.), 0.)

# Figure data experiment
visual_data = {'kind': 'bar', 'labels': ['observation 0', 'observation 1', 'observation 2'], 'ylabel': 'target / prediction', 'series': [{'label': 'target', 'y': targets.tolist()}, {'label': 'bias 1', 'y': predict(w, 1.0, X).tolist()}, {'label': 'bias 0', 'y': predict(w, 0.0, X).tolist()}]}

# Experiment: Verify the gradient by hand
gw, gb = jax.grad(mse, argnums=(0, 1))(w, 0.)
assert jnp.allclose(gw, jnp.array([-4/3, -4/3]))
assert jnp.allclose(gb, -2.)
print("Analytic gradients:", gw, gb)

# Experiment: Make the silent shape error visible
column_targets = targets[:, None]
pairwise = predict(w, 1., X) - column_targets
assert pairwise.shape == (3, 3)
assert jnp.allclose(jnp.mean(pairwise**2), 28/9)
print("Wrong residual shape:", pairwise.shape)

# Experiment: Change the reduction, predict the update
def mean_objective(weights):
    return jnp.mean((X @ weights - targets)**2)
def sum_objective(weights):
    return jnp.sum((X @ weights - targets)**2)
point = jnp.array([0.2, -0.4])
residual = X @ point - targets
analytic_mean = 2 * X.T @ residual / X.shape[0]
assert jnp.allclose(jax.grad(mean_objective)(point), analytic_mean, atol=1e-6)
assert jnp.allclose(jax.grad(sum_objective)(point), X.shape[0]*analytic_mean, atol=1e-6)

# Reference solution. Try the exercise before reading this.
residual = predict(w, 0., X) - targets
assert jnp.allclose(residual, -jnp.ones(3))
assert jnp.allclose(mse(w, 0.), 1.)

# Reference practice: Add an unseen observation
X_new = jnp.concatenate([X, jnp.array([[2., -1.]])])
y_new = jnp.concatenate([targets, jnp.array([6.])])
assert jnp.allclose(predict(w, 1., X_new), y_new)

# Reference practice: Reject and repair a column target
def checked_loss(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("one target required per prediction")
    return jnp.mean((prediction-target)**2)
try:
    checked_loss(predict(w, 1., X), column_targets)
except ValueError:
    pass
else:
    raise AssertionError("bad shape accepted")
assert checked_loss(predict(w, 1., X), column_targets[:, 0]) == 0
print("PASS: optimization-01")
