"""Linear algebra and loss intuition: worked experiments and reference solutions. CPU checks."""

# 1. Prepare the inputs
# Step 1 — 1. Prepare the inputs: This block establishes the values used by the following steps; run...
# Import jax for this computation.
import jax
import jax.numpy as jnp

# 2. Define the computation
# Step 2 — 2. Define the computation: predict contracts the feature axis through Xw.
# Construct `X` via `jnp.array([[1., 0.], [0., 1.], [1., 1.]])`
X = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
# Construct `w` via `jnp.array([2., -1.])`
w = jnp.array([2., -1.])
# Construct `targets` via `jnp.array([3., 0., 2.])`
targets = jnp.array([3., 0., 2.])
# Function `predict(w, bias, X)` implementing this stage's computation:
def predict(w, bias, X):
    # Return `X @ w + bias` to the caller.
    return X @ w + bias
# Function `mse(w, bias)` implementing this stage's computation:
def mse(w, bias):
    # Run `predict` to compute `prediction`.
    prediction = predict(w, bias, X)
    # Check tensor shape invariant: `prediction.shape == targets.shape`
    assert prediction.shape == targets.shape
    # Return `jnp.mean((prediction - targets) ** 2)` to the caller.
    return jnp.mean((prediction - targets) ** 2)

# 3. Measure and verify
# Step 3 — 3. Measure and verify: Predictions: [3., 0., 2.]; Loss: 0.0.
# Print the observed values to compare against the expected result.
print("Predictions:", predict(w, 1., X))
# Print diagnostic summary of the computed outputs.
print("Loss:", float(mse(w, 1.)))
# Assert that `jnp.allclose(predict(w, 1., X), targets)`.
assert jnp.allclose(predict(w, 1., X), targets)
# Assert that `jnp.allclose(mse(w, 1.), 0.)`.
assert jnp.allclose(mse(w, 1.), 0.)

# Step 1 — 1. Prepare the inputs: This block establishes the values used by the following steps; run...
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Step 2 — 2. Define the computation: predict contracts the feature axis through Xw.
# Construct `X` via `jnp.array([[1., 0.], [0., 1.], [1., 1.]])`
X = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
# Construct `w` via `jnp.array([2., -1.])`
w = jnp.array([2., -1.])
# Construct `targets` via `jnp.array([3., 0., 2.])`
targets = jnp.array([3., 0., 2.])
# Function `predict(w, bias, X)` implementing this stage's computation:
def predict(w, bias, X):
    # Return `X @ w + bias` to the caller.
    return X @ w + bias
# Function `mse(w, bias)` implementing this stage's computation:
def mse(w, bias):
    # Run `predict` to compute `prediction`.
    prediction = predict(w, bias, X)
    # Check tensor shape invariant: `prediction.shape == targets.shape`
    assert prediction.shape == targets.shape
    # Return `jnp.mean((prediction - targets) ** 2)` to the caller.
    return jnp.mean((prediction - targets) ** 2)
# Step 3 — 3. Measure and verify: Predictions: [3., 0., 2.]; Loss: 0.0.
# Print the observed values to compare against the expected result.
print("Predictions:", predict(w, 1., X))
# Print diagnostic summary of the computed outputs.
print("Loss:", float(mse(w, 1.)))
# Assert that `jnp.allclose(predict(w, 1., X), targets)`.
assert jnp.allclose(predict(w, 1., X), targets)
# Assert that `jnp.allclose(mse(w, 1.), 0.)`.
assert jnp.allclose(mse(w, 1.), 0.)

# Figure data experiment
# Compute figure data for: A scalar bias shifts every prediction
# Compute `visual_data` from `{'kind': 'bar', 'labels': ['observation 0', 'observa...`
visual_data = {'kind': 'bar', 'labels': ['observation 0', 'observation 1', 'observation 2'], 'ylabel': 'target / prediction', 'series': [{'label': 'target', 'y': targets.tolist()}, {'label': 'bias 1', 'y': predict(w, 1.0, X).tolist()}, {'label': 'bias 0', 'y': predict(w, 0.0, X).tolist()}]}

# Experiment: Verify the gradient by hand
# Experiment — Verify the gradient by hand: The matrix formula and autodiff agree for the same row-matched...
# Differentiate the objective to obtain `(gw, gb)` via automatic differentiation.
gw, gb = jax.grad(mse, argnums=(0, 1))(w, 0.)
# Assert that `jnp.allclose(gw, jnp.array([-4/3, -4/3]))`.
assert jnp.allclose(gw, jnp.array([-4/3, -4/3]))
# Assert that `jnp.allclose(gb, -2.)`.
assert jnp.allclose(gb, -2.)
# Print the observed values to compare against the expected result.
print("Analytic gradients:", gw, gb)

# Experiment: Make the silent shape error visible
# Experiment — Make the silent shape error visible: A valid broadcast can define an unintended objective without...
column_targets = targets[:, None]
# Run `predict` to compute `pairwise`.
pairwise = predict(w, 1., X) - column_targets
# Check tensor shape invariant: `pairwise.shape == (3, 3)`
assert pairwise.shape == (3, 3)
# Assert that `jnp.allclose(jnp.mean(pairwise**2), 28/9)`.
assert jnp.allclose(jnp.mean(pairwise**2), 28/9)
# Print the observed values to compare against the expected result.
print("Wrong residual shape:", pairwise.shape)

# Experiment: Change the reduction, predict the update
# Experiment — Change the reduction, predict the update: Loss reduction is part of the optimization specification.
def mean_objective(weights):
    # Return `jnp.mean((X @ weights - targets) ** 2)` to the caller.
    return jnp.mean((X @ weights - targets)**2)
# Function `sum_objective(weights)` implementing this stage's computation:
def sum_objective(weights):
    # Return `jnp.sum((X @ weights - targets) ** 2)` to the caller.
    return jnp.sum((X @ weights - targets)**2)
# Construct `point` via `jnp.array([0.2, -0.4])`
point = jnp.array([0.2, -0.4])
# Perform matrix contraction / projection to compute `residual`.
residual = X @ point - targets
# Perform matrix contraction / projection to compute `analytic_mean`.
analytic_mean = 2 * X.T @ residual / X.shape[0]
# Assert that `jnp.allclose(jax.grad(mean_objective)(point), analytic_mean, atol=1e-6)`.
assert jnp.allclose(jax.grad(mean_objective)(point), analytic_mean, atol=1e-6)
# Check tensor shape invariant: `jnp.allclose(jax.grad(sum_objective)(point), X.shape[0]*analytic_...`
assert jnp.allclose(jax.grad(sum_objective)(point), X.shape[0]*analytic_mean, atol=1e-6)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Set the bias to zero.
residual = predict(w, 0., X) - targets
# Assert that `jnp.allclose(residual, -jnp.ones(3))`.
assert jnp.allclose(residual, -jnp.ones(3))
# Assert that `jnp.allclose(mse(w, 0.), 1.)`.
assert jnp.allclose(mse(w, 0.), 1.)

# Reference practice: Add an unseen observation
# Add an unseen observation (Transfer / diagnosis): The fourth row checks a combination absent from the original...
# Construct `X_new` via `jnp.concatenate([X, jnp.array([[2., -1.]])])`
X_new = jnp.concatenate([X, jnp.array([[2., -1.]])])
# Construct `y_new` via `jnp.concatenate([targets, jnp.array([6.])])`
y_new = jnp.concatenate([targets, jnp.array([6.])])
# Assert that `jnp.allclose(predict(w, 1., X_new), y_new)`.
assert jnp.allclose(predict(w, 1., X_new), y_new)

# Reference practice: Reject and repair a column target
# Reject and repair a column target (Transfer / diagnosis): The failure is reproduced at the boundary and the repair is...
def checked_loss(prediction, target):
    # Guard input contract (`prediction.shape != target.shape`) and fail fast if violated.
    if prediction.shape != target.shape:
        raise ValueError("one target required per prediction")
    # Return `jnp.mean((prediction - target) ** 2)` to the caller.
    return jnp.mean((prediction-target)**2)
# Run the boundary check and catch the expected exception:
try:
    checked_loss(predict(w, 1., X), column_targets)
except ValueError:
    pass
else:
    raise AssertionError("bad shape accepted")
# Assert invariant `checked_loss(predict(w, 1., X), column_targets[:, 0]) == 0` holds
assert checked_loss(predict(w, 1., X), column_targets[:, 0]) == 0
print("PASS: optimization-01")
