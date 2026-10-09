"""Regularization and honest validation: worked experiments and reference solutions. CPU checks."""

# Create separate training and validation examples
# Step 1 — Create separate training and validation examples: The helper solves the explicitly stated mean-loss convention.
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Construct `X` via `jnp.array([[-1.0], [1.0]])`
X = jnp.array([[-1.0], [1.0]])
# Construct `y` via `jnp.array([-3.0, 3.0])`
y = jnp.array([-3.0, 3.0])
# Construct `Xv` via `jnp.array([[-2.0], [2.0]])`
Xv = jnp.array([[-2.0], [2.0]])
# Construct `yv` via `jnp.array([-4.0, 4.0])`
yv = jnp.array([-4.0, 4.0])

# Function `ridge(X, y, lam)` implementing this stage's computation:
def ridge(X, y, lam):
    # Return `jnp.linalg.solve(X.T @ X + X.shape[0] * lam * jnp.eye(X.shape[1]), X.T @ y)` to the caller.
    return jnp.linalg.solve(X.T @ X + X.shape[0] * lam * jnp.eye(X.shape[1]), X.T @ y)

# Function `mse(w, X, y)` implementing this stage's computation:
def mse(w, X, y):
    # Return `jnp.mean((X @ w - y) ** 2)` to the caller.
    return jnp.mean((X @ w - y) ** 2)

# Verify the penalized optimum
# Step 2 — Verify the penalized optimum: The data gradient and penalty gradient cancel at w=2.
lam = 0.5
# Run `ridge` to compute `w`.
w = ridge(X, y, lam)
# Compute `objective` from `lambda w: mse(w, X, y) + lam * jnp.dot(w, w)`
objective = lambda w: mse(w, X, y) + lam * jnp.dot(w, w)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(w, jnp.array([2.0]), atol=1e-06)
# Check numerical equivalence within tolerance: `jnp.allclose(jax.grad(objective)(w), jnp.zeros(1), atol=1e-06)`
assert jnp.allclose(jax.grad(objective)(w), jnp.zeros(1), atol=1e-06)

# Separate the quantities you report
# Step 3 — Separate the quantities you report: Training prediction loss is 1, validation loss is 0, and the...
train = mse(w, X, y)
# Run `mse` to compute `validation`.
validation = mse(w, Xv, yv)
# Run `objective` to compute `penalized`.
penalized = objective(w)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jnp.array([train, validation, penalized]), jnp.array([1.0, 0.0, 3.0]))
# Print the observed values to compare against the expected result.
print('train / validation / objective:', train, validation, penalized)

# Step 1 — Create separate training and validation examples: The helper solves the explicitly stated mean-loss convention.
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Construct `X` via `jnp.array([[-1.0], [1.0]])`
X = jnp.array([[-1.0], [1.0]])
# Construct `y` via `jnp.array([-3.0, 3.0])`
y = jnp.array([-3.0, 3.0])
# Construct `Xv` via `jnp.array([[-2.0], [2.0]])`
Xv = jnp.array([[-2.0], [2.0]])
# Construct `yv` via `jnp.array([-4.0, 4.0])`
yv = jnp.array([-4.0, 4.0])

# Function `ridge(X, y, lam)` implementing this stage's computation:
def ridge(X, y, lam):
    # Return `jnp.linalg.solve(X.T @ X + X.shape[0] * lam * jnp.eye(X.shape[1]), X.T @ y)` to the caller.
    return jnp.linalg.solve(X.T @ X + X.shape[0] * lam * jnp.eye(X.shape[1]), X.T @ y)

# Function `mse(w, X, y)` implementing this stage's computation:
def mse(w, X, y):
    # Return `jnp.mean((X @ w - y) ** 2)` to the caller.
    return jnp.mean((X @ w - y) ** 2)

# Step 2 — Verify the penalized optimum: The data gradient and penalty gradient cancel at w=2.
lam = 0.5
# Run `ridge` to compute `w`.
w = ridge(X, y, lam)
# Compute `objective` from `lambda w: mse(w, X, y) + lam * jnp.dot(w, w)`
objective = lambda w: mse(w, X, y) + lam * jnp.dot(w, w)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(w, jnp.array([2.0]), atol=1e-06)
# Check numerical equivalence within tolerance: `jnp.allclose(jax.grad(objective)(w), jnp.zeros(1), atol=1e-06)`
assert jnp.allclose(jax.grad(objective)(w), jnp.zeros(1), atol=1e-06)

# Step 3 — Separate the quantities you report: Training prediction loss is 1, validation loss is 0, and the...
train = mse(w, X, y)
# Run `mse` to compute `validation`.
validation = mse(w, Xv, yv)
# Run `objective` to compute `penalized`.
penalized = objective(w)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jnp.array([train, validation, penalized]), jnp.array([1.0, 0.0, 3.0]))
# Print the observed values to compare against the expected result.
print('train / validation / objective:', train, validation, penalized)

# Figure data experiment
# Compute figure data for: Training error and validation error select different penalties
# Generate a uniform grid of points in `grid`.
grid = jnp.linspace(0.0, 2.0, 41)
# Compute `visual_data` from `{'kind': 'line', 'x': grid.tolist(), 'xlabel': 'ridg...`
visual_data = {'kind': 'line', 'x': grid.tolist(), 'xlabel': 'ridge penalty', 'ylabel': 'mean squared prediction error', 'series': [{'label': 'training', 'y': [float(mse(ridge(X, y, float(a)), X, y)) for a in grid]}, {'label': 'validation', 'y': [float(mse(ridge(X, y, float(a)), Xv, yv)) for a in grid]}]}

# Experiment: Sweep the penalty
# Experiment — Sweep the penalty: Regularization deliberately trades training fit against a...
# Construct `candidates` via `jnp.array([0.0, 0.5, 2.0])`
candidates = jnp.array([0.0, 0.5, 2.0])
# Combine or mask array elements to form `fits`.
fits = jnp.stack([ridge(X, y, float(a)) for a in candidates])
# Construct `training` via `jnp.array([mse(a, X, y) for a in fits])`
training = jnp.array([mse(a, X, y) for a in fits])
# Construct `validation` via `jnp.array([mse(a, Xv, yv) for a in fits])`
validation = jnp.array([mse(a, Xv, yv) for a in fits])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(fits[:, 0], jnp.array([3.0, 2.0, 1.0]))
# Check numerical equivalence within tolerance: `jnp.allclose(training, jnp.array([0.0, 1.0, 4.0]))`
assert jnp.allclose(training, jnp.array([0.0, 1.0, 4.0]))
# Check numerical equivalence within tolerance: `jnp.allclose(validation, jnp.array([4.0, 0.0, 4.0]))`
assert jnp.allclose(validation, jnp.array([4.0, 0.0, 4.0]))
# Assert invariant `int(jnp.argmin(training)) != int(jnp.argmin(validation))` holds
assert int(jnp.argmin(training)) != int(jnp.argmin(validation))
# Print the observed values to compare against the expected result.
print('penalty / train / validation:', candidates, training, validation)

# Experiment: Duplicate the training set
# Experiment — Duplicate the training set: Using a mean data loss keeps the penalty tradeoff unchanged...
repeat = ridge(jnp.tile(X, (2, 1)), jnp.tile(y, 2), lam)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(repeat, w, atol=1e-06)

# Reference solution. Try the exercise before reading this.
# Exercise solution: For a target slope of 4 and penalty \lambda=1, derive the fitted slope...
# Construct `y4` via `jnp.array([-4.0, 4.0])`
y4 = jnp.array([-4.0, 4.0])
# Run `ridge` to compute `w4`.
w4 = ridge(X, y4, 1.0)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(w4, jnp.array([2.0]))
# Check numerical equivalence within tolerance: `jnp.allclose(jax.grad(lambda z: mse(z, X, y4) + jnp.dot(z, z))(w4...`
assert jnp.allclose(jax.grad(lambda z: mse(z, X, y4) + jnp.dot(z, z))(w4), jnp.zeros(1), atol=1e-06)

# Reference practice: Do not accidentally shrink the intercept
# Do not accidentally shrink the intercept (Transfer / diagnosis): The mask expresses a modeling choice: shrink feature effects...
# Construct `x` via `jnp.array([-1.0, 0.0, 1.0])`
x = jnp.array([-1.0, 0.0, 1.0])
# Construct `A` via `jnp.stack([x, jnp.ones_like(x)], axis=1)`
A = jnp.stack([x, jnp.ones_like(x)], axis=1)
# Compute `target` from `jnp.full((3,), 5.0)`
target = jnp.full((3,), 5.0)
# Construct `mask` via `jnp.diag(jnp.array([1.0, 0.0]))`
mask = jnp.diag(jnp.array([1.0, 0.0]))
# Perform matrix contraction / projection to compute `correct`.
correct = jnp.linalg.solve(A.T @ A + 3 * mask, A.T @ target)
# Run `ridge` to compute `all_penalized`.
all_penalized = ridge(A, target, 1.0)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(correct, jnp.array([0.0, 5.0]))
# Check numerical equivalence within tolerance: `jnp.allclose(all_penalized, jnp.array([0.0, 2.5]))`
assert jnp.allclose(all_penalized, jnp.array([0.0, 2.5]))

# Reference practice: Catch preprocessing leakage
# Catch preprocessing leakage (Transfer / diagnosis): Validation data must not influence fitted preprocessing.
# Construct `train_x` via `jnp.array([0.0, 2.0])`
train_x = jnp.array([0.0, 2.0])
# Construct `val_x` via `jnp.array([100.0])`
val_x = jnp.array([100.0])
# Aggregate array values to compute `train_mean`.
train_mean = jnp.mean(train_x)
# Aggregate array values to compute `leaked_mean`.
leaked_mean = jnp.mean(jnp.concatenate([train_x, val_x]))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(val_x - train_mean, jnp.array([99.0]))
# Check numerical equivalence within tolerance: `not jnp.allclose(train_mean, leaked_mean)`
assert not jnp.allclose(train_mean, leaked_mean)
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.mean(train_x - train_mean), 0.0)`
assert jnp.allclose(jnp.mean(train_x - train_mean), 0.0)
print("PASS: optimization-10")
