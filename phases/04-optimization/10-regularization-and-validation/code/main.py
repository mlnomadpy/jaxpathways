"""Regularization and honest validation: worked experiments and reference solutions. CPU checks."""

# Create separate training and validation examples
import jax
import jax.numpy as jnp
X = jnp.array([[-1.0], [1.0]])
y = jnp.array([-3.0, 3.0])
Xv = jnp.array([[-2.0], [2.0]])
yv = jnp.array([-4.0, 4.0])

def ridge(X, y, lam):
    return jnp.linalg.solve(X.T @ X + X.shape[0] * lam * jnp.eye(X.shape[1]), X.T @ y)

def mse(w, X, y):
    return jnp.mean((X @ w - y) ** 2)

# Verify the penalized optimum
lam = 0.5
w = ridge(X, y, lam)
objective = lambda w: mse(w, X, y) + lam * jnp.dot(w, w)
assert jnp.allclose(w, jnp.array([2.0]), atol=1e-06)
assert jnp.allclose(jax.grad(objective)(w), jnp.zeros(1), atol=1e-06)

# Separate the quantities you report
train = mse(w, X, y)
validation = mse(w, Xv, yv)
penalized = objective(w)
assert jnp.allclose(jnp.array([train, validation, penalized]), jnp.array([1.0, 0.0, 3.0]))
print('train / validation / objective:', train, validation, penalized)

import jax
import jax.numpy as jnp
X = jnp.array([[-1.0], [1.0]])
y = jnp.array([-3.0, 3.0])
Xv = jnp.array([[-2.0], [2.0]])
yv = jnp.array([-4.0, 4.0])

def ridge(X, y, lam):
    return jnp.linalg.solve(X.T @ X + X.shape[0] * lam * jnp.eye(X.shape[1]), X.T @ y)

def mse(w, X, y):
    return jnp.mean((X @ w - y) ** 2)

lam = 0.5
w = ridge(X, y, lam)
objective = lambda w: mse(w, X, y) + lam * jnp.dot(w, w)
assert jnp.allclose(w, jnp.array([2.0]), atol=1e-06)
assert jnp.allclose(jax.grad(objective)(w), jnp.zeros(1), atol=1e-06)

train = mse(w, X, y)
validation = mse(w, Xv, yv)
penalized = objective(w)
assert jnp.allclose(jnp.array([train, validation, penalized]), jnp.array([1.0, 0.0, 3.0]))
print('train / validation / objective:', train, validation, penalized)

# Figure data experiment
grid = jnp.linspace(0.0, 2.0, 41)
visual_data = {'kind': 'line', 'x': grid.tolist(), 'xlabel': 'ridge penalty', 'ylabel': 'mean squared prediction error', 'series': [{'label': 'training', 'y': [float(mse(ridge(X, y, float(a)), X, y)) for a in grid]}, {'label': 'validation', 'y': [float(mse(ridge(X, y, float(a)), Xv, yv)) for a in grid]}]}

# Experiment: Sweep the penalty
candidates = jnp.array([0.0, 0.5, 2.0])
fits = jnp.stack([ridge(X, y, float(a)) for a in candidates])
training = jnp.array([mse(a, X, y) for a in fits])
validation = jnp.array([mse(a, Xv, yv) for a in fits])
assert jnp.allclose(fits[:, 0], jnp.array([3.0, 2.0, 1.0]))
assert jnp.allclose(training, jnp.array([0.0, 1.0, 4.0]))
assert jnp.allclose(validation, jnp.array([4.0, 0.0, 4.0]))
assert int(jnp.argmin(training)) != int(jnp.argmin(validation))
print('penalty / train / validation:', candidates, training, validation)

# Experiment: Duplicate the training set
repeat = ridge(jnp.tile(X, (2, 1)), jnp.tile(y, 2), lam)
assert jnp.allclose(repeat, w, atol=1e-06)

# Reference solution. Try the exercise before reading this.
y4 = jnp.array([-4.0, 4.0])
w4 = ridge(X, y4, 1.0)
assert jnp.allclose(w4, jnp.array([2.0]))
assert jnp.allclose(jax.grad(lambda z: mse(z, X, y4) + jnp.dot(z, z))(w4), jnp.zeros(1), atol=1e-06)

# Reference practice: Do not accidentally shrink the intercept
x = jnp.array([-1.0, 0.0, 1.0])
A = jnp.stack([x, jnp.ones_like(x)], axis=1)
target = jnp.full((3,), 5.0)
mask = jnp.diag(jnp.array([1.0, 0.0]))
correct = jnp.linalg.solve(A.T @ A + 3 * mask, A.T @ target)
all_penalized = ridge(A, target, 1.0)
assert jnp.allclose(correct, jnp.array([0.0, 5.0]))
assert jnp.allclose(all_penalized, jnp.array([0.0, 2.5]))

# Reference practice: Catch preprocessing leakage
train_x = jnp.array([0.0, 2.0])
val_x = jnp.array([100.0])
train_mean = jnp.mean(train_x)
leaked_mean = jnp.mean(jnp.concatenate([train_x, val_x]))
assert jnp.allclose(val_x - train_mean, jnp.array([99.0]))
assert not jnp.allclose(train_mean, leaked_mean)
assert jnp.allclose(jnp.mean(train_x - train_mean), 0.0)
print("PASS: optimization-10")
