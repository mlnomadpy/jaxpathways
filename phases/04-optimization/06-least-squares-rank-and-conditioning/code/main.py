"""Least squares, rank, and conditioning: worked experiments and reference solutions. CPU checks."""

# Construct an imperfect fit
# Step 1 — Construct an imperfect fit: No weights can match all three targets, so a nonzero residual is...
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np
# Construct `X` via `jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])`
X = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
# Construct `y` via `jnp.array([1.0, 2.0, 2.0])`
y = jnp.array([1.0, 2.0, 2.0])
# Construct `expected` via `jnp.array([2 / 3, 5 / 3])`
expected = jnp.array([2 / 3, 5 / 3])

# Solve and inspect the residual
# Step 2 — Solve and inspect the residual: Perpendicular residuals verify optimality for this unconstrained...
w, _, rank, singular = jnp.linalg.lstsq(X, y, rcond=None)
# Perform matrix contraction / projection to compute `r`.
r = X @ w - y
# Assert invariant `int(rank) == 2` holds
assert int(rank) == 2
# Assert that `jnp.allclose(w, expected, atol=2e-06)`.
assert jnp.allclose(w, expected, atol=2e-06)
# Assert that `jnp.allclose(X.T @ r, jnp.zeros(2), atol=2e-06)`.
assert jnp.allclose(X.T @ r, jnp.zeros(2), atol=2e-06)

# Compare independent references
# Step 3 — Compare independent references: The hand result and a separate numerical implementation agree...
# Assert that `jnp.allclose(jnp.mean(r * r), 1 / 9, atol=2e-06)`.
assert jnp.allclose(jnp.mean(r * r), 1 / 9, atol=2e-06)
# Convert `np_w` to a host NumPy array for inspection or verification.
np_w = np.linalg.lstsq(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), rcond=None)[0]
# Assert that `np.allclose(np.asarray(w), np_w, atol=2e-06)`.
assert np.allclose(np.asarray(w), np_w, atol=2e-06)
# Print the observed values to compare against the expected result.
print('weights / rank / MSE:', w, rank, jnp.mean(r * r))

# Step 1 — Construct an imperfect fit: No weights can match all three targets, so a nonzero residual is...
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np
# Construct `X` via `jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])`
X = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
# Construct `y` via `jnp.array([1.0, 2.0, 2.0])`
y = jnp.array([1.0, 2.0, 2.0])
# Construct `expected` via `jnp.array([2 / 3, 5 / 3])`
expected = jnp.array([2 / 3, 5 / 3])

# Step 2 — Solve and inspect the residual: Perpendicular residuals verify optimality for this unconstrained...
w, _, rank, singular = jnp.linalg.lstsq(X, y, rcond=None)
# Perform matrix contraction / projection to compute `r`.
r = X @ w - y
# Assert invariant `int(rank) == 2` holds
assert int(rank) == 2
# Assert that `jnp.allclose(w, expected, atol=2e-06)`.
assert jnp.allclose(w, expected, atol=2e-06)
# Assert that `jnp.allclose(X.T @ r, jnp.zeros(2), atol=2e-06)`.
assert jnp.allclose(X.T @ r, jnp.zeros(2), atol=2e-06)

# Step 3 — Compare independent references: The hand result and a separate numerical implementation agree...
# Assert that `jnp.allclose(jnp.mean(r * r), 1 / 9, atol=2e-06)`.
assert jnp.allclose(jnp.mean(r * r), 1 / 9, atol=2e-06)
# Convert `np_w` to a host NumPy array for inspection or verification.
np_w = np.linalg.lstsq(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), rcond=None)[0]
# Assert that `np.allclose(np.asarray(w), np_w, atol=2e-06)`.
assert np.allclose(np.asarray(w), np_w, atol=2e-06)
# Print the observed values to compare against the expected result.
print('weights / rank / MSE:', w, rank, jnp.mean(r * r))

# Figure data experiment
# Compute figure data for: Least squares balances unavoidable residuals
# Perform matrix contraction / projection to compute `visual_data`.
visual_data = {'kind': 'bar', 'labels': ['row 0', 'row 1', 'row 2'], 'ylabel': 'target / prediction', 'series': [{'label': 'target', 'y': y.tolist()}, {'label': 'least-squares fit', 'y': (X @ w).tolist()}]}

# Experiment: Duplicate a feature
# Experiment — Duplicate a feature: Fit quality and coefficient uniqueness are different questions.
# Construct `a` via `jnp.array([1.0, 2.0, 3.0])`
a = jnp.array([1.0, 2.0, 3.0])
# Combine or mask array elements to form `duplicate`.
duplicate = jnp.stack([a, a], axis=1)
# Run `jnp.linalg.lstsq` to compute `(w_dup, _, rank_dup, _)`.
w_dup, _, rank_dup, _ = jnp.linalg.lstsq(duplicate, 3 * a, rcond=None)
# Assert invariant `int(rank_dup) == 1` holds
assert int(rank_dup) == 1
# Assert that `jnp.allclose(duplicate @ w_dup, 3 * a, atol=3e-06)`.
assert jnp.allclose(duplicate @ w_dup, 3 * a, atol=3e-06)
# Assert that `jnp.allclose(duplicate @ jnp.array([1.0, 2.0]), duplicate @ jnp.array([0.0, 3.0]))`.
assert jnp.allclose(duplicate @ jnp.array([1.0, 2.0]), duplicate @ jnp.array([0.0, 3.0]))

# Experiment: Perturb a weak direction
# Experiment — Perturb a weak direction: Nearly dependent columns can amplify target perturbations while...
# Construct `near` via `jnp.array([[1.0, 1.0], [0.0, 0.01], [0.0, 0.0]])`
near = jnp.array([[1.0, 1.0], [0.0, 0.01], [0.0, 0.0]])
# Construct `base` via `near @ jnp.array([1.0, 1.0])`
base = near @ jnp.array([1.0, 1.0])
# Construct `changed` via `base + jnp.array([0.0, 0.001, 0.0])`
changed = base + jnp.array([0.0, 0.001, 0.0])
# Run `jnp.linalg.lstsq` to compute `w0`.
w0 = jnp.linalg.lstsq(near, base, rcond=None)[0]
# Run `jnp.linalg.lstsq` to compute `w1`.
w1 = jnp.linalg.lstsq(near, changed, rcond=None)[0]
# Assert that `jnp.allclose(w1 - w0, jnp.array([-0.1, 0.1]), atol=5e-05)`.
assert jnp.allclose(w1 - w0, jnp.array([-0.1, 0.1]), atol=5e-05)
# Print the observed values to compare against the expected result.
print('condition / target change / weight change:', jnp.linalg.cond(near), jnp.linalg.norm(changed - base), jnp.linalg.norm(w1 - w0))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Add a constant bias column to inputs (-1,0,1) and fit targets (-1,1,3).
# Construct `x` via `jnp.array([-1.0, 0.0, 1.0])`
x = jnp.array([-1.0, 0.0, 1.0])
# Construct `design` via `jnp.stack([x, jnp.ones_like(x)], axis=1)`
design = jnp.stack([x, jnp.ones_like(x)], axis=1)
# Run `jnp.linalg.lstsq` to compute `fit`.
fit = jnp.linalg.lstsq(design, 2 * x + 1, rcond=None)[0]
# Assert that `jnp.allclose(fit, jnp.array([2.0, 1.0]), atol=2e-06)`.
assert jnp.allclose(fit, jnp.array([2.0, 1.0]), atol=2e-06)

# Reference practice: Change feature units
# Change feature units (Transfer / diagnosis): Feature units affect coefficients and conditioning.
# Construct `scaled` via `X * jnp.array([100.0, 1.0])`
scaled = X * jnp.array([100.0, 1.0])
# Run `jnp.linalg.lstsq` to compute `ws`.
ws = jnp.linalg.lstsq(scaled, y, rcond=None)[0]
# Assert that `jnp.allclose(ws, expected / jnp.array([100.0, 1.0]), atol=3e-06)`.
assert jnp.allclose(ws, expected / jnp.array([100.0, 1.0]), atol=3e-06)
# Assert that `jnp.allclose(scaled @ ws, X @ w, atol=3e-06)`.
assert jnp.allclose(scaled @ ws, X @ w, atol=3e-06)

# Reference practice: Explain the squared conditioning
# Explain the squared conditioning (Transfer / diagnosis): The normal equations can worsen numerical sensitivity.
# Construct `D` via `jnp.diag(jnp.array([1.0, 0.1]))`
D = jnp.diag(jnp.array([1.0, 0.1]))
# Assert that `jnp.allclose(jnp.linalg.cond(D), 10.0)`.
assert jnp.allclose(jnp.linalg.cond(D), 10.0)
# Assert that `jnp.allclose(jnp.linalg.cond(D.T @ D), 100.0, rtol=2e-06)`.
assert jnp.allclose(jnp.linalg.cond(D.T @ D), 100.0, rtol=2e-06)
print("PASS: optimization-06")
