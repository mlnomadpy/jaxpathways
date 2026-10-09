"""Probability, likelihood, and stable losses: worked experiments and reference solutions. CPU checks."""

# Write a loss from logits
# Step 1 — Write a loss from logits: A balanced probability assigns loss \log 2 to either outcome.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import optax

# Function `binary_loss(z, y)` implementing this stage's computation:
def binary_loss(z, y):
    # Return `jax.nn.softplus(z) - y * z` to the caller.
    return jax.nn.softplus(z) - y * z
# Construct `z` via `jnp.array(0.0)`
z = jnp.array(0.0)
# Construct `y` via `jnp.array(1.0)`
y = jnp.array(1.0)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(binary_loss(z, y), jnp.log(2.0))

# Check the derivative independently
# Step 2 — Check the derivative independently: The derivative points toward assigning more probability to the...
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.grad(binary_loss)(z, y), -0.5)
# Construct `logits` via `jnp.array([-2.0, 0.0, 2.0])`
logits = jnp.array([-2.0, 0.0, 2.0])
# Construct `labels` via `jnp.array([0.0, 1.0, 1.0])`
labels = jnp.array([0.0, 1.0, 1.0])
# Run `jax.nn.sigmoid` to compute `analytic`.
analytic = jax.nn.sigmoid(logits) - labels
# Differentiate the objective to obtain `actual` via automatic differentiation.
actual = jax.vmap(jax.grad(binary_loss))(logits, labels)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(actual, analytic, atol=1e-06)
# Check numerical equivalence within tolerance: `jnp.allclose(binary_loss(logits, labels), optax.sigmoid_binary_cr...`
assert jnp.allclose(binary_loss(logits, labels), optax.sigmoid_binary_cross_entropy(logits, labels), atol=1e-06)

# Check a confident mistake
# Step 3 — Check a confident mistake: The confident mistake has a large finite loss and derivative near -1.
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(binary_loss(jnp.array(-100.0), jnp.array(1.0)), 100.0)
# Check numerical equivalence within tolerance: `jnp.allclose(jax.grad(binary_loss)(jnp.array(-100.0), jnp.array(1...`
assert jnp.allclose(jax.grad(binary_loss)(jnp.array(-100.0), jnp.array(1.0)), -1.0)
# Print the observed values to compare against the expected result.
print('uncertain loss / confident-wrong loss:', binary_loss(z, y), binary_loss(-100.0, 1.0))

# Step 1 — Write a loss from logits: A balanced probability assigns loss \log 2 to either outcome.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import optax

# Function `binary_loss(z, y)` implementing this stage's computation:
def binary_loss(z, y):
    # Return `jax.nn.softplus(z) - y * z` to the caller.
    return jax.nn.softplus(z) - y * z
# Construct `z` via `jnp.array(0.0)`
z = jnp.array(0.0)
# Construct `y` via `jnp.array(1.0)`
y = jnp.array(1.0)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(binary_loss(z, y), jnp.log(2.0))

# Step 2 — Check the derivative independently: The derivative points toward assigning more probability to the...
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.grad(binary_loss)(z, y), -0.5)
# Construct `logits` via `jnp.array([-2.0, 0.0, 2.0])`
logits = jnp.array([-2.0, 0.0, 2.0])
# Construct `labels` via `jnp.array([0.0, 1.0, 1.0])`
labels = jnp.array([0.0, 1.0, 1.0])
# Run `jax.nn.sigmoid` to compute `analytic`.
analytic = jax.nn.sigmoid(logits) - labels
# Differentiate the objective to obtain `actual` via automatic differentiation.
actual = jax.vmap(jax.grad(binary_loss))(logits, labels)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(actual, analytic, atol=1e-06)
# Check numerical equivalence within tolerance: `jnp.allclose(binary_loss(logits, labels), optax.sigmoid_binary_cr...`
assert jnp.allclose(binary_loss(logits, labels), optax.sigmoid_binary_cross_entropy(logits, labels), atol=1e-06)

# Step 3 — Check a confident mistake: The confident mistake has a large finite loss and derivative near -1.
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(binary_loss(jnp.array(-100.0), jnp.array(1.0)), 100.0)
# Check numerical equivalence within tolerance: `jnp.allclose(jax.grad(binary_loss)(jnp.array(-100.0), jnp.array(1...`
assert jnp.allclose(jax.grad(binary_loss)(jnp.array(-100.0), jnp.array(1.0)), -1.0)
# Print the observed values to compare against the expected result.
print('uncertain loss / confident-wrong loss:', binary_loss(z, y), binary_loss(-100.0, 1.0))

# Figure data experiment
# Compute figure data for: Confidence is rewarded only when it matches the label
# Generate a uniform grid of points in `grid`.
grid = jnp.linspace(-10.0, 10.0, 81)
# Compute `visual_data` from `{'kind': 'line', 'x': grid.tolist(), 'xlabel': 'logi...`
visual_data = {'kind': 'line', 'x': grid.tolist(), 'xlabel': 'logit', 'ylabel': 'binary cross-entropy', 'series': [{'label': 'observed label 1', 'y': binary_loss(grid, 1.0).tolist()}, {'label': 'observed label 0', 'y': binary_loss(grid, 0.0).tolist()}]}

# Experiment: Break the probability-first implementation
# Experiment — Break the probability-first implementation: A stable final answer requires stable intermediate computations.
# Construct `p` via `jax.nn.sigmoid(jnp.array(100.0))`
p = jax.nn.sigmoid(jnp.array(100.0))
# Compute `naive` from `-jnp.log(1 - p)`
naive = -jnp.log(1 - p)
# Construct `stable` via `binary_loss(jnp.array(100.0), jnp.array(0.0))`
stable = binary_loss(jnp.array(100.0), jnp.array(0.0))
# Confirm that all computed values remain finite (no NaN or Inf).
assert not jnp.isfinite(naive)
# Check numerical equivalence within tolerance: `jnp.isfinite(stable) and jnp.allclose(stable, 100.0)`
assert jnp.isfinite(stable) and jnp.allclose(stable, 100.0)

# Experiment: Shift every class score
# Experiment — Shift every class score: Only relative class scores matter.
# Construct `scores` via `jnp.array([1.0, 2.0, 3.0])`
scores = jnp.array([1.0, 2.0, 3.0])
# Compute `target` from `2`
target = 2
# Evaluate numerically stable log-space cross-entropy/likelihood (`nll`).
nll = -jax.nn.log_softmax(scores)[target]
# Evaluate numerically stable log-space cross-entropy/likelihood (`shifted`).
shifted = -jax.nn.log_softmax(scores + 1000.0)[target]
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(nll, shifted, atol=1e-06)
# Run `jnp.log` to compute `reference_loss`.
reference_loss = jnp.log(jnp.exp(-2.0) + jnp.exp(-1.0) + 1.0)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(nll, reference_loss, atol=1e-06)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Compute the binary losses for positive labels at probabilities 0.8 and...
# Construct `probabilities` via `jnp.array([0.8, 0.2])`
probabilities = jnp.array([0.8, 0.2])
# Run `jnp.log` to compute `z_from_p`.
z_from_p = jnp.log(probabilities) - jnp.log1p(-probabilities)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(binary_loss(z_from_p, 1.0), -jnp.log(probabilities), atol=1e-06)
# Assert invariant `binary_loss(z_from_p[0], 1.0) < binary_loss(z_from_p[1], 1.0)` holds
assert binary_loss(z_from_p[0], 1.0) < binary_loss(z_from_p[1], 1.0)

# Reference practice: Fit a constant probability
# Fit a constant probability (Transfer / diagnosis): The constant model matches the observed class frequency.
# Construct `ys` via `jnp.array([1.0, 1.0, 1.0, 0.0])`
ys = jnp.array([1.0, 1.0, 1.0, 0.0])
# Aggregate array values to compute `constant_loss`.
constant_loss = lambda z: jnp.mean(binary_loss(z, ys))
# Run `jnp.log` to compute `optimum`.
optimum = jnp.log(3.0)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.grad(constant_loss)(optimum), 0.0, atol=1e-06)
# Assert invariant `constant_loss(optimum) < constant_loss(jnp.array(0.0))` holds
assert constant_loss(optimum) < constant_loss(jnp.array(0.0))

# Reference practice: Catch a reduction mistake
# Catch a reduction mistake (Transfer / diagnosis): Reduction conventions affect gradient scale and...
# Construct `zs` via `jnp.array([-1.0, 2.0])`
zs = jnp.array([-1.0, 2.0])
# Construct `ys2` via `jnp.array([0.0, 1.0])`
ys2 = jnp.array([0.0, 1.0])
# Run `binary_loss` to compute `original`.
original = binary_loss(zs, ys2)
# Run `binary_loss` to compute `doubled`.
doubled = binary_loss(jnp.tile(zs, 2), jnp.tile(ys2, 2))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jnp.mean(original), jnp.mean(doubled))
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.sum(doubled), 2 * jnp.sum(original))`
assert jnp.allclose(jnp.sum(doubled), 2 * jnp.sum(original))
# Differentiate the objective to obtain `mean_grad` via automatic differentiation.
mean_grad = lambda z: jax.grad(lambda t: jnp.mean(binary_loss(t * z, ys2)))(1.0)
# Differentiate the objective to obtain `duplicated_grad` via automatic differentiation.
duplicated_grad = jax.grad(lambda t: jnp.mean(binary_loss(t * jnp.tile(zs, 2), jnp.tile(ys2, 2))))(1.0)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(mean_grad(zs), duplicated_grad)
print("PASS: optimization-09")
