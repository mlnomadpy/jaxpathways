"""Curvature, Hessians, and learning rates: worked experiments and reference solutions. CPU checks."""

# Write the bowl
# Step 1 — Write the bowl: At the starting point the gradient is (1,10).
# Import jax for this computation.
import jax
import jax.numpy as jnp

# Function `loss(w)` implementing this stage's computation:
def loss(w):
    # Return `0.5 * (w[0] ** 2 + 10 * w[1] ** 2)` to the caller.
    return 0.5 * (w[0] ** 2 + 10 * w[1] ** 2)
# Construct `w` via `jnp.array([1.0, 1.0])`
w = jnp.array([1.0, 1.0])
# Construct `H` via `jnp.diag(jnp.array([1.0, 10.0]))`
H = jnp.diag(jnp.array([1.0, 10.0]))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.grad(loss)(w), H @ w)

# Check second derivatives
# Step 2 — Check second derivatives: The product agrees with the diagonal matrix without constructing a...
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.hessian(loss)(w), H)
# Construct `v` via `jnp.array([2.0, -1.0])`
v = jnp.array([2.0, -1.0])
# Differentiate the objective to obtain `hv` via automatic differentiation.
hv = jax.jvp(jax.grad(loss), (w,), (v,))[1]
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(hv, jnp.array([2.0, -10.0]))

# Run an explicit trajectory
# Step 3 — Run an explicit trajectory: The second coordinate vanishes; the first remains near 0.1216...
# Define `run(rate, steps)` to evaluate the objective and its automatic derivatives:
def run(rate, steps=20):

    # Define `step(w, _)` to evaluate the objective and its automatic derivatives:
    def step(w, _):
        # Differentiate the objective to obtain `w` via automatic differentiation.
        w = w - rate * jax.grad(loss)(w)
        # Return `(w, loss(w))` to the caller.
        return (w, loss(w))
    # Return `jax.lax.scan(step, jnp.array([1.0, 1.0]), None, length=steps)` to the caller.
    return jax.lax.scan(step, jnp.array([1.0, 1.0]), None, length=steps)
# Run `run` to compute `(final, history)`.
final, history = run(0.1)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(final, jnp.array([0.9 ** 20, 0.0]), atol=2e-06)
# Assert invariant `jnp.all(jnp.diff(history) <= 1e-06)` holds
assert jnp.all(jnp.diff(history) <= 1e-06)
# Print the observed values to compare against the expected result.
print('final weights / loss:', final, history[-1])

# Step 1 — Write the bowl: At the starting point the gradient is (1,10).
# Import jax for this computation.
import jax
import jax.numpy as jnp

# Function `loss(w)` implementing this stage's computation:
def loss(w):
    # Return `0.5 * (w[0] ** 2 + 10 * w[1] ** 2)` to the caller.
    return 0.5 * (w[0] ** 2 + 10 * w[1] ** 2)
# Construct `w` via `jnp.array([1.0, 1.0])`
w = jnp.array([1.0, 1.0])
# Construct `H` via `jnp.diag(jnp.array([1.0, 10.0]))`
H = jnp.diag(jnp.array([1.0, 10.0]))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.grad(loss)(w), H @ w)

# Step 2 — Check second derivatives: The product agrees with the diagonal matrix without constructing a...
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.hessian(loss)(w), H)
# Construct `v` via `jnp.array([2.0, -1.0])`
v = jnp.array([2.0, -1.0])
# Differentiate the objective to obtain `hv` via automatic differentiation.
hv = jax.jvp(jax.grad(loss), (w,), (v,))[1]
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(hv, jnp.array([2.0, -10.0]))

# Step 3 — Run an explicit trajectory: The second coordinate vanishes; the first remains near 0.1216...
# Define `run(rate, steps)` to evaluate the objective and its automatic derivatives:
def run(rate, steps=20):

    # Define `step(w, _)` to evaluate the objective and its automatic derivatives:
    def step(w, _):
        # Differentiate the objective to obtain `w` via automatic differentiation.
        w = w - rate * jax.grad(loss)(w)
        # Return `(w, loss(w))` to the caller.
        return (w, loss(w))
    # Return `jax.lax.scan(step, jnp.array([1.0, 1.0]), None, length=steps)` to the caller.
    return jax.lax.scan(step, jnp.array([1.0, 1.0]), None, length=steps)
# Run `run` to compute `(final, history)`.
final, history = run(0.1)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(final, jnp.array([0.9 ** 20, 0.0]), atol=2e-06)
# Assert invariant `jnp.all(jnp.diff(history) <= 1e-06)` holds
assert jnp.all(jnp.diff(history) <= 1e-06)
# Print the observed values to compare against the expected result.
print('final weights / loss:', final, history[-1])

# Figure data experiment
# Compute figure data for: Unequal curvature produces unequal progress
# Create device-backed JAX array `points`.
points = [jnp.array([1.0, 1.0])]
# Repeat the update loop over `range(20)` steps:
for _ in range(20):
    # Differentiate the objective to obtain gradients ``.
    points.append(points[-1] - 0.1 * jax.grad(loss)(points[-1]))
# Combine or mask array elements to form `trace`.
trace = jnp.stack(points)
# Compute `visual_data` from `{'kind': 'line', 'x': list(range(21)), 'xlabel': 'co...`
visual_data = {'kind': 'line', 'x': list(range(21)), 'xlabel': 'completed update', 'ylabel': 'coordinate value', 'series': [{'label': 'curvature 1', 'y': trace[:, 0].tolist()}, {'label': 'curvature 10', 'y': trace[:, 1].tolist()}]}

# Experiment: Cross the stability boundary
# Experiment — Cross the stability boundary: A learning-rate bound depends on curvature, not on a universal...
edge, _ = run(0.2)
# Run `run` to compute `(bad, bad_history)`.
bad, bad_history = run(0.21)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jnp.abs(edge[1]), 1.0, atol=2e-06)
# Check numerical equivalence within tolerance: `jnp.abs(bad[1]) > 6.0`
assert jnp.abs(bad[1]) > 6.0
# Assert invariant `bad_history[-1] > loss(w)` holds
assert bad_history[-1] > loss(w)

# Experiment: Change coordinates
# Experiment — Change coordinates: Coordinates can change optimization geometry while representing...
# Construct `z` via `jnp.array([w[0], jnp.sqrt(10.0) * w[1]])`
z = jnp.array([w[0], jnp.sqrt(10.0) * w[1]])
# Compute `round_loss` from `lambda z: 0.5 * jnp.dot(z, z)`
round_loss = lambda z: 0.5 * jnp.dot(z, z)
# Differentiate the objective to obtain `z_next` via automatic differentiation.
z_next = z - jax.grad(round_loss)(z)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(z_next, jnp.zeros(2))
# Check numerical equivalence within tolerance: `jnp.allclose(round_loss(z), loss(w))`
assert jnp.allclose(round_loss(z), loss(w))

# Reference solution. Try the exercise before reading this.
# Exercise solution: For curvatures (2,8), derive the stable interval and check one step at...
# Construct `H2` via `jnp.diag(jnp.array([2.0, 8.0]))`
H2 = jnp.diag(jnp.array([2.0, 8.0]))
# Perform matrix contraction / projection to compute `w_next`.
w_next = w - 0.1 * (H2 @ w)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(w_next, jnp.array([0.8, 0.2]), atol=1e-06)
# Check numerical equivalence within tolerance: `jnp.allclose(2 / jnp.max(jnp.linalg.eigvalsh(H2)), 0.25)`
assert jnp.allclose(2 / jnp.max(jnp.linalg.eigvalsh(H2)), 0.25)

# Reference practice: Reject the zero-gradient shortcut
# Reject the zero-gradient shortcut (Transfer / diagnosis): Stationarity is a first-order condition, not a proof of a...
saddle = lambda z: z[0] ** 2 - z[1] ** 2
# Construct `zero` via `jnp.zeros(2)`
zero = jnp.zeros(2)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.grad(saddle)(zero), zero)
# Assert invariant `saddle(jnp.array([0.0, 0.1])) < saddle(zero)` holds
assert saddle(jnp.array([0.0, 0.1])) < saddle(zero)
# Assert invariant `saddle(jnp.array([0.1, 0.0])) > saddle(zero)` holds
assert saddle(jnp.array([0.1, 0.0])) > saddle(zero)
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.linalg.eigvalsh(jax.hessian(saddle)(zero)), jnp....`
assert jnp.allclose(jnp.linalg.eigvalsh(jax.hessian(saddle)(zero)), jnp.array([-2.0, 2.0]))

# Reference practice: Transfer to a nonquadratic function
# Transfer to a nonquadratic function (Transfer / diagnosis): For a nonquadratic function the Hessian changes with the...
# Construct `point` via `jnp.array([0.0, jnp.log(2.0)])`
point = jnp.array([0.0, jnp.log(2.0)])
# Construct `direction` via `jnp.array([1.0, 3.0])`
direction = jnp.array([1.0, 3.0])
# Differentiate the objective to obtain `product` via automatic differentiation.
product = jax.jvp(jax.grad(lambda z: jnp.sum(jnp.exp(z))), (point,), (direction,))[1]
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(product, jnp.array([1.0, 6.0]), atol=1e-06)
print("PASS: optimization-08")
