"""Losses and value_and_grad: worked experiments and reference solutions. CPU checks."""

# Step 1: Set up imports and input tensors
import jax
import jax.numpy as jnp

# Step 2: Apply the core JAX transformation
x = jnp.array([1., 2., 3.])
# Compute `y` from `2. * x`
y = 2. * x
# Function `loss(weight)` implementing this stage's computation:
def loss(weight):
    # Return `jnp.mean((weight * x - y) ** 2)` to the caller.
    return jnp.mean((weight * x - y) ** 2)

# Step 3: Verify shapes and numerical invariants
value, gradient = jax.value_and_grad(loss)(1.0)
# Print the observed values to compare against the expected result.
# Assert that `jnp.allclose(value, 14./3.)`.
assert jnp.allclose(value, 14./3.)
# Assert that `jnp.allclose(gradient, -28./3.)`.
assert jnp.allclose(gradient, -28./3.)

# Losses and value_and_grad: A loss value tells us how the current prediction is scored.
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Construct `x` via `jnp.array([1., 2., 3.])`
x = jnp.array([1., 2., 3.])
# Compute `y` from `2. * x`
y = 2. * x
# Function `loss(weight)` implementing this stage's computation:
def loss(weight):
    # Return `jnp.mean((weight * x - y) ** 2)` to the caller.
    return jnp.mean((weight * x - y) ** 2)
# Differentiate the objective to obtain `(value, gradient)` via automatic differentiation.
value, gradient = jax.value_and_grad(loss)(1.0)
# Print the observed values to compare against the expected result.
print("Loss:", float(value), "gradient:", float(gradient))
# Assert that `jnp.allclose(value, 14./3.)`.
assert jnp.allclose(value, 14./3.)
# Assert that `jnp.allclose(gradient, -28./3.)`.
assert jnp.allclose(gradient, -28./3.)

# Figure data experiment
# Compute figure data for: Loss and slope answer different questions
# Generate a uniform grid of points in `grid`.
grid = jnp.linspace(0.0, 4.0, 81)
# Differentiate the objective to obtain gradients `visual_data`.
visual_data = {'kind': 'line', 'x': grid.tolist(), 'xlabel': 'weight', 'ylabel': 'loss or derivative (different quantities)', 'series': [{'label': 'mean squared loss', 'y': jax.vmap(loss)(grid).tolist()}, {'label': 'derivative', 'y': jax.vmap(jax.grad(loss))(grid).tolist()}]}

# Experiment: Check JAX against the handwritten gradient
# Experiment — Check JAX against the handwritten gradient: A derivative describes the direction.
def analytic_loss_gradient(weight):
    # Return `2.0 * jnp.mean((weight * x - y) * x)` to the caller.
    return 2. * jnp.mean((weight * x - y) * x)
# Iterate over `weight` to step through the computation:
for weight in (0., 1., 2.):
    # Differentiate the objective to obtain `(value_here, grad_here)` via automatic differentiation.
    value_here, grad_here = jax.value_and_grad(loss)(weight)
    # Print the observed values to compare against the expected result.
    print("weight / loss / gradient:", weight, float(value_here), float(grad_here))
    # Assert that `jnp.allclose(grad_here, analytic_loss_gradient(weight))`.
    assert jnp.allclose(grad_here, analytic_loss_gradient(weight))

# Experiment: Return residuals without changing the objective
# Experiment — Return residuals without changing the objective: The return structure separates logging data from a scalar...
def loss_with_residuals(weight):
    # Compute `residuals` from `weight * x - y`
    residuals = weight * x - y
    # Return `(jnp.mean(residuals ** 2), residuals)` to the caller.
    return jnp.mean(residuals ** 2), residuals
# Differentiate the objective to obtain `((aux_value, residuals), aux_grad...` via automatic differentiation.
(aux_value, residuals), aux_gradient = jax.value_and_grad(loss_with_residuals, has_aux=True)(1.)
# Check tensor shape invariant: `residuals.shape == (3,)`
assert residuals.shape == (3,)
# Assert that `jnp.allclose(residuals, jnp.array([-1., -2., -3.]))`.
assert jnp.allclose(residuals, jnp.array([-1., -2., -3.]))
# Assert that `jnp.allclose(aux_value, value)`.
assert jnp.allclose(aux_value, value)
# Assert that `jnp.allclose(aux_gradient, gradient)`.
assert jnp.allclose(aux_gradient, gradient)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Foundation · Starting from w=1, calculate the next weight and loss for...
small_step = 1. - 0.1 * gradient
# Compute `large_step` from `1. - 0.3 * gradient`
large_step = 1. - 0.3 * gradient
# Assert that `jnp.allclose(small_step, 29. / 15.)`.
assert jnp.allclose(small_step, 29. / 15.)
# Assert invariant `loss(small_step) < value` holds
assert loss(small_step) < value
# Assert invariant `loss(large_step) > value` holds
assert loss(large_step) > value
# Assert that `jnp.allclose(loss(2.), 0.)`.
assert jnp.allclose(loss(2.), 0.)

# Reference practice: Make the data boundary explicit
# Make the data boundary explicit (Practice): Keeping data explicit prepares the function for new batches...
def explicit_loss(weight, inputs, targets):
    # Return `jnp.mean((weight * inputs - targets) ** 2)` to the caller.
    return jnp.mean((weight * inputs - targets) ** 2)
# Differentiate the objective to obtain `(v_explicit, g_explicit)` via automatic differentiation.
v_explicit, g_explicit = jax.value_and_grad(explicit_loss)(1., x, y)
# Assert that `jnp.allclose(v_explicit, 14. / 3.)`.
assert jnp.allclose(v_explicit, 14. / 3.)
# Assert that `jnp.allclose(g_explicit, -28. / 3.)`.
assert jnp.allclose(g_explicit, -28. / 3.)

# Reference practice: Audit reduction and batch replication
# Audit reduction and batch replication (Challenge): Mean gradients stay fixed under exact batch replication; sum...
def summed_loss(weight, inputs, targets):
    # Return `jnp.sum((weight * inputs - targets) ** 2)` to the caller.
    return jnp.sum((weight * inputs - targets) ** 2)
# Differentiate the objective to obtain `g_mean` via automatic differentiation.
g_mean = jax.grad(explicit_loss)(1., x, y)
# Differentiate the objective to obtain `g_sum` via automatic differentiation.
g_sum = jax.grad(summed_loss)(1., x, y)
# Compute `x_twice, y_twice` from `jnp.tile(x, 2), jnp.tile(y, 2)`
x_twice, y_twice = jnp.tile(x, 2), jnp.tile(y, 2)
# Assert that `jnp.allclose(g_sum, len(x) * g_mean)`.
assert jnp.allclose(g_sum, len(x) * g_mean)
# Assert that `jnp.allclose(jax.grad(explicit_loss)(1., x_twice, y_twice), g_mean)`.
assert jnp.allclose(jax.grad(explicit_loss)(1., x_twice, y_twice), g_mean)
# Assert that `jnp.allclose(jax.grad(summed_loss)(1., x_twice, y_twice), 2. * g_sum)`.
assert jnp.allclose(jax.grad(summed_loss)(1., x_twice, y_twice), 2. * g_sum)
# Assert that `jnp.allclose(1. - 0.1 * g_mean, 1. - (0.1 / len(x)) * g_sum)`.
assert jnp.allclose(1. - 0.1 * g_mean, 1. - (0.1 / len(x)) * g_sum)
print("PASS: transforms-02")
