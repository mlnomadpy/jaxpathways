"""Losses and value_and_grad: worked experiments and reference solutions. CPU checks."""



import jax
import jax.numpy as jnp
x = jnp.array([1., 2., 3.])
y = 2. * x
def loss(weight):
    return jnp.mean((weight * x - y) ** 2)
value, gradient = jax.value_and_grad(loss)(1.0)
print("Loss:", float(value), "gradient:", float(gradient))
assert jnp.allclose(value, 14./3.)
assert jnp.allclose(gradient, -28./3.)

# Figure data experiment
grid = jnp.linspace(0.0, 4.0, 81)
visual_data = {'kind': 'line', 'x': grid.tolist(), 'xlabel': 'weight', 'ylabel': 'loss or derivative (different quantities)', 'series': [{'label': 'mean squared loss', 'y': jax.vmap(loss)(grid).tolist()}, {'label': 'derivative', 'y': jax.vmap(jax.grad(loss))(grid).tolist()}]}

# Experiment: Check JAX against the handwritten gradient
def analytic_loss_gradient(weight):
    return 2. * jnp.mean((weight * x - y) * x)
for weight in (0., 1., 2.):
    value_here, grad_here = jax.value_and_grad(loss)(weight)
    print("weight / loss / gradient:", weight, float(value_here), float(grad_here))
    assert jnp.allclose(grad_here, analytic_loss_gradient(weight))

# Experiment: Return residuals without changing the objective
def loss_with_residuals(weight):
    residuals = weight * x - y
    return jnp.mean(residuals ** 2), residuals
(aux_value, residuals), aux_gradient = jax.value_and_grad(loss_with_residuals, has_aux=True)(1.)
assert residuals.shape == (3,)
assert jnp.allclose(residuals, jnp.array([-1., -2., -3.]))
assert jnp.allclose(aux_value, value)
assert jnp.allclose(aux_gradient, gradient)

# Reference solution. Try the exercise before reading this.
small_step = 1. - 0.1 * gradient
large_step = 1. - 0.3 * gradient
assert jnp.allclose(small_step, 29. / 15.)
assert loss(small_step) < value
assert loss(large_step) > value
assert jnp.allclose(loss(2.), 0.)

# Reference practice: Make the data boundary explicit
def explicit_loss(weight, inputs, targets):
    return jnp.mean((weight * inputs - targets) ** 2)
v_explicit, g_explicit = jax.value_and_grad(explicit_loss)(1., x, y)
assert jnp.allclose(v_explicit, 14. / 3.)
assert jnp.allclose(g_explicit, -28. / 3.)

# Reference practice: Audit reduction and batch replication
def summed_loss(weight, inputs, targets):
    return jnp.sum((weight * inputs - targets) ** 2)
g_mean = jax.grad(explicit_loss)(1., x, y)
g_sum = jax.grad(summed_loss)(1., x, y)
x_twice, y_twice = jnp.tile(x, 2), jnp.tile(y, 2)
assert jnp.allclose(g_sum, len(x) * g_mean)
assert jnp.allclose(jax.grad(explicit_loss)(1., x_twice, y_twice), g_mean)
assert jnp.allclose(jax.grad(summed_loss)(1., x_twice, y_twice), 2. * g_sum)
assert jnp.allclose(1. - 0.1 * g_mean, 1. - (0.1 / len(x)) * g_sum)
print("PASS: transforms-02")
