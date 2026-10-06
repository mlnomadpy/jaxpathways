"""Your first gradient: worked experiments and reference solutions. CPU checks."""



def central_difference(function, x, h=1e-3):
    return (function(x + h) - function(x - h)) / (2 * h)

def square(x):
    return x * x

estimate = central_difference(square, 3.0)
print("Finite-difference estimate:", estimate)
assert abs(estimate - 6.0) < 1e-8

import jax
import jax.numpy as jnp
def f(x):
    return x ** 2
derivative = jax.grad(f)
print(float(derivative(3.0)))
assert jnp.allclose(derivative(3.0), 6.)
assert jnp.allclose(derivative(0.0), 0.)

# Figure data experiment
grid = jnp.linspace(-4.0, 4.0, 81)
point = 3.0
slope = derivative(point)
visual_data = {'kind': 'line', 'x': grid.tolist(), 'xlabel': 'input x', 'ylabel': 'function / tangent value', 'series': [{'label': 'f(x) = x squared', 'y': jax.vmap(f)(grid).tolist()}, {'label': 'tangent at x = 3', 'y': (f(point) + slope * (grid - point)).tolist()}]}
visual_data['markers'] = [{'x': point, 'y': float(f(point)), 'label': 'selected input'}]

# Experiment: Check the chain rule at three inputs
def composed(x):
    return (3. * x + 1.) ** 2
for point in (-1., 0., 2.):
    observed = jax.grad(composed)(point)
    expected = 6. * (3. * point + 1.)
    print("chain rule:", point, float(observed))
    assert jnp.allclose(observed, expected)

# Experiment: Choose a finite-difference scale
def cubic32(x):
    return x ** 3
point = jnp.array(3., dtype=jnp.float32)
for step in (1e-1, 1e-2, 1e-3, 1e-5, 1e-7):
    estimate = (cubic32(point + step) - cubic32(point - step)) / (2. * step)
    print("h / estimate / error:", step, float(estimate), float(jnp.abs(estimate - 27.)))
assert jnp.allclose(jax.grad(cubic32)(point), 27.)

# Reference solution. Try the exercise before reading this.
def cubic(x):
    return x ** 3
for point in (-2., 0., 3.):
    assert jnp.allclose(jax.grad(cubic)(point), 3. * point ** 2)
assert cubic(-0.1) < cubic(0.) < cubic(0.1)

# Reference practice: Differentiate the parameter you intended
def scalar_prediction(weight, x):
    return weight * x
assert jnp.allclose(jax.grad(scalar_prediction, argnums=0)(2., 3.), 3.)
assert jnp.allclose(jax.grad(scalar_prediction, argnums=1)(2., 3.), 2.)

# Reference practice: Repair an objective without hiding its meaning
def two_costs(x):
    return jnp.array([x ** 2, (x - 2.) ** 2])
def mean_cost(x):
    return jnp.mean(two_costs(x))
for point in (0., 1., 2.):
    assert jnp.allclose(jax.grad(mean_cost)(point), 2. * point - 2.)
    assert jnp.allclose(jax.grad(lambda z: jnp.sum(two_costs(z)))(point), 2. * jax.grad(mean_cost)(point))
print("PASS: first-gradient")
