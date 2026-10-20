"""Your first gradient: worked experiments and reference solutions. CPU checks."""

# Step 1: Set up imports and input tensors
import jax
import jax.numpy as jnp

# Step 2: Apply the core JAX transformation
def f(x):
    return x ** 2
# Differentiate the objective to obtain `derivative` via automatic differentiation.
derivative = jax.grad(f)

# Step 3: Verify shapes and numerical invariants
assert jnp.allclose(derivative(3.0), 6.)
assert jnp.allclose(derivative(0.0), 0.)

# Numerical estimate: Approximately 6.0.
def central_difference(function, x, h=1e-3):
    return (function(x + h) - function(x - h)) / (2 * h)

def square(x):
    return x * x

estimate = central_difference(square, 3.0)
print("Finite-difference estimate:", estimate)
assert abs(estimate - 6.0) < 1e-8

# Your first gradient: A derivative tells us how an output changes near a particular input.
import jax
import jax.numpy as jnp
def f(x):
    return x ** 2
# Differentiate the objective to obtain `derivative` via automatic differentiation.
derivative = jax.grad(f)
print(float(derivative(3.0)))
assert jnp.allclose(derivative(3.0), 6.)
assert jnp.allclose(derivative(0.0), 0.)

# Figure data experiment
# Compute figure data for: A tangent describes local change
# Generate a uniform grid of points in `grid`.
grid = jnp.linspace(-4.0, 4.0, 81)
point = 3.0
slope = derivative(point)
# Vectorize across the batch dimension without a Python loop (`visual_data`).
visual_data = {'kind': 'line', 'x': grid.tolist(), 'xlabel': 'input x', 'ylabel': 'function / tangent value', 'series': [{'label': 'f(x) = x squared', 'y': jax.vmap(f)(grid).tolist()}, {'label': 'tangent at x = 3', 'y': (f(point) + slope * (grid - point)).tolist()}]}
visual_data['markers'] = [{'x': point, 'y': float(f(point)), 'label': 'selected input'}]

# Experiment: Check the chain rule at three inputs
# Experiment — Check the chain rule at three inputs: The inner derivative contributes a factor of 3.
def composed(x):
    return (3. * x + 1.) ** 2
for point in (-1., 0., 2.):
    # Differentiate the objective to obtain `observed` via automatic differentiation.
    observed = jax.grad(composed)(point)
    expected = 6. * (3. * point + 1.)
    print("chain rule:", point, float(observed))
    assert jnp.allclose(observed, expected)

# Experiment: Choose a finite-difference scale
# Experiment — Choose a finite-difference scale: A central difference has truncation error at larger h and...
def cubic32(x):
    return x ** 3
point = jnp.array(3., dtype=jnp.float32)
for step in (1e-1, 1e-2, 1e-3, 1e-5, 1e-7):
    estimate = (cubic32(point + step) - cubic32(point - step)) / (2. * step)
    print("h / estimate / error:", step, float(estimate), float(jnp.abs(estimate - 27.)))
assert jnp.allclose(jax.grad(cubic32)(point), 27.)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Foundation · Define f(x)=x^3.
def cubic(x):
    return x ** 3
for point in (-2., 0., 3.):
    assert jnp.allclose(jax.grad(cubic)(point), 3. * point ** 2)
assert cubic(-0.1) < cubic(0.) < cubic(0.1)

# Reference practice: Differentiate the parameter you intended
# Differentiate the parameter you intended (Practice): A parameter gradient and an input sensitivity answer...
def scalar_prediction(weight, x):
    return weight * x
assert jnp.allclose(jax.grad(scalar_prediction, argnums=0)(2., 3.), 3.)
assert jnp.allclose(jax.grad(scalar_prediction, argnums=1)(2., 3.), 2.)

# Reference practice: Repair an objective without hiding its meaning
# Repair an objective without hiding its meaning (Challenge): Mean and sum agree on the minimizer here but differ by a...
def two_costs(x):
    return jnp.array([x ** 2, (x - 2.) ** 2])
def mean_cost(x):
    return jnp.mean(two_costs(x))
for point in (0., 1., 2.):
    assert jnp.allclose(jax.grad(mean_cost)(point), 2. * point - 2.)
    assert jnp.allclose(jax.grad(lambda z: jnp.sum(two_costs(z)))(point), 2. * jax.grad(mean_cost)(point))
print("PASS: first-gradient")
