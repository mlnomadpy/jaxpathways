"""Your first gradient: worked experiments and reference solutions. CPU checks."""



# Numerical estimate: Approximately 6.0.
def central_difference(function, x, h=1e-3):
    # Return `(function(x + h) - function(x - h)) / (2 * h)` to the caller.
    return (function(x + h) - function(x - h)) / (2 * h)

# Function `square(x)` implementing this stage's computation:
def square(x):
    # Return `x * x` to the caller.
    return x * x

# Run `central_difference` to compute `estimate`.
estimate = central_difference(square, 3.0)
# Print the observed values to compare against the expected result.
print("Finite-difference estimate:", estimate)
# Verify contract: `abs(estimate - 6.0) < 1e-08`.
assert abs(estimate - 6.0) < 1e-8

# Your first gradient: A derivative tells us how an output changes near a particular input.
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Function `f(x)` implementing this stage's computation:
def f(x):
    # Return `x ** 2` to the caller.
    return x ** 2
# Differentiate the objective to obtain `derivative` via automatic differentiation.
derivative = jax.grad(f)
# Print the observed values to compare against the expected result.
print(float(derivative(3.0)))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(derivative(3.0), 6.)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(derivative(0.0), 0.)

# Figure data experiment
# Compute figure data for: A tangent describes local change
# Generate a uniform grid of points in `grid`.
grid = jnp.linspace(-4.0, 4.0, 81)
# Evaluate `point` from the current inputs and state.
point = 3.0
# Run `derivative` to compute `slope`.
slope = derivative(point)
# Vectorize across the batch dimension without a Python loop (`visual_data`).
visual_data = {'kind': 'line', 'x': grid.tolist(), 'xlabel': 'input x', 'ylabel': 'function / tangent value', 'series': [{'label': 'f(x) = x squared', 'y': jax.vmap(f)(grid).tolist()}, {'label': 'tangent at x = 3', 'y': (f(point) + slope * (grid - point)).tolist()}]}
# Evaluate `visual_data['markers']` from the current inputs and state.
visual_data['markers'] = [{'x': point, 'y': float(f(point)), 'label': 'selected input'}]

# Experiment: Check the chain rule at three inputs
# Experiment — Check the chain rule at three inputs: The inner derivative contributes a factor of 3.
def composed(x):
    # Return `(3.0 * x + 1.0) ** 2` to the caller.
    return (3. * x + 1.) ** 2
# Iterate over `point` to step through the computation:
for point in (-1., 0., 2.):
    # Differentiate the objective to obtain `observed` via automatic differentiation.
    observed = jax.grad(composed)(point)
    # Evaluate `expected` from the current inputs and state.
    expected = 6. * (3. * point + 1.)
    # Print the observed values to compare against the expected result.
    print("chain rule:", point, float(observed))
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
    assert jnp.allclose(observed, expected)

# Experiment: Choose a finite-difference scale
# Experiment — Choose a finite-difference scale: A central difference has truncation error at larger h and...
def cubic32(x):
    # Return `x ** 3` to the caller.
    return x ** 3
# Initialize array `point` with explicit values and shape.
point = jnp.array(3., dtype=jnp.float32)
# Iterate over `step` to step through the computation:
for step in (1e-1, 1e-2, 1e-3, 1e-5, 1e-7):
    # Evaluate `estimate` from the current inputs and state.
    estimate = (cubic32(point + step) - cubic32(point - step)) / (2. * step)
    # Print the observed values to compare against the expected result.
    print("h / estimate / error:", step, float(estimate), float(jnp.abs(estimate - 27.)))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.grad(cubic32)(point), 27.)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Foundation · Define f(x)=x^3.
def cubic(x):
    # Return `x ** 3` to the caller.
    return x ** 3
# Iterate over `point` to step through the computation:
for point in (-2., 0., 3.):
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
    assert jnp.allclose(jax.grad(cubic)(point), 3. * point ** 2)
# Verify contract: `cubic(-0.1) < cubic(0.0) < cubic(0.1)`.
assert cubic(-0.1) < cubic(0.) < cubic(0.1)

# Reference practice: Differentiate the parameter you intended
# Differentiate the parameter you intended (Practice): A parameter gradient and an input sensitivity answer...
def scalar_prediction(weight, x):
    # Return `weight * x` to the caller.
    return weight * x
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.grad(scalar_prediction, argnums=0)(2., 3.), 3.)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jax.grad(scalar_prediction, argnums=1)(2., 3.), 2.)

# Reference practice: Repair an objective without hiding its meaning
# Repair an objective without hiding its meaning (Challenge): Mean and sum agree on the minimizer here but differ by a...
def two_costs(x):
    # Return `jnp.array([x ** 2, (x - 2.0) ** 2])` to the caller.
    return jnp.array([x ** 2, (x - 2.) ** 2])
# Function `mean_cost(x)` implementing this stage's computation:
def mean_cost(x):
    # Return `jnp.mean(two_costs(x))` to the caller.
    return jnp.mean(two_costs(x))
# Iterate over `point` to step through the computation:
for point in (0., 1., 2.):
    # Verify that the numerical values match the expected reference within tolerance.
    assert jnp.allclose(jax.grad(mean_cost)(point), 2. * point - 2.)
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
    assert jnp.allclose(jax.grad(lambda z: jnp.sum(two_costs(z)))(point), 2. * jax.grad(mean_cost)(point))
print("PASS: first-gradient")
