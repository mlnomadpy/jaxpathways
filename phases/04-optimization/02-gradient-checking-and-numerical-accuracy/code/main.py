"""Gradient checking and numerical accuracy: worked experiments and reference solutions. CPU checks."""

# 1. Prepare the inputs
# Step 1 — 1. Prepare the inputs: This block establishes the values used by the following steps; run...
# Import jax for this computation.
import jax
import jax.numpy as jnp

# 2. Define the computation
# Step 2 — 2. Define the computation: Each row of basis perturbs one coordinate.
def objective(w):
    # Return `jnp.sum((w - jnp.array([2.0, -1.0])) ** 2)` to the caller.
    return jnp.sum((w - jnp.array([2., -1.])) ** 2)
# Initialize array `w` with explicit values and shape.
w = jnp.array([0.5, 0.25], dtype=jnp.float32)
# Evaluate `h` from the current inputs and state.
h = 1e-2
# Initialize array `basis` with explicit values and shape.
basis = jnp.eye(2)
# Vectorize across the batch dimension with `jax.vmap` (`finite`).
finite = jax.vmap(lambda e: (objective(w+h*e)-objective(w-h*e))/(2*h))(basis)
# Differentiate the objective to obtain `automatic` via automatic differentiation.
automatic = jax.grad(objective)(w)
# Initialize array `analytic` with explicit values and shape.
analytic = 2 * (w - jnp.array([2., -1.]))

# 3. Measure and verify
# Step 3 — 3. Measure and verify: Both gradients are approximately [-3., 2.5].
# Print the observed values to compare against the expected result.
print("Finite difference:", finite)
# Print diagnostic summary of the computed outputs.
print("Autodiff:", automatic)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(automatic, analytic)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(finite, automatic, atol=2e-4, rtol=2e-4)

# Step 1 — 1. Prepare the inputs: This block establishes the values used by the following steps; run...
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Step 2 — 2. Define the computation: Each row of basis perturbs one coordinate.
def objective(w):
    # Return `jnp.sum((w - jnp.array([2.0, -1.0])) ** 2)` to the caller.
    return jnp.sum((w - jnp.array([2., -1.])) ** 2)
# Initialize array `w` with explicit values and shape.
w = jnp.array([0.5, 0.25], dtype=jnp.float32)
# Evaluate `h` from the current inputs and state.
h = 1e-2
# Initialize array `basis` with explicit values and shape.
basis = jnp.eye(2)
# Vectorize across the batch dimension with `jax.vmap` (`finite`).
finite = jax.vmap(lambda e: (objective(w+h*e)-objective(w-h*e))/(2*h))(basis)
# Differentiate the objective to obtain `automatic` via automatic differentiation.
automatic = jax.grad(objective)(w)
# Initialize array `analytic` with explicit values and shape.
analytic = 2 * (w - jnp.array([2., -1.]))
# Step 3 — 3. Measure and verify: Both gradients are approximately [-3., 2.5].
# Print the observed values to compare against the expected result.
print("Finite difference:", finite)
# Print diagnostic summary of the computed outputs.
print("Autodiff:", automatic)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(automatic, analytic)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(finite, automatic, atol=2e-4, rtol=2e-4)

# Figure data experiment
# Compute figure data for: Smaller perturbations eventually lose accuracy
# Run `jnp.logspace` to compute `steps`.
steps = jnp.logspace(-8.0, -1.0, 22)
# Evaluate `errors` from the current inputs and state.
errors = []
# Loop over `h_plot` in `steps`:
for h_plot in steps:
    # Vectorize across the batch dimension without a Python loop (`estimate`).
    estimate = jax.vmap(lambda e: (objective(w + h_plot * e) - objective(w - h_plot * e)) / (2 * h_plot))(basis)
    # Reduce across the target axis to summarize ``.
    errors.append(float(jnp.max(jnp.abs(estimate - analytic))))
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'line', 'x': steps.tolist(), 'xscale': 'log', 'xlabel': 'finite-difference step', 'ylabel': 'maximum absolute gradient error', 'series': [{'label': 'float32 central difference', 'y': errors}]}
# Evaluate `visual_data['yscale']` from the current inputs and state.
visual_data['yscale'] = 'log'

# Experiment: Sweep the perturbation size
# Experiment — Sweep the perturbation size: Nearby float32 inputs can coincide, so reducing the step does...
# Iterate over `step_size` to step through the computation:
for step_size in [1e-1, 1e-2, 1e-4, 1e-8]:
    # Vectorize across the batch dimension with `jax.vmap` (`estimate`).
    estimate = jax.vmap(lambda e: (objective(w+step_size*e)-objective(w-step_size*e))/(2*step_size))(basis)
    # Print the observed values to compare against the expected result.
    print("h / max error:", step_size, float(jnp.max(jnp.abs(estimate-analytic))))
# Vectorize across the batch dimension with `jax.vmap` (`tiny`).
tiny = jax.vmap(lambda e: (objective(w+1e-8*e)-objective(w-1e-8*e))/(2e-8))(basis)
# Verify that the numerical values match the expected reference within tolerance.
assert not jnp.allclose(tiny, analytic, atol=1e-3)

# Experiment: Check a new direction
# Experiment — Check a new direction: A new projection supplies a separate numerical check, though it...
# Initialize array `direction` with explicit values and shape.
direction = jnp.array([1., -2.])
# Evaluate `directional` from the current inputs and state.
directional = (objective(w+h*direction)-objective(w-h*direction))/(2*h)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(directional, -8., atol=2e-4)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.dot(automatic, direction), -8.)

# Experiment: A symmetric probe can hide a kink
# Experiment — A symmetric probe can hide a kink: Agreement with one numerical probe can conceal nondifferentiability.
h_kink = 0.01
# Evaluate `symmetric` from the current inputs and state.
symmetric = (jnp.abs(h_kink)-jnp.abs(-h_kink))/(2*h_kink)
# Evaluate `left` from the current inputs and state.
left = (jnp.abs(0.)-jnp.abs(-h_kink))/h_kink
# Evaluate `right` from the current inputs and state.
right = (jnp.abs(h_kink)-jnp.abs(0.))/h_kink
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(symmetric,0.)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(left,-1.) and jnp.allclose(right,1.)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jax.grad(jnp.abs)(jnp.array(-2.)),-1.)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jax.grad(jnp.abs)(jnp.array(2.)),1.)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Check the gradient again at w=[2., -1.].
# Initialize array `minimum` with explicit values and shape.
minimum = jnp.array([2., -1.])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.grad(objective)(minimum), jnp.zeros(2))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(objective(minimum), 0.)

# Reference practice: Move to a different smooth objective
# Move to a different smooth objective (Transfer / diagnosis): The analytic reference checks a nonquadratic function at a...
# Initialize array `z` with explicit values and shape.
z = jnp.array([0.2, -0.7])
# Function `trig_loss(z)` implementing this stage's computation:
def trig_loss(z):
    # Return `jnp.sum(jnp.sin(z))` to the caller.
    return jnp.sum(jnp.sin(z))
# Vectorize across the batch dimension with `jax.vmap` (`fd`).
fd = jax.vmap(lambda e: (trig_loss(z+1e-2*e)-trig_loss(z-1e-2*e))/(2e-2))(jnp.eye(2))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.grad(trig_loss)(z), jnp.cos(z))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(fd, jnp.cos(z), atol=5e-5, rtol=5e-5)

# Reference practice: Catch the wrong objective despite correct gradients
# Catch the wrong objective despite correct gradients (Transfer / diagnosis): The omitted term is a specification error; numerical...
def incomplete(z):
    # Return `(z[0] - 2.0) ** 2` to the caller.
    return (z[0]-2.)**2
# Differentiate the objective to obtain `bad` via automatic differentiation.
bad = jax.grad(incomplete)(w)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(bad, jnp.array([-3., 0.]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not jnp.allclose(bad, analytic)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jax.grad(objective)(w), analytic)
print("PASS: optimization-02")
