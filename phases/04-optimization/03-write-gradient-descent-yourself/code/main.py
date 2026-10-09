"""Write gradient descent yourself: worked experiments and reference solutions. CPU checks."""

# 1. Prepare the inputs
# Step 1 — 1. Prepare the inputs: This block establishes the values used by the following steps; run...
# Import jax for this computation.
import jax
import jax.numpy as jnp

# 2. Define the computation
# Step 2 — 2. Define the computation: train defines a state transition that subtracts the derivative.
def loss(w):
    # Return `(w - 2.0) ** 2` to the caller.
    return (w - 2.) ** 2
# Define `train(initial, rate, steps)` to evaluate the objective and its automatic derivatives:
def train(initial, rate, steps):
    # Define `step(w, _)` to evaluate the objective and its automatic derivatives:
    def step(w, _):
        # Differentiate the objective to obtain `next_w` via automatic differentiation.
        next_w = w - rate * jax.grad(loss)(w)
        # Return `(next_w, loss(next_w))` to the caller.
        return next_w, loss(next_w)
    # Return `jax.lax.scan(step, jnp.array(initial), None, length=steps)` to the caller.
    return jax.lax.scan(step, jnp.array(initial), None, length=steps)

# 3. Measure and verify
# Step 3 — 3. Measure and verify: The weight approaches 2.0 and the final loss is below 10^{-8}.
final, history = train(0., 0.1, 60)
# Print the observed values to compare against the expected result.
print("Final weight:", float(final))
# Print diagnostic summary of the computed outputs.
print("Final loss:", float(history[-1]))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(final, 2., atol=1e-4)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert history[-1] < 1e-8
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.all(jnp.diff(history) <= 1e-7)

# Step 1 — 1. Prepare the inputs: This block establishes the values used by the following steps; run...
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Step 2 — 2. Define the computation: train defines a state transition that subtracts the derivative.
def loss(w):
    # Return `(w - 2.0) ** 2` to the caller.
    return (w - 2.) ** 2
# Define `train(initial, rate, steps)` to evaluate the objective and its automatic derivatives:
def train(initial, rate, steps):
    # Define `step(w, _)` to evaluate the objective and its automatic derivatives:
    def step(w, _):
        # Differentiate the objective to obtain `next_w` via automatic differentiation.
        next_w = w - rate * jax.grad(loss)(w)
        # Return `(next_w, loss(next_w))` to the caller.
        return next_w, loss(next_w)
    # Return `jax.lax.scan(step, jnp.array(initial), None, length=steps)` to the caller.
    return jax.lax.scan(step, jnp.array(initial), None, length=steps)
# Step 3 — 3. Measure and verify: The weight approaches 2.0 and the final loss is below 10^{-8}.
final, history = train(0., 0.1, 60)
# Print the observed values to compare against the expected result.
print("Final weight:", float(final))
# Print diagnostic summary of the computed outputs.
print("Final loss:", float(history[-1]))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(final, 2., atol=1e-4)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert history[-1] < 1e-8
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.all(jnp.diff(history) <= 1e-7)

# Figure data experiment
# Compute figure data for: Gradient descent trades step size for stability
# Evaluate `rates` from the current inputs and state.
rates = [0.1, 0.5, 1.1]
# Evaluate `series` from the current inputs and state.
series = []
# Loop over `rate_plot` in `rates`:
for rate_plot in rates:
    # Create device-backed JAX array `position_plot`.
    position_plot = jnp.array(0.0)
    # Evaluate `losses` from the current inputs and state.
    losses = []
    # Repeat the update loop over `range(12)` steps:
    for _ in range(12):
        # Append the current step result to `losses`.
        losses.append(float((position_plot - 2) ** 2))
        # Evaluate `position_plot` from the current inputs and state.
        position_plot = position_plot - rate_plot * 2 * (position_plot - 2)
    # Append the current step result to `series`.
    series.append({'label': 'rate ' + str(rate_plot), 'y': losses})
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'line', 'x': list(range(12)), 'xlabel': 'completed update', 'ylabel': 'squared error', 'yscale': 'symlog', 'series': series}

# Experiment: Check the first two steps
# Experiment — Check the first two steps: A hand-derived trajectory checks update sign and the recording...
two_w, two_losses = train(0., 0.1, 2)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(two_w, 0.72)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(two_losses, jnp.array([2.56, 1.6384]))
# Print the observed values to compare against the expected result.
print("First two post-update losses:", two_losses)

# Experiment: Test the edge of stability
# Experiment — Test the edge of stability: Magnitude-one error multipliers do not contract even though the...
edge_w, edge_losses = train(0., 1., 5)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(edge_w, 4.)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(edge_losses, jnp.full(5, 4.))

# Experiment: A tiny update far from the solution
# Experiment — A tiny update far from the solution: A stopping rule must distinguish lack of progress from reaching...
# Initialize array `far` with explicit values and shape.
far = jnp.array(0.)
# Evaluate `quadratic` from the current inputs and state.
quadratic = lambda value: (value-2.)**2
# Differentiate the objective to obtain `grad_far` via automatic differentiation.
grad_far = jax.grad(quadratic)(far)
# Evaluate `next_far` from the current inputs and state.
next_far = far - 1e-8*grad_far
# Verify contract: `jnp.abs(next_far - far) < 1e-06`.
assert jnp.abs(next_far-far)<1e-6
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.abs(grad_far)>3.
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert quadratic(next_far)>3.9

# Reference solution. Try the exercise before reading this.
# Exercise solution: Run ten steps with learning rate 1.1.
unstable, unstable_history = train(0., 1.1, 10)
# Verify contract: `unstable_history[-1] > loss(0.0)`.
assert unstable_history[-1] > loss(0.)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert abs(1 - 2 * 1.1) > 1

# Reference practice: Transfer the stability analysis
# Transfer the stability analysis (Transfer / diagnosis): Changing curvature changes the stability interval to...
def scaled_loss(w):
    # Return `4 * (w - 3.0) ** 2` to the caller.
    return 4*(w-3.)**2
# Define `scaled_step(w, _)` to evaluate the objective and its automatic derivatives:
def scaled_step(w, _):
    # Differentiate the objective to obtain `new` via automatic differentiation.
    new = w-0.1*jax.grad(scaled_loss)(w)
    # Return `(new, scaled_loss(new))` to the caller.
    return new, scaled_loss(new)
# Run compiled structured control flow via `jax.lax` (`(scaled_w, scaled_history)`).
scaled_w, scaled_history = jax.lax.scan(scaled_step, jnp.array(0.), None, length=20)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(scaled_w, 3., atol=1e-5)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.all(jnp.diff(scaled_history) <= 1e-6)

# Reference practice: Reproduce an uphill update and repair it
# Reproduce an uphill update and repair it (Transfer / diagnosis): The sign error is isolated using the same input and rate;...
# Differentiate the objective to obtain `uphill` via automatic differentiation.
uphill = 0. + 0.1*jax.grad(loss)(0.)
# Differentiate the objective to obtain `downhill` via automatic differentiation.
downhill = 0. - 0.1*jax.grad(loss)(0.)
# Verify contract: `loss(uphill) > loss(0.0)`.
assert loss(uphill) > loss(0.)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert loss(downhill) < loss(0.)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(downhill, 0.4)
print("PASS: optimization-03")
