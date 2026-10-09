"""Optimize with Optax: worked experiments and reference solutions. CPU checks."""

# 1. Prepare the inputs
# Step 1 — 1. Prepare the inputs: This block establishes the values used by the following steps; run...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import optax

# 2. Define the computation
# Step 2 — 2. Define the computation: loss maps the two parameter leaves to row-wise predictions.
# Construct `params` via `{"weight": jnp.array(0.), "bias": jnp.array(0.)}`
params = {"weight": jnp.array(0.), "bias": jnp.array(0.)}
# Construct `x` via `jnp.linspace(-1., 1., 21)`
x = jnp.linspace(-1., 1., 21)
# Compute `y` from `2. * x + 1.`
y = 2. * x + 1.
# Function `loss(p)` implementing this stage's computation:
def loss(p):
    # Return `jnp.mean((p['weight'] * x + p['bias'] - y) ** 2)` to the caller.
    return jnp.mean((p["weight"] * x + p["bias"] - y) ** 2)
# Configure or step the Optax optimizer state (`optimizer`).
optimizer = optax.sgd(0.2)
# Run `optimizer.init` to compute `state`.
state = optimizer.init(params)

# 3. Measure and verify
# Step 3 — 3. Measure and verify: Weight approaches 2.0, bias approaches 1.0, and loss is below 10^{-8}.
initial_loss = loss(params)
# Repeat the update loop over `range(120)` steps:
for _ in range(120):
    # Differentiate the objective to obtain `grads` via automatic differentiation.
    grads = jax.grad(loss)(params)
    # Apply the computed gradient updates to update the model parameters.
    updates, state = optimizer.update(grads, state, params)
    # Configure or step the Optax optimizer state (`params`).
    params = optax.apply_updates(params, updates)
# Print the observed values to compare against the expected result.
print("Parameters:", params)
# Print diagnostic summary of the computed outputs.
print("Loss:", float(loss(params)))
# Assert invariant `loss(params) < 1e-8` holds
assert loss(params) < 1e-8
# Check numerical equivalence within tolerance: `jnp.allclose(params["weight"]`
assert jnp.allclose(params["weight"], 2., atol=1e-4)
# Check numerical equivalence within tolerance: `jnp.allclose(params["bias"]`
assert jnp.allclose(params["bias"], 1., atol=1e-4)

# Step 1 — 1. Prepare the inputs: This block establishes the values used by the following steps; run...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import optax
# Step 2 — 2. Define the computation: loss maps the two parameter leaves to row-wise predictions.
# Construct `params` via `{"weight": jnp.array(0.), "bias": jnp.array(0.)}`
params = {"weight": jnp.array(0.), "bias": jnp.array(0.)}
# Construct `x` via `jnp.linspace(-1., 1., 21)`
x = jnp.linspace(-1., 1., 21)
# Compute `y` from `2. * x + 1.`
y = 2. * x + 1.
# Function `loss(p)` implementing this stage's computation:
def loss(p):
    # Return `jnp.mean((p['weight'] * x + p['bias'] - y) ** 2)` to the caller.
    return jnp.mean((p["weight"] * x + p["bias"] - y) ** 2)
# Configure or step the Optax optimizer state (`optimizer`).
optimizer = optax.sgd(0.2)
# Run `optimizer.init` to compute `state`.
state = optimizer.init(params)
# Step 3 — 3. Measure and verify: Weight approaches 2.0, bias approaches 1.0, and loss is below 10^{-8}.
initial_loss = loss(params)
# Repeat the update loop over `range(120)` steps:
for _ in range(120):
    # Differentiate the objective to obtain `grads` via automatic differentiation.
    grads = jax.grad(loss)(params)
    # Apply the computed gradient updates to update the model parameters.
    updates, state = optimizer.update(grads, state, params)
    # Configure or step the Optax optimizer state (`params`).
    params = optax.apply_updates(params, updates)
# Print the observed values to compare against the expected result.
print("Parameters:", params)
# Print diagnostic summary of the computed outputs.
print("Loss:", float(loss(params)))
# Assert invariant `loss(params) < 1e-8` holds
assert loss(params) < 1e-8
# Check numerical equivalence within tolerance: `jnp.allclose(params["weight"]`
assert jnp.allclose(params["weight"], 2., atol=1e-4)
# Check numerical equivalence within tolerance: `jnp.allclose(params["bias"]`
assert jnp.allclose(params["bias"], 1., atol=1e-4)

# Figure data experiment
# Compute figure data for: The fitted line recovers slope and bias
# Allocate initialized array `visual_data` with the specified shape and dtype.
visual_data = {'kind': 'line', 'x': x.tolist(), 'xlabel': 'input', 'ylabel': 'prediction', 'series': [{'label': 'target', 'y': y.tolist()}, {'label': 'initial model', 'y': jnp.zeros_like(x).tolist()}, {'label': 'trained model', 'y': (params['weight'] * x + params['bias']).tolist()}]}

# Experiment: Check the first update against arithmetic
# Experiment — Check the first update against arithmetic: Independent arithmetic checks the gradient scale and the update...
# Construct `p0` via `{"weight": jnp.array(0.), "bias": jnp.array(0.)}`
p0 = {"weight": jnp.array(0.), "bias": jnp.array(0.)}
# Differentiate the objective to obtain `g0` via automatic differentiation.
g0 = jax.grad(loss)(p0)
# Apply the computed gradient updates to update the model parameters.
u0, s0 = optimizer.update(g0, optimizer.init(p0), p0)
# Configure or step the Optax optimizer state (`p1`).
p1 = optax.apply_updates(p0, u0)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(p1["weight"], 22/75)
# Check numerical equivalence within tolerance: `jnp.allclose(p1["bias"]`
assert jnp.allclose(p1["bias"], 0.4)
# Print the observed values to compare against the expected result.
print("First parameters:", p1)

# Experiment: Observe stateful momentum
# Experiment — Observe stateful momentum: Resetting momentum removes the prior-gradient contribution; it...
# Configure or step the Optax optimizer state (`momentum`).
momentum = optax.sgd(0.1, momentum=0.9)
# Construct `m0` via `momentum.init(jnp.array(0.))`
m0 = momentum.init(jnp.array(0.))
# Construct `u1, m1` via `momentum.update(jnp.array(2.), m0)`
u1, m1 = momentum.update(jnp.array(2.), m0)
# Construct `u2, m2` via `momentum.update(jnp.array(1.), m1)`
u2, m2 = momentum.update(jnp.array(1.), m1)
# Construct `reset_u2, _` via `momentum.update(jnp.array(1.), m0)`
reset_u2, _ = momentum.update(jnp.array(1.), m0)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(u1, -0.2)
# Check numerical equivalence within tolerance: `jnp.allclose(u2, -0.28)`
assert jnp.allclose(u2, -0.28)
# Check numerical equivalence within tolerance: `jnp.allclose(reset_u2, -0.1)`
assert jnp.allclose(reset_u2, -0.1)

# Experiment: Coast through a zero-gradient step
# Experiment — Coast through a zero-gradient step: Optimizer state carries information from earlier steps.
# Configure or step the Optax optimizer state (`momentum_tx`).
momentum_tx = optax.sgd(0.1, momentum=0.9)
# Construct `position` via `jnp.array(0.)`
position = jnp.array(0.)
# Run `momentum_tx.init` to compute `momentum_state`.
momentum_state = momentum_tx.init(position)
# Construct `coast_first, momentum_state` via `momentum_tx.update(jnp.array(2.),momentum_state,posi...`
coast_first, momentum_state = momentum_tx.update(jnp.array(2.),momentum_state,position)
# Configure or step the Optax optimizer state (`position`).
position = optax.apply_updates(position,coast_first)
# Construct `coast_second, momentum_state` via `momentum_tx.update(jnp.array(0.),momentum_state,posi...`
coast_second, momentum_state = momentum_tx.update(jnp.array(0.),momentum_state,position)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(coast_first,-0.2)
# Check numerical equivalence within tolerance: `jnp.allclose(coast_second,-0.18)`
assert jnp.allclose(coast_second,-0.18)

# Reference solution. Try the exercise before reading this.
# Exercise solution: At the original zero parameters, compare the first Optax SGD update...
# Construct `start` via `{"weight": jnp.array(0.), "bias": jnp.array(0.)}`
start = {"weight": jnp.array(0.), "bias": jnp.array(0.)}
# Differentiate the objective to obtain `grads` via automatic differentiation.
grads = jax.grad(loss)(start)
# Apply the computed gradient updates to update the model parameters.
updates, _ = optimizer.update(grads, optimizer.init(start), start)
# Configure or step the Optax optimizer state (`actual`).
actual = optax.apply_updates(start, updates)
# Transform every leaf of the parameter PyTree (`expected`).
expected = jax.tree.map(lambda p, g: p - 0.2 * g, start, grads)
# Verify that the numerical values match the expected reference within tolerance.
assert all(jnp.allclose(a, b) for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)))

# Reference practice: Replay a momentum continuation
# Replay a momentum continuation (Transfer / diagnosis): This verifies an in-memory optimizer transition.
# Construct `replay_u, replay_state` via `momentum.update(jnp.array(1.), m1)`
replay_u, replay_state = momentum.update(jnp.array(1.), m1)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(replay_u, u2)
# Check numerical equivalence within tolerance: `all(jnp.allclose(a,b) for a,b in zip(jax.tree.leaves(replay_state...`
assert all(jnp.allclose(a,b) for a,b in zip(jax.tree.leaves(replay_state), jax.tree.leaves(m2)))

# Reference practice: Catch double subtraction
# Catch double subtraction (Transfer / diagnosis): Comparing identical gradients exposes an update-sign error...
# Transform every leaf of the parameter PyTree (`wrong`).
wrong = jax.tree.map(lambda p,u: p-u, p0,u0)
# Configure or step the Optax optimizer state (`right`).
right = optax.apply_updates(p0,u0)
# Assert invariant `loss(wrong) > loss(p0)` holds
assert loss(wrong) > loss(p0)
# Assert invariant `loss(right) < loss(p0)` holds
assert loss(right) < loss(p0)
print("PASS: optimization-04")
