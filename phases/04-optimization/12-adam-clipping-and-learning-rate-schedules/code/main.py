"""Adam, clipping, and learning-rate schedules: worked experiments and reference solutions. CPU checks."""

# Build a transparent Adam reference
# Step 1 — Build a transparent Adam reference: The supplied gradients isolate optimizer arithmetic; they are not...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import optax
# Initialize array `params` with explicit values and shape.
params = jnp.zeros(2)
# Configure or step the Optax optimizer state (`adam`).
adam = optax.adam(0.1, b1=0.9, b2=0.99, eps=1e-08)
# Run `adam.init` to compute `state`.
state = adam.init(params)
# Initialize array `m` with explicit values and shape.
m = jnp.zeros(2)
# Initialize array `v` with explicit values and shape.
v = jnp.zeros(2)
# Initialize array `gradients` with explicit values and shape.
gradients = [jnp.array([2.0, -4.0]), jnp.array([1.0, 3.0])]

# Verify both moment updates
# Step 2 — Verify both moment updates: The first update is near (-0.1,0.1); the next depends on both...
# Iterate over `(t, g)` to step through the computation:
for t, g in enumerate(gradients, start=1):
    # Evaluate `m` from the current inputs and state.
    m = 0.9 * m + 0.1 * g
    # Evaluate `v` from the current inputs and state.
    v = 0.99 * v + 0.01 * g * g
    # Evaluate `expected` from the current inputs and state.
    expected = -0.1 * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.99 ** t)) + 1e-08)
    # Run `adam.update` to compute `(update, state)`.
    update, state = adam.update(g, state, params)
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
    assert jnp.allclose(update, expected, atol=2e-06, rtol=2e-06)
    # Apply the computed gradient updates to update the model parameters.
    params = optax.apply_updates(params, update)
    # Print diagnostic summary of the computed outputs.
    print('step / update:', t, update)

# Inspect one clipped update
# Step 3 — Inspect one clipped update: For this SGD chain, the update norm is 0.1.
# Initialize array `g` with explicit values and shape.
g = jnp.array([3.0, 4.0])
# Configure or step the Optax optimizer state (`clip`).
clip = optax.clip_by_global_norm(1.0)
# Initialize array `(clipped, _)` with explicit values and shape.
clipped, _ = clip.update(g, clip.init(jnp.zeros(2)))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(clipped, jnp.array([0.6, 0.8]), atol=1e-06)
# Configure or step the Optax optimizer state (`chain`).
chain = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(0.1))
# Initialize array `(update, _)` with explicit values and shape.
update, _ = chain.update(g, chain.init(jnp.zeros(2)))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(update, jnp.array([-0.06, -0.08]), atol=1e-06)

# Step 1 — Build a transparent Adam reference: The supplied gradients isolate optimizer arithmetic; they are not...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import optax
# Initialize array `params` with explicit values and shape.
params = jnp.zeros(2)
# Configure or step the Optax optimizer state (`adam`).
adam = optax.adam(0.1, b1=0.9, b2=0.99, eps=1e-08)
# Run `adam.init` to compute `state`.
state = adam.init(params)
# Initialize array `m` with explicit values and shape.
m = jnp.zeros(2)
# Initialize array `v` with explicit values and shape.
v = jnp.zeros(2)
# Initialize array `gradients` with explicit values and shape.
gradients = [jnp.array([2.0, -4.0]), jnp.array([1.0, 3.0])]

# Step 2 — Verify both moment updates: The first update is near (-0.1,0.1); the next depends on both...
# Iterate over `(t, g)` to step through the computation:
for t, g in enumerate(gradients, start=1):
    # Evaluate `m` from the current inputs and state.
    m = 0.9 * m + 0.1 * g
    # Evaluate `v` from the current inputs and state.
    v = 0.99 * v + 0.01 * g * g
    # Evaluate `expected` from the current inputs and state.
    expected = -0.1 * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.99 ** t)) + 1e-08)
    # Run `adam.update` to compute `(update, state)`.
    update, state = adam.update(g, state, params)
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
    assert jnp.allclose(update, expected, atol=2e-06, rtol=2e-06)
    # Apply the computed gradient updates to update the model parameters.
    params = optax.apply_updates(params, update)
    # Print diagnostic summary of the computed outputs.
    print('step / update:', t, update)

# Step 3 — Inspect one clipped update: For this SGD chain, the update norm is 0.1.
# Initialize array `g` with explicit values and shape.
g = jnp.array([3.0, 4.0])
# Configure or step the Optax optimizer state (`clip`).
clip = optax.clip_by_global_norm(1.0)
# Initialize array `(clipped, _)` with explicit values and shape.
clipped, _ = clip.update(g, clip.init(jnp.zeros(2)))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(clipped, jnp.array([0.6, 0.8]), atol=1e-06)
# Configure or step the Optax optimizer state (`chain`).
chain = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(0.1))
# Initialize array `(update, _)` with explicit values and shape.
update, _ = chain.update(g, chain.init(jnp.zeros(2)))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(update, jnp.array([-0.06, -0.08]), atol=1e-06)

# Figure data experiment
# Compute figure data for: Compare optimizer paths under a fixed update budget
# Function `plot_bowl(z)` implementing this stage's computation:
def plot_bowl(z):
    # Return `0.5 * (z[0] ** 2 + 10 * z[1] ** 2)` to the caller.
    return 0.5 * (z[0] ** 2 + 10 * z[1] ** 2)
# Evaluate `series` from the current inputs and state.
series = []
# Loop over `(name, tx)` in `[('SGD 0.05', optax.sgd(0.05)), ('momentum 0.05', optax.sgd(0.05, momentum=0.9)), ('Adam 0.1', optax.adam(0.1))]`:
for name, tx in [('SGD 0.05', optax.sgd(0.05)), ('momentum 0.05', optax.sgd(0.05, momentum=0.9)), ('Adam 0.1', optax.adam(0.1))]:
    # Create device-backed JAX array `p_plot`.
    p_plot = jnp.array([1.0, 1.0])
    # Run `tx.init` to compute `s_plot`.
    s_plot = tx.init(p_plot)
    # Evaluate `vals` from the current inputs and state.
    vals = [float(plot_bowl(p_plot))]
    # Repeat the update loop over `range(50)` steps:
    for _ in range(50):
        # Differentiate the objective to obtain gradients `(delta, s_plot)`.
        delta, s_plot = tx.update(jax.grad(plot_bowl)(p_plot), s_plot, p_plot)
        # Apply the computed gradient updates to update the model parameters.
        p_plot = optax.apply_updates(p_plot, delta)
        # Append the current step result to `vals`.
        vals.append(float(plot_bowl(p_plot)))
    # Append the current step result to `series`.
    series.append({'label': name, 'y': vals})
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'line', 'x': list(range(51)), 'xlabel': 'completed update', 'ylabel': 'objective', 'yscale': 'log', 'series': series}

# Experiment: Inspect the schedule boundary
# Experiment — Inspect the schedule boundary: A schedule is indexed by optimizer updates.
# Configure or step the Optax optimizer state (`schedule`).
schedule = optax.linear_schedule(init_value=0.1, end_value=0.01, transition_steps=4)
# Initialize array `rates` with explicit values and shape.
rates = jnp.array([schedule(i) for i in range(6)])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(rates, jnp.array([0.1, 0.0775, 0.055, 0.0325, 0.01, 0.01]), atol=1e-06)
# Configure or step the Optax optimizer state (`tx`).
tx = optax.sgd(schedule)
# Initialize array `p` with explicit values and shape.
p = jnp.array(0.0)
# Run `tx.init` to compute `s`.
s = tx.init(p)
# Repeat the update loop over `range(5)` steps:
for _ in range(5):
    # Initialize array `(delta, s)` with explicit values and shape.
    delta, s = tx.update(jnp.array(1.0), s, p)
    # Configure or step the Optax optimizer state (`p`).
    p = optax.apply_updates(p, delta)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(p, -0.275, atol=1e-06)
# Print the observed values to compare against the expected result.
print('schedule rates:', rates)

# Experiment: Compare a fixed update budget
# Experiment — Compare a fixed update budget: This is a controlled demonstration at stated rates and update...
def bowl(w):
    # Return `0.5 * (w[0] ** 2 + 10 * w[1] ** 2)` to the caller.
    return 0.5 * (w[0] ** 2 + 10 * w[1] ** 2)
# Initialize array `start` with explicit values and shape.
start = jnp.array([1.0, 1.0])
# Configure or step the Optax optimizer state (`optimizers`).
optimizers = {'sgd': optax.sgd(0.05), 'momentum': optax.sgd(0.05, momentum=0.9), 'adam': optax.adam(0.1)}
# Evaluate `traces` from the current inputs and state.
traces = {}
# Iterate over `(name, tx)` to step through the computation:
for name, tx in optimizers.items():

    # Define `step(carry, _)` to evaluate the objective and its automatic derivatives:
    def step(carry, _):
        # Evaluate `(p, s)` from the current inputs and state.
        p, s = carry
        # Differentiate the objective to obtain `(delta, s)` via automatic differentiation.
        delta, s = tx.update(jax.grad(bowl)(p), s, p)
        # Configure or step the Optax optimizer state (`p`).
        p = optax.apply_updates(p, delta)
        # Return `((p, s), bowl(p))` to the caller.
        return ((p, s), bowl(p))
    # Run compiled structured control flow via `jax.lax` (`((_, _), history)`).
    (_, _), history = jax.lax.scan(step, (start, tx.init(start)), None, length=50)
    # Confirm that all computed values remain finite (no NaN or Inf).
    assert jnp.all(jnp.isfinite(history))
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
    assert history[-1] < bowl(start)
    # Evaluate `traces[name]` from the current inputs and state.
    traces[name] = history
    # Print diagnostic summary of the computed outputs.
    print(name, 'initial / final loss:', bowl(start), history[-1])
# Verify that the numerical values match the expected reference within tolerance.
assert not jnp.allclose(traces['sgd'], traces['adam'])

# Reference solution. Try the exercise before reading this.
# Exercise solution: Compare global norm clipping with coordinate-wise clipping on (3,4).
coordinate = jnp.clip(g, -1.0, 1.0)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(coordinate, jnp.array([1.0, 1.0]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(clipped[0] / clipped[1], g[0] / g[1])
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not jnp.allclose(coordinate[0] / coordinate[1], g[0] / g[1])

# Reference practice: Catch schedule restart
# Catch schedule restart (Transfer / diagnosis): Restoring weights without optimizer state can silently...
# Configure or step the Optax optimizer state (`tx`).
tx = optax.sgd(schedule)
# Initialize array `p` with explicit values and shape.
p = jnp.array(0.0)
# Run `tx.init` to compute `s`.
s = tx.init(p)
# Repeat the update loop over `range(5)` steps:
for _ in range(5):
    # Initialize array `(delta, s)` with explicit values and shape.
    delta, s = tx.update(jnp.array(1.0), s, p)
    # Configure or step the Optax optimizer state (`p`).
    p = optax.apply_updates(p, delta)
# Initialize array `(continued, _)` with explicit values and shape.
continued, _ = tx.update(jnp.array(1.0), s, p)
# Initialize array `(restarted, _)` with explicit values and shape.
restarted, _ = tx.update(jnp.array(1.0), tx.init(p), p)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(continued, -0.01, atol=1e-06)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(restarted, -0.1, atol=1e-06)

# Reference practice: Check transformation order
# Check transformation order (Transfer / diagnosis): Transformation order changes the algorithm.
# Configure or step the Optax optimizer state (`before`).
before = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(0.1))
# Configure or step the Optax optimizer state (`after`).
after = optax.chain(optax.sgd(0.1), optax.clip_by_global_norm(1.0))
# Initialize array `(u_before, _)` with explicit values and shape.
u_before, _ = before.update(g, before.init(jnp.zeros(2)))
# Initialize array `(u_after, _)` with explicit values and shape.
u_after, _ = after.update(g, after.init(jnp.zeros(2)))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jnp.linalg.norm(u_before), 0.1, atol=1e-06)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.linalg.norm(u_after), 0.5, atol=1e-06)
print("PASS: optimization-12")
