"""Adam, clipping, and learning-rate schedules: worked experiments and reference solutions. CPU checks."""

# Build a transparent Adam reference
# Step 1 — Build a transparent Adam reference: The supplied gradients isolate optimizer arithmetic; they are not...
import jax
import jax.numpy as jnp
import optax
params = jnp.zeros(2)
# Configure or step the Optax optimizer state (`adam`).
adam = optax.adam(0.1, b1=0.9, b2=0.99, eps=1e-08)
state = adam.init(params)
m = jnp.zeros(2)
v = jnp.zeros(2)
gradients = [jnp.array([2.0, -4.0]), jnp.array([1.0, 3.0])]

# Verify both moment updates
# Step 2 — Verify both moment updates: The first update is near (-0.1,0.1); the next depends on both...
for t, g in enumerate(gradients, start=1):
    m = 0.9 * m + 0.1 * g
    v = 0.99 * v + 0.01 * g * g
    expected = -0.1 * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.99 ** t)) + 1e-08)
    update, state = adam.update(g, state, params)
    assert jnp.allclose(update, expected, atol=2e-06, rtol=2e-06)
    # Apply the computed gradient updates to update the model parameters.
    params = optax.apply_updates(params, update)
    print('step / update:', t, update)

# Inspect one clipped update
# Step 3 — Inspect one clipped update: For this SGD chain, the update norm is 0.1.
g = jnp.array([3.0, 4.0])
# Configure or step the Optax optimizer state (`clip`).
clip = optax.clip_by_global_norm(1.0)
clipped, _ = clip.update(g, clip.init(jnp.zeros(2)))
assert jnp.allclose(clipped, jnp.array([0.6, 0.8]), atol=1e-06)
# Configure or step the Optax optimizer state (`chain`).
chain = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(0.1))
update, _ = chain.update(g, chain.init(jnp.zeros(2)))
assert jnp.allclose(update, jnp.array([-0.06, -0.08]), atol=1e-06)

# Step 1 — Build a transparent Adam reference: The supplied gradients isolate optimizer arithmetic; they are not...
import jax
import jax.numpy as jnp
import optax
params = jnp.zeros(2)
# Configure or step the Optax optimizer state (`adam`).
adam = optax.adam(0.1, b1=0.9, b2=0.99, eps=1e-08)
state = adam.init(params)
m = jnp.zeros(2)
v = jnp.zeros(2)
gradients = [jnp.array([2.0, -4.0]), jnp.array([1.0, 3.0])]

# Step 2 — Verify both moment updates: The first update is near (-0.1,0.1); the next depends on both...
for t, g in enumerate(gradients, start=1):
    m = 0.9 * m + 0.1 * g
    v = 0.99 * v + 0.01 * g * g
    expected = -0.1 * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.99 ** t)) + 1e-08)
    update, state = adam.update(g, state, params)
    assert jnp.allclose(update, expected, atol=2e-06, rtol=2e-06)
    # Apply the computed gradient updates to update the model parameters.
    params = optax.apply_updates(params, update)
    print('step / update:', t, update)

# Step 3 — Inspect one clipped update: For this SGD chain, the update norm is 0.1.
g = jnp.array([3.0, 4.0])
# Configure or step the Optax optimizer state (`clip`).
clip = optax.clip_by_global_norm(1.0)
clipped, _ = clip.update(g, clip.init(jnp.zeros(2)))
assert jnp.allclose(clipped, jnp.array([0.6, 0.8]), atol=1e-06)
# Configure or step the Optax optimizer state (`chain`).
chain = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(0.1))
update, _ = chain.update(g, chain.init(jnp.zeros(2)))
assert jnp.allclose(update, jnp.array([-0.06, -0.08]), atol=1e-06)

# Figure data experiment
# Compute figure data for: Compare optimizer paths under a fixed update budget
def plot_bowl(z):
    return 0.5 * (z[0] ** 2 + 10 * z[1] ** 2)
series = []
for name, tx in [('SGD 0.05', optax.sgd(0.05)), ('momentum 0.05', optax.sgd(0.05, momentum=0.9)), ('Adam 0.1', optax.adam(0.1))]:
    # Create device-backed JAX array `p_plot`.
    p_plot = jnp.array([1.0, 1.0])
    s_plot = tx.init(p_plot)
    vals = [float(plot_bowl(p_plot))]
    for _ in range(50):
        # Differentiate the objective to obtain gradients `(delta, s_plot)`.
        delta, s_plot = tx.update(jax.grad(plot_bowl)(p_plot), s_plot, p_plot)
        # Apply the computed gradient updates to update the model parameters.
        p_plot = optax.apply_updates(p_plot, delta)
        vals.append(float(plot_bowl(p_plot)))
    series.append({'label': name, 'y': vals})
visual_data = {'kind': 'line', 'x': list(range(51)), 'xlabel': 'completed update', 'ylabel': 'objective', 'yscale': 'log', 'series': series}

# Experiment: Inspect the schedule boundary
# Experiment — Inspect the schedule boundary: A schedule is indexed by optimizer updates.
# Configure or step the Optax optimizer state (`schedule`).
schedule = optax.linear_schedule(init_value=0.1, end_value=0.01, transition_steps=4)
rates = jnp.array([schedule(i) for i in range(6)])
assert jnp.allclose(rates, jnp.array([0.1, 0.0775, 0.055, 0.0325, 0.01, 0.01]), atol=1e-06)
# Configure or step the Optax optimizer state (`tx`).
tx = optax.sgd(schedule)
p = jnp.array(0.0)
s = tx.init(p)
for _ in range(5):
    delta, s = tx.update(jnp.array(1.0), s, p)
    # Configure or step the Optax optimizer state (`p`).
    p = optax.apply_updates(p, delta)
assert jnp.allclose(p, -0.275, atol=1e-06)
print('schedule rates:', rates)

# Experiment: Compare a fixed update budget
# Experiment — Compare a fixed update budget: This is a controlled demonstration at stated rates and update...
def bowl(w):
    return 0.5 * (w[0] ** 2 + 10 * w[1] ** 2)
start = jnp.array([1.0, 1.0])
# Configure or step the Optax optimizer state (`optimizers`).
optimizers = {'sgd': optax.sgd(0.05), 'momentum': optax.sgd(0.05, momentum=0.9), 'adam': optax.adam(0.1)}
traces = {}
for name, tx in optimizers.items():

    # Define `step(carry, _)` to evaluate the objective and its automatic derivatives:
    def step(carry, _):
        p, s = carry
        # Differentiate the objective to obtain `(delta, s)` via automatic differentiation.
        delta, s = tx.update(jax.grad(bowl)(p), s, p)
        # Configure or step the Optax optimizer state (`p`).
        p = optax.apply_updates(p, delta)
        return ((p, s), bowl(p))
    # Run compiled structured control flow via `jax.lax` (`((_, _), history)`).
    (_, _), history = jax.lax.scan(step, (start, tx.init(start)), None, length=50)
    # Confirm that all computed values remain finite (no NaN or Inf).
    assert jnp.all(jnp.isfinite(history))
    assert history[-1] < bowl(start)
    traces[name] = history
    print(name, 'initial / final loss:', bowl(start), history[-1])
assert not jnp.allclose(traces['sgd'], traces['adam'])

# Reference solution. Try the exercise before reading this.
# Exercise solution: Compare global norm clipping with coordinate-wise clipping on (3,4).
coordinate = jnp.clip(g, -1.0, 1.0)
assert jnp.allclose(coordinate, jnp.array([1.0, 1.0]))
assert jnp.allclose(clipped[0] / clipped[1], g[0] / g[1])
assert not jnp.allclose(coordinate[0] / coordinate[1], g[0] / g[1])

# Reference practice: Catch schedule restart
# Catch schedule restart (Transfer / diagnosis): Restoring weights without optimizer state can silently...
# Configure or step the Optax optimizer state (`tx`).
tx = optax.sgd(schedule)
p = jnp.array(0.0)
s = tx.init(p)
for _ in range(5):
    delta, s = tx.update(jnp.array(1.0), s, p)
    # Configure or step the Optax optimizer state (`p`).
    p = optax.apply_updates(p, delta)
continued, _ = tx.update(jnp.array(1.0), s, p)
restarted, _ = tx.update(jnp.array(1.0), tx.init(p), p)
assert jnp.allclose(continued, -0.01, atol=1e-06)
assert jnp.allclose(restarted, -0.1, atol=1e-06)

# Reference practice: Check transformation order
# Check transformation order (Transfer / diagnosis): Transformation order changes the algorithm.
# Configure or step the Optax optimizer state (`before`).
before = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(0.1))
# Configure or step the Optax optimizer state (`after`).
after = optax.chain(optax.sgd(0.1), optax.clip_by_global_norm(1.0))
u_before, _ = before.update(g, before.init(jnp.zeros(2)))
u_after, _ = after.update(g, after.init(jnp.zeros(2)))
assert jnp.allclose(jnp.linalg.norm(u_before), 0.1, atol=1e-06)
assert jnp.allclose(jnp.linalg.norm(u_after), 0.5, atol=1e-06)
print("PASS: optimization-12")
