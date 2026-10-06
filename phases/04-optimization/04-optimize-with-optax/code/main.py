"""Optimize with Optax: worked experiments and reference solutions. CPU checks."""

# 1. Prepare the inputs
import jax
import jax.numpy as jnp
import optax

# 2. Define the computation
params = {"weight": jnp.array(0.), "bias": jnp.array(0.)}
x = jnp.linspace(-1., 1., 21)
y = 2. * x + 1.
def loss(p):
    return jnp.mean((p["weight"] * x + p["bias"] - y) ** 2)
optimizer = optax.sgd(0.2)
state = optimizer.init(params)

# 3. Measure and verify
initial_loss = loss(params)
for _ in range(120):
    grads = jax.grad(loss)(params)
    updates, state = optimizer.update(grads, state, params)
    params = optax.apply_updates(params, updates)
print("Parameters:", params)
print("Loss:", float(loss(params)))
assert loss(params) < 1e-8
assert jnp.allclose(params["weight"], 2., atol=1e-4)
assert jnp.allclose(params["bias"], 1., atol=1e-4)

import jax
import jax.numpy as jnp
import optax
params = {"weight": jnp.array(0.), "bias": jnp.array(0.)}
x = jnp.linspace(-1., 1., 21)
y = 2. * x + 1.
def loss(p):
    return jnp.mean((p["weight"] * x + p["bias"] - y) ** 2)
optimizer = optax.sgd(0.2)
state = optimizer.init(params)
initial_loss = loss(params)
for _ in range(120):
    grads = jax.grad(loss)(params)
    updates, state = optimizer.update(grads, state, params)
    params = optax.apply_updates(params, updates)
print("Parameters:", params)
print("Loss:", float(loss(params)))
assert loss(params) < 1e-8
assert jnp.allclose(params["weight"], 2., atol=1e-4)
assert jnp.allclose(params["bias"], 1., atol=1e-4)

# Figure data experiment
visual_data = {'kind': 'line', 'x': x.tolist(), 'xlabel': 'input', 'ylabel': 'prediction', 'series': [{'label': 'target', 'y': y.tolist()}, {'label': 'initial model', 'y': jnp.zeros_like(x).tolist()}, {'label': 'trained model', 'y': (params['weight'] * x + params['bias']).tolist()}]}

# Experiment: Check the first update against arithmetic
p0 = {"weight": jnp.array(0.), "bias": jnp.array(0.)}
g0 = jax.grad(loss)(p0)
u0, s0 = optimizer.update(g0, optimizer.init(p0), p0)
p1 = optax.apply_updates(p0, u0)
assert jnp.allclose(p1["weight"], 22/75)
assert jnp.allclose(p1["bias"], 0.4)
print("First parameters:", p1)

# Experiment: Observe stateful momentum
momentum = optax.sgd(0.1, momentum=0.9)
m0 = momentum.init(jnp.array(0.))
u1, m1 = momentum.update(jnp.array(2.), m0)
u2, m2 = momentum.update(jnp.array(1.), m1)
reset_u2, _ = momentum.update(jnp.array(1.), m0)
assert jnp.allclose(u1, -0.2)
assert jnp.allclose(u2, -0.28)
assert jnp.allclose(reset_u2, -0.1)

# Experiment: Coast through a zero-gradient step
momentum_tx = optax.sgd(0.1, momentum=0.9)
position = jnp.array(0.)
momentum_state = momentum_tx.init(position)
coast_first, momentum_state = momentum_tx.update(jnp.array(2.),momentum_state,position)
position = optax.apply_updates(position,coast_first)
coast_second, momentum_state = momentum_tx.update(jnp.array(0.),momentum_state,position)
assert jnp.allclose(coast_first,-0.2)
assert jnp.allclose(coast_second,-0.18)

# Reference solution. Try the exercise before reading this.
start = {"weight": jnp.array(0.), "bias": jnp.array(0.)}
grads = jax.grad(loss)(start)
updates, _ = optimizer.update(grads, optimizer.init(start), start)
actual = optax.apply_updates(start, updates)
expected = jax.tree.map(lambda p, g: p - 0.2 * g, start, grads)
assert all(jnp.allclose(a, b) for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)))

# Reference practice: Replay a momentum continuation
replay_u, replay_state = momentum.update(jnp.array(1.), m1)
assert jnp.allclose(replay_u, u2)
assert all(jnp.allclose(a,b) for a,b in zip(jax.tree.leaves(replay_state), jax.tree.leaves(m2)))

# Reference practice: Catch double subtraction
wrong = jax.tree.map(lambda p,u: p-u, p0,u0)
right = optax.apply_updates(p0,u0)
assert loss(wrong) > loss(p0)
assert loss(right) < loss(p0)
print("PASS: optimization-04")
