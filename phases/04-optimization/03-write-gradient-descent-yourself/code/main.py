"""Write gradient descent yourself: worked experiments and reference solutions. CPU checks."""

# 1. Prepare the inputs
import jax
import jax.numpy as jnp

# 2. Define the computation
def loss(w):
    return (w - 2.) ** 2
def train(initial, rate, steps):
    def step(w, _):
        next_w = w - rate * jax.grad(loss)(w)
        return next_w, loss(next_w)
    return jax.lax.scan(step, jnp.array(initial), None, length=steps)

# 3. Measure and verify
final, history = train(0., 0.1, 60)
print("Final weight:", float(final))
print("Final loss:", float(history[-1]))
assert jnp.allclose(final, 2., atol=1e-4)
assert history[-1] < 1e-8
assert jnp.all(jnp.diff(history) <= 1e-7)

import jax
import jax.numpy as jnp
def loss(w):
    return (w - 2.) ** 2
def train(initial, rate, steps):
    def step(w, _):
        next_w = w - rate * jax.grad(loss)(w)
        return next_w, loss(next_w)
    return jax.lax.scan(step, jnp.array(initial), None, length=steps)
final, history = train(0., 0.1, 60)
print("Final weight:", float(final))
print("Final loss:", float(history[-1]))
assert jnp.allclose(final, 2., atol=1e-4)
assert history[-1] < 1e-8
assert jnp.all(jnp.diff(history) <= 1e-7)

# Figure data experiment
rates = [0.1, 0.5, 1.1]
series = []
for rate_plot in rates:
    position_plot = jnp.array(0.0)
    losses = []
    for _ in range(12):
        losses.append(float((position_plot - 2) ** 2))
        position_plot = position_plot - rate_plot * 2 * (position_plot - 2)
    series.append({'label': 'rate ' + str(rate_plot), 'y': losses})
visual_data = {'kind': 'line', 'x': list(range(12)), 'xlabel': 'completed update', 'ylabel': 'squared error', 'yscale': 'symlog', 'series': series}

# Experiment: Check the first two steps
two_w, two_losses = train(0., 0.1, 2)
assert jnp.allclose(two_w, 0.72)
assert jnp.allclose(two_losses, jnp.array([2.56, 1.6384]))
print("First two post-update losses:", two_losses)

# Experiment: Test the edge of stability
edge_w, edge_losses = train(0., 1., 5)
assert jnp.allclose(edge_w, 4.)
assert jnp.allclose(edge_losses, jnp.full(5, 4.))

# Experiment: A tiny update far from the solution
far = jnp.array(0.)
quadratic = lambda value: (value-2.)**2
grad_far = jax.grad(quadratic)(far)
next_far = far - 1e-8*grad_far
assert jnp.abs(next_far-far)<1e-6
assert jnp.abs(grad_far)>3.
assert quadratic(next_far)>3.9

# Reference solution. Try the exercise before reading this.
unstable, unstable_history = train(0., 1.1, 10)
assert unstable_history[-1] > loss(0.)
assert abs(1 - 2 * 1.1) > 1

# Reference practice: Transfer the stability analysis
def scaled_loss(w):
    return 4*(w-3.)**2
def scaled_step(w, _):
    new = w-0.1*jax.grad(scaled_loss)(w)
    return new, scaled_loss(new)
scaled_w, scaled_history = jax.lax.scan(scaled_step, jnp.array(0.), None, length=20)
assert jnp.allclose(scaled_w, 3., atol=1e-5)
assert jnp.all(jnp.diff(scaled_history) <= 1e-6)

# Reference practice: Reproduce an uphill update and repair it
uphill = 0. + 0.1*jax.grad(loss)(0.)
downhill = 0. - 0.1*jax.grad(loss)(0.)
assert loss(uphill) > loss(0.)
assert loss(downhill) < loss(0.)
assert jnp.allclose(downhill, 0.4)
print("PASS: optimization-03")
