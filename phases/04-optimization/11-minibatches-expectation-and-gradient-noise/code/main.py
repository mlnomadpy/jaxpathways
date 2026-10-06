"""Minibatches, expectation, and gradient noise: worked experiments and reference solutions. CPU checks."""

# Compute per-example gradients
import jax
import jax.numpy as jnp
y = jnp.array([1.0, 2.0, 3.0, 4.0])
w = jnp.array(0.0)

def example_loss(w, y):
    return 0.5 * (w - y) ** 2
per = jax.vmap(jax.grad(example_loss), in_axes=(None, 0))(w, y)
full = jax.grad(lambda w: jnp.mean(jax.vmap(example_loss, in_axes=(None, 0))(w, y)))(w)
assert jnp.allclose(per, jnp.array([-1.0, -2.0, -3.0, -4.0]))
assert jnp.allclose(full, -2.5)

# Enumerate every pair
pairs = (per[:, None] + per[None, :]) / 2
assert pairs.shape == (4, 4)
assert jnp.allclose(jnp.mean(pairs), full)
assert jnp.allclose(jnp.var(per), 1.25)
assert jnp.allclose(jnp.var(pairs), 0.625)

# Sample and replay one batch
key = jax.random.key(7)
indices = jax.random.choice(key, 4, shape=(2,), replace=True)
replay = jax.random.choice(key, 4, shape=(2,), replace=True)
assert jnp.array_equal(indices, replay)
print('full gradient / single variance / pair variance:', full, jnp.var(per), jnp.var(pairs))
print('sample indices / sample gradient:', indices, jnp.mean(per[indices]))

import jax
import jax.numpy as jnp
y = jnp.array([1.0, 2.0, 3.0, 4.0])
w = jnp.array(0.0)

def example_loss(w, y):
    return 0.5 * (w - y) ** 2
per = jax.vmap(jax.grad(example_loss), in_axes=(None, 0))(w, y)
full = jax.grad(lambda w: jnp.mean(jax.vmap(example_loss, in_axes=(None, 0))(w, y)))(w)
assert jnp.allclose(per, jnp.array([-1.0, -2.0, -3.0, -4.0]))
assert jnp.allclose(full, -2.5)

pairs = (per[:, None] + per[None, :]) / 2
assert pairs.shape == (4, 4)
assert jnp.allclose(jnp.mean(pairs), full)
assert jnp.allclose(jnp.var(per), 1.25)
assert jnp.allclose(jnp.var(pairs), 0.625)

key = jax.random.key(7)
indices = jax.random.choice(key, 4, shape=(2,), replace=True)
replay = jax.random.choice(key, 4, shape=(2,), replace=True)
assert jnp.array_equal(indices, replay)
print('full gradient / single variance / pair variance:', full, jnp.var(per), jnp.var(pairs))
print('sample indices / sample gradient:', indices, jnp.mean(per[indices]))

# Figure data experiment
triples = (per[:, None, None] + per[None, :, None] + per[None, None, :]) / 3
visual_data = {'kind': 'bar', 'labels': ['batch 1', 'batch 2', 'batch 3'], 'ylabel': 'gradient variance', 'series': [{'label': 'exact enumeration', 'y': [float(jnp.var(per)), float(jnp.var(pairs)), float(jnp.var(triples))]}]}

# Experiment: Combine unequal batches
first = jnp.mean(per[:3])
last = jnp.mean(per[3:])
wrong = (first + last) / 2
correct = (3 * first + last) / 4
assert jnp.allclose(wrong, -3.0)
assert jnp.allclose(correct, full)

# Experiment: Take a noisy step at the full optimum
full_loss = lambda w: jnp.mean(0.5 * (w - y) ** 2)
optimum = jnp.array(2.5)
noisy = optimum - 0.1 * jax.grad(example_loss)(optimum, y[0])
assert jnp.allclose(jax.grad(full_loss)(optimum), 0.0)
assert full_loss(noisy) > full_loss(optimum)

# Reference solution. Try the exercise before reading this.
triples = (per[:, None, None] + per[None, :, None] + per[None, None, :]) / 3
assert triples.size == 64
assert jnp.allclose(jnp.mean(triples), full, atol=1e-06)
assert jnp.allclose(jnp.var(triples), 1.25 / 3, atol=1e-06)

# Reference practice: Accumulate before updating
g1 = jax.grad(lambda z: jnp.mean(0.5 * (z - y[:3]) ** 2))(w)
g2 = jax.grad(lambda z: jnp.mean(0.5 * (z - y[3:]) ** 2))(w)
accumulated = (3 * g1 + g2) / 4
assert jnp.allclose(w - 0.1 * accumulated, w - 0.1 * full)

# Reference practice: Detect a biased sampler
biased = 0.9 * per[0] + 0.1 * per[3]
assert jnp.allclose(biased, -1.3)
assert not jnp.allclose(biased, full)
print("PASS: optimization-11")
