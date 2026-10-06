"""Probability, likelihood, and stable losses: worked experiments and reference solutions. CPU checks."""

# Write a loss from logits
import jax
import jax.numpy as jnp
import optax

def binary_loss(z, y):
    return jax.nn.softplus(z) - y * z
z = jnp.array(0.0)
y = jnp.array(1.0)
assert jnp.allclose(binary_loss(z, y), jnp.log(2.0))

# Check the derivative independently
assert jnp.allclose(jax.grad(binary_loss)(z, y), -0.5)
logits = jnp.array([-2.0, 0.0, 2.0])
labels = jnp.array([0.0, 1.0, 1.0])
analytic = jax.nn.sigmoid(logits) - labels
actual = jax.vmap(jax.grad(binary_loss))(logits, labels)
assert jnp.allclose(actual, analytic, atol=1e-06)
assert jnp.allclose(binary_loss(logits, labels), optax.sigmoid_binary_cross_entropy(logits, labels), atol=1e-06)

# Check a confident mistake
assert jnp.allclose(binary_loss(jnp.array(-100.0), jnp.array(1.0)), 100.0)
assert jnp.allclose(jax.grad(binary_loss)(jnp.array(-100.0), jnp.array(1.0)), -1.0)
print('uncertain loss / confident-wrong loss:', binary_loss(z, y), binary_loss(-100.0, 1.0))

import jax
import jax.numpy as jnp
import optax

def binary_loss(z, y):
    return jax.nn.softplus(z) - y * z
z = jnp.array(0.0)
y = jnp.array(1.0)
assert jnp.allclose(binary_loss(z, y), jnp.log(2.0))

assert jnp.allclose(jax.grad(binary_loss)(z, y), -0.5)
logits = jnp.array([-2.0, 0.0, 2.0])
labels = jnp.array([0.0, 1.0, 1.0])
analytic = jax.nn.sigmoid(logits) - labels
actual = jax.vmap(jax.grad(binary_loss))(logits, labels)
assert jnp.allclose(actual, analytic, atol=1e-06)
assert jnp.allclose(binary_loss(logits, labels), optax.sigmoid_binary_cross_entropy(logits, labels), atol=1e-06)

assert jnp.allclose(binary_loss(jnp.array(-100.0), jnp.array(1.0)), 100.0)
assert jnp.allclose(jax.grad(binary_loss)(jnp.array(-100.0), jnp.array(1.0)), -1.0)
print('uncertain loss / confident-wrong loss:', binary_loss(z, y), binary_loss(-100.0, 1.0))

# Figure data experiment
grid = jnp.linspace(-10.0, 10.0, 81)
visual_data = {'kind': 'line', 'x': grid.tolist(), 'xlabel': 'logit', 'ylabel': 'binary cross-entropy', 'series': [{'label': 'observed label 1', 'y': binary_loss(grid, 1.0).tolist()}, {'label': 'observed label 0', 'y': binary_loss(grid, 0.0).tolist()}]}

# Experiment: Break the probability-first implementation
p = jax.nn.sigmoid(jnp.array(100.0))
naive = -jnp.log(1 - p)
stable = binary_loss(jnp.array(100.0), jnp.array(0.0))
assert not jnp.isfinite(naive)
assert jnp.isfinite(stable) and jnp.allclose(stable, 100.0)

# Experiment: Shift every class score
scores = jnp.array([1.0, 2.0, 3.0])
target = 2
nll = -jax.nn.log_softmax(scores)[target]
shifted = -jax.nn.log_softmax(scores + 1000.0)[target]
assert jnp.allclose(nll, shifted, atol=1e-06)
reference_loss = jnp.log(jnp.exp(-2.0) + jnp.exp(-1.0) + 1.0)
assert jnp.allclose(nll, reference_loss, atol=1e-06)

# Reference solution. Try the exercise before reading this.
probabilities = jnp.array([0.8, 0.2])
z_from_p = jnp.log(probabilities) - jnp.log1p(-probabilities)
assert jnp.allclose(binary_loss(z_from_p, 1.0), -jnp.log(probabilities), atol=1e-06)
assert binary_loss(z_from_p[0], 1.0) < binary_loss(z_from_p[1], 1.0)

# Reference practice: Fit a constant probability
ys = jnp.array([1.0, 1.0, 1.0, 0.0])
constant_loss = lambda z: jnp.mean(binary_loss(z, ys))
optimum = jnp.log(3.0)
assert jnp.allclose(jax.grad(constant_loss)(optimum), 0.0, atol=1e-06)
assert constant_loss(optimum) < constant_loss(jnp.array(0.0))

# Reference practice: Catch a reduction mistake
zs = jnp.array([-1.0, 2.0])
ys2 = jnp.array([0.0, 1.0])
original = binary_loss(zs, ys2)
doubled = binary_loss(jnp.tile(zs, 2), jnp.tile(ys2, 2))
assert jnp.allclose(jnp.mean(original), jnp.mean(doubled))
assert jnp.allclose(jnp.sum(doubled), 2 * jnp.sum(original))
mean_grad = lambda z: jax.grad(lambda t: jnp.mean(binary_loss(t * z, ys2)))(1.0)
duplicated_grad = jax.grad(lambda t: jnp.mean(binary_loss(t * jnp.tile(zs, 2), jnp.tile(ys2, 2))))(1.0)
assert jnp.allclose(mean_grad(zs), duplicated_grad)
print("PASS: optimization-09")
