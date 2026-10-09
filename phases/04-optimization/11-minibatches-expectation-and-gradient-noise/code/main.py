"""Minibatches, expectation, and gradient noise: worked experiments and reference solutions. CPU checks."""

# Compute per-example gradients
# Step 1 — Compute per-example gradients: Each example supplies a slope; the dataset objective uses their mean.
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Initialize array `y` with explicit values and shape.
y = jnp.array([1.0, 2.0, 3.0, 4.0])
# Initialize array `w` with explicit values and shape.
w = jnp.array(0.0)

# Function `example_loss(w, y)` implementing this stage's computation:
def example_loss(w, y):
    # Return `0.5 * (w - y) ** 2` to the caller.
    return 0.5 * (w - y) ** 2
# Differentiate the objective to obtain `per` via automatic differentiation.
per = jax.vmap(jax.grad(example_loss), in_axes=(None, 0))(w, y)
# Differentiate the objective to obtain `full` via automatic differentiation.
full = jax.grad(lambda w: jnp.mean(jax.vmap(example_loss, in_axes=(None, 0))(w, y)))(w)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(per, jnp.array([-1.0, -2.0, -3.0, -4.0]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(full, -2.5)

# Enumerate every pair
# Step 2 — Enumerate every pair: All ordered pairs are equally likely under independent uniform draws.
pairs = (per[:, None] + per[None, :]) / 2
# Verify that the output tensor shape matches our prediction.
assert pairs.shape == (4, 4)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.mean(pairs), full)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.var(per), 1.25)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.var(pairs), 0.625)

# Sample and replay one batch
# Step 3 — Sample and replay one batch: Replay checks reproducibility.
# Create or split explicit PRNG key(s) (`key`) for reproducible randomness.
key = jax.random.key(7)
# Sample deterministic random values into `indices` using an explicit PRNG key.
indices = jax.random.choice(key, 4, shape=(2,), replace=True)
# Sample deterministic random values into `replay` using an explicit PRNG key.
replay = jax.random.choice(key, 4, shape=(2,), replace=True)
# Verify contract: `jnp.array_equal(indices, replay)`.
assert jnp.array_equal(indices, replay)
# Print the observed values to compare against the expected result.
print('full gradient / single variance / pair variance:', full, jnp.var(per), jnp.var(pairs))
# Print diagnostic summary of the computed outputs.
print('sample indices / sample gradient:', indices, jnp.mean(per[indices]))

# Step 1 — Compute per-example gradients: Each example supplies a slope; the dataset objective uses their mean.
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Initialize array `y` with explicit values and shape.
y = jnp.array([1.0, 2.0, 3.0, 4.0])
# Initialize array `w` with explicit values and shape.
w = jnp.array(0.0)

# Function `example_loss(w, y)` implementing this stage's computation:
def example_loss(w, y):
    # Return `0.5 * (w - y) ** 2` to the caller.
    return 0.5 * (w - y) ** 2
# Differentiate the objective to obtain `per` via automatic differentiation.
per = jax.vmap(jax.grad(example_loss), in_axes=(None, 0))(w, y)
# Differentiate the objective to obtain `full` via automatic differentiation.
full = jax.grad(lambda w: jnp.mean(jax.vmap(example_loss, in_axes=(None, 0))(w, y)))(w)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(per, jnp.array([-1.0, -2.0, -3.0, -4.0]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(full, -2.5)

# Step 2 — Enumerate every pair: All ordered pairs are equally likely under independent uniform draws.
pairs = (per[:, None] + per[None, :]) / 2
# Verify that the output tensor shape matches our prediction.
assert pairs.shape == (4, 4)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.mean(pairs), full)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.var(per), 1.25)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.var(pairs), 0.625)

# Step 3 — Sample and replay one batch: Replay checks reproducibility.
# Create or split explicit PRNG key(s) (`key`) for reproducible randomness.
key = jax.random.key(7)
# Sample deterministic random values into `indices` using an explicit PRNG key.
indices = jax.random.choice(key, 4, shape=(2,), replace=True)
# Sample deterministic random values into `replay` using an explicit PRNG key.
replay = jax.random.choice(key, 4, shape=(2,), replace=True)
# Verify contract: `jnp.array_equal(indices, replay)`.
assert jnp.array_equal(indices, replay)
# Print the observed values to compare against the expected result.
print('full gradient / single variance / pair variance:', full, jnp.var(per), jnp.var(pairs))
# Print diagnostic summary of the computed outputs.
print('sample indices / sample gradient:', indices, jnp.mean(per[indices]))

# Figure data experiment
# Compute figure data for: Averaging independent gradients reduces variance
# Evaluate `triples` from the current inputs and state.
triples = (per[:, None, None] + per[None, :, None] + per[None, None, :]) / 3
# Reduce across the target axis to summarize `visual_data`.
visual_data = {'kind': 'bar', 'labels': ['batch 1', 'batch 2', 'batch 3'], 'ylabel': 'gradient variance', 'series': [{'label': 'exact enumeration', 'y': [float(jnp.var(per)), float(jnp.var(pairs)), float(jnp.var(triples))]}]}

# Experiment: Combine unequal batches
# Experiment — Combine unequal batches: Batch means require example-count weights when batch sizes differ.
# Aggregate array values to compute `first`.
first = jnp.mean(per[:3])
# Aggregate array values to compute `last`.
last = jnp.mean(per[3:])
# Evaluate `wrong` from the current inputs and state.
wrong = (first + last) / 2
# Evaluate `correct` from the current inputs and state.
correct = (3 * first + last) / 4
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(wrong, -3.0)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(correct, full)

# Experiment: Take a noisy step at the full optimum
# Experiment — Take a noisy step at the full optimum: A noisy gradient may point away from the full optimum on a...
# Aggregate array values to compute `full_loss`.
full_loss = lambda w: jnp.mean(0.5 * (w - y) ** 2)
# Initialize array `optimum` with explicit values and shape.
optimum = jnp.array(2.5)
# Differentiate the objective to obtain `noisy` via automatic differentiation.
noisy = optimum - 0.1 * jax.grad(example_loss)(optimum, y[0])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.grad(full_loss)(optimum), 0.0)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert full_loss(noisy) > full_loss(optimum)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Enumerate all 64 ordered batches of size 3.
triples = (per[:, None, None] + per[None, :, None] + per[None, None, :]) / 3
# Verify contract: `triples.size == 64`.
assert triples.size == 64
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.mean(triples), full, atol=1e-06)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.var(triples), 1.25 / 3, atol=1e-06)

# Reference practice: Accumulate before updating
# Accumulate before updating (Transfer / diagnosis): Exact accumulation matches the full mean gradient only when...
# Differentiate the objective to obtain `g1` via automatic differentiation.
g1 = jax.grad(lambda z: jnp.mean(0.5 * (z - y[:3]) ** 2))(w)
# Differentiate the objective to obtain `g2` via automatic differentiation.
g2 = jax.grad(lambda z: jnp.mean(0.5 * (z - y[3:]) ** 2))(w)
# Evaluate `accumulated` from the current inputs and state.
accumulated = (3 * g1 + g2) / 4
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(w - 0.1 * accumulated, w - 0.1 * full)

# Reference practice: Detect a biased sampler
# Detect a biased sampler (Transfer / diagnosis): Uniform mean-gradient claims require the corresponding...
biased = 0.9 * per[0] + 0.1 * per[3]
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(biased, -1.3)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not jnp.allclose(biased, full)
print("PASS: optimization-11")
