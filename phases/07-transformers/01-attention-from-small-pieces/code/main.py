"""Attention from small pieces: worked experiments and reference solutions. CPU checks."""

# 1. Prepare the experiment
# Step 1 — 1. Prepare the experiment: These imports provide JAX array transformations and NumPy...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# 2. Implement the mechanism
# Step 2 — 2. Implement the mechanism: Multiplying Q\in\mathbb{R}^{T_q\times D_k} by...
def attend(q, k, v):
    # Perform matrix contraction / projection to compute `scores`.
    scores = q @ k.T / jnp.sqrt(q.shape[-1])
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights = jax.nn.softmax(scores, axis=-1)
    # Return `(weights @ v, weights)` to the caller.
    return weights @ v, weights

# 3. Run and check
# Step 3 — 3. Run and check: Uniform weights 1/3; both output rows [2, 2].
# Construct `q` via `jnp.zeros((2, 2))`
q = jnp.zeros((2, 2))
# Construct `k` via `jnp.array([[1., 0.], [0., 1.], [1., 1.]])`
k = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
# Construct `v` via `jnp.array([[2., 0.], [0., 4.], [4., 2.]])`
v = jnp.array([[2., 0.], [0., 4.], [4., 2.]])
# Run `attend` to compute `(output, weights)`.
output, weights = attend(q, k, v)
# Print the observed values to compare against the expected result.
print("Weights / output:", weights, output)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(weights, jnp.full((2, 3), 1/3))
# Check numerical equivalence within tolerance: `jnp.allclose(output, jnp.array([[2., 2.], [2., 2.]]))`
assert jnp.allclose(output, jnp.array([[2., 2.], [2., 2.]]))
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.sum(weights, axis=-1), 1.)`
assert jnp.allclose(jnp.sum(weights, axis=-1), 1.)

# Step 1 — 1. Prepare the experiment: These imports provide JAX array transformations and NumPy...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Step 2 — 2. Implement the mechanism: Multiplying Q\in\mathbb{R}^{T_q\times D_k} by...
def attend(q, k, v):
    # Perform matrix contraction / projection to compute `scores`.
    scores = q @ k.T / jnp.sqrt(q.shape[-1])
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights = jax.nn.softmax(scores, axis=-1)
    # Return `(weights @ v, weights)` to the caller.
    return weights @ v, weights

# Step 3 — 3. Run and check: Uniform weights 1/3; both output rows [2, 2].
# Construct `q` via `jnp.zeros((2, 2))`
q = jnp.zeros((2, 2))
# Construct `k` via `jnp.array([[1., 0.], [0., 1.], [1., 1.]])`
k = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
# Construct `v` via `jnp.array([[2., 0.], [0., 4.], [4., 2.]])`
v = jnp.array([[2., 0.], [0., 4.], [4., 2.]])
# Run `attend` to compute `(output, weights)`.
output, weights = attend(q, k, v)
# Print the observed values to compare against the expected result.
print("Weights / output:", weights, output)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(weights, jnp.full((2, 3), 1/3))
# Check numerical equivalence within tolerance: `jnp.allclose(output, jnp.array([[2., 2.], [2., 2.]]))`
assert jnp.allclose(output, jnp.array([[2., 2.], [2., 2.]]))
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.sum(weights, axis=-1), 1.)`
assert jnp.allclose(jnp.sum(weights, axis=-1), 1.)

# Figure data experiment
# Compute figure data for: Each query distributes one unit of attention
# Create device-backed JAX array `changed_q`.
changed_q = jnp.array([[3.0, 0.0], [0.0, 3.0]])
# Run `attend` to compute `(_, focused)`.
_, focused = attend(changed_q, k, v)
# Compute `visual_data` from `{'kind': 'panels', 'panels': [{'kind': 'heatmap', 't...`
visual_data = {'kind': 'panels', 'panels': [{'kind': 'heatmap', 'title': 'Zero queries', 'values': weights.tolist(), 'rows': ['query 0', 'query 1'], 'columns': ['key 0', 'key 1', 'key 2'], 'unit': 'attention weight'}, {'kind': 'heatmap', 'title': 'Changed queries', 'values': focused.tolist(), 'rows': ['query 0', 'query 1'], 'columns': ['key 0', 'key 1', 'key 2'], 'unit': 'attention weight'}]}

# Experiment: Compare a nonzero query with NumPy
# Experiment — Compare a nonzero query with NumPy: A separate NumPy calculation checks nonuniform weighting and the...
# Construct `q_new` via `jnp.array([[2., 0.]])`
q_new = jnp.array([[2., 0.]])
# Run `attend` to compute `(actual, w_new)`.
actual, w_new = attend(q_new, k, v)
# Convert `scores_np` to a host NumPy array for inspection or verification.
scores_np = np.asarray(q_new) @ np.asarray(k).T / np.sqrt(2)
# Reduce along axis=-1 to compute `raw`.
raw = np.exp(scores_np - scores_np.max(axis=-1, keepdims=True))
# Reduce along axis=-1 to compute `reference_w`.
reference_w = raw / raw.sum(axis=-1, keepdims=True)
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(w_new, reference_w, rtol=1e-6, atol=1e-6)`
np.testing.assert_allclose(w_new, reference_w, rtol=1e-6, atol=1e-6)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(actual, reference_w @ np.asarray(v), rtol=1e-6, atol=1e-6)

# Experiment: Shift every value vector
# Experiment — Shift every value vector: Row-normalized attention preserves a shared value translation.
# Construct `shift` via `jnp.array([5., -2.])`
shift = jnp.array([5., -2.])
# Run `attend` to compute `(shifted, _)`.
shifted, _ = attend(q_new, k, v + shift)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(shifted, actual + shift)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Permute the three key/value pairs together, then permute only values.
# Construct `order` via `jnp.array([2, 0, 1])`
order = jnp.array([2, 0, 1])
# Run `attend` to compute `(paired, _)`.
paired, _ = attend(q_new, k[order], v[order])
# Run `attend` to compute `(mismatched, _)`.
mismatched, _ = attend(q_new, k, v[order])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(paired, actual)
# Check numerical equivalence within tolerance: `not jnp.allclose(mismatched, actual)`
assert not jnp.allclose(mismatched, actual)

# Reference practice: Use a different value dimension
# Use a different value dimension (Transfer / diagnosis): Queries and keys share Dk; values may have an independent...
# Construct `scalar_v` via `jnp.array([[3.], [6.], [9.]])`
scalar_v = jnp.array([[3.], [6.], [9.]])
# Run `attend` to compute `(scalar_out, _)`.
scalar_out, _ = attend(q, k, scalar_v)
# Check tensor shape invariant: `scalar_out.shape == (2, 1)`
assert scalar_out.shape == (2, 1)
# Check numerical equivalence within tolerance: `jnp.allclose(scalar_out, 6.)`
assert jnp.allclose(scalar_out, 6.)

# Reference practice: Expose a wrong softmax axis
# Expose a wrong softmax axis (Transfer / diagnosis): A legal operation can normalize the wrong axis.
# Construct `wrong_w` via `jax.nn.softmax(jnp.zeros((2, 3)), axis=0)`
wrong_w = jax.nn.softmax(jnp.zeros((2, 3)), axis=0)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(wrong_w.sum(axis=-1), 1.5)
# Construct `right_w` via `jax.nn.softmax(jnp.zeros((2, 3)), axis=-1)`
right_w = jax.nn.softmax(jnp.zeros((2, 3)), axis=-1)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(right_w.sum(axis=-1), 1.)
print("PASS: transformers-01")
