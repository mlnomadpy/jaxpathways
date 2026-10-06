"""Attention from small pieces: worked experiments and reference solutions. CPU checks."""

# 1. Prepare the experiment
import jax
import jax.numpy as jnp
import numpy as np

# 2. Implement the mechanism
def attend(q, k, v):
    scores = q @ k.T / jnp.sqrt(q.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ v, weights

# 3. Run and check
q = jnp.zeros((2, 2))
k = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
v = jnp.array([[2., 0.], [0., 4.], [4., 2.]])
output, weights = attend(q, k, v)
print("Weights / output:", weights, output)
assert jnp.allclose(weights, jnp.full((2, 3), 1/3))
assert jnp.allclose(output, jnp.array([[2., 2.], [2., 2.]]))
assert jnp.allclose(jnp.sum(weights, axis=-1), 1.)

import jax
import jax.numpy as jnp
import numpy as np

def attend(q, k, v):
    scores = q @ k.T / jnp.sqrt(q.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ v, weights

q = jnp.zeros((2, 2))
k = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
v = jnp.array([[2., 0.], [0., 4.], [4., 2.]])
output, weights = attend(q, k, v)
print("Weights / output:", weights, output)
assert jnp.allclose(weights, jnp.full((2, 3), 1/3))
assert jnp.allclose(output, jnp.array([[2., 2.], [2., 2.]]))
assert jnp.allclose(jnp.sum(weights, axis=-1), 1.)

# Figure data experiment
changed_q = jnp.array([[3.0, 0.0], [0.0, 3.0]])
_, focused = attend(changed_q, k, v)
visual_data = {'kind': 'panels', 'panels': [{'kind': 'heatmap', 'title': 'Zero queries', 'values': weights.tolist(), 'rows': ['query 0', 'query 1'], 'columns': ['key 0', 'key 1', 'key 2'], 'unit': 'attention weight'}, {'kind': 'heatmap', 'title': 'Changed queries', 'values': focused.tolist(), 'rows': ['query 0', 'query 1'], 'columns': ['key 0', 'key 1', 'key 2'], 'unit': 'attention weight'}]}

# Experiment: Compare a nonzero query with NumPy
q_new = jnp.array([[2., 0.]])
actual, w_new = attend(q_new, k, v)
scores_np = np.asarray(q_new) @ np.asarray(k).T / np.sqrt(2)
raw = np.exp(scores_np - scores_np.max(axis=-1, keepdims=True))
reference_w = raw / raw.sum(axis=-1, keepdims=True)
np.testing.assert_allclose(w_new, reference_w, rtol=1e-6, atol=1e-6)
np.testing.assert_allclose(actual, reference_w @ np.asarray(v), rtol=1e-6, atol=1e-6)

# Experiment: Shift every value vector
shift = jnp.array([5., -2.])
shifted, _ = attend(q_new, k, v + shift)
assert jnp.allclose(shifted, actual + shift)

# Reference solution. Try the exercise before reading this.
order = jnp.array([2, 0, 1])
paired, _ = attend(q_new, k[order], v[order])
mismatched, _ = attend(q_new, k, v[order])
assert jnp.allclose(paired, actual)
assert not jnp.allclose(mismatched, actual)

# Reference practice: Use a different value dimension
scalar_v = jnp.array([[3.], [6.], [9.]])
scalar_out, _ = attend(q, k, scalar_v)
assert scalar_out.shape == (2, 1)
assert jnp.allclose(scalar_out, 6.)

# Reference practice: Expose a wrong softmax axis
wrong_w = jax.nn.softmax(jnp.zeros((2, 3)), axis=0)
assert jnp.allclose(wrong_w.sum(axis=-1), 1.5)
right_w = jax.nn.softmax(jnp.zeros((2, 3)), axis=-1)
assert jnp.allclose(right_w.sum(axis=-1), 1.)
print("PASS: transformers-01")
