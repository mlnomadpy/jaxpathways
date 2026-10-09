"""Masking, tokenization, and sequence packing: worked experiments and reference solutions. CPU checks."""

# 1. Prepare the experiment
# Step 1 — 1. Prepare the experiment: These imports provide JAX array transformations and NumPy...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# 2. Implement the mechanism
# Step 2 — 2. Implement the mechanism: allowed key columns query 0: 1 0 0 0 query 1: 1 1 0 0 query 2: 1 1...
def masked_mean(values, allowed):
    # Combine or mask array elements to form `scores`.
    scores = jnp.where(allowed, 0., -jnp.inf)
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights = jax.nn.softmax(scores, axis=-1)
    # Return `(weights @ values, weights)` to the caller.
    return weights @ values, weights

# Function `make_mask(segment_ids, valid)` implementing this stage's computation:
def make_mask(segment_ids, valid):
    # Construct `positions` via `jnp.arange(segment_ids.shape[0])`
    positions = jnp.arange(segment_ids.shape[0])
    # Compute `causal` from `positions[:, None] >= positions[None, :]`
    causal = positions[:, None] >= positions[None, :]
    # Compute `same_segment` from `segment_ids[:, None] == segment_ids[None, :]`
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    # Return `causal & same_segment & valid[:, None] & valid[None, :]` to the caller.
    return causal & same_segment & valid[:, None] & valid[None, :]

# 3. Run and check
# Step 3 — 3. Run and check: Causal means [2, 3, 4.666667, 7.5]; no weight above the diagonal.
# Construct `values` via `jnp.array([[2.], [4.], [8.], [16.]])`
values = jnp.array([[2.], [4.], [8.], [16.]])
# Construct `causal` via `jnp.arange(4)[:, None] >= jnp.arange(4)[None, :]`
causal = jnp.arange(4)[:, None] >= jnp.arange(4)[None, :]
# Run `masked_mean` to compute `(out, weights)`.
out, weights = masked_mean(values, causal)
# Assert that `jnp.allclose(out[:, 0], jnp.array([2., 3., 14/3, 7.5]))`.
assert jnp.allclose(out[:, 0], jnp.array([2., 3., 14/3, 7.5]))
# Assert that `jnp.allclose(jnp.triu(weights, 1), 0.)`.
assert jnp.allclose(jnp.triu(weights, 1), 0.)
# Print the observed values to compare against the expected result.
print("Causal means:", out[:, 0])

# Step 1 — 1. Prepare the experiment: These imports provide JAX array transformations and NumPy...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Step 2 — 2. Implement the mechanism: allowed key columns query 0: 1 0 0 0 query 1: 1 1 0 0 query 2: 1 1...
def masked_mean(values, allowed):
    # Combine or mask array elements to form `scores`.
    scores = jnp.where(allowed, 0., -jnp.inf)
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights = jax.nn.softmax(scores, axis=-1)
    # Return `(weights @ values, weights)` to the caller.
    return weights @ values, weights

# Function `make_mask(segment_ids, valid)` implementing this stage's computation:
def make_mask(segment_ids, valid):
    # Construct `positions` via `jnp.arange(segment_ids.shape[0])`
    positions = jnp.arange(segment_ids.shape[0])
    # Compute `causal` from `positions[:, None] >= positions[None, :]`
    causal = positions[:, None] >= positions[None, :]
    # Compute `same_segment` from `segment_ids[:, None] == segment_ids[None, :]`
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    # Return `causal & same_segment & valid[:, None] & valid[None, :]` to the caller.
    return causal & same_segment & valid[:, None] & valid[None, :]

# Step 3 — 3. Run and check: Causal means [2, 3, 4.666667, 7.5]; no weight above the diagonal.
# Construct `values` via `jnp.array([[2.], [4.], [8.], [16.]])`
values = jnp.array([[2.], [4.], [8.], [16.]])
# Construct `causal` via `jnp.arange(4)[:, None] >= jnp.arange(4)[None, :]`
causal = jnp.arange(4)[:, None] >= jnp.arange(4)[None, :]
# Run `masked_mean` to compute `(out, weights)`.
out, weights = masked_mean(values, causal)
# Assert that `jnp.allclose(out[:, 0], jnp.array([2., 3., 14/3, 7.5]))`.
assert jnp.allclose(out[:, 0], jnp.array([2., 3., 14/3, 7.5]))
# Assert that `jnp.allclose(jnp.triu(weights, 1), 0.)`.
assert jnp.allclose(jnp.triu(weights, 1), 0.)
# Print the observed values to compare against the expected result.
print("Causal means:", out[:, 0])

# Figure data experiment
# Compute figure data for: A causal mask blocks future information
# Compute `visual_data` from `{'kind': 'heatmap', 'values': weights.tolist(), 'row...`
visual_data = {'kind': 'heatmap', 'values': weights.tolist(), 'rows': ['query ' + str(i) for i in range(4)], 'columns': ['key ' + str(i) for i in range(4)], 'unit': 'attention weight'}

# Experiment: Perturb the future
# Experiment — Perturb the future: Perturbation tests whether earlier representations depend on...
changed = values.at[-1, 0].set(1000.)
# Run `masked_mean` to compute `(changed_out, _)`.
changed_out, _ = masked_mean(changed, causal)
# Assert that `jnp.allclose(changed_out[:3], out[:3])`.
assert jnp.allclose(changed_out[:3], out[:3])
# Assert that `not jnp.allclose(changed_out[-1], out[-1])`.
assert not jnp.allclose(changed_out[-1], out[-1])

# Experiment: Pack two independent examples
# Experiment — Pack two independent examples: Causal order alone does not isolate independent packed examples.
# Construct `segments` via `jnp.array([0, 0, 1, 1])`
segments = jnp.array([0, 0, 1, 1])
# Construct `valid` via `jnp.ones(4, dtype=bool)`
valid = jnp.ones(4, dtype=bool)
# Run `make_mask` to compute `packed`.
packed = make_mask(segments, valid)
# Run `masked_mean` to compute `(packed_out, _)`.
packed_out, _ = masked_mean(values, packed)
# Assert that `jnp.allclose(packed_out[:, 0], jnp.array([2., 3., 8., 12.]))`.
assert jnp.allclose(packed_out[:, 0], jnp.array([2., 3., 8., 12.]))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Encode abcab with a three-character vocabulary.
vocab = {"a":0, "b":1, "c":2}
# Construct `ids` via `jnp.array([vocab[c] for c in "abcab"])`
ids = jnp.array([vocab[c] for c in "abcab"])
# Assert invariant `jnp.array_equal(ids[:-1], jnp.array([0,1,2,0]))` holds
assert jnp.array_equal(ids[:-1], jnp.array([0,1,2,0]))
# Assert invariant `jnp.array_equal(ids[1:], jnp.array([1,2,0,1]))` holds
assert jnp.array_equal(ids[1:], jnp.array([1,2,0,1]))
# Construct `seg` via `jnp.array([0,0,0,1,1])`
seg = jnp.array([0,0,0,1,1])
# Compute `loss_valid` from `seg[:-1] == seg[1:]`
loss_valid = seg[:-1] == seg[1:]
# Assert invariant `jnp.array_equal(loss_valid, jnp.array([True,True,False,True]))` holds
assert jnp.array_equal(loss_valid, jnp.array([True,True,False,True]))

# Reference practice: Repair padding-query NaNs
# Repair padding-query NaNs (Transfer / diagnosis): A fallback makes normalization defined; zeroing and loss...
# Construct `pad_valid` via `jnp.array([True,True,False,False])`
pad_valid = jnp.array([True,True,False,False])
# Construct `pad_mask` via `make_mask(jnp.zeros(4, dtype=int), pad_valid)`
pad_mask = make_mask(jnp.zeros(4, dtype=int), pad_valid)
# Combine or mask array elements to form `(bad_pad, _)`.
bad_pad, _ = masked_mean(values, pad_mask)
# Confirm that all computed values remain finite (no NaN or Inf).
assert not jnp.all(jnp.isfinite(bad_pad))
# Compute `fallback` from `(~pad_valid[:, None]) & jnp.eye(4, dtype=bool)`
fallback = (~pad_valid[:, None]) & jnp.eye(4, dtype=bool)
# Combine or mask array elements to form `(pad_out, _)`.
pad_out, _ = masked_mean(values, pad_mask | fallback)
# Combine or mask array elements to form `pad_out`.
pad_out = jnp.where(pad_valid[:, None], pad_out, 0.)
# Confirm that all computed values remain finite (no NaN or Inf).
assert jnp.all(jnp.isfinite(pad_out))
# Assert that `jnp.allclose(pad_out[:,0], jnp.array([2.,3.,0.,0.]))`.
assert jnp.allclose(pad_out[:,0], jnp.array([2.,3.,0.,0.]))

# Reference practice: Detect packed-example leakage
# Detect packed-example leakage (Transfer / diagnosis): The earlier values 2 and 4 belong to another example; they...
# Assert that `jnp.allclose(out[2,0], 14/3)`.
assert jnp.allclose(out[2,0], 14/3)
# Assert that `jnp.allclose(packed_out[2,0], 8.)`.
assert jnp.allclose(packed_out[2,0], 8.)
# Assert that `not jnp.allclose(out[2,0], packed_out[2,0])`.
assert not jnp.allclose(out[2,0], packed_out[2,0])
print("PASS: transformers-02")
