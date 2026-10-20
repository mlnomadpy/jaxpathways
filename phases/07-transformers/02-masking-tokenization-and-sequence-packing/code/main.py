"""Masking, tokenization, and sequence packing: worked experiments and reference solutions. CPU checks."""

# 1. Prepare the experiment
# Step 1 — 1. Prepare the experiment: These imports provide JAX array transformations and NumPy...
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
    return weights @ values, weights

def make_mask(segment_ids, valid):
    positions = jnp.arange(segment_ids.shape[0])
    causal = positions[:, None] >= positions[None, :]
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    return causal & same_segment & valid[:, None] & valid[None, :]

# 3. Run and check
# Step 3 — 3. Run and check: Causal means [2, 3, 4.666667, 7.5]; no weight above the diagonal.
values = jnp.array([[2.], [4.], [8.], [16.]])
causal = jnp.arange(4)[:, None] >= jnp.arange(4)[None, :]
out, weights = masked_mean(values, causal)
assert jnp.allclose(out[:, 0], jnp.array([2., 3., 14/3, 7.5]))
assert jnp.allclose(jnp.triu(weights, 1), 0.)
print("Causal means:", out[:, 0])

# Step 1 — 1. Prepare the experiment: These imports provide JAX array transformations and NumPy...
import jax
import jax.numpy as jnp
import numpy as np

# Step 2 — 2. Implement the mechanism: allowed key columns query 0: 1 0 0 0 query 1: 1 1 0 0 query 2: 1 1...
def masked_mean(values, allowed):
    # Combine or mask array elements to form `scores`.
    scores = jnp.where(allowed, 0., -jnp.inf)
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ values, weights

def make_mask(segment_ids, valid):
    positions = jnp.arange(segment_ids.shape[0])
    causal = positions[:, None] >= positions[None, :]
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    return causal & same_segment & valid[:, None] & valid[None, :]

# Step 3 — 3. Run and check: Causal means [2, 3, 4.666667, 7.5]; no weight above the diagonal.
values = jnp.array([[2.], [4.], [8.], [16.]])
causal = jnp.arange(4)[:, None] >= jnp.arange(4)[None, :]
out, weights = masked_mean(values, causal)
assert jnp.allclose(out[:, 0], jnp.array([2., 3., 14/3, 7.5]))
assert jnp.allclose(jnp.triu(weights, 1), 0.)
print("Causal means:", out[:, 0])

# Figure data experiment
# Compute figure data for: A causal mask blocks future information
visual_data = {'kind': 'heatmap', 'values': weights.tolist(), 'rows': ['query ' + str(i) for i in range(4)], 'columns': ['key ' + str(i) for i in range(4)], 'unit': 'attention weight'}

# Experiment: Perturb the future
# Experiment — Perturb the future: Perturbation tests whether earlier representations depend on...
changed = values.at[-1, 0].set(1000.)
changed_out, _ = masked_mean(changed, causal)
assert jnp.allclose(changed_out[:3], out[:3])
assert not jnp.allclose(changed_out[-1], out[-1])

# Experiment: Pack two independent examples
# Experiment — Pack two independent examples: Causal order alone does not isolate independent packed examples.
segments = jnp.array([0, 0, 1, 1])
valid = jnp.ones(4, dtype=bool)
packed = make_mask(segments, valid)
packed_out, _ = masked_mean(values, packed)
assert jnp.allclose(packed_out[:, 0], jnp.array([2., 3., 8., 12.]))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Encode abcab with a three-character vocabulary.
vocab = {"a":0, "b":1, "c":2}
ids = jnp.array([vocab[c] for c in "abcab"])
assert jnp.array_equal(ids[:-1], jnp.array([0,1,2,0]))
assert jnp.array_equal(ids[1:], jnp.array([1,2,0,1]))
seg = jnp.array([0,0,0,1,1])
loss_valid = seg[:-1] == seg[1:]
assert jnp.array_equal(loss_valid, jnp.array([True,True,False,True]))

# Reference practice: Repair padding-query NaNs
# Repair padding-query NaNs (Transfer / diagnosis): A fallback makes normalization defined; zeroing and loss...
pad_valid = jnp.array([True,True,False,False])
pad_mask = make_mask(jnp.zeros(4, dtype=int), pad_valid)
# Combine or mask array elements to form `(bad_pad, _)`.
bad_pad, _ = masked_mean(values, pad_mask)
# Confirm that all computed values remain finite (no NaN or Inf).
assert not jnp.all(jnp.isfinite(bad_pad))
fallback = (~pad_valid[:, None]) & jnp.eye(4, dtype=bool)
# Combine or mask array elements to form `(pad_out, _)`.
pad_out, _ = masked_mean(values, pad_mask | fallback)
# Combine or mask array elements to form `pad_out`.
pad_out = jnp.where(pad_valid[:, None], pad_out, 0.)
# Confirm that all computed values remain finite (no NaN or Inf).
assert jnp.all(jnp.isfinite(pad_out))
assert jnp.allclose(pad_out[:,0], jnp.array([2.,3.,0.,0.]))

# Reference practice: Detect packed-example leakage
# Detect packed-example leakage (Transfer / diagnosis): The earlier values 2 and 4 belong to another example; they...
assert jnp.allclose(out[2,0], 14/3)
assert jnp.allclose(packed_out[2,0], 8.)
assert not jnp.allclose(out[2,0], packed_out[2,0])
print("PASS: transformers-02")
