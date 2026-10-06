"""Masking, tokenization, and sequence packing: worked experiments and reference solutions. CPU checks."""

# 1. Prepare the experiment
import jax
import jax.numpy as jnp
import numpy as np

# 2. Implement the mechanism
def masked_mean(values, allowed):
    scores = jnp.where(allowed, 0., -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ values, weights

def make_mask(segment_ids, valid):
    positions = jnp.arange(segment_ids.shape[0])
    causal = positions[:, None] >= positions[None, :]
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    return causal & same_segment & valid[:, None] & valid[None, :]

# 3. Run and check
values = jnp.array([[2.], [4.], [8.], [16.]])
causal = jnp.arange(4)[:, None] >= jnp.arange(4)[None, :]
out, weights = masked_mean(values, causal)
assert jnp.allclose(out[:, 0], jnp.array([2., 3., 14/3, 7.5]))
assert jnp.allclose(jnp.triu(weights, 1), 0.)
print("Causal means:", out[:, 0])

import jax
import jax.numpy as jnp
import numpy as np

def masked_mean(values, allowed):
    scores = jnp.where(allowed, 0., -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ values, weights

def make_mask(segment_ids, valid):
    positions = jnp.arange(segment_ids.shape[0])
    causal = positions[:, None] >= positions[None, :]
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    return causal & same_segment & valid[:, None] & valid[None, :]

values = jnp.array([[2.], [4.], [8.], [16.]])
causal = jnp.arange(4)[:, None] >= jnp.arange(4)[None, :]
out, weights = masked_mean(values, causal)
assert jnp.allclose(out[:, 0], jnp.array([2., 3., 14/3, 7.5]))
assert jnp.allclose(jnp.triu(weights, 1), 0.)
print("Causal means:", out[:, 0])

# Figure data experiment
visual_data = {'kind': 'heatmap', 'values': weights.tolist(), 'rows': ['query ' + str(i) for i in range(4)], 'columns': ['key ' + str(i) for i in range(4)], 'unit': 'attention weight'}

# Experiment: Perturb the future
changed = values.at[-1, 0].set(1000.)
changed_out, _ = masked_mean(changed, causal)
assert jnp.allclose(changed_out[:3], out[:3])
assert not jnp.allclose(changed_out[-1], out[-1])

# Experiment: Pack two independent examples
segments = jnp.array([0, 0, 1, 1])
valid = jnp.ones(4, dtype=bool)
packed = make_mask(segments, valid)
packed_out, _ = masked_mean(values, packed)
assert jnp.allclose(packed_out[:, 0], jnp.array([2., 3., 8., 12.]))

# Reference solution. Try the exercise before reading this.
vocab = {"a":0, "b":1, "c":2}
ids = jnp.array([vocab[c] for c in "abcab"])
assert jnp.array_equal(ids[:-1], jnp.array([0,1,2,0]))
assert jnp.array_equal(ids[1:], jnp.array([1,2,0,1]))
seg = jnp.array([0,0,0,1,1])
loss_valid = seg[:-1] == seg[1:]
assert jnp.array_equal(loss_valid, jnp.array([True,True,False,True]))

# Reference practice: Repair padding-query NaNs
pad_valid = jnp.array([True,True,False,False])
pad_mask = make_mask(jnp.zeros(4, dtype=int), pad_valid)
bad_pad, _ = masked_mean(values, pad_mask)
assert not jnp.all(jnp.isfinite(bad_pad))
fallback = (~pad_valid[:, None]) & jnp.eye(4, dtype=bool)
pad_out, _ = masked_mean(values, pad_mask | fallback)
pad_out = jnp.where(pad_valid[:, None], pad_out, 0.)
assert jnp.all(jnp.isfinite(pad_out))
assert jnp.allclose(pad_out[:,0], jnp.array([2.,3.,0.,0.]))

# Reference practice: Detect packed-example leakage
assert jnp.allclose(out[2,0], 14/3)
assert jnp.allclose(packed_out[2,0], 8.)
assert not jnp.allclose(out[2,0], packed_out[2,0])
print("PASS: transformers-02")
