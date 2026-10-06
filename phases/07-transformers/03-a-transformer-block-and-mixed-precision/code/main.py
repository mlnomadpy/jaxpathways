"""A Transformer block and mixed precision: worked experiments and reference solutions. CPU checks."""

# 1. Prepare the experiment
import jax
import jax.numpy as jnp
import numpy as np

# 2. Implement the mechanism
def layer_norm(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x-mean)**2, axis=-1, keepdims=True)
    return (x-mean) / jnp.sqrt(variance + 1e-5)

def init_block(key, width=8):
    keys = jax.random.split(key, 6)
    shapes = [(width,width)]*4 + [(width,2*width),(2*width,width)]
    return {name: jax.random.normal(k,s)*0.1 for name,k,s in zip(
        ["q","k","v","o","up","down"], keys, shapes)}

def block_forward(p, x):
    h = layer_norm(x)
    q, k, v = h@p["q"], h@p["k"], h@p["v"]
    scores = q@k.T / jnp.sqrt(x.shape[-1])
    allowed = jnp.arange(x.shape[0])[:,None] >= jnp.arange(x.shape[0])[None,:]
    weights = jax.nn.softmax(jnp.where(allowed, scores, -jnp.inf), axis=-1)
    residual = x + (weights@v)@p["o"]
    return residual + jax.nn.gelu(layer_norm(residual)@p["up"])@p["down"]

# 3. Run and check
p = init_block(jax.random.key(7))
x = jnp.arange(32, dtype=jnp.float32).reshape(4,8)/10
output = block_forward(p, x)
assert output.shape == x.shape
assert jnp.all(jnp.isfinite(output))
zero = jax.tree.map(jnp.zeros_like, p)
assert jnp.allclose(block_forward(zero, x), x)
assert jnp.allclose(jax.jit(block_forward)(p,x), output, atol=1e-5)
print("Block shape:", output.shape)

import jax
import jax.numpy as jnp
import numpy as np

def layer_norm(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x-mean)**2, axis=-1, keepdims=True)
    return (x-mean) / jnp.sqrt(variance + 1e-5)

def init_block(key, width=8):
    keys = jax.random.split(key, 6)
    shapes = [(width,width)]*4 + [(width,2*width),(2*width,width)]
    return {name: jax.random.normal(k,s)*0.1 for name,k,s in zip(
        ["q","k","v","o","up","down"], keys, shapes)}

def block_forward(p, x):
    h = layer_norm(x)
    q, k, v = h@p["q"], h@p["k"], h@p["v"]
    scores = q@k.T / jnp.sqrt(x.shape[-1])
    allowed = jnp.arange(x.shape[0])[:,None] >= jnp.arange(x.shape[0])[None,:]
    weights = jax.nn.softmax(jnp.where(allowed, scores, -jnp.inf), axis=-1)
    residual = x + (weights@v)@p["o"]
    return residual + jax.nn.gelu(layer_norm(residual)@p["up"])@p["down"]

p = init_block(jax.random.key(7))
x = jnp.arange(32, dtype=jnp.float32).reshape(4,8)/10
output = block_forward(p, x)
assert output.shape == x.shape
assert jnp.all(jnp.isfinite(output))
zero = jax.tree.map(jnp.zeros_like, p)
assert jnp.allclose(block_forward(zero, x), x)
assert jnp.allclose(jax.jit(block_forward)(p,x), output, atol=1e-5)
print("Block shape:", output.shape)

# Figure data experiment
visual_data = {'kind': 'heatmap', 'values': (output - x).tolist(), 'rows': ['token ' + str(i) for i in range(4)], 'columns': ['f' + str(i) for i in range(8)], 'unit': 'output minus input', 'diverging': True}

# Experiment: Verify the residual identity and causal boundary
changed_x = x.at[-1].add(100.)
changed_output = block_forward(p, changed_x)
assert jnp.allclose(changed_output[:3], output[:3], atol=1e-5)
assert not jnp.allclose(changed_output[-1], output[-1])

# Experiment: Compare two precision policies
low_p = jax.tree.map(lambda z:z.astype(jnp.bfloat16), p)
low_x = x.astype(jnp.bfloat16)
low_out = block_forward(low_p, low_x).astype(jnp.float32)
error = jnp.max(jnp.abs(low_out-output))
print("bfloat16 max absolute error:", float(error))
assert jnp.all(jnp.isfinite(low_out))
assert error < 0.1

def mixed_block(p, x):
    h = layer_norm(x.astype(jnp.float32)).astype(jnp.bfloat16)
    q,k,v = h@p["q"], h@p["k"], h@p["v"]
    scores = (q@k.T).astype(jnp.float32)/jnp.sqrt(x.shape[-1])
    allowed = jnp.arange(x.shape[0])[:,None] >= jnp.arange(x.shape[0])[None,:]
    weights = jax.nn.softmax(jnp.where(allowed,scores,-jnp.inf),axis=-1).astype(jnp.bfloat16)
    residual = x+(weights@v)@p["o"]
    normalized = layer_norm(residual.astype(jnp.float32)).astype(jnp.bfloat16)
    activation = jax.nn.gelu((normalized@p["up"]).astype(jnp.float32)).astype(jnp.bfloat16)
    return residual + activation@p["down"]

mixed_out = mixed_block(low_p,low_x).astype(jnp.float32)
mixed_error = jnp.max(jnp.abs(mixed_out-output))
print("Float32 reduction policy max absolute error:",float(mixed_error))
assert jnp.all(jnp.isfinite(mixed_out))
assert mixed_error < 0.1

# Reference solution. Try the exercise before reading this.
def objective(params):
    return jnp.mean(block_forward(params,x)**2)
g = jax.grad(objective)(p)
assert all(a.shape==b.shape and jnp.all(jnp.isfinite(b)) for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(g)))
def shifted_objective(delta):
    altered = {**p, "down": p["down"].at[0,0].add(delta)}
    return objective(altered)
fd = (shifted_objective(1e-2)-shifted_objective(-1e-2))/(2e-2)
assert jnp.allclose(fd,g["down"][0,0],atol=1e-4,rtol=2e-2)

# Reference practice: Check normalization without autodiff
row = np.asarray(x[0])
reference = (row-row.mean())/np.sqrt(np.mean((row-row.mean())**2)+1e-5)
np.testing.assert_allclose(layer_norm(x)[0],reference,atol=1e-6)
assert jnp.allclose(layer_norm(jnp.ones((2,8))),0.)

# Reference practice: Repair a residual width mismatch
expanded = jax.nn.gelu(layer_norm(x)@p["up"])
assert expanded.shape == (4,16)
try:
    _ = x + expanded
except TypeError:
    pass
else:
    raise AssertionError("expected width mismatch")
assert (x + expanded@p["down"]).shape == x.shape
print("PASS: transformers-03")
