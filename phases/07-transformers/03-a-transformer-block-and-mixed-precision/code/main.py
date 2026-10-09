"""A Transformer block and mixed precision: worked experiments and reference solutions. CPU checks."""

# 1. Prepare the experiment
# Step 1 — 1. Prepare the experiment: These imports provide JAX array transformations and NumPy...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# 2. Implement the mechanism
# Step 2 — 2. Implement the mechanism: x → norm → Q/K/V → causal attention → output projection...
def layer_norm(x):
    # Reduce along axis=-1 to compute `mean`.
    mean = jnp.mean(x, axis=-1, keepdims=True)
    # Reduce along axis=-1 to compute `variance`.
    variance = jnp.mean((x-mean)**2, axis=-1, keepdims=True)
    # Return `(x - mean) / jnp.sqrt(variance + 1e-05)` to the caller.
    return (x-mean) / jnp.sqrt(variance + 1e-5)

# Function `init_block(key, width)` implementing this stage's computation:
def init_block(key, width=8):
    # Create or split explicit PRNG key(s) (`keys`) for reproducible randomness.
    keys = jax.random.split(key, 6)
    # Evaluate `shapes` from the current inputs and state.
    shapes = [(width,width)]*4 + [(width,2*width),(2*width,width)]
    # Return `{name: jax.random.normal(k, s) * 0.1 for name, k, s in zip(['q', 'k', 'v', 'o', 'up', 'down'], keys, shapes)}` to the caller.
    return {name: jax.random.normal(k,s)*0.1 for name,k,s in zip(
        ["q","k","v","o","up","down"], keys, shapes)}

# Function `block_forward(p, x)` implementing this stage's computation:
def block_forward(p, x):
    # Run `layer_norm` to compute `h`.
    h = layer_norm(x)
    # Perform matrix contraction / projection to compute `(q, k, v)`.
    q, k, v = h@p["q"], h@p["k"], h@p["v"]
    # Perform matrix contraction / projection to compute `scores`.
    scores = q@k.T / jnp.sqrt(x.shape[-1])
    # Create evenly spaced index values in `allowed`.
    allowed = jnp.arange(x.shape[0])[:,None] >= jnp.arange(x.shape[0])[None,:]
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights = jax.nn.softmax(jnp.where(allowed, scores, -jnp.inf), axis=-1)
    # Perform matrix contraction / projection to compute `residual`.
    residual = x + (weights@v)@p["o"]
    # Return `residual + jax.nn.gelu(layer_norm(residual) @ p['up']) @ p['down']` to the caller.
    return residual + jax.nn.gelu(layer_norm(residual)@p["up"])@p["down"]

# 3. Run and check
# Step 3 — 3. Run and check: Output shape (4, 8); finite values, zero-projection identity,...
# Create or split explicit PRNG key(s) (`p`) for reproducible randomness.
p = init_block(jax.random.key(7))
# Construct and reshape `x` into the target tensor dimensions.
x = jnp.arange(32, dtype=jnp.float32).reshape(4,8)/10
# Run `block_forward` to compute `output`.
output = block_forward(p, x)
# Verify that the output tensor shape matches our prediction.
assert output.shape == x.shape
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.all(jnp.isfinite(output))
# Transform every leaf of the parameter PyTree (`zero`).
zero = jax.tree.map(jnp.zeros_like, p)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(block_forward(zero, x), x)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jax.jit(block_forward)(p,x), output, atol=1e-5)
# Print the observed values to compare against the expected result.
print("Block shape:", output.shape)

# Step 1 — 1. Prepare the experiment: These imports provide JAX array transformations and NumPy...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Step 2 — 2. Implement the mechanism: x → norm → Q/K/V → causal attention → output projection...
def layer_norm(x):
    # Reduce along axis=-1 to compute `mean`.
    mean = jnp.mean(x, axis=-1, keepdims=True)
    # Reduce along axis=-1 to compute `variance`.
    variance = jnp.mean((x-mean)**2, axis=-1, keepdims=True)
    # Return `(x - mean) / jnp.sqrt(variance + 1e-05)` to the caller.
    return (x-mean) / jnp.sqrt(variance + 1e-5)

# Function `init_block(key, width)` implementing this stage's computation:
def init_block(key, width=8):
    # Create or split explicit PRNG key(s) (`keys`) for reproducible randomness.
    keys = jax.random.split(key, 6)
    # Evaluate `shapes` from the current inputs and state.
    shapes = [(width,width)]*4 + [(width,2*width),(2*width,width)]
    # Return `{name: jax.random.normal(k, s) * 0.1 for name, k, s in zip(['q', 'k', 'v', 'o', 'up', 'down'], keys, shapes)}` to the caller.
    return {name: jax.random.normal(k,s)*0.1 for name,k,s in zip(
        ["q","k","v","o","up","down"], keys, shapes)}

# Function `block_forward(p, x)` implementing this stage's computation:
def block_forward(p, x):
    # Run `layer_norm` to compute `h`.
    h = layer_norm(x)
    # Perform matrix contraction / projection to compute `(q, k, v)`.
    q, k, v = h@p["q"], h@p["k"], h@p["v"]
    # Perform matrix contraction / projection to compute `scores`.
    scores = q@k.T / jnp.sqrt(x.shape[-1])
    # Create evenly spaced index values in `allowed`.
    allowed = jnp.arange(x.shape[0])[:,None] >= jnp.arange(x.shape[0])[None,:]
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights = jax.nn.softmax(jnp.where(allowed, scores, -jnp.inf), axis=-1)
    # Perform matrix contraction / projection to compute `residual`.
    residual = x + (weights@v)@p["o"]
    # Return `residual + jax.nn.gelu(layer_norm(residual) @ p['up']) @ p['down']` to the caller.
    return residual + jax.nn.gelu(layer_norm(residual)@p["up"])@p["down"]

# Step 3 — 3. Run and check: Output shape (4, 8); finite values, zero-projection identity,...
# Create or split explicit PRNG key(s) (`p`) for reproducible randomness.
p = init_block(jax.random.key(7))
# Construct and reshape `x` into the target tensor dimensions.
x = jnp.arange(32, dtype=jnp.float32).reshape(4,8)/10
# Run `block_forward` to compute `output`.
output = block_forward(p, x)
# Verify that the output tensor shape matches our prediction.
assert output.shape == x.shape
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.all(jnp.isfinite(output))
# Transform every leaf of the parameter PyTree (`zero`).
zero = jax.tree.map(jnp.zeros_like, p)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(block_forward(zero, x), x)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jax.jit(block_forward)(p,x), output, atol=1e-5)
# Print the observed values to compare against the expected result.
print("Block shape:", output.shape)

# Figure data experiment
# Compute figure data for: A residual block adds a correction to its input
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'heatmap', 'values': (output - x).tolist(), 'rows': ['token ' + str(i) for i in range(4)], 'columns': ['f' + str(i) for i in range(8)], 'unit': 'output minus input', 'diverging': True}

# Experiment: Verify the residual identity and causal boundary
# Experiment — Verify the residual identity and causal boundary: Masking and per-token operations preserve causal independence...
changed_x = x.at[-1].add(100.)
# Run `block_forward` to compute `changed_output`.
changed_output = block_forward(p, changed_x)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(changed_output[:3], output[:3], atol=1e-5)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not jnp.allclose(changed_output[-1], output[-1])

# Experiment: Compare two precision policies
# Experiment — Compare two precision policies: Storage/activation casts and sensitive-reduction precision are...
# Transform every leaf of the parameter PyTree (`low_p`).
low_p = jax.tree.map(lambda z:z.astype(jnp.bfloat16), p)
# Cast or evaluate `low_x` in explicit floating-point precision.
low_x = x.astype(jnp.bfloat16)
# Cast or evaluate `low_out` in explicit floating-point precision.
low_out = block_forward(low_p, low_x).astype(jnp.float32)
# Aggregate array values to compute `error`.
error = jnp.max(jnp.abs(low_out-output))
# Print the observed values to compare against the expected result.
print("bfloat16 max absolute error:", float(error))
# Confirm that all computed values remain finite (no NaN or Inf).
assert jnp.all(jnp.isfinite(low_out))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert error < 0.1

# Function `mixed_block(p, x)` implementing this stage's computation:
def mixed_block(p, x):
    # Cast or evaluate `h` in explicit floating-point precision.
    h = layer_norm(x.astype(jnp.float32)).astype(jnp.bfloat16)
    # Perform matrix contraction / projection to compute `(q, k, v)`.
    q,k,v = h@p["q"], h@p["k"], h@p["v"]
    # Cast or evaluate `scores` in explicit floating-point precision.
    scores = (q@k.T).astype(jnp.float32)/jnp.sqrt(x.shape[-1])
    # Create evenly spaced index values in `allowed`.
    allowed = jnp.arange(x.shape[0])[:,None] >= jnp.arange(x.shape[0])[None,:]
    # Cast or evaluate `weights` in explicit floating-point precision.
    weights = jax.nn.softmax(jnp.where(allowed,scores,-jnp.inf),axis=-1).astype(jnp.bfloat16)
    # Perform matrix contraction / projection to compute `residual`.
    residual = x+(weights@v)@p["o"]
    # Cast or evaluate `normalized` in explicit floating-point precision.
    normalized = layer_norm(residual.astype(jnp.float32)).astype(jnp.bfloat16)
    # Cast or evaluate `activation` in explicit floating-point precision.
    activation = jax.nn.gelu((normalized@p["up"]).astype(jnp.float32)).astype(jnp.bfloat16)
    # Return `residual + activation @ p['down']` to the caller.
    return residual + activation@p["down"]

# Cast or evaluate `mixed_out` in explicit floating-point precision.
mixed_out = mixed_block(low_p,low_x).astype(jnp.float32)
# Aggregate array values to compute `mixed_error`.
mixed_error = jnp.max(jnp.abs(mixed_out-output))
# Print the observed values to compare against the expected result.
print("Float32 reduction policy max absolute error:",float(mixed_error))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.all(jnp.isfinite(mixed_out))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert mixed_error < 0.1

# Reference solution. Try the exercise before reading this.
# Exercise solution: Differentiate the mean squared block output with respect to every...
def objective(params):
    # Return `jnp.mean(block_forward(params, x) ** 2)` to the caller.
    return jnp.mean(block_forward(params,x)**2)
# Differentiate the objective to obtain `g` via automatic differentiation.
g = jax.grad(objective)(p)
# Verify that the output tensor shape matches our prediction.
assert all(a.shape==b.shape and jnp.all(jnp.isfinite(b)) for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(g)))
# Function `shifted_objective(delta)` implementing this stage's computation:
def shifted_objective(delta):
    # Evaluate `altered` from the current inputs and state.
    altered = {**p, "down": p["down"].at[0,0].add(delta)}
    # Return `objective(altered)` to the caller.
    return objective(altered)
# Evaluate `fd` from the current inputs and state.
fd = (shifted_objective(1e-2)-shifted_objective(-1e-2))/(2e-2)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(fd,g["down"][0,0],atol=1e-4,rtol=2e-2)

# Reference practice: Check normalization without autodiff
# Check normalization without autodiff (Transfer / diagnosis): The independent reduction and constant-row case check the...
row = np.asarray(x[0])
# Aggregate array values to compute `reference`.
reference = (row-row.mean())/np.sqrt(np.mean((row-row.mean())**2)+1e-5)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(layer_norm(x)[0],reference,atol=1e-6)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(layer_norm(jnp.ones((2,8))),0.)

# Reference practice: Repair a residual width mismatch
# Repair a residual width mismatch (Transfer / diagnosis): Residual paths require matching widths; the down projection...
expanded = jax.nn.gelu(layer_norm(x)@p["up"])
# Verify that the output tensor shape matches our prediction.
assert expanded.shape == (4,16)
# Run the boundary check and catch the expected exception:
try:
    _ = x + expanded
except TypeError:
    pass
else:
    raise AssertionError("expected width mismatch")
# Verify that the output tensor shape matches our prediction.
assert (x + expanded@p["down"]).shape == x.shape
print("PASS: transformers-03")
