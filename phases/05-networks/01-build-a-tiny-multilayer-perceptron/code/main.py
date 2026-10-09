"""Build a tiny multilayer perceptron: worked experiments and reference solutions. CPU checks."""

# 1. Define the fixture and parameter tree
# Step 1 — 1. Define the fixture and parameter tree: The parameter count is 16+8+8+1=33.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np
import optax
# Initialize array `x` with explicit values and shape.
x = jnp.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
# Initialize array `y` with explicit values and shape.
y = jnp.array([0.,1.,1.,0.])
# Create or split explicit PRNG key(s) (`(k1, k2)`) for reproducible randomness.
k1, k2 = jax.random.split(jax.random.key(0))
# Sample deterministic random values into `params` using an explicit PRNG key.
params = {'w1':jax.random.normal(k1,(2,8))*.4, 'b1':jnp.zeros(8),
          'w2':jax.random.normal(k2,(8,1))*.4, 'b2':jnp.zeros(1)}
# Verify contract: `sum((a.size for a in jax.tree.leaves(params))) == 33`.
assert sum(a.size for a in jax.tree.leaves(params)) == 33

# 2. Write prediction and verify the loss
# Step 2 — 2. Write prediction and verify the loss: The zero-score baseline is about 0.693147.
def logits(p, batch):
    # Return `(jnp.tanh(batch @ p['w1'] + p['b1']) @ p['w2'] + p['b2']).squeeze(-1)` to the caller.
    return (jnp.tanh(batch@p['w1']+p['b1'])@p['w2']+p['b2']).squeeze(-1)
# Function `loss(p, batch, labels)` implementing this stage's computation:
def loss(p, batch, labels):
    # Return `jnp.mean(optax.sigmoid_binary_cross_entropy(logits(p, batch), labels))` to the caller.
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits(p,batch),labels))
# Convert `scores` to a host NumPy array for inspection or verification.
scores = np.asarray(logits(params,x))
# Aggregate array values to compute `expected`.
expected = np.mean(np.logaddexp(0.,scores)-np.asarray(y)*scores)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(loss(params,x,y), expected, rtol=1e-6)
# Transform every leaf of the parameter PyTree (`zeros`).
zeros = jax.tree.map(jnp.zeros_like, params)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(loss(zeros,x,y),np.log(2.),rtol=1e-6)
# Verify that the output tensor shape matches our prediction.
assert logits(params,jnp.ones((1,2))).shape == (1,)

# 3. Train and evaluate untouched points
# Step 3 — 3. Train and evaluate untouched points: The independent held-out seed changes input coordinates.
# Configure or step the Optax optimizer state (`tx`).
tx = optax.adam(.03)
# Run `tx.init` to compute `state`.
state = tx.init(params)
# Define and JIT-compile `step(p, s)` so XLA traces and fuses the operations:
@jax.jit
# Function `step(p, s)` implementing this stage's computation:
def step(p,s):
    # Differentiate the objective to obtain `(value, grads)` via automatic differentiation.
    value,grads = jax.value_and_grad(loss)(p,x,y)
    # Run `tx.update` to compute `(updates, s)`.
    updates,s = tx.update(grads,s,p)
    # Return `(optax.apply_updates(p, updates), s, value)` to the caller.
    return optax.apply_updates(p,updates),s,value
# Evaluate `loss(params, x, y)` and convert the result into Python scalar/collection `initial`.
initial = float(loss(params,x,y))
# Repeat the update loop over `range(200)` steps:
# Run `step` to compute `(params, state, _)`.
for _ in range(200): params,state,_ = step(params,state)
# Draw pseudorandom samples for `rng` using the explicit RNG state.
rng = np.random.default_rng(12)
# Convert `held_x` to a host NumPy array for inspection or verification.
held_x = np.repeat(np.asarray(x),8,axis=0)+rng.normal(0,.12,(32,2))
# Cast or evaluate `held_y` in explicit floating-point precision.
held_y = (held_x[:,0]*held_x[:,1]<0).astype(np.float32)
# Initialize array `held_scores` with explicit values and shape.
held_scores = np.asarray(logits(params,jnp.array(held_x)))
# Aggregate array values to compute `accuracy`.
accuracy = np.mean((held_scores>0)==held_y)
# Evaluate `loss(params, x, y)` and convert the result into Python scalar/collection `final`.
final = float(loss(params,x,y))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert final < .03 and accuracy >= .95
# Print diagnostic summary of the computed outputs.
print('Training loss:', initial, '->', final, 'held-out accuracy:', accuracy)

# Step 1 — 1. Define the fixture and parameter tree: The parameter count is 16+8+8+1=33.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np
import optax
# Initialize array `x` with explicit values and shape.
x = jnp.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
# Initialize array `y` with explicit values and shape.
y = jnp.array([0.,1.,1.,0.])
# Create or split explicit PRNG key(s) (`(k1, k2)`) for reproducible randomness.
k1, k2 = jax.random.split(jax.random.key(0))
# Sample deterministic random values into `params` using an explicit PRNG key.
params = {'w1':jax.random.normal(k1,(2,8))*.4, 'b1':jnp.zeros(8),
          'w2':jax.random.normal(k2,(8,1))*.4, 'b2':jnp.zeros(1)}
# Verify contract: `sum((a.size for a in jax.tree.leaves(params))) == 33`.
assert sum(a.size for a in jax.tree.leaves(params)) == 33

# Step 2 — 2. Write prediction and verify the loss: The zero-score baseline is about 0.693147.
def logits(p, batch):
    # Return `(jnp.tanh(batch @ p['w1'] + p['b1']) @ p['w2'] + p['b2']).squeeze(-1)` to the caller.
    return (jnp.tanh(batch@p['w1']+p['b1'])@p['w2']+p['b2']).squeeze(-1)
# Function `loss(p, batch, labels)` implementing this stage's computation:
def loss(p, batch, labels):
    # Return `jnp.mean(optax.sigmoid_binary_cross_entropy(logits(p, batch), labels))` to the caller.
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits(p,batch),labels))
# Convert `scores` to a host NumPy array for inspection or verification.
scores = np.asarray(logits(params,x))
# Aggregate array values to compute `expected`.
expected = np.mean(np.logaddexp(0.,scores)-np.asarray(y)*scores)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(loss(params,x,y), expected, rtol=1e-6)
# Transform every leaf of the parameter PyTree (`zeros`).
zeros = jax.tree.map(jnp.zeros_like, params)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(loss(zeros,x,y),np.log(2.),rtol=1e-6)
# Verify that the output tensor shape matches our prediction.
assert logits(params,jnp.ones((1,2))).shape == (1,)

# Step 3 — 3. Train and evaluate untouched points: The independent held-out seed changes input coordinates.
# Configure or step the Optax optimizer state (`tx`).
tx = optax.adam(.03)
# Run `tx.init` to compute `state`.
state = tx.init(params)
# Define and JIT-compile `step(p, s)` so XLA traces and fuses the operations:
@jax.jit
# Function `step(p, s)` implementing this stage's computation:
def step(p,s):
    # Differentiate the objective to obtain `(value, grads)` via automatic differentiation.
    value,grads = jax.value_and_grad(loss)(p,x,y)
    # Run `tx.update` to compute `(updates, s)`.
    updates,s = tx.update(grads,s,p)
    # Return `(optax.apply_updates(p, updates), s, value)` to the caller.
    return optax.apply_updates(p,updates),s,value
# Evaluate `loss(params, x, y)` and convert the result into Python scalar/collection `initial`.
initial = float(loss(params,x,y))
# Repeat the update loop over `range(200)` steps:
# Run `step` to compute `(params, state, _)`.
for _ in range(200): params,state,_ = step(params,state)
# Draw pseudorandom samples for `rng` using the explicit RNG state.
rng = np.random.default_rng(12)
# Convert `held_x` to a host NumPy array for inspection or verification.
held_x = np.repeat(np.asarray(x),8,axis=0)+rng.normal(0,.12,(32,2))
# Cast or evaluate `held_y` in explicit floating-point precision.
held_y = (held_x[:,0]*held_x[:,1]<0).astype(np.float32)
# Initialize array `held_scores` with explicit values and shape.
held_scores = np.asarray(logits(params,jnp.array(held_x)))
# Aggregate array values to compute `accuracy`.
accuracy = np.mean((held_scores>0)==held_y)
# Evaluate `loss(params, x, y)` and convert the result into Python scalar/collection `final`.
final = float(loss(params,x,y))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert final < .03 and accuracy >= .95
# Print diagnostic summary of the computed outputs.
print('Training loss:', initial, '->', final, 'held-out accuracy:', accuracy)

# Figure data experiment
# Compute figure data for: A tiny MLP separates the XOR regions
# Generate a uniform grid of points in `axis`.
axis = jnp.linspace(-1.6, 1.6, 61)
# Run `jnp.meshgrid` to compute `(gx, gy)`.
gx, gy = jnp.meshgrid(axis, axis)
# Combine or mask array elements to form `grid`.
grid = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)
# Rearrange tensor axes to match the required layout for `prob`.
prob = jax.nn.sigmoid(logits(params, grid)).reshape(gx.shape)
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'field', 'values': prob.tolist(), 'extent': [-1.6, 1.6, -1.6, 1.6], 'xlabel': 'feature 0', 'ylabel': 'feature 1', 'unit': 'P(label 1)', 'points': x.tolist(), 'labels': y.tolist()}

# Experiment: Duplication preserves a mean objective
# Experiment — Duplication preserves a mean objective: Averaging normalizes repeated identical data.
repeated_x=jnp.repeat(x,3,axis=0);repeated_y=jnp.repeat(y,3)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(loss(params,repeated_x,repeated_y),loss(params,x,y),rtol=1e-5)
# Differentiate the objective to obtain `original_g` via automatic differentiation.
original_g=jax.grad(loss)(params,x,y);repeated_g=jax.grad(loss)(params,repeated_x,repeated_y)
# Iterate over `(a, b)` to step through the computation:
for a,b in zip(jax.tree.leaves(original_g),jax.tree.leaves(repeated_g)):
    # Verify that computed values match the expected reference within numerical tolerance.
    np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-7)

# Experiment: An affine stack stays affine
# Experiment — An affine stack stays affine: This exact affine identity explains why removing the...
def affine_stack(batch):return (batch@params['w1']+params['b1'])@params['w2']+params['b2']
# Initialize array `midpoint` with explicit values and shape.
midpoint=jnp.array([[.2,-.3]]);delta=jnp.array([[.4,.1]])
# Combine or mask array elements to form ``.
np.testing.assert_allclose(affine_stack(midpoint),.5*(affine_stack(midpoint+delta)+affine_stack(midpoint-delta)),atol=1e-6)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Swap class zero and one.
complemented=dict(params,w2=-params['w2'],b2=-params['b2'])
# Initialize array `changed_x` with explicit values and shape.
changed_x=jnp.array([[-.8,.7],[.7,.9],[-1.1,-.8]])
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(logits(complemented,changed_x),-logits(params,changed_x),atol=1e-6)
# Convert `expected_labels` to a host NumPy array for inspection or verification.
expected_labels=(np.asarray(changed_x)[:,0]*np.asarray(changed_x)[:,1]>0)
# Verify contract: `np.array_equal(np.asarray(logits(complemented, changed_x) > 0), expe...`.
assert np.array_equal(np.asarray(logits(complemented,changed_x)>0),expected_labels)

# Reference practice: Reorder hidden units without changing predictions
# Reorder hidden units without changing predictions (Transfer): A hidden unit has no intrinsic index.
# Initialize array `order` with explicit values and shape.
order=jnp.array([7,0,6,1,5,2,4,3])
# Initialize array `probe` with explicit values and shape.
probe=jnp.array([[.2,-.9],[-.4,.6],[.8,.1]])
# Evaluate `params, w1=params['w1'][:, order], b1=params['b1'][order], w2=params['w2'][order, :]` and convert the result into Python scalar/collection `permuted`.
permuted=dict(params,w1=params['w1'][:,order],b1=params['b1'][order],w2=params['w2'][order,:])
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(logits(permuted,probe),logits(params,probe),rtol=1e-5,atol=1e-5)
# Evaluate `params, w1=params['w1'][:, order], b1=params['b1'][order]` and convert the result into Python scalar/collection `broken`.
broken=dict(params,w1=params['w1'][:,order],b1=params['b1'][order])
# Verify that the numerical values match the expected reference within tolerance.
assert not np.allclose(logits(broken,probe),logits(params,probe),atol=1e-5)
# Print the observed values to compare against the expected result.
print('Consistent hidden permutation preserves logits; one-sided permutation changes them.')

# Reference practice: Diagnose accidental pairwise broadcasting
# Diagnose accidental pairwise broadcasting (Intermediate): The incorrect scalar still looks plausible, but it compares...
# Configure or step the Optax optimizer state (`bad`).
bad=optax.sigmoid_binary_cross_entropy(logits(params,x),y[:,None])
# Verify that the output tensor shape matches our prediction.
assert bad.shape==(4,4)
# Configure or step the Optax optimizer state (`good`).
good=optax.sigmoid_binary_cross_entropy(logits(params,x),y)
# Verify that the output tensor shape matches our prediction.
assert good.shape==(4,)
# Reduce across the target axis to summarize ``.
np.testing.assert_allclose(jnp.mean(good),loss(params,x,y),rtol=1e-6)
print("PASS: networks-01")
