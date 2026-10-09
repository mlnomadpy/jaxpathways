# Build a tiny multilayer perceptron

Phase 05: Neural network training · about 85 minutes · CPU

## What you will be able to do

- Trace the shapes from observations to logits
- Explain why the hidden activation matters
- Use stable binary cross-entropy
- Separate learning from held-out measurement

## The problem

Four points can give us a useful puzzle: how do we separate the two diagonal pairs in XOR? A single straight line cannot do it. We’ll add a small hidden layer and a nonlinear activation, check the shapes, and train a classifier. Then we’ll test nearby points to see what it has learned beyond the four training examples.

## The idea

A multilayer perceptron combines learned linear maps with nonlinear activations. The hidden layer builds features that the output layer can combine. The activation lets the composition represent boundaries that a single linear map cannot.

## Why hidden layers need nonlinear operations

XOR gives the same label to opposite corners of a square. One straight separating line cannot isolate both opposite corners from the other two. Hidden units can form intermediate regions that an output layer combines.

Remove every nonlinear activation and the weight matrices simply multiply into one linear map. More matrices alone do not solve this geometry. Biases change offsets but do not remove the limitation.

Read the decision field both at and between the training points. The four observed labels constrain only a tiny part of the plane. A smooth-looking boundary is still an extrapolation. Connect each layer's actual width to the feature vector it produces before interpreting that extrapolation.

### Pause and reason

Does fitting all four XOR points establish generalization to another dataset?

<details><summary>Compare your reasoning</summary>

No. It checks representation and optimization on this teaching fixture. A different population requires its own held-out evaluation.

</details>

## Trace the shapes from observations to logits

Each row is one observation with two features. The first weight matrix has shape $(2, 8)$, so a batch $(B, 2)$ becomes $(B, 8)$. A bias vector $(8,)$ broadcasts across rows; it does not create eight observations. tanh changes values without changing shape. The second matrix $(8, 1)$ produces one logit per row. Squeeze only the final axis to produce $(B,)$, preserving the batch dimension even when $B$ is one. A logit is an unrestricted real score; a positive logit predicts class one when the decision threshold is probability $0.5$.

```text
(B, 2) → affine → (B, 8) → tanh → (B, 8) → affine → (B, 1) → (B,)
```

## Explain why the hidden activation matters

XOR assigns matching signs to class zero and opposite signs to class one. A single line cannot place both diagonal pairs on opposite sides. Two affine layers without an activation still collapse to one affine map: ($x$ W1 + b1) W2 + b2. More parameters alone therefore do not solve this obstacle. tanh introduces curved intermediate features that can support a nonlinear boundary. Eight hidden units are deliberately small enough to inspect. They are not a principled best width, and successful training at one seed does not guarantee every initialization will converge.

## Use stable binary cross-entropy

Do not calculate $\log\sigma(z)$ by separately taking the sigmoid and logarithm. At large negative scores the probability may round to zero. The stable scalar loss is $\operatorname{logaddexp}(0,z)-yz$. Average these losses over the batch so duplicating every row does not double the objective. Labels are zero or one, logits and labels both have shape $(B,)$, and gradients are taken with respect to the parameter tree. The derivative with respect to each score is $\sigma(z)-y$, divided by batch size. We verify the implementation independently with NumPy and verify a zero-parameter example against $\log2$.

## Separate learning from held-out measurement

The four XOR corners are the complete training set. Held-out points are perturbed versions of those corners, generated from a separate fixed seed. Their labels follow the original sign rule. They check nearby transfer rather than merely memorizing four exact inputs. They do not test arbitrary extrapolation across the plane; this small fixture is especially easy. Record initial and final loss, the seed, hidden width, learning rate, update count, and held-out accuracy. Keep evaluation outside the update loop and do not select a hyperparameter by repeatedly inspecting the held-out answers.

## Read the classification loss one example at a time

The logit $z$ is the model’s raw score; the label $y$ is either $0$ or $1$. A positive logit favors class $1$. When $z=0$, the sigmoid probability is one half, so either label gives a loss of $\log 2$. That is a useful sanity check. In code, logaddexp evaluates the softplus term stably instead of separately computing an exponential and logarithm.

$$
\ell(z,y)=\underbrace{\log(1+e^z)}_{\mathrm{softplus}(z)}-yz
$$

## 1. Define the fixture and parameter tree

Create main.py and copy this block. Before running, write the four XOR labels and the expected shape of each parameter.

```python
# Step 1 — 1. Define the fixture and parameter tree: The parameter count is 16+8+8+1=33.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np
import optax
# Construct `x` via `jnp.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])`
x = jnp.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
# Construct `y` via `jnp.array([0.,1.,1.,0.])`
y = jnp.array([0.,1.,1.,0.])
# Create or split explicit PRNG key(s) (`(k1, k2)`) for reproducible randomness.
k1, k2 = jax.random.split(jax.random.key(0))
# Sample deterministic random values into `params` using an explicit PRNG key.
params = {'w1':jax.random.normal(k1,(2,8))*.4, 'b1':jnp.zeros(8),
          'w2':jax.random.normal(k2,(8,1))*.4, 'b2':jnp.zeros(1)}
# Assert invariant `sum(a.size for a in jax.tree.leaves(params)) == 33` holds
assert sum(a.size for a in jax.tree.leaves(params)) == 33
```

The parameter count is $16+8+8+1=33$. Independent keys initialize the two layers; biases start at zero.

## 2. Write prediction and verify the loss

Append these functions. The NumPy reference has no autodiff or optimizer and tests the objective itself.

```python
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
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(loss(params,x,y), expected, rtol=1e-6)`
np.testing.assert_allclose(loss(params,x,y), expected, rtol=1e-6)
# Transform every leaf of the parameter PyTree (`zeros`).
zeros = jax.tree.map(jnp.zeros_like, params)
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(loss(zeros,x,y),np.log(2.),rtol=1e-6)`
np.testing.assert_allclose(loss(zeros,x,y),np.log(2.),rtol=1e-6)
# Check tensor shape invariant: `logits(params,jnp.ones((1,2))).shape == (1,)`
assert logits(params,jnp.ones((1,2))).shape == (1,)
```

The zero-score baseline is about $0.693147$. A loss below it alone does not prove held-out skill.

## 3. Train and evaluate untouched points

Append the compiled step and loop. Prediction: can the nonlinear model beat chance on points near each training corner?

```python
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
# Construct `held_scores` via `np.asarray(logits(params,jnp.array(held_x)))`
held_scores = np.asarray(logits(params,jnp.array(held_x)))
# Aggregate array values to compute `accuracy`.
accuracy = np.mean((held_scores>0)==held_y)
# Evaluate `loss(params, x, y)` and convert the result into Python scalar/collection `final`.
final = float(loss(params,x,y))
# Assert invariant `final < .03 and accuracy >= .95` holds
assert final < .03 and accuracy >= .95
# Print diagnostic summary of the computed outputs.
print('Training loss:', initial, '->', final, 'held-out accuracy:', accuracy)
```

The independent held-out seed changes input coordinates. Passing this check establishes behavior on this fixture only.

## Run the example

```python
# Step 1 — 1. Define the fixture and parameter tree: The parameter count is 16+8+8+1=33.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np
import optax
# Construct `x` via `jnp.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])`
x = jnp.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
# Construct `y` via `jnp.array([0.,1.,1.,0.])`
y = jnp.array([0.,1.,1.,0.])
# Create or split explicit PRNG key(s) (`(k1, k2)`) for reproducible randomness.
k1, k2 = jax.random.split(jax.random.key(0))
# Sample deterministic random values into `params` using an explicit PRNG key.
params = {'w1':jax.random.normal(k1,(2,8))*.4, 'b1':jnp.zeros(8),
          'w2':jax.random.normal(k2,(8,1))*.4, 'b2':jnp.zeros(1)}
# Assert invariant `sum(a.size for a in jax.tree.leaves(params)) == 33` holds
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
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(loss(params,x,y), expected, rtol=1e-6)`
np.testing.assert_allclose(loss(params,x,y), expected, rtol=1e-6)
# Transform every leaf of the parameter PyTree (`zeros`).
zeros = jax.tree.map(jnp.zeros_like, params)
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(loss(zeros,x,y),np.log(2.),rtol=1e-6)`
np.testing.assert_allclose(loss(zeros,x,y),np.log(2.),rtol=1e-6)
# Check tensor shape invariant: `logits(params,jnp.ones((1,2))).shape == (1,)`
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
# Construct `held_scores` via `np.asarray(logits(params,jnp.array(held_x)))`
held_scores = np.asarray(logits(params,jnp.array(held_x)))
# Aggregate array values to compute `accuracy`.
accuracy = np.mean((held_scores>0)==held_y)
# Evaluate `loss(params, x, y)` and convert the result into Python scalar/collection `final`.
final = float(loss(params,x,y))
# Assert invariant `final < .03 and accuracy >= .95` holds
assert final < .03 and accuracy >= .95
# Print diagnostic summary of the computed outputs.
print('Training loss:', initial, '->', final, 'held-out accuracy:', accuracy)
```

Expected: Loss falls below $0.03$ and nearby held-out accuracy is at least $0.95$. Exact initialization and losses may vary across backend/version.

## A tiny MLP separates the XOR regions

**Predict:** Can a single straight boundary separate these labels?

![A tiny MLP separates the XOR regions](../../phases/05-networks/01-build-a-tiny-multilayer-perceptron/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The axes are the two input features. Background color shows the model’s predicted probability of label $1$: dark purple is near $1$, and the pale regions are near $0$. The markers give observed labels independently of that background: triangles are label $1$, circles are label $0$.

The triangles at $(-1,1)$ and $(1,-1)$ sit in dark regions. The circles at $(-1,-1)$ and $(1,1)$ sit in pale regions. Thus the learned pattern assigns label $1$ to opposite-sign corners and label $0$ to same-sign corners, matching XOR.

### Connect it to the computation

A single straight boundary cannot separate these alternating corners. The hidden nonlinear layer allows the model to bend the boundary and create separated regions for the same class. The transition band between light and dark marks inputs where the predicted probability changes rapidly.

Most colored locations are grid queries, not additional labeled observations. Their colors show how this trained model extends its predictions between and beyond the examples. A dark region alone is separate from calibrated confidence or accuracy there; those claims need appropriate held-out data.

```python
# Compute figure data for: A tiny MLP separates the XOR regions
# Generate a uniform grid of points in `axis`.
axis = jnp.linspace(-1.6, 1.6, 61)
# Run `jnp.meshgrid` to compute `(gx, gy)`.
gx, gy = jnp.meshgrid(axis, axis)
# Combine or mask array elements to form `grid`.
grid = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)
# Rearrange tensor axes to match the required layout for `prob`.
prob = jax.nn.sigmoid(logits(params, grid)).reshape(gx.shape)
# Compute `visual_data` from `{'kind': 'field', 'values': prob.tolist(), 'extent':...`
visual_data = {'kind': 'field', 'values': prob.tolist(), 'extent': [-1.6, 1.6, -1.6, 1.6], 'xlabel': 'feature 0', 'ylabel': 'feature 1', 'unit': 'P(label 1)', 'points': x.tolist(), 'labels': y.tolist()}
```

## Recorded reference execution

CPU run: 2026-10-08T14:02:33.082414+00:00. JAX 0.9.2.

```text
Training loss: 0.7047799825668335 -> 0.0015878621488809586 held-out accuracy: 1.0
Training loss: 0.7047799825668335 -> 0.0015878621488809586 held-out accuracy: 1.0
Consistent hidden permutation preserves logits; one-sided permutation changes them.
PASS: networks-01

```

## Duplication preserves a mean objective

**Predict before running:** If every observation is repeated three times, should the loss and gradient change?

```python
# Experiment — Duplication preserves a mean objective: Averaging normalizes repeated identical data.
repeated_x=jnp.repeat(x,3,axis=0)
repeated_y=jnp.repeat(y,3)
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(loss(params,repeated_x,repeated_y),los...`
np.testing.assert_allclose(loss(params,repeated_x,repeated_y),loss(params,x,y),rtol=1e-5)
# Differentiate the objective to obtain `original_g` via automatic differentiation.
original_g=jax.grad(loss)(params,x,y)
repeated_g=jax.grad(loss)(params,repeated_x,repeated_y)
# Iterate over `(a, b)` to step through the computation:
for a,b in zip(jax.tree.leaves(original_g),jax.tree.leaves(repeated_g)):
    # Check numerical equivalence within tolerance: `np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-7)`
    np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-7)
```

**Expected:** The losses and corresponding gradient leaves agree within tolerance.

Averaging normalizes repeated identical data. Summing would change the learning-rate interpretation.

## An affine stack stays affine

**Predict before running:** Can removing tanh create a nonlinear XOR separator merely by keeping two matrices?

```python
# Experiment — An affine stack stays affine: This exact affine identity explains why removing the...
def affine_stack(batch):return (batch@params['w1']+params['b1'])@params['w2']+params['b2']
# Construct `midpoint` via `jnp.array([[.2,-.3]])`
midpoint=jnp.array([[.2,-.3]])
delta=jnp.array([[.4,.1]])
# Combine or mask array elements to form ``.
np.testing.assert_allclose(affine_stack(midpoint),.5*(affine_stack(midpoint+delta)+affine_stack(midpoint-delta)),atol=1e-6)
```

**Expected:** The midpoint identity holds for the affine stack.

This exact affine identity explains why removing the nonlinearity limits the class of functions, regardless of width.

## Make it yours

Swap class zero and one. Transform only the trained output layer to preserve the boundary and verify predictions on new coordinates.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Construct `changed_x` via `jnp.array([[-.8,.7],[.7,.9],[-1.1,-.8]])`
2. Check numerical equivalence within tolerance: `np.testing.assert_allclose(logits(complemented,changed_x),-logits...`
3. Convert `expected_labels` to a host NumPy array for inspection or verification.
4. Assert invariant `np.array_equal(np.asarray(logits(complemented,changed_x)>0),expec...` holds

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Swap class zero and one.
complemented = dict(...)  # TODO: compute complemented
# Construct `changed_x` via `jnp.array([[-.8,.7],[.7,.9],[-1.1,-.8]])`
changed_x = jnp.array(...)  # TODO: compute changed_x
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(logits(complemented,changed_x),-logits...`
np.testing.assert_allclose(logits(complemented,changed_x),-logits(params,changed_x),atol=1e-6)
# Convert `expected_labels` to a host NumPy array for inspection or verification.
expected_labels = ...  # TODO: compute expected_labels
# Assert invariant `np.array_equal(np.asarray(logits(complemented,changed_x)>0),expec...` holds
assert np.array_equal(np.asarray(logits(complemented,changed_x)  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Swap class zero and one.
complemented=dict(params,w2=-params['w2'],b2=-params['b2'])
# Construct `changed_x` via `jnp.array([[-.8,.7],[.7,.9],[-1.1,-.8]])`
changed_x=jnp.array([[-.8,.7],[.7,.9],[-1.1,-.8]])
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(logits(complemented,changed_x),-logits...`
np.testing.assert_allclose(logits(complemented,changed_x),-logits(params,changed_x),atol=1e-6)
# Convert `expected_labels` to a host NumPy array for inspection or verification.
expected_labels=(np.asarray(changed_x)[:,0]*np.asarray(changed_x)[:,1]>0)
# Assert invariant `np.array_equal(np.asarray(logits(complemented,changed_x)>0),expec...` holds
assert np.array_equal(np.asarray(logits(complemented,changed_x)>0),expected_labels)
```

</details>

## Reorder hidden units without changing predictions

**Transfer**

Permute the hidden units and make the matching change to the output layer. Predict whether the logits change. Test the result on new coordinates, then show why permuting only one side breaks the contract.

<details><summary>Hint</summary>

Columns of the first kernel and rows of the second kernel describe the same hidden units.

</details>

### How to write: Reorder hidden units without changing predictions — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Construct `order` via `jnp.array([7,0,6,1,5,2,4,3])`
2. Construct `probe` via `jnp.array([[.2,-.9],[-.4,.6],[.8,.1]])`
3. Evaluate `params, w1=params['w1'][:, order], b1=params['b1'][order], w2=params['w2'][order, :]` and convert the result into Python scalar/collection `permuted`.
4. Check numerical equivalence within tolerance: `np.testing.assert_allclose(logits(permuted,probe),logits(params,p...`
5. Evaluate `params, w1=params['w1'][:, order], b1=params['b1'][order]` and convert the result into Python scalar/collection `broken`.

**Starter code scaffold (fill in the TODOs):**

```python
# Reorder hidden units without changing predictions (Transfer): A hidden unit has no intrinsic index.
# Construct `order` via `jnp.array([7,0,6,1,5,2,4,3])`
order = jnp.array(...)  # TODO: compute order
# Construct `probe` via `jnp.array([[.2,-.9],[-.4,.6],[.8,.1]])`
probe = jnp.array(...)  # TODO: compute probe
# Evaluate `params, w1=params['w1'][:, order], b1=params['b1'][order], w2=params['w2'][order, :]` and convert the result into Python scalar/collection `permuted`.
permuted = dict(...)  # TODO: compute permuted
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(logits(permuted,probe),logits(params,p...`
np.testing.assert_allclose(logits(permuted,probe),logits(params,probe),rtol = ...  # TODO: compute np.testing.assert_allclose(logits(permuted,probe),logits(params,probe),rtol
# Evaluate `params, w1=params['w1'][:, order], b1=params['b1'][order]` and convert the result into Python scalar/collection `broken`.
broken = dict(...)  # TODO: compute broken
# Verify that the numerical values match the expected reference within tolerance.
assert not np.allclose(logits(broken,probe),logits(params,probe),atol=1e-5)  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print('Consistent hidden permutation preserves logits; one-sided permutation changes them.')
```

<details><summary>Reference solution and reasoning</summary>

```python
# Reorder hidden units without changing predictions (Transfer): A hidden unit has no intrinsic index.
# Construct `order` via `jnp.array([7,0,6,1,5,2,4,3])`
order=jnp.array([7,0,6,1,5,2,4,3])
# Construct `probe` via `jnp.array([[.2,-.9],[-.4,.6],[.8,.1]])`
probe=jnp.array([[.2,-.9],[-.4,.6],[.8,.1]])
# Evaluate `params, w1=params['w1'][:, order], b1=params['b1'][order], w2=params['w2'][order, :]` and convert the result into Python scalar/collection `permuted`.
permuted=dict(params,w1=params['w1'][:,order],b1=params['b1'][order],w2=params['w2'][order,:])
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(logits(permuted,probe),logits(params,p...`
np.testing.assert_allclose(logits(permuted,probe),logits(params,probe),rtol=1e-5,atol=1e-5)
# Evaluate `params, w1=params['w1'][:, order], b1=params['b1'][order]` and convert the result into Python scalar/collection `broken`.
broken=dict(params,w1=params['w1'][:,order],b1=params['b1'][order])
# Verify that the numerical values match the expected reference within tolerance.
assert not np.allclose(logits(broken,probe),logits(params,probe),atol=1e-5)
# Print the observed values to compare against the expected result.
print('Consistent hidden permutation preserves logits; one-sided permutation changes them.')
```

A hidden unit has no intrinsic index. Reordering its incoming weights, bias and outgoing weights together preserves the function. This is a parameter-layout invariant, not another training run.

</details>

## Diagnose accidental pairwise broadcasting

**Intermediate**

Demonstrate how labels shaped $(B, 1)$ and scores shaped $(B,)$ create a $(B, B)$ loss. Repair the contract before averaging.

<details><summary>Hint</summary>

Inspect the unreduced loss shape; a scalar mean can hide the error.

</details>

### How to write: Diagnose accidental pairwise broadcasting — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `x.sum(axis=...) / x.mean(axis=..., keepdims=...)` — Reduces values along the named `axis` (the axis you name is collapsed unless `keepdims=True`).
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `optax.adam(lr) / optax.apply_updates(params, updates)` — Optax gradient transformations and numerically stable loss functions over parameter PyTrees.

**Step-by-step implementation plan:**
1. Diagnose accidental pairwise broadcasting (Intermediate): The incorrect scalar still looks plausible, but it compares...
2. Configure or step the Optax optimizer state (`bad`).
3. Check tensor shape invariant: `bad.shape==(4,4)`
4. Configure or step the Optax optimizer state (`good`).
5. Check tensor shape invariant: `good.shape==(4,)`

**Starter code scaffold (fill in the TODOs):**

```python
# Diagnose accidental pairwise broadcasting (Intermediate): The incorrect scalar still looks plausible, but it compares...
# Configure or step the Optax optimizer state (`bad`).
bad = optax.sigmoid_binary_cross_entropy(...)  # TODO: compute bad
# Check tensor shape invariant: `bad.shape==(4,4)`
assert bad.shape  # TODO: complete assertion check
# Configure or step the Optax optimizer state (`good`).
good = optax.sigmoid_binary_cross_entropy(...)  # TODO: compute good
# Check tensor shape invariant: `good.shape==(4,)`
assert good.shape  # TODO: complete assertion check
# Reduce across the target axis to summarize ``.
np.testing.assert_allclose(jnp.mean(good),loss(params,x,y),rtol = ...  # TODO: compute np.testing.assert_allclose(jnp.mean(good),loss(params,x,y),rtol
```

<details><summary>Reference solution and reasoning</summary>

```python
# Diagnose accidental pairwise broadcasting (Intermediate): The incorrect scalar still looks plausible, but it compares...
# Configure or step the Optax optimizer state (`bad`).
bad=optax.sigmoid_binary_cross_entropy(logits(params,x),y[:,None])
# Check tensor shape invariant: `bad.shape==(4,4)`
assert bad.shape==(4,4)
# Configure or step the Optax optimizer state (`good`).
good=optax.sigmoid_binary_cross_entropy(logits(params,x),y)
# Check tensor shape invariant: `good.shape==(4,)`
assert good.shape==(4,)
# Reduce across the target axis to summarize ``.
np.testing.assert_allclose(jnp.mean(good),loss(params,x,y),rtol=1e-6)
```

The incorrect scalar still looks plausible, but it compares each score with every label. A shape check catches the silent objective change.

</details>

## Check your understanding

Why does the two-layer XOR model need a nonlinear activation?

1. Without it the composition is still affine
2. Two affine layers always separate XOR
3. Cross-entropy itself changes the prediction boundary

<details><summary>Answer and explanation</summary>

Without it the composition is still affine

The optimizer can change coefficients but cannot enlarge the affine function class. The activation supplies nonlinear intermediate features.

</details>

## Diagnose the result

Inspect unreduced loss shapes before tuning learning rate. Compare the loss with NumPy, check finite gradients, and test $B=1$. If fit succeeds but perturbed held-out points fail, investigate the learned boundary rather than claiming generalization from four exact predictions.

## Carry forward

- Trace the shapes from observations to logits
- Explain why the hidden activation matters
- Use stable binary cross-entropy
- Separate learning from held-out measurement

## Keep your evidence

Keep the 33-parameter shape calculation, independent NumPy cross-entropy, train and held-out seeds, initial/final loss, nearby held-out accuracy, and one repaired score/label shape failure.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [Optax sigmoid binary cross-entropy](https://optax.readthedocs.io/en/latest/api/losses.html#optax.losses.sigmoid_binary_cross_entropy)
- [JAX automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html)

