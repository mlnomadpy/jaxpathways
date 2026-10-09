"""A compiled train and evaluation step: worked experiments and reference solutions. CPU checks."""

# 1. Construct data, model, and optimizer
# Step 1 — 1. Construct data, model, and optimizer: The held-out seed is 11; every optimizer call below uses seed-10...
# Import flax (nnx) for this computation.
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
# Function `dataset(seed, n)` implementing this stage's computation:
def dataset(seed,n=48):
    # Cast or evaluate `a` in explicit floating-point precision.
    a=np.random.default_rng(seed).normal(size=(n,2)).astype(np.float32)
    # Cast or evaluate `b` in explicit floating-point precision.
    b=(a[:,0]+.5*a[:,1]>0).astype(np.float32)
    # Return `(jnp.array(a), jnp.array(b))` to the caller.
    return jnp.array(a),jnp.array(b)
# Run `dataset` to compute `(train_x, train_y)`.
train_x,train_y=dataset(10)
# Run `dataset` to compute `(test_x, test_y)`.
test_x,test_y=dataset(11)
# Define `Classifier` module / container with explicit state and forward pass:
class Classifier(nnx.Module):
    # Function `__init__(self, seed)` implementing this stage's computation:
    def __init__(self,seed=0):
        # Run `nnx.Rngs` to compute `rngs`.
        rngs=nnx.Rngs(seed)
        # Run `nnx.Linear` to compute `self.hidden`.
        self.hidden=nnx.Linear(2,8,rngs=rngs)
        # Run `nnx.Linear` to compute `self.out`.
        self.out=nnx.Linear(8,1,rngs=rngs)
    # Function `__call__(self, x)` implementing this stage's computation:
    # Return `self.out(nnx.tanh(self.hidden(x))).squeeze(-1)` to the caller.
    def __call__(self,x):return self.out(nnx.tanh(self.hidden(x))).squeeze(-1)
# Function `objective(m, x, y)` implementing this stage's computation:
# Return `jnp.mean(optax.sigmoid_binary_cross_entropy(m(x), y))` to the caller.
def objective(m,x,y):return jnp.mean(optax.sigmoid_binary_cross_entropy(m(x),y))
# Function `initialize()` implementing this stage's computation:
def initialize():
    # Run `Classifier` to compute `m`.
    m=Classifier(0)
    # Return `(m, nnx.Optimizer(m, optax.adam(0.03), wrt=nnx.Param))` to the caller.
    return m,nnx.Optimizer(m,optax.adam(.03),wrt=nnx.Param)
# Run `initialize` to compute `(model, optimizer)`.
model,optimizer=initialize()

# 2. Define the two compiled functions
# Step 2 — 2. Define the two compiled functions: The 80-update counter is an optimizer invariant.
# Define and JIT-compile `train_step(m, o, x, y)` so XLA traces and fuses the operations:
@nnx.jit
# Function `train_step(m, o, x, y)` implementing this stage's computation:
def train_step(m,o,x,y):
    # Differentiate the objective to obtain `(value, grads)` via automatic differentiation.
    value,grads=nnx.value_and_grad(objective)(m,x,y)
    # Update state in place with the new values.
    o.update(m,grads)
    # Return `value` to the caller.
    return value
# Define and JIT-compile `evaluate(m, x, y)` so XLA traces and fuses the operations:
@nnx.jit
# Function `evaluate(m, x, y)` implementing this stage's computation:
def evaluate(m,x,y):
    # Run `m` to compute `scores`.
    scores=m(x)
    # Return `(jnp.mean(optax.sigmoid_binary_cross_entropy(scores, y)), jnp.mean((scores > 0) == y))` to the caller.
    return jnp.mean(optax.sigmoid_binary_cross_entropy(scores,y)),jnp.mean((scores>0)==y)
# Function `snapshot(m)` implementing this stage's computation:
# Return `[np.array(a, copy=True) for a in jax.tree.leaves(nnx.state(m))]` to the caller.
def snapshot(m):return [np.array(a,copy=True) for a in jax.tree.leaves(nnx.state(m))]
# Evaluate `objective(model, train_x, train_y)` and convert the result into Python scalar/collection `before_loss`.
before_loss=float(objective(model,train_x,train_y))
# Repeat the update loop over `range(80)` steps:
# Execute `for _ in range(80):train_step(model,optimizer,train_x,train_y)`.
for _ in range(80):train_step(model,optimizer,train_x,train_y)
# Assert that `int(optimizer.step[...])==80`.
assert int(optimizer.step[...])==80

# 3. Verify held-out metrics and state isolation
# Step 3 — 3. Verify held-out metrics and state isolation: The synthetic fixture normally exceeds 0.85 held-out accuracy.
frozen=snapshot(model)
step_count=int(optimizer.step[...])
# Run `evaluate` to compute `(test_loss, test_accuracy)`.
test_loss,test_accuracy=evaluate(model,test_x,test_y)
# Iterate over `(a, b)` to step through the computation:
for a,b in zip(frozen,snapshot(model)):np.testing.assert_array_equal(a,b)
# Assert that `int(optimizer.step[...])==step_count`.
assert int(optimizer.step[...])==step_count
# Convert `scores` to a host NumPy array for inspection or verification.
# Convert `labels` to a host NumPy array for inspection or verification.
scores=np.asarray(model(test_x))
labels=np.asarray(test_y)
# Compute `np.testing.assert_allclose(test_loss,np.mean(np.logaddexp(0.,scores)-labels*scores),rtol` as `1e-5,atol=1e-6)`.
np.testing.assert_allclose(test_loss,np.mean(np.logaddexp(0.,scores)-labels*scores),rtol=1e-5,atol=1e-6)
# Compute `np.testing.assert_allclose(test_accuracy,np.mean((scores>0)` as `=labels),atol=1e-6)`.
np.testing.assert_allclose(test_accuracy,np.mean((scores>0)==labels),atol=1e-6)
# Assert that `float(test_accuracy)>.85 and float(objective(model,train_x,train_y))<before_loss*.3`.
assert float(test_accuracy)>.85 and float(objective(model,train_x,train_y))<before_loss*.3
# Print the observed values to compare against the expected result.
print('Updates:',step_count,'held-out loss:',float(test_loss),'accuracy:',float(test_accuracy))

# Step 1 — 1. Construct data, model, and optimizer: The held-out seed is 11; every optimizer call below uses seed-10...
# Import flax (nnx) for this computation.
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
# Function `dataset(seed, n)` implementing this stage's computation:
def dataset(seed,n=48):
    # Cast or evaluate `a` in explicit floating-point precision.
    a=np.random.default_rng(seed).normal(size=(n,2)).astype(np.float32)
    # Cast or evaluate `b` in explicit floating-point precision.
    b=(a[:,0]+.5*a[:,1]>0).astype(np.float32)
    # Return `(jnp.array(a), jnp.array(b))` to the caller.
    return jnp.array(a),jnp.array(b)
# Run `dataset` to compute `(train_x, train_y)`.
train_x,train_y=dataset(10)
# Run `dataset` to compute `(test_x, test_y)`.
test_x,test_y=dataset(11)
# Define `Classifier` module / container with explicit state and forward pass:
class Classifier(nnx.Module):
    # Function `__init__(self, seed)` implementing this stage's computation:
    def __init__(self,seed=0):
        # Run `nnx.Rngs` to compute `rngs`.
        rngs=nnx.Rngs(seed)
        # Run `nnx.Linear` to compute `self.hidden`.
        self.hidden=nnx.Linear(2,8,rngs=rngs)
        # Run `nnx.Linear` to compute `self.out`.
        self.out=nnx.Linear(8,1,rngs=rngs)
    # Function `__call__(self, x)` implementing this stage's computation:
    # Return `self.out(nnx.tanh(self.hidden(x))).squeeze(-1)` to the caller.
    def __call__(self,x):return self.out(nnx.tanh(self.hidden(x))).squeeze(-1)
# Function `objective(m, x, y)` implementing this stage's computation:
# Return `jnp.mean(optax.sigmoid_binary_cross_entropy(m(x), y))` to the caller.
def objective(m,x,y):return jnp.mean(optax.sigmoid_binary_cross_entropy(m(x),y))
# Function `initialize()` implementing this stage's computation:
def initialize():
    # Run `Classifier` to compute `m`.
    m=Classifier(0)
    # Return `(m, nnx.Optimizer(m, optax.adam(0.03), wrt=nnx.Param))` to the caller.
    return m,nnx.Optimizer(m,optax.adam(.03),wrt=nnx.Param)
# Run `initialize` to compute `(model, optimizer)`.
model,optimizer=initialize()

# Step 2 — 2. Define the two compiled functions: The 80-update counter is an optimizer invariant.
# Define and JIT-compile `train_step(m, o, x, y)` so XLA traces and fuses the operations:
@nnx.jit
# Function `train_step(m, o, x, y)` implementing this stage's computation:
def train_step(m,o,x,y):
    # Differentiate the objective to obtain `(value, grads)` via automatic differentiation.
    value,grads=nnx.value_and_grad(objective)(m,x,y)
    # Update state in place with the new values.
    o.update(m,grads)
    # Return `value` to the caller.
    return value
# Define and JIT-compile `evaluate(m, x, y)` so XLA traces and fuses the operations:
@nnx.jit
# Function `evaluate(m, x, y)` implementing this stage's computation:
def evaluate(m,x,y):
    # Run `m` to compute `scores`.
    scores=m(x)
    # Return `(jnp.mean(optax.sigmoid_binary_cross_entropy(scores, y)), jnp.mean((scores > 0) == y))` to the caller.
    return jnp.mean(optax.sigmoid_binary_cross_entropy(scores,y)),jnp.mean((scores>0)==y)
# Function `snapshot(m)` implementing this stage's computation:
# Return `[np.array(a, copy=True) for a in jax.tree.leaves(nnx.state(m))]` to the caller.
def snapshot(m):return [np.array(a,copy=True) for a in jax.tree.leaves(nnx.state(m))]
# Evaluate `objective(model, train_x, train_y)` and convert the result into Python scalar/collection `before_loss`.
before_loss=float(objective(model,train_x,train_y))
# Repeat the update loop over `range(80)` steps:
# Execute `for _ in range(80):train_step(model,optimizer,train_x,train_y)`.
for _ in range(80):train_step(model,optimizer,train_x,train_y)
# Assert that `int(optimizer.step[...])==80`.
assert int(optimizer.step[...])==80

# Step 3 — 3. Verify held-out metrics and state isolation: The synthetic fixture normally exceeds 0.85 held-out accuracy.
frozen=snapshot(model)
step_count=int(optimizer.step[...])
# Run `evaluate` to compute `(test_loss, test_accuracy)`.
test_loss,test_accuracy=evaluate(model,test_x,test_y)
# Iterate over `(a, b)` to step through the computation:
for a,b in zip(frozen,snapshot(model)):np.testing.assert_array_equal(a,b)
# Assert that `int(optimizer.step[...])==step_count`.
assert int(optimizer.step[...])==step_count
# Convert `scores` to a host NumPy array for inspection or verification.
# Convert `labels` to a host NumPy array for inspection or verification.
scores=np.asarray(model(test_x))
labels=np.asarray(test_y)
# Compute `np.testing.assert_allclose(test_loss,np.mean(np.logaddexp(0.,scores)-labels*scores),rtol` as `1e-5,atol=1e-6)`.
np.testing.assert_allclose(test_loss,np.mean(np.logaddexp(0.,scores)-labels*scores),rtol=1e-5,atol=1e-6)
# Compute `np.testing.assert_allclose(test_accuracy,np.mean((scores>0)` as `=labels),atol=1e-6)`.
np.testing.assert_allclose(test_accuracy,np.mean((scores>0)==labels),atol=1e-6)
# Assert that `float(test_accuracy)>.85 and float(objective(model,train_x,train_y))<before_loss*.3`.
assert float(test_accuracy)>.85 and float(objective(model,train_x,train_y))<before_loss*.3
# Print the observed values to compare against the expected result.
print('Updates:',step_count,'held-out loss:',float(test_loss),'accuracy:',float(test_accuracy))

# Figure data experiment
# Compute figure data for: Inspect predictions on held-out inputs
# Generate a uniform grid of points in `axis`.
axis = jnp.linspace(-3.0, 3.0, 61)
# Run `jnp.meshgrid` to compute `(gx, gy)`.
gx, gy = jnp.meshgrid(axis, axis)
# Combine or mask array elements to form `grid`.
grid = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)
# Rearrange tensor axes to match the required layout for `prob`.
prob = jax.nn.sigmoid(model(grid)).reshape(gx.shape)
# Compute `visual_data` from `{'kind': 'field', 'values': prob.tolist(), 'extent':...`
visual_data = {'kind': 'field', 'values': prob.tolist(), 'extent': [-3.0, 3.0, -3.0, 3.0], 'xlabel': 'feature 0', 'ylabel': 'feature 1', 'unit': 'P(label 1)', 'points': test_x.tolist(), 'labels': test_y.tolist()}

# Experiment: Replay the complete update sequence
# Experiment — Replay the complete update sequence: Replay reconstructs optimizer state from the start.
replay,replay_optimizer=initialize()
# Repeat the update loop over `range(80)` steps:
# Execute `for _ in range(80):train_step(replay,replay_optimizer,train_x,train_y)`.
for _ in range(80):train_step(replay,replay_optimizer,train_x,train_y)
# Iterate over `(a, b)` to step through the computation:
for a,b in zip(snapshot(model),snapshot(replay)):np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
# Compute `np.testing.assert_allclose(evaluate(replay,test_x,test_y),evaluate(model,test_x,test_y),rtol` as `1e-6)`.
np.testing.assert_allclose(evaluate(replay,test_x,test_y),evaluate(model,test_x,test_y),rtol=1e-6)

# Experiment: Duplicating evaluation data
# Experiment — Duplicating evaluation data: Means normalize the observation count.
doubled=evaluate(model,jnp.concatenate([test_x,test_x]),jnp.concatenate([test_y,test_y]))
# Compute `np.testing.assert_allclose(doubled,evaluate(model,test_x,test_y),rtol` as `1e-5,atol=1e-6)`.
np.testing.assert_allclose(doubled,evaluate(model,test_x,test_y),rtol=1e-5,atol=1e-6)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Compute held-out metrics in three equal chunks and reconstruct the...
old=snapshot(model)
# Compute `pieces` from `[evaluate(model,test_x[i:i+16],test_y[i:i+16]) for i...`
pieces=[evaluate(model,test_x[i:i+16],test_y[i:i+16]) for i in range(0,48,16)]
# Reduce along axis=0 to compute `aggregate`.
aggregate=np.mean(np.asarray(pieces),axis=0)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(aggregate,np.asarray(evaluate(model,test_x,test_y)),rtol=1e-5,atol=1e-6)
# Iterate over `(a, b)` to step through the computation:
for a,b in zip(old,snapshot(model)):np.testing.assert_array_equal(a,b)

# Reference practice: Aggregate unequal batches
# Aggregate unequal batches (Transfer): A batch is a packaging choice, not a unit of evidence.
# Compute `sizes` from `np.array([7,41])`
sizes=np.array([7,41])
# Convert `metrics` to a host NumPy array for inspection or verification.
metrics=np.asarray([evaluate(model,test_x[:7],test_y[:7]),evaluate(model,test_x[7:],test_y[7:])])
# Reduce along axis=0 to compute `weighted`.
weighted=(metrics*sizes[:,None]).sum(axis=0)/sizes.sum()
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(weighted,np.asarray(evaluate(model,test_x,test_y)),rtol=1e-5,atol=1e-6)
# Compute `correct` from `np.array([6,20])`
correct=np.array([6,20])
counts=np.array([7,41])
# Compute `expected` from `26/48`
expected=26/48
# Execute `np.testing.assert_allclose(correct.sum()/counts.sum(),expected)`.
np.testing.assert_allclose(correct.sum()/counts.sum(),expected)
# Assert that `not np.isclose(np.mean(correct/counts),expected)`.
assert not np.isclose(np.mean(correct/counts),expected)
# Print the observed values to compare against the expected result.
print('Count-weighted held-out metrics:',weighted,'constructed accuracy:',expected)

# Reference practice: Catch an evaluation function that trains
# Catch an evaluation function that trains (Intermediate): A plausible returned loss is separate from evaluation.
graph,state=nnx.split(model)
# Transform every leaf of the parameter PyTree (`bad_model`).
bad_model=nnx.merge(graph,jax.tree.map(lambda a:jnp.array(a,copy=True),state))
# Configure or step the Optax optimizer state (`bad_optimizer`).
bad_optimizer=nnx.Optimizer(bad_model,optax.adam(.03),wrt=nnx.Param)
# Run `snapshot` to compute `prior`.
prior=snapshot(bad_model)
# Run `train_step` to perform the next check or state transition.
train_step(bad_model,bad_optimizer,test_x,test_y)
# Assert that `int(bad_optimizer.step[...])==1`.
assert int(bad_optimizer.step[...])==1
# Assert that `any(not np.array_equal(a,b) for a,b in zip(prior,snapshot(bad_model)))`.
assert any(not np.array_equal(a,b) for a,b in zip(prior,snapshot(bad_model)))
# Assert that `int(optimizer.step[...])==80`.
assert int(optimizer.step[...])==80
print("PASS: networks-03")
