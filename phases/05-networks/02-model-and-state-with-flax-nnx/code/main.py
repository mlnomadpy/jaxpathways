"""Model and state with Flax NNX: worked experiments and reference solutions. CPU checks."""

# 1. Define typed state and the model
# Step 1 — 1. Define typed state and the model: nnx.Rngs manages initializer keys.
# Import flax (nnx) for this computation.
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
# Define `TinyMLP` module / container with explicit state and forward pass:
class TinyMLP(nnx.Module):
    # Function `__init__(self, seed)` implementing this stage's computation:
    def __init__(self,seed=0):
        # Run `nnx.Rngs` to compute `rngs`.
        rngs=nnx.Rngs(seed)
        # Run `nnx.Linear` to compute `self.hidden`.
        self.hidden=nnx.Linear(2,3,rngs=rngs)
        # Run `nnx.Linear` to compute `self.out`.
        self.out=nnx.Linear(3,1,rngs=rngs)
        # Create device-backed JAX array `self.calls`.
        self.calls=nnx.Variable(jnp.array(0,dtype=jnp.int32))
    # Function `__call__(self, x, record)` implementing this stage's computation:
    def __call__(self,x,record=False):
        # Branch on condition `record`:
        if record:self.calls[...] += 1
        # Return `self.out(nnx.tanh(self.hidden(x))).squeeze(-1)` to the caller.
        return self.out(nnx.tanh(self.hidden(x))).squeeze(-1)
# Run `TinyMLP` to compute `model`.
model=TinyMLP()
# Initialize array `batch` with explicit values and shape.
batch=jnp.array([[1.,2.],[-1.,.5],[0.,0.]])

# 2. Verify equations and inspect the filters
# Step 2 — 2. Verify equations and inspect the filters: The matching forward pass verifies the network wiring.
scores=model(batch,record=True)
# Convert `h` to a host NumPy array for inspection or verification.
h=np.tanh(np.asarray(batch)@np.asarray(model.hidden.kernel[...])+np.asarray(model.hidden.bias[...]))
# Convert `expected` to a host NumPy array for inspection or verification.
expected=(h@np.asarray(model.out.kernel[...])+np.asarray(model.out.bias[...])).squeeze(-1)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(scores,expected,rtol=1e-5,atol=1e-6)
# Run `nnx.state` to compute `param_state`.
param_state=nnx.state(model,nnx.Param)
# Run `nnx.state` to compute `other_state`.
other_state=nnx.state(model,nnx.Not(nnx.Param))
# Verify contract: `sum((a.size for a in jax.tree.leaves(param_state))) == 13`.
assert sum(a.size for a in jax.tree.leaves(param_state))==13
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert sum(a.size for a in jax.tree.leaves(other_state))==1
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert int(model.calls[...])==1

# 3. Split, merge, and test independence
# Step 3 — 3. Split, merge, and test independence: nnx.grad differentiates Param variables by default.
graphdef,state=nnx.split(model)
# Transform every leaf of the parameter PyTree (`clone`).
clone=nnx.merge(graphdef,jax.tree.map(lambda a:jnp.array(a,copy=True),state))
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(clone(batch),model(batch),rtol=1e-6)
# Run `clone` to perform the next check or state transition.
clone(batch,record=True)
# Verify contract: `int(clone.calls[...]) == 2 and int(model.calls[...]) == 1`.
assert int(clone.calls[...])==2 and int(model.calls[...])==1
# Differentiate the objective to obtain `grads` via automatic differentiation.
grads=nnx.grad(lambda m:jnp.mean(m(batch)**2))(model)
# Verify contract: `sum((a.size for a in jax.tree.leaves(grads))) == 13`.
assert sum(a.size for a in jax.tree.leaves(grads))==13
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert 'calls' not in grads
# Print the observed values to compare against the expected result.
print('Trainable scalars: 13; other state: 1; clone counters:',int(model.calls[...]),int(clone.calls[...]))

# Step 1 — 1. Define typed state and the model: nnx.Rngs manages initializer keys.
# Import flax (nnx) for this computation.
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
# Define `TinyMLP` module / container with explicit state and forward pass:
class TinyMLP(nnx.Module):
    # Function `__init__(self, seed)` implementing this stage's computation:
    def __init__(self,seed=0):
        # Run `nnx.Rngs` to compute `rngs`.
        rngs=nnx.Rngs(seed)
        # Run `nnx.Linear` to compute `self.hidden`.
        self.hidden=nnx.Linear(2,3,rngs=rngs)
        # Run `nnx.Linear` to compute `self.out`.
        self.out=nnx.Linear(3,1,rngs=rngs)
        # Create device-backed JAX array `self.calls`.
        self.calls=nnx.Variable(jnp.array(0,dtype=jnp.int32))
    # Function `__call__(self, x, record)` implementing this stage's computation:
    def __call__(self,x,record=False):
        # Branch on condition `record`:
        if record:self.calls[...] += 1
        # Return `self.out(nnx.tanh(self.hidden(x))).squeeze(-1)` to the caller.
        return self.out(nnx.tanh(self.hidden(x))).squeeze(-1)
# Run `TinyMLP` to compute `model`.
model=TinyMLP()
# Initialize array `batch` with explicit values and shape.
batch=jnp.array([[1.,2.],[-1.,.5],[0.,0.]])

# Step 2 — 2. Verify equations and inspect the filters: The matching forward pass verifies the network wiring.
scores=model(batch,record=True)
# Convert `h` to a host NumPy array for inspection or verification.
h=np.tanh(np.asarray(batch)@np.asarray(model.hidden.kernel[...])+np.asarray(model.hidden.bias[...]))
# Convert `expected` to a host NumPy array for inspection or verification.
expected=(h@np.asarray(model.out.kernel[...])+np.asarray(model.out.bias[...])).squeeze(-1)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(scores,expected,rtol=1e-5,atol=1e-6)
# Run `nnx.state` to compute `param_state`.
param_state=nnx.state(model,nnx.Param)
# Run `nnx.state` to compute `other_state`.
other_state=nnx.state(model,nnx.Not(nnx.Param))
# Verify contract: `sum((a.size for a in jax.tree.leaves(param_state))) == 13`.
assert sum(a.size for a in jax.tree.leaves(param_state))==13
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert sum(a.size for a in jax.tree.leaves(other_state))==1
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert int(model.calls[...])==1

# Step 3 — 3. Split, merge, and test independence: nnx.grad differentiates Param variables by default.
graphdef,state=nnx.split(model)
# Transform every leaf of the parameter PyTree (`clone`).
clone=nnx.merge(graphdef,jax.tree.map(lambda a:jnp.array(a,copy=True),state))
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(clone(batch),model(batch),rtol=1e-6)
# Run `clone` to perform the next check or state transition.
clone(batch,record=True)
# Verify contract: `int(clone.calls[...]) == 2 and int(model.calls[...]) == 1`.
assert int(clone.calls[...])==2 and int(model.calls[...])==1
# Differentiate the objective to obtain `grads` via automatic differentiation.
grads=nnx.grad(lambda m:jnp.mean(m(batch)**2))(model)
# Verify contract: `sum((a.size for a in jax.tree.leaves(grads))) == 13`.
assert sum(a.size for a in jax.tree.leaves(grads))==13
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert 'calls' not in grads
# Print the observed values to compare against the expected result.
print('Trainable scalars: 13; other state: 1; clone counters:',int(model.calls[...]),int(clone.calls[...]))

# Experiment: Bias shifts every score equally
# Experiment — Bias shifts every score equally: An affine output bias changes the decision threshold uniformly;...
before=np.asarray(model(batch))
# Evaluate `saved` from the current inputs and state.
saved=model.out.bias[...]
# Evaluate `model.out.bias[...]` from the current inputs and state.
model.out.bias[...] = saved+.25
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(model(batch),before+.25,atol=1e-6)
# Evaluate `model.out.bias[...]` from the current inputs and state.
model.out.bias[...] = saved

# Experiment: Repeated construction reproduces state
# Experiment — Repeated construction reproduces state: Replay is checked in one pinned environment.
replay=TinyMLP(0)
# Iterate over `(a, b)` to step through the computation:
for a,b in zip(jax.tree.leaves(nnx.state(replay,nnx.Param)),jax.tree.leaves(param_state)):
    # Verify that computed values match the expected reference within numerical tolerance.
    np.testing.assert_array_equal(a,b)
# Run `TinyMLP` to compute `different`.
different=TinyMLP(1)
# Verify contract: `not np.array_equal(np.asarray(different.hidden.kernel[...]), np.asar...`.
assert not np.array_equal(np.asarray(different.hidden.kernel[...]),np.asarray(model.hidden.kernel[...]))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Apply the model to one new row and to five rows.
# Initialize array `single` with explicit values and shape.
single=jnp.array([[.3,-.7]])
# Verify that the output tensor shape matches our prediction.
assert model(single).shape==(1,)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert model(jnp.repeat(single,5,axis=0)).shape==(5,)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert sum(a.size for a in jax.tree.leaves(nnx.state(model,nnx.Param)))==13

# Reference practice: Change a clone without changing its source
# Change a clone without changing its source (Transfer): The bias is broadcast over observations, so every logit...
# Transform every leaf of the parameter PyTree (`original_leaves`).
original_leaves=[np.array(a,copy=True) for a in jax.tree.leaves(nnx.state(model,nnx.Param))]
# Convert `old_clone` to a host NumPy array for inspection or verification.
old_clone=np.asarray(clone(batch)).copy()
# Evaluate `clone.out.bias[...]` from the current inputs and state.
clone.out.bias[...] = clone.out.bias[...] + 1.
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(clone(batch),old_clone+1.,rtol=1e-5,atol=1e-6)
# Iterate over `(before, after)` to step through the computation:
for before,after in zip(original_leaves,jax.tree.leaves(nnx.state(model,nnx.Param))):
    # Verify that computed values match the expected reference within numerical tolerance.
    np.testing.assert_array_equal(before,after)
# Print the observed values to compare against the expected result.
print('Clone logits shifted by one; original parameters unchanged.')

# Reference practice: Diagnose a shared module reference
# Diagnose a shared module reference (Intermediate): Sharing a graph is useful for tied layers but wrong for an...
saved_count=int(model.calls[...])
# Evaluate `shared` from the current inputs and state.
shared=model
# Run `shared` to perform the next check or state transition.
shared(batch,record=True)
# Verify contract: `int(model.calls[...]) == saved_count + 1`.
assert int(model.calls[...])==saved_count+1
# Run `nnx.split` to compute `(graph, state)`.
graph,state=nnx.split(model)
# Transform every leaf of the parameter PyTree (`independent`).
independent=nnx.merge(graph,jax.tree.map(lambda a:jnp.array(a,copy=True),state))
# Run `independent` to perform the next check or state transition.
independent(batch,record=True)
# Verify contract: `int(model.calls[...]) == saved_count + 1`.
assert int(model.calls[...])==saved_count+1
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert int(independent.calls[...])==saved_count+2
print("PASS: networks-02")
