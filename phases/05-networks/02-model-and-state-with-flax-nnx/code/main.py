"""Model and state with Flax NNX: worked experiments and reference solutions. CPU checks."""

# 1. Define typed state and the model
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
class TinyMLP(nnx.Module):
    def __init__(self,seed=0):
        rngs=nnx.Rngs(seed)
        self.hidden=nnx.Linear(2,3,rngs=rngs)
        self.out=nnx.Linear(3,1,rngs=rngs)
        self.calls=nnx.Variable(jnp.array(0,dtype=jnp.int32))
    def __call__(self,x,record=False):
        if record:self.calls[...] += 1
        return self.out(nnx.tanh(self.hidden(x))).squeeze(-1)
model=TinyMLP()
batch=jnp.array([[1.,2.],[-1.,.5],[0.,0.]])

# 2. Verify equations and inspect the filters
scores=model(batch,record=True)
h=np.tanh(np.asarray(batch)@np.asarray(model.hidden.kernel[...])+np.asarray(model.hidden.bias[...]))
expected=(h@np.asarray(model.out.kernel[...])+np.asarray(model.out.bias[...])).squeeze(-1)
np.testing.assert_allclose(scores,expected,rtol=1e-5,atol=1e-6)
param_state=nnx.state(model,nnx.Param)
other_state=nnx.state(model,nnx.Not(nnx.Param))
assert sum(a.size for a in jax.tree.leaves(param_state))==13
assert sum(a.size for a in jax.tree.leaves(other_state))==1
assert int(model.calls[...])==1

# 3. Split, merge, and test independence
graphdef,state=nnx.split(model)
clone=nnx.merge(graphdef,jax.tree.map(lambda a:jnp.array(a,copy=True),state))
np.testing.assert_allclose(clone(batch),model(batch),rtol=1e-6)
clone(batch,record=True)
assert int(clone.calls[...])==2 and int(model.calls[...])==1
grads=nnx.grad(lambda m:jnp.mean(m(batch)**2))(model)
assert sum(a.size for a in jax.tree.leaves(grads))==13
assert 'calls' not in grads
print('Trainable scalars: 13; other state: 1; clone counters:',int(model.calls[...]),int(clone.calls[...]))

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
class TinyMLP(nnx.Module):
    def __init__(self,seed=0):
        rngs=nnx.Rngs(seed)
        self.hidden=nnx.Linear(2,3,rngs=rngs)
        self.out=nnx.Linear(3,1,rngs=rngs)
        self.calls=nnx.Variable(jnp.array(0,dtype=jnp.int32))
    def __call__(self,x,record=False):
        if record:self.calls[...] += 1
        return self.out(nnx.tanh(self.hidden(x))).squeeze(-1)
model=TinyMLP()
batch=jnp.array([[1.,2.],[-1.,.5],[0.,0.]])

scores=model(batch,record=True)
h=np.tanh(np.asarray(batch)@np.asarray(model.hidden.kernel[...])+np.asarray(model.hidden.bias[...]))
expected=(h@np.asarray(model.out.kernel[...])+np.asarray(model.out.bias[...])).squeeze(-1)
np.testing.assert_allclose(scores,expected,rtol=1e-5,atol=1e-6)
param_state=nnx.state(model,nnx.Param)
other_state=nnx.state(model,nnx.Not(nnx.Param))
assert sum(a.size for a in jax.tree.leaves(param_state))==13
assert sum(a.size for a in jax.tree.leaves(other_state))==1
assert int(model.calls[...])==1

graphdef,state=nnx.split(model)
clone=nnx.merge(graphdef,jax.tree.map(lambda a:jnp.array(a,copy=True),state))
np.testing.assert_allclose(clone(batch),model(batch),rtol=1e-6)
clone(batch,record=True)
assert int(clone.calls[...])==2 and int(model.calls[...])==1
grads=nnx.grad(lambda m:jnp.mean(m(batch)**2))(model)
assert sum(a.size for a in jax.tree.leaves(grads))==13
assert 'calls' not in grads
print('Trainable scalars: 13; other state: 1; clone counters:',int(model.calls[...]),int(clone.calls[...]))

# Experiment: Bias shifts every score equally
before=np.asarray(model(batch))
saved=model.out.bias[...]
model.out.bias[...] = saved+.25
np.testing.assert_allclose(model(batch),before+.25,atol=1e-6)
model.out.bias[...] = saved

# Experiment: Repeated construction reproduces state
replay=TinyMLP(0)
for a,b in zip(jax.tree.leaves(nnx.state(replay,nnx.Param)),jax.tree.leaves(param_state)):
    np.testing.assert_array_equal(a,b)
different=TinyMLP(1)
assert not np.array_equal(np.asarray(different.hidden.kernel[...]),np.asarray(model.hidden.kernel[...]))

# Reference solution. Try the exercise before reading this.
single=jnp.array([[.3,-.7]])
assert model(single).shape==(1,)
assert model(jnp.repeat(single,5,axis=0)).shape==(5,)
assert sum(a.size for a in jax.tree.leaves(nnx.state(model,nnx.Param)))==13

# Reference practice: Change a clone without changing its source
original_leaves=[np.array(a,copy=True) for a in jax.tree.leaves(nnx.state(model,nnx.Param))]
old_clone=np.asarray(clone(batch)).copy()
clone.out.bias[...] = clone.out.bias[...] + 1.
np.testing.assert_allclose(clone(batch),old_clone+1.,rtol=1e-5,atol=1e-6)
for before,after in zip(original_leaves,jax.tree.leaves(nnx.state(model,nnx.Param))):
    np.testing.assert_array_equal(before,after)
print('Clone logits shifted by one; original parameters unchanged.')

# Reference practice: Diagnose a shared module reference
saved_count=int(model.calls[...])
shared=model
shared(batch,record=True)
assert int(model.calls[...])==saved_count+1
graph,state=nnx.split(model)
independent=nnx.merge(graph,jax.tree.map(lambda a:jnp.array(a,copy=True),state))
independent(batch,record=True)
assert int(model.calls[...])==saved_count+1
assert int(independent.calls[...])==saved_count+2
print("PASS: networks-02")
