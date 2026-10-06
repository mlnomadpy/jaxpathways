"""A compiled train and evaluation step: worked experiments and reference solutions. CPU checks."""

# 1. Construct data, model, and optimizer
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
def dataset(seed,n=48):
    a=np.random.default_rng(seed).normal(size=(n,2)).astype(np.float32)
    b=(a[:,0]+.5*a[:,1]>0).astype(np.float32)
    return jnp.array(a),jnp.array(b)
train_x,train_y=dataset(10)
test_x,test_y=dataset(11)
class Classifier(nnx.Module):
    def __init__(self,seed=0):
        rngs=nnx.Rngs(seed)
        self.hidden=nnx.Linear(2,8,rngs=rngs)
        self.out=nnx.Linear(8,1,rngs=rngs)
    def __call__(self,x):return self.out(nnx.tanh(self.hidden(x))).squeeze(-1)
def objective(m,x,y):return jnp.mean(optax.sigmoid_binary_cross_entropy(m(x),y))
def initialize():
    m=Classifier(0)
    return m,nnx.Optimizer(m,optax.adam(.03),wrt=nnx.Param)
model,optimizer=initialize()

# 2. Define the two compiled functions
@nnx.jit
def train_step(m,o,x,y):
    value,grads=nnx.value_and_grad(objective)(m,x,y)
    o.update(m,grads)
    return value
@nnx.jit
def evaluate(m,x,y):
    scores=m(x)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(scores,y)),jnp.mean((scores>0)==y)
def snapshot(m):return [np.array(a,copy=True) for a in jax.tree.leaves(nnx.state(m))]
before_loss=float(objective(model,train_x,train_y))
for _ in range(80):train_step(model,optimizer,train_x,train_y)
assert int(optimizer.step[...])==80

# 3. Verify held-out metrics and state isolation
frozen=snapshot(model);step_count=int(optimizer.step[...])
test_loss,test_accuracy=evaluate(model,test_x,test_y)
for a,b in zip(frozen,snapshot(model)):np.testing.assert_array_equal(a,b)
assert int(optimizer.step[...])==step_count
scores=np.asarray(model(test_x));labels=np.asarray(test_y)
np.testing.assert_allclose(test_loss,np.mean(np.logaddexp(0.,scores)-labels*scores),rtol=1e-5,atol=1e-6)
np.testing.assert_allclose(test_accuracy,np.mean((scores>0)==labels),atol=1e-6)
assert float(test_accuracy)>.85 and float(objective(model,train_x,train_y))<before_loss*.3
print('Updates:',step_count,'held-out loss:',float(test_loss),'accuracy:',float(test_accuracy))

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
def dataset(seed,n=48):
    a=np.random.default_rng(seed).normal(size=(n,2)).astype(np.float32)
    b=(a[:,0]+.5*a[:,1]>0).astype(np.float32)
    return jnp.array(a),jnp.array(b)
train_x,train_y=dataset(10)
test_x,test_y=dataset(11)
class Classifier(nnx.Module):
    def __init__(self,seed=0):
        rngs=nnx.Rngs(seed)
        self.hidden=nnx.Linear(2,8,rngs=rngs)
        self.out=nnx.Linear(8,1,rngs=rngs)
    def __call__(self,x):return self.out(nnx.tanh(self.hidden(x))).squeeze(-1)
def objective(m,x,y):return jnp.mean(optax.sigmoid_binary_cross_entropy(m(x),y))
def initialize():
    m=Classifier(0)
    return m,nnx.Optimizer(m,optax.adam(.03),wrt=nnx.Param)
model,optimizer=initialize()

@nnx.jit
def train_step(m,o,x,y):
    value,grads=nnx.value_and_grad(objective)(m,x,y)
    o.update(m,grads)
    return value
@nnx.jit
def evaluate(m,x,y):
    scores=m(x)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(scores,y)),jnp.mean((scores>0)==y)
def snapshot(m):return [np.array(a,copy=True) for a in jax.tree.leaves(nnx.state(m))]
before_loss=float(objective(model,train_x,train_y))
for _ in range(80):train_step(model,optimizer,train_x,train_y)
assert int(optimizer.step[...])==80

frozen=snapshot(model);step_count=int(optimizer.step[...])
test_loss,test_accuracy=evaluate(model,test_x,test_y)
for a,b in zip(frozen,snapshot(model)):np.testing.assert_array_equal(a,b)
assert int(optimizer.step[...])==step_count
scores=np.asarray(model(test_x));labels=np.asarray(test_y)
np.testing.assert_allclose(test_loss,np.mean(np.logaddexp(0.,scores)-labels*scores),rtol=1e-5,atol=1e-6)
np.testing.assert_allclose(test_accuracy,np.mean((scores>0)==labels),atol=1e-6)
assert float(test_accuracy)>.85 and float(objective(model,train_x,train_y))<before_loss*.3
print('Updates:',step_count,'held-out loss:',float(test_loss),'accuracy:',float(test_accuracy))

# Figure data experiment
axis = jnp.linspace(-3.0, 3.0, 61)
gx, gy = jnp.meshgrid(axis, axis)
grid = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)
prob = jax.nn.sigmoid(model(grid)).reshape(gx.shape)
visual_data = {'kind': 'field', 'values': prob.tolist(), 'extent': [-3.0, 3.0, -3.0, 3.0], 'xlabel': 'feature 0', 'ylabel': 'feature 1', 'unit': 'P(label 1)', 'points': test_x.tolist(), 'labels': test_y.tolist()}

# Experiment: Replay the complete update sequence
replay,replay_optimizer=initialize()
for _ in range(80):train_step(replay,replay_optimizer,train_x,train_y)
for a,b in zip(snapshot(model),snapshot(replay)):np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
np.testing.assert_allclose(evaluate(replay,test_x,test_y),evaluate(model,test_x,test_y),rtol=1e-6)

# Experiment: Duplicating evaluation data
doubled=evaluate(model,jnp.concatenate([test_x,test_x]),jnp.concatenate([test_y,test_y]))
np.testing.assert_allclose(doubled,evaluate(model,test_x,test_y),rtol=1e-5,atol=1e-6)

# Reference solution. Try the exercise before reading this.
old=snapshot(model)
pieces=[evaluate(model,test_x[i:i+16],test_y[i:i+16]) for i in range(0,48,16)]
aggregate=np.mean(np.asarray(pieces),axis=0)
np.testing.assert_allclose(aggregate,np.asarray(evaluate(model,test_x,test_y)),rtol=1e-5,atol=1e-6)
for a,b in zip(old,snapshot(model)):np.testing.assert_array_equal(a,b)

# Reference practice: Aggregate unequal batches
sizes=np.array([7,41])
metrics=np.asarray([evaluate(model,test_x[:7],test_y[:7]),evaluate(model,test_x[7:],test_y[7:])])
weighted=(metrics*sizes[:,None]).sum(axis=0)/sizes.sum()
np.testing.assert_allclose(weighted,np.asarray(evaluate(model,test_x,test_y)),rtol=1e-5,atol=1e-6)
correct=np.array([6,20]);counts=np.array([7,41])
expected=26/48
np.testing.assert_allclose(correct.sum()/counts.sum(),expected)
assert not np.isclose(np.mean(correct/counts),expected)
print('Count-weighted held-out metrics:',weighted,'constructed accuracy:',expected)

# Reference practice: Catch an evaluation function that trains
graph,state=nnx.split(model)
bad_model=nnx.merge(graph,jax.tree.map(lambda a:jnp.array(a,copy=True),state))
bad_optimizer=nnx.Optimizer(bad_model,optax.adam(.03),wrt=nnx.Param)
prior=snapshot(bad_model)
train_step(bad_model,bad_optimizer,test_x,test_y)
assert int(bad_optimizer.step[...])==1
assert any(not np.array_equal(a,b) for a,b in zip(prior,snapshot(bad_model)))
assert int(optimizer.step[...])==80
print("PASS: networks-03")
