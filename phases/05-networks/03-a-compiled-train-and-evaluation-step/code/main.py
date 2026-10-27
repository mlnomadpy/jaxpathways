"""A compiled train and evaluation step: worked experiments and reference solutions. CPU checks."""

# 1. Construct data, model, and optimizer
# Step 1 : 1. Construct data, model, and optimizer
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
# Define `Classifier` module / container with explicit state and forward pass:
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
# Step 2 : 2. Define the two compiled functions: The 80-update counter is an optimizer invariant.
# Define and JIT-compile `train_step(m, o, x, y)` so XLA traces and fuses the operations:
@nnx.jit
def train_step(m,o,x,y):
    value,grads=nnx.value_and_grad(objective)(m,x,y)
    # Apply the parameter gradients to the mutable Flax model and optimizer state.
    o.update(m,grads)
    return value
# Define and JIT-compile `evaluate(m, x, y)` so XLA traces and fuses the operations:
@nnx.jit
def evaluate(m,x,y):
    scores=m(x)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(scores,y)),jnp.mean((scores>0)==y)
def snapshot(m):return [np.array(a,copy=True) for a in jax.tree.leaves(nnx.state(m))]
before_loss=float(objective(model,train_x,train_y))
for _ in range(80):train_step(model,optimizer,train_x,train_y)
assert int(optimizer.step[...])==80

# 3. Verify held-out metrics and state isolation
# Step 3 : 3. Verify held-out metrics and state isolation: The synthetic fixture normally exceeds 0.85 held-out accuracy.
frozen=snapshot(model)
step_count=int(optimizer.step[...])
test_loss,test_accuracy=evaluate(model,test_x,test_y)
for a,b in zip(frozen,snapshot(model)):np.testing.assert_array_equal(a,b)
assert int(optimizer.step[...])==step_count
# Transfer the device result to the host for an independent NumPy comparison.
# Transfer the device result to the host for an independent NumPy comparison.
scores=np.asarray(model(test_x))
labels=np.asarray(test_y)
np.testing.assert_allclose(test_loss,np.mean(np.logaddexp(0.,scores)-labels*scores),rtol=1e-5,atol=1e-6)
np.testing.assert_allclose(test_accuracy,np.mean((scores>0)==labels),atol=1e-6)
assert float(test_accuracy)>.85 and float(objective(model,train_x,train_y))<before_loss*.3
print('Updates:',step_count,'held-out loss:',float(test_loss),'accuracy:',float(test_accuracy))

# Step 1 : 1. Construct data, model, and optimizer
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
# Define `Classifier` module / container with explicit state and forward pass:
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

# Step 2 : 2. Define the two compiled functions: The 80-update counter is an optimizer invariant.
# Define and JIT-compile `train_step(m, o, x, y)` so XLA traces and fuses the operations:
@nnx.jit
def train_step(m,o,x,y):
    value,grads=nnx.value_and_grad(objective)(m,x,y)
    # Apply the parameter gradients to the mutable Flax model and optimizer state.
    o.update(m,grads)
    return value
# Define and JIT-compile `evaluate(m, x, y)` so XLA traces and fuses the operations:
@nnx.jit
def evaluate(m,x,y):
    scores=m(x)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(scores,y)),jnp.mean((scores>0)==y)
def snapshot(m):return [np.array(a,copy=True) for a in jax.tree.leaves(nnx.state(m))]
before_loss=float(objective(model,train_x,train_y))
for _ in range(80):train_step(model,optimizer,train_x,train_y)
assert int(optimizer.step[...])==80

# Step 3 : 3. Verify held-out metrics and state isolation: The synthetic fixture normally exceeds 0.85 held-out accuracy.
frozen=snapshot(model)
step_count=int(optimizer.step[...])
test_loss,test_accuracy=evaluate(model,test_x,test_y)
for a,b in zip(frozen,snapshot(model)):np.testing.assert_array_equal(a,b)
assert int(optimizer.step[...])==step_count
# Transfer the device result to the host for an independent NumPy comparison.
# Transfer the device result to the host for an independent NumPy comparison.
scores=np.asarray(model(test_x))
labels=np.asarray(test_y)
np.testing.assert_allclose(test_loss,np.mean(np.logaddexp(0.,scores)-labels*scores),rtol=1e-5,atol=1e-6)
np.testing.assert_allclose(test_accuracy,np.mean((scores>0)==labels),atol=1e-6)
assert float(test_accuracy)>.85 and float(objective(model,train_x,train_y))<before_loss*.3
print('Updates:',step_count,'held-out loss:',float(test_loss),'accuracy:',float(test_accuracy))

# Figure data experiment
axis = jnp.linspace(-3.0, 3.0, 61)
gx, gy = jnp.meshgrid(axis, axis)
grid = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)
# Rearrange tensor axes to match the required layout for `prob`.
prob = jax.nn.sigmoid(model(grid)).reshape(gx.shape)
visual_data = {'kind': 'field', 'values': prob.tolist(), 'extent': [-3.0, 3.0, -3.0, 3.0], 'xlabel': 'feature 0', 'ylabel': 'feature 1', 'unit': 'P(label 1)', 'points': test_x.tolist(), 'labels': test_y.tolist()}

# Experiment: Replay the complete update sequence
# Experiment : Replay the complete update sequence: Replay reconstructs optimizer state from the start.
replay,replay_optimizer=initialize()
for _ in range(80):train_step(replay,replay_optimizer,train_x,train_y)
for a,b in zip(snapshot(model),snapshot(replay)):np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
np.testing.assert_allclose(evaluate(replay,test_x,test_y),evaluate(model,test_x,test_y),rtol=1e-6)

# Experiment: Duplicating evaluation data
# Experiment : Duplicating evaluation data: Means normalize the observation count.
doubled=evaluate(model,jnp.concatenate([test_x,test_x]),jnp.concatenate([test_y,test_y]))
np.testing.assert_allclose(doubled,evaluate(model,test_x,test_y),rtol=1e-5,atol=1e-6)

# Experiment: Stateful evaluation: running statistics and dropout keys
# Experiment : Stateful evaluation: BatchNorm statistics and dropout RNG are state, distinct from parameters.
from flax import nnx as state_nnx
import jax as state_jax
import jax.numpy as state_jnp
import numpy as state_np
import optax as state_optax
class StatefulProbe(state_nnx.Module):
    def __init__(self, seed=4):
        rngs = state_nnx.Rngs(params=seed, dropout=seed+1)
        self.hidden = state_nnx.Linear(2, 4, rngs=rngs)
        self.norm = state_nnx.BatchNorm(num_features=4, rngs=rngs)
        self.dropout = state_nnx.Dropout(rate=0.5, rngs=rngs)
        self.out = state_nnx.Linear(4, 1, rngs=rngs)
    def __call__(self, x, *, training):
        x = self.norm(self.hidden(x), use_running_average=not training)
        x = self.dropout(state_nnx.relu(x), deterministic=not training)
        return self.out(x).squeeze(-1)
def stateful_loss(model, x, y):
    return state_jnp.mean(state_optax.sigmoid_binary_cross_entropy(model(x, training=True), y))
@state_nnx.jit
def stateful_train_step(model, optimizer, x, y):
    loss, grads = state_nnx.value_and_grad(stateful_loss)(model, x, y)
    optimizer.update(model, grads)
    return loss
def copy_state(state):
    return state_jax.tree.map(lambda value: state_jnp.array(value, copy=True), state)
def same_state(left, right):
    def comparable(value):
        if state_jax.dtypes.issubdtype(value.dtype, state_jax.dtypes.prng_key):
            return state_jax.random.key_data(value)
        return value
    return all(bool(state_np.array_equal(comparable(a), comparable(b))) for a, b in zip(state_jax.tree.leaves(left), state_jax.tree.leaves(right)))
state_x = state_jnp.asarray([[0., 1.], [1., 2.], [-1., 1.], [2., -1.], [1., -2.], [-2., 0.], [0., -1.], [2., 2.]])
state_y = state_jnp.asarray([1., 1., 0., 1., 0., 0., 0., 1.])
state_model = StatefulProbe()
state_optimizer = state_nnx.Optimizer(state_model, state_optax.sgd(0.02), wrt=state_nnx.Param)
for _ in range(3):
    stateful_train_step(state_model, state_optimizer, state_x, state_y)
params_before = copy_state(state_nnx.state(state_model, state_nnx.Param))
stats_before = copy_state(state_nnx.state(state_model, state_nnx.BatchStat))
rng_before = copy_state(state_nnx.state(state_model, state_nnx.RngState))
eval_a = state_model(state_x, training=False)
eval_b = state_model(state_x, training=False)
state_np.testing.assert_array_equal(eval_a, eval_b)
assert same_state(params_before, state_nnx.state(state_model, state_nnx.Param))
assert same_state(stats_before, state_nnx.state(state_model, state_nnx.BatchStat))
assert same_state(rng_before, state_nnx.state(state_model, state_nnx.RngState))
# Negative control: training mode on an independent clone must change BatchStat and consume a dropout key.
graphdef, state_snapshot = state_nnx.split(state_model)
bad_mode_model = state_nnx.merge(graphdef, copy_state(state_snapshot))
bad_stats = copy_state(state_nnx.state(bad_mode_model, state_nnx.BatchStat))
bad_rng = copy_state(state_nnx.state(bad_mode_model, state_nnx.RngState))
_ = bad_mode_model(state_x, training=True)
assert not same_state(bad_stats, state_nnx.state(bad_mode_model, state_nnx.BatchStat))
assert not same_state(bad_rng, state_nnx.state(bad_mode_model, state_nnx.RngState))
assert same_state(params_before, state_nnx.state(state_model, state_nnx.Param))
print('Deterministic evaluation repeated exactly; Param, BatchStat, and RngState unchanged. Negative control changed BatchStat and RngState; parameters stayed separate.')

# Reference solution. Try the exercise before reading this.
# Exercise solution
old=snapshot(model)
pieces=[evaluate(model,test_x[i:i+16],test_y[i:i+16]) for i in range(0,48,16)]
# Reduce along axis=0 to compute `aggregate`.
aggregate=np.mean(np.asarray(pieces),axis=0)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(aggregate,np.asarray(evaluate(model,test_x,test_y)),rtol=1e-5,atol=1e-6)
for a,b in zip(old,snapshot(model)):np.testing.assert_array_equal(a,b)

# Reference practice: Aggregate unequal batches
sizes=np.array([7,41])
# Transfer the device result to the host for an independent NumPy comparison.
metrics=np.asarray([evaluate(model,test_x[:7],test_y[:7]),evaluate(model,test_x[7:],test_y[7:])])
# Reduce along axis=0 to compute `weighted`.
weighted=(metrics*sizes[:,None]).sum(axis=0)/sizes.sum()
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(weighted,np.asarray(evaluate(model,test_x,test_y)),rtol=1e-5,atol=1e-6)
correct=np.array([6,20])
counts=np.array([7,41])
expected=26/48
np.testing.assert_allclose(correct.sum()/counts.sum(),expected)
assert not np.isclose(np.mean(correct/counts),expected)
print('Count-weighted held-out metrics:',weighted,'constructed accuracy:',expected)

# Reference practice: Catch an evaluation function that trains
# Catch an evaluation function that trains (Intermediate): A plausible returned loss is separate from evaluation.
graph,state=nnx.split(model)
bad_model=nnx.merge(graph,jax.tree.map(lambda a:jnp.array(a,copy=True),state))
bad_optimizer=nnx.Optimizer(bad_model,optax.adam(.03),wrt=nnx.Param)
prior=snapshot(bad_model)
train_step(bad_model,bad_optimizer,test_x,test_y)
assert int(bad_optimizer.step[...])==1
assert any(not np.array_equal(a,b) for a,b in zip(prior,snapshot(bad_model)))
assert int(optimizer.step[...])==80
print("PASS: networks-03")
