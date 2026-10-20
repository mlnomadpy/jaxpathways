"""Train a CNN on a small dataset: worked experiments and reference solutions. CPU checks."""

# 1. Generate two disjoint synthetic splits
# Step 1 — 1. Generate two disjoint synthetic splits: Two seeds produce separate examples.
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
def bars(seed,n,noise=.08):
    # Draw pseudorandom samples for `rng` using the explicit RNG state.
    rng=np.random.default_rng(seed)
    # Allocate initialized array `images` with the specified shape and dtype.
    images=np.zeros((n,8,8,1),np.float32)
    labels=np.arange(n,dtype=np.int32)%2
    for i,label in enumerate(labels):
        # Draw pseudorandom samples for `position` using the explicit RNG state.
        position=int(rng.integers(2,6))
        if label==0:images[i,position,:,0]=1.
        else:images[i,:,position,0]=1.
    # Accumulate the next contribution into `images`.
    images+=rng.normal(0,noise,images.shape).astype(np.float32)
    return jnp.array(images),jnp.array(labels)
train_x,train_y=bars(20,48)
test_x,test_y=bars(21,24)
# Check tensor shape invariant: `train_x.shape==(48,8,8,1) and test_x.shape==(24,8,8,1)`
assert train_x.shape==(48,8,8,1) and test_x.shape==(24,8,8,1)
assert not np.array_equal(np.asarray(train_x[:24]),np.asarray(test_x))

# 2. Build and independently inspect the convolution
# Step 2 — 2. Build and independently inspect the convolution: The first output at (0,0) consumes exactly the top-left 3\times3...
# Define `BarCNN` module / container with explicit state and forward pass:
class BarCNN(nnx.Module):
    def __init__(self):
        rngs=nnx.Rngs(0)
        # Combine or mask array elements to form `self.conv`.
        self.conv=nnx.Conv(1,4,(3,3),padding='VALID',rngs=rngs)
        self.head=nnx.Linear(4,2,rngs=rngs)
    def __call__(self,x):
        # Apply nonlinear activation or probability normalization to compute `features`.
        features=jnp.mean(nnx.relu(self.conv(x)),axis=(1,2))
        return self.head(features)
model=BarCNN()
raw=model.conv(train_x[:1])
# Check tensor shape invariant: `raw.shape==(1,6,6,4)`
assert raw.shape==(1,6,6,4)
# Convert `patch` to a host NumPy array for inspection or verification.
patch=np.asarray(train_x[0,:3,:3,:])
# Convert `kernel` to a host NumPy array for inspection or verification.
kernel=np.asarray(model.conv.kernel[...])
# Aggregate array values to compute `expected`.
expected=np.sum(patch*kernel[:,:,:,0])+np.asarray(model.conv.bias[...])[0]
np.testing.assert_allclose(raw[0,0,0,0],expected,rtol=1e-5,atol=1e-6)
def loss(m,x,y):return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(m(x),y))
# Configure or step the Optax optimizer state (`optimizer`).
optimizer=nnx.Optimizer(model,optax.adam(.03),wrt=nnx.Param)
@nnx.jit
def train_step(m,o,x,y):
    # Evaluate both scalar loss and parameter gradients in one pass (`(value, grads)`).
    value,grads=nnx.value_and_grad(loss)(m,x,y)
    # Update state in place with the new values.
    o.update(m,grads)
    return value

# 3. Fit, then measure the held-out split once
# Step 3 — 3. Fit, then measure the held-out split once: The output is a measured CPU result for synthetic bars.
initial=float(loss(model,train_x,train_y))
# Execute `for _ in range(80):train_step(model,optimizer,train_x,train_y)`.
for _ in range(80):train_step(model,optimizer,train_x,train_y)
# Convert `scores` to a host NumPy array for inspection or verification.
# Convert `labels` to a host NumPy array for inspection or verification.
scores=np.asarray(model(test_x))
labels=np.asarray(test_y)
predictions=scores.argmax(axis=-1)
# Reduce along axis=-1 to compute `shifted`.
shifted=scores-scores.max(axis=-1,keepdims=True)
# Reduce along axis=-1 to compute `reference_loss`.
reference_loss=np.mean(np.log(np.exp(shifted).sum(axis=-1))-shifted[np.arange(len(labels)),labels])
np.testing.assert_allclose(loss(model,test_x,test_y),reference_loss,rtol=1e-5,atol=1e-6)
# Allocate initialized array `confusion` with the specified shape and dtype.
confusion=np.zeros((2,2),dtype=int)
np.add.at(confusion,(labels,predictions),1)
# Aggregate array values to compute `accuracy`.
accuracy=np.mean(predictions==labels)
assert confusion.sum()==24 and np.isclose(np.trace(confusion)/24,accuracy)
assert accuracy>=.9 and float(loss(model,train_x,train_y))<initial*.3
print('Held-out loss:',reference_loss,'accuracy:',accuracy,'confusion:\n',confusion)

# Step 1 — 1. Generate two disjoint synthetic splits: Two seeds produce separate examples.
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
def bars(seed,n,noise=.08):
    # Draw pseudorandom samples for `rng` using the explicit RNG state.
    rng=np.random.default_rng(seed)
    # Allocate initialized array `images` with the specified shape and dtype.
    images=np.zeros((n,8,8,1),np.float32)
    labels=np.arange(n,dtype=np.int32)%2
    for i,label in enumerate(labels):
        # Draw pseudorandom samples for `position` using the explicit RNG state.
        position=int(rng.integers(2,6))
        if label==0:images[i,position,:,0]=1.
        else:images[i,:,position,0]=1.
    # Accumulate the next contribution into `images`.
    images+=rng.normal(0,noise,images.shape).astype(np.float32)
    return jnp.array(images),jnp.array(labels)
train_x,train_y=bars(20,48)
test_x,test_y=bars(21,24)
# Check tensor shape invariant: `train_x.shape==(48,8,8,1) and test_x.shape==(24,8,8,1)`
assert train_x.shape==(48,8,8,1) and test_x.shape==(24,8,8,1)
assert not np.array_equal(np.asarray(train_x[:24]),np.asarray(test_x))

# Step 2 — 2. Build and independently inspect the convolution: The first output at (0,0) consumes exactly the top-left 3\times3...
# Define `BarCNN` module / container with explicit state and forward pass:
class BarCNN(nnx.Module):
    def __init__(self):
        rngs=nnx.Rngs(0)
        # Combine or mask array elements to form `self.conv`.
        self.conv=nnx.Conv(1,4,(3,3),padding='VALID',rngs=rngs)
        self.head=nnx.Linear(4,2,rngs=rngs)
    def __call__(self,x):
        # Apply nonlinear activation or probability normalization to compute `features`.
        features=jnp.mean(nnx.relu(self.conv(x)),axis=(1,2))
        return self.head(features)
model=BarCNN()
raw=model.conv(train_x[:1])
# Check tensor shape invariant: `raw.shape==(1,6,6,4)`
assert raw.shape==(1,6,6,4)
# Convert `patch` to a host NumPy array for inspection or verification.
patch=np.asarray(train_x[0,:3,:3,:])
# Convert `kernel` to a host NumPy array for inspection or verification.
kernel=np.asarray(model.conv.kernel[...])
# Aggregate array values to compute `expected`.
expected=np.sum(patch*kernel[:,:,:,0])+np.asarray(model.conv.bias[...])[0]
np.testing.assert_allclose(raw[0,0,0,0],expected,rtol=1e-5,atol=1e-6)
def loss(m,x,y):return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(m(x),y))
# Configure or step the Optax optimizer state (`optimizer`).
optimizer=nnx.Optimizer(model,optax.adam(.03),wrt=nnx.Param)
@nnx.jit
def train_step(m,o,x,y):
    # Evaluate both scalar loss and parameter gradients in one pass (`(value, grads)`).
    value,grads=nnx.value_and_grad(loss)(m,x,y)
    # Update state in place with the new values.
    o.update(m,grads)
    return value

# Step 3 — 3. Fit, then measure the held-out split once: The output is a measured CPU result for synthetic bars.
initial=float(loss(model,train_x,train_y))
# Execute `for _ in range(80):train_step(model,optimizer,train_x,train_y)`.
for _ in range(80):train_step(model,optimizer,train_x,train_y)
# Convert `scores` to a host NumPy array for inspection or verification.
# Convert `labels` to a host NumPy array for inspection or verification.
scores=np.asarray(model(test_x))
labels=np.asarray(test_y)
predictions=scores.argmax(axis=-1)
# Reduce along axis=-1 to compute `shifted`.
shifted=scores-scores.max(axis=-1,keepdims=True)
# Reduce along axis=-1 to compute `reference_loss`.
reference_loss=np.mean(np.log(np.exp(shifted).sum(axis=-1))-shifted[np.arange(len(labels)),labels])
np.testing.assert_allclose(loss(model,test_x,test_y),reference_loss,rtol=1e-5,atol=1e-6)
# Allocate initialized array `confusion` with the specified shape and dtype.
confusion=np.zeros((2,2),dtype=int)
np.add.at(confusion,(labels,predictions),1)
# Aggregate array values to compute `accuracy`.
accuracy=np.mean(predictions==labels)
assert confusion.sum()==24 and np.isclose(np.trace(confusion)/24,accuracy)
assert accuracy>=.9 and float(loss(model,train_x,train_y))<initial*.3
print('Held-out loss:',reference_loss,'accuracy:',accuracy,'confusion:\n',confusion)

# Figure data experiment
# Compute figure data for: See the image task and its classification errors
visual_data = {'kind': 'panels', 'panels': [{'kind': 'heatmap', 'title': 'Held-out image 0', 'values': test_x[0, :, :, 0].tolist(), 'unit': 'pixel value'}, {'kind': 'heatmap', 'title': 'Held-out confusion', 'values': confusion.tolist(), 'rows': ['actual horizontal', 'actual vertical'], 'columns': ['pred. horizontal', 'pred. vertical'], 'unit': 'example count'}]}

# Experiment: Trace a second patch and a second filter
# Experiment — Trace a second patch and a second filter: The same learned filter is reused at each valid location;...
patch=np.asarray(test_x[0,2:5,3:6,:])
# Convert `weights` to a host NumPy array for inspection or verification.
weights=np.asarray(model.conv.kernel[...])[:,:,:,2]
# Aggregate array values to compute `expected`.
expected=np.sum(patch*weights)+np.asarray(model.conv.bias[...])[2]
np.testing.assert_allclose(model.conv(test_x[:1])[0,2,3,2],expected,rtol=1e-5,atol=1e-6)

# Experiment: Predict the chance classifier
# Experiment — Predict the chance classifier: A majority or constant-label baseline is part of evaluation.
always_zero=np.zeros_like(labels)
# Aggregate array values to compute `baseline`.
baseline=np.mean(always_zero==labels)
assert baseline==.5
# Allocate initialized array `baseline_confusion` with the specified shape and dtype.
baseline_confusion=np.zeros((2,2),dtype=int)
np.add.at(baseline_confusion,(labels,always_zero),1)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(baseline_confusion,np.array([[12,0],[12,0]]))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Apply the trained model to one 10\times10 image.
larger=jnp.zeros((1,10,10,1))
larger=larger.at[0,4,:,0].set(1.)
# Check tensor shape invariant: `model.conv(larger).shape==(1,8,8,4)`
assert model.conv(larger).shape==(1,8,8,4)
# Check tensor shape invariant: `model(larger).shape==(1,2)`
assert model(larger).shape==(1,2)
assert sum(a.size for a in jax.tree.leaves(nnx.state(model,nnx.Param)))==50

# Reference practice: Read a confusion matrix as conditional rates
# Read a confusion matrix as conditional rates (Transfer): Each row asks where examples of one true class went.
# Reduce along axis=1 to compute `counts`.
counts=confusion.sum(axis=1)
# Combine or mask array elements to form `rates`.
rates=np.divide(confusion,counts[:,None],out=np.full(confusion.shape,np.nan),where=counts[:,None]!=0)
# Allocate initialized array `` with the specified shape and dtype.
np.testing.assert_allclose(rates.sum(axis=1),np.ones(2))
# Execute `np.testing.assert_allclose((np.diag(rates)*counts).sum()/counts.sum(),accuracy)`.
np.testing.assert_allclose((np.diag(rates)*counts).sum()/counts.sum(),accuracy)
missing=np.array([[3,1],[0,0]])
# Reduce along axis=1 to compute `n`.
n=missing.sum(axis=1)
# Combine or mask array elements to form `missing_rates`.
missing_rates=np.divide(missing,n[:,None],out=np.full(missing.shape,np.nan),where=n[:,None]!=0)
# Ensure all array elements remain finite: `np.isnan(missing_rates[1]).all()`
assert np.isnan(missing_rates[1]).all()
print('True-class counts:',counts,'row-normalized confusion:',rates)
print('Absent class recall: undefined, not zero.')

# Reference practice: Catch a channel-order mistake
# Catch a channel-order mistake (Intermediate): The mistaken array has eight channels where the layer...
wrong=jnp.transpose(test_x[:2],(0,3,1,2))
caught=False
# Run the boundary check and catch the expected exception:
try:model(wrong)
except (ValueError,TypeError):caught=True
assert caught
# Rearrange tensor axes to match the required layout for `repaired`.
repaired=jnp.transpose(wrong,(0,2,3,1))
np.testing.assert_allclose(model(repaired),model(test_x[:2]),rtol=1e-6)
print("PASS: networks-04")
