"""A sharded training step: worked experiments and reference solutions. CPU checks."""

# Start a fresh CPU runtime
import jax
# Run this file in a fresh process; in a notebook restart the kernel first.
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 4)
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
devices = jax.devices("cpu")
if len(devices) != 4:
    raise RuntimeError("Expected four CPU devices. Restart the notebook kernel, then Run all; do not run an earlier JAX cell first.")
mesh = Mesh(np.array(devices), ("data",))
rows = NamedSharding(mesh, P("data", None))
replicated = NamedSharding(mesh, P())

# Place and compute
host_x = np.array([[1,0],[0,1],[1,1],[2,0],[0,2],[2,1],[1,2],[2,2]],dtype=np.float32)
true_w = np.array([2.,-1.],dtype=np.float32)
host_y = host_x @ true_w
labels_sharding = NamedSharding(mesh,P("data"))
x = jax.device_put(host_x, rows)
y = jax.device_put(host_y, labels_sharding)
w = jax.device_put(np.zeros(2,dtype=np.float32),replicated)
def loss(weights, features, labels):
    return jnp.mean((features @ weights-labels)**2)
def update(weights, features, labels):
    value, gradient = jax.value_and_grad(loss)(weights,features,labels)
    return weights - 0.1*gradient, value, gradient
train_step = jax.jit(update, in_shardings=(replicated,rows,labels_sharding),out_shardings=(replicated,replicated,replicated))
new_w, before, gradient = train_step(w,x,y)

# Inspect and verify
expected_gradient = 2 * host_x.T @ (-host_y) / len(host_y)
np.testing.assert_allclose(np.asarray(gradient),expected_gradient,rtol=1e-6,atol=1e-6)
np.testing.assert_allclose(np.asarray(new_w),-0.1*expected_gradient,rtol=1e-6,atol=1e-6)
expected_loss = np.mean(host_y**2)
np.testing.assert_allclose(np.asarray(before), expected_loss,rtol=1e-6)
assert float(loss(new_w,x,y)) < float(before)
for shard in new_w.addressable_shards:
    np.testing.assert_allclose(np.asarray(shard.data),np.asarray(new_w))
print("Initial loss:",float(before))
print("Gradient:",np.asarray(gradient),"updated weights:",np.asarray(new_w))
print("Updated loss:",float(loss(new_w,x,y)))

import jax
# Run this file in a fresh process; in a notebook restart the kernel first.
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 4)
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
devices = jax.devices("cpu")
if len(devices) != 4:
    raise RuntimeError("Expected four CPU devices. Restart the notebook kernel, then Run all; do not run an earlier JAX cell first.")
mesh = Mesh(np.array(devices), ("data",))
rows = NamedSharding(mesh, P("data", None))
replicated = NamedSharding(mesh, P())

host_x = np.array([[1,0],[0,1],[1,1],[2,0],[0,2],[2,1],[1,2],[2,2]],dtype=np.float32)
true_w = np.array([2.,-1.],dtype=np.float32)
host_y = host_x @ true_w
labels_sharding = NamedSharding(mesh,P("data"))
x = jax.device_put(host_x, rows)
y = jax.device_put(host_y, labels_sharding)
w = jax.device_put(np.zeros(2,dtype=np.float32),replicated)
def loss(weights, features, labels):
    return jnp.mean((features @ weights-labels)**2)
def update(weights, features, labels):
    value, gradient = jax.value_and_grad(loss)(weights,features,labels)
    return weights - 0.1*gradient, value, gradient
train_step = jax.jit(update, in_shardings=(replicated,rows,labels_sharding),out_shardings=(replicated,replicated,replicated))
new_w, before, gradient = train_step(w,x,y)

expected_gradient = 2 * host_x.T @ (-host_y) / len(host_y)
np.testing.assert_allclose(np.asarray(gradient),expected_gradient,rtol=1e-6,atol=1e-6)
np.testing.assert_allclose(np.asarray(new_w),-0.1*expected_gradient,rtol=1e-6,atol=1e-6)
expected_loss = np.mean(host_y**2)
np.testing.assert_allclose(np.asarray(before), expected_loss,rtol=1e-6)
assert float(loss(new_w,x,y)) < float(before)
for shard in new_w.addressable_shards:
    np.testing.assert_allclose(np.asarray(shard.data),np.asarray(new_w))
print("Initial loss:",float(before))
print("Gradient:",np.asarray(gradient),"updated weights:",np.asarray(new_w))
print("Updated loss:",float(loss(new_w,x,y)))

# Figure data experiment
visual_data = {'kind': 'bar', 'labels': ['weight 0', 'weight 1'], 'ylabel': 'gradient coordinate', 'series': [{'label': 'sharded JAX', 'y': np.asarray(gradient).tolist()}, {'label': 'NumPy reference', 'y': expected_gradient.tolist()}]}

# Experiment: Permutation across shards
reverse_x = jax.device_put(host_x[::-1].copy(),rows)
reverse_y = jax.device_put(host_y[::-1].copy(),labels_sharding)
reverse_w,reverse_loss,reverse_gradient = train_step(w,reverse_x,reverse_y)
np.testing.assert_allclose(np.asarray(reverse_gradient),expected_gradient,rtol=1e-6,atol=1e-6)
np.testing.assert_allclose(np.asarray(reverse_w),np.asarray(new_w),rtol=1e-6,atol=1e-6)
print("Permutation preserves the global update")

# Experiment: Ten updates against a host training loop
cpu_weights = w
reference_weights = np.zeros(2,dtype=np.float32)
for step_index in range(10):
    cpu_weights,_,_ = train_step(cpu_weights,x,y)
    residual = host_x @ reference_weights-host_y
    reference_weights -= 0.1*(2*host_x.T @ residual/len(host_y))
    np.testing.assert_allclose(np.asarray(cpu_weights),reference_weights,rtol=1e-5,atol=1e-6)
print("Ten-step weights:",np.asarray(cpu_weights))

# Reference solution. Try the exercise before reading this.
changed_y = host_x @ np.array([-1.,3.],dtype=np.float32)
start = np.array([0.2,-0.1],dtype=np.float32)
def changed_update(weights,features,labels):
    return weights-0.05*jax.grad(loss)(weights,features,labels)
changed_step = jax.jit(changed_update,in_shardings=(replicated,rows,labels_sharding),out_shardings=replicated)
result = changed_step(jax.device_put(start,replicated),x,jax.device_put(changed_y,labels_sharding))
reference_gradient = 2*host_x.T @ (host_x@start-changed_y)/len(changed_y)
np.testing.assert_allclose(np.asarray(result),start-0.05*reference_gradient,rtol=1e-6,atol=1e-6)

local_gradients = []
for features,labels in zip(np.split(host_x,4),np.split(host_y,4)):
    local_gradients.append(2*features.T @ (-labels)/len(labels))
wrong = np.sum(local_gradients,axis=0)
right = np.mean(local_gradients,axis=0)
np.testing.assert_allclose(wrong,4*expected_gradient,rtol=1e-6)
np.testing.assert_allclose(right,expected_gradient,rtol=1e-6)
# Unequal partitions: weight local mean gradients by their observation counts.
parts = [(host_x[:3],host_y[:3]),(host_x[3:],host_y[3:])]
weighted = sum(len(b)*(2*a.T @ (-b)/len(b)) for a,b in parts)/len(host_y)
np.testing.assert_allclose(weighted,expected_gradient,rtol=1e-6)
print("Summed local means are four times too large; weighted aggregation repaired")

# Reference practice: Transfer to a different target model
changed_y = host_x @ np.array([-1.,3.],dtype=np.float32)
start = np.array([0.2,-0.1],dtype=np.float32)
def changed_update(weights,features,labels):
    return weights-0.05*jax.grad(loss)(weights,features,labels)
changed_step = jax.jit(changed_update,in_shardings=(replicated,rows,labels_sharding),out_shardings=replicated)
result = changed_step(jax.device_put(start,replicated),x,jax.device_put(changed_y,labels_sharding))
reference_gradient = 2*host_x.T @ (host_x@start-changed_y)/len(changed_y)
np.testing.assert_allclose(np.asarray(result),start-0.05*reference_gradient,rtol=1e-6,atol=1e-6)

# Reference practice: Diagnose summed local means
local_gradients = []
for features,labels in zip(np.split(host_x,4),np.split(host_y,4)):
    local_gradients.append(2*features.T @ (-labels)/len(labels))
wrong = np.sum(local_gradients,axis=0)
right = np.mean(local_gradients,axis=0)
np.testing.assert_allclose(wrong,4*expected_gradient,rtol=1e-6)
np.testing.assert_allclose(right,expected_gradient,rtol=1e-6)
# Unequal partitions: weight local mean gradients by their observation counts.
parts = [(host_x[:3],host_y[:3]),(host_x[3:],host_y[3:])]
weighted = sum(len(b)*(2*a.T @ (-b)/len(b)) for a,b in parts)/len(host_y)
np.testing.assert_allclose(weighted,expected_gradient,rtol=1e-6)
print("Summed local means are four times too large; weighted aggregation repaired")
print("PASS: distributed-02")
