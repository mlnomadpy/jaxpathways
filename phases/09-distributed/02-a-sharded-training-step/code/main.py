"""A sharded training step: worked experiments and reference solutions. CPU checks."""

# Start a fresh CPU runtime
# Step 1 — Start a fresh CPU runtime: Configuration happens before devices() or array creation...
# Import jax for this computation.
import jax
# Run this file in a fresh process; in a notebook restart the kernel first.
jax.config.update("jax_platforms", "cpu")
# Update state in place with the new values.
jax.config.update("jax_num_cpu_devices", 4)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
# Query the active JAX devices into `devices`.
devices = jax.devices("cpu")
# Guard input contract (`len(devices) != 4`) and fail fast if violated.
if len(devices) != 4:
    raise RuntimeError("Expected four CPU devices. Restart the notebook kernel, then Run all; do not run an earlier JAX cell first.")
# Configure multi-device placement / sharding specification (`mesh`).
mesh = Mesh(np.array(devices), ("data",))
# Configure multi-device placement / sharding specification (`rows`).
rows = NamedSharding(mesh, P("data", None))
# Configure multi-device placement / sharding specification (`replicated`).
replicated = NamedSharding(mesh, P())

# Place and compute
# Step 2 — Place and compute: Mesh axis names describe placement.
# Compute `host_x` from `np.array([[1,0],[0,1],[1,1],[2,0],[0,2],[2,1],[1,2],...`
host_x = np.array([[1,0],[0,1],[1,1],[2,0],[0,2],[2,1],[1,2],[2,2]],dtype=np.float32)
# Compute `true_w` from `np.array([2.,-1.],dtype=np.float32)`
true_w = np.array([2.,-1.],dtype=np.float32)
# Perform matrix / vector contraction (`@`) to compute `host_y`.
host_y = host_x @ true_w
# Configure multi-device placement / sharding specification (`labels_sharding`).
labels_sharding = NamedSharding(mesh,P("data"))
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host_x, rows)
# Place `y` explicitly onto the target JAX device.
y = jax.device_put(host_y, labels_sharding)
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(np.zeros(2,dtype=np.float32),replicated)
# Function `loss(weights, features, labels)` implementing this stage's computation:
def loss(weights, features, labels):
    # Return `jnp.mean((features @ weights - labels) ** 2)` to the caller.
    return jnp.mean((features @ weights-labels)**2)
# Define `update(weights, features, labels)` to evaluate the objective and its automatic derivatives:
def update(weights, features, labels):
    # Differentiate the objective to obtain `(value, gradient)` via automatic differentiation.
    value, gradient = jax.value_and_grad(loss)(weights,features,labels)
    # Return `(weights - 0.1 * gradient, value, gradient)` to the caller.
    return weights - 0.1*gradient, value, gradient
# Wrap with `jax.jit` (`train_step`) so XLA traces and compiles the function.
train_step = jax.jit(update, in_shardings=(replicated,rows,labels_sharding),out_shardings=(replicated,replicated,replicated))
# Run `train_step` to compute `(new_w, before, gradient)`.
new_w, before, gradient = train_step(w,x,y)

# Inspect and verify
# Step 3 — Inspect and verify: Assertions compare against host calculations or hand-derived...
expected_gradient = 2 * host_x.T @ (-host_y) / len(host_y)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(gradient),expected_gradient,rtol=1e-6,atol=1e-6)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(new_w),-0.1*expected_gradient,rtol=1e-6,atol=1e-6)
# Aggregate array values to compute `expected_loss`.
expected_loss = np.mean(host_y**2)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(before), expected_loss,rtol=1e-6)
# Assert invariant `float(loss(new_w,x,y)) < float(before)` holds
assert float(loss(new_w,x,y)) < float(before)
# Iterate over `shard` to step through the computation:
for shard in new_w.addressable_shards:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(shard.data),np.asarray(new_w))
# Print the observed values to compare against the expected result.
print("Initial loss:",float(before))
# Print diagnostic summary of the computed outputs.
print("Gradient:",np.asarray(gradient),"updated weights:",np.asarray(new_w))
# Print diagnostic summary of the computed outputs.
print("Updated loss:",float(loss(new_w,x,y)))

# Step 1 — Start a fresh CPU runtime: Configuration happens before devices() or array creation...
# Import jax for this computation.
import jax
# Run this file in a fresh process; in a notebook restart the kernel first.
jax.config.update("jax_platforms", "cpu")
# Update state in place with the new values.
jax.config.update("jax_num_cpu_devices", 4)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
# Query the active JAX devices into `devices`.
devices = jax.devices("cpu")
# Guard input contract (`len(devices) != 4`) and fail fast if violated.
if len(devices) != 4:
    raise RuntimeError("Expected four CPU devices. Restart the notebook kernel, then Run all; do not run an earlier JAX cell first.")
# Configure multi-device placement / sharding specification (`mesh`).
mesh = Mesh(np.array(devices), ("data",))
# Configure multi-device placement / sharding specification (`rows`).
rows = NamedSharding(mesh, P("data", None))
# Configure multi-device placement / sharding specification (`replicated`).
replicated = NamedSharding(mesh, P())

# Step 2 — Place and compute: Mesh axis names describe placement.
# Compute `host_x` from `np.array([[1,0],[0,1],[1,1],[2,0],[0,2],[2,1],[1,2],...`
host_x = np.array([[1,0],[0,1],[1,1],[2,0],[0,2],[2,1],[1,2],[2,2]],dtype=np.float32)
# Compute `true_w` from `np.array([2.,-1.],dtype=np.float32)`
true_w = np.array([2.,-1.],dtype=np.float32)
# Perform matrix / vector contraction (`@`) to compute `host_y`.
host_y = host_x @ true_w
# Configure multi-device placement / sharding specification (`labels_sharding`).
labels_sharding = NamedSharding(mesh,P("data"))
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host_x, rows)
# Place `y` explicitly onto the target JAX device.
y = jax.device_put(host_y, labels_sharding)
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(np.zeros(2,dtype=np.float32),replicated)
# Function `loss(weights, features, labels)` implementing this stage's computation:
def loss(weights, features, labels):
    # Return `jnp.mean((features @ weights - labels) ** 2)` to the caller.
    return jnp.mean((features @ weights-labels)**2)
# Define `update(weights, features, labels)` to evaluate the objective and its automatic derivatives:
def update(weights, features, labels):
    # Differentiate the objective to obtain `(value, gradient)` via automatic differentiation.
    value, gradient = jax.value_and_grad(loss)(weights,features,labels)
    # Return `(weights - 0.1 * gradient, value, gradient)` to the caller.
    return weights - 0.1*gradient, value, gradient
# Wrap with `jax.jit` (`train_step`) so XLA traces and compiles the function.
train_step = jax.jit(update, in_shardings=(replicated,rows,labels_sharding),out_shardings=(replicated,replicated,replicated))
# Run `train_step` to compute `(new_w, before, gradient)`.
new_w, before, gradient = train_step(w,x,y)

# Step 3 — Inspect and verify: Assertions compare against host calculations or hand-derived...
expected_gradient = 2 * host_x.T @ (-host_y) / len(host_y)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(gradient),expected_gradient,rtol=1e-6,atol=1e-6)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(new_w),-0.1*expected_gradient,rtol=1e-6,atol=1e-6)
# Aggregate array values to compute `expected_loss`.
expected_loss = np.mean(host_y**2)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(before), expected_loss,rtol=1e-6)
# Assert invariant `float(loss(new_w,x,y)) < float(before)` holds
assert float(loss(new_w,x,y)) < float(before)
# Iterate over `shard` to step through the computation:
for shard in new_w.addressable_shards:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(shard.data),np.asarray(new_w))
# Print the observed values to compare against the expected result.
print("Initial loss:",float(before))
# Print diagnostic summary of the computed outputs.
print("Gradient:",np.asarray(gradient),"updated weights:",np.asarray(new_w))
# Print diagnostic summary of the computed outputs.
print("Updated loss:",float(loss(new_w,x,y)))

# Figure data experiment
# Compute figure data for: Sharded differentiation matches the global reference
# Convert `visual_data` to a host NumPy array for inspection or verification.
visual_data = {'kind': 'bar', 'labels': ['weight 0', 'weight 1'], 'ylabel': 'gradient coordinate', 'series': [{'label': 'sharded JAX', 'y': np.asarray(gradient).tolist()}, {'label': 'NumPy reference', 'y': expected_gradient.tolist()}]}

# Experiment: Permutation across shards
# Experiment — Permutation across shards: The global mean is invariant to row order.
reverse_x = jax.device_put(host_x[::-1].copy(),rows)
# Place `reverse_y` explicitly onto the target JAX device.
reverse_y = jax.device_put(host_y[::-1].copy(),labels_sharding)
# Run `train_step` to compute `(reverse_w, reverse_loss, reverse_gradient)`.
reverse_w,reverse_loss,reverse_gradient = train_step(w,reverse_x,reverse_y)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(reverse_gradient),expected_gradient,rtol=1e-6,atol=1e-6)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(reverse_w),np.asarray(new_w),rtol=1e-6,atol=1e-6)
# Print the observed values to compare against the expected result.
print("Permutation preserves the global update")

# Experiment: Ten updates against a host training loop
# Experiment — Ten updates against a host training loop: Checking every update catches transient mismatches that a single...
cpu_weights = w
# Allocate initialized array `reference_weights` with the specified shape and dtype.
reference_weights = np.zeros(2,dtype=np.float32)
# Iterate over `step_index` to step through the computation:
for step_index in range(10):
    # Run `train_step` to compute `(cpu_weights, _, _)`.
    cpu_weights,_,_ = train_step(cpu_weights,x,y)
    # Perform matrix contraction / projection to compute `residual`.
    residual = host_x @ reference_weights-host_y
    # Accumulate the next contribution into `reference_weights`.
    reference_weights -= 0.1*(2*host_x.T @ residual/len(host_y))
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(cpu_weights),reference_weights,rtol=1e-5,atol=1e-6)
# Print the observed values to compare against the expected result.
print("Ten-step weights:",np.asarray(cpu_weights))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Run a changed target/initial-state update, then diagnose the...
# Compute `changed_y` from `host_x @ np.array([-1.,3.],dtype=np.float32)`
changed_y = host_x @ np.array([-1.,3.],dtype=np.float32)
# Compute `start` from `np.array([0.2,-0.1],dtype=np.float32)`
start = np.array([0.2,-0.1],dtype=np.float32)
# Define `changed_update(weights, features, labels)` to evaluate the objective and its automatic derivatives:
def changed_update(weights,features,labels):
    # Return `weights - 0.05 * jax.grad(loss)(weights, features, labels)` to the caller.
    return weights-0.05*jax.grad(loss)(weights,features,labels)
# Wrap with `jax.jit` (`changed_step`) so XLA traces and compiles the function.
changed_step = jax.jit(changed_update,in_shardings=(replicated,rows,labels_sharding),out_shardings=replicated)
# Place `result` explicitly onto the target JAX device.
result = changed_step(jax.device_put(start,replicated),x,jax.device_put(changed_y,labels_sharding))
# Perform matrix contraction / projection to compute `reference_gradient`.
reference_gradient = 2*host_x.T @ (host_x@start-changed_y)/len(changed_y)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(result),start-0.05*reference_gradient,rtol=1e-6,atol=1e-6)

# Compute `local_gradients` from `[]`
local_gradients = []
# Iterate over `(features, labels)` to step through the computation:
for features,labels in zip(np.split(host_x,4),np.split(host_y,4)):
    # Perform matrix contraction / projection to compute ``.
    local_gradients.append(2*features.T @ (-labels)/len(labels))
# Reduce along axis=0 to compute `wrong`.
wrong = np.sum(local_gradients,axis=0)
# Reduce along axis=0 to compute `right`.
right = np.mean(local_gradients,axis=0)
# Compute `np.testing.assert_allclose(wrong,4*expected_gradient,rtol` as `1e-6)`.
np.testing.assert_allclose(wrong,4*expected_gradient,rtol=1e-6)
# Compute `np.testing.assert_allclose(right,expected_gradient,rtol` as `1e-6)`.
np.testing.assert_allclose(right,expected_gradient,rtol=1e-6)
# Unequal partitions: weight local mean gradients by their observation counts.
parts = [(host_x[:3],host_y[:3]),(host_x[3:],host_y[3:])]
# Perform matrix contraction / projection to compute `weighted`.
weighted = sum(len(b)*(2*a.T @ (-b)/len(b)) for a,b in parts)/len(host_y)
# Compute `np.testing.assert_allclose(weighted,expected_gradient,rtol` as `1e-6)`.
np.testing.assert_allclose(weighted,expected_gradient,rtol=1e-6)
# Print diagnostic summary of the computed outputs.
print("Summed local means are four times too large; weighted aggregation repaired")

# Reference practice: Transfer to a different target model
# Transfer to a different target model (Challenge): The independent formula follows the new target, initial...
# Compute `changed_y` from `host_x @ np.array([-1.,3.],dtype=np.float32)`
changed_y = host_x @ np.array([-1.,3.],dtype=np.float32)
# Compute `start` from `np.array([0.2,-0.1],dtype=np.float32)`
start = np.array([0.2,-0.1],dtype=np.float32)
# Define `changed_update(weights, features, labels)` to evaluate the objective and its automatic derivatives:
def changed_update(weights,features,labels):
    # Return `weights - 0.05 * jax.grad(loss)(weights, features, labels)` to the caller.
    return weights-0.05*jax.grad(loss)(weights,features,labels)
# Wrap with `jax.jit` (`changed_step`) so XLA traces and compiles the function.
changed_step = jax.jit(changed_update,in_shardings=(replicated,rows,labels_sharding),out_shardings=replicated)
# Place `result` explicitly onto the target JAX device.
result = changed_step(jax.device_put(start,replicated),x,jax.device_put(changed_y,labels_sharding))
# Perform matrix contraction / projection to compute `reference_gradient`.
reference_gradient = 2*host_x.T @ (host_x@start-changed_y)/len(changed_y)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(result),start-0.05*reference_gradient,rtol=1e-6,atol=1e-6)

# Reference practice: Diagnose summed local means
# Diagnose summed local means (Challenge): Each local gradient already divides by its local count.
local_gradients = []
# Iterate over `(features, labels)` to step through the computation:
for features,labels in zip(np.split(host_x,4),np.split(host_y,4)):
    # Perform matrix contraction / projection to compute ``.
    local_gradients.append(2*features.T @ (-labels)/len(labels))
# Reduce along axis=0 to compute `wrong`.
wrong = np.sum(local_gradients,axis=0)
# Reduce along axis=0 to compute `right`.
right = np.mean(local_gradients,axis=0)
# Compute `np.testing.assert_allclose(wrong,4*expected_gradient,rtol` as `1e-6)`.
np.testing.assert_allclose(wrong,4*expected_gradient,rtol=1e-6)
# Compute `np.testing.assert_allclose(right,expected_gradient,rtol` as `1e-6)`.
np.testing.assert_allclose(right,expected_gradient,rtol=1e-6)
# Unequal partitions: weight local mean gradients by their observation counts.
parts = [(host_x[:3],host_y[:3]),(host_x[3:],host_y[3:])]
# Perform matrix contraction / projection to compute `weighted`.
weighted = sum(len(b)*(2*a.T @ (-b)/len(b)) for a,b in parts)/len(host_y)
# Compute `np.testing.assert_allclose(weighted,expected_gradient,rtol` as `1e-6)`.
np.testing.assert_allclose(weighted,expected_gradient,rtol=1e-6)
# Print the observed values to compare against the expected result.
print("Summed local means are four times too large; weighted aggregation repaired")
print("PASS: distributed-02")
