"""Practice with four virtual CPU devices: worked experiments and reference solutions. CPU checks."""

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
# Construct and reshape `host` into the target tensor dimensions.
host = np.arange(8, dtype=np.float32).reshape(4, 2)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host, rows)
# Wrap with `jax.jit` (`y`) so XLA traces and compiles the function.
y = jax.jit(lambda a: 2 * a + 1, in_shardings=rows, out_shardings=rows)(x)
# Synchronize host execution until asynchronous device computation completes.
y.block_until_ready()

# Inspect and verify
# Step 3 — Inspect and verify: Assertions compare against host calculations or hand-derived...
# Print the observed values to compare against the expected result.
print("JAX version:", jax.__version__)
# Print diagnostic summary of the computed outputs.
print("CPU devices:", len(devices))
# Iterate over `shard` to step through the computation:
for shard in x.addressable_shards:
    # Print diagnostic summary of the computed outputs.
    print("Device", shard.device.id, "index", shard.index, "values", np.asarray(shard.data))
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(y), 2 * host + 1)
# Verify contract: `len(x.addressable_shards) == 4`.
assert len(x.addressable_shards) == 4
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert all(shard.data.shape == (1, 2) for shard in x.addressable_shards)
# Print the observed values to compare against the expected result.
print("Global shape:", x.shape, "result:", np.asarray(y).tolist())

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
# Construct and reshape `host` into the target tensor dimensions.
host = np.arange(8, dtype=np.float32).reshape(4, 2)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host, rows)
# Wrap with `jax.jit` (`y`) so XLA traces and compiles the function.
y = jax.jit(lambda a: 2 * a + 1, in_shardings=rows, out_shardings=rows)(x)
# Synchronize host execution until asynchronous device computation completes.
y.block_until_ready()

# Step 3 — Inspect and verify: Assertions compare against host calculations or hand-derived...
# Print the observed values to compare against the expected result.
print("JAX version:", jax.__version__)
# Print diagnostic summary of the computed outputs.
print("CPU devices:", len(devices))
# Iterate over `shard` to step through the computation:
for shard in x.addressable_shards:
    # Print diagnostic summary of the computed outputs.
    print("Device", shard.device.id, "index", shard.index, "values", np.asarray(shard.data))
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(y), 2 * host + 1)
# Verify contract: `len(x.addressable_shards) == 4`.
assert len(x.addressable_shards) == 4
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert all(shard.data.shape == (1, 2) for shard in x.addressable_shards)
# Print the observed values to compare against the expected result.
print("Global shape:", x.shape, "result:", np.asarray(y).tolist())

# Figure data experiment
# Compute figure data for: Four logical devices own four different rows
# Run `np.empty` to compute `owners`.
owners = np.empty(host.shape, dtype=int)
# Loop over `s` in `x.addressable_shards`:
for s in x.addressable_shards:
    # Evaluate `owners[s.index]` from the current inputs and state.
    owners[s.index] = s.device.id
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'heatmap', 'values': owners.tolist(), 'unit': 'logical CPU device ID', 'rows': ['row ' + str(i) for i in range(4)], 'columns': ['feature 0', 'feature 1']}

# Experiment: Keep the shape, change the values
# Experiment — Keep the shape, change the values: Placement follows the declared shape/specification, not the...
negative = -host - 2
# Place `negative_x` explicitly onto the target JAX device.
negative_x = jax.device_put(negative, rows)
# Wrap with `jax.jit` (`negative_y`) so XLA traces and compiles the function.
negative_y = jax.jit(lambda a: 2*a+1, in_shardings=rows, out_shardings=rows)(negative_x)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(negative_y), 2*negative+1)
# Verify that the output tensor shape matches our prediction.
assert all(s.data.shape == (1,2) for s in negative_x.addressable_shards)
# Print the observed values to compare against the expected result.
print("Negative input result:", np.asarray(negative_y).tolist())

# Experiment: Replicate instead of partitioning
# Experiment — Replicate instead of partitioning: Replication stores full copies.
copies = jax.device_put(host, replicated)
# Verify contract: `copies.is_fully_replicated`.
assert copies.is_fully_replicated
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert len(copies.addressable_shards) == 4
# Iterate over `shard` to step through the computation:
for shard in copies.addressable_shards:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), host)
# Print the observed values to compare against the expected result.
print("Replicated local shapes:", [s.data.shape for s in copies.addressable_shards])

# Reference solution. Try the exercise before reading this.
# Exercise solution: Predict and verify the per-device shapes for twelve rows, then...
# Construct and reshape `twelve` into the target tensor dimensions.
twelve = np.arange(24,dtype=np.float32).reshape(12,2)
# Place `z` explicitly onto the target JAX device.
z = jax.device_put(twelve, rows)
# Wrap with `jax.jit` (`z2`) so XLA traces and compiles the function.
z2 = jax.jit(lambda a:a*a, in_shardings=rows, out_shardings=rows)(z)
# Verify that the output tensor shape matches our prediction.
assert all(s.data.shape == (3,2) for s in z2.addressable_shards)
# Iterate over `shard` to step through the computation:
for shard in z2.addressable_shards:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), (twelve*twelve)[shard.index])

# Construct and reshape `uneven` into the target tensor dimensions.
uneven = np.arange(10,dtype=np.float32).reshape(5,2)
# Run the boundary check and catch the expected exception:
try:
    jax.device_put(uneven, rows)
except ValueError:
    print("Expected indivisible row dimension")
else:
    raise AssertionError("Expected divisibility failure")
# Place `fixed` explicitly onto the target JAX device.
fixed = jax.device_put(uneven, replicated)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(fixed), uneven)
# Verify contract: `fixed.is_fully_replicated`.
assert fixed.is_fully_replicated

# Reference practice: Move from four rows to twelve
# Move from four rows to twelve (Challenge): The twelve rows split into four groups of three.
# Construct and reshape `twelve` into the target tensor dimensions.
twelve = np.arange(24,dtype=np.float32).reshape(12,2)
# Place `z` explicitly onto the target JAX device.
z = jax.device_put(twelve, rows)
# Wrap with `jax.jit` (`z2`) so XLA traces and compiles the function.
z2 = jax.jit(lambda a:a*a, in_shardings=rows, out_shardings=rows)(z)
# Verify that the output tensor shape matches our prediction.
assert all(s.data.shape == (3,2) for s in z2.addressable_shards)
# Iterate over `shard` to step through the computation:
for shard in z2.addressable_shards:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), (twelve*twelve)[shard.index])

# Reference practice: Repair a batch that does not divide evenly
# Repair a batch that does not divide evenly (Challenge): Five rows cannot be evenly split across the four-way data axis.
# Construct and reshape `uneven` into the target tensor dimensions.
uneven = np.arange(10,dtype=np.float32).reshape(5,2)
# Run the boundary check and catch the expected exception:
try:
    jax.device_put(uneven, rows)
except ValueError:
    print("Expected indivisible row dimension")
else:
    raise AssertionError("Expected divisibility failure")
# Place `fixed` explicitly onto the target JAX device.
fixed = jax.device_put(uneven, replicated)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(fixed), uneven)
# Verify contract: `fixed.is_fully_replicated`.
assert fixed.is_fully_replicated
print("PASS: welcome-cpu")
