"""Arrays, meshes, and sharding: worked experiments and reference solutions. CPU checks."""

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
host = np.arange(32, dtype=np.float32).reshape(8,4)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host, rows)
# Configure multi-device placement / sharding specification (`columns`).
columns = NamedSharding(mesh, P(None, "data"))
# Place `x_columns` explicitly onto the target JAX device.
x_columns = jax.device_put(host, columns)
# Wrap with `jax.jit` (`sum_rows`) so XLA traces and compiles the function.
sum_rows = jax.jit(lambda a:a.sum(axis=0), in_shardings=rows, out_shardings=replicated)(x)
# Synchronize host execution until asynchronous device computation completes.
sum_rows.block_until_ready()

# Inspect and verify
# Step 3 — Inspect and verify: Assertions compare against host calculations or hand-derived...
np.testing.assert_array_equal(np.asarray(sum_rows), np.array([112,120,128,136],dtype=np.float32))
# Iterate over `shard` to step through the computation:
for shard in x.addressable_shards:
    # Check tensor shape invariant: `shard.data.shape == (2,4)`
    assert shard.data.shape == (2,4)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), host[shard.index])
# Iterate over `shard` to step through the computation:
for shard in x_columns.addressable_shards:
    # Check tensor shape invariant: `shard.data.shape == (8,1)`
    assert shard.data.shape == (8,1)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), host[shard.index])
# Print the observed values to compare against the expected result.
print("Row-shard shapes:", [s.data.shape for s in x.addressable_shards])
# Print diagnostic summary of the computed outputs.
print("Column-shard shapes:", [s.data.shape for s in x_columns.addressable_shards])
# Print diagnostic summary of the computed outputs.
print("Column sums:", np.asarray(sum_rows))

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
host = np.arange(32, dtype=np.float32).reshape(8,4)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host, rows)
# Configure multi-device placement / sharding specification (`columns`).
columns = NamedSharding(mesh, P(None, "data"))
# Place `x_columns` explicitly onto the target JAX device.
x_columns = jax.device_put(host, columns)
# Wrap with `jax.jit` (`sum_rows`) so XLA traces and compiles the function.
sum_rows = jax.jit(lambda a:a.sum(axis=0), in_shardings=rows, out_shardings=replicated)(x)
# Synchronize host execution until asynchronous device computation completes.
sum_rows.block_until_ready()

# Step 3 — Inspect and verify: Assertions compare against host calculations or hand-derived...
np.testing.assert_array_equal(np.asarray(sum_rows), np.array([112,120,128,136],dtype=np.float32))
# Iterate over `shard` to step through the computation:
for shard in x.addressable_shards:
    # Check tensor shape invariant: `shard.data.shape == (2,4)`
    assert shard.data.shape == (2,4)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), host[shard.index])
# Iterate over `shard` to step through the computation:
for shard in x_columns.addressable_shards:
    # Check tensor shape invariant: `shard.data.shape == (8,1)`
    assert shard.data.shape == (8,1)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), host[shard.index])
# Print the observed values to compare against the expected result.
print("Row-shard shapes:", [s.data.shape for s in x.addressable_shards])
# Print diagnostic summary of the computed outputs.
print("Column-shard shapes:", [s.data.shape for s in x_columns.addressable_shards])
# Print diagnostic summary of the computed outputs.
print("Column sums:", np.asarray(sum_rows))

# Figure data experiment
# Compute figure data for: Row sharding and column sharding place the same array differently
# Compute `panels` from `[]`
panels = []
# Loop over `(name, array)` in `[('Row partition', x), ('Column partition', x_columns)]`:
for name, array in [('Row partition', x), ('Column partition', x_columns)]:
    # Run `np.empty` to compute `owner`.
    owner = np.empty(host.shape, dtype=int)
    # Loop over `s` in `array.addressable_shards`:
    for s in array.addressable_shards:
        # Compute `owner[s.index]` from `s.device.id`
        owner[s.index] = s.device.id
    # Append the current step result to `panels`.
    panels.append({'kind': 'heatmap', 'title': name, 'values': owner.tolist(), 'unit': 'logical CPU device ID'})
# Compute `visual_data` from `{'kind': 'panels', 'panels': panels}`
visual_data = {'kind': 'panels', 'panels': panels}

# Experiment: Reshard the same global values
# Experiment — Reshard the same global values: Resharding changes ownership of values, not their mathematical...
moved = jax.device_put(x, columns)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(moved), host)
# Check tensor shape invariant: `all(s.data.shape == (8,1) for s in moved.addressable_shards)`
assert all(s.data.shape == (8,1) for s in moved.addressable_shards)
# Print the observed values to compare against the expected result.
print("Resharded values preserved")

# Experiment: Reduce a dimension that is not partitioned
# Experiment — Reduce a dimension that is not partitioned: All four columns for a row are already local in row...
# Configure multi-device placement / sharding specification (`vector_rows`).
vector_rows = NamedSharding(mesh, P("data"))
# Wrap with `jax.jit` (`per_row`) so XLA traces and compiles the function.
per_row = jax.jit(lambda a:a.sum(axis=1), in_shardings=rows, out_shardings=vector_rows)(x)
# Create evenly spaced index values in ``.
np.testing.assert_array_equal(np.asarray(per_row), np.arange(6,119,16,dtype=np.float32))
# Check tensor shape invariant: `all(s.data.shape == (2,) for s in per_row.addressable_shards)`
assert all(s.data.shape == (2,) for s in per_row.addressable_shards)
# Print the observed values to compare against the expected result.
print("Row sums:", np.asarray(per_row))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Compare row and column layouts of a (12,8) array, and repair a ten-row...
# Construct and reshape `other` into the target tensor dimensions.
other = np.arange(96,dtype=np.float32).reshape(12,8)
# Iterate over `(spec, expected)` to step through the computation:
for spec, expected in [(rows,(3,8)),(columns,(12,2))]:
    # Place `placed` explicitly onto the target JAX device.
    placed = jax.device_put(other,spec)
    # Iterate over `shard` to step through the computation:
    for shard in placed.addressable_shards:
        # Check tensor shape invariant: `shard.data.shape == expected`
        assert shard.data.shape == expected
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_array_equal(np.asarray(shard.data),other[shard.index])

# Construct and reshape `original` into the target tensor dimensions.
original = np.arange(40,dtype=np.float32).reshape(10,4)
# Run the boundary check and catch the expected exception:
try:
    jax.device_put(original, rows)
except ValueError:
    print("Expected indivisible batch")
else:
    raise AssertionError("Expected divisibility failure")
# Place `padded` explicitly onto the target JAX device.
padded = jax.device_put(np.pad(original,((0,2),(0,0))),rows)
# Configure multi-device placement / sharding specification (`mask_sharding`).
mask_sharding = NamedSharding(mesh,P("data"))
# Compute `mask` from `jax.device_put(np.array([1]*10+[0]*2,dtype=np.float3...`
mask = jax.device_put(np.array([1]*10+[0]*2,dtype=np.float32),mask_sharding)
# Wrap with `jax.jit` (`masked_mean`) so XLA traces and compiles the function.
masked_mean = jax.jit(lambda a,m:(a*m[:,None]).sum(axis=0)/m.sum(),in_shardings=(rows,mask_sharding),out_shardings=replicated)(padded,mask)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(masked_mean),original.mean(axis=0),rtol=1e-6)
# Assert that `not np.allclose(np.asarray(padded.mean(axis=0)), original.mean(axis=0))`.
assert not np.allclose(np.asarray(padded.mean(axis=0)), original.mean(axis=0))

# Reference practice: Change both dimensions
# Change both dimensions (Challenge): The row layout owns three complete feature vectors per...
# Construct and reshape `other` into the target tensor dimensions.
other = np.arange(96,dtype=np.float32).reshape(12,8)
# Iterate over `(spec, expected)` to step through the computation:
for spec, expected in [(rows,(3,8)),(columns,(12,2))]:
    # Place `placed` explicitly onto the target JAX device.
    placed = jax.device_put(other,spec)
    # Iterate over `shard` to step through the computation:
    for shard in placed.addressable_shards:
        # Check tensor shape invariant: `shard.data.shape == expected`
        assert shard.data.shape == expected
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_array_equal(np.asarray(shard.data),other[shard.index])

# Reference practice: Pad an uneven batch without corrupting its mean
# Pad an uneven batch without corrupting its mean (Challenge): Padding repairs the shape but adds artificial rows.
# Construct and reshape `original` into the target tensor dimensions.
original = np.arange(40,dtype=np.float32).reshape(10,4)
# Run the boundary check and catch the expected exception:
try:
    jax.device_put(original, rows)
except ValueError:
    print("Expected indivisible batch")
else:
    raise AssertionError("Expected divisibility failure")
# Place `padded` explicitly onto the target JAX device.
padded = jax.device_put(np.pad(original,((0,2),(0,0))),rows)
# Configure multi-device placement / sharding specification (`mask_sharding`).
mask_sharding = NamedSharding(mesh,P("data"))
# Compute `mask` from `jax.device_put(np.array([1]*10+[0]*2,dtype=np.float3...`
mask = jax.device_put(np.array([1]*10+[0]*2,dtype=np.float32),mask_sharding)
# Wrap with `jax.jit` (`masked_mean`) so XLA traces and compiles the function.
masked_mean = jax.jit(lambda a,m:(a*m[:,None]).sum(axis=0)/m.sum(),in_shardings=(rows,mask_sharding),out_shardings=replicated)(padded,mask)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(masked_mean),original.mean(axis=0),rtol=1e-6)
# Assert that `not np.allclose(np.asarray(padded.mean(axis=0)), original.mean(axis=0))`.
assert not np.allclose(np.asarray(padded.mean(axis=0)), original.mean(axis=0))
print("PASS: distributed-01")
