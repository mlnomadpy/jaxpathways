"""Arrays, meshes, and sharding: worked experiments and reference solutions. CPU checks."""

# Start a fresh CPU runtime
# Step 1 — Start a fresh CPU runtime: Configuration happens before devices() or array creation...
import jax
jax.config.update("jax_platforms", "cpu")
# Select the CPU backend and virtual device count before initializing JAX.
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
# Step 2 — Place and compute: Mesh axis names describe placement.
host = np.arange(32, dtype=np.float32).reshape(8,4)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host, rows)
columns = NamedSharding(mesh, P(None, "data"))
# Place `x_columns` explicitly onto the target JAX device.
x_columns = jax.device_put(host, columns)
# Wrap with `jax.jit` (`sum_rows`) so XLA traces and compiles the function.
sum_rows = jax.jit(lambda a:a.sum(axis=0), in_shardings=rows, out_shardings=replicated)(x)
# Wait for device work to finish before timing it or comparing its result on the host.
sum_rows.block_until_ready()

# Inspect and verify
# Step 3 — Inspect and verify: Assertions compare against host calculations or hand-derived...
np.testing.assert_array_equal(np.asarray(sum_rows), np.array([112,120,128,136],dtype=np.float32))
for shard in x.addressable_shards:
    # Check tensor shape invariant: `shard.data.shape == (2,4)`
    assert shard.data.shape == (2,4)
    # Transfer the device result to the host for an independent NumPy comparison.
    np.testing.assert_array_equal(np.asarray(shard.data), host[shard.index])
for shard in x_columns.addressable_shards:
    # Check tensor shape invariant: `shard.data.shape == (8,1)`
    assert shard.data.shape == (8,1)
    # Transfer the device result to the host for an independent NumPy comparison.
    np.testing.assert_array_equal(np.asarray(shard.data), host[shard.index])
print("Row-shard shapes:", [s.data.shape for s in x.addressable_shards])
print("Column-shard shapes:", [s.data.shape for s in x_columns.addressable_shards])
print("Column sums:", np.asarray(sum_rows))

# Step 1 — Start a fresh CPU runtime: Configuration happens before devices() or array creation...
import jax
jax.config.update("jax_platforms", "cpu")
# Select the CPU backend and virtual device count before initializing JAX.
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

# Step 2 — Place and compute: Mesh axis names describe placement.
host = np.arange(32, dtype=np.float32).reshape(8,4)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host, rows)
columns = NamedSharding(mesh, P(None, "data"))
# Place `x_columns` explicitly onto the target JAX device.
x_columns = jax.device_put(host, columns)
# Wrap with `jax.jit` (`sum_rows`) so XLA traces and compiles the function.
sum_rows = jax.jit(lambda a:a.sum(axis=0), in_shardings=rows, out_shardings=replicated)(x)
# Wait for device work to finish before timing it or comparing its result on the host.
sum_rows.block_until_ready()

# Step 3 — Inspect and verify: Assertions compare against host calculations or hand-derived...
np.testing.assert_array_equal(np.asarray(sum_rows), np.array([112,120,128,136],dtype=np.float32))
for shard in x.addressable_shards:
    # Check tensor shape invariant: `shard.data.shape == (2,4)`
    assert shard.data.shape == (2,4)
    # Transfer the device result to the host for an independent NumPy comparison.
    np.testing.assert_array_equal(np.asarray(shard.data), host[shard.index])
for shard in x_columns.addressable_shards:
    # Check tensor shape invariant: `shard.data.shape == (8,1)`
    assert shard.data.shape == (8,1)
    # Transfer the device result to the host for an independent NumPy comparison.
    np.testing.assert_array_equal(np.asarray(shard.data), host[shard.index])
print("Row-shard shapes:", [s.data.shape for s in x.addressable_shards])
print("Column-shard shapes:", [s.data.shape for s in x_columns.addressable_shards])
print("Column sums:", np.asarray(sum_rows))

# Figure data experiment
panels = []
for name, array in [('Row partition', x), ('Column partition', x_columns)]:
    owner = np.empty(host.shape, dtype=int)
    for s in array.addressable_shards:
        owner[s.index] = s.device.id
    panels.append({'kind': 'heatmap', 'title': name, 'values': owner.tolist(), 'unit': 'logical CPU device ID'})
visual_data = {'kind': 'panels', 'panels': panels}

# Experiment: Reshard the same global values
# Experiment — Reshard the same global values: Resharding changes ownership of values, not their mathematical...
moved = jax.device_put(x, columns)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_array_equal(np.asarray(moved), host)
# Check tensor shape invariant: `all(s.data.shape == (8,1) for s in moved.addressable_shards)`
assert all(s.data.shape == (8,1) for s in moved.addressable_shards)
print("Resharded values preserved")

# Experiment: Reduce a dimension that is not partitioned
# Experiment — Reduce a dimension that is not partitioned: All four columns for a row are already local in row...
vector_rows = NamedSharding(mesh, P("data"))
# Wrap with `jax.jit` (`per_row`) so XLA traces and compiles the function.
per_row = jax.jit(lambda a:a.sum(axis=1), in_shardings=rows, out_shardings=vector_rows)(x)
# Create evenly spaced index values in ``.
np.testing.assert_array_equal(np.asarray(per_row), np.arange(6,119,16,dtype=np.float32))
# Check tensor shape invariant: `all(s.data.shape == (2,) for s in per_row.addressable_shards)`
assert all(s.data.shape == (2,) for s in per_row.addressable_shards)
print("Row sums:", np.asarray(per_row))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Compare row and column layouts of a (12,8) array, and repair a ten-row...
other = np.arange(96,dtype=np.float32).reshape(12,8)
for spec, expected in [(rows,(3,8)),(columns,(12,2))]:
    # Place `placed` explicitly onto the target JAX device.
    placed = jax.device_put(other,spec)
    for shard in placed.addressable_shards:
        # Check tensor shape invariant: `shard.data.shape == expected`
        assert shard.data.shape == expected
        # Transfer the device result to the host for an independent NumPy comparison.
        np.testing.assert_array_equal(np.asarray(shard.data),other[shard.index])

original = np.arange(40,dtype=np.float32).reshape(10,4)
try:
    jax.device_put(original, rows)
except ValueError:
    print("Expected indivisible batch")
else:
    raise AssertionError("Expected divisibility failure")
# Place `padded` explicitly onto the target JAX device.
padded = jax.device_put(np.pad(original,((0,2),(0,0))),rows)
mask_sharding = NamedSharding(mesh,P("data"))
mask = jax.device_put(np.array([1]*10+[0]*2,dtype=np.float32),mask_sharding)
# Wrap with `jax.jit` (`masked_mean`) so XLA traces and compiles the function.
masked_mean = jax.jit(lambda a,m:(a*m[:,None]).sum(axis=0)/m.sum(),in_shardings=(rows,mask_sharding),out_shardings=replicated)(padded,mask)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(masked_mean),original.mean(axis=0),rtol=1e-6)
assert not np.allclose(np.asarray(padded.mean(axis=0)), original.mean(axis=0))

# Reference practice: Change both dimensions
# Change both dimensions (Challenge): The row layout owns three complete feature vectors per...
other = np.arange(96,dtype=np.float32).reshape(12,8)
for spec, expected in [(rows,(3,8)),(columns,(12,2))]:
    # Place `placed` explicitly onto the target JAX device.
    placed = jax.device_put(other,spec)
    for shard in placed.addressable_shards:
        # Check tensor shape invariant: `shard.data.shape == expected`
        assert shard.data.shape == expected
        # Transfer the device result to the host for an independent NumPy comparison.
        np.testing.assert_array_equal(np.asarray(shard.data),other[shard.index])

# Reference practice: Pad an uneven batch without corrupting its mean
# Pad an uneven batch without corrupting its mean (Challenge): Padding repairs the shape but adds artificial rows.
original = np.arange(40,dtype=np.float32).reshape(10,4)
try:
    jax.device_put(original, rows)
except ValueError:
    print("Expected indivisible batch")
else:
    raise AssertionError("Expected divisibility failure")
# Place `padded` explicitly onto the target JAX device.
padded = jax.device_put(np.pad(original,((0,2),(0,0))),rows)
mask_sharding = NamedSharding(mesh,P("data"))
mask = jax.device_put(np.array([1]*10+[0]*2,dtype=np.float32),mask_sharding)
# Wrap with `jax.jit` (`masked_mean`) so XLA traces and compiles the function.
masked_mean = jax.jit(lambda a,m:(a*m[:,None]).sum(axis=0)/m.sum(),in_shardings=(rows,mask_sharding),out_shardings=replicated)(padded,mask)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(masked_mean),original.mean(axis=0),rtol=1e-6)
assert not np.allclose(np.asarray(padded.mean(axis=0)), original.mean(axis=0))
print("PASS: distributed-01")
