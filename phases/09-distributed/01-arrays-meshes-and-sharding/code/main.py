"""Arrays, meshes, and sharding: worked experiments and reference solutions. CPU checks."""

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
host = np.arange(32, dtype=np.float32).reshape(8,4)
x = jax.device_put(host, rows)
columns = NamedSharding(mesh, P(None, "data"))
x_columns = jax.device_put(host, columns)
sum_rows = jax.jit(lambda a:a.sum(axis=0), in_shardings=rows, out_shardings=replicated)(x)
sum_rows.block_until_ready()

# Inspect and verify
np.testing.assert_array_equal(np.asarray(sum_rows), np.array([112,120,128,136],dtype=np.float32))
for shard in x.addressable_shards:
    assert shard.data.shape == (2,4)
    np.testing.assert_array_equal(np.asarray(shard.data), host[shard.index])
for shard in x_columns.addressable_shards:
    assert shard.data.shape == (8,1)
    np.testing.assert_array_equal(np.asarray(shard.data), host[shard.index])
print("Row-shard shapes:", [s.data.shape for s in x.addressable_shards])
print("Column-shard shapes:", [s.data.shape for s in x_columns.addressable_shards])
print("Column sums:", np.asarray(sum_rows))

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

host = np.arange(32, dtype=np.float32).reshape(8,4)
x = jax.device_put(host, rows)
columns = NamedSharding(mesh, P(None, "data"))
x_columns = jax.device_put(host, columns)
sum_rows = jax.jit(lambda a:a.sum(axis=0), in_shardings=rows, out_shardings=replicated)(x)
sum_rows.block_until_ready()

np.testing.assert_array_equal(np.asarray(sum_rows), np.array([112,120,128,136],dtype=np.float32))
for shard in x.addressable_shards:
    assert shard.data.shape == (2,4)
    np.testing.assert_array_equal(np.asarray(shard.data), host[shard.index])
for shard in x_columns.addressable_shards:
    assert shard.data.shape == (8,1)
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
moved = jax.device_put(x, columns)
np.testing.assert_array_equal(np.asarray(moved), host)
assert all(s.data.shape == (8,1) for s in moved.addressable_shards)
print("Resharded values preserved")

# Experiment: Reduce a dimension that is not partitioned
vector_rows = NamedSharding(mesh, P("data"))
per_row = jax.jit(lambda a:a.sum(axis=1), in_shardings=rows, out_shardings=vector_rows)(x)
np.testing.assert_array_equal(np.asarray(per_row), np.arange(6,119,16,dtype=np.float32))
assert all(s.data.shape == (2,) for s in per_row.addressable_shards)
print("Row sums:", np.asarray(per_row))

# Reference solution. Try the exercise before reading this.
other = np.arange(96,dtype=np.float32).reshape(12,8)
for spec, expected in [(rows,(3,8)),(columns,(12,2))]:
    placed = jax.device_put(other,spec)
    for shard in placed.addressable_shards:
        assert shard.data.shape == expected
        np.testing.assert_array_equal(np.asarray(shard.data),other[shard.index])

original = np.arange(40,dtype=np.float32).reshape(10,4)
try:
    jax.device_put(original, rows)
except ValueError:
    print("Expected indivisible batch")
else:
    raise AssertionError("Expected divisibility failure")
padded = jax.device_put(np.pad(original,((0,2),(0,0))),rows)
mask_sharding = NamedSharding(mesh,P("data"))
mask = jax.device_put(np.array([1]*10+[0]*2,dtype=np.float32),mask_sharding)
masked_mean = jax.jit(lambda a,m:(a*m[:,None]).sum(axis=0)/m.sum(),in_shardings=(rows,mask_sharding),out_shardings=replicated)(padded,mask)
np.testing.assert_allclose(np.asarray(masked_mean),original.mean(axis=0),rtol=1e-6)
assert not np.allclose(np.asarray(padded.mean(axis=0)), original.mean(axis=0))

# Reference practice: Change both dimensions
other = np.arange(96,dtype=np.float32).reshape(12,8)
for spec, expected in [(rows,(3,8)),(columns,(12,2))]:
    placed = jax.device_put(other,spec)
    for shard in placed.addressable_shards:
        assert shard.data.shape == expected
        np.testing.assert_array_equal(np.asarray(shard.data),other[shard.index])

# Reference practice: Pad an uneven batch without corrupting its mean
original = np.arange(40,dtype=np.float32).reshape(10,4)
try:
    jax.device_put(original, rows)
except ValueError:
    print("Expected indivisible batch")
else:
    raise AssertionError("Expected divisibility failure")
padded = jax.device_put(np.pad(original,((0,2),(0,0))),rows)
mask_sharding = NamedSharding(mesh,P("data"))
mask = jax.device_put(np.array([1]*10+[0]*2,dtype=np.float32),mask_sharding)
masked_mean = jax.jit(lambda a,m:(a*m[:,None]).sum(axis=0)/m.sum(),in_shardings=(rows,mask_sharding),out_shardings=replicated)(padded,mask)
np.testing.assert_allclose(np.asarray(masked_mean),original.mean(axis=0),rtol=1e-6)
assert not np.allclose(np.asarray(padded.mean(axis=0)), original.mean(axis=0))
print("PASS: distributed-01")
