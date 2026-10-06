"""Practice with four virtual CPU devices: worked experiments and reference solutions. CPU checks."""

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
host = np.arange(8, dtype=np.float32).reshape(4, 2)
x = jax.device_put(host, rows)
y = jax.jit(lambda a: 2 * a + 1, in_shardings=rows, out_shardings=rows)(x)
y.block_until_ready()

# Inspect and verify
print("JAX version:", jax.__version__)
print("CPU devices:", len(devices))
for shard in x.addressable_shards:
    print("Device", shard.device.id, "index", shard.index, "values", np.asarray(shard.data))
np.testing.assert_array_equal(np.asarray(y), 2 * host + 1)
assert len(x.addressable_shards) == 4
assert all(shard.data.shape == (1, 2) for shard in x.addressable_shards)
print("Global shape:", x.shape, "result:", np.asarray(y).tolist())

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

host = np.arange(8, dtype=np.float32).reshape(4, 2)
x = jax.device_put(host, rows)
y = jax.jit(lambda a: 2 * a + 1, in_shardings=rows, out_shardings=rows)(x)
y.block_until_ready()

print("JAX version:", jax.__version__)
print("CPU devices:", len(devices))
for shard in x.addressable_shards:
    print("Device", shard.device.id, "index", shard.index, "values", np.asarray(shard.data))
np.testing.assert_array_equal(np.asarray(y), 2 * host + 1)
assert len(x.addressable_shards) == 4
assert all(shard.data.shape == (1, 2) for shard in x.addressable_shards)
print("Global shape:", x.shape, "result:", np.asarray(y).tolist())

# Figure data experiment
owners = np.empty(host.shape, dtype=int)
for s in x.addressable_shards:
    owners[s.index] = s.device.id
visual_data = {'kind': 'heatmap', 'values': owners.tolist(), 'unit': 'logical CPU device ID', 'rows': ['row ' + str(i) for i in range(4)], 'columns': ['feature 0', 'feature 1']}

# Experiment: Keep the shape, change the values
negative = -host - 2
negative_x = jax.device_put(negative, rows)
negative_y = jax.jit(lambda a: 2*a+1, in_shardings=rows, out_shardings=rows)(negative_x)
np.testing.assert_array_equal(np.asarray(negative_y), 2*negative+1)
assert all(s.data.shape == (1,2) for s in negative_x.addressable_shards)
print("Negative input result:", np.asarray(negative_y).tolist())

# Experiment: Replicate instead of partitioning
copies = jax.device_put(host, replicated)
assert copies.is_fully_replicated
assert len(copies.addressable_shards) == 4
for shard in copies.addressable_shards:
    np.testing.assert_array_equal(np.asarray(shard.data), host)
print("Replicated local shapes:", [s.data.shape for s in copies.addressable_shards])

# Reference solution. Try the exercise before reading this.
twelve = np.arange(24,dtype=np.float32).reshape(12,2)
z = jax.device_put(twelve, rows)
z2 = jax.jit(lambda a:a*a, in_shardings=rows, out_shardings=rows)(z)
assert all(s.data.shape == (3,2) for s in z2.addressable_shards)
for shard in z2.addressable_shards:
    np.testing.assert_array_equal(np.asarray(shard.data), (twelve*twelve)[shard.index])

uneven = np.arange(10,dtype=np.float32).reshape(5,2)
try:
    jax.device_put(uneven, rows)
except ValueError:
    print("Expected indivisible row dimension")
else:
    raise AssertionError("Expected divisibility failure")
fixed = jax.device_put(uneven, replicated)
np.testing.assert_array_equal(np.asarray(fixed), uneven)
assert fixed.is_fully_replicated

# Reference practice: Move from four rows to twelve
twelve = np.arange(24,dtype=np.float32).reshape(12,2)
z = jax.device_put(twelve, rows)
z2 = jax.jit(lambda a:a*a, in_shardings=rows, out_shardings=rows)(z)
assert all(s.data.shape == (3,2) for s in z2.addressable_shards)
for shard in z2.addressable_shards:
    np.testing.assert_array_equal(np.asarray(shard.data), (twelve*twelve)[shard.index])

# Reference practice: Repair a batch that does not divide evenly
uneven = np.arange(10,dtype=np.float32).reshape(5,2)
try:
    jax.device_put(uneven, rows)
except ValueError:
    print("Expected indivisible row dimension")
else:
    raise AssertionError("Expected divisibility failure")
fixed = jax.device_put(uneven, replicated)
np.testing.assert_array_equal(np.asarray(fixed), uneven)
assert fixed.is_fully_replicated
print("PASS: welcome-cpu")
