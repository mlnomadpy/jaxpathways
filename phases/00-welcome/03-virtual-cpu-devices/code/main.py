"""Practice with four virtual CPU devices: worked experiments and reference solutions. CPU checks."""

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
host = np.arange(8, dtype=np.float32).reshape(4, 2)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host, rows)
# Wrap with `jax.jit` (`y`) so XLA traces and compiles the function.
y = jax.jit(lambda a: 2 * a + 1, in_shardings=rows, out_shardings=rows)(x)
# Wait for device work to finish before timing it or comparing its result on the host.
y.block_until_ready()

# Inspect and verify
# Step 3 — Inspect and verify: Assertions compare against host calculations or hand-derived...
print("JAX version:", jax.__version__)
print("CPU devices:", len(devices))
for shard in x.addressable_shards:
    print("Device", shard.device.id, "index", shard.index, "values", np.asarray(shard.data))
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_array_equal(np.asarray(y), 2 * host + 1)
assert len(x.addressable_shards) == 4
# Check tensor shape invariant: `all(shard.data.shape == (1, 2) for shard in x.addressable_shards)`
assert all(shard.data.shape == (1, 2) for shard in x.addressable_shards)
print("Global shape:", x.shape, "result:", np.asarray(y).tolist())

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
host = np.arange(8, dtype=np.float32).reshape(4, 2)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host, rows)
# Wrap with `jax.jit` (`y`) so XLA traces and compiles the function.
y = jax.jit(lambda a: 2 * a + 1, in_shardings=rows, out_shardings=rows)(x)
# Wait for device work to finish before timing it or comparing its result on the host.
y.block_until_ready()

# Step 3 — Inspect and verify: Assertions compare against host calculations or hand-derived...
print("JAX version:", jax.__version__)
print("CPU devices:", len(devices))
for shard in x.addressable_shards:
    print("Device", shard.device.id, "index", shard.index, "values", np.asarray(shard.data))
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_array_equal(np.asarray(y), 2 * host + 1)
assert len(x.addressable_shards) == 4
# Check tensor shape invariant: `all(shard.data.shape == (1, 2) for shard in x.addressable_shards)`
assert all(shard.data.shape == (1, 2) for shard in x.addressable_shards)
print("Global shape:", x.shape, "result:", np.asarray(y).tolist())

# Figure data experiment
owners = np.empty(host.shape, dtype=int)
for s in x.addressable_shards:
    owners[s.index] = s.device.id
visual_data = {'kind': 'heatmap', 'values': owners.tolist(), 'unit': 'logical CPU device ID', 'rows': ['row ' + str(i) for i in range(4)], 'columns': ['feature 0', 'feature 1']}

# Experiment: Keep the shape, change the values
# Experiment — Keep the shape, change the values: Placement follows the declared shape/specification, not the...
negative = -host - 2
# Place `negative_x` explicitly onto the target JAX device.
negative_x = jax.device_put(negative, rows)
# Wrap with `jax.jit` (`negative_y`) so XLA traces and compiles the function.
negative_y = jax.jit(lambda a: 2*a+1, in_shardings=rows, out_shardings=rows)(negative_x)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_array_equal(np.asarray(negative_y), 2*negative+1)
# Check tensor shape invariant: `all(s.data.shape == (1,2) for s in negative_x.addressable_shards)`
assert all(s.data.shape == (1,2) for s in negative_x.addressable_shards)
print("Negative input result:", np.asarray(negative_y).tolist())

# Experiment: Replicate instead of partitioning
# Experiment — Replicate instead of partitioning: Replication stores full copies.
copies = jax.device_put(host, replicated)
assert copies.is_fully_replicated
assert len(copies.addressable_shards) == 4
for shard in copies.addressable_shards:
    # Transfer the device result to the host for an independent NumPy comparison.
    np.testing.assert_array_equal(np.asarray(shard.data), host)
print("Replicated local shapes:", [s.data.shape for s in copies.addressable_shards])

# Reference solution. Try the exercise before reading this.
# Exercise solution: Predict and verify the per-device shapes for twelve rows, then...
twelve = np.arange(24,dtype=np.float32).reshape(12,2)
# Place `z` explicitly onto the target JAX device.
z = jax.device_put(twelve, rows)
# Wrap with `jax.jit` (`z2`) so XLA traces and compiles the function.
z2 = jax.jit(lambda a:a*a, in_shardings=rows, out_shardings=rows)(z)
# Check tensor shape invariant: `all(s.data.shape == (3,2) for s in z2.addressable_shards)`
assert all(s.data.shape == (3,2) for s in z2.addressable_shards)
for shard in z2.addressable_shards:
    # Transfer the device result to the host for an independent NumPy comparison.
    np.testing.assert_array_equal(np.asarray(shard.data), (twelve*twelve)[shard.index])

uneven = np.arange(10,dtype=np.float32).reshape(5,2)
try:
    jax.device_put(uneven, rows)
except ValueError:
    print("Expected indivisible row dimension")
else:
    raise AssertionError("Expected divisibility failure")
# Place `fixed` explicitly onto the target JAX device.
fixed = jax.device_put(uneven, replicated)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_array_equal(np.asarray(fixed), uneven)
assert fixed.is_fully_replicated

# Reference practice: Move from four rows to twelve
# Move from four rows to twelve (Challenge): The twelve rows split into four groups of three.
twelve = np.arange(24,dtype=np.float32).reshape(12,2)
# Place `z` explicitly onto the target JAX device.
z = jax.device_put(twelve, rows)
# Wrap with `jax.jit` (`z2`) so XLA traces and compiles the function.
z2 = jax.jit(lambda a:a*a, in_shardings=rows, out_shardings=rows)(z)
# Check tensor shape invariant: `all(s.data.shape == (3,2) for s in z2.addressable_shards)`
assert all(s.data.shape == (3,2) for s in z2.addressable_shards)
for shard in z2.addressable_shards:
    # Transfer the device result to the host for an independent NumPy comparison.
    np.testing.assert_array_equal(np.asarray(shard.data), (twelve*twelve)[shard.index])

# Reference practice: Repair a batch that does not divide evenly
# Repair a batch that does not divide evenly (Challenge): Five rows cannot be evenly split across the four-way data axis.
uneven = np.arange(10,dtype=np.float32).reshape(5,2)
try:
    jax.device_put(uneven, rows)
except ValueError:
    print("Expected indivisible row dimension")
else:
    raise AssertionError("Expected divisibility failure")
# Place `fixed` explicitly onto the target JAX device.
fixed = jax.device_put(uneven, replicated)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_array_equal(np.asarray(fixed), uneven)
assert fixed.is_fully_replicated
print("PASS: welcome-cpu")
