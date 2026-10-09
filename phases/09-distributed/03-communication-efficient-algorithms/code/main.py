"""Communication-efficient algorithms: worked experiments and reference solutions. CPU checks."""

# Configure four logical devices
# Step 1 — Configure four logical devices: Compare both the numerical global result and the placement needed...
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update('jax_platforms', 'cpu')
# Update state in place with the new values.
jax.config.update('jax_num_cpu_devices', 4)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from time import perf_counter

# Place the data and model
# Step 2 — Place the data and model: Compare both the numerical global result and the placement needed...
# Verify contract: `jax.local_device_count() == 4`.
assert jax.local_device_count() == 4, 'Restart with four logical CPU devices'
# Configure multi-device placement / sharding specification (`mesh`).
mesh = Mesh(np.asarray(jax.devices()), ('data',))
# Configure multi-device placement / sharding specification (`rows`).
rows = NamedSharding(mesh, P('data', None))
# Configure multi-device placement / sharding specification (`replicated`).
replicated = NamedSharding(mesh, P())
# Configure multi-device placement / sharding specification (`features`).
features = NamedSharding(mesh, P('data'))
# Construct and reshape `xh` into the target tensor dimensions.
xh = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 128 - 0.5
# Initialize array `yh` with explicit values and shape.
yh = xh @ np.linspace(-0.4, 0.3, 8, dtype=np.float32)
# Initialize array `wh` with explicit values and shape.
wh = np.linspace(0.1, -0.2, 8, dtype=np.float32)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(xh, rows)
# Configure multi-device placement / sharding specification (`y`).
y = jax.device_put(yh, NamedSharding(mesh, P('data')))
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(wh, replicated)

# Implement partial gradients and collectives
# Step 3 — Implement partial gradients and collectives: Compare both the numerical global result and the placement needed...
def local_contribution(a, b, weights):
    # Return `2 * a.T @ (a @ weights - b) / 16` to the caller.
    return 2 * a.T @ (a @ weights - b) / 16

# Verify the result and placement
# Step 4 — Verify the result and placement: Compare both the numerical global result and the placement needed...
def reduce_all(a, b, weights):
    # Return `jax.lax.psum(local_contribution(a, b, weights), 'data')` to the caller.
    return jax.lax.psum(local_contribution(a, b, weights), 'data')

# Measure completed calls
# Step 5 — Measure completed calls: Compare both the numerical global result and the placement needed...
def reduce_shard(a, b, weights):
    # Return `jax.lax.psum_scatter(local_contribution(a, b, weights), 'data', tiled=True)` to the caller.
    return jax.lax.psum_scatter(local_contribution(a, b, weights), 'data', tiled=True)

# Measure completed calls
# Step 6 — Measure completed calls: Compare both the numerical global result and the placement needed...
# Configure multi-device placement / sharding specification (`all_gradient`).
all_gradient = jax.jit(jax.shard_map(reduce_all, mesh=mesh,
    in_specs=(P('data', None), P('data'), P()), out_specs=P()))
# Configure multi-device placement / sharding specification (`shard_gradient`).
shard_gradient = jax.jit(jax.shard_map(reduce_shard, mesh=mesh,
    in_specs=(P('data', None), P('data'), P()), out_specs=P('data')))
# Configure multi-device placement / sharding specification (`gather`).
gather = jax.jit(jax.shard_map(lambda g: jax.lax.all_gather(g, 'data', tiled=True),
    mesh=mesh, in_specs=P('data'), out_specs=P(), check_vma=False))
# Run `all_gradient` to compute `ga`.
ga = all_gradient(x, y, w)
# Run `shard_gradient` to compute `gs`.
gs = shard_gradient(x, y, w)
# Run `gather` to compute `gg`.
gg = gather(gs)
# Perform matrix contraction / projection to compute `reference`.
reference = 2 * xh.T @ (xh @ wh - yh) / 16
# Iterate over `result` to step through the computation:
for result in [ga, gs, gg]:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(result), reference, atol=2e-6, rtol=2e-6)
# Verify that the output tensor shape matches our prediction.
assert all(s.data.shape == (8,) for s in ga.addressable_shards)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert all(s.data.shape == (2,) for s in gs.addressable_shards)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert all(s.data.shape == (8,) for s in gg.addressable_shards)
# Trace or lower the function to inspect its compiler representation (`hlo_all`).
hlo_all = str(all_gradient.lower(x, y, w).compiler_ir(dialect='stablehlo'))
# Run `str` to compute `hlo_shard`.
hlo_shard = str(shard_gradient.lower(x, y, w).compiler_ir(dialect='stablehlo'))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert 'all_reduce' in hlo_all and 'reduce_scatter' in hlo_shard

# Measure completed calls
# Step 7 — Measure completed calls: Compare both the numerical global result and the placement needed...
def completed_samples(fn, *args):
    # Synchronize host execution until asynchronous device computation completes.
    fn(*args).block_until_ready()
    # Evaluate `samples` from the current inputs and state.
    samples = []
    # Repeat the update loop over `range(7)` steps:
    for _ in range(7):
        # Record execution timing or profiler trace in `start`.
        start = perf_counter()
        # Synchronize host execution until asynchronous device computation completes.
        fn(*args).block_until_ready()
        # Record execution timing or profiler trace in ``.
        samples.append((perf_counter() - start) * 1e6)
    # Return `samples` to the caller.
    return samples
# Run `completed_samples` to compute `all_us`.
all_us = completed_samples(all_gradient, x, y, w)
# Run `completed_samples` to compute `shard_us`.
shard_us = completed_samples(shard_gradient, x, y, w)
# Run `completed_samples` to compute `gather_us`.
gather_us = completed_samples(gather, gs)
# Idealized ring payload per rank; not a network measurement.
ranks, gradient_bytes = 4, wh.nbytes
# Evaluate `ring_scatter_bytes` from the current inputs and state.
ring_scatter_bytes = (ranks - 1) / ranks * gradient_bytes
# Evaluate `ring_all_bytes` from the current inputs and state.
ring_all_bytes = 2 * ring_scatter_bytes
# Print the observed values to compare against the expected result.
print('Gradient reference:', reference.tolist())
# Print diagnostic summary of the computed outputs.
print('Local result shapes: replicated (8,), reduce-scatter (2,), gathered (8,)')
# Print diagnostic summary of the computed outputs.
print('Idealized ring bytes per rank:', ring_all_bytes, ring_scatter_bytes)
# Print diagnostic summary of the computed outputs.
print('Completed CPU median microseconds:', {
    'all_reduce': float(np.median(all_us)), 'reduce_scatter': float(np.median(shard_us)),
    'all_gather_only': float(np.median(gather_us))})

# Step 1 — Configure four logical devices: Compare both the numerical global result and the placement needed...
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update('jax_platforms', 'cpu')
# Update state in place with the new values.
jax.config.update('jax_num_cpu_devices', 4)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from time import perf_counter

# Step 2 — Place the data and model: Compare both the numerical global result and the placement needed...
# Verify contract: `jax.local_device_count() == 4`.
assert jax.local_device_count() == 4, 'Restart with four logical CPU devices'
# Configure multi-device placement / sharding specification (`mesh`).
mesh = Mesh(np.asarray(jax.devices()), ('data',))
# Configure multi-device placement / sharding specification (`rows`).
rows = NamedSharding(mesh, P('data', None))
# Configure multi-device placement / sharding specification (`replicated`).
replicated = NamedSharding(mesh, P())
# Configure multi-device placement / sharding specification (`features`).
features = NamedSharding(mesh, P('data'))
# Construct and reshape `xh` into the target tensor dimensions.
xh = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 128 - 0.5
# Initialize array `yh` with explicit values and shape.
yh = xh @ np.linspace(-0.4, 0.3, 8, dtype=np.float32)
# Initialize array `wh` with explicit values and shape.
wh = np.linspace(0.1, -0.2, 8, dtype=np.float32)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(xh, rows)
# Configure multi-device placement / sharding specification (`y`).
y = jax.device_put(yh, NamedSharding(mesh, P('data')))
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(wh, replicated)

# Step 3 — Implement partial gradients and collectives: Compare both the numerical global result and the placement needed...
def local_contribution(a, b, weights):
    # Return `2 * a.T @ (a @ weights - b) / 16` to the caller.
    return 2 * a.T @ (a @ weights - b) / 16

# Step 4 — Verify the result and placement: Compare both the numerical global result and the placement needed...
def reduce_all(a, b, weights):
    # Return `jax.lax.psum(local_contribution(a, b, weights), 'data')` to the caller.
    return jax.lax.psum(local_contribution(a, b, weights), 'data')

# Step 5 — Measure completed calls: Compare both the numerical global result and the placement needed...
def reduce_shard(a, b, weights):
    # Return `jax.lax.psum_scatter(local_contribution(a, b, weights), 'data', tiled=True)` to the caller.
    return jax.lax.psum_scatter(local_contribution(a, b, weights), 'data', tiled=True)

# Step 6 — Measure completed calls: Compare both the numerical global result and the placement needed...
# Configure multi-device placement / sharding specification (`all_gradient`).
all_gradient = jax.jit(jax.shard_map(reduce_all, mesh=mesh,
    in_specs=(P('data', None), P('data'), P()), out_specs=P()))
# Configure multi-device placement / sharding specification (`shard_gradient`).
shard_gradient = jax.jit(jax.shard_map(reduce_shard, mesh=mesh,
    in_specs=(P('data', None), P('data'), P()), out_specs=P('data')))
# Configure multi-device placement / sharding specification (`gather`).
gather = jax.jit(jax.shard_map(lambda g: jax.lax.all_gather(g, 'data', tiled=True),
    mesh=mesh, in_specs=P('data'), out_specs=P(), check_vma=False))
# Run `all_gradient` to compute `ga`.
ga = all_gradient(x, y, w)
# Run `shard_gradient` to compute `gs`.
gs = shard_gradient(x, y, w)
# Run `gather` to compute `gg`.
gg = gather(gs)
# Perform matrix contraction / projection to compute `reference`.
reference = 2 * xh.T @ (xh @ wh - yh) / 16
# Iterate over `result` to step through the computation:
for result in [ga, gs, gg]:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(result), reference, atol=2e-6, rtol=2e-6)
# Verify that the output tensor shape matches our prediction.
assert all(s.data.shape == (8,) for s in ga.addressable_shards)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert all(s.data.shape == (2,) for s in gs.addressable_shards)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert all(s.data.shape == (8,) for s in gg.addressable_shards)
# Trace or lower the function to inspect its compiler representation (`hlo_all`).
hlo_all = str(all_gradient.lower(x, y, w).compiler_ir(dialect='stablehlo'))
# Run `str` to compute `hlo_shard`.
hlo_shard = str(shard_gradient.lower(x, y, w).compiler_ir(dialect='stablehlo'))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert 'all_reduce' in hlo_all and 'reduce_scatter' in hlo_shard

# Step 7 — Measure completed calls: Compare both the numerical global result and the placement needed...
def completed_samples(fn, *args):
    # Synchronize host execution until asynchronous device computation completes.
    fn(*args).block_until_ready()
    # Evaluate `samples` from the current inputs and state.
    samples = []
    # Repeat the update loop over `range(7)` steps:
    for _ in range(7):
        # Record execution timing or profiler trace in `start`.
        start = perf_counter()
        # Synchronize host execution until asynchronous device computation completes.
        fn(*args).block_until_ready()
        # Record execution timing or profiler trace in ``.
        samples.append((perf_counter() - start) * 1e6)
    # Return `samples` to the caller.
    return samples
# Run `completed_samples` to compute `all_us`.
all_us = completed_samples(all_gradient, x, y, w)
# Run `completed_samples` to compute `shard_us`.
shard_us = completed_samples(shard_gradient, x, y, w)
# Run `completed_samples` to compute `gather_us`.
gather_us = completed_samples(gather, gs)
# Idealized ring payload per rank; not a network measurement.
ranks, gradient_bytes = 4, wh.nbytes
# Evaluate `ring_scatter_bytes` from the current inputs and state.
ring_scatter_bytes = (ranks - 1) / ranks * gradient_bytes
# Evaluate `ring_all_bytes` from the current inputs and state.
ring_all_bytes = 2 * ring_scatter_bytes
# Print the observed values to compare against the expected result.
print('Gradient reference:', reference.tolist())
# Print diagnostic summary of the computed outputs.
print('Local result shapes: replicated (8,), reduce-scatter (2,), gathered (8,)')
# Print diagnostic summary of the computed outputs.
print('Idealized ring bytes per rank:', ring_all_bytes, ring_scatter_bytes)
# Print diagnostic summary of the computed outputs.
print('Completed CPU median microseconds:', {
    'all_reduce': float(np.median(all_us)), 'reduce_scatter': float(np.median(shard_us)),
    'all_gather_only': float(np.median(gather_us))})

# Figure data experiment
# Compute figure data for: One mathematical gradient, different local storage
# Evaluate `visual_data` from the current inputs and state.
visual_data={'panels':[{'kind':'bar','labels':['all-reduce','reduce-scatter','after gather'],'ylabel':'values stored per device','series':[{'label':'observed local shape','y':[ga.addressable_shards[0].data.size,gs.addressable_shards[0].data.size,gg.addressable_shards[0].data.size]}]},{'kind':'bar','labels':['all-reduce','reduce-scatter','scatter + gather'],'ylabel':'modeled bytes per rank','series':[{'label':'idealized ring payload','y':[ring_all_bytes,ring_scatter_bytes,ring_scatter_bytes*2]}]}]}

# Experiment: Update slices before gathering
# Experiment — Update slices before gathering: An elementwise update can consume the partitioned gradient.
# Initialize array `v_host` with explicit values and shape.
v_host = np.linspace(-0.02, 0.03, 8, dtype=np.float32)
# Place `ws` explicitly onto the target JAX device.
ws = jax.device_put(wh, features)
# Place `vs` explicitly onto the target JAX device.
vs = jax.device_put(v_host, features)
# Evaluate `new_v` from the current inputs and state.
new_v = 0.9 * vs + gs
# Evaluate `new_w` from the current inputs and state.
new_w = ws - 0.1 * new_v
# Run `gather` to compute `updated`.
updated = gather(new_w)
# Evaluate `expected_w` from the current inputs and state.
expected_w = wh - 0.1 * (0.9 * v_host + reference)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(updated), expected_w, atol=2e-6, rtol=2e-6)
# Iterate over `shard` to step through the computation:
for shard in updated.addressable_shards:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(shard.data), expected_w, atol=2e-6, rtol=2e-6)
# Verify that the output tensor shape matches our prediction.
assert all(s.data.shape == (2,) for s in new_v.addressable_shards)
# Print diagnostic summary of the computed outputs.
print('Sharded momentum matches the independent elementwise update.')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Repeat the gradient comparison after reversing both observation rows...
x2 = jax.device_put(xh[::-1].copy(), rows)
# Configure multi-device placement / sharding specification (`y2`).
y2 = jax.device_put(yh[::-1].copy(), NamedSharding(mesh, P('data')))
# Cast or evaluate `w2h` in explicit floating-point precision.
w2h = wh + np.float32(0.17)
# Place `w2` explicitly onto the target JAX device.
w2 = jax.device_put(w2h, replicated)
# Perform matrix contraction / projection to compute `r2`.
r2 = 2 * xh.T @ (xh @ w2h - yh) / 16
# Iterate over `answer` to step through the computation:
for answer in [all_gradient(x2, y2, w2), gather(shard_gradient(x2, y2, w2))]:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(answer), r2, atol=2e-6, rtol=2e-6)
    # Iterate over `shard` to step through the computation:
    for shard in answer.addressable_shards:
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_allclose(np.asarray(shard.data), r2, atol=2e-6, rtol=2e-6)
# Perform matrix contraction / projection to compute `wrong_labels`.
wrong_labels = 2 * xh.T @ (xh @ w2h - yh[::-1]) / 16
# Verify contract: `np.max(np.abs(wrong_labels - r2)) > 0.01`.
assert np.max(np.abs(wrong_labels-r2)) > 0.01
# Print the observed values to compare against the expected result.
print('Joint permutation preserves the gradient; label-only reversal does not.')

# Reference practice: Account for the complete communication cycle
# Account for the complete communication cycle (Transfer): The two-phase payload equals the all-reduce payload under...
budget = []
# Iterate over `p` to step through the computation:
for p in [2, 4, 8]:
    # Evaluate `size` from the current inputs and state.
    size = 2**20
    # Evaluate `rs` from the current inputs and state.
    rs = size * (p - 1) // p
    # Evaluate `ar` from the current inputs and state.
    ar = 2 * rs
    # Evaluate `cycle` from the current inputs and state.
    cycle = rs + rs
    # Verify contract: `ar == cycle`.
    assert ar == cycle
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
    assert size // p * p == size
    # Append the current step result to `budget`.
    budget.append({'ranks': p, 'all_reduce_bytes': ar,
                   'scatter_gather_bytes': cycle, 'shard_storage_bytes': size // p})
# Verify contract: `[row['all_reduce_bytes'] for row in budget] == [1048576, 1572864, 18...`.
assert [row['all_reduce_bytes'] for row in budget] == [1048576, 1572864, 1835008]
# Print the observed values to compare against the expected result.
print('Idealized payload/storage budget, not observed network traffic:', budget)
print("PASS: distributed-03")
