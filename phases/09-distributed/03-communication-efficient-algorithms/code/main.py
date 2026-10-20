"""Communication-efficient algorithms: worked experiments and reference solutions. CPU checks."""

# Configure four logical devices
# Step 1 — Configure four logical devices: Compare both the numerical global result and the placement needed...
import jax
# Update state in place with the new values.
jax.config.update('jax_platforms', 'cpu')
# Update state in place with the new values.
jax.config.update('jax_num_cpu_devices', 4)
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from time import perf_counter

# Place the data and model
# Step 2 — Place the data and model: Compare both the numerical global result and the placement needed...
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
yh = xh @ np.linspace(-0.4, 0.3, 8, dtype=np.float32)
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
    return 2 * a.T @ (a @ weights - b) / 16

# Verify the result and placement
# Step 4 — Verify the result and placement: Compare both the numerical global result and the placement needed...
def reduce_all(a, b, weights):
    return jax.lax.psum(local_contribution(a, b, weights), 'data')

# Measure completed calls
# Step 5 — Measure completed calls: Compare both the numerical global result and the placement needed...
def reduce_shard(a, b, weights):
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
ga = all_gradient(x, y, w)
gs = shard_gradient(x, y, w)
gg = gather(gs)
reference = 2 * xh.T @ (xh @ wh - yh) / 16
for result in [ga, gs, gg]:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(result), reference, atol=2e-6, rtol=2e-6)
# Check tensor shape invariant: `all(s.data.shape == (8,) for s in ga.addressable_shards)`
assert all(s.data.shape == (8,) for s in ga.addressable_shards)
# Check tensor shape invariant: `all(s.data.shape == (2,) for s in gs.addressable_shards)`
assert all(s.data.shape == (2,) for s in gs.addressable_shards)
# Check tensor shape invariant: `all(s.data.shape == (8,) for s in gg.addressable_shards)`
assert all(s.data.shape == (8,) for s in gg.addressable_shards)
# Trace or lower the function to inspect its compiler representation (`hlo_all`).
hlo_all = str(all_gradient.lower(x, y, w).compiler_ir(dialect='stablehlo'))
hlo_shard = str(shard_gradient.lower(x, y, w).compiler_ir(dialect='stablehlo'))
assert 'all_reduce' in hlo_all and 'reduce_scatter' in hlo_shard

# Measure completed calls
# Step 7 — Measure completed calls: Compare both the numerical global result and the placement needed...
def completed_samples(fn, *args):
    # Synchronize host execution until asynchronous device computation completes.
    fn(*args).block_until_ready()
    samples = []
    for _ in range(7):
        start = perf_counter()
        # Synchronize host execution until asynchronous device computation completes.
        fn(*args).block_until_ready()
        samples.append((perf_counter() - start) * 1e6)
    return samples
all_us = completed_samples(all_gradient, x, y, w)
shard_us = completed_samples(shard_gradient, x, y, w)
gather_us = completed_samples(gather, gs)
# Idealized ring payload per rank; not a network measurement.
ranks, gradient_bytes = 4, wh.nbytes
ring_scatter_bytes = (ranks - 1) / ranks * gradient_bytes
ring_all_bytes = 2 * ring_scatter_bytes
print('Gradient reference:', reference.tolist())
print('Local result shapes: replicated (8,), reduce-scatter (2,), gathered (8,)')
print('Idealized ring bytes per rank:', ring_all_bytes, ring_scatter_bytes)
print('Completed CPU median microseconds:', {
    'all_reduce': float(np.median(all_us)), 'reduce_scatter': float(np.median(shard_us)),
    'all_gather_only': float(np.median(gather_us))})

# Step 1 — Configure four logical devices: Compare both the numerical global result and the placement needed...
import jax
# Update state in place with the new values.
jax.config.update('jax_platforms', 'cpu')
# Update state in place with the new values.
jax.config.update('jax_num_cpu_devices', 4)
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from time import perf_counter

# Step 2 — Place the data and model: Compare both the numerical global result and the placement needed...
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
yh = xh @ np.linspace(-0.4, 0.3, 8, dtype=np.float32)
wh = np.linspace(0.1, -0.2, 8, dtype=np.float32)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(xh, rows)
# Configure multi-device placement / sharding specification (`y`).
y = jax.device_put(yh, NamedSharding(mesh, P('data')))
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(wh, replicated)

# Step 3 — Implement partial gradients and collectives: Compare both the numerical global result and the placement needed...
def local_contribution(a, b, weights):
    return 2 * a.T @ (a @ weights - b) / 16

# Step 4 — Verify the result and placement: Compare both the numerical global result and the placement needed...
def reduce_all(a, b, weights):
    return jax.lax.psum(local_contribution(a, b, weights), 'data')

# Step 5 — Measure completed calls: Compare both the numerical global result and the placement needed...
def reduce_shard(a, b, weights):
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
ga = all_gradient(x, y, w)
gs = shard_gradient(x, y, w)
gg = gather(gs)
reference = 2 * xh.T @ (xh @ wh - yh) / 16
for result in [ga, gs, gg]:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(result), reference, atol=2e-6, rtol=2e-6)
# Check tensor shape invariant: `all(s.data.shape == (8,) for s in ga.addressable_shards)`
assert all(s.data.shape == (8,) for s in ga.addressable_shards)
# Check tensor shape invariant: `all(s.data.shape == (2,) for s in gs.addressable_shards)`
assert all(s.data.shape == (2,) for s in gs.addressable_shards)
# Check tensor shape invariant: `all(s.data.shape == (8,) for s in gg.addressable_shards)`
assert all(s.data.shape == (8,) for s in gg.addressable_shards)
# Trace or lower the function to inspect its compiler representation (`hlo_all`).
hlo_all = str(all_gradient.lower(x, y, w).compiler_ir(dialect='stablehlo'))
hlo_shard = str(shard_gradient.lower(x, y, w).compiler_ir(dialect='stablehlo'))
assert 'all_reduce' in hlo_all and 'reduce_scatter' in hlo_shard

# Step 7 — Measure completed calls: Compare both the numerical global result and the placement needed...
def completed_samples(fn, *args):
    # Synchronize host execution until asynchronous device computation completes.
    fn(*args).block_until_ready()
    samples = []
    for _ in range(7):
        start = perf_counter()
        # Synchronize host execution until asynchronous device computation completes.
        fn(*args).block_until_ready()
        samples.append((perf_counter() - start) * 1e6)
    return samples
all_us = completed_samples(all_gradient, x, y, w)
shard_us = completed_samples(shard_gradient, x, y, w)
gather_us = completed_samples(gather, gs)
# Idealized ring payload per rank; not a network measurement.
ranks, gradient_bytes = 4, wh.nbytes
ring_scatter_bytes = (ranks - 1) / ranks * gradient_bytes
ring_all_bytes = 2 * ring_scatter_bytes
print('Gradient reference:', reference.tolist())
print('Local result shapes: replicated (8,), reduce-scatter (2,), gathered (8,)')
print('Idealized ring bytes per rank:', ring_all_bytes, ring_scatter_bytes)
print('Completed CPU median microseconds:', {
    'all_reduce': float(np.median(all_us)), 'reduce_scatter': float(np.median(shard_us)),
    'all_gather_only': float(np.median(gather_us))})

# Figure data experiment
# Compute figure data for: One mathematical gradient, different local storage
visual_data={'panels':[{'kind':'bar','labels':['all-reduce','reduce-scatter','after gather'],'ylabel':'values stored per device','series':[{'label':'observed local shape','y':[ga.addressable_shards[0].data.size,gs.addressable_shards[0].data.size,gg.addressable_shards[0].data.size]}]},{'kind':'bar','labels':['all-reduce','reduce-scatter','scatter + gather'],'ylabel':'modeled bytes per rank','series':[{'label':'idealized ring payload','y':[ring_all_bytes,ring_scatter_bytes,ring_scatter_bytes*2]}]}]}

# Experiment: Update slices before gathering
# Experiment — Update slices before gathering: An elementwise update can consume the partitioned gradient.
v_host = np.linspace(-0.02, 0.03, 8, dtype=np.float32)
# Place `ws` explicitly onto the target JAX device.
ws = jax.device_put(wh, features)
# Place `vs` explicitly onto the target JAX device.
vs = jax.device_put(v_host, features)
new_v = 0.9 * vs + gs
new_w = ws - 0.1 * new_v
updated = gather(new_w)
expected_w = wh - 0.1 * (0.9 * v_host + reference)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(updated), expected_w, atol=2e-6, rtol=2e-6)
for shard in updated.addressable_shards:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(shard.data), expected_w, atol=2e-6, rtol=2e-6)
# Check tensor shape invariant: `all(s.data.shape == (2,) for s in new_v.addressable_shards)`
assert all(s.data.shape == (2,) for s in new_v.addressable_shards)
print('Sharded momentum matches the independent elementwise update.')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Repeat the gradient comparison after reversing both observation rows...
x2 = jax.device_put(xh[::-1].copy(), rows)
# Configure multi-device placement / sharding specification (`y2`).
y2 = jax.device_put(yh[::-1].copy(), NamedSharding(mesh, P('data')))
w2h = wh + np.float32(0.17)
# Place `w2` explicitly onto the target JAX device.
w2 = jax.device_put(w2h, replicated)
r2 = 2 * xh.T @ (xh @ w2h - yh) / 16
for answer in [all_gradient(x2, y2, w2), gather(shard_gradient(x2, y2, w2))]:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(answer), r2, atol=2e-6, rtol=2e-6)
    for shard in answer.addressable_shards:
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_allclose(np.asarray(shard.data), r2, atol=2e-6, rtol=2e-6)
wrong_labels = 2 * xh.T @ (xh @ w2h - yh[::-1]) / 16
assert np.max(np.abs(wrong_labels-r2)) > 0.01
print('Joint permutation preserves the gradient; label-only reversal does not.')

# Reference practice: Account for the complete communication cycle
# Account for the complete communication cycle (Transfer): The two-phase payload equals the all-reduce payload under...
budget = []
for p in [2, 4, 8]:
    size = 2**20
    rs = size * (p - 1) // p
    ar = 2 * rs
    cycle = rs + rs
    assert ar == cycle
    assert size // p * p == size
    budget.append({'ranks': p, 'all_reduce_bytes': ar,
                   'scatter_gather_bytes': cycle, 'shard_storage_bytes': size // p})
assert [row['all_reduce_bytes'] for row in budget] == [1048576, 1572864, 1835008]
print('Idealized payload/storage budget, not observed network traffic:', budget)
print("PASS: distributed-03")
