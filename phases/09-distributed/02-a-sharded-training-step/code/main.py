"""A sharded training step: worked experiments and reference solutions. CPU checks."""

# Start a fresh CPU runtime
# Step 1 — Start a fresh CPU runtime
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
host_x = np.array([[1,0],[0,1],[1,1],[2,0],[0,2],[2,1],[1,2],[2,2]],dtype=np.float32)
true_w = np.array([2.,-1.],dtype=np.float32)
host_y = host_x @ true_w
labels_sharding = NamedSharding(mesh,P("data"))
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host_x, rows)
# Place `y` explicitly onto the target JAX device.
y = jax.device_put(host_y, labels_sharding)
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(np.zeros(2,dtype=np.float32),replicated)
def loss(weights, features, labels):
    return jnp.mean((features @ weights-labels)**2)
def update(weights, features, labels):
    value, gradient = jax.value_and_grad(loss)(weights,features,labels)
    return weights - 0.1*gradient, value, gradient
# Wrap with `jax.jit` (`train_step`) so XLA traces and compiles the function.
train_step = jax.jit(update, in_shardings=(replicated,rows,labels_sharding),out_shardings=(replicated,replicated,replicated))
new_w, before, gradient = train_step(w,x,y)

# Inspect and verify
# Step 3 — Inspect and verify
expected_gradient = 2 * host_x.T @ (-host_y) / len(host_y)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(gradient),expected_gradient,rtol=1e-6,atol=1e-6)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(new_w),-0.1*expected_gradient,rtol=1e-6,atol=1e-6)
expected_loss = np.mean(host_y**2)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(before), expected_loss,rtol=1e-6)
assert float(loss(new_w,x,y)) < float(before)
for shard in new_w.addressable_shards:
    # Transfer the device result to the host for an independent NumPy comparison.
    np.testing.assert_allclose(np.asarray(shard.data),np.asarray(new_w))
print("Initial loss:",float(before))
print("Gradient:",np.asarray(gradient),"updated weights:",np.asarray(new_w))
print("Updated loss:",float(loss(new_w,x,y)))

# Run the 2D parameter-sharded update
# Step 4 — Run an actual 2D parameter-sharded training update
# This independent task uses 8 padded rows, 4 input features, and 4 output columns.
# It runs on a 2x2 logical CPU mesh and compares against an unsharded NumPy oracle.
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh2d = Mesh(np.asarray(jax.devices("cpu")).reshape(2, 2), ("data", "model"))
x_shard = NamedSharding(mesh2d, P("data", None))
weight_shard = NamedSharding(mesh2d, P(None, "model"))
target_shard = NamedSharding(mesh2d, P("data", "model"))
mask_shard = NamedSharding(mesh2d, P("data"))
replicated_scalar = NamedSharding(mesh2d, P())

x_host = (np.arange(32, dtype=np.float32).reshape(8, 4) - 7) / 10
weight_host = (np.arange(16, dtype=np.float32).reshape(4, 4) - 5) / 20
truth_host = np.full((4, 4), 0.2, dtype=np.float32)
target_host = x_host @ truth_host
momentum_host = np.full((4, 4), 0.1, dtype=np.float32)
x_2d = jax.device_put(x_host, x_shard)
target_2d = jax.device_put(target_host, target_shard)
weight_2d = jax.device_put(weight_host, weight_shard)
momentum_2d = jax.device_put(momentum_host, weight_shard)

# The global masked mean is numerator / (valid examples * output columns).
def masked_loss_2d(weight, features, targets, valid):
    residual = features @ weight - targets
    numerator = jnp.sum(jnp.square(residual) * valid[:, None])
    denominator = jnp.sum(valid) * targets.shape[1]
    return numerator / denominator

@jax.jit(
    in_shardings=(weight_shard, x_shard, target_shard, mask_shard),
    out_shardings=(replicated_scalar, weight_shard),
)
def value_and_grad_2d(weight, features, targets, valid):
    return jax.value_and_grad(masked_loss_2d)(weight, features, targets, valid)

learning_rate_2d = 0.05
momentum_rate_2d = 0.8
@jax.jit(
    in_shardings=(weight_shard, weight_shard, weight_shard),
    out_shardings=(weight_shard, weight_shard),
)
def momentum_update_2d(weight, momentum, gradient):
    next_momentum = momentum_rate_2d * momentum + gradient
    next_weight = weight - learning_rate_2d * next_momentum
    return next_weight, next_momentum

# Five valid rows create unequal local counts (4 and 1); three create an empty shard (3 and 0).
for valid_count_2d in (5, 3):
    valid_host_2d = np.r_[np.ones(valid_count_2d, np.float32),
                          np.zeros(8 - valid_count_2d, np.float32)]
    valid_2d = jax.device_put(valid_host_2d, mask_shard)
    loss_2d, gradient_2d = value_and_grad_2d(
        weight_2d, x_2d, target_2d, valid_2d
    )
    next_weight_2d, next_momentum_2d = momentum_update_2d(
        weight_2d, momentum_2d, gradient_2d
    )
    loss_2d.block_until_ready()

    # Independent unsharded oracle: no JAX sharding or autodiff is used here.
    residual_host_2d = x_host @ weight_host - target_host
    oracle_loss_2d = np.sum(residual_host_2d ** 2 * valid_host_2d[:, None]) / (valid_count_2d * 4)
    oracle_gradient_2d = (
        2 * x_host.T @ (residual_host_2d * valid_host_2d[:, None])
        / (valid_count_2d * 4)
    )
    oracle_momentum_2d = momentum_rate_2d * momentum_host + oracle_gradient_2d
    oracle_weight_2d = weight_host - learning_rate_2d * oracle_momentum_2d
    np.testing.assert_allclose(np.asarray(loss_2d), oracle_loss_2d, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(gradient_2d), oracle_gradient_2d, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(next_weight_2d), oracle_weight_2d, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(next_momentum_2d), oracle_momentum_2d, rtol=2e-5, atol=2e-6)
    local_valid_counts_2d = [
        int(np.asarray(shard.data).sum())
        for shard in valid_2d.addressable_shards
    ]
    expected_counts_2d = [4, 4, 1, 1] if valid_count_2d == 5 else [3, 3, 0, 0]
    assert sorted(local_valid_counts_2d) == sorted(expected_counts_2d)
    local_means_2d = []
    for row_start_2d in (0, 4):
        row_stop_2d = row_start_2d + 4
        shard_count_2d = valid_host_2d[row_start_2d:row_stop_2d].sum()
        shard_numerator_2d = np.sum(
            residual_host_2d[row_start_2d:row_stop_2d] ** 2
            * valid_host_2d[row_start_2d:row_stop_2d, None]
        )
        if shard_count_2d:
            local_means_2d.append(shard_numerator_2d / (shard_count_2d * 4))
        else:
            assert shard_numerator_2d == 0
    if valid_count_2d == 5:
        # Negative control: equal averaging of 4-example and 1-example means is wrong.
        naive_local_mean_2d = float(np.mean(local_means_2d))
        assert not np.isclose(naive_local_mean_2d, oracle_loss_2d)
        print(f"valid=5; naive equal mean of nonempty shard means={naive_local_mean_2d:.6f} (wrong); weighted global loss={oracle_loss_2d:.6f}")
    else:
        print("valid=3; empty data shard contributes zero sum and zero count")
    print(f"valid={valid_count_2d}; device valid counts={local_valid_counts_2d}; loss={float(loss_2d):.6f}")

# Report per-device ownership from the actual addressable shards. All are float32.
def shard_report_2d(name, array):
    rows = []
    for shard in array.addressable_shards:
        rows.append((shard.device.id, shard.data.shape, int(shard.data.size * shard.data.dtype.itemsize)))
    print(f"{name} per device (device id, local shape, bytes): {rows}")

for name_2d, value_2d in [
    ("batch features", x_2d),
    ("targets", target_2d),
    ("parameters", weight_2d),
    ("momentum state", momentum_2d),
    ("parameter gradient", gradient_2d),
    ("updated parameters", next_weight_2d),
    ("updated momentum", next_momentum_2d),
]:
    shard_report_2d(name_2d, value_2d)
print("2D parameter-sharded forward/backward/update matches the unsharded NumPy oracle")

# Step 1 — Start a fresh CPU runtime
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
host_x = np.array([[1,0],[0,1],[1,1],[2,0],[0,2],[2,1],[1,2],[2,2]],dtype=np.float32)
true_w = np.array([2.,-1.],dtype=np.float32)
host_y = host_x @ true_w
labels_sharding = NamedSharding(mesh,P("data"))
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host_x, rows)
# Place `y` explicitly onto the target JAX device.
y = jax.device_put(host_y, labels_sharding)
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(np.zeros(2,dtype=np.float32),replicated)
def loss(weights, features, labels):
    return jnp.mean((features @ weights-labels)**2)
def update(weights, features, labels):
    value, gradient = jax.value_and_grad(loss)(weights,features,labels)
    return weights - 0.1*gradient, value, gradient
# Wrap with `jax.jit` (`train_step`) so XLA traces and compiles the function.
train_step = jax.jit(update, in_shardings=(replicated,rows,labels_sharding),out_shardings=(replicated,replicated,replicated))
new_w, before, gradient = train_step(w,x,y)

# Step 3 — Inspect and verify
expected_gradient = 2 * host_x.T @ (-host_y) / len(host_y)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(gradient),expected_gradient,rtol=1e-6,atol=1e-6)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(new_w),-0.1*expected_gradient,rtol=1e-6,atol=1e-6)
expected_loss = np.mean(host_y**2)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(before), expected_loss,rtol=1e-6)
assert float(loss(new_w,x,y)) < float(before)
for shard in new_w.addressable_shards:
    # Transfer the device result to the host for an independent NumPy comparison.
    np.testing.assert_allclose(np.asarray(shard.data),np.asarray(new_w))
print("Initial loss:",float(before))
print("Gradient:",np.asarray(gradient),"updated weights:",np.asarray(new_w))
print("Updated loss:",float(loss(new_w,x,y)))

# Step 4 — Run an actual 2D parameter-sharded training update
# This independent task uses 8 padded rows, 4 input features, and 4 output columns.
# It runs on a 2x2 logical CPU mesh and compares against an unsharded NumPy oracle.
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh2d = Mesh(np.asarray(jax.devices("cpu")).reshape(2, 2), ("data", "model"))
x_shard = NamedSharding(mesh2d, P("data", None))
weight_shard = NamedSharding(mesh2d, P(None, "model"))
target_shard = NamedSharding(mesh2d, P("data", "model"))
mask_shard = NamedSharding(mesh2d, P("data"))
replicated_scalar = NamedSharding(mesh2d, P())

x_host = (np.arange(32, dtype=np.float32).reshape(8, 4) - 7) / 10
weight_host = (np.arange(16, dtype=np.float32).reshape(4, 4) - 5) / 20
truth_host = np.full((4, 4), 0.2, dtype=np.float32)
target_host = x_host @ truth_host
momentum_host = np.full((4, 4), 0.1, dtype=np.float32)
x_2d = jax.device_put(x_host, x_shard)
target_2d = jax.device_put(target_host, target_shard)
weight_2d = jax.device_put(weight_host, weight_shard)
momentum_2d = jax.device_put(momentum_host, weight_shard)

# The global masked mean is numerator / (valid examples * output columns).
def masked_loss_2d(weight, features, targets, valid):
    residual = features @ weight - targets
    numerator = jnp.sum(jnp.square(residual) * valid[:, None])
    denominator = jnp.sum(valid) * targets.shape[1]
    return numerator / denominator

@jax.jit(
    in_shardings=(weight_shard, x_shard, target_shard, mask_shard),
    out_shardings=(replicated_scalar, weight_shard),
)
def value_and_grad_2d(weight, features, targets, valid):
    return jax.value_and_grad(masked_loss_2d)(weight, features, targets, valid)

learning_rate_2d = 0.05
momentum_rate_2d = 0.8
@jax.jit(
    in_shardings=(weight_shard, weight_shard, weight_shard),
    out_shardings=(weight_shard, weight_shard),
)
def momentum_update_2d(weight, momentum, gradient):
    next_momentum = momentum_rate_2d * momentum + gradient
    next_weight = weight - learning_rate_2d * next_momentum
    return next_weight, next_momentum

# Five valid rows create unequal local counts (4 and 1); three create an empty shard (3 and 0).
for valid_count_2d in (5, 3):
    valid_host_2d = np.r_[np.ones(valid_count_2d, np.float32),
                          np.zeros(8 - valid_count_2d, np.float32)]
    valid_2d = jax.device_put(valid_host_2d, mask_shard)
    loss_2d, gradient_2d = value_and_grad_2d(
        weight_2d, x_2d, target_2d, valid_2d
    )
    next_weight_2d, next_momentum_2d = momentum_update_2d(
        weight_2d, momentum_2d, gradient_2d
    )
    loss_2d.block_until_ready()

    # Independent unsharded oracle: no JAX sharding or autodiff is used here.
    residual_host_2d = x_host @ weight_host - target_host
    oracle_loss_2d = np.sum(residual_host_2d ** 2 * valid_host_2d[:, None]) / (valid_count_2d * 4)
    oracle_gradient_2d = (
        2 * x_host.T @ (residual_host_2d * valid_host_2d[:, None])
        / (valid_count_2d * 4)
    )
    oracle_momentum_2d = momentum_rate_2d * momentum_host + oracle_gradient_2d
    oracle_weight_2d = weight_host - learning_rate_2d * oracle_momentum_2d
    np.testing.assert_allclose(np.asarray(loss_2d), oracle_loss_2d, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(gradient_2d), oracle_gradient_2d, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(next_weight_2d), oracle_weight_2d, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(next_momentum_2d), oracle_momentum_2d, rtol=2e-5, atol=2e-6)
    local_valid_counts_2d = [
        int(np.asarray(shard.data).sum())
        for shard in valid_2d.addressable_shards
    ]
    expected_counts_2d = [4, 4, 1, 1] if valid_count_2d == 5 else [3, 3, 0, 0]
    assert sorted(local_valid_counts_2d) == sorted(expected_counts_2d)
    local_means_2d = []
    for row_start_2d in (0, 4):
        row_stop_2d = row_start_2d + 4
        shard_count_2d = valid_host_2d[row_start_2d:row_stop_2d].sum()
        shard_numerator_2d = np.sum(
            residual_host_2d[row_start_2d:row_stop_2d] ** 2
            * valid_host_2d[row_start_2d:row_stop_2d, None]
        )
        if shard_count_2d:
            local_means_2d.append(shard_numerator_2d / (shard_count_2d * 4))
        else:
            assert shard_numerator_2d == 0
    if valid_count_2d == 5:
        # Negative control: equal averaging of 4-example and 1-example means is wrong.
        naive_local_mean_2d = float(np.mean(local_means_2d))
        assert not np.isclose(naive_local_mean_2d, oracle_loss_2d)
        print(f"valid=5; naive equal mean of nonempty shard means={naive_local_mean_2d:.6f} (wrong); weighted global loss={oracle_loss_2d:.6f}")
    else:
        print("valid=3; empty data shard contributes zero sum and zero count")
    print(f"valid={valid_count_2d}; device valid counts={local_valid_counts_2d}; loss={float(loss_2d):.6f}")

# Report per-device ownership from the actual addressable shards. All are float32.
def shard_report_2d(name, array):
    rows = []
    for shard in array.addressable_shards:
        rows.append((shard.device.id, shard.data.shape, int(shard.data.size * shard.data.dtype.itemsize)))
    print(f"{name} per device (device id, local shape, bytes): {rows}")

for name_2d, value_2d in [
    ("batch features", x_2d),
    ("targets", target_2d),
    ("parameters", weight_2d),
    ("momentum state", momentum_2d),
    ("parameter gradient", gradient_2d),
    ("updated parameters", next_weight_2d),
    ("updated momentum", next_momentum_2d),
]:
    shard_report_2d(name_2d, value_2d)
print("2D parameter-sharded forward/backward/update matches the unsharded NumPy oracle")

# Figure data experiment
# Derive the table from actual local shards rather than hardcoded byte labels.
table_rows = []
for label, array, spec_label in [
    ('batch X', x_2d, 'P(data,None)'),
    ('target Y', target_2d, 'P(data,model)'),
    ('weights W', weight_2d, 'P(None,model)'),
    ('momentum', momentum_2d, 'P(None,model)'),
    ('gradient dW', gradient_2d, 'P(None,model)'),
]:
    shards = array.addressable_shards
    shapes = {tuple(s.data.shape) for s in shards}
    sizes = {int(s.data.size * s.data.dtype.itemsize) for s in shards}
    assert len(shapes) == len(sizes) == 1
    table_rows.append([label, str(tuple(array.shape)) + '\n' + spec_label,
                       str(next(iter(shapes))), next(iter(sizes))])
visual_data = {'kind': 'table', 'headers': ['Array','Global shape/spec','Each device','Bytes/device'],
               'rows': table_rows, 'widths': [.17,.40,.23,.20]}


# Experiment: Permutation across shards
# Experiment — Permutation across shards: The global mean is invariant to row order.
reverse_x = jax.device_put(host_x[::-1].copy(),rows)
# Place `reverse_y` explicitly onto the target JAX device.
reverse_y = jax.device_put(host_y[::-1].copy(),labels_sharding)
reverse_w,reverse_loss,reverse_gradient = train_step(w,reverse_x,reverse_y)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(reverse_gradient),expected_gradient,rtol=1e-6,atol=1e-6)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(reverse_w),np.asarray(new_w),rtol=1e-6,atol=1e-6)
print("Permutation preserves the global update")

# Experiment: Ten updates against a host training loop
# Experiment — Ten updates against a host training loop
cpu_weights = w
reference_weights = np.zeros(2,dtype=np.float32)
for step_index in range(10):
    cpu_weights,_,_ = train_step(cpu_weights,x,y)
    residual = host_x @ reference_weights-host_y
    # Accumulate the next contribution into `reference_weights`.
    reference_weights -= 0.1*(2*host_x.T @ residual/len(host_y))
    # Transfer the device result to the host for an independent NumPy comparison.
    np.testing.assert_allclose(np.asarray(cpu_weights),reference_weights,rtol=1e-5,atol=1e-6)
print("Ten-step weights:",np.asarray(cpu_weights))

# Experiment: Parameter-sharded update with unequal and empty valid counts
# This independent task uses 8 padded rows, 4 input features, and 4 output columns.
# It runs on a 2x2 logical CPU mesh and compares against an unsharded NumPy oracle.
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh2d = Mesh(np.asarray(jax.devices("cpu")).reshape(2, 2), ("data", "model"))
x_shard = NamedSharding(mesh2d, P("data", None))
weight_shard = NamedSharding(mesh2d, P(None, "model"))
target_shard = NamedSharding(mesh2d, P("data", "model"))
mask_shard = NamedSharding(mesh2d, P("data"))
replicated_scalar = NamedSharding(mesh2d, P())

x_host = (np.arange(32, dtype=np.float32).reshape(8, 4) - 7) / 10
weight_host = (np.arange(16, dtype=np.float32).reshape(4, 4) - 5) / 20
truth_host = np.full((4, 4), 0.2, dtype=np.float32)
target_host = x_host @ truth_host
momentum_host = np.full((4, 4), 0.1, dtype=np.float32)
x_2d = jax.device_put(x_host, x_shard)
target_2d = jax.device_put(target_host, target_shard)
weight_2d = jax.device_put(weight_host, weight_shard)
momentum_2d = jax.device_put(momentum_host, weight_shard)

# The global masked mean is numerator / (valid examples * output columns).
def masked_loss_2d(weight, features, targets, valid):
    residual = features @ weight - targets
    numerator = jnp.sum(jnp.square(residual) * valid[:, None])
    denominator = jnp.sum(valid) * targets.shape[1]
    return numerator / denominator

@jax.jit(
    in_shardings=(weight_shard, x_shard, target_shard, mask_shard),
    out_shardings=(replicated_scalar, weight_shard),
)
def value_and_grad_2d(weight, features, targets, valid):
    return jax.value_and_grad(masked_loss_2d)(weight, features, targets, valid)

learning_rate_2d = 0.05
momentum_rate_2d = 0.8
@jax.jit(
    in_shardings=(weight_shard, weight_shard, weight_shard),
    out_shardings=(weight_shard, weight_shard),
)
def momentum_update_2d(weight, momentum, gradient):
    next_momentum = momentum_rate_2d * momentum + gradient
    next_weight = weight - learning_rate_2d * next_momentum
    return next_weight, next_momentum

# Five valid rows create unequal local counts (4 and 1); three create an empty shard (3 and 0).
for valid_count_2d in (5, 3):
    valid_host_2d = np.r_[np.ones(valid_count_2d, np.float32),
                          np.zeros(8 - valid_count_2d, np.float32)]
    valid_2d = jax.device_put(valid_host_2d, mask_shard)
    loss_2d, gradient_2d = value_and_grad_2d(
        weight_2d, x_2d, target_2d, valid_2d
    )
    next_weight_2d, next_momentum_2d = momentum_update_2d(
        weight_2d, momentum_2d, gradient_2d
    )
    loss_2d.block_until_ready()

    # Independent unsharded oracle: no JAX sharding or autodiff is used here.
    residual_host_2d = x_host @ weight_host - target_host
    oracle_loss_2d = np.sum(residual_host_2d ** 2 * valid_host_2d[:, None]) / (valid_count_2d * 4)
    oracle_gradient_2d = (
        2 * x_host.T @ (residual_host_2d * valid_host_2d[:, None])
        / (valid_count_2d * 4)
    )
    oracle_momentum_2d = momentum_rate_2d * momentum_host + oracle_gradient_2d
    oracle_weight_2d = weight_host - learning_rate_2d * oracle_momentum_2d
    np.testing.assert_allclose(np.asarray(loss_2d), oracle_loss_2d, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(gradient_2d), oracle_gradient_2d, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(next_weight_2d), oracle_weight_2d, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(next_momentum_2d), oracle_momentum_2d, rtol=2e-5, atol=2e-6)
    local_valid_counts_2d = [
        int(np.asarray(shard.data).sum())
        for shard in valid_2d.addressable_shards
    ]
    expected_counts_2d = [4, 4, 1, 1] if valid_count_2d == 5 else [3, 3, 0, 0]
    assert sorted(local_valid_counts_2d) == sorted(expected_counts_2d)
    local_means_2d = []
    for row_start_2d in (0, 4):
        row_stop_2d = row_start_2d + 4
        shard_count_2d = valid_host_2d[row_start_2d:row_stop_2d].sum()
        shard_numerator_2d = np.sum(
            residual_host_2d[row_start_2d:row_stop_2d] ** 2
            * valid_host_2d[row_start_2d:row_stop_2d, None]
        )
        if shard_count_2d:
            local_means_2d.append(shard_numerator_2d / (shard_count_2d * 4))
        else:
            assert shard_numerator_2d == 0
    if valid_count_2d == 5:
        # Negative control: equal averaging of 4-example and 1-example means is wrong.
        naive_local_mean_2d = float(np.mean(local_means_2d))
        assert not np.isclose(naive_local_mean_2d, oracle_loss_2d)
        print(f"valid=5; naive equal mean of nonempty shard means={naive_local_mean_2d:.6f} (wrong); weighted global loss={oracle_loss_2d:.6f}")
    else:
        print("valid=3; empty data shard contributes zero sum and zero count")
    print(f"valid={valid_count_2d}; device valid counts={local_valid_counts_2d}; loss={float(loss_2d):.6f}")

# Report per-device ownership from the actual addressable shards. All are float32.
def shard_report_2d(name, array):
    rows = []
    for shard in array.addressable_shards:
        rows.append((shard.device.id, shard.data.shape, int(shard.data.size * shard.data.dtype.itemsize)))
    print(f"{name} per device (device id, local shape, bytes): {rows}")

for name_2d, value_2d in [
    ("batch features", x_2d),
    ("targets", target_2d),
    ("parameters", weight_2d),
    ("momentum state", momentum_2d),
    ("parameter gradient", gradient_2d),
    ("updated parameters", next_weight_2d),
    ("updated momentum", next_momentum_2d),
]:
    shard_report_2d(name_2d, value_2d)
print("2D parameter-sharded forward/backward/update matches the unsharded NumPy oracle")

# Reference solution. Try the exercise before reading this.
# Exercise solution
changed_y = host_x @ np.array([-1.,3.],dtype=np.float32)
start = np.array([0.2,-0.1],dtype=np.float32)
def changed_update(weights,features,labels):
    return weights-0.05*jax.grad(loss)(weights,features,labels)
# Wrap with `jax.jit` (`changed_step`) so XLA traces and compiles the function.
changed_step = jax.jit(changed_update,in_shardings=(replicated,rows,labels_sharding),out_shardings=replicated)
# Place `result` explicitly onto the target JAX device.
result = changed_step(jax.device_put(start,replicated),x,jax.device_put(changed_y,labels_sharding))
reference_gradient = 2*host_x.T @ (host_x@start-changed_y)/len(changed_y)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(result),start-0.05*reference_gradient,rtol=1e-6,atol=1e-6)

local_gradients = []
for features,labels in zip(np.split(host_x,4),np.split(host_y,4)):
    local_gradients.append(2*features.T @ (-labels)/len(labels))
# Reduce along axis=0 to compute `wrong`.
wrong = np.sum(local_gradients,axis=0)
# Reduce along axis=0 to compute `right`.
right = np.mean(local_gradients,axis=0)
np.testing.assert_allclose(wrong,4*expected_gradient,rtol=1e-6)
np.testing.assert_allclose(right,expected_gradient,rtol=1e-6)
# Unequal partitions: weight local mean gradients by their observation counts.
parts = [(host_x[:3],host_y[:3]),(host_x[3:],host_y[3:])]
weighted = sum(len(b)*(2*a.T @ (-b)/len(b)) for a,b in parts)/len(host_y)
np.testing.assert_allclose(weighted,expected_gradient,rtol=1e-6)
print("Summed local means are four times too large; weighted aggregation repaired")

# Reference practice: Transfer to a different target model
# Transfer to a different target model (Challenge)
changed_y = host_x @ np.array([-1.,3.],dtype=np.float32)
start = np.array([0.2,-0.1],dtype=np.float32)
def changed_update(weights,features,labels):
    return weights-0.05*jax.grad(loss)(weights,features,labels)
# Wrap with `jax.jit` (`changed_step`) so XLA traces and compiles the function.
changed_step = jax.jit(changed_update,in_shardings=(replicated,rows,labels_sharding),out_shardings=replicated)
# Place `result` explicitly onto the target JAX device.
result = changed_step(jax.device_put(start,replicated),x,jax.device_put(changed_y,labels_sharding))
reference_gradient = 2*host_x.T @ (host_x@start-changed_y)/len(changed_y)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(result),start-0.05*reference_gradient,rtol=1e-6,atol=1e-6)

# Reference practice: Diagnose summed local means
# Diagnose summed local means (Challenge): Each local gradient already divides by its local count.
local_gradients = []
for features,labels in zip(np.split(host_x,4),np.split(host_y,4)):
    local_gradients.append(2*features.T @ (-labels)/len(labels))
# Reduce along axis=0 to compute `wrong`.
wrong = np.sum(local_gradients,axis=0)
# Reduce along axis=0 to compute `right`.
right = np.mean(local_gradients,axis=0)
np.testing.assert_allclose(wrong,4*expected_gradient,rtol=1e-6)
np.testing.assert_allclose(right,expected_gradient,rtol=1e-6)
# Unequal partitions: weight local mean gradients by their observation counts.
parts = [(host_x[:3],host_y[:3]),(host_x[3:],host_y[3:])]
weighted = sum(len(b)*(2*a.T @ (-b)/len(b)) for a,b in parts)/len(host_y)
np.testing.assert_allclose(weighted,expected_gradient,rtol=1e-6)
print("Summed local means are four times too large; weighted aggregation repaired")

# Reference practice: Change the mesh-sized regression fixture
# Transfer the fixture to twelve padded rows with seven valid examples.
x12_host = (np.arange(48, dtype=np.float32).reshape(12, 4) - 11) / 12
truth12_host = np.full((4, 4), 0.15, dtype=np.float32)
target12_host = x12_host @ truth12_host
valid12_host = np.r_[np.ones(7, np.float32), np.zeros(5, np.float32)]
x12_2d = jax.device_put(x12_host, x_shard)
target12_2d = jax.device_put(target12_host, target_shard)
valid12_2d = jax.device_put(valid12_host, mask_shard)
loss12_2d, gradient12_2d = value_and_grad_2d(
    weight_2d, x12_2d, target12_2d, valid12_2d
)
next_weight12_2d, next_momentum12_2d = momentum_update_2d(
    weight_2d, momentum_2d, gradient12_2d
)
residual12_host = x12_host @ weight_host - target12_host
oracle_loss12 = np.sum(residual12_host[:7] ** 2) / (7 * 4)
oracle_gradient12 = 2 * x12_host[:7].T @ residual12_host[:7] / (7 * 4)
oracle_momentum12 = momentum_rate_2d * momentum_host + oracle_gradient12
oracle_weight12 = weight_host - learning_rate_2d * oracle_momentum12
np.testing.assert_allclose(np.asarray(loss12_2d), oracle_loss12, rtol=2e-5, atol=2e-6)
np.testing.assert_allclose(np.asarray(gradient12_2d), oracle_gradient12, rtol=2e-5, atol=2e-6)
np.testing.assert_allclose(np.asarray(next_weight12_2d), oracle_weight12, rtol=2e-5, atol=2e-6)
np.testing.assert_allclose(np.asarray(next_momentum12_2d), oracle_momentum12, rtol=2e-5, atol=2e-6)
counts12 = [int(np.asarray(s.data).sum()) for s in valid12_2d.addressable_shards]
assert sorted(counts12) == [1, 1, 6, 6]
assert all(s.data.shape == (4, 2) and s.data.nbytes == 32 for s in next_weight12_2d.addressable_shards)
print(f"12-row transfer: valid counts={counts12}; loss={float(loss12_2d):.6f}; parameter shards=(4,2), 32 bytes/device")
print("PASS: distributed-02")
