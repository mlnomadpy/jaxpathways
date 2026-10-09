"""Resilient distributed training: worked experiments and reference solutions. CPU checks."""

# Initialize the runtime before device access
# Step 1 — Initialize the runtime before device access: Every controller must use consistent state, data and collective...
# Import os for this computation.
import os
import tempfile
from pathlib import Path
import jax
# Read `multi` from environment configuration (with a default fallback).
multi = os.environ.get('COURSE_MULTIPROCESS') == '1'
# Read `resume` from environment configuration (with a default fallback).
resume = os.environ.get('COURSE_RESUME') == '1'
# Branch on condition `multi`:
if multi:
    # Every controller runs this before querying devices or creating arrays.
    jax.distributed.initialize()
else:
    jax.config.update('jax_platforms', 'cpu')
    jax.config.update('jax_num_cpu_devices', 4)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Define the global fixture and placements
# Step 2 — Define the global fixture and placements: Every controller must use consistent state, data and collective...
count = jax.device_count()
# Verify contract: `count >= 2`.
assert count >= 2, 'Use at least two devices for the placement exercise'
# Configure multi-device placement / sharding specification (`mesh`).
mesh = Mesh(np.asarray(jax.devices()), ('data',))
# Configure multi-device placement / sharding specification (`rep`).
rep = NamedSharding(mesh, P())
# Configure multi-device placement / sharding specification (`row`).
row = NamedSharding(mesh, P('data', None))
# Configure multi-device placement / sharding specification (`label`).
label = NamedSharding(mesh, P('data'))
# Evaluate `(batch_size, size, width)` from the current inputs and state.
batch_size, size, width = 2 * count, 4 * count, 2 * count
# Identical tiny synthetic fixture on each controller; not distributed data ingestion.
data = np.random.default_rng(41).normal(size=(size, width)).astype(np.float32)
# Initialize array `truth` with explicit values and shape.
truth = np.linspace(-0.3, 0.4, width, dtype=np.float32)
# Perform matrix / vector contraction (`@`) to compute `targets`.
targets = data @ truth

# Carry complete transition state
# Step 3 — Carry complete transition state: Every controller must use consistent state, data and collective...
def placed(value, sharding=rep):
    # Convert `a` to a host NumPy array for inspection or verification.
    a = np.asarray(value)
    # Return `jax.make_array_from_callback(a.shape, sharding, lambda index: a[index])` to the caller.
    return jax.make_array_from_callback(a.shape, sharding, lambda index: a[index])

# Checkpoint and verify replay
# Step 4 — Checkpoint and verify replay: Every controller must use consistent state, data and collective...
def initial():
    # Create or split explicit PRNG key(s) (`(key, order_key)`) for reproducible randomness.
    key, order_key = jax.random.split(jax.random.PRNGKey(17))
    # Return `{'weights': placed(np.zeros(width, np.float32)), 'momentum': placed(np.zeros(width, np.float32)), 'key': placed(np.asarray(key)), 'order': placed(np.asarray(jax.random.permutation(order_key, size))), 'cursor': placed(np.int32(0)), 'step': placed(np.int32(0))}` to the caller.
    return { 'weights': placed(np.zeros(width, np.float32)),
             'momentum': placed(np.zeros(width, np.float32)),
             'key': placed(np.asarray(key)),
             'order': placed(np.asarray(jax.random.permutation(order_key, size))),
             'cursor': placed(np.int32(0)), 'step': placed(np.int32(0)) }

# Checkpoint and verify replay
# Step 5 — Checkpoint and verify replay: Every controller must use consistent state, data and collective...
# Define and JIT-compile `update(w, velocity, x, y)` so XLA traces and fuses the operations:
@jax.jit
# Function `update(w, velocity, x, y)` implementing this stage's computation:
def update(w, velocity, x, y):
    # Differentiate the objective to obtain `(loss, grad)` via automatic differentiation.
    loss, grad = jax.value_and_grad(lambda a: jnp.mean((x @ a - y)**2))(w)
    # Evaluate `v` from the current inputs and state.
    v = 0.8 * velocity + grad
    # Return `(w - 0.04 * v, v, loss)` to the caller.
    return w - 0.04 * v, v, loss

# Checkpoint and verify replay
# Step 6 — Checkpoint and verify replay: Every controller must use consistent state, data and collective...
def transition(state):
    # Evaluate `state['cursor']` and convert the result into Python scalar/collection `cursor`.
    cursor = int(state['cursor'])
    # Convert `order` to a host NumPy array for inspection or verification.
    order = np.asarray(state['order'])
    # Evaluate `key` from the current inputs and state.
    key = state['key']
    # Branch on condition `cursor == size`:
    if cursor == size:
        key, shuffle_key = jax.random.split(key)
        order = np.asarray(jax.random.permutation(shuffle_key, size))
        cursor = 0
    # Evaluate `ids` from the current inputs and state.
    ids = order[cursor:cursor + batch_size]
    # Evaluate `(x, y)` from the current inputs and state.
    x, y = placed(data[ids], row), placed(targets[ids], label)
    # Run `update` to compute `(w, v, loss)`.
    w, v, loss = update(state['weights'], state['momentum'], x, y)
    # Synchronize host execution until asynchronous device computation completes.
    loss.block_until_ready()
    # Independent global derivative and momentum update on the host.
    old_w, old_v = np.asarray(state['weights']), np.asarray(state['momentum'])
    # Perform matrix contraction / projection to compute `g`.
    g = 2 * data[ids].T @ (data[ids] @ old_w - targets[ids]) / batch_size
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(v), 0.8 * old_v + g, rtol=2e-5, atol=2e-6)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(w), old_w - 0.04 * np.asarray(v), rtol=2e-5, atol=2e-6)
    # Convert `new` to a host NumPy array for inspection or verification.
    new = dict(weights=w, momentum=v, key=placed(np.asarray(key)),
               order=placed(order), cursor=placed(np.int32(cursor+batch_size)),
               step=placed(np.int32(int(state['step'])+1)))
    # Return `(new, ids.copy(), float(loss))` to the caller.
    return new, ids.copy(), float(loss)

# Checkpoint and verify replay
# Step 7 — Checkpoint and verify replay: Every controller must use consistent state, data and collective...
state = initial()
# Run `transition` to compute `(state, first_ids, _)`.
state, first_ids, _ = transition(state)
# Shared path is mandatory for real multi-controller execution.
if multi:
    root = Path(os.environ['COURSE_CHECKPOINT_DIR']).resolve()
else:
    root = Path(os.environ['COURSE_CHECKPOINT_DIR']).resolve() if os.environ.get('COURSE_CHECKPOINT_DIR') else Path(tempfile.mkdtemp(prefix='jax-distributed-recovery-'))
# Evaluate `checkpoint_path` from the current inputs and state.
checkpoint_path = root / 'step-one'
# Guard input contract (`resume and (not checkpoint_path.exists())`) and fail fast if violated.
if resume and not checkpoint_path.exists():
    raise FileNotFoundError('Resume needs an existing step-one checkpoint')
# Guard input contract (`not resume and checkpoint_path.exists()`) and fail fast if violated.
if not resume and checkpoint_path.exists():
    raise FileExistsError('Use a new folder or set COURSE_RESUME=1 for the accepted checkpoint')
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as checkpointer:
    # Branch on condition `not resume`:
    if not resume:
        checkpointer.save(checkpoint_path, state)
        checkpointer.wait_until_finished()
    # Target carries the current mesh/sharding, rather than inferring it from old metadata.
    restored = checkpointer.restore(checkpoint_path, target=initial())
    # Iterate over `name` to step through the computation:
    for name in state:
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_array_equal(np.asarray(restored[name]), np.asarray(state[name]))
    # Evaluate `(baseline, resumed)` from the current inputs and state.
    baseline, resumed = state, restored
    # Evaluate `(baseline_losses, resumed_losses, replay_ids)` from the current inputs and state.
    baseline_losses, resumed_losses, replay_ids = [], [], []
    # Repeat the update loop over `range(3)` steps:
    for _ in range(3):
        # Run `transition` to compute `(baseline, ids_a, loss_a)`.
        baseline, ids_a, loss_a = transition(baseline)
        # Run `transition` to compute `(resumed, ids_b, loss_b)`.
        resumed, ids_b, loss_b = transition(resumed)
        # Verify that computed values match the expected reference within numerical tolerance.
        np.testing.assert_array_equal(ids_a, ids_b)
        # Loop over `name` in `baseline`:
        for name in baseline:
            # Convert `` to a host NumPy array for inspection or verification.
            np.testing.assert_allclose(np.asarray(baseline[name]), np.asarray(resumed[name]), rtol=2e-5, atol=2e-6)
        # Append the current step result to `baseline_losses`.
        baseline_losses.append(loss_a)
        # Append the current step result to `resumed_losses`.
        resumed_losses.append(loss_b)
        # Append the current step result to `replay_ids`.
        replay_ids.append(ids_a.tolist())
# Print the observed values to compare against the expected result.
print('Recorded backend/processes/devices:', jax.default_backend(), jax.process_count(), count)
# Print diagnostic summary of the computed outputs.
print('Restored step/cursor:', int(restored['step']), int(restored['cursor']))
# Print diagnostic summary of the computed outputs.
print('Next sample IDs across epoch boundary:', replay_ids)
# Print diagnostic summary of the computed outputs.
print('Uninterrupted losses:', baseline_losses)
# Print diagnostic summary of the computed outputs.
print('Restored losses:', resumed_losses)
# Print diagnostic summary of the computed outputs.
print('Checkpoint:', checkpoint_path)

# Step 1 — Initialize the runtime before device access: Every controller must use consistent state, data and collective...
# Import os for this computation.
import os
import tempfile
from pathlib import Path
import jax
# Read `multi` from environment configuration (with a default fallback).
multi = os.environ.get('COURSE_MULTIPROCESS') == '1'
# Read `resume` from environment configuration (with a default fallback).
resume = os.environ.get('COURSE_RESUME') == '1'
# Branch on condition `multi`:
if multi:
    # Every controller runs this before querying devices or creating arrays.
    jax.distributed.initialize()
else:
    jax.config.update('jax_platforms', 'cpu')
    jax.config.update('jax_num_cpu_devices', 4)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Step 2 — Define the global fixture and placements: Every controller must use consistent state, data and collective...
count = jax.device_count()
# Verify contract: `count >= 2`.
assert count >= 2, 'Use at least two devices for the placement exercise'
# Configure multi-device placement / sharding specification (`mesh`).
mesh = Mesh(np.asarray(jax.devices()), ('data',))
# Configure multi-device placement / sharding specification (`rep`).
rep = NamedSharding(mesh, P())
# Configure multi-device placement / sharding specification (`row`).
row = NamedSharding(mesh, P('data', None))
# Configure multi-device placement / sharding specification (`label`).
label = NamedSharding(mesh, P('data'))
# Evaluate `(batch_size, size, width)` from the current inputs and state.
batch_size, size, width = 2 * count, 4 * count, 2 * count
# Identical tiny synthetic fixture on each controller; not distributed data ingestion.
data = np.random.default_rng(41).normal(size=(size, width)).astype(np.float32)
# Initialize array `truth` with explicit values and shape.
truth = np.linspace(-0.3, 0.4, width, dtype=np.float32)
# Perform matrix / vector contraction (`@`) to compute `targets`.
targets = data @ truth

# Step 3 — Carry complete transition state: Every controller must use consistent state, data and collective...
def placed(value, sharding=rep):
    # Convert `a` to a host NumPy array for inspection or verification.
    a = np.asarray(value)
    # Return `jax.make_array_from_callback(a.shape, sharding, lambda index: a[index])` to the caller.
    return jax.make_array_from_callback(a.shape, sharding, lambda index: a[index])

# Step 4 — Checkpoint and verify replay: Every controller must use consistent state, data and collective...
def initial():
    # Create or split explicit PRNG key(s) (`(key, order_key)`) for reproducible randomness.
    key, order_key = jax.random.split(jax.random.PRNGKey(17))
    # Return `{'weights': placed(np.zeros(width, np.float32)), 'momentum': placed(np.zeros(width, np.float32)), 'key': placed(np.asarray(key)), 'order': placed(np.asarray(jax.random.permutation(order_key, size))), 'cursor': placed(np.int32(0)), 'step': placed(np.int32(0))}` to the caller.
    return { 'weights': placed(np.zeros(width, np.float32)),
             'momentum': placed(np.zeros(width, np.float32)),
             'key': placed(np.asarray(key)),
             'order': placed(np.asarray(jax.random.permutation(order_key, size))),
             'cursor': placed(np.int32(0)), 'step': placed(np.int32(0)) }

# Step 5 — Checkpoint and verify replay: Every controller must use consistent state, data and collective...
# Define and JIT-compile `update(w, velocity, x, y)` so XLA traces and fuses the operations:
@jax.jit
# Function `update(w, velocity, x, y)` implementing this stage's computation:
def update(w, velocity, x, y):
    # Differentiate the objective to obtain `(loss, grad)` via automatic differentiation.
    loss, grad = jax.value_and_grad(lambda a: jnp.mean((x @ a - y)**2))(w)
    # Evaluate `v` from the current inputs and state.
    v = 0.8 * velocity + grad
    # Return `(w - 0.04 * v, v, loss)` to the caller.
    return w - 0.04 * v, v, loss

# Step 6 — Checkpoint and verify replay: Every controller must use consistent state, data and collective...
def transition(state):
    # Evaluate `state['cursor']` and convert the result into Python scalar/collection `cursor`.
    cursor = int(state['cursor'])
    # Convert `order` to a host NumPy array for inspection or verification.
    order = np.asarray(state['order'])
    # Evaluate `key` from the current inputs and state.
    key = state['key']
    # Branch on condition `cursor == size`:
    if cursor == size:
        key, shuffle_key = jax.random.split(key)
        order = np.asarray(jax.random.permutation(shuffle_key, size))
        cursor = 0
    # Evaluate `ids` from the current inputs and state.
    ids = order[cursor:cursor + batch_size]
    # Evaluate `(x, y)` from the current inputs and state.
    x, y = placed(data[ids], row), placed(targets[ids], label)
    # Run `update` to compute `(w, v, loss)`.
    w, v, loss = update(state['weights'], state['momentum'], x, y)
    # Synchronize host execution until asynchronous device computation completes.
    loss.block_until_ready()
    # Independent global derivative and momentum update on the host.
    old_w, old_v = np.asarray(state['weights']), np.asarray(state['momentum'])
    # Perform matrix contraction / projection to compute `g`.
    g = 2 * data[ids].T @ (data[ids] @ old_w - targets[ids]) / batch_size
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(v), 0.8 * old_v + g, rtol=2e-5, atol=2e-6)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.asarray(w), old_w - 0.04 * np.asarray(v), rtol=2e-5, atol=2e-6)
    # Convert `new` to a host NumPy array for inspection or verification.
    new = dict(weights=w, momentum=v, key=placed(np.asarray(key)),
               order=placed(order), cursor=placed(np.int32(cursor+batch_size)),
               step=placed(np.int32(int(state['step'])+1)))
    # Return `(new, ids.copy(), float(loss))` to the caller.
    return new, ids.copy(), float(loss)

# Step 7 — Checkpoint and verify replay: Every controller must use consistent state, data and collective...
state = initial()
# Run `transition` to compute `(state, first_ids, _)`.
state, first_ids, _ = transition(state)
# Shared path is mandatory for real multi-controller execution.
if multi:
    root = Path(os.environ['COURSE_CHECKPOINT_DIR']).resolve()
else:
    root = Path(os.environ['COURSE_CHECKPOINT_DIR']).resolve() if os.environ.get('COURSE_CHECKPOINT_DIR') else Path(tempfile.mkdtemp(prefix='jax-distributed-recovery-'))
# Evaluate `checkpoint_path` from the current inputs and state.
checkpoint_path = root / 'step-one'
# Guard input contract (`resume and (not checkpoint_path.exists())`) and fail fast if violated.
if resume and not checkpoint_path.exists():
    raise FileNotFoundError('Resume needs an existing step-one checkpoint')
# Guard input contract (`not resume and checkpoint_path.exists()`) and fail fast if violated.
if not resume and checkpoint_path.exists():
    raise FileExistsError('Use a new folder or set COURSE_RESUME=1 for the accepted checkpoint')
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as checkpointer:
    # Branch on condition `not resume`:
    if not resume:
        checkpointer.save(checkpoint_path, state)
        checkpointer.wait_until_finished()
    # Target carries the current mesh/sharding, rather than inferring it from old metadata.
    restored = checkpointer.restore(checkpoint_path, target=initial())
    # Iterate over `name` to step through the computation:
    for name in state:
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_array_equal(np.asarray(restored[name]), np.asarray(state[name]))
    # Evaluate `(baseline, resumed)` from the current inputs and state.
    baseline, resumed = state, restored
    # Evaluate `(baseline_losses, resumed_losses, replay_ids)` from the current inputs and state.
    baseline_losses, resumed_losses, replay_ids = [], [], []
    # Repeat the update loop over `range(3)` steps:
    for _ in range(3):
        # Run `transition` to compute `(baseline, ids_a, loss_a)`.
        baseline, ids_a, loss_a = transition(baseline)
        # Run `transition` to compute `(resumed, ids_b, loss_b)`.
        resumed, ids_b, loss_b = transition(resumed)
        # Verify that computed values match the expected reference within numerical tolerance.
        np.testing.assert_array_equal(ids_a, ids_b)
        # Loop over `name` in `baseline`:
        for name in baseline:
            # Convert `` to a host NumPy array for inspection or verification.
            np.testing.assert_allclose(np.asarray(baseline[name]), np.asarray(resumed[name]), rtol=2e-5, atol=2e-6)
        # Append the current step result to `baseline_losses`.
        baseline_losses.append(loss_a)
        # Append the current step result to `resumed_losses`.
        resumed_losses.append(loss_b)
        # Append the current step result to `replay_ids`.
        replay_ids.append(ids_a.tolist())
# Print the observed values to compare against the expected result.
print('Recorded backend/processes/devices:', jax.default_backend(), jax.process_count(), count)
# Print diagnostic summary of the computed outputs.
print('Restored step/cursor:', int(restored['step']), int(restored['cursor']))
# Print diagnostic summary of the computed outputs.
print('Next sample IDs across epoch boundary:', replay_ids)
# Print diagnostic summary of the computed outputs.
print('Uninterrupted losses:', baseline_losses)
# Print diagnostic summary of the computed outputs.
print('Restored losses:', resumed_losses)
# Print diagnostic summary of the computed outputs.
print('Checkpoint:', checkpoint_path)

# Figure data experiment
# Compute figure data for: Restored updates reproduce the reference trajectory
# Evaluate `visual_data` from the current inputs and state.
visual_data={'kind':'line','x':[1,2,3],'xlabel':'update after checkpoint','ylabel':'batch mean squared loss','series':[{'label':'uninterrupted','y':baseline_losses},{'label':'restored','y':resumed_losses}]}

# Experiment: Keep weights but discard momentum
# Experiment — Keep weights but discard momentum: The data stream is correct in this injected failure.
incomplete = {**restored, 'momentum': placed(np.zeros(width, np.float32))}
# Run `transition` to compute `(wrong_next, wrong_ids, _)`.
wrong_next, wrong_ids, _ = transition(incomplete)
# Run `transition` to compute `(right_next, right_ids, _)`.
right_next, right_ids, _ = transition(restored)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_array_equal(wrong_ids, right_ids)
# Aggregate array values to compute `weight_gap`.
weight_gap = float(np.max(np.abs(np.asarray(wrong_next['weights']) - np.asarray(right_next['weights']))))
# Verify contract: `weight_gap > 1e-05`.
assert weight_gap > 1e-5
# Print the observed values to compare against the expected result.
print('Same next sample IDs, different weights after missing momentum:', weight_gap)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Extend uninterrupted and restored execution by two additional steps.
for _ in range(2):
    # Run `transition` to compute `(baseline, ids_a, _)`.
    baseline, ids_a, _ = transition(baseline)
    # Run `transition` to compute `(resumed, ids_b, _)`.
    resumed, ids_b, _ = transition(resumed)
    # Verify that computed values match the expected reference within numerical tolerance.
    np.testing.assert_array_equal(ids_a, ids_b)
    # Loop over `name` in `baseline`:
    for name in baseline:
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_allclose(np.asarray(baseline[name]), np.asarray(resumed[name]), rtol=2e-5, atol=2e-6)
    # Loop over `shard` in `resumed['weights'].addressable_shards`:
    for shard in resumed['weights'].addressable_shards:
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_allclose(np.asarray(shard.data), np.asarray(baseline['weights']), rtol=2e-5, atol=2e-6)
# Verify contract: `int(resumed['step']) == 6`.
assert int(resumed['step']) == 6
# Print the observed values to compare against the expected result.
print('Five resumed transitions and all addressable parameter replicas agree.')

# Reference practice: Find a delayed random-state failure
# Find a delayed random-state failure (Transfer): A successful first resumed update does not establish future...
# Create or split explicit PRNG key(s) (`wrong_key_state`) for reproducible randomness.
wrong_key_state = {**restored, 'key': placed(np.asarray(jax.random.PRNGKey(999)))}
# Run `transition` to compute `(correct_a, ids_a, _)`.
correct_a, ids_a, _ = transition(restored)
# Run `transition` to compute `(wrong_a, ids_b, _)`.
wrong_a, ids_b, _ = transition(wrong_key_state)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_array_equal(ids_a, ids_b)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(correct_a['weights']), np.asarray(wrong_a['weights']), rtol=2e-5, atol=2e-6)
# Run `transition` to compute `(correct_b, ids_c, _)`.
correct_b, ids_c, _ = transition(correct_a)
# Run `transition` to compute `(wrong_b, ids_d, _)`.
wrong_b, ids_d, _ = transition(wrong_a)
# Verify contract: `not np.array_equal(ids_c, ids_d)`.
assert not np.array_equal(ids_c, ids_d)
# Print the observed values to compare against the expected result.
print('Current-epoch batch matches; next-epoch batch exposes the wrong key.')
print("PASS: distributed-04")
