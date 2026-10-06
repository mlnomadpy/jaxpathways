"""Resilient distributed training: worked experiments and reference solutions. CPU checks."""

# Initialize the runtime before device access
import os
import tempfile
from pathlib import Path
import jax
multi = os.environ.get('COURSE_MULTIPROCESS') == '1'
resume = os.environ.get('COURSE_RESUME') == '1'
if multi:
    # Every controller runs this before querying devices or creating arrays.
    jax.distributed.initialize()
else:
    jax.config.update('jax_platforms', 'cpu')
    jax.config.update('jax_num_cpu_devices', 4)
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Define the global fixture and placements
count = jax.device_count()
assert count >= 2, 'Use at least two devices for the placement exercise'
mesh = Mesh(np.asarray(jax.devices()), ('data',))
rep = NamedSharding(mesh, P())
row = NamedSharding(mesh, P('data', None))
label = NamedSharding(mesh, P('data'))
batch_size, size, width = 2 * count, 4 * count, 2 * count
# Identical tiny synthetic fixture on each controller; not distributed data ingestion.
data = np.random.default_rng(41).normal(size=(size, width)).astype(np.float32)
truth = np.linspace(-0.3, 0.4, width, dtype=np.float32)
targets = data @ truth

# Carry complete transition state
def placed(value, sharding=rep):
    a = np.asarray(value)
    return jax.make_array_from_callback(a.shape, sharding, lambda index: a[index])

# Checkpoint and verify replay
def initial():
    key, order_key = jax.random.split(jax.random.PRNGKey(17))
    return { 'weights': placed(np.zeros(width, np.float32)),
             'momentum': placed(np.zeros(width, np.float32)),
             'key': placed(np.asarray(key)),
             'order': placed(np.asarray(jax.random.permutation(order_key, size))),
             'cursor': placed(np.int32(0)), 'step': placed(np.int32(0)) }

# Checkpoint and verify replay
@jax.jit
def update(w, velocity, x, y):
    loss, grad = jax.value_and_grad(lambda a: jnp.mean((x @ a - y)**2))(w)
    v = 0.8 * velocity + grad
    return w - 0.04 * v, v, loss

# Checkpoint and verify replay
def transition(state):
    cursor = int(state['cursor'])
    order = np.asarray(state['order'])
    key = state['key']
    if cursor == size:
        key, shuffle_key = jax.random.split(key)
        order = np.asarray(jax.random.permutation(shuffle_key, size))
        cursor = 0
    ids = order[cursor:cursor + batch_size]
    x, y = placed(data[ids], row), placed(targets[ids], label)
    w, v, loss = update(state['weights'], state['momentum'], x, y)
    loss.block_until_ready()
    # Independent global derivative and momentum update on the host.
    old_w, old_v = np.asarray(state['weights']), np.asarray(state['momentum'])
    g = 2 * data[ids].T @ (data[ids] @ old_w - targets[ids]) / batch_size
    np.testing.assert_allclose(np.asarray(v), 0.8 * old_v + g, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(w), old_w - 0.04 * np.asarray(v), rtol=2e-5, atol=2e-6)
    new = dict(weights=w, momentum=v, key=placed(np.asarray(key)),
               order=placed(order), cursor=placed(np.int32(cursor+batch_size)),
               step=placed(np.int32(int(state['step'])+1)))
    return new, ids.copy(), float(loss)

# Checkpoint and verify replay
state = initial()
state, first_ids, _ = transition(state)
# Shared path is mandatory for real multi-controller execution.
if multi:
    root = Path(os.environ['COURSE_CHECKPOINT_DIR']).resolve()
else:
    root = Path(os.environ['COURSE_CHECKPOINT_DIR']).resolve() if os.environ.get('COURSE_CHECKPOINT_DIR') else Path(tempfile.mkdtemp(prefix='jax-distributed-recovery-'))
checkpoint_path = root / 'step-one'
if resume and not checkpoint_path.exists():
    raise FileNotFoundError('Resume needs an existing step-one checkpoint')
if not resume and checkpoint_path.exists():
    raise FileExistsError('Use a new folder or set COURSE_RESUME=1 for the accepted checkpoint')
with ocp.StandardCheckpointer() as checkpointer:
    if not resume:
        checkpointer.save(checkpoint_path, state)
        checkpointer.wait_until_finished()
    # Target carries the current mesh/sharding, rather than inferring it from old metadata.
    restored = checkpointer.restore(checkpoint_path, target=initial())
    for name in state:
        np.testing.assert_array_equal(np.asarray(restored[name]), np.asarray(state[name]))
    baseline, resumed = state, restored
    baseline_losses, resumed_losses, replay_ids = [], [], []
    for _ in range(3):
        baseline, ids_a, loss_a = transition(baseline)
        resumed, ids_b, loss_b = transition(resumed)
        np.testing.assert_array_equal(ids_a, ids_b)
        for name in baseline:
            np.testing.assert_allclose(np.asarray(baseline[name]), np.asarray(resumed[name]), rtol=2e-5, atol=2e-6)
        baseline_losses.append(loss_a)
        resumed_losses.append(loss_b)
        replay_ids.append(ids_a.tolist())
print('Recorded backend/processes/devices:', jax.default_backend(), jax.process_count(), count)
print('Restored step/cursor:', int(restored['step']), int(restored['cursor']))
print('Next sample IDs across epoch boundary:', replay_ids)
print('Uninterrupted losses:', baseline_losses)
print('Restored losses:', resumed_losses)
print('Checkpoint:', checkpoint_path)


import os
import tempfile
from pathlib import Path
import jax
multi = os.environ.get('COURSE_MULTIPROCESS') == '1'
resume = os.environ.get('COURSE_RESUME') == '1'
if multi:
    # Every controller runs this before querying devices or creating arrays.
    jax.distributed.initialize()
else:
    jax.config.update('jax_platforms', 'cpu')
    jax.config.update('jax_num_cpu_devices', 4)
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

count = jax.device_count()
assert count >= 2, 'Use at least two devices for the placement exercise'
mesh = Mesh(np.asarray(jax.devices()), ('data',))
rep = NamedSharding(mesh, P())
row = NamedSharding(mesh, P('data', None))
label = NamedSharding(mesh, P('data'))
batch_size, size, width = 2 * count, 4 * count, 2 * count
# Identical tiny synthetic fixture on each controller; not distributed data ingestion.
data = np.random.default_rng(41).normal(size=(size, width)).astype(np.float32)
truth = np.linspace(-0.3, 0.4, width, dtype=np.float32)
targets = data @ truth

def placed(value, sharding=rep):
    a = np.asarray(value)
    return jax.make_array_from_callback(a.shape, sharding, lambda index: a[index])

def initial():
    key, order_key = jax.random.split(jax.random.PRNGKey(17))
    return { 'weights': placed(np.zeros(width, np.float32)),
             'momentum': placed(np.zeros(width, np.float32)),
             'key': placed(np.asarray(key)),
             'order': placed(np.asarray(jax.random.permutation(order_key, size))),
             'cursor': placed(np.int32(0)), 'step': placed(np.int32(0)) }

@jax.jit
def update(w, velocity, x, y):
    loss, grad = jax.value_and_grad(lambda a: jnp.mean((x @ a - y)**2))(w)
    v = 0.8 * velocity + grad
    return w - 0.04 * v, v, loss

def transition(state):
    cursor = int(state['cursor'])
    order = np.asarray(state['order'])
    key = state['key']
    if cursor == size:
        key, shuffle_key = jax.random.split(key)
        order = np.asarray(jax.random.permutation(shuffle_key, size))
        cursor = 0
    ids = order[cursor:cursor + batch_size]
    x, y = placed(data[ids], row), placed(targets[ids], label)
    w, v, loss = update(state['weights'], state['momentum'], x, y)
    loss.block_until_ready()
    # Independent global derivative and momentum update on the host.
    old_w, old_v = np.asarray(state['weights']), np.asarray(state['momentum'])
    g = 2 * data[ids].T @ (data[ids] @ old_w - targets[ids]) / batch_size
    np.testing.assert_allclose(np.asarray(v), 0.8 * old_v + g, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(w), old_w - 0.04 * np.asarray(v), rtol=2e-5, atol=2e-6)
    new = dict(weights=w, momentum=v, key=placed(np.asarray(key)),
               order=placed(order), cursor=placed(np.int32(cursor+batch_size)),
               step=placed(np.int32(int(state['step'])+1)))
    return new, ids.copy(), float(loss)

state = initial()
state, first_ids, _ = transition(state)
# Shared path is mandatory for real multi-controller execution.
if multi:
    root = Path(os.environ['COURSE_CHECKPOINT_DIR']).resolve()
else:
    root = Path(os.environ['COURSE_CHECKPOINT_DIR']).resolve() if os.environ.get('COURSE_CHECKPOINT_DIR') else Path(tempfile.mkdtemp(prefix='jax-distributed-recovery-'))
checkpoint_path = root / 'step-one'
if resume and not checkpoint_path.exists():
    raise FileNotFoundError('Resume needs an existing step-one checkpoint')
if not resume and checkpoint_path.exists():
    raise FileExistsError('Use a new folder or set COURSE_RESUME=1 for the accepted checkpoint')
with ocp.StandardCheckpointer() as checkpointer:
    if not resume:
        checkpointer.save(checkpoint_path, state)
        checkpointer.wait_until_finished()
    # Target carries the current mesh/sharding, rather than inferring it from old metadata.
    restored = checkpointer.restore(checkpoint_path, target=initial())
    for name in state:
        np.testing.assert_array_equal(np.asarray(restored[name]), np.asarray(state[name]))
    baseline, resumed = state, restored
    baseline_losses, resumed_losses, replay_ids = [], [], []
    for _ in range(3):
        baseline, ids_a, loss_a = transition(baseline)
        resumed, ids_b, loss_b = transition(resumed)
        np.testing.assert_array_equal(ids_a, ids_b)
        for name in baseline:
            np.testing.assert_allclose(np.asarray(baseline[name]), np.asarray(resumed[name]), rtol=2e-5, atol=2e-6)
        baseline_losses.append(loss_a)
        resumed_losses.append(loss_b)
        replay_ids.append(ids_a.tolist())
print('Recorded backend/processes/devices:', jax.default_backend(), jax.process_count(), count)
print('Restored step/cursor:', int(restored['step']), int(restored['cursor']))
print('Next sample IDs across epoch boundary:', replay_ids)
print('Uninterrupted losses:', baseline_losses)
print('Restored losses:', resumed_losses)
print('Checkpoint:', checkpoint_path)


# Figure data experiment
visual_data={'kind':'line','x':[1,2,3],'xlabel':'update after checkpoint','ylabel':'batch mean squared loss','series':[{'label':'uninterrupted','y':baseline_losses},{'label':'restored','y':resumed_losses}]}

# Experiment: Keep weights but discard momentum
incomplete = {**restored, 'momentum': placed(np.zeros(width, np.float32))}
wrong_next, wrong_ids, _ = transition(incomplete)
right_next, right_ids, _ = transition(restored)
np.testing.assert_array_equal(wrong_ids, right_ids)
weight_gap = float(np.max(np.abs(np.asarray(wrong_next['weights']) - np.asarray(right_next['weights']))))
assert weight_gap > 1e-5
print('Same next sample IDs, different weights after missing momentum:', weight_gap)

# Reference solution. Try the exercise before reading this.
for _ in range(2):
    baseline, ids_a, _ = transition(baseline)
    resumed, ids_b, _ = transition(resumed)
    np.testing.assert_array_equal(ids_a, ids_b)
    for name in baseline:
        np.testing.assert_allclose(np.asarray(baseline[name]), np.asarray(resumed[name]), rtol=2e-5, atol=2e-6)
    for shard in resumed['weights'].addressable_shards:
        np.testing.assert_allclose(np.asarray(shard.data), np.asarray(baseline['weights']), rtol=2e-5, atol=2e-6)
assert int(resumed['step']) == 6
print('Five resumed transitions and all addressable parameter replicas agree.')

# Reference practice: Find a delayed random-state failure
wrong_key_state = {**restored, 'key': placed(np.asarray(jax.random.PRNGKey(999)))}
correct_a, ids_a, _ = transition(restored)
wrong_a, ids_b, _ = transition(wrong_key_state)
np.testing.assert_array_equal(ids_a, ids_b)
np.testing.assert_allclose(np.asarray(correct_a['weights']), np.asarray(wrong_a['weights']), rtol=2e-5, atol=2e-6)
correct_b, ids_c, _ = transition(correct_a)
wrong_b, ids_d, _ = transition(wrong_a)
assert not np.array_equal(ids_c, ids_d)
print('Current-epoch batch matches; next-epoch batch exposes the wrong key.')
print("PASS: distributed-04")
