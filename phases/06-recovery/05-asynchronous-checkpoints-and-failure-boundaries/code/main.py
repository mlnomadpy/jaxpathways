"""Asynchronous checkpoints and failure boundaries: worked experiments and reference solutions. CPU checks."""

# Define functional state and publication
# Step 1 — Define functional state and publication: The manifest has its own acceptance boundary.
# Import json for this computation.
import json
import tempfile
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
# Read or serialize artifact data on disk (`root`).
root=Path(tempfile.mkdtemp(prefix='jax-async-lesson-'))
# Function `transition(state)` implementing this stage's computation:
def transition(state):
    # Return `{'weight': state['weight'] + 0.25, 'step': state['step'] + 1}` to the caller.
    return {'weight':state['weight']+.25,'step':state['step']+1}
# Function `publish(path, step)` implementing this stage's computation:
def publish(path, step):
    # Compute `pending` from `root/'LATEST.pending'`
    pending=root/'LATEST.pending'
    # Write the serialized artifact payload to disk.
    pending.write_text(json.dumps({'path':str(path),'step':step}))
    # Run `pending.replace` to perform the next check or state transition.
    pending.replace(root/'LATEST.json')
# Function `accepted()` implementing this stage's computation:
def accepted():
    # Return `json.loads((root / 'LATEST.json').read_text())` to the caller.
    return json.loads((root/'LATEST.json').read_text())

# Save, continue and verify
# Step 2 — Save, continue and verify: Wait on the real background writer, restore against the snapshot...
# Construct `state` via `{'weight':jnp.arange(1024,dtype=jnp.float32),'step':...`
state={'weight':jnp.arange(1024,dtype=jnp.float32),'step':jnp.array(0,jnp.int32)}
# Run `transition` to compute `snapshot`.
snapshot=transition(state)
# Compute `path` from `root/'step_1'`
path=root/'step_1'
# Record execution timing or profiler trace in `t0`.
t0=time.perf_counter()
# Enter `ocp.AsyncCheckpointer(ocp.StandardCheckpointH` context block:
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    # Run `cp.save` to perform the next check or state transition.
    cp.save(path,args=ocp.args.StandardSave(snapshot))
    # Record execution timing or profiler trace in `returned`.
    returned=time.perf_counter()
    # Assert invariant `not (root/'LATEST.json').exists()` holds
    assert not (root/'LATEST.json').exists()
    # Run `transition` to compute `live`.
    live=transition(snapshot)
    # Run `cp.wait_until_finished` to perform the next check or state transition.
    cp.wait_until_finished()
    # Record execution timing or profiler trace in `committed`.
    committed=time.perf_counter()
    # Run `cp.restore` to compute `restored`.
    restored=cp.restore(path,args=ocp.args.StandardRestore(snapshot))
    # Execute `np.testing.assert_array_equal(restored['weight'],snapshot['w`
    np.testing.assert_array_equal(restored['weight'],snapshot['weight'])
    # Assert invariant `int(restored['step'])==1 and int(live['step'])==2` holds
    assert int(restored['step'])==1 and int(live['step'])==2
    # Run `publish` to perform the next check or state transition.
    publish(path,1)
# Compute `report` from `{'save_call_seconds':returned-t0,'remaining_work_and...`
report={'save_call_seconds':returned-t0,'remaining_work_and_wait_seconds':committed-returned,
        'accepted_step':accepted()['step'],'live_step':int(live['step'])}
# Assert invariant `report['accepted_step']==1` holds
assert report['accepted_step']==1

# Interrupt publication deliberately
# A deliberately interrupted application publication leaves the previous commit selected.
unpublished=root/'step_2'
# Enter `ocp.AsyncCheckpointer(ocp.StandardCheckpointH` context block:
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    # Run `cp.save` to perform the next check or state transition.
    cp.save(unpublished,args=ocp.args.StandardSave(live))
    # Run `cp.wait_until_finished` to perform the next check or state transition.
    cp.wait_until_finished()
    # Run the boundary check and catch the expected exception:
    try:
        raise RuntimeError('injected interruption before updating LATEST')
    except RuntimeError:
        pass
# Assert invariant `accepted()['step']==1` holds
assert accepted()['step']==1
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as cp:
    # Run `cp.restore` to compute `previous`.
    previous=cp.restore(accepted()['path'],target=snapshot)
# Execute `np.testing.assert_array_equal(previous['weight'],snapshot['w`
np.testing.assert_array_equal(previous['weight'],snapshot['weight'])
# Assert invariant `int(previous['step'])==1` holds
assert int(previous['step'])==1
# Print the observed values to compare against the expected result.
print('Checkpoint workspace:',root)
# Print diagnostic summary of the computed outputs.
print(json.dumps(report,indent=2))
# Print diagnostic summary of the computed outputs.
print('Unpublished step 2 exists; application recovery still selects verified step 1.')

# Step 1 — Define functional state and publication: The manifest has its own acceptance boundary.
# Import json for this computation.
import json
import tempfile
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
# Read or serialize artifact data on disk (`root`).
root=Path(tempfile.mkdtemp(prefix='jax-async-lesson-'))
# Function `transition(state)` implementing this stage's computation:
def transition(state):
    # Return `{'weight': state['weight'] + 0.25, 'step': state['step'] + 1}` to the caller.
    return {'weight':state['weight']+.25,'step':state['step']+1}
# Function `publish(path, step)` implementing this stage's computation:
def publish(path, step):
    # Compute `pending` from `root/'LATEST.pending'`
    pending=root/'LATEST.pending'
    # Write the serialized artifact payload to disk.
    pending.write_text(json.dumps({'path':str(path),'step':step}))
    # Run `pending.replace` to perform the next check or state transition.
    pending.replace(root/'LATEST.json')
# Function `accepted()` implementing this stage's computation:
def accepted():
    # Return `json.loads((root / 'LATEST.json').read_text())` to the caller.
    return json.loads((root/'LATEST.json').read_text())

# Step 2 — Save, continue and verify: Wait on the real background writer, restore against the snapshot...
# Construct `state` via `{'weight':jnp.arange(1024,dtype=jnp.float32),'step':...`
state={'weight':jnp.arange(1024,dtype=jnp.float32),'step':jnp.array(0,jnp.int32)}
# Run `transition` to compute `snapshot`.
snapshot=transition(state)
# Compute `path` from `root/'step_1'`
path=root/'step_1'
# Record execution timing or profiler trace in `t0`.
t0=time.perf_counter()
# Enter `ocp.AsyncCheckpointer(ocp.StandardCheckpointH` context block:
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    # Run `cp.save` to perform the next check or state transition.
    cp.save(path,args=ocp.args.StandardSave(snapshot))
    # Record execution timing or profiler trace in `returned`.
    returned=time.perf_counter()
    # Assert invariant `not (root/'LATEST.json').exists()` holds
    assert not (root/'LATEST.json').exists()
    # Run `transition` to compute `live`.
    live=transition(snapshot)
    # Run `cp.wait_until_finished` to perform the next check or state transition.
    cp.wait_until_finished()
    # Record execution timing or profiler trace in `committed`.
    committed=time.perf_counter()
    # Run `cp.restore` to compute `restored`.
    restored=cp.restore(path,args=ocp.args.StandardRestore(snapshot))
    # Execute `np.testing.assert_array_equal(restored['weight'],snapshot['w`
    np.testing.assert_array_equal(restored['weight'],snapshot['weight'])
    # Assert invariant `int(restored['step'])==1 and int(live['step'])==2` holds
    assert int(restored['step'])==1 and int(live['step'])==2
    # Run `publish` to perform the next check or state transition.
    publish(path,1)
# Compute `report` from `{'save_call_seconds':returned-t0,'remaining_work_and...`
report={'save_call_seconds':returned-t0,'remaining_work_and_wait_seconds':committed-returned,
        'accepted_step':accepted()['step'],'live_step':int(live['step'])}
# Assert invariant `report['accepted_step']==1` holds
assert report['accepted_step']==1

# A deliberately interrupted application publication leaves the previous commit selected.
unpublished=root/'step_2'
# Enter `ocp.AsyncCheckpointer(ocp.StandardCheckpointH` context block:
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    # Run `cp.save` to perform the next check or state transition.
    cp.save(unpublished,args=ocp.args.StandardSave(live))
    # Run `cp.wait_until_finished` to perform the next check or state transition.
    cp.wait_until_finished()
    # Run the boundary check and catch the expected exception:
    try:
        raise RuntimeError('injected interruption before updating LATEST')
    except RuntimeError:
        pass
# Assert invariant `accepted()['step']==1` holds
assert accepted()['step']==1
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as cp:
    # Run `cp.restore` to compute `previous`.
    previous=cp.restore(accepted()['path'],target=snapshot)
# Execute `np.testing.assert_array_equal(previous['weight'],snapshot['w`
np.testing.assert_array_equal(previous['weight'],snapshot['weight'])
# Assert invariant `int(previous['step'])==1` holds
assert int(previous['step'])==1
# Print the observed values to compare against the expected result.
print('Checkpoint workspace:',root)
# Print diagnostic summary of the computed outputs.
print(json.dumps(report,indent=2))
# Print diagnostic summary of the computed outputs.
print('Unpublished step 2 exists; application recovery still selects verified step 1.')

# Figure data experiment
# Compute figure data for: A live step can be ahead of the accepted checkpoint
# Compute `visual_data` from `{'kind':'bar','labels':['saved snapshot','live state...`
visual_data={'kind':'bar','labels':['saved snapshot','live state','accepted checkpoint'],'ylabel':'completed step','series':[{'label':'local publication experiment','y':[int(snapshot["step"]),int(live["step"]),report["accepted_step"]]}]}

# Experiment: Trigger an asynchronous error
# Experiment — Trigger an asynchronous error: The controlled failure happens in the finalization callback.
def injected_failure():
    raise RuntimeError('injected finalization callback failure')
# Compute `failed_path` from `root/'callback_failure'`
failed_path=root/'callback_failure'
# Run `ocp.AsyncCheckpointer` to compute `cp`.
cp=ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler(),async_options=ocp.options.AsyncOptions(post_finalization_callback=injected_failure))
# Run the boundary check and catch the expected exception:
try:
    cp.save(failed_path,args=ocp.args.StandardSave(live))
    # Run the boundary check and catch the expected exception:
    try:
        cp.wait_until_finished()
    except RuntimeError as error:
        assert 'injected' in str(error)
        print('Expected background failure surfaced:',error)
    else:
        raise AssertionError('background error was hidden')
finally:
    cp.close()
# Assert invariant `accepted()['step']==1` holds
assert accepted()['step']==1

# Reference solution. Try the exercise before reading this.
# Exercise solution: Verify and publish the already written second checkpoint.
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as cp:
    # Run `cp.restore` to compute `verified`.
    verified=cp.restore(unpublished,target=live)
# Execute `np.testing.assert_array_equal(verified['weight'],live['weigh`
np.testing.assert_array_equal(verified['weight'],live['weight'])
# Assert invariant `int(verified['step'])==2` holds
assert int(verified['step'])==2
# Run `publish` to perform the next check or state transition.
publish(unpublished,2)
# Execute `np.testing.assert_array_equal(transition(verified)['weight']`
np.testing.assert_array_equal(transition(verified)['weight'],transition(live)['weight'])
# Assert invariant `accepted()['step']==2 and int(snapshot['step'])==1` holds
assert accepted()['step']==2 and int(snapshot['step'])==1
# Print the observed values to compare against the expected result.
print('Step two verified and accepted; next transition agrees.')

# Reference practice: Keep only a genuinely accepted target
# Keep only a genuinely accepted target (Transfer): File presence is a necessary local preflight, not proof that...
def preflight(record):
    # Guard input contract (`not Path(record['path']).is_dir() or int(record['step']) < 0`) and fail fast if violated.
    if not Path(record['path']).is_dir() or int(record['step'])<0:
        raise ValueError('checkpoint target unavailable')
    # Return `record` to the caller.
    return record
# Run `accepted` to compute `before`.
before=accepted()
# Run the boundary check and catch the expected exception:
try:preflight({'path':str(root/'never_saved'),'step':99})
except ValueError:pass
else:raise AssertionError('missing target accepted')
# Assert invariant `accepted()==before` holds
assert accepted()==before
# Run `preflight` to perform the next check or state transition.
preflight(before)
# Print the observed values to compare against the expected result.
print('Missing target rejected; accepted checkpoint unchanged.')
print("PASS: recovery-05")
