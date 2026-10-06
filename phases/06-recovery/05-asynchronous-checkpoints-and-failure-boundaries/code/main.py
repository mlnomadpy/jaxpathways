"""Asynchronous checkpoints and failure boundaries: worked experiments and reference solutions. CPU checks."""

# Define functional state and publication
import json
import tempfile
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
root=Path(tempfile.mkdtemp(prefix='jax-async-lesson-'))
def transition(state):
    return {'weight':state['weight']+.25,'step':state['step']+1}
def publish(path, step):
    pending=root/'LATEST.pending'
    pending.write_text(json.dumps({'path':str(path),'step':step}))
    pending.replace(root/'LATEST.json')
def accepted():
    return json.loads((root/'LATEST.json').read_text())

# Save, continue and verify
state={'weight':jnp.arange(1024,dtype=jnp.float32),'step':jnp.array(0,jnp.int32)}
snapshot=transition(state)
path=root/'step_1'
t0=time.perf_counter()
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    cp.save(path,args=ocp.args.StandardSave(snapshot))
    returned=time.perf_counter()
    assert not (root/'LATEST.json').exists()
    live=transition(snapshot)
    cp.wait_until_finished()
    committed=time.perf_counter()
    restored=cp.restore(path,args=ocp.args.StandardRestore(snapshot))
    np.testing.assert_array_equal(restored['weight'],snapshot['weight'])
    assert int(restored['step'])==1 and int(live['step'])==2
    publish(path,1)
report={'save_call_seconds':returned-t0,'remaining_work_and_wait_seconds':committed-returned,
        'accepted_step':accepted()['step'],'live_step':int(live['step'])}
assert report['accepted_step']==1

# Interrupt publication deliberately
# A deliberately interrupted application publication leaves the previous commit selected.
unpublished=root/'step_2'
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    cp.save(unpublished,args=ocp.args.StandardSave(live))
    cp.wait_until_finished()
    try:
        raise RuntimeError('injected interruption before updating LATEST')
    except RuntimeError:
        pass
assert accepted()['step']==1
with ocp.StandardCheckpointer() as cp:
    previous=cp.restore(accepted()['path'],target=snapshot)
np.testing.assert_array_equal(previous['weight'],snapshot['weight'])
assert int(previous['step'])==1
print('Checkpoint workspace:',root)
print(json.dumps(report,indent=2))
print('Unpublished step 2 exists; application recovery still selects verified step 1.')


import json
import tempfile
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
root=Path(tempfile.mkdtemp(prefix='jax-async-lesson-'))
def transition(state):
    return {'weight':state['weight']+.25,'step':state['step']+1}
def publish(path, step):
    pending=root/'LATEST.pending'
    pending.write_text(json.dumps({'path':str(path),'step':step}))
    pending.replace(root/'LATEST.json')
def accepted():
    return json.loads((root/'LATEST.json').read_text())

state={'weight':jnp.arange(1024,dtype=jnp.float32),'step':jnp.array(0,jnp.int32)}
snapshot=transition(state)
path=root/'step_1'
t0=time.perf_counter()
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    cp.save(path,args=ocp.args.StandardSave(snapshot))
    returned=time.perf_counter()
    assert not (root/'LATEST.json').exists()
    live=transition(snapshot)
    cp.wait_until_finished()
    committed=time.perf_counter()
    restored=cp.restore(path,args=ocp.args.StandardRestore(snapshot))
    np.testing.assert_array_equal(restored['weight'],snapshot['weight'])
    assert int(restored['step'])==1 and int(live['step'])==2
    publish(path,1)
report={'save_call_seconds':returned-t0,'remaining_work_and_wait_seconds':committed-returned,
        'accepted_step':accepted()['step'],'live_step':int(live['step'])}
assert report['accepted_step']==1

# A deliberately interrupted application publication leaves the previous commit selected.
unpublished=root/'step_2'
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    cp.save(unpublished,args=ocp.args.StandardSave(live))
    cp.wait_until_finished()
    try:
        raise RuntimeError('injected interruption before updating LATEST')
    except RuntimeError:
        pass
assert accepted()['step']==1
with ocp.StandardCheckpointer() as cp:
    previous=cp.restore(accepted()['path'],target=snapshot)
np.testing.assert_array_equal(previous['weight'],snapshot['weight'])
assert int(previous['step'])==1
print('Checkpoint workspace:',root)
print(json.dumps(report,indent=2))
print('Unpublished step 2 exists; application recovery still selects verified step 1.')


# Figure data experiment
visual_data={'kind':'bar','labels':['saved snapshot','live state','accepted checkpoint'],'ylabel':'completed step','series':[{'label':'local publication experiment','y':[int(snapshot["step"]),int(live["step"]),report["accepted_step"]]}]}

# Experiment: Trigger an asynchronous error
def injected_failure():
    raise RuntimeError('injected finalization callback failure')
failed_path=root/'callback_failure'
cp=ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler(),async_options=ocp.options.AsyncOptions(post_finalization_callback=injected_failure))
try:
    cp.save(failed_path,args=ocp.args.StandardSave(live))
    try:
        cp.wait_until_finished()
    except RuntimeError as error:
        assert 'injected' in str(error)
        print('Expected background failure surfaced:',error)
    else:
        raise AssertionError('background error was hidden')
finally:
    cp.close()
assert accepted()['step']==1


# Reference solution. Try the exercise before reading this.
with ocp.StandardCheckpointer() as cp:
    verified=cp.restore(unpublished,target=live)
np.testing.assert_array_equal(verified['weight'],live['weight'])
assert int(verified['step'])==2
publish(unpublished,2)
np.testing.assert_array_equal(transition(verified)['weight'],transition(live)['weight'])
assert accepted()['step']==2 and int(snapshot['step'])==1
print('Step two verified and accepted; next transition agrees.')

# Reference practice: Keep only a genuinely accepted target
def preflight(record):
    if not Path(record['path']).is_dir() or int(record['step'])<0:
        raise ValueError('checkpoint target unavailable')
    return record
before=accepted()
try:preflight({'path':str(root/'never_saved'),'step':99})
except ValueError:pass
else:raise AssertionError('missing target accepted')
assert accepted()==before
preflight(before)
print('Missing target rejected; accepted checkpoint unchanged.')
print("PASS: recovery-05")
