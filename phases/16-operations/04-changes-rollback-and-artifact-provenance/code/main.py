"""Changes, rollback, and artifact provenance: worked experiments and reference solutions. CPU checks."""

# Create the local artifact helpers
# Step 1: Create the local artifact helpers
"""Bounded local JAX workload operations. No scheduler, cloud, or GPU emulator."""
# Import pathlib (Path) for this computation.
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time


# Function `digest(value)` implementing this stage's computation:
def digest(value):
    # Return `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()` to the caller.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


# Function `atomic_json(path, value)` implementing this stage's computation:
def atomic_json(path, value):
    """Replace one local JSON file only after its bytes have been flushed."""
    # Read or serialize artifact data on disk (`path`).
    # Execute the next step of the computation.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Run `tempfile.mkstemp` to compute `(fd, temporary)`.
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    # Run the boundary check and catch the expected exception:
    try:
        # Enter `os.fdopen(fd, 'w')` context block:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

# Define the supervised worker
# This file is materialized only inside the caller-owned temporary run folder.
WORKER = r'''
import hashlib, json, os, sys, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
root = Path(sys.argv[1])
cfg = json.loads((root/'config.json').read_text())
worker_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def emit(event, **fields):
    print(json.dumps(dict(event=event, monotonic_s=time.monotonic(), **fields)), flush=True)
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
def write_checkpoint(state):
    target = root/'checkpoint.json'
    temporary = root/'checkpoint.pending'
    with temporary.open('w') as stream:
        json.dump(dict(state,checksum=digest(state)),stream,sort_keys=True,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,target)
    emit('checkpoint',step=state['step'],state_hash=digest(state))
try:
    emit('started',pid=os.getpid(),backend=jax.default_backend(),device_count=jax.device_count())
    if jax.default_backend() != cfg['backend'] or jax.device_count() < cfg['min_devices']:
        raise ValueError('runtime backend/device contract failed')
    x = jnp.linspace(-1.,1.,64)
    y = 2*x+1
    data_hash = digest(dict(x=np.asarray(x).tolist(),y=np.asarray(y).tolist()))
    training = {name:cfg[name] for name in ('seed','learning_rate','momentum','batch_size')}
    config_hash = digest(training)
    if cfg['resume']:
        saved = json.loads((root/'checkpoint.json').read_text())
        checksum = saved.pop('checksum',None)
        if checksum != digest(saved) or saved.get('worker_hash') != worker_hash:
            raise ValueError('checkpoint checksum/source mismatch')
        if saved['schema_version'] != 1 or saved['config_hash'] != config_hash or saved['data_hash'] != data_hash:
            raise ValueError('checkpoint provenance mismatch')
        state = saved
        step = saved['step']
        params=jnp.array(saved['params'])
        velocity=jnp.array(saved['velocity'])
        key=jnp.array(saved['key'],dtype=jnp.uint32)
        if not isinstance(step,int) or isinstance(step,bool) or step < 0 or step > cfg['steps'] or params.shape != (2,) or velocity.shape != (2,) or key.shape != (2,):
            raise ValueError('invalid checkpoint state shape or step')
        if not bool(jnp.all(jnp.isfinite(params))) or not bool(jnp.all(jnp.isfinite(velocity))):
            raise ValueError('nonfinite checkpoint state')
        emit('restored',step=step,state_hash=digest(saved))
    else:
        step=0
        params=jnp.zeros(2)
        velocity=jnp.zeros(2)
        key=jax.random.PRNGKey(cfg['seed'])
    def update(params,velocity,key):
        key, sample = jax.random.split(key)
        indices=jax.random.choice(sample,64,(cfg['batch_size'],),replace=False)
        objective=lambda p:jnp.mean((p[0]*x[indices]+p[1]-y[indices])**2)
        loss, grad=jax.value_and_grad(objective)(params)
        velocity=cfg['momentum']*velocity+grad
        params=params-cfg['learning_rate']*velocity
        return params,velocity,key,loss
    compiled=jax.jit(update)
    # Compile and synchronize once without consuming actual training state.
    begin=time.perf_counter()
    warm=compiled(params,velocity,key)
    jax.block_until_ready(warm)
    emit('ready',compile_warmup_s=time.perf_counter()-begin,step=step)
    while step < cfg['steps']:
        if cfg.get('stall_at') == step:
            emit('stall_injected',step=step)
            time.sleep(cfg.get('stall_seconds',10.))
        begin=time.perf_counter()
        params,velocity,key,loss=compiled(params,velocity,key)
        jax.block_until_ready((params,velocity,key,loss))
        elapsed=time.perf_counter()-begin
        step+=1
        state=dict(schema_version=1,worker_hash=worker_hash,step=step,params=np.asarray(params).tolist(),velocity=np.asarray(velocity).tolist(),key=np.asarray(key).tolist(),config_hash=config_hash,data_hash=data_hash)
        emit('progress',step=step,loss=float(loss),examples=cfg['batch_size'],update_s=elapsed,state_hash=digest(state))
        if step % cfg['checkpoint_every'] == 0 or step == cfg['steps']:
            write_checkpoint(state)
        if cfg.get('fail_after') == step:
            raise RuntimeError('controlled failure after update')
    write_checkpoint(state)
    emit('completed',step=step,state_hash=digest(state))
except Exception as error:
    emit('failed',kind=type(error).__name__,message=str(error))
    raise SystemExit(23)
'''

# Add bounded launch and result collection
# Step 3 — Add bounded launch and result collection: The supervisor owns the child handle, captures its output and...
def default_config(**changes):
    # Evaluate `seed=3, learning_rate=0.04, momentum=0.8, batch_size=8, steps=8, checkpoint_every=2, backend='cpu', min_devices=1, resume=False, fail_after=None, stall_at=None, stall_seconds=10.0` and convert the result into Python scalar/collection `cfg`.
    cfg=dict(seed=3,learning_rate=.04,momentum=.8,batch_size=8,steps=8,
             checkpoint_every=2,backend='cpu',min_devices=1,resume=False,
             fail_after=None,stall_at=None,stall_seconds=10.)
    # Update state in place with the new values.
    cfg.update(changes)
    # Iterate over `name` to step through the computation:
    for name in ('steps','checkpoint_every','batch_size','min_devices'):
        # Guard input contract (`not isinstance(cfg[name], int) or isinstance(cfg[name], bool) or cfg[name] < 1`) and fail fast if violated.
        if not isinstance(cfg[name],int) or isinstance(cfg[name],bool) or cfg[name] < 1: raise ValueError(name+' must be a positive integer')
    # Guard input contract (`cfg['batch_size'] > 64 or not 0 <= cfg['momentum'] < 1 or (not 0 < cfg['learning_rate'] < 1)`) and fail fast if violated.
    if cfg['batch_size'] > 64 or not 0 <= cfg['momentum'] < 1 or not 0 < cfg['learning_rate'] < 1:
        raise ValueError('invalid training configuration')
    # Return `cfg` to the caller.
    return cfg


# Function `launch(root, config, timeout)` implementing this stage's computation:
def launch(root, config=None, timeout=10.):
    # Read or serialize artifact data on disk (`root`).
    # Execute the next step of the computation.
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    # Run `default_config` to compute `cfg`.
    cfg=default_config(**(config or {}))
    # Guard input contract (`timeout <= 0`) and fail fast if violated.
    if timeout <= 0: raise ValueError('timeout must be positive')
    # Run `atomic_json` to perform the next check or state transition.
    atomic_json(root/'config.json',cfg)
    # Evaluate `worker` from the current inputs and state.
    # Read or serialize artifact data on disk (``).
    worker=root/'worker.py'
    worker.write_text(WORKER)
    # Configure environment variable before initializing the runtime.
    env=dict(os.environ,JAX_PLATFORMS=cfg['backend'],PYTHONUNBUFFERED='1')
    # Record execution timing or profiler trace in `started`.
    started=time.perf_counter()
    # Read or serialize artifact data on disk (`process`).
    process=subprocess.Popen([sys.executable,str(worker),str(root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
    # Compute `timed_out` from `False`
    timed_out=False
    try:
        stdout,stderr=process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True
        process.terminate()
        try: stdout,stderr=process.communicate(timeout=1.)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout,stderr=process.communicate(timeout=1.)
    # Record execution timing or profiler trace in `wall`.
    wall=time.perf_counter()-started
    # Evaluate `events` from the current inputs and state.
    # Compute `events` from `[]`
    events=[]
    malformed=[]
    # Loop over `line` in `stdout.splitlines()`:
    for line in stdout.splitlines():
        # Branch on condition `not line.strip()`:
        if not line.strip(): continue
        try:
            event=json.loads(line)
            if not isinstance(event,dict) or 'event' not in event: raise ValueError('not an event')
            events.append(event)
        except (ValueError,TypeError): malformed.append(line)
    # Compute `status` from `'timed_out' if timed_out else 'completed' if process...`
    status='timed_out' if timed_out else 'completed' if process.returncode == 0 and not malformed and events and events[-1]['event']=='completed' else 'failed'
    # Compute deterministic cryptographic digest `result` for provenance verification.
    result=dict(status=status,returncode=process.returncode,wall_s=wall,events=events,stderr=stderr,
                worker_hash=hashlib.sha256(WORKER.encode()).hexdigest(),config=cfg,malformed_stdout=malformed)
    # Run `atomic_json` to perform the next check or state transition.
    atomic_json(root/'run.json',result)
    # Return `result` to the caller.
    return result

# Add content-addressed publication and selection
# Step 4 — Add content-addressed publication and selection: Publication and activation are separate; rollback uses the same...
def publish(store, artifact):
    """Content-address one JSON artifact. Publishing does not activate it."""
    # Read or serialize artifact data on disk (`store`).
    # Run `digest` to compute `identifier`.
    store=Path(store)
    identifier=digest(artifact)
    # Run `atomic_json` to perform the next check or state transition.
    atomic_json(store/'artifacts'/(identifier+'.json'),artifact)
    # Return `identifier` to the caller.
    return identifier


# Function `read_artifact(store, identifier)` implementing this stage's computation:
def read_artifact(store, identifier):
    # Guard input contract (`len(identifier) != 64 or any((c not in '0123456789abcdef' for c in identifier))`) and fail fast if violated.
    if len(identifier) != 64 or any(c not in '0123456789abcdef' for c in identifier):
        raise ValueError('invalid artifact identifier')
    # Read or serialize artifact data on disk (`artifact`).
    artifact=json.loads((Path(store)/'artifacts'/(identifier+'.json')).read_text())
    # Guard input contract (`digest(artifact) != identifier`) and fail fast if violated.
    if digest(artifact) != identifier: raise ValueError('artifact digest mismatch')
    # Return `artifact` to the caller.
    return artifact


# Function `artifact_metrics(artifact)` implementing this stage's computation:
def artifact_metrics(artifact):
    """Independent held-out arithmetic for this fixed linear-model release contract."""
    # Run `artifact.get` to compute `params`.
    params=artifact.get('state',{}).get('params',[])
    # Guard input contract (`len(params) != 2 or any((not math.isfinite(float(v)) for v in params))`) and fail fast if violated.
    if len(params)!=2 or any(not math.isfinite(float(v)) for v in params):
        raise ValueError('invalid model parameters')
    # Compute `inputs` from `(-1.5,-.37,.22,1.5)`
    inputs=(-1.5,-.37,.22,1.5)
    # Run `sum` to compute `mse`.
    mse=sum((params[0]*x+params[1]-(2*x+1))**2 for x in inputs)/len(inputs)
    # Return `dict(held_out_mse=mse, acceptance_limit=1.0, passed=mse <= 1.0)` to the caller.
    return dict(held_out_mse=mse,acceptance_limit=1.,passed=mse<=1.)


# Function `activate(store, identifier)` implementing this stage's computation:
def activate(store, identifier):
    # Run `read_artifact` to compute `artifact`.
    artifact=read_artifact(store,identifier)
    # Guard input contract (`artifact.get('validation', {}).get('passed') is not True`) and fail fast if violated.
    if artifact.get('validation',{}).get('passed') is not True:
        raise ValueError('artifact has no passing validation evidence')
    # Compute `required` from `['worker_hash','config_hash','data_hash','state']`
    required=['worker_hash','config_hash','data_hash','state']
    # Guard input contract (`any((name not in artifact for name in required))`) and fail fast if violated.
    if any(name not in artifact for name in required): raise ValueError('artifact provenance incomplete')
    # Guard input contract (`not artifact_metrics(artifact)['passed']`) and fail fast if violated.
    if not artifact_metrics(artifact)['passed']: raise ValueError('held-out model validation failed')
    # Evaluate `previous` from the current inputs and state.
    # Read or serialize artifact data on disk (`pointer`).
    previous=None
    pointer=Path(store)/'active.json'
    # Branch on condition `pointer.exists()`:
    if pointer.exists():previous=json.loads(pointer.read_text())['current']
    # Run `atomic_json` to perform the next check or state transition.
    atomic_json(pointer,dict(current=identifier,previous=previous))
    # Return `identifier` to the caller.
    return identifier


# Function `rollback(store)` implementing this stage's computation:
def rollback(store):
    # Read or serialize artifact data on disk (`pointer`).
    pointer=json.loads((Path(store)/'active.json').read_text())
    # Guard input contract (`not pointer.get('previous')`) and fail fast if violated.
    if not pointer.get('previous'):raise ValueError('no previous version')
    # Return `activate(store, pointer['previous'])` to the caller.
    return activate(store,pointer['previous'])

# Rehearse a local release and rollback
# Step 5 — Rehearse a local release and rollback: The artifact contains a checkpoint from an actual successful JAX job.
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='ops-release-') as folder:
    # Read or serialize artifact data on disk (`base`).
    # Run `launch` to compute `run`.
    base=Path(folder)
    run=launch(base/'run')
    # Read or serialize artifact data on disk (`state`).
    state=json.loads((base/'run'/'checkpoint.json').read_text())
    # Compute `store` from `base/'registry'`
    store=base/'registry'
    # Evaluate `worker_hash=run['worker_hash'], config_hash=state['config_hash'], data_hash=state['data_hash'], state=state, validation={'passed': True, 'check': 'completed CPU fixture and checksum'}` and convert the result into Python scalar/collection `artifact`.
    artifact=dict(worker_hash=run['worker_hash'],config_hash=state['config_hash'],data_hash=state['data_hash'],state=state,validation={'passed':True,'check':'completed CPU fixture and checksum'})
    # Run `publish` to compute `version_a`.
    version_a=publish(store,artifact)
    # Assert invariant `not (store/'active.json').exists()` holds
    assert not (store/'active.json').exists()
    # Run `activate` to perform the next check or state transition.
    activate(store,version_a)
    # Run `publish` to compute `version_b`.
    version_b=publish(store,dict(artifact,release_note='reviewed metadata revision'))
    # Run `activate` to perform the next check or state transition.
    activate(store,version_b)
    # Run `publish` to compute `rejected`.
    rejected=publish(store,dict(artifact,validation={'passed':False}))
    # Compute `before` from `(store/'active.json').read_bytes()`
    before=(store/'active.json').read_bytes()
    try: activate(store,rejected)
    except ValueError: rejected_without_change=(store/'active.json').read_bytes()==before
    else: raise AssertionError('bad release activated')
    # Assert invariant `rejected_without_change` holds
    assert rejected_without_change
    # Assert invariant `rollback(store)==version_a` holds
    assert rollback(store)==version_a
    # Read or serialize artifact data on disk (`chosen`).
    chosen=json.loads((store/'active.json').read_text())
    # Assert invariant `chosen['current']==version_a` holds
    assert chosen['current']==version_a
# Print the observed values to compare against the expected result.
print('first / second content IDs:',version_a,version_b)
# Print diagnostic summary of the computed outputs.
print('failed gate kept selection:',rejected_without_change,'rollback restored first:',chosen['current']==version_a)

# Complete runnable example (operations-04)
"""Bounded local JAX workload operations. No scheduler, cloud, or GPU emulator."""
# Import pathlib (Path) for this computation.
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time


# Function `digest(value)` implementing this stage's computation:
def digest(value):
    # Return `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()` to the caller.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


# Function `atomic_json(path, value)` implementing this stage's computation:
def atomic_json(path, value):
    """Replace one local JSON file only after its bytes have been flushed."""
    # Read or serialize artifact data on disk (`path`).
    # Execute the next step of the computation.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Run `tempfile.mkstemp` to compute `(fd, temporary)`.
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    # Run the boundary check and catch the expected exception:
    try:
        # Enter `os.fdopen(fd, 'w')` context block:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

# This file is materialized only inside the caller-owned temporary run folder.
WORKER = r'''
import hashlib, json, os, sys, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
root = Path(sys.argv[1])
cfg = json.loads((root/'config.json').read_text())
worker_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def emit(event, **fields):
    print(json.dumps(dict(event=event, monotonic_s=time.monotonic(), **fields)), flush=True)
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
def write_checkpoint(state):
    target = root/'checkpoint.json'
    temporary = root/'checkpoint.pending'
    with temporary.open('w') as stream:
        json.dump(dict(state,checksum=digest(state)),stream,sort_keys=True,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,target)
    emit('checkpoint',step=state['step'],state_hash=digest(state))
try:
    emit('started',pid=os.getpid(),backend=jax.default_backend(),device_count=jax.device_count())
    if jax.default_backend() != cfg['backend'] or jax.device_count() < cfg['min_devices']:
        raise ValueError('runtime backend/device contract failed')
    x = jnp.linspace(-1.,1.,64)
    y = 2*x+1
    data_hash = digest(dict(x=np.asarray(x).tolist(),y=np.asarray(y).tolist()))
    training = {name:cfg[name] for name in ('seed','learning_rate','momentum','batch_size')}
    config_hash = digest(training)
    if cfg['resume']:
        saved = json.loads((root/'checkpoint.json').read_text())
        checksum = saved.pop('checksum',None)
        if checksum != digest(saved) or saved.get('worker_hash') != worker_hash:
            raise ValueError('checkpoint checksum/source mismatch')
        if saved['schema_version'] != 1 or saved['config_hash'] != config_hash or saved['data_hash'] != data_hash:
            raise ValueError('checkpoint provenance mismatch')
        state = saved
        step = saved['step']
        params=jnp.array(saved['params'])
        velocity=jnp.array(saved['velocity'])
        key=jnp.array(saved['key'],dtype=jnp.uint32)
        if not isinstance(step,int) or isinstance(step,bool) or step < 0 or step > cfg['steps'] or params.shape != (2,) or velocity.shape != (2,) or key.shape != (2,):
            raise ValueError('invalid checkpoint state shape or step')
        if not bool(jnp.all(jnp.isfinite(params))) or not bool(jnp.all(jnp.isfinite(velocity))):
            raise ValueError('nonfinite checkpoint state')
        emit('restored',step=step,state_hash=digest(saved))
    else:
        step=0
        params=jnp.zeros(2)
        velocity=jnp.zeros(2)
        key=jax.random.PRNGKey(cfg['seed'])
    def update(params,velocity,key):
        key, sample = jax.random.split(key)
        indices=jax.random.choice(sample,64,(cfg['batch_size'],),replace=False)
        objective=lambda p:jnp.mean((p[0]*x[indices]+p[1]-y[indices])**2)
        loss, grad=jax.value_and_grad(objective)(params)
        velocity=cfg['momentum']*velocity+grad
        params=params-cfg['learning_rate']*velocity
        return params,velocity,key,loss
    compiled=jax.jit(update)
    # Compile and synchronize once without consuming actual training state.
    begin=time.perf_counter()
    warm=compiled(params,velocity,key)
    jax.block_until_ready(warm)
    emit('ready',compile_warmup_s=time.perf_counter()-begin,step=step)
    while step < cfg['steps']:
        if cfg.get('stall_at') == step:
            emit('stall_injected',step=step)
            time.sleep(cfg.get('stall_seconds',10.))
        begin=time.perf_counter()
        params,velocity,key,loss=compiled(params,velocity,key)
        jax.block_until_ready((params,velocity,key,loss))
        elapsed=time.perf_counter()-begin
        step+=1
        state=dict(schema_version=1,worker_hash=worker_hash,step=step,params=np.asarray(params).tolist(),velocity=np.asarray(velocity).tolist(),key=np.asarray(key).tolist(),config_hash=config_hash,data_hash=data_hash)
        emit('progress',step=step,loss=float(loss),examples=cfg['batch_size'],update_s=elapsed,state_hash=digest(state))
        if step % cfg['checkpoint_every'] == 0 or step == cfg['steps']:
            write_checkpoint(state)
        if cfg.get('fail_after') == step:
            raise RuntimeError('controlled failure after update')
    write_checkpoint(state)
    emit('completed',step=step,state_hash=digest(state))
except Exception as error:
    emit('failed',kind=type(error).__name__,message=str(error))
    raise SystemExit(23)
'''

# Step 3 — Add bounded launch and result collection: The supervisor owns the child handle, captures its output and...
def default_config(**changes):
    # Evaluate `seed=3, learning_rate=0.04, momentum=0.8, batch_size=8, steps=8, checkpoint_every=2, backend='cpu', min_devices=1, resume=False, fail_after=None, stall_at=None, stall_seconds=10.0` and convert the result into Python scalar/collection `cfg`.
    cfg=dict(seed=3,learning_rate=.04,momentum=.8,batch_size=8,steps=8,
             checkpoint_every=2,backend='cpu',min_devices=1,resume=False,
             fail_after=None,stall_at=None,stall_seconds=10.)
    # Update state in place with the new values.
    cfg.update(changes)
    # Iterate over `name` to step through the computation:
    for name in ('steps','checkpoint_every','batch_size','min_devices'):
        # Guard input contract (`not isinstance(cfg[name], int) or isinstance(cfg[name], bool) or cfg[name] < 1`) and fail fast if violated.
        if not isinstance(cfg[name],int) or isinstance(cfg[name],bool) or cfg[name] < 1: raise ValueError(name+' must be a positive integer')
    # Guard input contract (`cfg['batch_size'] > 64 or not 0 <= cfg['momentum'] < 1 or (not 0 < cfg['learning_rate'] < 1)`) and fail fast if violated.
    if cfg['batch_size'] > 64 or not 0 <= cfg['momentum'] < 1 or not 0 < cfg['learning_rate'] < 1:
        raise ValueError('invalid training configuration')
    # Return `cfg` to the caller.
    return cfg


# Function `launch(root, config, timeout)` implementing this stage's computation:
def launch(root, config=None, timeout=10.):
    # Read or serialize artifact data on disk (`root`).
    # Execute the next step of the computation.
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    # Run `default_config` to compute `cfg`.
    cfg=default_config(**(config or {}))
    # Guard input contract (`timeout <= 0`) and fail fast if violated.
    if timeout <= 0: raise ValueError('timeout must be positive')
    # Run `atomic_json` to perform the next check or state transition.
    atomic_json(root/'config.json',cfg)
    # Evaluate `worker` from the current inputs and state.
    # Read or serialize artifact data on disk (``).
    worker=root/'worker.py'
    worker.write_text(WORKER)
    # Configure environment variable before initializing the runtime.
    env=dict(os.environ,JAX_PLATFORMS=cfg['backend'],PYTHONUNBUFFERED='1')
    # Record execution timing or profiler trace in `started`.
    started=time.perf_counter()
    # Read or serialize artifact data on disk (`process`).
    process=subprocess.Popen([sys.executable,str(worker),str(root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
    # Compute `timed_out` from `False`
    timed_out=False
    try:
        stdout,stderr=process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True
        process.terminate()
        try: stdout,stderr=process.communicate(timeout=1.)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout,stderr=process.communicate(timeout=1.)
    # Record execution timing or profiler trace in `wall`.
    wall=time.perf_counter()-started
    # Evaluate `events` from the current inputs and state.
    # Compute `events` from `[]`
    events=[]
    malformed=[]
    # Loop over `line` in `stdout.splitlines()`:
    for line in stdout.splitlines():
        # Branch on condition `not line.strip()`:
        if not line.strip(): continue
        try:
            event=json.loads(line)
            if not isinstance(event,dict) or 'event' not in event: raise ValueError('not an event')
            events.append(event)
        except (ValueError,TypeError): malformed.append(line)
    # Compute `status` from `'timed_out' if timed_out else 'completed' if process...`
    status='timed_out' if timed_out else 'completed' if process.returncode == 0 and not malformed and events and events[-1]['event']=='completed' else 'failed'
    # Compute deterministic cryptographic digest `result` for provenance verification.
    result=dict(status=status,returncode=process.returncode,wall_s=wall,events=events,stderr=stderr,
                worker_hash=hashlib.sha256(WORKER.encode()).hexdigest(),config=cfg,malformed_stdout=malformed)
    # Run `atomic_json` to perform the next check or state transition.
    atomic_json(root/'run.json',result)
    # Return `result` to the caller.
    return result

# Step 4 — Add content-addressed publication and selection: Publication and activation are separate; rollback uses the same...
def publish(store, artifact):
    """Content-address one JSON artifact. Publishing does not activate it."""
    # Read or serialize artifact data on disk (`store`).
    # Run `digest` to compute `identifier`.
    store=Path(store)
    identifier=digest(artifact)
    # Run `atomic_json` to perform the next check or state transition.
    atomic_json(store/'artifacts'/(identifier+'.json'),artifact)
    # Return `identifier` to the caller.
    return identifier


# Function `read_artifact(store, identifier)` implementing this stage's computation:
def read_artifact(store, identifier):
    # Guard input contract (`len(identifier) != 64 or any((c not in '0123456789abcdef' for c in identifier))`) and fail fast if violated.
    if len(identifier) != 64 or any(c not in '0123456789abcdef' for c in identifier):
        raise ValueError('invalid artifact identifier')
    # Read or serialize artifact data on disk (`artifact`).
    artifact=json.loads((Path(store)/'artifacts'/(identifier+'.json')).read_text())
    # Guard input contract (`digest(artifact) != identifier`) and fail fast if violated.
    if digest(artifact) != identifier: raise ValueError('artifact digest mismatch')
    # Return `artifact` to the caller.
    return artifact


# Function `artifact_metrics(artifact)` implementing this stage's computation:
def artifact_metrics(artifact):
    """Independent held-out arithmetic for this fixed linear-model release contract."""
    # Run `artifact.get` to compute `params`.
    params=artifact.get('state',{}).get('params',[])
    # Guard input contract (`len(params) != 2 or any((not math.isfinite(float(v)) for v in params))`) and fail fast if violated.
    if len(params)!=2 or any(not math.isfinite(float(v)) for v in params):
        raise ValueError('invalid model parameters')
    # Compute `inputs` from `(-1.5,-.37,.22,1.5)`
    inputs=(-1.5,-.37,.22,1.5)
    # Run `sum` to compute `mse`.
    mse=sum((params[0]*x+params[1]-(2*x+1))**2 for x in inputs)/len(inputs)
    # Return `dict(held_out_mse=mse, acceptance_limit=1.0, passed=mse <= 1.0)` to the caller.
    return dict(held_out_mse=mse,acceptance_limit=1.,passed=mse<=1.)


# Function `activate(store, identifier)` implementing this stage's computation:
def activate(store, identifier):
    # Run `read_artifact` to compute `artifact`.
    artifact=read_artifact(store,identifier)
    # Guard input contract (`artifact.get('validation', {}).get('passed') is not True`) and fail fast if violated.
    if artifact.get('validation',{}).get('passed') is not True:
        raise ValueError('artifact has no passing validation evidence')
    # Compute `required` from `['worker_hash','config_hash','data_hash','state']`
    required=['worker_hash','config_hash','data_hash','state']
    # Guard input contract (`any((name not in artifact for name in required))`) and fail fast if violated.
    if any(name not in artifact for name in required): raise ValueError('artifact provenance incomplete')
    # Guard input contract (`not artifact_metrics(artifact)['passed']`) and fail fast if violated.
    if not artifact_metrics(artifact)['passed']: raise ValueError('held-out model validation failed')
    # Evaluate `previous` from the current inputs and state.
    # Read or serialize artifact data on disk (`pointer`).
    previous=None
    pointer=Path(store)/'active.json'
    # Branch on condition `pointer.exists()`:
    if pointer.exists():previous=json.loads(pointer.read_text())['current']
    # Run `atomic_json` to perform the next check or state transition.
    atomic_json(pointer,dict(current=identifier,previous=previous))
    # Return `identifier` to the caller.
    return identifier


# Function `rollback(store)` implementing this stage's computation:
def rollback(store):
    # Read or serialize artifact data on disk (`pointer`).
    pointer=json.loads((Path(store)/'active.json').read_text())
    # Guard input contract (`not pointer.get('previous')`) and fail fast if violated.
    if not pointer.get('previous'):raise ValueError('no previous version')
    # Return `activate(store, pointer['previous'])` to the caller.
    return activate(store,pointer['previous'])

# Step 5 — Rehearse a local release and rollback: The artifact contains a checkpoint from an actual successful JAX job.
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='ops-release-') as folder:
    # Read or serialize artifact data on disk (`base`).
    # Run `launch` to compute `run`.
    base=Path(folder)
    run=launch(base/'run')
    # Read or serialize artifact data on disk (`state`).
    state=json.loads((base/'run'/'checkpoint.json').read_text())
    # Compute `store` from `base/'registry'`
    store=base/'registry'
    # Evaluate `worker_hash=run['worker_hash'], config_hash=state['config_hash'], data_hash=state['data_hash'], state=state, validation={'passed': True, 'check': 'completed CPU fixture and checksum'}` and convert the result into Python scalar/collection `artifact`.
    artifact=dict(worker_hash=run['worker_hash'],config_hash=state['config_hash'],data_hash=state['data_hash'],state=state,validation={'passed':True,'check':'completed CPU fixture and checksum'})
    # Run `publish` to compute `version_a`.
    version_a=publish(store,artifact)
    # Assert invariant `not (store/'active.json').exists()` holds
    assert not (store/'active.json').exists()
    # Run `activate` to perform the next check or state transition.
    activate(store,version_a)
    # Run `publish` to compute `version_b`.
    version_b=publish(store,dict(artifact,release_note='reviewed metadata revision'))
    # Run `activate` to perform the next check or state transition.
    activate(store,version_b)
    # Run `publish` to compute `rejected`.
    rejected=publish(store,dict(artifact,validation={'passed':False}))
    # Compute `before` from `(store/'active.json').read_bytes()`
    before=(store/'active.json').read_bytes()
    try: activate(store,rejected)
    except ValueError: rejected_without_change=(store/'active.json').read_bytes()==before
    else: raise AssertionError('bad release activated')
    # Assert invariant `rejected_without_change` holds
    assert rejected_without_change
    # Assert invariant `rollback(store)==version_a` holds
    assert rollback(store)==version_a
    # Read or serialize artifact data on disk (`chosen`).
    chosen=json.loads((store/'active.json').read_text())
    # Assert invariant `chosen['current']==version_a` holds
    assert chosen['current']==version_a
# Print the observed values to compare against the expected result.
print('first / second content IDs:',version_a,version_b)
# Print diagnostic summary of the computed outputs.
print('failed gate kept selection:',rejected_without_change,'rollback restored first:',chosen['current']==version_a)

# Figure data experiment
# Compute figure data for: The selected artifact changes only after a passing gate
# Enter managed runtime/context scope for this block:
with tempfile.TemporaryDirectory() as folder:
    # Run `publish` to compute `a`.
    # Execute the next step of the computation.
    a=publish(folder,artifact)
    activate(folder,a)
    # Compute `values` from `[1]`
    values=[1]
    # Run `publish` to compute `b`.
    # Read or serialize artifact data on disk (``).
    b=publish(folder,dict(artifact,release_note='B'))
    values.append(1 if json.loads((Path(folder)/'active.json').read_text())['current']==a else 2)
    # Run `activate` to perform the next check or state transition.
    # Read or serialize artifact data on disk (``).
    activate(folder,b)
    values.append(2 if json.loads((Path(folder)/'active.json').read_text())['current']==b else 1)
    # Run `publish` to compute `bad`.
    bad=publish(folder,dict(artifact,validation={'passed':False}))
    try:activate(folder,bad)
    except ValueError:pass
    # Read or serialize artifact data on disk (``).
    values.append(2 if json.loads((Path(folder)/'active.json').read_text())['current']==b else 1)
    # Run `rollback` to perform the next check or state transition.
    # Read or serialize artifact data on disk (``).
    rollback(folder)
    values.append(1 if json.loads((Path(folder)/'active.json').read_text())['current']==a else 2)
# Assert invariant `values==[1,1,2,2,1]` holds
assert values==[1,1,2,2,1]
# Compute `visual_data` from `{'kind':'bar','labels':['activate A','publish B','ac...`
visual_data={'kind':'bar','labels':['activate A','publish B','activate B','reject candidate','rollback'],'xlabel':'executed local operation','ylabel':'selected artifact code: 1=A, 2=B','series':[{'label':'observed active pointer','y':values}]}

# Experiment: Identity follows content rather than key insertion order
# Experiment — Identity follows content rather than key insertion order: Canonical encoding removes an incidental representation...
# Assert invariant `digest({'a':1` holds
assert digest({'a':1,'b':2})==digest({'b':2,'a':1})
# Assert invariant `digest({'a':1` holds
assert digest({'a':1,'b':2})!=digest({'a':1,'b':3})

# Experiment: Detect corruption before activation
# Experiment — Detect corruption before activation: A filename containing a hash is not enough: the reader must...
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory() as folder:
    # Run `publish` to compute `identifier`.
    identifier=publish(folder,artifact)
    # Read or serialize artifact data on disk (``).
    (Path(folder)/'artifacts'/(identifier+'.json')).write_text('{}')
    # Run the boundary check and catch the expected exception:
    try: activate(folder,identifier)
    except ValueError as error: assert 'digest' in str(error)
    else: raise AssertionError('corruption accepted')
    # Assert invariant `not (Path(folder)/'active.json').exists()` holds
    assert not (Path(folder)/'active.json').exists()

# Experiment: Reject an actually degraded model
# Experiment — Reject an actually degraded model: Measured validation and claimed validation are distinct.
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory() as folder:
    # Run `publish` to compute `stable`.
    # Execute the next step of the computation.
    stable=publish(folder,artifact)
    activate(folder,stable)
    # Evaluate `artifact, state=dict(artifact['state'], params=[100.0, -100.0])` and convert the result into Python scalar/collection `degraded`.
    degraded=dict(artifact,state=dict(artifact['state'],params=[100.,-100.]))
    # Run `publish` to compute `candidate`.
    candidate=publish(folder,degraded)
    # Assert invariant `artifact_metrics(degraded)['held_out_mse'] > 1.` holds
    assert artifact_metrics(degraded)['held_out_mse'] > 1.
    # Read or serialize artifact data on disk (`before`).
    before=(Path(folder)/'active.json').read_bytes()
    try:activate(folder,candidate)
    except ValueError as error:assert 'held-out' in str(error)
    else:raise AssertionError('degraded model activated')
    # Assert invariant `before==(Path(folder)/'active.json').read_bytes()` holds
    assert before==(Path(folder)/'active.json').read_bytes()

# Reference solution. Try the exercise before reading this.
# Exercise solution: Publish two compatible validated artifacts that differ in an explicit...
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory() as folder:
    # Run `publish` to compute `one`.
    one=publish(folder,dict(artifact,release_note='candidate one'))
    # Run `publish` to compute `two`.
    two=publish(folder,dict(artifact,release_note='candidate two'))
    # Run `activate` to perform the next check or state transition.
    # Run `activate` to perform the next check or state transition.
    activate(folder,one)
    activate(folder,two)
    # Assert invariant `rollback(folder)==one` holds
    assert rollback(folder)==one
    # Assert invariant `read_artifact(folder` holds
    assert read_artifact(folder,one)['release_note']=='candidate one'

# Reference practice: Reject an incomplete provenance record
# Reject an incomplete provenance record (Transfer / diagnosis): The gate establishes only the declared local contract; it...
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory() as folder:
    # Evaluate `artifact` and convert the result into Python scalar/collection `incomplete`.
    incomplete=dict(artifact)
    del incomplete['worker_hash']
    # Run `publish` to compute `identifier`.
    identifier=publish(folder,incomplete)
    # Run the boundary check and catch the expected exception:
    try: activate(folder,identifier)
    except ValueError: pass
    else: raise AssertionError('missing provenance accepted')

# Reference practice: Check rollback without history
# Check rollback without history (Transfer / diagnosis): A runbook needs a defined failure path when no compatible...
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory() as folder:
    # Run `publish` to compute `identifier`.
    # Execute the next step of the computation.
    identifier=publish(folder,artifact)
    activate(folder,identifier)
    # Read or serialize artifact data on disk (`before`).
    before=(Path(folder)/'active.json').read_bytes()
    # Run the boundary check and catch the expected exception:
    try: rollback(folder)
    except ValueError: pass
    else: raise AssertionError('invented rollback history')
    # Assert invariant `before==(Path(folder)/'active.json').read_bytes()` holds
    assert before==(Path(folder)/'active.json').read_bytes()
print("PASS: operations-04")
