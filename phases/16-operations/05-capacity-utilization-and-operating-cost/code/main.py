"""Capacity, utilization, and operating cost: worked experiments and reference solutions. CPU checks."""

# Create the local artifact helpers
# Step 1: Create the local artifact helpers
"""Bounded local JAX subprocess supervisor and lifecycle verifier."""
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
    # Compute `path` as `Path(path)`.
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
        if os.path.exists(temporary):
            os.unlink(temporary)

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
        if not isinstance(cfg[name],int) or isinstance(cfg[name],bool) or cfg[name] < 1:
            raise ValueError(name+' must be a positive integer')
    # Guard input contract (`cfg['batch_size'] > 64 or not 0 <= cfg['momentum'] < 1 or (not 0 < cfg['learning_rate'] < 1)`) and fail fast if violated.
    if cfg['batch_size'] > 64 or not 0 <= cfg['momentum'] < 1 or not 0 < cfg['learning_rate'] < 1:
        raise ValueError('invalid training configuration')
    # Return `cfg` to the caller.
    return cfg


# Function `launch(root, config, timeout)` implementing this stage's computation:
def launch(root, config=None, timeout=10.):
    # Compute `root` as `Path(root)`.
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    # Run `default_config` to compute `cfg`.
    cfg=default_config(**(config or {}))
    # Guard input contract (`timeout <= 0`) and fail fast if violated.
    if timeout <= 0:
        raise ValueError('timeout must be positive')
    # Run `atomic_json` to perform the next check or state transition.
    atomic_json(root/'config.json',cfg)
    # Compute `worker` as `root/'worker.py'`.
    worker=root/'worker.py'
    worker.write_text(WORKER)
    # Configure environment variable before initializing the runtime.
    env=dict(os.environ,JAX_PLATFORMS=cfg['backend'],PYTHONUNBUFFERED='1')
    # Record execution timing or profiler trace in `started`.
    started=time.perf_counter()
    # Read or serialize artifact data on disk (`process`).
    process=subprocess.Popen([sys.executable,str(worker),str(root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
    # Compute `timed_out` as `False`.
    timed_out=False
    try:
        stdout,stderr=process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True
        process.terminate()
        try:
            stdout,stderr=process.communicate(timeout=1.)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout,stderr=process.communicate(timeout=1.)
    # Record execution timing or profiler trace in `wall`.
    wall=time.perf_counter()-started
    # Compute `events` from `[]`
    events=[]
    malformed=[]
    # Loop over `line` in `stdout.splitlines()`:
    for line in stdout.splitlines():
        # Branch on condition `not line.strip()`:
        if not line.strip():
            continue
        try:
            event=json.loads(line)
            if not isinstance(event,dict) or 'event' not in event:
                raise ValueError('not an event')
            events.append(event)
        except (ValueError,TypeError):
            malformed.append(line)
    # Compute `status` from `'timed_out' if timed_out else 'completed' if process...`
    status='timed_out' if timed_out else 'completed' if process.returncode == 0 and not malformed and events and events[-1]['event']=='completed' else 'failed'
    # Compute deterministic cryptographic digest `result` for provenance verification.
    result=dict(status=status,returncode=process.returncode,wall_s=wall,events=events,stderr=stderr,
                worker_hash=hashlib.sha256(WORKER.encode()).hexdigest(),config=cfg,malformed_stdout=malformed)
    # Run `atomic_json` to perform the next check or state transition.
    atomic_json(root/'run.json',result)
    # Return `result` to the caller.
    return result

# Add metric summaries and capacity arithmetic
# Step 4 — Add metric summaries and capacity arithmetic: Keep measured quantities separate from invented planning inputs...
def summarize(result):
    # Compute `updates` from `[e for e in result['events'] if e['event']=='progress']`
    updates=[e for e in result['events'] if e['event']=='progress']
    # Compute `checkpoints` from `[e['step'] for e in result['events'] if e['event']==...`
    checkpoints=[e['step'] for e in result['events'] if e['event']=='checkpoint']
    # Run `sum` to compute `count`.
    # Run `sum` to compute `update_s`.
    count=sum(e['examples'] for e in updates)
    update_s=sum(e['update_s'] for e in updates)
    # Run `max` to compute `latest`.
    latest=max((e['step'] for e in updates),default=0)
    # Run `max` to compute `checkpoint`.
    checkpoint=max(checkpoints,default=0)
    # Return `dict(status=result['status'], updates=len(updates), examples=count, last_step=latest, checkpoint_step=checkpoint, uncheckpointed_updates=max(0, latest - checkpoint), update_s=update_s, update_examples_per_s=count / update_s if update_s else 0.0, job_examples_per_s=count / result['wall_s'] if result['wall_s'] else 0.0, update_duty_fraction=update_s / result['wall_s'] if result['wall_s'] else 0.0)` to the caller.
    return dict(status=result['status'],updates=len(updates),examples=count,last_step=latest,checkpoint_step=checkpoint,
                uncheckpointed_updates=max(0,latest-checkpoint),update_s=update_s,
                update_examples_per_s=count/update_s if update_s else 0.,
                job_examples_per_s=count/result['wall_s'] if result['wall_s'] else 0.,
                update_duty_fraction=update_s/result['wall_s'] if result['wall_s'] else 0.)

# Function `capacity(measured_job_s, arrivals_per_hour, workers, hourly_rate, ...)` implementing this stage's computation:
def capacity(measured_job_s, arrivals_per_hour, workers, hourly_rate, reserve_fraction=.25):
    """Illustrative one-job-per-worker plan; inputs are not cloud price discovery."""
    # Compute `values` from `[measured_job_s,arrivals_per_hour,hourly_rate,reserv...`
    values=[measured_job_s,arrivals_per_hour,hourly_rate,reserve_fraction]
    # Guard input contract (`any((not math.isfinite(x) for x in values)) or measured_job_s <= 0 or arrivals_per_hour < 0 or (hourly_rate < 0)`) and fail fast if violated.
    if any(not math.isfinite(x) for x in values) or measured_job_s <= 0 or arrivals_per_hour < 0 or hourly_rate < 0:
        raise ValueError('invalid finite capacity inputs')
    # Guard input contract (`not isinstance(workers, int) or isinstance(workers, bool) or workers < 1 or (not 0 <= reserve_fraction < 1)`) and fail fast if violated.
    if not isinstance(workers,int) or isinstance(workers,bool) or workers < 1 or not 0 <= reserve_fraction < 1:
        raise ValueError('invalid workers or reserve')
    # Compute `offered` from `arrivals_per_hour*measured_job_s/3600`
    offered=arrivals_per_hour*measured_job_s/3600
    # Compute `load` from `offered/workers`
    load=offered/workers
    # Run `max` to compute `required`.
    required=max(1,math.ceil(offered/(1-reserve_fraction)))
    # Return `dict(offered_worker_hours_per_hour=offered, load_fraction=load, minimum_workers_with_reserve=required, hourly_budget=workers * hourly_rate, hypothetical_cost_per_job=measured_job_s * hourly_rate / 3600, within_reserve=load <= 1 - reserve_fraction)` to the caller.
    return dict(offered_worker_hours_per_hour=offered,load_fraction=load,
                minimum_workers_with_reserve=required,hourly_budget=workers*hourly_rate,
                hypothetical_cost_per_job=measured_job_s*hourly_rate/3600,
                within_reserve=load <= 1-reserve_fraction)

# Measure jobs before evaluating assumptions
# Step 5 — Measure jobs before evaluating assumptions: Every duration comes from a completed child; arrival and price...
samples=[]
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='ops-capacity-') as folder:
    # Loop over `steps` in `[8, 32, 64]`:
    for steps in [8,32,64]:
        # Read or serialize artifact data on disk (`run`).
        run=launch(Path(folder)/str(steps),dict(steps=steps))
        # Assert invariant `run['status']=='completed'` holds
        assert run['status']=='completed'
        # Append the current step result to `samples`.
        samples.append((steps,run,summarize(run)))
# Run `max` to compute `measured`.
measured=max(run['wall_s'] for _,run,_ in samples)
# Run `capacity` to compute `plan`.
plan=capacity(measured_job_s=measured,arrivals_per_hour=600,workers=2,hourly_rate=1.5,reserve_fraction=.25)
# Assert invariant `plan['hourly_budget']==3.` holds
assert plan['hourly_budget']==3.
# Print the observed values to compare against the expected result.
print('measured local job seconds:',[r['wall_s'] for _,r,_ in samples])
# Print diagnostic summary of the computed outputs.
print('illustrative capacity plan:',plan)
# Print diagnostic summary of the computed outputs.
print('assumptions: 600 arrivals/hour, 2 workers, hypothetical 1.50 per worker-hour, 25% reserve')

# Complete runnable example (operations-05)
"""Bounded local JAX subprocess supervisor and lifecycle verifier."""
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
    # Compute `path` as `Path(path)`.
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
        if os.path.exists(temporary):
            os.unlink(temporary)

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
        if not isinstance(cfg[name],int) or isinstance(cfg[name],bool) or cfg[name] < 1:
            raise ValueError(name+' must be a positive integer')
    # Guard input contract (`cfg['batch_size'] > 64 or not 0 <= cfg['momentum'] < 1 or (not 0 < cfg['learning_rate'] < 1)`) and fail fast if violated.
    if cfg['batch_size'] > 64 or not 0 <= cfg['momentum'] < 1 or not 0 < cfg['learning_rate'] < 1:
        raise ValueError('invalid training configuration')
    # Return `cfg` to the caller.
    return cfg


# Function `launch(root, config, timeout)` implementing this stage's computation:
def launch(root, config=None, timeout=10.):
    # Compute `root` as `Path(root)`.
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    # Run `default_config` to compute `cfg`.
    cfg=default_config(**(config or {}))
    # Guard input contract (`timeout <= 0`) and fail fast if violated.
    if timeout <= 0:
        raise ValueError('timeout must be positive')
    # Run `atomic_json` to perform the next check or state transition.
    atomic_json(root/'config.json',cfg)
    # Compute `worker` as `root/'worker.py'`.
    worker=root/'worker.py'
    worker.write_text(WORKER)
    # Configure environment variable before initializing the runtime.
    env=dict(os.environ,JAX_PLATFORMS=cfg['backend'],PYTHONUNBUFFERED='1')
    # Record execution timing or profiler trace in `started`.
    started=time.perf_counter()
    # Read or serialize artifact data on disk (`process`).
    process=subprocess.Popen([sys.executable,str(worker),str(root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
    # Compute `timed_out` as `False`.
    timed_out=False
    try:
        stdout,stderr=process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True
        process.terminate()
        try:
            stdout,stderr=process.communicate(timeout=1.)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout,stderr=process.communicate(timeout=1.)
    # Record execution timing or profiler trace in `wall`.
    wall=time.perf_counter()-started
    # Compute `events` from `[]`
    events=[]
    malformed=[]
    # Loop over `line` in `stdout.splitlines()`:
    for line in stdout.splitlines():
        # Branch on condition `not line.strip()`:
        if not line.strip():
            continue
        try:
            event=json.loads(line)
            if not isinstance(event,dict) or 'event' not in event:
                raise ValueError('not an event')
            events.append(event)
        except (ValueError,TypeError):
            malformed.append(line)
    # Compute `status` from `'timed_out' if timed_out else 'completed' if process...`
    status='timed_out' if timed_out else 'completed' if process.returncode == 0 and not malformed and events and events[-1]['event']=='completed' else 'failed'
    # Compute deterministic cryptographic digest `result` for provenance verification.
    result=dict(status=status,returncode=process.returncode,wall_s=wall,events=events,stderr=stderr,
                worker_hash=hashlib.sha256(WORKER.encode()).hexdigest(),config=cfg,malformed_stdout=malformed)
    # Run `atomic_json` to perform the next check or state transition.
    atomic_json(root/'run.json',result)
    # Return `result` to the caller.
    return result

# Step 4 — Add metric summaries and capacity arithmetic: Keep measured quantities separate from invented planning inputs...
def summarize(result):
    # Compute `updates` from `[e for e in result['events'] if e['event']=='progress']`
    updates=[e for e in result['events'] if e['event']=='progress']
    # Compute `checkpoints` from `[e['step'] for e in result['events'] if e['event']==...`
    checkpoints=[e['step'] for e in result['events'] if e['event']=='checkpoint']
    # Run `sum` to compute `count`.
    # Run `sum` to compute `update_s`.
    count=sum(e['examples'] for e in updates)
    update_s=sum(e['update_s'] for e in updates)
    # Run `max` to compute `latest`.
    latest=max((e['step'] for e in updates),default=0)
    # Run `max` to compute `checkpoint`.
    checkpoint=max(checkpoints,default=0)
    # Return `dict(status=result['status'], updates=len(updates), examples=count, last_step=latest, checkpoint_step=checkpoint, uncheckpointed_updates=max(0, latest - checkpoint), update_s=update_s, update_examples_per_s=count / update_s if update_s else 0.0, job_examples_per_s=count / result['wall_s'] if result['wall_s'] else 0.0, update_duty_fraction=update_s / result['wall_s'] if result['wall_s'] else 0.0)` to the caller.
    return dict(status=result['status'],updates=len(updates),examples=count,last_step=latest,checkpoint_step=checkpoint,
                uncheckpointed_updates=max(0,latest-checkpoint),update_s=update_s,
                update_examples_per_s=count/update_s if update_s else 0.,
                job_examples_per_s=count/result['wall_s'] if result['wall_s'] else 0.,
                update_duty_fraction=update_s/result['wall_s'] if result['wall_s'] else 0.)

# Function `capacity(measured_job_s, arrivals_per_hour, workers, hourly_rate, ...)` implementing this stage's computation:
def capacity(measured_job_s, arrivals_per_hour, workers, hourly_rate, reserve_fraction=.25):
    """Illustrative one-job-per-worker plan; inputs are not cloud price discovery."""
    # Compute `values` from `[measured_job_s,arrivals_per_hour,hourly_rate,reserv...`
    values=[measured_job_s,arrivals_per_hour,hourly_rate,reserve_fraction]
    # Guard input contract (`any((not math.isfinite(x) for x in values)) or measured_job_s <= 0 or arrivals_per_hour < 0 or (hourly_rate < 0)`) and fail fast if violated.
    if any(not math.isfinite(x) for x in values) or measured_job_s <= 0 or arrivals_per_hour < 0 or hourly_rate < 0:
        raise ValueError('invalid finite capacity inputs')
    # Guard input contract (`not isinstance(workers, int) or isinstance(workers, bool) or workers < 1 or (not 0 <= reserve_fraction < 1)`) and fail fast if violated.
    if not isinstance(workers,int) or isinstance(workers,bool) or workers < 1 or not 0 <= reserve_fraction < 1:
        raise ValueError('invalid workers or reserve')
    # Compute `offered` from `arrivals_per_hour*measured_job_s/3600`
    offered=arrivals_per_hour*measured_job_s/3600
    # Compute `load` from `offered/workers`
    load=offered/workers
    # Run `max` to compute `required`.
    required=max(1,math.ceil(offered/(1-reserve_fraction)))
    # Return `dict(offered_worker_hours_per_hour=offered, load_fraction=load, minimum_workers_with_reserve=required, hourly_budget=workers * hourly_rate, hypothetical_cost_per_job=measured_job_s * hourly_rate / 3600, within_reserve=load <= 1 - reserve_fraction)` to the caller.
    return dict(offered_worker_hours_per_hour=offered,load_fraction=load,
                minimum_workers_with_reserve=required,hourly_budget=workers*hourly_rate,
                hypothetical_cost_per_job=measured_job_s*hourly_rate/3600,
                within_reserve=load <= 1-reserve_fraction)

# Step 5 — Measure jobs before evaluating assumptions: Every duration comes from a completed child; arrival and price...
samples=[]
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='ops-capacity-') as folder:
    # Loop over `steps` in `[8, 32, 64]`:
    for steps in [8,32,64]:
        # Read or serialize artifact data on disk (`run`).
        run=launch(Path(folder)/str(steps),dict(steps=steps))
        # Assert invariant `run['status']=='completed'` holds
        assert run['status']=='completed'
        # Append the current step result to `samples`.
        samples.append((steps,run,summarize(run)))
# Run `max` to compute `measured`.
measured=max(run['wall_s'] for _,run,_ in samples)
# Run `capacity` to compute `plan`.
plan=capacity(measured_job_s=measured,arrivals_per_hour=600,workers=2,hourly_rate=1.5,reserve_fraction=.25)
# Assert invariant `plan['hourly_budget']==3.` holds
assert plan['hourly_budget']==3.
# Print the observed values to compare against the expected result.
print('measured local job seconds:',[r['wall_s'] for _,r,_ in samples])
# Print diagnostic summary of the computed outputs.
print('illustrative capacity plan:',plan)
# Print diagnostic summary of the computed outputs.
print('assumptions: 600 arrivals/hour, 2 workers, hypothetical 1.50 per worker-hour, 25% reserve')

# Figure data experiment
# Compute figure data for: Short jobs spend time outside the update loop
# Compute `visual_data` from `{'panels':[{'kind':'bar','labels':[str(n) for n,_,_ ...`
visual_data={'panels':[{'kind':'bar','labels':[str(n) for n,_,_ in samples],'xlabel':'updates in fresh local process','ylabel':'complete job seconds','title':'Complete process lifetime','series':[{'label':'wall time','y':[r['wall_s'] for _,r,_ in samples]}]},{'kind':'bar','labels':[str(n) for n,_,_ in samples],'xlabel':'updates in fresh local process','ylabel':'synchronized update seconds','title':'Training updates only — separate vertical scale','series':[{'label':'update time','y':[st['update_s'] for _,_,st in samples]}]}]}

# Experiment: Check units with known arithmetic
# Experiment — Check units with known arithmetic: This independent arithmetic checks the calculator without...
known=capacity(90.,80.,4,2.,.25)
# Assert invariant `known['offered_worker_hours_per_hour']==2.` holds
assert known['offered_worker_hours_per_hour']==2.
# Assert invariant `known['load_fraction']==.5` holds
assert known['load_fraction']==.5
# Assert invariant `known['minimum_workers_with_reserve']==3` holds
assert known['minimum_workers_with_reserve']==3
# Assert invariant `known['hypothetical_cost_per_job']==.05` holds
assert known['hypothetical_cost_per_job']==.05
# Assert invariant `known['hourly_budget']==8.` holds
assert known['hourly_budget']==8.

# Experiment: Double the arrivals without changing the job
# Experiment — Double the arrivals without changing the job: Demand changes the required capacity; it does not automatically...
one=capacity(measured,600,2,1.5,.25)
# Run `capacity` to compute `two`.
two=capacity(measured,1200,2,1.5,.25)
# Assert that `abs(two['load_fraction']-2*one['load_fraction'])<1e-12`.
assert abs(two['load_fraction']-2*one['load_fraction'])<1e-12
# Assert invariant `two['hypothetical_cost_per_job']==one['hypothetical_cost_per_job']` holds
assert two['hypothetical_cost_per_job']==one['hypothetical_cost_per_job']
# Print the observed values to compare against the expected result.
print('nominal load at two demand scenarios:',one['load_fraction'],two['load_fraction'])

# Reference solution. Try the exercise before reading this.
# Exercise solution: Using the measured service-time input, calculate the minimum worker...
changed=capacity(measured,1200,2,1.5,.4)
# Compute `required` from `changed['minimum_workers_with_reserve']`
required=changed['minimum_workers_with_reserve']
# Compute `demand` from `1200*measured/3600`
demand=1200*measured/3600
# Assert invariant `demand/required <= .6 + 1e-12` holds
assert demand/required <= .6 + 1e-12
# Branch on condition `required > 1`:
if required>1: assert demand/(required-1) > .6
# Print the observed values to compare against the expected result.
print('required workers under changed hypothetical reserve:',required)

# Reference practice: Account for failed attempts
# Account for failed attempts (Transfer / diagnosis): Failure overhead belongs in the budget even when it does not...
per_attempt=measured*1.5/3600
# Compute `per_completion` from `125*per_attempt/100`
per_completion=125*per_attempt/100
# Assert that `abs(per_completion-1.25*per_attempt)<1e-12`.
assert abs(per_completion-1.25*per_attempt)<1e-12

# Reference practice: Reject impossible capacity inputs
# Reject impossible capacity inputs (Transfer / diagnosis): Input validation makes the assumptions visible and prevents...
# Iterate over `change` to step through the computation:
for change in [dict(reserve_fraction=1.),dict(workers=0),dict(measured_job_s=float('nan'))]:
    # Evaluate `measured_job_s=measured, arrivals_per_hour=600, workers=2, hourly_rate=1.5` and convert the result into Python scalar/collection `inputs`.
    # Compute `inputs` as `dict(measured_job_s=measured,arrivals_per_hour=600,workers=2,hourly_rate=1.5)`.
    inputs=dict(measured_job_s=measured,arrivals_per_hour=600,workers=2,hourly_rate=1.5)
    inputs.update(change)
    # Run the boundary check and catch the expected exception:
    try:capacity(**inputs)
    except ValueError:pass
    else:raise AssertionError('invalid plan accepted')
print("PASS: operations-05")
