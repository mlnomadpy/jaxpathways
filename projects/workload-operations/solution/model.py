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


# Function `default_config()` implementing this stage's computation:
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
    # Initialize list `events` for the stage values.
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
    # Compute `status` as `'timed_out' if timed_out else 'completed' if process.returncode == 0 and not malfo`.
    status='timed_out' if timed_out else 'completed' if process.returncode == 0 and not malformed and events and events[-1]['event']=='completed' else 'failed'
    # Compute deterministic cryptographic digest `result` for provenance verification.
    result=dict(status=status,returncode=process.returncode,wall_s=wall,events=events,stderr=stderr,
                worker_hash=hashlib.sha256(WORKER.encode()).hexdigest(),config=cfg,malformed_stdout=malformed)
    # Run `atomic_json` to perform the next check or state transition.
    atomic_json(root/'run.json',result)
    # Return `result` to the caller.
    return result


# Function `summarize(result)` implementing this stage's computation:
def summarize(result):
    # Initialize list `updates` for the stage values.
    updates=[e for e in result['events'] if e['event']=='progress']
    # Initialize list `checkpoints` for the stage values.
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


# Function `publish(store, artifact)` implementing this stage's computation:
def publish(store, artifact):
    """Content-address one JSON artifact. Publishing does not activate it."""
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
    if digest(artifact) != identifier:
        raise ValueError('artifact digest mismatch')
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
    # Evaluate the compound expression for `inputs`.
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
    # Initialize list `required` for the stage values.
    required=['worker_hash','config_hash','data_hash','state']
    # Guard input contract (`any((name not in artifact for name in required))`) and fail fast if violated.
    if any(name not in artifact for name in required):
        raise ValueError('artifact provenance incomplete')
    # Guard input contract (`not artifact_metrics(artifact)['passed']`) and fail fast if violated.
    if not artifact_metrics(artifact)['passed']:
        raise ValueError('held-out model validation failed')
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


# Function `capacity(measured_job_s, arrivals_per_hour, workers, hourly_rate, ...)` implementing this stage's computation:
def capacity(measured_job_s, arrivals_per_hour, workers, hourly_rate, reserve_fraction=.25):
    """Illustrative one-job-per-worker plan; inputs are not cloud price discovery."""
    # Initialize list `values` for the stage values.
    values=[measured_job_s,arrivals_per_hour,hourly_rate,reserve_fraction]
    # Guard input contract (`any((not math.isfinite(x) for x in values)) or measured_job_s <= 0 or arrivals_per_hour < 0 or (hourly_rate < 0)`) and fail fast if violated.
    if any(not math.isfinite(x) for x in values) or measured_job_s <= 0 or arrivals_per_hour < 0 or hourly_rate < 0:
        raise ValueError('invalid finite capacity inputs')
    # Guard input contract (`not isinstance(workers, int) or isinstance(workers, bool) or workers < 1 or (not 0 <= reserve_fraction < 1)`) and fail fast if violated.
    if not isinstance(workers,int) or isinstance(workers,bool) or workers < 1 or not 0 <= reserve_fraction < 1:
        raise ValueError('invalid workers or reserve')
    # Compute `offered` as `arrivals_per_hour*measured_job_s/3600`.
    offered=arrivals_per_hour*measured_job_s/3600
    # Compute `load` as `offered/workers`.
    load=offered/workers
    # Run `max` to compute `required`.
    required=max(1,math.ceil(offered/(1-reserve_fraction)))
    # Return `dict(offered_worker_hours_per_hour=offered, load_fraction=load, minimum_workers_with_reserve=required, hourly_budget=workers * hourly_rate, hypothetical_cost_per_job=measured_job_s * hourly_rate / 3600, within_reserve=load <= 1 - reserve_fraction)` to the caller.
    return dict(offered_worker_hours_per_hour=offered,load_fraction=load,
                minimum_workers_with_reserve=required,hourly_budget=workers*hourly_rate,
                hypothetical_cost_per_job=measured_job_s*hourly_rate/3600,
                within_reserve=load <= 1-reserve_fraction)
