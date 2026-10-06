"""Workload operations starter. The small worker is supplied and inspectable.
Implement the supervisor, metrics, artifact selection and planning functions.
Do not import the instructor solution. Use only caller-owned temporary folders.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time

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
    target = root/'checkpoint.json'; temporary = root/'checkpoint.pending'
    with temporary.open('w') as stream:
        json.dump(dict(state,checksum=digest(state)),stream,sort_keys=True,allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,target)
    emit('checkpoint',step=state['step'],state_hash=digest(state))
try:
    emit('started',pid=os.getpid(),backend=jax.default_backend(),device_count=jax.device_count())
    if jax.default_backend() != cfg['backend'] or jax.device_count() < cfg['min_devices']:
        raise ValueError('runtime backend/device contract failed')
    x = jnp.linspace(-1.,1.,64); y = 2*x+1
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
        step = saved['step']; params=jnp.array(saved['params']); velocity=jnp.array(saved['velocity']); key=jnp.array(saved['key'],dtype=jnp.uint32)
        if not isinstance(step,int) or isinstance(step,bool) or step < 0 or step > cfg['steps'] or params.shape != (2,) or velocity.shape != (2,) or key.shape != (2,):
            raise ValueError('invalid checkpoint state shape or step')
        if not bool(jnp.all(jnp.isfinite(params))) or not bool(jnp.all(jnp.isfinite(velocity))):
            raise ValueError('nonfinite checkpoint state')
        emit('restored',step=step,state_hash=digest(saved))
    else:
        step=0; params=jnp.zeros(2);velocity=jnp.zeros(2);key=jax.random.PRNGKey(cfg['seed'])
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
    begin=time.perf_counter();warm=compiled(params,velocity,key);jax.block_until_ready(warm)
    emit('ready',compile_warmup_s=time.perf_counter()-begin,step=step)
    while step < cfg['steps']:
        if cfg.get('stall_at') == step:
            emit('stall_injected',step=step)
            time.sleep(cfg.get('stall_seconds',10.))
        begin=time.perf_counter()
        params,velocity,key,loss=compiled(params,velocity,key);jax.block_until_ready((params,velocity,key,loss))
        elapsed=time.perf_counter()-begin;step+=1
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


def digest(value):
    """SHA-256 of sorted compact JSON with nonfinite values rejected."""
    raise NotImplementedError('Canonical content identity')

def atomic_json(path, value):
    """Write/flush/fsync a sibling temporary file; replace the selected path."""
    raise NotImplementedError('Atomic local selection')

def default_config(**changes):
    """Return validated defaults plus changes. See README for default values."""
    raise NotImplementedError('Validate the workload contract')

def launch(root, config=None, timeout=10.):
    """Write worker/config, run a real child, capture events, and reap on timeout.

    Return status, returncode, wall_s, events, stderr, worker_hash, config and
    malformed_stdout; atomically retain the same record in root/run.json.
    """
    raise NotImplementedError('Supervise the process, not an invented status')

def summarize(result):
    """Return updates/examples/last_step/checkpoint_step/uncheckpointed_updates,
    update_s, update_examples_per_s, job_examples_per_s, update_duty_fraction
    and status. Derive every value from actual events and wall_s.
    """
    raise NotImplementedError('Work and timing boundaries')

def publish(store, artifact):
    """Write artifacts/<digest>.json without changing active.json; return digest."""
    raise NotImplementedError('Publish independently of activation')

def read_artifact(store, identifier):
    """Require a 64-character lowercase hex ID and verify payload digest."""
    raise NotImplementedError('Read with integrity checks')

def artifact_metrics(artifact):
    """Evaluate params on x=(-1.5,-.37,.22,1.5), y=2*x+1; MSE limit 1."""
    raise NotImplementedError('Compute the actual release quality gate')

def activate(store, identifier):
    """Verify digest, provenance, declared validation and actual held-out MSE, then select.
    active.json contains current and previous content IDs.
    """
    raise NotImplementedError('Gate before pointer mutation')

def rollback(store):
    """Select previous through the same checked activation path."""
    raise NotImplementedError('Checked rollback')

def capacity(measured_job_s, arrivals_per_hour, workers, hourly_rate, reserve_fraction=.25):
    """Return offered_worker_hours_per_hour, load_fraction,
    minimum_workers_with_reserve, hourly_budget, hypothetical_cost_per_job,
    within_reserve. Validate finite input and document assumptions.
    """
    raise NotImplementedError('Dimensional capacity arithmetic')
