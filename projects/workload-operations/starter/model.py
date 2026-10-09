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
    # Key APIs to use: `hashlib.sha256`, `json.dumps`, `encode`, `hexdigest`
    # Step 1: Return `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()` to the caller.
    raise NotImplementedError('Canonical content identity')

def atomic_json(path, value):
    """Write/flush/fsync a sibling temporary file; replace the selected path."""
    # Key APIs to use: `disk`, `Path`, `parent.mkdir`, `tempfile.mkstemp`, `os.fdopen`
    # Step 1: Read or serialize artifact data on disk (`path`).
    # Step 2: Execute the next step of the computation.
    # Step 3: Run `tempfile.mkstemp` to compute `(fd, temporary)`.
    raise NotImplementedError('Atomic local selection')

def default_config(**changes):
    """Return validated defaults plus changes. See README for default values."""
    # Key APIs to use: `cfg.update`, `in`, `contract`, `or`
    # Step 1: Evaluate `seed=3, learning_rate=0.04, momentum=0.8, batch_size=8, steps=8, checkpoint_every=2, backend='cpu', min_devices=1, resume=False, fail_after=None, stall_at=None, stall_seconds=10.0` and convert the result into Python scalar/collection `cfg`.
    # Step 2: Update state in place with the new values.
    # Step 3: Loop over `name` in `('steps', 'checkpoint_every', 'batch_size', 'min_devices')`:
    # Step 4: Inside block: Guard input contract (`not isinstance(cfg[name], int) or isinstance(cfg[name], bool) or cfg[name] < 1`) and fail fast if violated.
    # Step 5: Guard input contract (`cfg['batch_size'] > 64 or not 0 <= cfg['momentum'] < 1 or (not 0 < cfg['learning_rate'] < 1)`) and fail fast if violated.
    # Step 6: Return `cfg` to the caller.
    raise NotImplementedError('Validate the workload contract')

def launch(root, config=None, timeout=10.):
    """Write worker/config, run a real child, capture events, and reap on timeout.

    Return status, returncode, wall_s, events, stderr, worker_hash, config and
    malformed_stdout; atomically retain the same record in root/run.json.
    """
    # Key APIs to use: `disk`, `Path`, `root.mkdir`, `default_config`, `contract`
    # Step 1: Read or serialize artifact data on disk (`root`).
    # Step 2: Execute the next step of the computation.
    # Step 3: Run `default_config` to compute `cfg`.
    # Step 4: Guard input contract (`timeout <= 0`) and fail fast if violated.
    # Step 5: Run `atomic_json` to perform the next check or state transition.
    # Step 6: Evaluate `worker` from the current inputs and state.
    raise NotImplementedError('Supervise the process, not an invented status')

def summarize(result):
    """Return updates/examples/last_step/checkpoint_step/uncheckpointed_updates,
    update_s, update_examples_per_s, job_examples_per_s, update_duty_fraction
    and status. Derive every value from actual events and wall_s.
    """
    # Key APIs to use: `sum`, `max`
    # Step 1: Evaluate `updates` from the current inputs and state.
    # Step 2: Evaluate `checkpoints` from the current inputs and state.
    # Step 3: Run `sum` to compute `count`.
    # Step 4: Run `sum` to compute `update_s`.
    # Step 5: Run `max` to compute `latest`.
    # Step 6: Run `max` to compute `checkpoint`.
    raise NotImplementedError('Work and timing boundaries')

def publish(store, artifact):
    """Write artifacts/<digest>.json without changing active.json; return digest."""
    # Key APIs to use: `disk`, `Path`, `digest`, `atomic_json`
    # Step 1: Read or serialize artifact data on disk (`store`).
    # Step 2: Run `digest` to compute `identifier`.
    # Step 3: Run `atomic_json` to perform the next check or state transition.
    # Step 4: Return `identifier` to the caller.
    raise NotImplementedError('Publish independently of activation')

def read_artifact(store, identifier):
    """Require a 64-character lowercase hex ID and verify payload digest."""
    # Key APIs to use: `contract`, `any`, `disk`, `json.loads`, `Path`
    # Step 1: Guard input contract (`len(identifier) != 64 or any((c not in '0123456789abcdef' for c in identifier))`) and fail fast if violated.
    # Step 2: Read or serialize artifact data on disk (`artifact`).
    # Step 3: Guard input contract (`digest(artifact) != identifier`) and fail fast if violated.
    # Step 4: Return `artifact` to the caller.
    raise NotImplementedError('Read with integrity checks')

def artifact_metrics(artifact):
    """Evaluate params on x=(-1.5,-.37,.22,1.5), y=2*x+1; MSE limit 1."""
    # Key APIs to use: `artifact.get`, `get`, `contract`, `any`, `math.isfinite`
    # Step 1: Run `artifact.get` to compute `params`.
    # Step 2: Guard input contract (`len(params) != 2 or any((not math.isfinite(float(v)) for v in params))`) and fail fast if violated.
    # Step 3: Evaluate `inputs` from the current inputs and state.
    # Step 4: Run `sum` to compute `mse`.
    # Step 5: Return `dict(held_out_mse=mse, acceptance_limit=1.0, passed=mse <= 1.0)` to the caller.
    raise NotImplementedError('Compute the actual release quality gate')

def activate(store, identifier):
    """Verify digest, provenance, declared validation and actual held-out MSE, then select.
    active.json contains current and previous content IDs.
    """
    # Key APIs to use: `read_artifact`, `contract`, `artifact.get`, `get`, `any`
    # Step 1: Run `read_artifact` to compute `artifact`.
    # Step 2: Guard input contract (`artifact.get('validation', {}).get('passed') is not True`) and fail fast if violated.
    # Step 3: Evaluate `required` from the current inputs and state.
    # Step 4: Guard input contract (`any((name not in artifact for name in required))`) and fail fast if violated.
    # Step 5: Guard input contract (`not artifact_metrics(artifact)['passed']`) and fail fast if violated.
    # Step 6: Evaluate `previous` from the current inputs and state.
    raise NotImplementedError('Gate before pointer mutation')

def rollback(store):
    """Select previous through the same checked activation path."""
    # Key APIs to use: `disk`, `json.loads`, `Path`, `read_text`, `contract`
    # Step 1: Read or serialize artifact data on disk (`pointer`).
    # Step 2: Guard input contract (`not pointer.get('previous')`) and fail fast if violated.
    # Step 3: Return `activate(store, pointer['previous'])` to the caller.
    raise NotImplementedError('Checked rollback')

def capacity(measured_job_s, arrivals_per_hour, workers, hourly_rate, reserve_fraction=.25):
    """Return offered_worker_hours_per_hour, load_fraction,
    minimum_workers_with_reserve, hourly_budget, hypothetical_cost_per_job,
    within_reserve. Validate finite input and document assumptions.
    """
    # Key APIs to use: `contract`, `any`, `math.isfinite`, `or`, `max`
    # Step 1: Evaluate `values` from the current inputs and state.
    # Step 2: Guard input contract (`any((not math.isfinite(x) for x in values)) or measured_job_s <= 0 or arrivals_per_hour < 0 or (hourly_rate < 0)`) and fail fast if violated.
    # Step 3: Guard input contract (`not isinstance(workers, int) or isinstance(workers, bool) or workers < 1 or (not 0 <= reserve_fraction < 1)`) and fail fast if violated.
    # Step 4: Evaluate `offered` from the current inputs and state.
    # Step 5: Evaluate `load` from the current inputs and state.
    # Step 6: Run `max` to compute `required`.
    raise NotImplementedError('Dimensional capacity arithmetic')
