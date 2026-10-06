# Health, logs, and workload observability

Phase 16: Workload operations · about 85 minutes · CPU

## What you will be able to do

- Build useful summaries from structured events emitted by a real worker.
- Define throughput with explicit numerator, denominator and synchronization boundary.
- Estimate uncheckpointed work from progress and checkpoint events.

## The problem

A process is alive and prints messages, yet useful training may have stopped. What signals distinguish real progress from noise, and what could be lost if the process fails now? We will measure synchronized updates, count examples and compare progress with the latest committed checkpoint using events emitted by actual runs.

## The idea

Observability should answer a decision. Progress step and processed examples tell us whether work advances. Update duration estimates the speed of the compiled training operation; total job duration includes startup and operational overhead. Checkpoint step tells us how much completed work remains uncommitted. We keep these quantities separate instead of calling every ratio utilization.

## An event needs meaning and units

Each event has a monotonic timestamp and an event name. A progress event adds completed step, scalar loss, examples processed, synchronized update seconds and a hash of the resulting complete state. Checkpoint events record the step that was actually replaced on disk. A heartbeat would show liveness only; it must not increment examples or claim training progress. Monotonic time is useful for local durations but is not a globally comparable timestamp across unrelated hosts.

## Time completed computation

JAX may return control before an operation finishes. The worker calls block_until_ready on the result before stopping its update timer. That timer includes dispatch and completion of one compiled minibatch update, but excludes JSON encoding, checkpoint writes, process startup and compilation. The ready event records a separate compile/warmup duration. A benchmark that times dispatch alone would understate service time.

## Two throughputs answer different questions

For $12$ updates of $8$ examples, the numerator is $96$ examples. Divide by the sum of synchronized update durations to describe the update path. Divide by supervisor wall time to describe the short job end to end. The latter is lower here because this tiny workload spends much of its lifetime outside the compiled updates. Neither rate should be relabeled as a device hardware utilization percentage.

$$
q_{\mathrm{update}}=\frac{N}{\sum_i t_i},\qquad q_{\mathrm{job}}=\frac{N}{T_{\mathrm{wall}}}
$$

## Freshness is about committed progress

The controlled incident fails after update $5$, while the checkpoint cadence is every $2$ updates. Its last committed checkpoint is step $4$, leaving one completed but uncommitted update. This is a count of potential replay work, not a time-based recovery objective. To express time at risk, use appropriate measured durations and explicitly distinguish completed work from work that might still be in flight.

$$
\mathrm{uncommitted\ updates}=\max(0,\mathrm{last\ progress\ step}-\mathrm{checkpoint\ step})
$$

## A declining loss is not a health guarantee

The fixture fits a line with minibatch momentum updates. Loss may rise between batches because different examples are sampled, and momentum can overshoot. A lower recent loss does not prove input freshness, checkpoint durability or correct evaluation. Conversely, a noisy loss sequence does not prove a stalled process. Diagnose loss behavior using the optimization lessons; diagnose missing progress using event and process evidence.

## Scope of this collector

This bounded lab captures stdout and stderr in memory and analyzes events after the child exits. That is appropriate for a few dozen events. A long-running service needs streamed, bounded logs, a durable sink, timestamps aligned for its fleet, and monitored queue/backpressure behavior. The lesson provides measurement contracts and actual failure evidence, not an operational telemetry backend.

## Create the local artifact helpers

Create main.py with this block. Run python3 main.py in the CPU course environment.

```python
"""Bounded local JAX workload operations. No scheduler, cloud, or GPU emulator."""
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    """Replace one local JSON file only after its bytes have been flushed."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
```

The only filesystem writes are to the caller-selected run directory. Stable JSON encoding makes hashes reproducible.

## Define the supervised worker

Append this block to main.py. Run python3 main.py in the CPU course environment.

```python
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
```

The string is a real Python program launched in a separate process. It emits structured events, synchronizes JAX updates and preserves complete state.

## Add bounded launch and result collection

Append this block to main.py. Run python3 main.py in the CPU course environment.

```python
def default_config(**changes):
    cfg=dict(seed=3,learning_rate=.04,momentum=.8,batch_size=8,steps=8,
             checkpoint_every=2,backend='cpu',min_devices=1,resume=False,
             fail_after=None,stall_at=None,stall_seconds=10.)
    cfg.update(changes)
    for name in ('steps','checkpoint_every','batch_size','min_devices'):
        if not isinstance(cfg[name],int) or isinstance(cfg[name],bool) or cfg[name] < 1: raise ValueError(name+' must be a positive integer')
    if cfg['batch_size'] > 64 or not 0 <= cfg['momentum'] < 1 or not 0 < cfg['learning_rate'] < 1:
        raise ValueError('invalid training configuration')
    return cfg


def launch(root, config=None, timeout=10.):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    cfg=default_config(**(config or {}))
    if timeout <= 0: raise ValueError('timeout must be positive')
    atomic_json(root/'config.json',cfg)
    worker=root/'worker.py';worker.write_text(WORKER)
    env=dict(os.environ,JAX_PLATFORMS=cfg['backend'],PYTHONUNBUFFERED='1')
    started=time.perf_counter()
    process=subprocess.Popen([sys.executable,str(worker),str(root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
    timed_out=False
    try:
        stdout,stderr=process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True;process.terminate()
        try: stdout,stderr=process.communicate(timeout=1.)
        except subprocess.TimeoutExpired:
            process.kill();stdout,stderr=process.communicate(timeout=1.)
    wall=time.perf_counter()-started
    events=[]; malformed=[]
    for line in stdout.splitlines():
        if not line.strip(): continue
        try:
            event=json.loads(line)
            if not isinstance(event,dict) or 'event' not in event: raise ValueError('not an event')
            events.append(event)
        except (ValueError,TypeError): malformed.append(line)
    status='timed_out' if timed_out else 'completed' if process.returncode == 0 and not malformed and events and events[-1]['event']=='completed' else 'failed'
    result=dict(status=status,returncode=process.returncode,wall_s=wall,events=events,stderr=stderr,
                worker_hash=hashlib.sha256(WORKER.encode()).hexdigest(),config=cfg,malformed_stdout=malformed)
    atomic_json(root/'run.json',result)
    return result
```

The supervisor owns the child handle, captures its output and always waits for termination after a timeout.

## Summarize event boundaries

Append this block to main.py. Run python3 main.py in the CPU course environment.

```python
def summarize(result):
    updates=[e for e in result['events'] if e['event']=='progress']
    checkpoints=[e['step'] for e in result['events'] if e['event']=='checkpoint']
    count=sum(e['examples'] for e in updates);update_s=sum(e['update_s'] for e in updates)
    latest=max((e['step'] for e in updates),default=0)
    checkpoint=max(checkpoints,default=0)
    return dict(status=result['status'],updates=len(updates),examples=count,last_step=latest,checkpoint_step=checkpoint,
                uncheckpointed_updates=max(0,latest-checkpoint),update_s=update_s,
                update_examples_per_s=count/update_s if update_s else 0.,
                job_examples_per_s=count/result['wall_s'] if result['wall_s'] else 0.,
                update_duty_fraction=update_s/result['wall_s'] if result['wall_s'] else 0.)
```

The summary retains work counts, timing boundaries and checkpoint lag separately.

## Run a healthy job and a controlled incident

Append this block to main.py. Run python3 main.py in the CPU course environment.

```python
with tempfile.TemporaryDirectory(prefix='ops-signals-') as folder:
    observed=launch(Path(folder)/'healthy',dict(steps=12))
    interrupted=launch(Path(folder)/'failed',dict(steps=12,fail_after=5))
healthy=summarize(observed); incident=summarize(interrupted)
progress=[e for e in observed['events'] if e['event']=='progress']
assert healthy['examples']==96 and healthy['updates']==12
assert incident['last_step']==5 and incident['checkpoint_step']==4
assert incident['uncheckpointed_updates']==1
assert healthy['job_examples_per_s'] < healthy['update_examples_per_s']
print('healthy:',healthy)
print('incident:',incident)
```

Both summaries are derived from real child output; failures are not invented status rows.

## Run the example

```python
"""Bounded local JAX workload operations. No scheduler, cloud, or GPU emulator."""
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    """Replace one local JSON file only after its bytes have been flushed."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
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

def default_config(**changes):
    cfg=dict(seed=3,learning_rate=.04,momentum=.8,batch_size=8,steps=8,
             checkpoint_every=2,backend='cpu',min_devices=1,resume=False,
             fail_after=None,stall_at=None,stall_seconds=10.)
    cfg.update(changes)
    for name in ('steps','checkpoint_every','batch_size','min_devices'):
        if not isinstance(cfg[name],int) or isinstance(cfg[name],bool) or cfg[name] < 1: raise ValueError(name+' must be a positive integer')
    if cfg['batch_size'] > 64 or not 0 <= cfg['momentum'] < 1 or not 0 < cfg['learning_rate'] < 1:
        raise ValueError('invalid training configuration')
    return cfg


def launch(root, config=None, timeout=10.):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    cfg=default_config(**(config or {}))
    if timeout <= 0: raise ValueError('timeout must be positive')
    atomic_json(root/'config.json',cfg)
    worker=root/'worker.py';worker.write_text(WORKER)
    env=dict(os.environ,JAX_PLATFORMS=cfg['backend'],PYTHONUNBUFFERED='1')
    started=time.perf_counter()
    process=subprocess.Popen([sys.executable,str(worker),str(root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
    timed_out=False
    try:
        stdout,stderr=process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True;process.terminate()
        try: stdout,stderr=process.communicate(timeout=1.)
        except subprocess.TimeoutExpired:
            process.kill();stdout,stderr=process.communicate(timeout=1.)
    wall=time.perf_counter()-started
    events=[]; malformed=[]
    for line in stdout.splitlines():
        if not line.strip(): continue
        try:
            event=json.loads(line)
            if not isinstance(event,dict) or 'event' not in event: raise ValueError('not an event')
            events.append(event)
        except (ValueError,TypeError): malformed.append(line)
    status='timed_out' if timed_out else 'completed' if process.returncode == 0 and not malformed and events and events[-1]['event']=='completed' else 'failed'
    result=dict(status=status,returncode=process.returncode,wall_s=wall,events=events,stderr=stderr,
                worker_hash=hashlib.sha256(WORKER.encode()).hexdigest(),config=cfg,malformed_stdout=malformed)
    atomic_json(root/'run.json',result)
    return result

def summarize(result):
    updates=[e for e in result['events'] if e['event']=='progress']
    checkpoints=[e['step'] for e in result['events'] if e['event']=='checkpoint']
    count=sum(e['examples'] for e in updates);update_s=sum(e['update_s'] for e in updates)
    latest=max((e['step'] for e in updates),default=0)
    checkpoint=max(checkpoints,default=0)
    return dict(status=result['status'],updates=len(updates),examples=count,last_step=latest,checkpoint_step=checkpoint,
                uncheckpointed_updates=max(0,latest-checkpoint),update_s=update_s,
                update_examples_per_s=count/update_s if update_s else 0.,
                job_examples_per_s=count/result['wall_s'] if result['wall_s'] else 0.,
                update_duty_fraction=update_s/result['wall_s'] if result['wall_s'] else 0.)

with tempfile.TemporaryDirectory(prefix='ops-signals-') as folder:
    observed=launch(Path(folder)/'healthy',dict(steps=12))
    interrupted=launch(Path(folder)/'failed',dict(steps=12,fail_after=5))
healthy=summarize(observed); incident=summarize(interrupted)
progress=[e for e in observed['events'] if e['event']=='progress']
assert healthy['examples']==96 and healthy['updates']==12
assert incident['last_step']==5 and incident['checkpoint_step']==4
assert incident['uncheckpointed_updates']==1
assert healthy['job_examples_per_s'] < healthy['update_examples_per_s']
print('healthy:',healthy)
print('incident:',incident)
```

Expected: The healthy run records $96$ processed examples. The failure occurs at step $5$, with checkpoint step $4$, so one completed update must be replayed.

## Observed progress and committed progress diverge before failure

**Predict:** Will the checkpoint line always equal the progress line?

![Observed progress and committed progress diverge before failure](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis is completed update number for the failed run. The vertical axis is step index. The progress line reaches $5$, while the checkpoint line advances at steps $2$ and $4$, ending at $4$. The vertical gap at the final point is one update. These values come from actual emitted events, not a hypothetical fleet trace.

### Connect it to the computation

The staircase records when state becomes available for recovery. A gap is expected between checkpoints; it becomes operationally relevant when the process fails. Resuming from step $4$ repeats update $5$ using its restored optimizer and random state. This plot does not show elapsed recovery time or prove power-loss durability.

```python
committed=0;points=[];saved=[]
for event in interrupted['events']:
    if event['event']=='checkpoint': committed=event['step']
    if event['event']=='progress':
        points.append(event['step']);saved.append(committed)
# At each completed step include a checkpoint emitted immediately afterward.
for i,step_number in enumerate(points):
    saved[i]=max([e['step'] for e in interrupted['events'] if e['event']=='checkpoint' and e['step']<=step_number],default=0)
visual_data={'kind':'line','x':points,'xlabel':'completed update number','ylabel':'step index','series':[{'label':'observed progress','y':points},{'label':'committed checkpoint','y':saved}]}
```

## Recorded reference execution

CPU run: 2026-10-06T01:27:03.401245+00:00. JAX 0.9.2.

```text
healthy: {'status': 'completed', 'updates': 12, 'examples': 96, 'last_step': 12, 'checkpoint_step': 12, 'uncheckpointed_updates': 0, 'update_s': 0.0005469173192977905, 'update_examples_per_s': 175529.2740102989, 'job_examples_per_s': 157.8143481242871, 'update_duty_fraction': 0.0008990770856548272}
incident: {'status': 'failed', 'updates': 5, 'examples': 40, 'last_step': 5, 'checkpoint_step': 4, 'uncheckpointed_updates': 1, 'update_s': 0.00021733413450419903, 'update_examples_per_s': 184048.40128427767, 'job_examples_per_s': 68.04208061803776, 'update_duty_fraction': 0.00036969666752465424}
healthy: {'status': 'completed', 'updates': 12, 'examples': 96, 'last_step': 12, 'checkpoint_step': 12, 'uncheckpointed_updates': 0, 'update_s': 0.00046362518332898617, 'update_examples_per_s': 207063.81674672506, 'job_examples_per_s': 161.20108241721204, 'update_duty_fraction': 0.0007785091811303223}
incident: {'status': 'failed', 'updates': 5, 'examples': 40, 'last_step': 5, 'checkpoint_step': 4, 'uncheckpointed_updates': 1, 'update_s': 0.0002401240635663271, 'update_examples_per_s': 166580.55592562965, 'job_examples_per_s': 68.72298729861558, 'update_duty_fraction': 0.0004125510742640164}
examples / synchronized seconds: 96 0.00046362518332898617
measured update duty fraction, not hardware utilization: 0.0007785091811303223
PASS: operations-02

```

## Recompute throughput independently

**Predict before running:** Should averaging per-step rates equal total examples divided by total time?

```python
total_examples=sum(e['examples'] for e in progress)
total_time=sum(e['update_s'] for e in progress)
assert abs(total_examples/total_time-healthy['update_examples_per_s']) < 1e-6
assert all(e['update_s']>0 for e in progress)
print('examples / synchronized seconds:',total_examples,total_time)
```

**Expected:** The summary matches the independently accumulated numerator and denominator.

Summing work and time weights slow steps appropriately; an unweighted mean of rates generally answers a different question.

## Locate the incident boundary

**Predict before running:** Is the last successful update necessarily present in the checkpoint?

```python
names=[e['event'] for e in interrupted['events']]
assert names[-1]=='failed'
assert [e['step'] for e in interrupted['events'] if e['event']=='checkpoint']==[2,4]
assert [e['step'] for e in interrupted['events'] if e['event']=='progress'][-1]==5
```

**Expected:** The event order exposes exactly one completed update after the latest checkpoint.

A progress log is not a committed state artifact. Recovery must select the committed boundary.

## Make it yours

Compute the fraction of supervisor wall time spent inside synchronized updates, and state precisely what that fraction excludes.

<details><summary>Reference solution</summary>

```python
duty=sum(e['update_s'] for e in progress)/observed['wall_s']
assert 0 < duty < 1
assert abs(duty-healthy['update_duty_fraction']) < 1e-9
print('measured update duty fraction, not hardware utilization:',duty)
```

</details>

## Catch an average-of-rates error

**Transfer / diagnosis**

Two updates process $8$ examples each in $1$ and $3$ seconds. Compare the average rate with the rate of all completed work.

<details><summary>Hint</summary>

Sum examples and seconds first.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
durations=[1.,3.];counts=[8,8]
wrong=sum(n/t for n,t in zip(counts,durations))/2
right=sum(counts)/sum(durations)
assert abs(wrong-16/3)<1e-12 and right==4.
```

The short step gets excessive influence when rates are averaged without accounting for their duration.

</details>

## Change the checkpoint cadence

**Transfer / diagnosis**

Run with checkpoint cadence $3$ and fail after update $5$. Predict the committed step and replay count.

<details><summary>Hint</summary>

Expect the most recent multiple of the checkpoint interval before the failure.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
with tempfile.TemporaryDirectory() as folder:
    changed=launch(folder,dict(checkpoint_every=3,fail_after=5))
stats=summarize(changed)
assert stats['checkpoint_step']==3 and stats['uncheckpointed_updates']==2
```

The run tests a changed operational policy rather than assuming every failure loses exactly one update.

</details>

## Check your understanding

What does update duty fraction establish in this lab?

1. GPU compute-unit occupancy
2. Cluster utilization across hosts
3. The fraction of this job’s wall time inside the measured synchronized update boundary

<details><summary>Answer and explanation</summary>

The fraction of this job’s wall time inside the measured synchronized update boundary

The numerator and denominator describe an application timing boundary. They do not measure device occupancy, memory bandwidth or fleet allocation.

</details>

## Diagnose the result

If throughput is implausibly high, check synchronization and whether examples were counted twice. If checkpoint freshness looks perfect during a failure, verify that checkpoint events are emitted only after replacement completes. If timestamps appear out of order across hosts, do not subtract unrelated monotonic clocks.

## Carry forward

- Choose metrics that answer a concrete operational decision.
- Specify the timing boundary before reporting a throughput or utilization-like ratio.

## Keep your evidence

Keep raw structured events, observed work counts, synchronized timing units and checkpoint lag. Derive job and update throughput with different boundaries.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [Python subprocess management](https://docs.python.org/3/library/subprocess.html)
- [Python os.replace and fsync](https://docs.python.org/3/library/os.html#os.replace)
- [JAX asynchronous dispatch and synchronization](https://docs.jax.dev/en/latest/async_dispatch.html)
- [JAX device discovery](https://docs.jax.dev/en/latest/_autosummary/jax.devices.html)

