# Capacity, utilization, and operating cost

Phase 16: Workload operations · about 90 minutes · CPU

## What you will be able to do

- Measure end-to-end local service time at more than one workload length.
- Compute offered load, worker reserve and illustrative cost with consistent units.
- Identify assumptions that prevent a CPU fixture from establishing accelerator fleet capacity.

## The problem

A short local training job finishes quickly. How many workers would a stream of jobs require, and what can this measurement actually tell us about cost? We will measure complete CPU jobs of several lengths, separate update work from startup overhead and apply transparent capacity arithmetic to explicitly hypothetical arrival rates and prices.

## The idea

Capacity planning connects a workload duration to an arrival process and resource policy. A measured job lasting $t$ seconds consumes $t/3600$ worker-hours if one worker is reserved for its lifetime. Multiply by jobs arriving per hour to obtain offered worker demand. Divide by worker count for nominal load, then leave a stated reserve. This first-order plan is useful arithmetic, not a queueing guarantee or a cloud price quote.

## Measure the service boundary you intend to provision

We run jobs with $8$, $32$ and $64$ updates in fresh child processes. Supervisor wall time includes imports, runtime startup, compilation, updates, logs and checkpoints. If a production worker stays warm across many jobs, this measurement overstates repeated startup cost; if production adds queueing or remote data, it understates those delays. Write the boundary alongside the number. We use the maximum of these three local observations as an illustrative conservative input, not a statistical tail estimate.

## Convert seconds into worker demand

Let $\lambda$ be arrivals per hour, $t$ seconds per job and $m$ workers, each running one job at a time. Offered demand is $\lambda t/3600$ worker-hours per hour. Nominal load is this demand divided by $m$. The units cancel to a fraction. A load above one means average demand exceeds nominal capacity under the model; a load below one does not guarantee a latency target.

$$
\rho=\frac{\lambda t}{3600m}
$$

## State the reserve rather than hiding it

For reserve fraction $r$, require nominal load no greater than $1-r$. The smallest integer worker count in this simple model is the ceiling below, with at least one worker in our function. For $90$-second jobs arriving $80$ times an hour, demand is $2$ worker-hours per hour. A $25$% reserve requires at least $3$ workers; four workers would run at nominal load $0.5$.

$$
m_{\min}=\left\lceil\frac{\lambda t}{3600(1-r)}\right\rceil
$$

## A cost number needs a declared price model

We use an invented planning rate of $1.50$ currency units per worker-hour, explicitly not a current cloud price. Reserving two workers for an hour costs $3$ under that model, whether busy or idle. A job’s active-time cost is $pt/3600$ for hourly rate $p$, but dividing the reserved hourly bill by actual completed jobs can produce a different cost per completion. Include idle capacity, failed attempts, storage and transfer charges when those belong to the real budget.

## Do not confuse duty fraction with hardware utilization

The worker records seconds inside synchronized compiled updates. Dividing their sum by job wall time is an application duty fraction. A small value in a tiny job mostly reflects startup and operational overhead. It does not measure CPU core occupancy, accelerator compute utilization, memory-bandwidth pressure or scheduler allocation. Those require their own target-specific measurements. Increasing update count can amortize startup without making each update intrinsically faster.

## Capacity is a distribution, not just a mean

Real arrivals can burst, job durations can vary, failures consume retries, and a worker may be unavailable. Three short measurements cannot estimate a high percentile reliably. Retain individual timings, choose a larger representative study on the actual target, and test the queue/latency objective before buying capacity. The local examples execute real jobs; all arrival, reserve and price scenarios remain labeled assumptions.

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

## Add metric summaries and capacity arithmetic

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

def capacity(measured_job_s, arrivals_per_hour, workers, hourly_rate, reserve_fraction=.25):
    """Illustrative one-job-per-worker plan; inputs are not cloud price discovery."""
    values=[measured_job_s,arrivals_per_hour,hourly_rate,reserve_fraction]
    if any(not math.isfinite(x) for x in values) or measured_job_s <= 0 or arrivals_per_hour < 0 or hourly_rate < 0:
        raise ValueError('invalid finite capacity inputs')
    if not isinstance(workers,int) or isinstance(workers,bool) or workers < 1 or not 0 <= reserve_fraction < 1:
        raise ValueError('invalid workers or reserve')
    offered=arrivals_per_hour*measured_job_s/3600
    load=offered/workers
    required=max(1,math.ceil(offered/(1-reserve_fraction)))
    return dict(offered_worker_hours_per_hour=offered,load_fraction=load,
                minimum_workers_with_reserve=required,hourly_budget=workers*hourly_rate,
                hypothetical_cost_per_job=measured_job_s*hourly_rate/3600,
                within_reserve=load <= 1-reserve_fraction)
```

Keep measured quantities separate from invented planning inputs and validate the units.

## Measure jobs before evaluating assumptions

Append this block to main.py. Run python3 main.py in the CPU course environment.

```python
samples=[]
with tempfile.TemporaryDirectory(prefix='ops-capacity-') as folder:
    for steps in [8,32,64]:
        run=launch(Path(folder)/str(steps),dict(steps=steps))
        assert run['status']=='completed'
        samples.append((steps,run,summarize(run)))
measured=max(run['wall_s'] for _,run,_ in samples)
plan=capacity(measured_job_s=measured,arrivals_per_hour=600,workers=2,hourly_rate=1.5,reserve_fraction=.25)
assert plan['hourly_budget']==3.
print('measured local job seconds:',[r['wall_s'] for _,r,_ in samples])
print('illustrative capacity plan:',plan)
print('assumptions: 600 arrivals/hour, 2 workers, hypothetical 1.50 per worker-hour, 25% reserve')
```

Every duration comes from a completed child; arrival and price inputs are labeled hypothetical.

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

def capacity(measured_job_s, arrivals_per_hour, workers, hourly_rate, reserve_fraction=.25):
    """Illustrative one-job-per-worker plan; inputs are not cloud price discovery."""
    values=[measured_job_s,arrivals_per_hour,hourly_rate,reserve_fraction]
    if any(not math.isfinite(x) for x in values) or measured_job_s <= 0 or arrivals_per_hour < 0 or hourly_rate < 0:
        raise ValueError('invalid finite capacity inputs')
    if not isinstance(workers,int) or isinstance(workers,bool) or workers < 1 or not 0 <= reserve_fraction < 1:
        raise ValueError('invalid workers or reserve')
    offered=arrivals_per_hour*measured_job_s/3600
    load=offered/workers
    required=max(1,math.ceil(offered/(1-reserve_fraction)))
    return dict(offered_worker_hours_per_hour=offered,load_fraction=load,
                minimum_workers_with_reserve=required,hourly_budget=workers*hourly_rate,
                hypothetical_cost_per_job=measured_job_s*hourly_rate/3600,
                within_reserve=load <= 1-reserve_fraction)

samples=[]
with tempfile.TemporaryDirectory(prefix='ops-capacity-') as folder:
    for steps in [8,32,64]:
        run=launch(Path(folder)/str(steps),dict(steps=steps))
        assert run['status']=='completed'
        samples.append((steps,run,summarize(run)))
measured=max(run['wall_s'] for _,run,_ in samples)
plan=capacity(measured_job_s=measured,arrivals_per_hour=600,workers=2,hourly_rate=1.5,reserve_fraction=.25)
assert plan['hourly_budget']==3.
print('measured local job seconds:',[r['wall_s'] for _,r,_ in samples])
print('illustrative capacity plan:',plan)
print('assumptions: 600 arrivals/hour, 2 workers, hypothetical 1.50 per worker-hour, 25% reserve')
```

Expected: Three actual local durations feed a first-order plan whose arrival rate, worker count, reserve and price are explicitly hypothetical. No hardware utilization or cloud-cost claim is inferred from CPU timings.

## Short jobs spend time outside the update loop

**Predict:** Will doubling the number of updates double total process time?

![Short jobs spend time outside the update loop](../../phases/16-operations/05-capacity-utilization-and-operating-cost/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal categories in both panels are actual jobs with $8$, $32$ and $64$ updates. The upper panel shows complete supervisor wall time in seconds. The lower panel shows the sum of synchronized update times in seconds. The panels use separate linear vertical scales, so compare their labeled numerical values rather than bar heights across panels. The labels retain the actual measurements from this run.

### Connect it to the computation

The lower-panel values are much smaller than the complete-job values because startup, compilation, logging and checkpoint writes are outside the update timer. Calculate each pair’s ratio from its labels to see how much of the short job is spent in measured updates. The ratio can be large for such small workloads. Host scheduling and filesystem work vary, so close wall-time values need not increase monotonically with update count. This figure explains timing boundaries and startup amortization; it does not measure hardware utilization or prove scaling on an accelerator.

```python
visual_data={'panels':[{'kind':'bar','labels':[str(n) for n,_,_ in samples],'xlabel':'updates in fresh local process','ylabel':'complete job seconds','title':'Complete process lifetime','series':[{'label':'wall time','y':[r['wall_s'] for _,r,_ in samples]}]},{'kind':'bar','labels':[str(n) for n,_,_ in samples],'xlabel':'updates in fresh local process','ylabel':'synchronized update seconds','title':'Training updates only — separate vertical scale','series':[{'label':'update time','y':[st['update_s'] for _,_,st in samples]}]}]}
```

## Recorded reference execution

CPU run: 2026-10-06T01:27:27.024822+00:00. JAX 0.9.2.

```text
measured local job seconds: [0.49369908310472965, 0.4885221249423921, 0.5049728329759091]
illustrative capacity plan: {'offered_worker_hours_per_hour': 0.08416213882931818, 'load_fraction': 0.04208106941465909, 'minimum_workers_with_reserve': 1, 'hourly_budget': 3.0, 'hypothetical_cost_per_job': 0.00021040534707329547, 'within_reserve': True}
assumptions: 600 arrivals/hour, 2 workers, hypothetical 1.50 per worker-hour, 25% reserve
measured local job seconds: [0.49332916690036654, 0.4850145000964403, 0.5038862081710249]
illustrative capacity plan: {'offered_worker_hours_per_hour': 0.08398103469517082, 'load_fraction': 0.04199051734758541, 'minimum_workers_with_reserve': 1, 'hourly_budget': 3.0, 'hypothetical_cost_per_job': 0.00020995258673792705, 'within_reserve': True}
assumptions: 600 arrivals/hour, 2 workers, hypothetical 1.50 per worker-hour, 25% reserve
nominal load at two demand scenarios: 0.04199051734758541 0.08398103469517082
required workers under changed hypothetical reserve: 1
PASS: operations-05

```

## Check units with known arithmetic

**Predict before running:** What load and minimum worker count follow from $90$-second jobs arriving $80$ times per hour?

```python
known=capacity(90.,80.,4,2.,.25)
assert known['offered_worker_hours_per_hour']==2.
assert known['load_fraction']==.5
assert known['minimum_workers_with_reserve']==3
assert known['hypothetical_cost_per_job']==.05
assert known['hourly_budget']==8.
```

**Expected:** Demand, load, reserve and cost match a hand calculation.

This independent arithmetic checks the calculator without treating invented inputs as measured infrastructure.

## Double the arrivals without changing the job

**Predict before running:** Does doubling demand require a proportional change in measured update speed?

```python
one=capacity(measured,600,2,1.5,.25)
two=capacity(measured,1200,2,1.5,.25)
assert abs(two['load_fraction']-2*one['load_fraction'])<1e-12
assert two['hypothetical_cost_per_job']==one['hypothetical_cost_per_job']
print('nominal load at two demand scenarios:',one['load_fraction'],two['load_fraction'])
```

**Expected:** Nominal load doubles while per-job active-time cost stays fixed under the same simple model.

Demand changes the required capacity; it does not automatically make each job faster or slower in a model without contention.

## Make it yours

Using the measured service-time input, calculate the minimum worker count for $1200$ arrivals per hour with $40$% reserve, then independently verify that the chosen count satisfies the load bound.

<details><summary>Reference solution</summary>

```python
changed=capacity(measured,1200,2,1.5,.4)
required=changed['minimum_workers_with_reserve']
demand=1200*measured/3600
assert demand/required <= .6 + 1e-12
if required>1: assert demand/(required-1) > .6
print('required workers under changed hypothetical reserve:',required)
```

</details>

## Account for failed attempts

**Transfer / diagnosis**

Assume $100$ completed jobs require $125$ equal-duration attempts. Compare active-time cost per attempt with active-time cost per completion.

<details><summary>Hint</summary>

Every attempt consumes resources under this hypothetical model.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
per_attempt=measured*1.5/3600
per_completion=125*per_attempt/100
assert abs(per_completion-1.25*per_attempt)<1e-12
```

Failure overhead belongs in the budget even when it does not count as successful throughput.

</details>

## Reject impossible capacity inputs

**Transfer / diagnosis**

Try a full reserve, zero workers and a nonfinite measured duration.

<details><summary>Hint</summary>

A plan should not silently divide by zero or emit a meaningless worker count.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
for change in [dict(reserve_fraction=1.),dict(workers=0),dict(measured_job_s=float('nan'))]:
    inputs=dict(measured_job_s=measured,arrivals_per_hour=600,workers=2,hourly_rate=1.5);inputs.update(change)
    try:capacity(**inputs)
    except ValueError:pass
    else:raise AssertionError('invalid plan accepted')
```

Input validation makes the assumptions visible and prevents numeric output from looking authoritative when the inputs are invalid.

</details>

## Check your understanding

Which claim is supported by the measured update duty fraction?

1. The fraction of local job wall time inside synchronized updates
2. The accelerator will be occupied by the same fraction
3. The same worker count meets any bursty arrival pattern

<details><summary>Answer and explanation</summary>

The fraction of local job wall time inside synchronized updates

The timing boundary is an application measurement. Hardware occupancy and queue latency require additional evidence from the actual target workload.

</details>

## Diagnose the result

If estimated cost is unexpectedly low, check seconds-to-hours conversion, idle reserved capacity and repeated failed attempts. If worker count looks sufficient but latency is poor, inspect burstiness and service-time variation. If longer jobs look faster per example, separate startup amortization from changes in the compiled update itself.

## Carry forward

- Carry units and assumptions through every capacity calculation.
- Measured local duration can inform a bounded model without pretending to establish fleet performance.

## Keep your evidence

Keep measured job/update samples and independent unit calculations. Label demand, reserve and price assumptions as hypothetical before deriving a resource plan.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [Python subprocess management](https://docs.python.org/3/library/subprocess.html)
- [Python os.replace and fsync](https://docs.python.org/3/library/os.html#os.replace)
- [JAX asynchronous dispatch and synchronization](https://docs.jax.dev/en/latest/async_dispatch.html)
- [JAX device discovery](https://docs.jax.dev/en/latest/_autosummary/jax.devices.html)

