"""Changes, rollback, and artifact provenance: worked experiments and reference solutions. CPU checks."""

# Publish without selecting
def publish(store, artifact):
    """Content-address one JSON artifact. Publishing does not activate it."""
    store=Path(store)
    identifier=digest(artifact)
    atomic_json(store/'artifacts'/(identifier+'.json'),artifact)
    return identifier

# Evaluate evidence before pointer mutation
def activation_evidence_passes(artifact):
    required=['worker_hash','config_hash','data_hash','state']
    return (artifact.get('validation',{}).get('passed') is True
            and all(name in artifact for name in required)
            and artifact_metrics(artifact)['passed'])

# Read the selected artifact in the consumer
def consumer_predict(store, x):
    """A fresh local consumer resolves selection and loads that exact artifact."""
    pointer=json.loads((Path(store)/'active.json').read_text())
    artifact=read_artifact(store,pointer['current'])
    params=artifact['state']['params']
    return float(params[0])*float(x)+float(params[1])

# Complete runnable example (operations-04)
"""Bounded local JAX subprocess supervisor and lifecycle verifier."""
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
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

# Step 3 — Add bounded launch and result collection
def default_config(**changes):
    cfg=dict(seed=3,learning_rate=.04,momentum=.8,batch_size=8,steps=8,
             checkpoint_every=2,backend='cpu',min_devices=1,resume=False,
             fail_after=None,stall_at=None,stall_seconds=10.)
    # Apply configuration overrides before validating the resulting fields.
    cfg.update(changes)
    for name in ('steps','checkpoint_every','batch_size','min_devices'):
        if not isinstance(cfg[name],int) or isinstance(cfg[name],bool) or cfg[name] < 1:
            raise ValueError(name+' must be a positive integer')
    if cfg['batch_size'] > 64 or not 0 <= cfg['momentum'] < 1 or not 0 < cfg['learning_rate'] < 1:
        raise ValueError('invalid training configuration')
    return cfg


def launch(root, config=None, timeout=10.):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    cfg=default_config(**(config or {}))
    if timeout <= 0:
        raise ValueError('timeout must be positive')
    atomic_json(root/'config.json',cfg)
    worker=root/'worker.py'
    worker.write_text(WORKER)
    env=dict(os.environ,JAX_PLATFORMS=cfg['backend'],PYTHONUNBUFFERED='1')
    started=time.perf_counter()
    process=subprocess.Popen([sys.executable,str(worker),str(root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
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
    wall=time.perf_counter()-started
    events=[]
    malformed=[]
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event=json.loads(line)
            if not isinstance(event,dict) or 'event' not in event:
                raise ValueError('not an event')
            events.append(event)
        except (ValueError,TypeError):
            malformed.append(line)
    status='timed_out' if timed_out else 'completed' if process.returncode == 0 and not malformed and events and events[-1]['event']=='completed' else 'failed'
    result=dict(status=status,returncode=process.returncode,wall_s=wall,events=events,stderr=stderr,
                worker_hash=hashlib.sha256(WORKER.encode()).hexdigest(),config=cfg,malformed_stdout=malformed)
    atomic_json(root/'run.json',result)
    return result

# Step 4 — Add content-addressed publication and selection
def publish(store, artifact):
    """Content-address one JSON artifact. Publishing does not activate it."""
    store=Path(store)
    identifier=digest(artifact)
    atomic_json(store/'artifacts'/(identifier+'.json'),artifact)
    return identifier


def read_artifact(store, identifier):
    if len(identifier) != 64 or any(c not in '0123456789abcdef' for c in identifier):
        raise ValueError('invalid artifact identifier')
    artifact=json.loads((Path(store)/'artifacts'/(identifier+'.json')).read_text())
    if digest(artifact) != identifier:
        raise ValueError('artifact digest mismatch')
    return artifact


def artifact_metrics(artifact):
    """Independent held-out arithmetic for this fixed linear-model release contract."""
    params=artifact.get('state',{}).get('params',[])
    if len(params)!=2 or any(not math.isfinite(float(v)) for v in params):
        raise ValueError('invalid model parameters')
    inputs=(-1.5,-.37,.22,1.5)
    mse=sum((params[0]*x+params[1]-(2*x+1))**2 for x in inputs)/len(inputs)
    return dict(held_out_mse=mse,acceptance_limit=1.,passed=mse<=1.)


def activation_evidence_passes(artifact):
    required=['worker_hash','config_hash','data_hash','state']
    return (artifact.get('validation',{}).get('passed') is True
            and all(name in artifact for name in required)
            and artifact_metrics(artifact)['passed'])

def activate(store, identifier):
    artifact=read_artifact(store,identifier)
    if not activation_evidence_passes(artifact):
        raise ValueError('artifact evidence or held-out validation failed')
    previous=None
    pointer=Path(store)/'active.json'
    if pointer.exists():previous=json.loads(pointer.read_text())['current']
    atomic_json(pointer,dict(current=identifier,previous=previous))
    return identifier


def consumer_predict(store, x):
    """A fresh local consumer resolves selection and loads that exact artifact."""
    pointer=json.loads((Path(store)/'active.json').read_text())
    artifact=read_artifact(store,pointer['current'])
    params=artifact['state']['params']
    return float(params[0])*float(x)+float(params[1])


def rollback(store):
    pointer=json.loads((Path(store)/'active.json').read_text())
    if not pointer.get('previous'):raise ValueError('no previous version')
    return activate(store,pointer['previous'])

# Step 5 — Observe consumer behavior across activation, rejection and rollback.
# This local, single-writer fixture has no authentication or production access control.
with tempfile.TemporaryDirectory(prefix='ops-release-') as folder:
    base=Path(folder)
    run=launch(base/'run')
    state=json.loads((base/'run'/'checkpoint.json').read_text())
    store=base/'registry'
    evidence=dict(worker_hash=run['worker_hash'],config_hash=state['config_hash'],data_hash=state['data_hash'],validation={'passed':True,'check':'completed CPU fixture and checksum'})
    # Two valid model payloads have different predictions at x=1.
    # Recompute the checkpoint checksum after changing each model state.
    state.pop('checksum',None)
    state_a=dict(state,params=[2.0,1.0])
    state_a['checksum']=digest(state_a)
    state_b=dict(state,params=[2.4,1.0])
    state_b['checksum']=digest(state_b)
    artifact_a=dict(evidence,state=state_a)
    artifact_a['validation'].update(artifact_metrics(artifact_a))
    version_a=publish(store,artifact_a)
    activate(store,version_a)
    before_b=consumer_predict(store,1.0)
    artifact_b=dict(evidence,state=state_b,release_note='validated slope revision')
    artifact_b['validation']=dict(evidence['validation'])
    artifact_b['validation'].update(artifact_metrics(artifact_b))
    version_b=publish(store,artifact_b)
    assert consumer_predict(store,1.0)==before_b==3.0  # publication alone changes no behavior
    activate(store,version_b)
    after_b=consumer_predict(store,1.0)
    assert after_b==3.4
    rejected_artifact=dict(artifact_b,validation={'passed':False,'check':'deliberate negative control'})
    rejected=publish(store,rejected_artifact)
    pointer_before=(store/'active.json').read_bytes()
    behavior_before=consumer_predict(store,1.0)
    try:
        activate(store,rejected)
    except ValueError:
        rejected_without_change=(store/'active.json').read_bytes()==pointer_before and consumer_predict(store,1.0)==behavior_before
    else:
        raise AssertionError('bad release activated')
    assert rejected_without_change
    rollback(store)
    after_rollback=consumer_predict(store,1.0)
    assert after_rollback==3.0
    selected=json.loads((store/'active.json').read_text())['current']
    assert selected==version_a
artifact=artifact_a  # later negative-control exercises start from this valid fixture artifact
print('consumer x=1 before B / after B / after rollback:',before_b,after_b,after_rollback)
print('A and B IDs:',version_a,version_b)
print('rejected activation unchanged selection and behavior:',rejected_without_change)
print('rollback selected A:',selected==version_a)


# Figure data experiment
with tempfile.TemporaryDirectory() as folder:
    a=publish(folder,artifact)
    activate(folder,a)
    values=[1]
    b=publish(folder,dict(artifact,release_note='B'))
    values.append(1 if json.loads((Path(folder)/'active.json').read_text())['current']==a else 2)
    activate(folder,b)
    values.append(2 if json.loads((Path(folder)/'active.json').read_text())['current']==b else 1)
    bad=publish(folder,dict(artifact,validation={'passed':False}))
    try:activate(folder,bad)
    except ValueError:pass
    values.append(2 if json.loads((Path(folder)/'active.json').read_text())['current']==b else 1)
    rollback(folder)
    values.append(1 if json.loads((Path(folder)/'active.json').read_text())['current']==a else 2)
assert values==[1,1,2,2,1]
visual_data={'kind':'bar','labels':['activate A','publish B','activate B','reject candidate','rollback'],'xlabel':'executed local operation','ylabel':'selected artifact code: 1=A, 2=B','series':[{'label':'observed active pointer','y':values}]}

# Experiment: Identity follows content rather than key insertion order
# Experiment — Identity follows content rather than key insertion order
assert digest({'a':1,'b':2})==digest({'b':2,'a':1})
assert digest({'a':1,'b':2})!=digest({'a':1,'b':3})

# Experiment: Detect corruption before activation
# Experiment — Detect corruption before activation
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory() as folder:
    identifier=publish(folder,artifact)
    # Write the serialized artifact payload to disk.
    (Path(folder)/'artifacts'/(identifier+'.json')).write_text('{}')
    try:
        activate(folder,identifier)
    except ValueError as error:
        assert 'digest' in str(error)
    else:
        raise AssertionError('corruption accepted')
    assert not (Path(folder)/'active.json').exists()

# Experiment: Reject an actually degraded model
# Experiment — Reject an actually degraded model: Measured validation and claimed validation are distinct.
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory() as folder:
    stable=publish(folder,artifact)
    activate(folder,stable)
    degraded=dict(artifact,state=dict(artifact['state'],params=[100.,-100.]))
    candidate=publish(folder,degraded)
    assert artifact_metrics(degraded)['held_out_mse'] > 1.
    before=(Path(folder)/'active.json').read_bytes()
    try:activate(folder,candidate)
    except ValueError as error:assert 'held-out' in str(error)
    else:raise AssertionError('degraded model activated')
    assert before==(Path(folder)/'active.json').read_bytes()

# Reference solution. Try the exercise before reading this.
# Reference interpretation: A=[2.0,1.0] predicts 3.0 at x=1; B=[2.4,1.0] predicts 3.4. Publishing B leaves A selected, so prediction stays 3.0. Activating B changes it to 3.4. A failed validation keeps pointer bytes and output at 3.4. Rollback selects A and prediction returns to 3.0. The assertion checks consumer behavior as well as selected identity.

# Reference practice: Reject an incomplete provenance record
# Reject an incomplete provenance record (Transfer / diagnosis)
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory() as folder:
    incomplete=dict(artifact)
    del incomplete['worker_hash']
    identifier=publish(folder,incomplete)
    try:
        activate(folder,identifier)
    except ValueError:
        pass
    else:
        raise AssertionError('missing provenance accepted')

# Reference practice: Check rollback without history
# Check rollback without history (Transfer / diagnosis)
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory() as folder:
    identifier=publish(folder,artifact)
    activate(folder,identifier)
    before=(Path(folder)/'active.json').read_bytes()
    try:
        rollback(folder)
    except ValueError:
        pass
    else:
        raise AssertionError('invented rollback history')
    assert before==(Path(folder)/'active.json').read_bytes()
print("PASS: operations-04")
