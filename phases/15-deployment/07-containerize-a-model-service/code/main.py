"""Containerize a model service and verify its boundary: worked experiments and reference solutions. CPU checks."""

# Freeze a tiny model independently of training
# Step 1 — Freeze a tiny model independently of training: The independent probe computes 2\times2+1=5.
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
MODEL = {'schema': 1, 'weight': 2., 'bias': 1.}

assert MODEL['weight']*2+MODEL['bias']==5

# Write the service boundary
# The deployment contract uses exported coefficients, not a training environment.
SERVICE = '''import hashlib,json,math,os,sys
raw=open(os.environ['MODEL_PATH'],'rb').read()
if hashlib.sha256(raw).hexdigest()!=os.environ['MODEL_SHA256']:
    raise ValueError('artifact mismatch')
m=json.loads(raw)
if not isinstance(m,dict) or set(m)!={'schema','weight','bias'}:
    raise ValueError('model schema')
if type(m['schema']) is not int or m['schema']!=1:
    raise ValueError('model schema version')
if any(type(m[k]) not in (int,float) or not math.isfinite(m[k]) for k in ('weight','bias')):
    raise ValueError('invalid coefficient')
x=json.load(sys.stdin)['inputs']
if not isinstance(x,list) or not 1<=len(x)<=32:
    raise ValueError('batch limit')
if any(type(v) not in (int,float) or not math.isfinite(v) for v in x):
    raise ValueError('invalid input')
print(json.dumps({'predictions':[m['weight']*v+m['bias'] for v in x]},allow_nan=False))
'''

# Launch fresh processes and check both outcomes
# Step 3 — Launch fresh processes and check both outcomes: Two valid batches pass and three deliberately invalid cases fail.
accepted, rejected = 0, 0
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory(prefix='container-contract-') as folder:
    root = Path(folder)
    artifact = root / 'model.json'
    service = root / 'service.py'
    artifact.write_text(json.dumps(MODEL, sort_keys=True))
    service.write_text(SERVICE)
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    env = dict(os.environ, MODEL_PATH=str(artifact), MODEL_SHA256=sha)
    for values in [[-.5], [-2., 0., 1.5]]:
        completed = subprocess.run([sys.executable, str(service)], input=json.dumps({'inputs': values}), text=True, capture_output=True, env=env, check=True)
        actual = json.loads(completed.stdout)['predictions']
        assert actual == [2 * value + 1 for value in values]
        # Accumulate the next contribution into `accepted`.
        accepted += 1
    for payload, changes in [({'inputs': []}, {}), ({'inputs': [True]}, {}), ({'inputs': [0.]}, {'MODEL_SHA256': '0' * 64})]:
        completed = subprocess.run([sys.executable, str(service)], input=json.dumps(payload), text=True, capture_output=True, env=dict(env, **changes))
        # Assert that the computed values satisfy the numerical and structural contract.
        # Accumulate the next contribution into `rejected`.
        assert completed.returncode != 0
        rejected += 1
print('Fresh-process valid requests:', accepted, 'rejected boundaries:', rejected)
print('This companion tests the process/artifact contract. Run the Docker lab for container evidence.')

# Containerize a model service and verify its boundary
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
MODEL = {'schema': 1, 'weight': 2., 'bias': 1.}
# The deployment contract uses exported coefficients, not a training environment.
SERVICE = '''import hashlib,json,math,os,sys
raw=open(os.environ['MODEL_PATH'],'rb').read()
if hashlib.sha256(raw).hexdigest()!=os.environ['MODEL_SHA256']:
    raise ValueError('artifact mismatch')
m=json.loads(raw)
if not isinstance(m,dict) or set(m)!={'schema','weight','bias'}:
    raise ValueError('model schema')
if type(m['schema']) is not int or m['schema']!=1:
    raise ValueError('model schema version')
if any(type(m[k]) not in (int,float) or not math.isfinite(m[k]) for k in ('weight','bias')):
    raise ValueError('invalid coefficient')
x=json.load(sys.stdin)['inputs']
if not isinstance(x,list) or not 1<=len(x)<=32:
    raise ValueError('batch limit')
if any(type(v) not in (int,float) or not math.isfinite(v) for v in x):
    raise ValueError('invalid input')
print(json.dumps({'predictions':[m['weight']*v+m['bias'] for v in x]},allow_nan=False))
'''
accepted, rejected = 0, 0
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory(prefix='container-contract-') as folder:
    root = Path(folder)
    artifact = root / 'model.json'
    service = root / 'service.py'
    artifact.write_text(json.dumps(MODEL, sort_keys=True))
    service.write_text(SERVICE)
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    env = dict(os.environ, MODEL_PATH=str(artifact), MODEL_SHA256=sha)
    for values in [[-.5], [-2., 0., 1.5]]:
        completed = subprocess.run([sys.executable, str(service)], input=json.dumps({'inputs': values}), text=True, capture_output=True, env=env, check=True)
        actual = json.loads(completed.stdout)['predictions']
        assert actual == [2 * value + 1 for value in values]
        # Accumulate the next contribution into `accepted`.
        accepted += 1
    for payload, changes in [({'inputs': []}, {}), ({'inputs': [True]}, {}), ({'inputs': [0.]}, {'MODEL_SHA256': '0' * 64})]:
        completed = subprocess.run([sys.executable, str(service)], input=json.dumps(payload), text=True, capture_output=True, env=dict(env, **changes))
        # Assert that the computed values satisfy the numerical and structural contract.
        # Accumulate the next contribution into `rejected`.
        assert completed.returncode != 0
        rejected += 1
print('Fresh-process valid requests:', accepted, 'rejected boundaries:', rejected)
print('This companion tests the process/artifact contract. Run the Docker lab for container evidence.')

# Figure data experiment
visual_data={'kind':'bar','labels':['valid predictions','expected rejections'],'xlabel':'local process contract','ylabel':'observed case count','series':[{'label':'executed CPU cases','y':[accepted,rejected]}]}

# Experiment: Change the batch without changing the model
# Experiment — Change the batch without changing the model
batch=[-.25,.75,2.]
assert [MODEL['weight']*x+MODEL['bias'] for x in batch]==[.5,2.5,5.]
print('Independent three-input predictions: 0.5, 2.5, 5.0')

# Experiment: Rehash an invalid model and watch validation reject it
# Experiment — Rehash an invalid model and watch validation reject it
invalid_model = dict(MODEL, weight=float('inf'))
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory() as directory:
    invalid_path = Path(directory)/'model.json'
    runner_path = Path(directory)/'service.py'
    # Write the serialized artifact payload to disk.
    invalid_path.write_text(json.dumps(invalid_model))
    runner_path.write_text(SERVICE)
    matching_digest = hashlib.sha256(invalid_path.read_bytes()).hexdigest()
    invalid_result = subprocess.run([sys.executable,str(runner_path)], input=json.dumps({'inputs':[0.]}), text=True,capture_output=True,env=dict(os.environ,MODEL_PATH=str(invalid_path),MODEL_SHA256=matching_digest),timeout=30)
    assert invalid_result.returncode != 0 and 'invalid coefficient' in invalid_result.stderr
print('Matching digest did not bypass model validation.')

# Reference solution. Try the exercise before reading this.
# Exercise solution: send changed artifact bytes through the fresh-process service loader.
# SERVICE is the service program defined earlier in this lesson. Its integrity check
# runs before model parsing, request evaluation, or prediction output.
original = json.dumps(MODEL, sort_keys=True).encode()
recorded_digest = hashlib.sha256(original).hexdigest()
mutated = json.dumps(dict(MODEL, bias=2.), sort_keys=True).encode()
assert hashlib.sha256(mutated).hexdigest() != recorded_digest
with tempfile.TemporaryDirectory(prefix='changed-artifact-exercise-') as folder:
    artifact = Path(folder) / 'model.json'
    service = Path(folder) / 'service.py'
    artifact.write_bytes(mutated)  # altered bytes reach the actual loader
    service.write_text(SERVICE)
    env = dict(os.environ, MODEL_PATH=str(artifact), MODEL_SHA256=recorded_digest)
    rejected = subprocess.run(
        [sys.executable, str(service)], input=json.dumps({'inputs': [2.]}),
        text=True, capture_output=True, env=env, timeout=30)
    assert rejected.returncode != 0
    assert 'artifact mismatch' in rejected.stderr
    assert rejected.stdout == ''  # no prediction was activated or returned
print('Changed artifact reached fresh-process loader: rejected before prediction; no output.')

# Reference practice: Reject a nonfinite request
# Reject a nonfinite request (transfer)
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)
    (p/'model').write_text(json.dumps(MODEL))
    (p/'service.py').write_text(SERVICE)
    env=dict(os.environ,MODEL_PATH=str(p/'model'),MODEL_SHA256=hashlib.sha256((p/'model').read_bytes()).hexdigest())
    bad=subprocess.run([sys.executable,str(p/'service.py')],input=json.dumps({'inputs':[float('inf')]}),text=True,capture_output=True,env=env)
    assert bad.returncode != 0
print('Nonfinite input rejected before output.')

# Reference practice: Check changed behavior after a legitimate artifact replacement
# Check changed behavior after a legitimate artifact replacement (Transfer / diagnosis): The old model produced 5; the new one produces 7.
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory() as directory:
    model_path=Path(directory)/'model.json'
    runner_path=Path(directory)/'service.py'
    model_path.write_text(json.dumps(dict(MODEL,weight=3.)))
    runner_path.write_text(SERVICE)
    digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
    changed=subprocess.run([sys.executable,str(runner_path)],input=json.dumps({'inputs':[2.]}),text=True,capture_output=True,check=True,env=dict(os.environ,MODEL_PATH=str(model_path),MODEL_SHA256=digest),timeout=30)
    values=json.loads(changed.stdout)['predictions']
    assert values==[7.] and values!=[5.]
print('New artifact loads, but the old behavioral expectation no longer passes.')
print("PASS: deployment-07")
