"""Containerize a model service and verify its boundary: worked experiments and reference solutions. CPU checks."""

# Freeze a tiny model independently of training
# Step 1 — Freeze a tiny model independently of training: The independent probe computes 2\times2+1=5.
# Import hashlib, json, os, subprocess, sys, tempfile for this computation.
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
# Compute `MODEL` from `{'schema': 1, 'weight': 2., 'bias': 1.}`
MODEL = {'schema': 1, 'weight': 2., 'bias': 1.}

# Assert invariant `MODEL['weight']*2+MODEL['bias']==5` holds
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
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='container-contract-') as folder:
    # Compute `root` from `Path(folder)`
    root = Path(folder)
    artifact = root / 'model.json'
    service = root / 'service.py'
    # Compute `artifact.write_text(json.dumps(MODEL, sort_keys` as `True))`.
    artifact.write_text(json.dumps(MODEL, sort_keys=True))
    service.write_text(SERVICE)
    # Compute deterministic cryptographic digest `sha` for provenance verification.
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    # Configure environment variable before initializing the runtime.
    env = dict(os.environ, MODEL_PATH=str(artifact), MODEL_SHA256=sha)
    # Loop over `values` in `[[-0.5], [-2.0, 0.0, 1.5]]`:
    for values in [[-.5], [-2., 0., 1.5]]:
        # Read or serialize artifact data on disk (`completed`).
        completed = subprocess.run([sys.executable, str(service)], input=json.dumps({'inputs': values}), text=True, capture_output=True, env=env, check=True)
        # Read or serialize artifact data on disk (`actual`).
        actual = json.loads(completed.stdout)['predictions']
        # Assert invariant `actual == [2 * value + 1 for value in values]` holds
        assert actual == [2 * value + 1 for value in values]
        # Accumulate the next contribution into `accepted`.
        accepted += 1
    # Loop over `(payload, changes)` in `[({'inputs': []}, {}), ({'inputs': [True]}, {}), ({'inputs': [0.0]}, {'MODEL_SHA256': '0' * 64})]`:
    for payload, changes in [({'inputs': []}, {}), ({'inputs': [True]}, {}), ({'inputs': [0.]}, {'MODEL_SHA256': '0' * 64})]:
        # Read or serialize artifact data on disk (`completed`).
        completed = subprocess.run([sys.executable, str(service)], input=json.dumps(payload), text=True, capture_output=True, env=dict(env, **changes))
        # Assert that the computed values satisfy the numerical and structural contract.
        # Accumulate the next contribution into `rejected`.
        assert completed.returncode != 0
        rejected += 1
# Print the observed values to compare against the expected result.
print('Fresh-process valid requests:', accepted, 'rejected boundaries:', rejected)
# Print diagnostic summary of the computed outputs.
print('This companion tests the process/artifact contract. Run the Docker lab for container evidence.')

# Containerize a model service and verify its boundary: Containerization packages a runtime environment; a running...
# Import hashlib, json, os, subprocess, sys, tempfile for this computation.
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
# Compute `MODEL` from `{'schema': 1, 'weight': 2., 'bias': 1.}`
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
# Compute `accepted, rejected` from `0, 0`
accepted, rejected = 0, 0
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='container-contract-') as folder:
    # Compute `root` from `Path(folder)`
    root = Path(folder)
    artifact = root / 'model.json'
    service = root / 'service.py'
    # Compute `artifact.write_text(json.dumps(MODEL, sort_keys` as `True))`.
    artifact.write_text(json.dumps(MODEL, sort_keys=True))
    service.write_text(SERVICE)
    # Compute deterministic cryptographic digest `sha` for provenance verification.
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    # Configure environment variable before initializing the runtime.
    env = dict(os.environ, MODEL_PATH=str(artifact), MODEL_SHA256=sha)
    # Loop over `values` in `[[-0.5], [-2.0, 0.0, 1.5]]`:
    for values in [[-.5], [-2., 0., 1.5]]:
        # Read or serialize artifact data on disk (`completed`).
        completed = subprocess.run([sys.executable, str(service)], input=json.dumps({'inputs': values}), text=True, capture_output=True, env=env, check=True)
        # Read or serialize artifact data on disk (`actual`).
        actual = json.loads(completed.stdout)['predictions']
        # Assert invariant `actual == [2 * value + 1 for value in values]` holds
        assert actual == [2 * value + 1 for value in values]
        # Accumulate the next contribution into `accepted`.
        accepted += 1
    # Loop over `(payload, changes)` in `[({'inputs': []}, {}), ({'inputs': [True]}, {}), ({'inputs': [0.0]}, {'MODEL_SHA256': '0' * 64})]`:
    for payload, changes in [({'inputs': []}, {}), ({'inputs': [True]}, {}), ({'inputs': [0.]}, {'MODEL_SHA256': '0' * 64})]:
        # Read or serialize artifact data on disk (`completed`).
        completed = subprocess.run([sys.executable, str(service)], input=json.dumps(payload), text=True, capture_output=True, env=dict(env, **changes))
        # Assert that the computed values satisfy the numerical and structural contract.
        # Accumulate the next contribution into `rejected`.
        assert completed.returncode != 0
        rejected += 1
# Print the observed values to compare against the expected result.
print('Fresh-process valid requests:', accepted, 'rejected boundaries:', rejected)
# Print diagnostic summary of the computed outputs.
print('This companion tests the process/artifact contract. Run the Docker lab for container evidence.')

# Figure data experiment
# Compute figure data for: Observe the process boundary before containerizing it
# Compute `visual_data` from `{'kind':'bar','labels':['valid predictions','expecte...`
visual_data={'kind':'bar','labels':['valid predictions','expected rejections'],'xlabel':'local process contract','ylabel':'observed case count','series':[{'label':'executed CPU cases','y':[accepted,rejected]}]}

# Experiment: Change the batch without changing the model
# Experiment — Change the batch without changing the model: Batching changes transport shape, not the per-observation...
batch=[-.25,.75,2.]
# Assert invariant `[MODEL['weight']*x+MODEL['bias'] for x in batch]==[.5` holds
assert [MODEL['weight']*x+MODEL['bias'] for x in batch]==[.5,2.5,5.]
# Print the observed values to compare against the expected result.
print('Independent three-input predictions: 0.5, 2.5, 5.0')

# Experiment: Rehash an invalid model and watch validation reject it
# Experiment — Rehash an invalid model and watch validation reject it: Python accepts the nonstandard Infinity token in this fixture...
invalid_model = dict(MODEL, weight=float('inf'))
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory() as directory:
    # Read or serialize artifact data on disk (`invalid_path`).
    invalid_path = Path(directory)/'model.json'
    # Read or serialize artifact data on disk (`runner_path`).
    runner_path = Path(directory)/'service.py'
    # Write the serialized artifact payload to disk.
    invalid_path.write_text(json.dumps(invalid_model))
    runner_path.write_text(SERVICE)
    # Compute deterministic cryptographic digest `matching_digest` for provenance verification.
    matching_digest = hashlib.sha256(invalid_path.read_bytes()).hexdigest()
    # Configure environment variable before initializing the runtime.
    invalid_result = subprocess.run([sys.executable,str(runner_path)], input=json.dumps({'inputs':[0.]}), text=True,capture_output=True,env=dict(os.environ,MODEL_PATH=str(invalid_path),MODEL_SHA256=matching_digest),timeout=30)
    # Assert that `invalid_result.returncode != 0 and 'invalid coefficient' in invalid_result.stderr`.
    assert invalid_result.returncode != 0 and 'invalid coefficient' in invalid_result.stderr
# Print the observed values to compare against the expected result.
print('Matching digest did not bypass model validation.')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Reject an artifact whose bytes changed after its expected digest was...
original=json.dumps(MODEL,sort_keys=True).encode()
# Read or serialize artifact data on disk (`mutated`).
mutated=json.dumps(dict(MODEL,bias=2.),sort_keys=True).encode()
# Assert that `hashlib.sha256(original).hexdigest()!=hashlib.sha256(mutated).hexdigest()`.
assert hashlib.sha256(original).hexdigest()!=hashlib.sha256(mutated).hexdigest()
# Print the observed values to compare against the expected result.
print('Changed artifact requires a new recorded digest.')

# Reference practice: Reject a nonfinite request
# Reject a nonfinite request (transfer): A request can parse as JSON in Python while still violating...
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory() as tmp:
    # Compute `p` as `Path(tmp)`.
    p=Path(tmp)
    (p/'model').write_text(json.dumps(MODEL))
    (p/'service.py').write_text(SERVICE)
    # Configure environment variable before initializing the runtime.
    env=dict(os.environ,MODEL_PATH=str(p/'model'),MODEL_SHA256=hashlib.sha256((p/'model').read_bytes()).hexdigest())
    # Read or serialize artifact data on disk (`bad`).
    bad=subprocess.run([sys.executable,str(p/'service.py')],input=json.dumps({'inputs':[float('inf')]}),text=True,capture_output=True,env=env)
    # Assert invariant `bad.returncode != 0` holds
    assert bad.returncode != 0
# Print the observed values to compare against the expected result.
print('Nonfinite input rejected before output.')

# Reference practice: Check changed behavior after a legitimate artifact replacement
# Check changed behavior after a legitimate artifact replacement (Transfer / diagnosis): The old model produced 5; the new one produces 7.
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory() as directory:
    # Read or serialize artifact data on disk (`runner_path`).
    model_path=Path(directory)/'model.json'
    runner_path=Path(directory)/'service.py'
    # Compute `model_path.write_text(json.dumps(dict(MODEL,weight` as `3.)))`.
    model_path.write_text(json.dumps(dict(MODEL,weight=3.)))
    runner_path.write_text(SERVICE)
    # Compute deterministic cryptographic digest `digest` for provenance verification.
    digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
    # Configure environment variable before initializing the runtime.
    changed=subprocess.run([sys.executable,str(runner_path)],input=json.dumps({'inputs':[2.]}),text=True,capture_output=True,check=True,env=dict(os.environ,MODEL_PATH=str(model_path),MODEL_SHA256=digest),timeout=30)
    # Read or serialize artifact data on disk (`values`).
    values=json.loads(changed.stdout)['predictions']
    # Assert invariant `values==[7.] and values!=[5.]` holds
    assert values==[7.] and values!=[5.]
# Print the observed values to compare against the expected result.
print('New artifact loads, but the old behavioral expectation no longer passes.')
print("PASS: deployment-07")
