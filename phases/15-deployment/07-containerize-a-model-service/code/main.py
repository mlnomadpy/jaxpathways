"""Containerize a model service and verify its boundary: worked experiments and reference solutions. CPU checks."""



import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
MODEL = {'schema': 1, 'weight': 2., 'bias': 1.}
# The deployment contract uses exported coefficients, not a training environment.
SERVICE = '''import hashlib,json,math,os,sys
raw=open(os.environ['MODEL_PATH'],'rb').read()
if hashlib.sha256(raw).hexdigest()!=os.environ['MODEL_SHA256']: raise ValueError('artifact mismatch')
m=json.loads(raw)
x=json.load(sys.stdin)['inputs']
if not isinstance(x,list) or not 1<=len(x)<=32: raise ValueError('batch limit')
if any(type(v) not in (int,float) or not math.isfinite(v) for v in x): raise ValueError('invalid input')
print(json.dumps({'predictions':[m['weight']*v+m['bias'] for v in x]},allow_nan=False))
'''
accepted, rejected = 0, 0
with tempfile.TemporaryDirectory(prefix='container-contract-') as folder:
    root = Path(folder); artifact = root / 'model.json'; service = root / 'service.py'
    artifact.write_text(json.dumps(MODEL, sort_keys=True)); service.write_text(SERVICE)
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    env = dict(os.environ, MODEL_PATH=str(artifact), MODEL_SHA256=sha)
    for values in [[-.5], [-2., 0., 1.5]]:
        completed = subprocess.run([sys.executable, str(service)], input=json.dumps({'inputs': values}), text=True, capture_output=True, env=env, check=True)
        actual = json.loads(completed.stdout)['predictions']
        assert actual == [2 * value + 1 for value in values]
        accepted += 1
    for payload, changes in [({'inputs': []}, {}), ({'inputs': [True]}, {}), ({'inputs': [0.]}, {'MODEL_SHA256': '0' * 64})]:
        completed = subprocess.run([sys.executable, str(service)], input=json.dumps(payload), text=True, capture_output=True, env=dict(env, **changes))
        assert completed.returncode != 0; rejected += 1
print('Fresh-process valid requests:', accepted, 'rejected boundaries:', rejected)
print('This companion tests the process/artifact contract. Run the Docker lab for container evidence.')


# Figure data experiment
visual_data={'kind':'bar','labels':['valid predictions','expected rejections'],'xlabel':'local process contract','ylabel':'observed case count','series':[{'label':'executed CPU cases','y':[accepted,rejected]}]}

# Experiment: Change the batch without changing the model
batch=[-.25,.75,2.]
assert [MODEL['weight']*x+MODEL['bias'] for x in batch]==[.5,2.5,5.]
print('Independent three-input predictions: 0.5, 2.5, 5.0')

# Reference solution. Try the exercise before reading this.
original=json.dumps(MODEL,sort_keys=True).encode()
mutated=json.dumps(dict(MODEL,bias=2.),sort_keys=True).encode()
assert hashlib.sha256(original).hexdigest()!=hashlib.sha256(mutated).hexdigest()
print('Changed artifact requires a new recorded digest.')

# Reference practice: Reject a nonfinite request
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp);(p/'model').write_text(json.dumps(MODEL));(p/'service.py').write_text(SERVICE)
    env=dict(os.environ,MODEL_PATH=str(p/'model'),MODEL_SHA256=hashlib.sha256((p/'model').read_bytes()).hexdigest())
    bad=subprocess.run([sys.executable,str(p/'service.py')],input=json.dumps({'inputs':[float('inf')]}),text=True,capture_output=True,env=env)
    assert bad.returncode != 0
print('Nonfinite input rejected before output.')
print("PASS: deployment-07")
