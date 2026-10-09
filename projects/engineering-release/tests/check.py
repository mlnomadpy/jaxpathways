"""Cumulative learner policy checks plus real tracking and fresh-process serving integration."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--implementation', default='starter')
parser.add_argument('--stage', choices=['1','2','3','4','all'], default='all')
args = parser.parse_args()
source = ROOT/'solution/engineering.py' if args.implementation == 'solution' else ROOT/'starter/engineering.py' if args.implementation == 'starter' else Path(args.implementation).resolve()
spec = importlib.util.spec_from_file_location('learner', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
stage = 4 if args.stage == 'all' else int(args.stage)

def rejects(call):
    try:
        call()
    except (ValueError, TypeError):
        return
    raise AssertionError('invalid contract accepted')

rows = [dict(id=str(i),group=str(i),split='train' if i<2 else 'validation',x=float(i),y=float(2*i+1)) for i in range(4)]
assert m.validate_rows(rows) == hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
rejects(lambda:m.validate_rows(rows+[rows[0]]))
bad=[dict(r) for r in rows]
bad[-1]['group']=bad[0]['group']
rejects(lambda:m.validate_rows(bad))
bad=[dict(r) for r in rows]
bad[1]['x']=float('nan')
rejects(lambda:m.validate_rows(bad))
for errors,slices,expected in [([.01]*9+[4.],['low']*9+['high'],False),([.02,.03,.04],['low','high','high'],True),([.01],['low'],False)]:
    result=m.gate_metrics(errors,slices,.5)
    assert math.isclose(result['overall']['mse'],sum(errors)/len(errors)) and result['passed'] is expected
rejects(lambda:m.gate_metrics([float('nan')],['low'],.5))
rejects(lambda:m.gate_metrics([1.],[],.5))
print('PASS stage 1: data identity, leakage, nonfinite inputs, independent weighted metrics and missing-slice rejection',flush=True)
if stage==1:sys.exit()
case=dict(unanswerable=False,expected='CPU',document='d1')
answer=dict(answer='CPU',citations=['d1'],abstain=False)
assert m.llm_case(case,answer,['d1'])
assert not m.llm_case(case,dict(answer,citations=['invented']),['d1'])
assert not m.llm_case(case,dict(answer,answer='GPU'),['d1'])
unknown=dict(unanswerable=True)
assert m.llm_case(unknown,dict(answer='',citations=[],abstain=True),['d1'])
assert not m.llm_case(unknown,dict(answer,abstain=True),['d1'])
assert not m.llm_case(case,dict(answer,abstain='false'),['d1'])
print('PASS stage 2: changed expected answer, unknown citations, unsupported cases and malformed abstention',flush=True)
if stage==2:sys.exit()
bundle=dict(model_hash=m.digest({'weight':2}),data_hash=m.digest(rows),evaluation_hash=m.digest([.02,.03]),image_digest='sha256:'+'1'*64,owner='course',target='cpu-fixture',passed=True)
approval=m.approval_for(bundle,'reviewer',100,200)
with tempfile.TemporaryDirectory() as folder:
    m.activate(folder,'v1',bundle,approval,150,'cpu-fixture')
    pointer=Path(folder)/'active.json'
    before=pointer.read_bytes()
    for b,a,t,target in [(dict(bundle,model_hash=m.digest({'weight':3})),approval,150,'cpu-fixture'),(bundle,approval,200,'cpu-fixture'),(bundle,approval,99,'cpu-fixture'),(bundle,approval,150,'edge'),(dict(bundle,passed=False),approval,150,'cpu-fixture')]:
        rejects(lambda:m.activate(folder,'bad',b,a,t,target))
        assert pointer.read_bytes()==before
    second=dict(bundle,owner='second-team')
    approved=m.approval_for(second,'reviewer',100,250)
    assert m.activate(folder,'v2',second,approved,160,'cpu-fixture')['previous']=='v1'
    assert m.activate(folder,'v1',bundle,approval,170,'cpu-fixture')['current']=='v1'
rejects(lambda:m.approval_for(bundle,'reviewer',200,100))
print('PASS stage 3: bundle binding, time/target checks, atomic rejection and revalidated local rollback',flush=True)
if stage==3:sys.exit()
with tempfile.TemporaryDirectory(prefix='engineering-check-') as folder:
    run=subprocess.run([sys.executable,str(ROOT/'mlflow_lab.py'),'--output',folder],capture_output=True,text=True,timeout=90)
    if run.returncode:raise AssertionError(run.stdout+run.stderr)
    report=json.loads((Path(folder)/'report.json').read_text())
    assert report['validation_mse'][1] < .001 < report['validation_mse'][0]
    assert report['trace_count']==1 and report['span_count']==3 and report['champion_version']=='2'
    artifact=Path(folder)/'model.json'
    model=json.loads(artifact.read_text())
    env=dict(os.environ,MODEL_PATH=str(artifact),MODEL_SHA256=report['model_sha256'])
    result=subprocess.run([sys.executable,str(ROOT/'service.py'),'--predict'],input=json.dumps({'inputs':[-.5,0.,1.]}),text=True,capture_output=True,env=env,check=True)
    assert json.loads(result.stdout)['predictions']==[model['bias']-.5*model['weight'],model['bias'],model['bias']+model['weight']]
    errors=[(model['weight']*x+model['bias']-(2*x+1))**2 for x in [-1.5,-.5,.5,1.5]]
    quality=m.gate_metrics(errors,['low','low','high','high'],.001)
    assert quality['passed']
    tracked_bundle=dict(bundle,model_hash=report['model_sha256'],evaluation_hash=m.digest(quality))
    approved=m.approval_for(tracked_bundle,'fixture-reviewer',100,200)
    selected=m.activate(Path(folder)/'selection',report['champion_version'],tracked_bundle,approved,150,'cpu-fixture')
    assert selected['bundle_hash']==m.digest(tracked_bundle)
    for payload in [{'inputs':[]},{'inputs':[True]},{'inputs':[float('inf')]}]:
        invalid=subprocess.run([sys.executable,str(ROOT/'service.py'),'--predict'],input=json.dumps(payload),text=True,capture_output=True,env=env)
        assert invalid.returncode!=0
print('PASS stage 4: real MLflow runs, pyfunc reload, registry aliases, prompt version, nested trace and fresh-process service',flush=True)
print('PASS stage 4: MLflow tracking, registry aliases, and service verification complete.')
