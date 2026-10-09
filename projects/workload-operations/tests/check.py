"""Actual bounded subprocess drills; no scheduler or cloud calls."""
import argparse
import hashlib
import importlib.util
import json
import math
import platform
from pathlib import Path
import tempfile
import time
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--stage',type=int,choices=[1,2,3,4,5],default=5)
p.add_argument('--implementation',default='starter')
p.add_argument('--report',type=Path)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
path=root/a.implementation/'model.py' if a.implementation in ('starter','solution') else Path(a.implementation)
spec=importlib.util.spec_from_file_location('learner',path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
results={}
with tempfile.TemporaryDirectory(prefix='ops-check-') as folder:
 base=Path(folder)
 good=m.launch(base/'good',dict(steps=8))
 assert good['status']=='completed' and good['returncode']==0
 assert good['events'][0]['event']=='started' and good['events'][-1]['event']=='completed'
 assert good['events'][0]['backend']=='cpu' and good['events'][0]['device_count']>=1
 wrong=m.launch(base/'wrong',dict(min_devices=999))
 assert wrong['status']=='failed' and wrong['returncode']!=0
 assert any(e['event']=='failed' and 'runtime' in e['message'] for e in wrong['events'])
 stalled=m.launch(base/'stalled',dict(stall_at=0,stall_seconds=20),timeout=6)
 assert stalled['status']=='timed_out' and stalled['returncode']!=0
 assert any(e['event']=='stall_injected' for e in stalled['events']),stalled
 assert stalled['wall_s'] < 9
 results['lifecycle']={r['status']:r['returncode'] for r in [good,wrong,stalled]}
 print('PASS stage 1: real successful, preflight-failed and timed-out/reaped child processes',flush=True)
 if a.stage>=2:
  stats=m.summarize(good)
  progress=[e for e in good['events'] if e['event']=='progress']
  assert [e['step'] for e in progress]==list(range(1,9))
  assert stats['examples']==64 and stats['updates']==8 and stats['uncheckpointed_updates']==0
  np.testing.assert_allclose(stats['update_examples_per_s'],64/sum(e['update_s'] for e in progress))
  np.testing.assert_allclose(stats['job_examples_per_s'],64/good['wall_s'])
  assert 0<stats['update_duty_fraction']<1
  assert all(math.isfinite(e['loss']) and e['update_s']>0 for e in progress)
  assert all(y['monotonic_s']>=x['monotonic_s'] for x,y in zip(good['events'],good['events'][1:]))
  results['observability']=stats
  print('PASS stage 2: real structured events, units, counters, synchronized update and end-to-end boundaries',flush=True)
 if a.stage>=3:
  failed=m.launch(base/'recovery',dict(fail_after=3))
  assert failed['status']=='failed' and m.summarize(failed)['uncheckpointed_updates']==1
  checkpoint=json.loads((base/'recovery'/'checkpoint.json').read_text())
  assert checkpoint['step']==2
  resumed=m.launch(base/'recovery',dict(resume=True))
  assert resumed['status']=='completed'
  original={e['step']:e for e in good['events'] if e['event']=='progress'}
  restored=[e for e in resumed['events'] if e['event']=='progress']
  assert [e['step'] for e in restored]==list(range(3,9))
  for event in restored:
   assert event['state_hash']==original[event['step']]['state_hash']
   np.testing.assert_array_equal(event['loss'],original[event['step']]['loss'])
  assert json.loads((base/'good'/'checkpoint.json').read_text())==json.loads((base/'recovery'/'checkpoint.json').read_text())
  again=m.launch(base/'recovery',dict(resume=True))
  assert again['status']=='completed'
  # Changed config is rejected instead of silently continuing a different job.
  incompatible=m.launch(base/'recovery',dict(resume=True,learning_rate=.05))
  assert incompatible['status']=='failed'
  cp=json.loads((base/'recovery'/'checkpoint.json').read_text())
  cp['velocity'][0]+=1
  (base/'recovery'/'checkpoint.json').write_text(json.dumps(cp))
  corrupt=m.launch(base/'recovery',dict(resume=True))
  assert corrupt['status']=='failed'
  assert any(e['event']=='failed' and 'checksum' in e['message'] for e in corrupt['events'])
  results['recovery']=dict(restored_step=2,replayed_steps=[e['step'] for e in restored],next_update_equal=True)
  print('PASS stage 3: actual failure, complete checkpoint replay, finished restart, incompatible/corrupt restore rejection',flush=True)
 if a.stage>=4:
  store=base/'registry'
  state=json.loads((base/'good'/'checkpoint.json').read_text())
  artifact=dict(worker_hash=good['worker_hash'],config_hash=state['config_hash'],data_hash=state['data_hash'],state=state,validation={'passed':True})
  first=m.publish(store,artifact)
  assert len(first)==64
  assert first==hashlib.sha256(json.dumps(artifact,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
  assert not (store/'active.json').exists()
  m.activate(store,first)
  second=m.publish(store,dict(artifact,release_note='reviewed metadata change'))
  m.activate(store,second)
  assert m.rollback(store)==first
  assert json.loads((store/'active.json').read_text())['current']==first
  rejected=m.publish(store,dict(artifact,validation={'passed':False}))
  before=(store/'active.json').read_bytes()
  try:m.activate(store,rejected)
  except ValueError:pass
  else:raise AssertionError('failed validation activated')
  assert before==(store/'active.json').read_bytes()
  (store/'artifacts'/(second+'.json')).write_text('{}')
  try:m.activate(store,second)
  except ValueError:pass
  else:raise AssertionError('corrupt artifact activated')
  assert before==(store/'active.json').read_bytes()
  degraded=dict(artifact,state=dict(artifact['state'],params=[100.,-100.]))
  damaged=m.publish(store,degraded)
  try:m.activate(store,damaged)
  except ValueError:pass
  else:raise AssertionError('degraded model activated despite actual held-out error')
  assert before==(store/'active.json').read_bytes()
  results['provenance']=dict(first=first,rolled_back=True,failed_validation_rejected=True,corruption_rejected=True)
  print('PASS stage 4: content hashes, inactive publication, atomic selection, rollback and rejected corruption',flush=True)
 if a.stage>=5:
  for duration,arrivals,workers,rate,reserve in [(90,80,4,2,.25),(12,600,3,.5,.2),(good['wall_s'],120,2,1.5,.3)]:
   plan=m.capacity(duration,arrivals,workers,rate,reserve)
   demand=duration*arrivals/3600
   np.testing.assert_allclose(plan['load_fraction'],demand/workers)
   np.testing.assert_allclose(plan['hypothetical_cost_per_job'],duration*rate/3600)
   assert plan['minimum_workers_with_reserve']==max(1,math.ceil(demand/(1-reserve)))
   assert plan['within_reserve']==(demand/workers<=1-reserve)
  for invalid in [dict(workers=0),dict(reserve_fraction=1),dict(measured_job_s=float('nan')),dict(arrivals_per_hour=-1)]:
   values=dict(measured_job_s=1.,arrivals_per_hour=1.,workers=1,hourly_rate=1.)
   values.update(invalid)
   try:m.capacity(**values)
   except ValueError:pass
   else:raise AssertionError('invalid planning input accepted')
  results['capacity']=m.capacity(good['wall_s'],120,2,1.5,.3)
  print('PASS stage 5: independently checked units, reserve arithmetic, measured local input and invalid-input rejection',flush=True)
if a.report:
 a.report.write_text(json.dumps(dict(status='passed',stage=a.stage,python=platform.python_version(),implementationHash=hashlib.sha256(path.read_bytes()).hexdigest(),checkerHash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=results),indent=2)+'\n')
