"""MLOps: data contracts, CI gates and monitoring: worked experiments and reference solutions. CPU checks."""



# Offline MLOps: train a tiny model, serialize it, reload it, and gate predictions.
# No network, external service, or supplied squared-error array is used.
import hashlib, json, math, tempfile
from pathlib import Path

def digest(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()

def validate_rows(rows):
    if not rows: raise ValueError('empty dataset')
    ids=set(); group_split={}
    for row in rows:
        if set(row)!={'id','group','split','x','y','slice'}: raise ValueError('schema mismatch')
        if not row['id'] or row['id'] in ids: raise ValueError('duplicate or empty ID')
        if row['split'] not in {'train','validation','test'}: raise ValueError('unknown split')
        if row['slice'] not in {'low','high'}: raise ValueError('unknown slice')
        for key in ('x','y'):
            if isinstance(row[key],bool) or not isinstance(row[key],(int,float)) or not math.isfinite(row[key]): raise ValueError('invalid number')
        if row['group'] in group_split and group_split[row['group']]!=row['split']:
            raise ValueError('source group leaks across splits')
        ids.add(row['id']); group_split[row['group']]=row['split']
    return digest(rows)

def train_linear(rows):
    train=[r for r in rows if r['split']=='train']
    if len(train)<2: raise ValueError('need at least two training rows')
    mx=math.fsum(r['x'] for r in train)/len(train); my=math.fsum(r['y'] for r in train)/len(train)
    denom=math.fsum((r['x']-mx)**2 for r in train)
    if denom==0: raise ValueError('training inputs have no variation')
    slope=math.fsum((r['x']-mx)*(r['y']-my) for r in train)/denom
    return {'format':'linear-v1','slope':slope,'intercept':my-slope*mx,'training_manifest':digest(train)}

def predict(artifact,row):
    if artifact.get('format')!='linear-v1': raise ValueError('unknown artifact format')
    base=artifact['slope']*row['x']+artifact['intercept']
    return base+artifact.get('slice_offsets',{}).get(row['slice'],0.)

def evaluate_reloaded(rows,artifact_path,limit):
    artifact=json.loads(Path(artifact_path).read_text())
    test=[r for r in rows if r['split']=='test']
    if not test: raise ValueError('empty evaluation set')
    per=[]
    for r in test:
        pred=predict(artifact,r); err=(pred-r['y'])**2
        per.append({'id':r['id'],'slice':r['slice'],'target':r['y'],'prediction':pred,'squared_error':err})
    metrics={}
    for label in ('low','high'):
        sample=[v['squared_error'] for v in per if v['slice']==label]
        metrics[label]={'count':len(sample),'mse':math.fsum(sample)/len(sample) if sample else None}
    metrics['overall']={'count':len(per),'mse':math.fsum(v['squared_error'] for v in per)/len(per)}
    metrics['passed']=all(metrics[k]['count']>0 and metrics[k]['mse']<=limit for k in ('low','high','overall'))
    return per,metrics,artifact

# Four training sources and ten distinct, held-out sources. One required slice is intentionally risky.
rows=[dict(id=f'train-{i}',group=f'train-group-{i}',split='train',x=float(i),y=2.*i+1.,slice='low') for i in range(4)]
rows += [dict(id=f'low-{i}',group=f'low-group-{i}',split='test',x=float(i),y=2.*i+1.1,slice='low') for i in range(9)]
rows += [dict(id='high-0',group='high-group-0',split='test',x=8.,y=19.,slice='high')]
manifest=validate_rows(rows)
artifact=train_linear(rows)
with tempfile.TemporaryDirectory(prefix='mlops-reload-') as work:
    artifact_path=Path(work)/'candidate.json'
    artifact_path.write_text(json.dumps(artifact,sort_keys=True))
    per,metrics,reloaded=evaluate_reloaded(rows,artifact_path,.5)
    assert reloaded==artifact and len(per)==10
    assert metrics['low']['count']==9 and metrics['high']['count']==1
    assert math.isclose(metrics['low']['mse'],.01,abs_tol=1e-12)
    assert math.isclose(metrics['high']['mse'],4.,abs_tol=1e-12)
    assert math.isclose(metrics['overall']['mse'],.409,abs_tol=1e-12) and not metrics['passed']
    # Candidate B changes one model parameter and is independently reloaded.
    # Evaluation rows, targets, IDs, split, slices, counts and gate policy remain identical.
    candidate_b=dict(artifact,slice_offsets={'high':1.9})
    candidate_b_path=Path(work)/'candidate-b.json'
    candidate_b_path.write_text(json.dumps(candidate_b,sort_keys=True))
    paired_per,paired_metrics,reloaded_b=evaluate_reloaded(rows,candidate_b_path,.5)
    assert digest(reloaded_b)!=digest(reloaded)
    assert [(v['id'],v['target'],v['slice']) for v in per]==[(v['id'],v['target'],v['slice']) for v in paired_per]
    assert math.isclose(paired_metrics['low']['mse'],.01,abs_tol=1e-12)
    assert math.isclose(paired_metrics['high']['mse'],.01,abs_tol=1e-12) and paired_metrics['passed']
    # Separate label-correction sensitivity check: keep artifact A fixed and correct one held-out target.
    corrected_labels=[dict(r) for r in rows]; corrected_labels[-1]['y']=17.1
    corrected_path=Path(work)/'candidate-a-again.json'; corrected_path.write_text(json.dumps(artifact,sort_keys=True))
    corrected_per,corrected_metrics,_=evaluate_reloaded(corrected_labels,corrected_path,.5)
    assert [(v['id'],v['slice']) for v in per]==[(v['id'],v['slice']) for v in corrected_per]
    assert math.isclose(corrected_metrics['high']['mse'],.01,abs_tol=1e-12) and corrected_metrics['passed']
    # Leakage and missing-slice evidence fail explicitly.
    leaked=[dict(r) for r in rows]; leaked[-1]['group']=rows[0]['group']
    try: validate_rows(leaked)
    except ValueError: pass
    else: raise AssertionError('cross-split source leakage accepted')
    missing=[dict(r) for r in rows if r['slice']!='high']
    missing_path=Path(work)/'candidate.json'; missing_path.write_text(json.dumps(artifact))
    _,missing_metrics,_=evaluate_reloaded(missing,missing_path,.5)
    assert not missing_metrics['passed'] and missing_metrics['high']['count']==0
print('manifest:',manifest)
print('artifact fingerprint:',digest(reloaded))
print('per-example prediction sample:',per[0])
print('baseline metrics:',metrics)
print('candidate B fingerprint:',digest(reloaded_b))
print('paired candidate B metrics (same ten IDs, targets, splits, slices and policy):',paired_metrics)
print('label-correction sensitivity with artifact A held fixed:',corrected_metrics)
print('missing required slice:',missing_metrics['high'],'release:',missing_metrics['passed'])

# Figure data experiment
visual_data={'kind':'bar','labels':['overall (10)','low (9)','high (1)'],'xlabel':'evaluation group (same fixed cases)','ylabel':'mean squared error','series':[{'label':'candidate A','y':[metrics[k]['mse'] for k in ['overall','low','high']]},{'label':'candidate B','y':[paired_metrics[k]['mse'] for k in ['overall','low','high']]},{'label':'illustrative limit','y':[.5,.5,.5]}]}

# Experiment: Tighten one shared release policy without changing either model
# Change only the shared policy; preserve both serialized artifacts and evaluation rows.
with tempfile.TemporaryDirectory(prefix='mlops-policy-change-') as folder:
    policy_root=Path(folder)
    policy_results={}
    for name,saved_model in [('A',artifact),('B',candidate_b)]:
        model_path=policy_root/(name+'.json')
        model_path.write_text(json.dumps(saved_model,sort_keys=True))
        fixed_cases,strict_metrics,_=evaluate_reloaded(rows,model_path,.005)
        _,original_metrics,_=evaluate_reloaded(rows,model_path,.5)
        assert [(r['id'],r['target'],r['slice']) for r in fixed_cases]==[(r['id'],r['target'],r['slice']) for r in per]
        assert strict_metrics['overall']['count']==original_metrics['overall']['count']==10
        assert strict_metrics['low']==original_metrics['low']
        assert strict_metrics['high']==original_metrics['high']
        assert not strict_metrics['passed']
        policy_results[name]={'original_pass':original_metrics['passed'],'strict_pass':strict_metrics['passed']}
    assert policy_results=={'A':{'original_pass':False,'strict_pass':False},'B':{'original_pass':True,'strict_pass':False}}
print('Same predictions, changed policy:',policy_results)


# Reference solution. Try the exercise before reading this.
# This separate check changes one candidate parameter while freezing the labels.
changed_b=dict(artifact,slice_offsets={'high':.5})
with tempfile.TemporaryDirectory() as work:
    path=Path(work)/'changed-b.json'; path.write_text(json.dumps(changed_b))
    per_c,metrics_c,_=evaluate_reloaded(rows,path,.5)
assert [(v['id'],v['target'],v['slice']) for v in per_c]==[(v['id'],v['target'],v['slice']) for v in per]
assert math.isclose(metrics_c['high']['mse'],2.25,abs_tol=1e-12)
assert not metrics_c['passed']
high_example=next(v for v in per_c if v['slice']=='high')
assert math.isclose(high_example['squared_error'],2.25,abs_tol=1e-12)
print('candidate B high-slice example:',high_example)
print('candidate B with offset 0.5:',metrics_c)
print('Artifact A and B are distinct:',digest(artifact)!=digest(changed_b))

# Reference practice: Change the population mix
# `metrics` was computed from per-example errors after loading the local artifact.
low_mse=metrics['low']['mse']; high_mse=metrics['high']['mse']
new_low_count=1; new_high_count=9
weighted=(new_low_count*low_mse+new_high_count*high_mse)/(new_low_count+new_high_count)
assert math.isclose(weighted,3.601)
assert low_mse==metrics['low']['mse'] and high_mse==metrics['high']['mse']
print('new weighted MSE:',weighted,'unchanged slice MSEs:',low_mse,high_mse)
print("PASS: operations-06")
