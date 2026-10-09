"""Cumulative public contracts; independent references, not a private exam."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import jax
import jax.numpy as jnp

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--implementation',default='solution')
parser.add_argument('--stage',default='all')
parser.add_argument('--resume')
parser.add_argument('--artifact')
parser.add_argument('--output')
args=parser.parse_args()
path=ROOT/'solution/model.py' if args.implementation=='solution' else Path(args.implementation).resolve()
spec=importlib.util.spec_from_file_location('learner_model',path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def rejects(fn):
    try:fn()
    except (ValueError,KeyError,TypeError):return
    raise AssertionError('invalid input was accepted')

def host_objective(p,x,t,labels):
    zi=x@np.asarray(p['image'],np.float64)
    zt=t@np.asarray(p['text'],np.float64)
    zi/=np.maximum(np.linalg.norm(zi,axis=1,keepdims=True),1e-6)
    zt/=np.maximum(np.linalg.norm(zt,axis=1,keepdims=True),1e-6)
    scores=zi@zt.T/.2
    positive=labels[:,None]==labels[None,:]
    exp=np.exp(scores)
    return float((np.mean(-np.log((exp*positive).sum(1)/exp.sum(1)))+np.mean(-np.log((exp*positive).sum(0)/exp.sum(0))))/2)

train=m.fixture(1)
held=m.fixture(7,per_class=5)
if args.resume:
    state=m.load_checkpoint(args.resume,train)
    rows=[]
    for _ in range(5):
        state,event=m.transition(state,train)
        rows.append(event)
    Path(args.output).write_text(json.dumps({'state':m._host_state(state),'events':rows}))
    sys.exit()
if args.artifact:
    art=m.load_artifact(args.artifact)
    Path(args.output).write_text(json.dumps(m.infer(art,images=held['images'][:8]).tolist()))
    sys.exit()
last=5 if args.stage=='all' else int(args.stage)
assert 1<=last<=5
x=m.image_features(train['images'])
t=m.text_features(train['captions'])
labels=train['labels']
assert x.shape==(48,64) and t.shape==(48,4)
np.testing.assert_array_equal(m.text_features(['thin vertical','horizontal thick']),[[1,0,1,0],[0,1,0,1]])
known=np.arange(64,dtype=np.float32).reshape(1,8,8)/64
np.testing.assert_array_equal(m.image_features(known),known.reshape(1,64))
for bad in ['vertical vertical','vertical','thin thick','vertical unknown','vertical thin thick']:
    rejects(lambda bad=bad:m.text_features([bad]))
for bad in [known.astype(np.float64),known[0],np.full((1,8,8),np.nan,np.float32),known+2]:
    rejects(lambda bad=bad:m.image_features(bad))
m.check_splits(train,held)
rejects(lambda:m.check_splits(train,train))
assert m.dataset_hash(train)!=m.dataset_hash(held)
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder)
    np.save(root/'train.npy',train['images'][0])
    np.save(root/'held.npy',held['images'][0])
    manifest={'provenance':'Locally generated test fixture; not a real dataset','license':'Original synthetic fixture',
              'records':[{'id':'a','group':'source-a','split':'train','image':'train.npy','caption':'vertical thin'},
                         {'id':'b','group':'source-b','split':'held','image':'held.npy','caption':'vertical thin'}]}
    (root/'pairs.json').write_text(json.dumps(manifest))
    loaded=m.load_pairs(root/'pairs.json','train')
    np.testing.assert_array_equal(loaded['images'][0],train['images'][0])
    m.check_splits(loaded,m.load_pairs(root/'pairs.json','held'))
    manifest['records'][1]['group']='source-a'
    (root/'pairs.json').write_text(json.dumps(manifest))
    rejects(lambda:m.load_pairs(root/'pairs.json','train'))

print('PASS stage 1: pair IDs, independent pixel/token contracts, leakage and malformed inputs',flush=True)
if last==1:sys.exit()
params=m.init_params(3)
for offset in [0.0,0.017]:
    p={k:np.asarray(v)+np.float32(offset) for k,v in params.items()}
    expected=host_objective(p,x,t,labels)
    actual=float(m.objective(p,jnp.asarray(x),jnp.asarray(t),jnp.asarray(labels)))
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
    grads=jax.grad(m.objective)(p,jnp.asarray(x),jnp.asarray(t),jnp.asarray(labels))
    for key,index in [('image',(2,1)),('image',(24,3)),('text',(1,2))]:
        plus={k:v.astype(np.float64).copy() for k,v in p.items()}
        minus={k:v.astype(np.float64).copy() for k,v in p.items()}
        plus[key][index]+=1e-4
        minus[key][index]-=1e-4
        fd=(host_objective(plus,x,t,labels)-host_objective(minus,x,t,labels))/2e-4
        np.testing.assert_allclose(float(grads[key][index]),fd,atol=2e-4,rtol=2e-3)
zi,zt=m.embeddings(params,jnp.asarray(x),jnp.asarray(t))
np.testing.assert_allclose(np.linalg.norm(zi,axis=1),1,atol=2e-6)
np.testing.assert_allclose(np.linalg.norm(zt,axis=1),1,atol=2e-6)
# Duplicate semantic pairs are positives: duplicating the batch leaves this objective unchanged.
np.testing.assert_allclose(float(m.objective(params,jnp.tile(x,(2,1)),jnp.tile(t,(2,1)),jnp.tile(labels,2))),float(m.objective(params,x,t,labels)),atol=2e-6)
zero_x=jnp.zeros_like(jnp.asarray(x))
zero_value,zero_grads=jax.value_and_grad(m.objective)(params,zero_x,jnp.asarray(t),jnp.asarray(labels))
assert np.isfinite(float(zero_value))
assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(zero_grads))
zero_params={k:jnp.zeros_like(v) for k,v in params.items()}
assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(jax.grad(m.objective)(zero_params,jnp.asarray(x),jnp.asarray(t),jnp.asarray(labels))))
print('PASS stage 2: independent symmetric multi-positive loss, changed-parameter finite differences, normalization',flush=True)
if last==2:sys.exit()
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    blank=m.fixture(11,per_class=1)
    blank['images'][0]=0
    blank_state,blank_event=m.transition(m.initial_state(blank,batch_size=4),blank)
    assert np.isfinite(blank_event['loss']) and all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(blank_state['params']))
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(blank_state['momentum']))
    trained,h=m.train(train,seed=0)
    second,_=m.train(train,seed=19)
    for state in [trained,second]:
        before=m.digest(m._host_state(state))
        ev=m.evaluate(state['params'],held)
        assert ev['image_to_text_correct']>=19 and ev['text_to_image_correct']>=19
        assert m.digest(m._host_state(state))==before
    assert np.mean(h[-10:])<np.mean(h[:10])
    saved=m.initial_state(train)
    for _ in range(2):saved,_=m.transition(saved,train)
    m.save_checkpoint(root/'checkpoint',saved)
    command=[sys.executable,str(Path(__file__).resolve()),'--implementation',str(path),'--resume',str(root/'checkpoint'),'--output',str(root/'resumed.json')]
    subprocess.run(command,check=True,capture_output=True,text=True,timeout=60)
    independent=json.loads((root/'resumed.json').read_text())
    rows=[]
    baseline=saved
    for _ in range(5):
        baseline,event=m.transition(baseline,train)
        rows.append(event)
    assert independent['events']==rows
    assert independent['state']==m._host_state(baseline)
    rejects(lambda:m.load_checkpoint(root/'checkpoint',held))
    rejects(lambda:m.load_checkpoint(root/'checkpoint',train,batch_size=8))
    original_rate=m.TRAINING['learning_rate']
    m.TRAINING['learning_rate']=original_rate*2
    rejects(lambda:m.load_checkpoint(root/'checkpoint',train))
    m.TRAINING['learning_rate']=original_rate
    record=json.loads((root/'checkpoint/checkpoint.json').read_text())
    record['payload']['state']['step']+=1
    (root/'checkpoint/checkpoint.json').write_text(json.dumps(record))
    rejects(lambda:m.load_checkpoint(root/'checkpoint',train))
    shifted=m.evaluate(trained['params'],m.fixture(9,per_class=5,shift=1))
    print('Clean/shifted image retrieval:',ev['image_to_text_correct'],shifted['image_to_text_correct'],'out of 20')
    print('PASS stage 3: two training seeds, fixed held-out data, fresh-process complete replay, corruption/configuration rejection',flush=True)
    if last==3:sys.exit()
    calibration=m.calibrate(trained['params'],train)
    assert calibration['data_hash']==m.dataset_hash(train) and calibration['data_hash']!=m.dataset_hash(held)
    zero,scale=m.quantize_weights(np.zeros((4,4),np.float32))
    assert np.all(zero==0) and np.all(scale>0)
    policies=[]
    for policy in ['fp32','w8a32','w8a8']:
        destination=root/policy
        m.export_artifact(destination,trained['params'],calibration,policy)
        art=m.load_artifact(destination)
        for branch,features in [('image',m.image_features(held['images'][:8])),('text',m.text_features(held['captions'][:8]))]:
            w=np.asarray(trained['params'][branch])
            qw,s=m.quantize_weights(w)
            if policy=='fp32':raw=features.astype(np.float64)@w.astype(np.float64)
            elif policy=='w8a32':raw=features.astype(np.float64)@(qw.astype(np.float64)*s)
            else:
                a=calibration['input_scales'][branch]
                qx=np.clip(np.rint(features/a),-127,127).astype(np.int64)
                assert features.shape[1]*127**2 < np.iinfo(np.int32).max
                raw=(qx@qw.astype(np.int64))*a*s
            oracle=raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-6)
            actual=m.infer(art,images=held['images'][:8]) if branch=='image' else m.infer(art,captions=held['captions'][:8])
            np.testing.assert_allclose(actual,oracle,atol=3e-6,rtol=3e-5)
        for count in [1,8]:
            np.testing.assert_allclose(m.infer(art,images=held['images'][:count]),np.asarray(m.encoder(trained['params'],'image',policy,calibration)(jnp.asarray(m.image_features(held['images'][:count])))),atol=3e-6,rtol=3e-5)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--implementation',str(path),'--artifact',str(destination),'--output',str(root/'loaded.json')],check=True,capture_output=True,text=True,timeout=60)
        np.testing.assert_allclose(json.loads((root/'loaded.json').read_text()),m.infer(art,images=held['images'][:8]),atol=3e-6,rtol=3e-5)
        policies.append(art)
    print('PASS stage 4: calibration provenance, int64 independent oracle, 12 real exports, fresh-process reload, singleton parity',flush=True)
    if last==4:sys.exit()
    art=policies[-1]
    rejects(lambda:m.infer(art))
    rejects(lambda:m.infer(art,images=held['images'][:1],captions=['vertical thin']))
    rejects(lambda:m.infer(art,images=held['images'][:3]))
    measure=m.measure(art,held['images'][:8],repeats=12)
    assert len(measure['milliseconds'])==12 and min(measure['milliseconds'])>0
    np.testing.assert_allclose(measure['p50_ms'],np.median(measure['milliseconds']))
    # Two examples from each class supply a balanced release gate.
    idx=np.array([0,1,5,6,10,11,15,16])
    gate_x=held['images'][idx]
    gate_y=held['labels'][idx]
    m.activate(root/'registry',root/'fp32',gate_x,gate_y)
    m.activate(root/'registry',root/'w8a8',gate_x,gate_y)
    m.activate(root/'registry',root/'fp32',gate_x,gate_y)
    before=(root/'registry/ACTIVE.json').read_bytes()
    bad={k:jnp.zeros_like(v) for k,v in trained['params'].items()}
    m.export_artifact(root/'bad',bad,calibration)
    rejects(lambda:m.activate(root/'registry',root/'bad',gate_x,gate_y))
    assert (root/'registry/ACTIVE.json').read_bytes()==before
    damaged=root/'w8a8/image-1.jax'
    payload=damaged.read_bytes()
    damaged.write_bytes(payload+b'corrupted')
    rejects(lambda:m.load_artifact(root/'w8a8'))
    print('Observed local request milliseconds:',measure['p50_ms'],measure['p95_ms'])
    print('PASS stage 5: missing-modality/shape rejection, actual inference timing, measured release gate, rollback and corrupt artifact',flush=True)
print('All public stages passed; no real-data, semantic-language or target-device qualification.')
