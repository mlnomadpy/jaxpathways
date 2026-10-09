"""Cumulative image lifecycle checks, including new-process restore and export."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/jaxpathways-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/jaxpathways-image-cache")
import subprocess
import sys
import tempfile
import time
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--implementation',default='starter')
p.add_argument('--stage',type=int,choices=range(1,9),default=8)
p.add_argument('--report',type=Path)
p.add_argument('--figures',action='store_true')
a=p.parse_args()
path=ROOT/a.implementation/'model.py' if a.implementation in ('starter','solution') else Path(a.implementation)
spec=importlib.util.spec_from_file_location('learner',path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
results={}
train=m.make_dataset(11,96,'train')
held=m.make_dataset(31,60,'test')
shift=m.make_dataset(31,60,'occlusion',True)


def np_forward(params,x):
    """Independent loop over spatial receptive fields, no JAX convolution."""
    params={k:np.asarray(v,dtype=np.float64) for k,v in params.items()}
    z=np.empty((len(x),6,6,6))
    for row in range(6):
        for col in range(6):
            for channel in range(6):
                z[:,row,col,channel]=(x[:,row:row+3,col:col+3,:]*params['conv'][:,:,:,channel]).sum(axis=(1,2,3))+params['conv_bias'][channel]
    return np.maximum(z,0).mean(axis=(1,2))@params['head']+params['head_bias']


def compare_state(left,right):
    for field in ['params','velocity']:
        for name in left[field]:np.testing.assert_array_equal(left[field][name],right[field][name])
    for field in ['key','order']:np.testing.assert_array_equal(left[field],right[field])
    for field in ['step','position','epoch','seed','data_hash','settings']:assert left[field]==right[field]


m.validate_splits(train,held)
x=m.preprocess(train['images'])
assert x.shape==(96,8,8,1) and x.dtype==np.float32
np.testing.assert_allclose(x,2*train['images'].astype(np.float32)/255-1,atol=1e-7)
np.testing.assert_array_equal(x,m.preprocess(train['images'].transpose(0,3,1,2),layout='NCHW'))
rgb=np.repeat(train['images'][:2],3,axis=-1)
np.testing.assert_allclose(m.preprocess(rgb),x[:2],atol=2e-7)
red=np.zeros((1,16,12,3),np.uint8)
red[:,:,:,0]=255
np.testing.assert_allclose(m.preprocess(red),2*.2126-1,atol=1e-7)
for value,kwargs in [(train['images'].astype(float),{}),(np.full((1,8,8,1),2.),{'input_range':'unit_float'}),(np.zeros((1,8,8,4),np.uint8),{}),(np.empty((0,8,8,1),np.uint8),{})]:
 try:m.preprocess(value,**kwargs)
 except ValueError:pass
 else:raise AssertionError('malformed image accepted')
try:m.validate_splits(train,dict(held,groups=[train['groups'][0]]+held['groups'][1:]))
except ValueError:pass
else:raise AssertionError('source-group leakage accepted')
with tempfile.TemporaryDirectory() as d:
 root=Path(d)
 Image.fromarray(rgb[0]).save(root/'sample.png')
 read=m.read_image(root/'sample.png')
 np.testing.assert_array_equal(read,rgb[:1])
 manifest=dict(split='external-test',source='self-generated test PNG',license='original fixture',items=[dict(path='sample.png',id='png-1',group='subject-1',label='vertical')])
 (root/'data.json').write_text(json.dumps(manifest))
 external=m.load_external_manifest(root/'data.json')
 assert external['images'].shape==(1,8,8,3)
 assert len(external['provenance']['file_hashes'][0])==64
 (root/'broken.png').write_bytes(b'not an image')
 try:m.read_image(root/'broken.png')
 except Exception:pass
 else:raise AssertionError('corrupt file decoded')
print('PASS stage 1: real image arrays, independent channel/range/layout/resize checks, local PNG ingestion, provenance and leakage guards',flush=True)

if a.stage>=2:
 for seed in [0,4]:
  params=m.init_params(seed)
  for batch in [1,3,7]:
   np.testing.assert_allclose(m.forward(params,jnp.asarray(x[:batch])),np_forward(params,x[:batch]),rtol=2e-5,atol=2e-6)
  assert sum(v.size for v in params.values())==81
  point={k:np.asarray(v,dtype=np.float64).copy() for k,v in params.items()}
  def independent_loss(p):
   logits=np_forward(p,x[:5])
   shifted=logits-logits.max(1,keepdims=True)
   return np.mean(np.log(np.exp(shifted).sum(1))-shifted[np.arange(5),train['labels'][:5]])
  eps=1e-4
  plus={k:v.copy() for k,v in point.items()}
  minus={k:v.copy() for k,v in point.items()}
  plus['head'][2,1]+=eps
  minus['head'][2,1]-=eps
  fd=(independent_loss(plus)-independent_loss(minus))/(2*eps)
  grad=jax.grad(m.objective)(params,jnp.asarray(x[:5]),jnp.asarray(train['labels'][:5]))
  np.testing.assert_allclose(grad['head'][2,1],fd,atol=2e-5,rtol=1e-3)
 try:m.objective(params,jnp.asarray(x[:3]),jnp.ones((3,1),jnp.int32))
 except ValueError:pass
 else:raise AssertionError('column label broadcast accepted')
 print('PASS stage 2: independent convolution/pooling/dense forward, 81 parameters, changed batches/seeds, finite differences and label axes',flush=True)

if a.stage>=3:
 curves=[]
 state=m.start(train,seed=0)
 tiny=m.start(train,seed=3)
 for _ in range(20):
  state,record=m.advance(state,train,10)
  curves.append(dict(step=state['step'],train_loss=m.evaluate(state['params'],train)['loss'],validation_loss=m.evaluate(state['params'],held)['loss']))
 assert curves[-1]['train_loss']<.1 and m.evaluate(state['params'],held)['accuracy']>.95
 replay,r=m.advance(m.start(train,seed=0),train,200)
 compare_state(state,replay)
 other,_=m.advance(m.start(train,seed=4),train,200)
 assert m.evaluate(other['params'],held)['accuracy']>.9
 subset={k:(v[:12] if k in ('images','labels','ids','groups') else v) for k,v in train.items()}
 fitted,_=m.advance(m.start(subset,seed=3),subset,150)
 assert m.evaluate(fitted['params'],subset)['accuracy']==1.
 results['training']=dict(steps=200,seed=0,clean_accuracy=m.evaluate(state['params'],held)['accuracy'],train_loss=curves[-1]['train_loss'],changed_seed_accuracy=m.evaluate(other['params'],held)['accuracy'])
 print('PASS stage 3: actual augmented image training, tiny-batch fit, full replay and changed training seed',flush=True)

if a.stage>=4:
 with tempfile.TemporaryDirectory() as d:
  root=Path(d)
  prefix,_=m.advance(m.start(train,seed=0),train,13)
  m.save_checkpoint(root/'checkpoint.npz',prefix)
  full,expected=m.advance(prefix,train,11)
  # Fresh interpreter imports the learner implementation, reads dataset/checkpoint, and continues.
  child="""import importlib.util,sys,json,numpy as np
from pathlib import Path
s=importlib.util.spec_from_file_location('learner',sys.argv[1])
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
r=Path(sys.argv[2])
data=m.make_dataset(11,96,'train')
state=m.restore_checkpoint(r/'checkpoint.npz',data)
state,records=m.advance(state,data,11)
m.save_checkpoint(r/'resumed.npz',state)
(r/'records.json').write_text(json.dumps(records))
"""
  run=subprocess.run([sys.executable,'-c',child,str(path.resolve()),str(root)],capture_output=True,text=True,timeout=45)
  assert run.returncode==0,run.stderr
  restored=m.restore_checkpoint(root/'resumed.npz',train)
  compare_state(full,restored)
  assert expected==json.loads((root/'records.json').read_text())
  modified=dict(train,images=train['images'].copy())
  modified['images'][0,0,0,0]^=1
  for data,settings in [(modified,None),(train,dict(learning_rate=.09))]:
   try:m.restore_checkpoint(root/'checkpoint.npz',data,settings)
   except ValueError:pass
   else:raise AssertionError('changed data/settings accepted for recovery')
 results['recovery']=dict(committed_step=13,next_steps=[v['step'] for v in expected],fresh_process=True,ids_augmented_inputs_losses_state_equal=True)
 print('PASS stage 4: fresh-process full optimizer/key/order/position restore; next IDs, augmentations, losses and state match; incompatible input rejected',flush=True)

if a.stage>=5:
 before={k:np.asarray(v).copy() for k,v in state['params'].items()}
 whole=m.evaluate(state['params'],held,batch_size=60)
 parts=m.evaluate(state['params'],held,batch_size=7)
 for k in before:np.testing.assert_array_equal(state['params'][k],before[k])
 np.testing.assert_allclose(whole['loss'],parts['loss'],atol=1e-7)
 assert whole['confusion']==parts['confusion']
 independent=np_forward(state['params'],m.preprocess(held['images']))
 np.testing.assert_allclose(whole['logits'],independent,atol=1e-4,rtol=1e-5)
 clean=whole
 shifted=m.evaluate(state['params'],shift)
 assert shifted['accuracy']<clean['accuracy'] and len(shifted['error_ids'])>0
 assert np.array(shifted['confusion']).sum()==60
 results['evaluation']=dict(clean_accuracy=clean['accuracy'],corruption_accuracy=shifted['accuracy'],corruption_confusion=shifted['confusion'],error_ids=shifted['error_ids'])
 print('PASS stage 5: frozen evaluation, independent logits, unequal batches, confusion counts and declared corruption failures',flush=True)

if a.stage>=6:
 q=m.calibrate(state['params'],train)
 assert str(q['calibration_hash'])==m.dataset_hash(train) and str(q['calibration_hash'])!=m.dataset_hash(held)
 quantized,clipping=m.integer_forward(q,m.preprocess(held['images']))
 intmetrics=m.metrics(quantized,held['labels'],held['ids'])
 assert intmetrics['accuracy']>.9
 # The integer accumulation is independently checked at one receptive field.
 ix=m.quantize(m.preprocess(held['images'][:1]),q['input_scale'])
 patch=ix[0,:3,:3,0].reshape(9)
 reference=np.array([sum(int(patch[i])*int(q['conv'][i,c]) for i in range(9)) for c in range(6)],np.int32)
 acc=m.patches(ix).astype(np.int32)@q['conv'].astype(np.int32)
 np.testing.assert_array_equal(acc[0,0,0],reference)
 assert 9*127*127<np.iinfo(np.int32).max and 6*127*127<np.iinfo(np.int32).max
 assert q['conv'].dtype==q['head'].dtype==np.int8
 lower_policies={}
 for policy in ['float16','bfloat16']:
  policy_report=m.evaluate(state['params'],held,compute=policy)
  lower_policies[policy]=dict(accuracy=policy_report['accuracy'],loss=policy_report['loss'],max_logit_error=float(np.max(np.abs(policy_report['logits']-clean['logits']))),accumulation='float32',optimizer_storage='float32')
  assert policy_report['accuracy']>.95
  outputs=m.forward(state['params'],jnp.asarray(x[:9]),compute=policy)
  assert outputs.dtype==jnp.float32 and np.isfinite(outputs).all()
 zero=m.calibrate(jax.tree.map(jnp.zeros_like,state['params']),train)
 assert np.isfinite(m.integer_forward(zero,x[:2])[0]).all()
 narrow=m.calibrate(state['params'],train,percentile=30.)
 _,nclip=m.integer_forward(narrow,m.preprocess(held['images']))
 assert nclip['hidden_clipped']>clipping['hidden_clipped']
 try:m.integer_forward(q,x[:1],accumulator='int16')
 except ValueError:pass
 else:raise AssertionError('unsupported accumulator accepted')
 results['precision']=dict(lower_float_policies=lower_policies,int8_accuracy=intmetrics['accuracy'],max_logit_error=float(np.max(np.abs(quantized-clean['logits']))),clipping=clipping,aggressive_calibration_clipping=nclip,weight_bytes_float32=sum(np.asarray(v).nbytes for v in state['params'].values()),quantized_payload_bytes=sum(np.asarray(v).nbytes for v in q.values() if np.asarray(v).dtype.kind!='U'))
 print('PASS stage 6: actual int8 products/int32 accumulation, calibration clipping, zero channels, FP16/BF16 execution and unsupported-policy rejection',flush=True)

if a.stage>=7:
 with tempfile.TemporaryDirectory() as d:
  root=Path(d)
  manifest=m.export_model(root/'artifact',state,q)
  runtime=m.select_runtime(root/'artifact',held['images'][:4],clean['logits'][:4])
  for bad,kwargs in [(np.empty((0,8,8,1),np.uint8),{}),(held['images'][:1],{'precision':'int16'}),(np.zeros((1,8,8,4),np.uint8),{})]:
   try:m.infer(runtime,bad,**kwargs)
   except ValueError:pass
   else:raise AssertionError('malformed inference request accepted')
  try:m.select_runtime(root/'artifact',held['images'][:4],clean['logits'][:4]+1)
  except ValueError:pass
  else:raise AssertionError('failed parity canary accepted')
  for count in [1,2,4,7]:np.testing.assert_allclose(m.infer(runtime,held['images'][:count]),clean['logits'][:count],atol=2e-5,rtol=2e-5)
  np.testing.assert_array_equal(m.infer(runtime,held['images'][:7],precision='int8'),quantized[:7])
  np.save(root/'request.npy',held['images'][:7])
  child="""import importlib.util,sys,numpy as np
from pathlib import Path
s=importlib.util.spec_from_file_location('learner',sys.argv[1])
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
r=Path(sys.argv[2])
runtime=m.load_model(r/'artifact')
np.save(r/'response.npy',m.infer(runtime,np.load(r/'request.npy')))
"""
  run=subprocess.run([sys.executable,'-c',child,str(path.resolve()),str(root)],capture_output=True,text=True,timeout=45)
  assert run.returncode==0,run.stderr
  np.testing.assert_allclose(np.load(root/'response.npy'),clean['logits'][:7],atol=2e-5,rtol=2e-5)
  try:m.load_model(root/'artifact',dict(m.PREPROCESS,version='changed'))
  except ValueError:pass
  else:raise AssertionError('preprocess mismatch accepted')
  benchmarks=[m.benchmark(runtime,held['images'][:4],precision=policy) for policy in ['float32','int8']]
  for b in benchmarks:assert len(b['samples_s'])==12 and min(b['samples_s'])>0 and b['images_per_s']>0
  results['export']=dict(actual_serialized_jax=True,fresh_process=True,request_batches=[1,2,4,7],artifact_bytes=sum(v.stat().st_size for v in (root/'artifact').iterdir()),manifest=manifest)
  previous=runtime
  # A valid second bundle must pass parity before serving; rollback retains the prior verified handle.
  m.export_model(root/'candidate',state,q)
  runtime=m.select_runtime(root/'candidate',held['images'][:4],clean['logits'][:4])
  runtime=previous
  np.testing.assert_allclose(m.infer(runtime,held['images'][:1]),clean['logits'][:1],atol=2e-5,rtol=2e-5)
  (root/'candidate'/'model-1.jax').write_bytes(b'corrupted')
  try:runtime=m.select_runtime(root/'candidate',held['images'][:4],clean['logits'][:4])
  except ValueError:pass
  else:raise AssertionError('corrupt candidate selected')
  assert runtime is previous
  (root/'artifact'/'model-1.jax').write_bytes(b'corrupted')
  try:m.load_model(root/'artifact')
  except ValueError:pass
  else:raise AssertionError('corrupt exported graph loaded')
 print('PASS stage 7: actual JAX export/reload, fresh interpreter inference, fixed graph signatures and adaptive batching, preprocessing/integrity guards',flush=True)

if a.stage>=8:
 results['inference_measurements']=benchmarks
 assert all(b['backend']=='cpu' for b in benchmarks)
 print('PASS stage 8: completed CPU warmup and repeated synchronous end-to-end adapter timings; no native edge speed claim',flush=True)

if a.figures:
 if a.stage<8:raise ValueError('figures require all stages')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 ROOT.joinpath('outputs').mkdir(exist_ok=True)
 fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
 ax=axes[0,0]
 ax.plot([v['step'] for v in curves],[v['train_loss'] for v in curves],label='train (no augmentation)')
 ax.plot([v['step'] for v in curves],[v['validation_loss'] for v in curves],label='frozen held-out')
 ax.set(xlabel='completed updates',ylabel='mean cross-entropy',title='Same post-update loss definition')
 ax.legend()
 for ax,report,title in [(axes[0,1],clean,'Clean held-out counts'),(axes[1,0],shifted,'Declared occlusion + noise counts')]:
  matrix=np.asarray(report['confusion'])
  ax.imshow(matrix,vmin=0,vmax=20,cmap='Purples')
  ax.set(xticks=range(3),yticks=range(3),xticklabels=m.CLASSES,yticklabels=m.CLASSES,xlabel='predicted class',ylabel='true class',title=title)
  for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>10 else 'black')
 axes[1,1].bar(['float32','int8 integer reference'],[clean['accuracy'],intmetrics['accuracy']])
 axes[1,1].set(ylim=(0,1.1),ylabel='held-out accuracy',title='Same clean examples, different arithmetic')
 fig.savefig(ROOT/'outputs/lifecycle.png',dpi=140)
 fig.savefig(ROOT/'outputs/lifecycle.svg')
 plt.close(fig)
 fig,axes=plt.subplots(2,4,figsize=(10,5),layout='constrained')
 wrong=np.flatnonzero(shifted['predictions']!=shift['labels'])[:4]
 for col,i in enumerate(wrong):
  axes[0,col].imshow(held['images'][i,:,:,0],vmin=0,vmax=255,cmap='gray')
  axes[0,col].set_title('clean: '+m.CLASSES[held['labels'][i]])
  axes[1,col].imshow(shift['images'][i,:,:,0],vmin=0,vmax=255,cmap='gray')
  axes[1,col].set_title('true '+m.CLASSES[shift['labels'][i]]+'\npred '+m.CLASSES[shifted['predictions'][i]])
  for row in [0,1]:axes[row,col].axis('off')
 fig.savefig(ROOT/'outputs/errors.png',dpi=140)
 fig.savefig(ROOT/'outputs/errors.svg')
 plt.close(fig)
 results['curves']=curves
if a.report:
 a.report.write_text(json.dumps(dict(schemaVersion=1,status='passed',stage=a.stage,implementationHash=hashlib.sha256(path.read_bytes()).hexdigest(),checkerHash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),environment=dict(python=platform.python_version(),jax=jax.__version__,numpy=np.__version__,backend=jax.default_backend(),device=str(jax.devices()[0])),results=results),indent=2)+'\n')
