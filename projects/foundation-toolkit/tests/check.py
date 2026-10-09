import argparse
import importlib.util
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import numpy as np
import jax
import jax.numpy as jnp

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--implementation',default='solution')
p.add_argument('--stage',default='all')
p.add_argument('--resume')
p.add_argument('--output')
args=p.parse_args()
path=ROOT/'solution/toolkit.py' if args.implementation=='solution' else Path(args.implementation).resolve()
spec=importlib.util.spec_from_file_location('learner',path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
if args.resume:
    final,positions=m.simulate(m.load_state(args.resume),7)
    with Path(args.output).open('wb') as f:np.savez(f,**{k:np.asarray(v) for k,v in final.items()},trajectory=np.asarray(positions))
    sys.exit()
last=4 if args.stage=='all' else int(args.stage)
assert 1<=last<=4

def reject(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('invalid input accepted')

report=m.environment_report()
assert report['backend']==jax.default_backend() and report['devices']
assert report['input_shape']==[2,3] and report['output_shape']==[2] and report['dtype']=='float32'
assert report['row_sums']==[3,12] and all(report[k] for k in ['python','jax','numpy'])
print('PASS stage 1: actual completed output, independent row sums and environment report',flush=True)
if last==1:sys.exit()
for x,mask in [(np.array([[1,4],[3,4],[99,99]],np.float32),np.array([True,True,False])),
               (np.array([[7,2]],np.float32),np.array([True])),
               (np.random.default_rng(8).normal(size=(9,3)).astype(np.float32),np.array([True]*7+[False]*2))]:
    original=x.copy()
    result=m.normalize_columns(x,mask)
    selected=x[mask].astype(np.float64)
    mean=selected.mean(0)
    scale=np.sqrt(np.mean((selected-mean)**2,axis=0))
    scale=np.where(scale>0,scale,1)
    expected=np.where(mask[:,None],(x-mean)/scale,0)
    np.testing.assert_allclose(result['values'],expected,atol=2e-6)
    np.testing.assert_allclose(result['mean'],mean,atol=2e-6)
    np.testing.assert_allclose(result['scale'],scale,atol=2e-6)
    assert int(result['count'])==int(mask.sum())
    np.testing.assert_array_equal(x,original)
reject(lambda:m.normalize_columns(np.ones((2,2),np.float32),np.array([False,False])))
reject(lambda:m.normalize_columns(np.ones((2,2),np.float32),np.ones((2,1),bool)))
print('PASS stage 2: independent masked statistics, constant columns, singleton, changed shape and input purity',flush=True)
if last==2:sys.exit()
for weights in [np.array([.3,-.2],np.float32),np.array([-1.,2.],np.float32)]:
    for n in [1,7]:
        x=np.linspace(-2,1,n,dtype=np.float32)
        y=.7*x+.4
        residual=weights[0]*x+weights[1]-y
        expected=np.stack([residual*x+.2*weights[0],residual+.2*weights[1]],axis=1)
        g=m.batch_gradients(weights,x,y)
        assert g.shape==(n,2)
        np.testing.assert_allclose(np.asarray(g),expected,atol=2e-6)
        # Reducing per-example gradients agrees with an independently derived mean gradient.
        np.testing.assert_allclose(np.asarray(g).mean(0),expected.astype(np.float64).mean(0),atol=2e-6)
reject(lambda:m.batch_gradients(np.ones(2,np.float32),np.ones(3,np.float32),np.ones((3,1),np.float32)))
print('PASS stage 3: compiled per-example gradients, two parameter probes, singleton and independent polynomial derivatives',flush=True)
if last==3:sys.exit()
for seed,batch in [(3,3),(11,1),(23,5)]:
    initial=m.initial_state(seed,batch)
    original={k:np.asarray(v).copy() for k,v in initial.items()}
    key,noise_key=jax.random.split(initial['key'])
    noise=np.asarray(jax.random.normal(noise_key,(batch,2)))
    expected_v=.9*original['velocity']+.1*noise
    expected_x=original['position']+.05*expected_v
    next_state,_=m.transition(initial)
    np.testing.assert_allclose(np.asarray(next_state['velocity']),expected_v,atol=2e-7)
    np.testing.assert_allclose(np.asarray(next_state['position']),expected_x,atol=2e-7)
    np.testing.assert_array_equal(next_state['key'],key)
    final,trajectory=m.simulate(initial,12)
    prefix,_=m.simulate(initial,5)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint=Path(tmp)/'state.npz'
        output=Path(tmp)/'resumed.npz'
        m.save_state(checkpoint,prefix)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--implementation',str(path),'--resume',str(checkpoint),'--output',str(output)],check=True,timeout=30,capture_output=True,text=True)
        with np.load(output,allow_pickle=False) as resumed:
            for k in final:np.testing.assert_array_equal(np.asarray(final[k]),resumed[k])
            np.testing.assert_array_equal(np.asarray(trajectory[5:]),resumed['trajectory'])
    for k in initial:np.testing.assert_array_equal(initial[k],original[k])
    changed,_=m.simulate(m.initial_state(seed+1,batch),12)
    assert not np.allclose(changed['position'],final['position'])
with tempfile.TemporaryDirectory() as tmp:
    original={k:np.asarray(v) for k,v in m.initial_state().items()}
    for field,bad in [('step',np.array(2**32,np.int64)),('position',original['position'].astype(np.float64)),('key',original['key'].astype(np.uint64))]:
        path=Path(tmp)/'bad.npz'
        np.savez(path,**{**original,field:bad})
        reject(lambda:m.load_state(path))
reject(lambda:m.simulate(m.initial_state(),0))
reject(lambda:m.simulate({**m.initial_state(),'key':jnp.zeros(3,jnp.uint32)},2))
print('PASS stage 4: independent update order, scan trajectory, three seeds/shapes, fresh-process key/state replay and purity',flush=True)
print(json.dumps(report,indent=2))
