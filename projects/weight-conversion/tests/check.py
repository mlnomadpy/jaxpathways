"""Changed seeds/shapes, layer localization, gradients and a fresh artifact process."""
import argparse,importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import torch
ROOT=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--implementation',default='solution');a.add_argument('--stage',default='all');a.add_argument('--restore');a.add_argument('--output');args=a.parse_args()
path=ROOT/'solution/bridge.py' if args.implementation=='solution' else Path(args.implementation).resolve()
spec=importlib.util.spec_from_file_location('learner',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if args.restore:
    restored=m.load_flax(args.restore);x=jnp.asarray([[.3,-1.,2.],[-.2,.7,0.]],jnp.float32)
    Path(args.output).write_text(json.dumps(np.asarray(restored(x)['output']).tolist()));raise SystemExit()
last=3 if args.stage=='all' else int(args.stage);assert 1<=last<=3
torch.set_num_threads(1)
def reject(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('invalid conversion accepted')
for seed in [5,21]:
    torch.manual_seed(seed);source=m.TorchModel().eval();state=source.state_dict();target=m.convert(state)
    for batch,scale in [(1,1e-3),(4,1.),(7,5.)]:
        x=np.random.default_rng(seed+batch).normal(size=(batch,3)).astype(np.float32)*scale
        tx=torch.tensor(x,requires_grad=True);torch_values=source(tx);flax_values=target(jnp.asarray(x))
        for key in torch_values:np.testing.assert_allclose(flax_values[key],torch_values[key].detach().numpy(),atol=2e-6,rtol=2e-5)
        torch_values['output'].sum().backward()
        np.testing.assert_allclose(jax.grad(lambda a:jnp.sum(target(a)['output']))(jnp.asarray(x)),tx.grad.numpy(),atol=5e-6,rtol=5e-5)
    reject(lambda:m.convert({k:v for k,v in state.items() if k!='norm.bias'}))
    reject(lambda:m.convert(dict(state,extra=torch.zeros(1))))
    reject(lambda:m.convert(dict(state,**{'hidden.weight':state['hidden.weight'].T})))
    reject(lambda:m.convert(dict(state,**{'out.bias':torch.tensor([float('nan'),0.])})))
print('PASS stage 1: actual PyTorch/Flax layers and input gradients; incomplete/bad tensors rejected',flush=True)
if last==1:raise SystemExit()
assert m.error_report([0.,1.],[1e-7,1.000001])['passed']
assert not m.error_report([0.,1.],[1e-3,1.])['passed']
r=m.error_report([1.,-2.],[1.1,-2.]);np.testing.assert_allclose(r['max_abs'],.1,atol=1e-12);np.testing.assert_allclose(r['relative_l2'],.1/np.sqrt(5),atol=1e-12)
reject(lambda:m.error_report([0.],[float('nan')]))
wrong=m.convert(state,eps=.1);x=jnp.asarray([[.3,-1.,2.],[-.2,.7,0.]],jnp.float32)
reference={k:v.detach().numpy() for k,v in source(torch.tensor(np.asarray(x))).items()}
checks={k:m.error_report(reference[k],v)['passed'] for k,v in wrong(x).items()}
assert checks['hidden'] and not checks['norm']
print('PASS stage 2: near-zero error budget and first wrong-epsilon layer localized',flush=True)
if last==2:raise SystemExit()
with tempfile.TemporaryDirectory() as folder:
    artifact=Path(folder)/'flax.npz';output=Path(folder)/'result.json';m.save_flax(target,artifact)
    subprocess.run([sys.executable,__file__,'--implementation',str(path),'--restore',str(artifact),'--output',str(output)],check=True,capture_output=True,text=True,timeout=60)
    np.testing.assert_allclose(json.loads(output.read_text()),reference['output'],atol=2e-6,rtol=2e-5)
    with np.load(artifact,allow_pickle=False) as z:broken={k:z[k] for k in z.files if k!='norm_scale'}
    np.savez(artifact,**broken);reject(lambda:m.load_flax(artifact))
print('PASS stage 3: converted artifact inference in a fresh process; incomplete archive rejected',flush=True)
