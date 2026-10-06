"""Convert PyTorch weights to Flax and locate numerical errors: worked experiments and reference solutions. CPU checks."""

# Make the source boundaries inspectable
"""Explicit CPU PyTorch -> Flax NNX mapping; no generic architecture converter."""
import numpy as np
import jax
import jax.numpy as jnp
import torch
from flax import nnx

class TorchModel(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.hidden=torch.nn.Linear(3,5)
        self.norm=torch.nn.LayerNorm(5,eps=eps)
        self.out=torch.nn.Linear(5,2)
    def forward(self,x):
        h=self.hidden(x);n=self.norm(h);a=torch.nn.functional.gelu(n,approximate='none')
        return {'hidden':h,'norm':n,'activation':a,'output':self.out(a)}

assert TorchModel().eval()(torch.zeros((2,3)))['output'].shape == (2,2)


# Write the same operations in Flax
class FlaxModel(nnx.Module):
    def __init__(self,eps=1e-5):
        self.hidden=nnx.Linear(3,5,rngs=nnx.Rngs(0))
        self.norm=nnx.LayerNorm(5,epsilon=eps,use_fast_variance=False,rngs=nnx.Rngs(1))
        self.out=nnx.Linear(5,2,rngs=nnx.Rngs(2))
    def __call__(self,x):
        h=self.hidden(x);n=self.norm(h);a=jax.nn.gelu(n,approximate=False)
        return {'hidden':h,'norm':n,'activation':a,'output':self.out(a)}

assert FlaxModel()(jnp.zeros((2,3)))['output'].shape == (2,2)


# Map each parameter and declare tolerances
def convert(state,eps=1e-5):
    shapes={'hidden.weight':(5,3),'hidden.bias':(5,),'norm.weight':(5,),'norm.bias':(5,),'out.weight':(2,5),'out.bias':(2,)}
    if set(state)!=set(shapes):raise ValueError('missing or unexpected state key')
    arrays={}
    for name,shape in shapes.items():
        value=state[name].detach().cpu().numpy()
        if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('unexpected shape, dtype or nonfinite tensor: '+name)
        arrays[name]=value.copy()
    target=FlaxModel(eps)
    target.hidden.kernel[...]=jnp.asarray(arrays['hidden.weight'].T)
    target.hidden.bias[...]=jnp.asarray(arrays['hidden.bias'])
    target.norm.scale[...]=jnp.asarray(arrays['norm.weight'])
    target.norm.bias[...]=jnp.asarray(arrays['norm.bias'])
    target.out.kernel[...]=jnp.asarray(arrays['out.weight'].T)
    target.out.bias[...]=jnp.asarray(arrays['out.bias'])
    return target

def error_report(reference,actual,atol=2e-6,rtol=2e-5):
    reference=np.asarray(reference,dtype=np.float64);actual=np.asarray(actual,dtype=np.float64)
    if reference.shape!=actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('shape or finite-value mismatch')
    if min(atol,rtol)<0 or not np.isfinite([atol,rtol]).all():raise ValueError('invalid tolerances')
    absolute=np.abs(actual-reference)
    budget=atol+rtol*np.abs(reference)
    return {'max_abs':float(absolute.max()),'relative_l2':float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference),1e-12)),
            'passed':bool(np.all(absolute<=budget))}

step_source = TorchModel().eval()
step_target = convert(step_source.state_dict())
step_input = np.array([[1., -2., .5]], np.float32)
step_expected = step_input @ step_source.hidden.weight.detach().numpy().T + step_source.hidden.bias.detach().numpy()
np.testing.assert_allclose(step_target(jnp.asarray(step_input))['hidden'], step_expected, rtol=2e-5, atol=2e-6)


# Save and reload the converted state
def save_flax(model,path):
    np.savez(path,hidden_kernel=np.asarray(model.hidden.kernel[...]),hidden_bias=np.asarray(model.hidden.bias[...]),
             norm_scale=np.asarray(model.norm.scale[...]),norm_bias=np.asarray(model.norm.bias[...]),
             out_kernel=np.asarray(model.out.kernel[...]),out_bias=np.asarray(model.out.bias[...]),
             epsilon=np.array(model.norm.epsilon),schema=np.array(1))

def load_flax(path):
    shapes={'hidden_kernel':(3,5),'hidden_bias':(5,),'norm_scale':(5,),'norm_bias':(5,),'out_kernel':(5,2),'out_bias':(2,)}
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=set(shapes)|{'epsilon','schema'}:raise ValueError('unexpected archive schema')
        if archive['schema'].shape!=() or int(archive['schema'])!=1:raise ValueError('unknown schema')
        if archive['epsilon'].shape!=():raise ValueError('epsilon must be scalar')
        eps=float(archive['epsilon'])
        if not np.isfinite(eps) or eps<=0:raise ValueError('invalid epsilon')
        arrays={}
        for key,shape in shapes.items():
            value=archive[key]
            if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('invalid array '+key)
            arrays[key]=value.copy()
    model=FlaxModel(eps)
    for module,name,key in [(model.hidden,'kernel','hidden_kernel'),(model.hidden,'bias','hidden_bias'),(model.norm,'scale','norm_scale'),(model.norm,'bias','norm_bias'),(model.out,'kernel','out_kernel'),(model.out,'bias','out_bias')]:
        getattr(module,name)[...]=jnp.asarray(arrays[key])
    return model

import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as step_folder:
    step_path = Path(step_folder) / 'roundtrip.npz'
    save_flax(step_target, step_path)
    step_reloaded = load_flax(step_path)
    np.testing.assert_allclose(step_reloaded(jnp.asarray(step_input))['output'], step_target(jnp.asarray(step_input))['output'], rtol=2e-5, atol=2e-6)


# Compare input regimes and reproduce a mismatch
import tempfile
from pathlib import Path
torch.set_num_threads(1)
torch.manual_seed(9)
source=TorchModel().eval()
# A real local state_dict file, loaded using the tensor-only loading option.
with tempfile.TemporaryDirectory() as folder:
    checkpoint=Path(folder)/'weights.pt';torch.save(source.state_dict(),checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    target=convert(state)
    # Save the converted weights independently of the live PyTorch object.
    converted=Path(folder)/'flax-weights.npz'
    save_flax(target,converted)
    target=load_flax(converted)
    errors={name:[] for name in ['hidden','norm','activation','output']}
    for batch,scale in [(1,1.),(7,1e-3),(5,4.)]:
        inputs=np.random.default_rng(batch).normal(size=(batch,3)).astype(np.float32)*scale
        tx=torch.tensor(inputs,requires_grad=True);torch_values=source(tx);flax_values=target(jnp.asarray(inputs))
        for name in errors:
            report=error_report(torch_values[name].detach().numpy(),flax_values[name]);assert report['passed'],(name,report)
            errors[name].append(report['max_abs'])
        torch_values['output'].sum().backward()
        input_gradient=jax.grad(lambda z:jnp.sum(target(z)['output']))(jnp.asarray(inputs))
        assert error_report(tx.grad.numpy(),input_gradient,atol=5e-6,rtol=5e-5)['passed']
    wrong=convert(state,eps=.1)
    reference={k:v.detach().numpy() for k,v in source(torch.tensor(inputs)).items()}
    wrong_reports={k:error_report(reference[k],v) for k,v in wrong(jnp.asarray(inputs)).items()}
    assert wrong_reports['hidden']['passed'] and not wrong_reports['norm']['passed']
    for bad in [dict(state,unexpected=torch.zeros(1)),{k:v for k,v in state.items() if k!='norm.bias'}]:
        try:convert(bad)
        except ValueError:pass
        else:raise AssertionError('incomplete mapping accepted')
print('Maximum absolute error per layer:',{k:max(v) for k,v in errors.items()})
print('Wrong epsilon: first mismatch is norm; intermediate outputs and input gradients verified on CPU.')


"""Explicit CPU PyTorch -> Flax NNX mapping; no generic architecture converter."""
import numpy as np
import jax
import jax.numpy as jnp
import torch
from flax import nnx

class TorchModel(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.hidden=torch.nn.Linear(3,5)
        self.norm=torch.nn.LayerNorm(5,eps=eps)
        self.out=torch.nn.Linear(5,2)
    def forward(self,x):
        h=self.hidden(x);n=self.norm(h);a=torch.nn.functional.gelu(n,approximate='none')
        return {'hidden':h,'norm':n,'activation':a,'output':self.out(a)}

class FlaxModel(nnx.Module):
    def __init__(self,eps=1e-5):
        self.hidden=nnx.Linear(3,5,rngs=nnx.Rngs(0))
        self.norm=nnx.LayerNorm(5,epsilon=eps,use_fast_variance=False,rngs=nnx.Rngs(1))
        self.out=nnx.Linear(5,2,rngs=nnx.Rngs(2))
    def __call__(self,x):
        h=self.hidden(x);n=self.norm(h);a=jax.nn.gelu(n,approximate=False)
        return {'hidden':h,'norm':n,'activation':a,'output':self.out(a)}

def convert(state,eps=1e-5):
    shapes={'hidden.weight':(5,3),'hidden.bias':(5,),'norm.weight':(5,),'norm.bias':(5,),'out.weight':(2,5),'out.bias':(2,)}
    if set(state)!=set(shapes):raise ValueError('missing or unexpected state key')
    arrays={}
    for name,shape in shapes.items():
        value=state[name].detach().cpu().numpy()
        if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('unexpected shape, dtype or nonfinite tensor: '+name)
        arrays[name]=value.copy()
    target=FlaxModel(eps)
    target.hidden.kernel[...]=jnp.asarray(arrays['hidden.weight'].T)
    target.hidden.bias[...]=jnp.asarray(arrays['hidden.bias'])
    target.norm.scale[...]=jnp.asarray(arrays['norm.weight'])
    target.norm.bias[...]=jnp.asarray(arrays['norm.bias'])
    target.out.kernel[...]=jnp.asarray(arrays['out.weight'].T)
    target.out.bias[...]=jnp.asarray(arrays['out.bias'])
    return target

def error_report(reference,actual,atol=2e-6,rtol=2e-5):
    reference=np.asarray(reference,dtype=np.float64);actual=np.asarray(actual,dtype=np.float64)
    if reference.shape!=actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('shape or finite-value mismatch')
    if min(atol,rtol)<0 or not np.isfinite([atol,rtol]).all():raise ValueError('invalid tolerances')
    absolute=np.abs(actual-reference)
    budget=atol+rtol*np.abs(reference)
    return {'max_abs':float(absolute.max()),'relative_l2':float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference),1e-12)),
            'passed':bool(np.all(absolute<=budget))}

def save_flax(model,path):
    np.savez(path,hidden_kernel=np.asarray(model.hidden.kernel[...]),hidden_bias=np.asarray(model.hidden.bias[...]),
             norm_scale=np.asarray(model.norm.scale[...]),norm_bias=np.asarray(model.norm.bias[...]),
             out_kernel=np.asarray(model.out.kernel[...]),out_bias=np.asarray(model.out.bias[...]),
             epsilon=np.array(model.norm.epsilon),schema=np.array(1))

def load_flax(path):
    shapes={'hidden_kernel':(3,5),'hidden_bias':(5,),'norm_scale':(5,),'norm_bias':(5,),'out_kernel':(5,2),'out_bias':(2,)}
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=set(shapes)|{'epsilon','schema'}:raise ValueError('unexpected archive schema')
        if archive['schema'].shape!=() or int(archive['schema'])!=1:raise ValueError('unknown schema')
        if archive['epsilon'].shape!=():raise ValueError('epsilon must be scalar')
        eps=float(archive['epsilon'])
        if not np.isfinite(eps) or eps<=0:raise ValueError('invalid epsilon')
        arrays={}
        for key,shape in shapes.items():
            value=archive[key]
            if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('invalid array '+key)
            arrays[key]=value.copy()
    model=FlaxModel(eps)
    for module,name,key in [(model.hidden,'kernel','hidden_kernel'),(model.hidden,'bias','hidden_bias'),(model.norm,'scale','norm_scale'),(model.norm,'bias','norm_bias'),(model.out,'kernel','out_kernel'),(model.out,'bias','out_bias')]:
        getattr(module,name)[...]=jnp.asarray(arrays[key])
    return model

import tempfile
from pathlib import Path
torch.set_num_threads(1)
torch.manual_seed(9)
source=TorchModel().eval()
# A real local state_dict file, loaded using the tensor-only loading option.
with tempfile.TemporaryDirectory() as folder:
    checkpoint=Path(folder)/'weights.pt';torch.save(source.state_dict(),checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    target=convert(state)
    # Save the converted weights independently of the live PyTorch object.
    converted=Path(folder)/'flax-weights.npz'
    save_flax(target,converted)
    target=load_flax(converted)
    errors={name:[] for name in ['hidden','norm','activation','output']}
    for batch,scale in [(1,1.),(7,1e-3),(5,4.)]:
        inputs=np.random.default_rng(batch).normal(size=(batch,3)).astype(np.float32)*scale
        tx=torch.tensor(inputs,requires_grad=True);torch_values=source(tx);flax_values=target(jnp.asarray(inputs))
        for name in errors:
            report=error_report(torch_values[name].detach().numpy(),flax_values[name]);assert report['passed'],(name,report)
            errors[name].append(report['max_abs'])
        torch_values['output'].sum().backward()
        input_gradient=jax.grad(lambda z:jnp.sum(target(z)['output']))(jnp.asarray(inputs))
        assert error_report(tx.grad.numpy(),input_gradient,atol=5e-6,rtol=5e-5)['passed']
    wrong=convert(state,eps=.1)
    reference={k:v.detach().numpy() for k,v in source(torch.tensor(inputs)).items()}
    wrong_reports={k:error_report(reference[k],v) for k,v in wrong(jnp.asarray(inputs)).items()}
    assert wrong_reports['hidden']['passed'] and not wrong_reports['norm']['passed']
    for bad in [dict(state,unexpected=torch.zeros(1)),{k:v for k,v in state.items() if k!='norm.bias'}]:
        try:convert(bad)
        except ValueError:pass
        else:raise AssertionError('incomplete mapping accepted')
print('Maximum absolute error per layer:',{k:max(v) for k,v in errors.items()})
print('Wrong epsilon: first mismatch is norm; intermediate outputs and input gradients verified on CPU.')


# Figure data experiment
visual_data={'kind':'bar','labels':list(errors),'xlabel':'operation in execution order','ylabel':'maximum absolute output error','series':[{'label':'matched architecture (max across inputs)','y':[max(v) for v in errors.values()]},{'label':'wrong epsilon (last input batch)','y':[wrong_reports[k]['max_abs'] for k in errors]}]}

# Experiment: Inspect the wrong-epsilon signature
assert wrong_reports['hidden']['passed']
assert not wrong_reports['norm']['passed']
print('Per-layer wrong-epsilon report:',wrong_reports)

# Experiment: Test the tolerance rule near zero
near_zero = error_report([0.], [1e-6])
near_zero_fail = error_report([0.], [3e-6])
large_value = error_report([100.], [100.001])
assert near_zero['passed'] and not near_zero_fail['passed'] and large_value['passed']
print('Near-zero, excessive near-zero, large-value:', near_zero['passed'], near_zero_fail['passed'], large_value['passed'])


# Reference solution. Try the exercise before reading this.
assert error_report([0.,1.],[1e-7,1.000001])['passed']
assert not error_report([0.,1.],[1e-3,1.])['passed']
print('Near-zero acceptance and rejection verified.')

# Reference practice: Reject a plausible but wrong tensor layout
bad_state=dict(state);bad_state['hidden.weight']=state['hidden.weight'].T
try:convert(bad_state)
except ValueError:print('Wrong source layout rejected')
else:raise AssertionError('layout mismatch accepted')

# Reference practice: Check a nonuniform output sensitivity
probe_inputs = np.array([[.2, -.7, 1.1], [1., .4, -.5]], np.float32)
cotangent = np.array([[1., -.5], [2., .25]], np.float32)
probe_torch = torch.tensor(probe_inputs, requires_grad=True)
weighted_source = (source(probe_torch)['output'] * torch.tensor(cotangent)).sum()
weighted_source.backward()
weighted_target = jax.grad(lambda z: jnp.sum(target(z)['output'] * jnp.asarray(cotangent)))(jnp.asarray(probe_inputs))
vjp_report = error_report(probe_torch.grad.numpy(), weighted_target, atol=5e-6, rtol=5e-5)
assert vjp_report['passed'], vjp_report
print('Nonuniform cotangent input-gradient parity:', vjp_report)

print("PASS: deployment-08")
