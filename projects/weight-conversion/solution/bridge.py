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
