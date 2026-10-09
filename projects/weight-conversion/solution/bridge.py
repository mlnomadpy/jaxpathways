"""Explicit CPU PyTorch -> Flax NNX mapping; no generic architecture converter."""
# Import numpy for this computation.
import numpy as np
import jax
import jax.numpy as jnp
import torch
from flax import nnx

# Define `TorchModel` module / container with explicit state and forward pass:
class TorchModel(torch.nn.Module):
    # Function `__init__(self, eps)` implementing this stage's computation:
    def __init__(self, eps=1e-5):
        # Run `super` to perform the next check or state transition.
        super().__init__()
        # Run `torch.nn.Linear` to compute `self.hidden`.
        self.hidden=torch.nn.Linear(3,5)
        # Run `torch.nn.LayerNorm` to compute `self.norm`.
        self.norm=torch.nn.LayerNorm(5,eps=eps)
        # Run `torch.nn.Linear` to compute `self.out`.
        self.out=torch.nn.Linear(5,2)
    # Function `forward(self, x)` implementing this stage's computation:
    def forward(self,x):
        # Run `self.hidden` to compute `h`.
        # Run `self.norm` to compute `n`.
        # Run `torch.nn.functional.gelu` to compute `a`.
        h=self.hidden(x)
        n=self.norm(h)
        a=torch.nn.functional.gelu(n,approximate='none')
        # Return `{'hidden': h, 'norm': n, 'activation': a, 'output': self.out(a)}` to the caller.
        return {'hidden':h,'norm':n,'activation':a,'output':self.out(a)}

# Define `FlaxModel` module / container with explicit state and forward pass:
class FlaxModel(nnx.Module):
    # Function `__init__(self, eps)` implementing this stage's computation:
    def __init__(self,eps=1e-5):
        # Run `nnx.Linear` to compute `self.hidden`.
        self.hidden=nnx.Linear(3,5,rngs=nnx.Rngs(0))
        # Run `nnx.LayerNorm` to compute `self.norm`.
        self.norm=nnx.LayerNorm(5,epsilon=eps,use_fast_variance=False,rngs=nnx.Rngs(1))
        # Run `nnx.Linear` to compute `self.out`.
        self.out=nnx.Linear(5,2,rngs=nnx.Rngs(2))
    # Function `__call__(self, x)` implementing this stage's computation:
    def __call__(self,x):
        # Run `self.hidden` to compute `h`.
        # Run `self.norm` to compute `n`.
        # Apply nonlinear activation or probability normalization to compute `a`.
        h=self.hidden(x)
        n=self.norm(h)
        a=jax.nn.gelu(n,approximate=False)
        # Return `{'hidden': h, 'norm': n, 'activation': a, 'output': self.out(a)}` to the caller.
        return {'hidden':h,'norm':n,'activation':a,'output':self.out(a)}

# Function `convert(state, eps)` implementing this stage's computation:
def convert(state,eps=1e-5):
    # Construct dictionary `shapes` with the structured fields for this stage.
    shapes={'hidden.weight':(5,3),'hidden.bias':(5,),'norm.weight':(5,),'norm.bias':(5,),'out.weight':(2,5),'out.bias':(2,)}
    # Guard input contract (`set(state) != set(shapes)`) and fail fast if violated.
    if set(state)!=set(shapes):raise ValueError('missing or unexpected state key')
    # Construct dictionary `arrays` with the structured fields for this stage.
    arrays={}
    # Iterate over `(name, shape)` to step through the computation:
    for name,shape in shapes.items():
        # Compute `value` as `state[name].detach().cpu().numpy()`.
        value=state[name].detach().cpu().numpy()
        # Guard input contract (`value.shape != shape or value.dtype != np.float32 or (not np.isfinite(value).all())`) and fail fast if violated.
        if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('unexpected shape, dtype or nonfinite tensor: '+name)
        # Run `value.copy` to compute `arrays[name]`.
        arrays[name]=value.copy()
    # Run `FlaxModel` to compute `target`.
    target=FlaxModel(eps)
    # Create device-backed JAX array `target.hidden.kernel[...]`.
    target.hidden.kernel[...]=jnp.asarray(arrays['hidden.weight'].T)
    # Create device-backed JAX array `target.hidden.bias[...]`.
    target.hidden.bias[...]=jnp.asarray(arrays['hidden.bias'])
    # Create device-backed JAX array `target.norm.scale[...]`.
    target.norm.scale[...]=jnp.asarray(arrays['norm.weight'])
    # Create device-backed JAX array `target.norm.bias[...]`.
    target.norm.bias[...]=jnp.asarray(arrays['norm.bias'])
    # Create device-backed JAX array `target.out.kernel[...]`.
    target.out.kernel[...]=jnp.asarray(arrays['out.weight'].T)
    # Create device-backed JAX array `target.out.bias[...]`.
    target.out.bias[...]=jnp.asarray(arrays['out.bias'])
    # Return `target` to the caller.
    return target

# Function `error_report(reference, actual, atol, rtol)` implementing this stage's computation:
def error_report(reference,actual,atol=2e-6,rtol=2e-5):
    # Convert `reference` to a host NumPy array for inspection or verification.
    # Convert `actual` to a host NumPy array for inspection or verification.
    reference=np.asarray(reference,dtype=np.float64)
    actual=np.asarray(actual,dtype=np.float64)
    # Guard input contract (`reference.shape != actual.shape or not np.isfinite(reference).all() or (not np.isfinite(actual).all())`) and fail fast if violated.
    if reference.shape!=actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('shape or finite-value mismatch')
    # Guard input contract (`min(atol, rtol) < 0 or not np.isfinite([atol, rtol]).all()`) and fail fast if violated.
    if min(atol,rtol)<0 or not np.isfinite([atol,rtol]).all():raise ValueError('invalid tolerances')
    # Run `np.abs` to compute `absolute`.
    absolute=np.abs(actual-reference)
    # Compute `budget` as `atol+rtol*np.abs(reference)`.
    budget=atol+rtol*np.abs(reference)
    # Return `{'max_abs': float(absolute.max()), 'relative_l2': float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-12)), 'passed': bool(np.all(absolute <= budget))}` to the caller.
    return {'max_abs':float(absolute.max()),'relative_l2':float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference),1e-12)),
            'passed':bool(np.all(absolute<=budget))}

# Function `save_flax(model, path)` implementing this stage's computation:
def save_flax(model,path):
    # Convert `` to a host NumPy array for inspection or verification.
    np.savez(path,hidden_kernel=np.asarray(model.hidden.kernel[...]),hidden_bias=np.asarray(model.hidden.bias[...]),
             norm_scale=np.asarray(model.norm.scale[...]),norm_bias=np.asarray(model.norm.bias[...]),
             out_kernel=np.asarray(model.out.kernel[...]),out_bias=np.asarray(model.out.bias[...]),
             epsilon=np.array(model.norm.epsilon),schema=np.array(1))

# Function `load_flax(path)` implementing this stage's computation:
def load_flax(path):
    # Construct dictionary `shapes` with the structured fields for this stage.
    shapes={'hidden_kernel':(3,5),'hidden_bias':(5,),'norm_scale':(5,),'norm_bias':(5,),'out_kernel':(5,2),'out_bias':(2,)}
    # Enter `np.load(path, allow_pickle=False)` context block:
    with np.load(path,allow_pickle=False) as archive:
        # Guard input contract (`set(archive.files) != set(shapes) | {'epsilon', 'schema'}`) and fail fast if violated.
        if set(archive.files)!=set(shapes)|{'epsilon','schema'}:raise ValueError('unexpected archive schema')
        # Guard input contract (`archive['schema'].shape != () or int(archive['schema']) != 1`) and fail fast if violated.
        if archive['schema'].shape!=() or int(archive['schema'])!=1:raise ValueError('unknown schema')
        # Guard input contract (`archive['epsilon'].shape != ()`) and fail fast if violated.
        if archive['epsilon'].shape!=():raise ValueError('epsilon must be scalar')
        # Evaluate `archive['epsilon']` and convert the result into Python scalar/collection `eps`.
        eps=float(archive['epsilon'])
        # Guard input contract (`not np.isfinite(eps) or eps <= 0`) and fail fast if violated.
        if not np.isfinite(eps) or eps<=0:raise ValueError('invalid epsilon')
        # Construct dictionary `arrays` with the structured fields for this stage.
        arrays={}
        # Iterate over `(key, shape)` to step through the computation:
        for key,shape in shapes.items():
            # Compute `value` as `archive[key]`.
            value=archive[key]
            # Guard input contract (`value.shape != shape or value.dtype != np.float32 or (not np.isfinite(value).all())`) and fail fast if violated.
            if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('invalid array '+key)
            # Run `value.copy` to compute `arrays[key]`.
            arrays[key]=value.copy()
    # Run `FlaxModel` to compute `model`.
    model=FlaxModel(eps)
    # Loop over `(module, name, key)` in `[(model.hidden, 'kernel', 'hidden_kernel'), (model.hidden, 'bias', 'hidden_bias'), (model.norm, 'scale', 'norm_scale'), (model.norm, 'bias', 'norm_bias'), (model.out, 'kernel', 'out_kernel'), (model.out, 'bias', 'out_bias')]`:
    for module,name,key in [(model.hidden,'kernel','hidden_kernel'),(model.hidden,'bias','hidden_bias'),(model.norm,'scale','norm_scale'),(model.norm,'bias','norm_bias'),(model.out,'kernel','out_kernel'),(model.out,'bias','out_bias')]:
        # Create device-backed JAX array `getattr(module, name)[...]`.
        getattr(module,name)[...]=jnp.asarray(arrays[key])
    # Return `model` to the caller.
    return model
