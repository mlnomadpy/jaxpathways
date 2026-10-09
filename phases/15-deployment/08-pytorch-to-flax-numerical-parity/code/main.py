"""Convert PyTorch weights to Flax and locate numerical errors: worked experiments and reference solutions. CPU checks."""

# Make the source boundaries inspectable
# Step 1: Make the source boundaries inspectable
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
        h=self.hidden(x);n=self.norm(h);a=torch.nn.functional.gelu(n,approximate='none')
        # Return `{'hidden': h, 'norm': n, 'activation': a, 'output': self.out(a)}` to the caller.
        return {'hidden':h,'norm':n,'activation':a,'output':self.out(a)}

# Verify that the output tensor shape matches our prediction.
assert TorchModel().eval()(torch.zeros((2,3)))['output'].shape == (2,2)

# Write the same operations in Flax
# Step 2 — Write the same operations in Flax: The probe again returns shape (2,2).
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
        h=self.hidden(x);n=self.norm(h);a=jax.nn.gelu(n,approximate=False)
        # Return `{'hidden': h, 'norm': n, 'activation': a, 'output': self.out(a)}` to the caller.
        return {'hidden':h,'norm':n,'activation':a,'output':self.out(a)}

# Verify that the output tensor shape matches our prediction.
assert FlaxModel()(jnp.zeros((2,3)))['output'].shape == (2,2)

# Map each parameter and declare tolerances
# Step 3 — Map each parameter and declare tolerances: The hidden-layer check passes after transposing the source kernel.
def convert(state,eps=1e-5):
    # Evaluate `shapes` from the current inputs and state.
    shapes={'hidden.weight':(5,3),'hidden.bias':(5,),'norm.weight':(5,),'norm.bias':(5,),'out.weight':(2,5),'out.bias':(2,)}
    # Guard input contract (`set(state) != set(shapes)`) and fail fast if violated.
    if set(state)!=set(shapes):raise ValueError('missing or unexpected state key')
    # Evaluate `arrays` from the current inputs and state.
    arrays={}
    # Iterate over `(name, shape)` to step through the computation:
    for name,shape in shapes.items():
        # Evaluate `value` from the current inputs and state.
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
    reference=np.asarray(reference,dtype=np.float64);actual=np.asarray(actual,dtype=np.float64)
    # Guard input contract (`reference.shape != actual.shape or not np.isfinite(reference).all() or (not np.isfinite(actual).all())`) and fail fast if violated.
    if reference.shape!=actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('shape or finite-value mismatch')
    # Guard input contract (`min(atol, rtol) < 0 or not np.isfinite([atol, rtol]).all()`) and fail fast if violated.
    if min(atol,rtol)<0 or not np.isfinite([atol,rtol]).all():raise ValueError('invalid tolerances')
    # Run `np.abs` to compute `absolute`.
    absolute=np.abs(actual-reference)
    # Evaluate `budget` from the current inputs and state.
    budget=atol+rtol*np.abs(reference)
    # Return `{'max_abs': float(absolute.max()), 'relative_l2': float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-12)), 'passed': bool(np.all(absolute <= budget))}` to the caller.
    return {'max_abs':float(absolute.max()),'relative_l2':float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference),1e-12)),
            'passed':bool(np.all(absolute<=budget))}

# Run `TorchModel` to compute `step_source`.
step_source = TorchModel().eval()
# Run `convert` to compute `step_target`.
step_target = convert(step_source.state_dict())
# Initialize array `step_input` with explicit values and shape.
step_input = np.array([[1., -2., .5]], np.float32)
# Perform matrix contraction / projection to compute `step_expected`.
step_expected = step_input @ step_source.hidden.weight.detach().numpy().T + step_source.hidden.bias.detach().numpy()
# Create device-backed JAX array ``.
np.testing.assert_allclose(step_target(jnp.asarray(step_input))['hidden'], step_expected, rtol=2e-5, atol=2e-6)

# Save and reload the converted state
# Step 4 — Save and reload the converted state: The temporary round trip preserves a probe output.
def save_flax(model,path):
    # Convert `` to a host NumPy array for inspection or verification.
    np.savez(path,hidden_kernel=np.asarray(model.hidden.kernel[...]),hidden_bias=np.asarray(model.hidden.bias[...]),
             norm_scale=np.asarray(model.norm.scale[...]),norm_bias=np.asarray(model.norm.bias[...]),
             out_kernel=np.asarray(model.out.kernel[...]),out_bias=np.asarray(model.out.bias[...]),
             epsilon=np.array(model.norm.epsilon),schema=np.array(1))

# Function `load_flax(path)` implementing this stage's computation:
def load_flax(path):
    # Evaluate `shapes` from the current inputs and state.
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
        # Evaluate `arrays` from the current inputs and state.
        arrays={}
        # Iterate over `(key, shape)` to step through the computation:
        for key,shape in shapes.items():
            # Evaluate `value` from the current inputs and state.
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

# Import tempfile for this computation.
import tempfile
from pathlib import Path
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory() as step_folder:
    # Read or serialize artifact data on disk (`step_path`).
    step_path = Path(step_folder) / 'roundtrip.npz'
    # Run `save_flax` to perform the next check or state transition.
    save_flax(step_target, step_path)
    # Run `load_flax` to compute `step_reloaded`.
    step_reloaded = load_flax(step_path)
    # Create device-backed JAX array ``.
    np.testing.assert_allclose(step_reloaded(jnp.asarray(step_input))['output'], step_target(jnp.asarray(step_input))['output'], rtol=2e-5, atol=2e-6)

# Compare input regimes and reproduce a mismatch
# Step 5 — Compare input regimes and reproduce a mismatch: The original epsilon passes all declared gates.
# Import tempfile for this computation.
import tempfile
from pathlib import Path
# Run `torch.set_num_threads` to perform the next check or state transition.
torch.set_num_threads(1)
# Run `torch.manual_seed` to perform the next check or state transition.
torch.manual_seed(9)
# Run `TorchModel` to compute `source`.
source=TorchModel().eval()
# A real local state_dict file, loaded using the tensor-only loading option.
with tempfile.TemporaryDirectory() as folder:
    # Read or serialize artifact data on disk (`checkpoint`).
    # Execute the next step of the computation.
    checkpoint=Path(folder)/'weights.pt';torch.save(source.state_dict(),checkpoint)
    # Run `torch.load` to compute `state`.
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    # Run `convert` to compute `target`.
    target=convert(state)
    # Save the converted weights independently of the live PyTorch object.
    converted=Path(folder)/'flax-weights.npz'
    # Run `save_flax` to perform the next check or state transition.
    save_flax(target,converted)
    # Run `load_flax` to compute `target`.
    target=load_flax(converted)
    # Evaluate `errors` from the current inputs and state.
    errors={name:[] for name in ['hidden','norm','activation','output']}
    # Loop over `(batch, scale)` in `[(1, 1.0), (7, 0.001), (5, 4.0)]`:
    for batch,scale in [(1,1.),(7,1e-3),(5,4.)]:
        # Cast or evaluate `inputs` in explicit floating-point precision.
        inputs=np.random.default_rng(batch).normal(size=(batch,3)).astype(np.float32)*scale
        # Run `torch.tensor` to compute `tx`.
        # Run `source` to compute `torch_values`.
        # Create device-backed JAX array `flax_values`.
        tx=torch.tensor(inputs,requires_grad=True);torch_values=source(tx);flax_values=target(jnp.asarray(inputs))
        # Loop over `name` in `errors`:
        for name in errors:
            # Run `error_report` to compute `report`.
            # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
            report=error_report(torch_values[name].detach().numpy(),flax_values[name]);assert report['passed'],(name,report)
            # Execute the next step of the computation.
            errors[name].append(report['max_abs'])
        # Reduce across the target axis to summarize ``.
        torch_values['output'].sum().backward()
        # Create device-backed JAX array `input_gradient`.
        input_gradient=jax.grad(lambda z:jnp.sum(target(z)['output']))(jnp.asarray(inputs))
        # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
        assert error_report(tx.grad.numpy(),input_gradient,atol=5e-6,rtol=5e-5)['passed']
    # Run `convert` to compute `wrong`.
    wrong=convert(state,eps=.1)
    # Evaluate `reference` from the current inputs and state.
    reference={k:v.detach().numpy() for k,v in source(torch.tensor(inputs)).items()}
    # Create device-backed JAX array `wrong_reports`.
    wrong_reports={k:error_report(reference[k],v) for k,v in wrong(jnp.asarray(inputs)).items()}
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
    assert wrong_reports['hidden']['passed'] and not wrong_reports['norm']['passed']
    # Loop over `bad` in `[dict(state, unexpected=torch.zeros(1)), {k: v for k, v in state.items() if k != 'norm.bias'}]`:
    for bad in [dict(state,unexpected=torch.zeros(1)),{k:v for k,v in state.items() if k!='norm.bias'}]:
        try:convert(bad)
        except ValueError:pass
        else:raise AssertionError('incomplete mapping accepted')
# Print the observed values to compare against the expected result.
print('Maximum absolute error per layer:',{k:max(v) for k,v in errors.items()})
# Print diagnostic summary of the computed outputs.
print('Wrong epsilon: first mismatch is norm; intermediate outputs and input gradients verified on CPU.')

# Complete runnable example (deployment-08)
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
        h=self.hidden(x);n=self.norm(h);a=torch.nn.functional.gelu(n,approximate='none')
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
        h=self.hidden(x);n=self.norm(h);a=jax.nn.gelu(n,approximate=False)
        # Return `{'hidden': h, 'norm': n, 'activation': a, 'output': self.out(a)}` to the caller.
        return {'hidden':h,'norm':n,'activation':a,'output':self.out(a)}

# Function `convert(state, eps)` implementing this stage's computation:
def convert(state,eps=1e-5):
    # Evaluate `shapes` from the current inputs and state.
    shapes={'hidden.weight':(5,3),'hidden.bias':(5,),'norm.weight':(5,),'norm.bias':(5,),'out.weight':(2,5),'out.bias':(2,)}
    # Guard input contract (`set(state) != set(shapes)`) and fail fast if violated.
    if set(state)!=set(shapes):raise ValueError('missing or unexpected state key')
    # Evaluate `arrays` from the current inputs and state.
    arrays={}
    # Iterate over `(name, shape)` to step through the computation:
    for name,shape in shapes.items():
        # Evaluate `value` from the current inputs and state.
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
    reference=np.asarray(reference,dtype=np.float64);actual=np.asarray(actual,dtype=np.float64)
    # Guard input contract (`reference.shape != actual.shape or not np.isfinite(reference).all() or (not np.isfinite(actual).all())`) and fail fast if violated.
    if reference.shape!=actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('shape or finite-value mismatch')
    # Guard input contract (`min(atol, rtol) < 0 or not np.isfinite([atol, rtol]).all()`) and fail fast if violated.
    if min(atol,rtol)<0 or not np.isfinite([atol,rtol]).all():raise ValueError('invalid tolerances')
    # Run `np.abs` to compute `absolute`.
    absolute=np.abs(actual-reference)
    # Evaluate `budget` from the current inputs and state.
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
    # Evaluate `shapes` from the current inputs and state.
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
        # Evaluate `arrays` from the current inputs and state.
        arrays={}
        # Iterate over `(key, shape)` to step through the computation:
        for key,shape in shapes.items():
            # Evaluate `value` from the current inputs and state.
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

# Import tempfile for this computation.
import tempfile
from pathlib import Path
# Run `torch.set_num_threads` to perform the next check or state transition.
torch.set_num_threads(1)
# Run `torch.manual_seed` to perform the next check or state transition.
torch.manual_seed(9)
# Run `TorchModel` to compute `source`.
source=TorchModel().eval()
# A real local state_dict file, loaded using the tensor-only loading option.
with tempfile.TemporaryDirectory() as folder:
    # Read or serialize artifact data on disk (`checkpoint`).
    # Execute the next step of the computation.
    checkpoint=Path(folder)/'weights.pt';torch.save(source.state_dict(),checkpoint)
    # Run `torch.load` to compute `state`.
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    # Run `convert` to compute `target`.
    target=convert(state)
    # Save the converted weights independently of the live PyTorch object.
    converted=Path(folder)/'flax-weights.npz'
    # Run `save_flax` to perform the next check or state transition.
    save_flax(target,converted)
    # Run `load_flax` to compute `target`.
    target=load_flax(converted)
    # Evaluate `errors` from the current inputs and state.
    errors={name:[] for name in ['hidden','norm','activation','output']}
    # Loop over `(batch, scale)` in `[(1, 1.0), (7, 0.001), (5, 4.0)]`:
    for batch,scale in [(1,1.),(7,1e-3),(5,4.)]:
        # Cast or evaluate `inputs` in explicit floating-point precision.
        inputs=np.random.default_rng(batch).normal(size=(batch,3)).astype(np.float32)*scale
        # Run `torch.tensor` to compute `tx`.
        # Run `source` to compute `torch_values`.
        # Create device-backed JAX array `flax_values`.
        tx=torch.tensor(inputs,requires_grad=True);torch_values=source(tx);flax_values=target(jnp.asarray(inputs))
        # Loop over `name` in `errors`:
        for name in errors:
            # Run `error_report` to compute `report`.
            # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
            report=error_report(torch_values[name].detach().numpy(),flax_values[name]);assert report['passed'],(name,report)
            # Execute the next step of the computation.
            errors[name].append(report['max_abs'])
        # Reduce across the target axis to summarize ``.
        torch_values['output'].sum().backward()
        # Create device-backed JAX array `input_gradient`.
        input_gradient=jax.grad(lambda z:jnp.sum(target(z)['output']))(jnp.asarray(inputs))
        # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
        assert error_report(tx.grad.numpy(),input_gradient,atol=5e-6,rtol=5e-5)['passed']
    # Run `convert` to compute `wrong`.
    wrong=convert(state,eps=.1)
    # Evaluate `reference` from the current inputs and state.
    reference={k:v.detach().numpy() for k,v in source(torch.tensor(inputs)).items()}
    # Create device-backed JAX array `wrong_reports`.
    wrong_reports={k:error_report(reference[k],v) for k,v in wrong(jnp.asarray(inputs)).items()}
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
    assert wrong_reports['hidden']['passed'] and not wrong_reports['norm']['passed']
    # Loop over `bad` in `[dict(state, unexpected=torch.zeros(1)), {k: v for k, v in state.items() if k != 'norm.bias'}]`:
    for bad in [dict(state,unexpected=torch.zeros(1)),{k:v for k,v in state.items() if k!='norm.bias'}]:
        try:convert(bad)
        except ValueError:pass
        else:raise AssertionError('incomplete mapping accepted')
# Print diagnostic summary of the computed outputs.
print('Maximum absolute error per layer:',{k:max(v) for k,v in errors.items()})
# Print diagnostic summary of the computed outputs.
print('Wrong epsilon: first mismatch is norm; intermediate outputs and input gradients verified on CPU.')

# Figure data experiment
# Compute figure data for: Locate the first divergent layer
# Evaluate `visual_data` from the current inputs and state.
visual_data={'kind':'bar','labels':list(errors),'xlabel':'operation in execution order','ylabel':'maximum absolute output error','series':[{'label':'matched architecture (max across inputs)','y':[max(v) for v in errors.values()]},{'label':'wrong epsilon (last input batch)','y':[wrong_reports[k]['max_abs'] for k in errors]}]}

# Experiment: Inspect the wrong-epsilon signature
# Experiment — Inspect the wrong-epsilon signature: Localize the first divergent operation before remapping...
# Verify contract: `wrong_reports['hidden']['passed']`.
assert wrong_reports['hidden']['passed']
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not wrong_reports['norm']['passed']
# Print the observed values to compare against the expected result.
print('Per-layer wrong-epsilon report:',wrong_reports)

# Experiment: Test the tolerance rule near zero
# Experiment — Test the tolerance rule near zero: The gate checks each element against a reference-dependent budget.
near_zero = error_report([0.], [1e-6])
# Run `error_report` to compute `near_zero_fail`.
near_zero_fail = error_report([0.], [3e-6])
# Run `error_report` to compute `large_value`.
large_value = error_report([100.], [100.001])
# Verify contract: `near_zero['passed'] and (not near_zero_fail['passed']) and large_val...`.
assert near_zero['passed'] and not near_zero_fail['passed'] and large_value['passed']
# Print the observed values to compare against the expected result.
print('Near-zero, excessive near-zero, large-value:', near_zero['passed'], near_zero_fail['passed'], large_value['passed'])

# Reference solution. Try the exercise before reading this.
# Exercise solution: Compare a near-zero reference against two candidate errors.
# Verify contract: `error_report([0.0, 1.0], [1e-07, 1.000001])['passed']`.
assert error_report([0.,1.],[1e-7,1.000001])['passed']
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not error_report([0.,1.],[1e-3,1.])['passed']
# Print the observed values to compare against the expected result.
print('Near-zero acceptance and rejection verified.')

# Reference practice: Reject a plausible but wrong tensor layout
# Reject a plausible but wrong tensor layout (Transfer): Asymmetric dimensions expose orientation errors that square...
bad_state=dict(state);bad_state['hidden.weight']=state['hidden.weight'].T
# Run the boundary check and catch the expected exception:
try:convert(bad_state)
except ValueError:print('Wrong source layout rejected')
else:raise AssertionError('layout mismatch accepted')

# Reference practice: Check a nonuniform output sensitivity
# Check a nonuniform output sensitivity (Transfer / diagnosis): This checks J^{\mathsf T}v for a second, nonuniform direction.
# Initialize array `probe_inputs` with explicit values and shape.
probe_inputs = np.array([[.2, -.7, 1.1], [1., .4, -.5]], np.float32)
# Initialize array `cotangent` with explicit values and shape.
cotangent = np.array([[1., -.5], [2., .25]], np.float32)
# Run `torch.tensor` to compute `probe_torch`.
probe_torch = torch.tensor(probe_inputs, requires_grad=True)
# Aggregate array values to compute `weighted_source`.
weighted_source = (source(probe_torch)['output'] * torch.tensor(cotangent)).sum()
# Run `weighted_source.backward` to perform the next check or state transition.
weighted_source.backward()
# Differentiate the objective to obtain `weighted_target` via automatic differentiation.
weighted_target = jax.grad(lambda z: jnp.sum(target(z)['output'] * jnp.asarray(cotangent)))(jnp.asarray(probe_inputs))
# Run `error_report` to compute `vjp_report`.
vjp_report = error_report(probe_torch.grad.numpy(), weighted_target, atol=5e-6, rtol=5e-5)
# Verify contract: `vjp_report['passed']`.
assert vjp_report['passed'], vjp_report
# Print the observed values to compare against the expected result.
print('Nonuniform cotangent input-gradient parity:', vjp_report)
print("PASS: deployment-08")
