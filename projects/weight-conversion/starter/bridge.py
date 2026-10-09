"""Explicit CPU PyTorch -> Flax NNX mapping; no generic architecture converter."""
import numpy as np
import jax
import jax.numpy as jnp
import torch
from flax import nnx

class TorchModel(torch.nn.Module):

    def __init__(self, eps=1e-05):
        super().__init__()
        self.hidden = torch.nn.Linear(3, 5)
        self.norm = torch.nn.LayerNorm(5, eps=eps)
        self.out = torch.nn.Linear(5, 2)

    def forward(self, x):
        h = self.hidden(x)
        n = self.norm(h)
        a = torch.nn.functional.gelu(n, approximate='none')
        return {'hidden': h, 'norm': n, 'activation': a, 'output': self.out(a)}

class FlaxModel(nnx.Module):

    def __init__(self, eps=1e-05):
        self.hidden = nnx.Linear(3, 5, rngs=nnx.Rngs(0))
        self.norm = nnx.LayerNorm(5, epsilon=eps, use_fast_variance=False, rngs=nnx.Rngs(1))
        self.out = nnx.Linear(5, 2, rngs=nnx.Rngs(2))

    def __call__(self, x):
        h = self.hidden(x)
        n = self.norm(h)
        a = jax.nn.gelu(n, approximate=False)
        return {'hidden': h, 'norm': n, 'activation': a, 'output': self.out(a)}

def convert(state, eps=1e-05):
    """Implement the contract in the project guide; run the staged checker."""
    # Key APIs to use: `contract`, `shapes.items`, `detach`, `cpu`, `numpy`
    # Step 1: Evaluate `shapes` from the current inputs and state.
    # Step 2: Guard input contract (`set(state) != set(shapes)`) and fail fast if violated.
    # Step 3: Evaluate `arrays` from the current inputs and state.
    # Step 4: Loop over `(name, shape)` in `shapes.items()`:
    # Step 5: Inside block: Evaluate `value` from the current inputs and state.
    # Step 6: Inside block: Guard input contract (`value.shape != shape or value.dtype != np.float32 or (not np.isfinite(value).all())`) and fail fast if violated.
    raise NotImplementedError('convert')

def error_report(reference, actual, atol=2e-06, rtol=2e-05):
    """Implement the contract in the project guide; run the staged checker."""
    # Key APIs to use: `np.asarray`, `contract`, `np.isfinite`, `all`, `or`
    # Step 1: Convert `reference` to a host NumPy array for inspection or verification.
    # Step 2: Convert `actual` to a host NumPy array for inspection or verification.
    # Step 3: Guard input contract (`reference.shape != actual.shape or not np.isfinite(reference).all() or (not np.isfinite(actual).all())`) and fail fast if violated.
    # Step 4: Guard input contract (`min(atol, rtol) < 0 or not np.isfinite([atol, rtol]).all()`) and fail fast if violated.
    # Step 5: Run `np.abs` to compute `absolute`.
    # Step 6: Evaluate `budget` from the current inputs and state.
    raise NotImplementedError('error_report')

def save_flax(model, path):
    """Implement the contract in the project guide; run the staged checker."""
    # Key APIs to use: `np.savez`, `np.asarray`, `np.array`
    # Step 1: Convert `` to a host NumPy array for inspection or verification.
    raise NotImplementedError('save_flax')

def load_flax(path):
    """Implement the contract in the project guide; run the staged checker."""
    # Key APIs to use: `np.load`, `contract`, `np.isfinite`, `shapes.items`, `or`
    # Step 1: Evaluate `shapes` from the current inputs and state.
    # Step 2: Enter managed runtime/context scope for this block:
    # Step 3: Inside block: Guard input contract (`set(archive.files) != set(shapes) | {'epsilon', 'schema'}`) and fail fast if violated.
    # Step 4: Inside block: Guard input contract (`archive['schema'].shape != () or int(archive['schema']) != 1`) and fail fast if violated.
    # Step 5: Run `FlaxModel` to compute `model`.
    # Step 6: Loop over `(module, name, key)` in `[(model.hidden, 'kernel', 'hidden_kernel'), (model.hidden, 'bias', 'hidden_bias'), (model.norm, 'scale', 'norm_scale'), (model.norm, 'bias', 'norm_bias'), (model.out, 'kernel', 'out_kernel'), (model.out, 'bias', 'out_bias')]`:
    raise NotImplementedError('load_flax')
