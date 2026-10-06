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
    raise NotImplementedError('convert')

def error_report(reference, actual, atol=2e-06, rtol=2e-05):
    """Implement the contract in the project guide; run the staged checker."""
    raise NotImplementedError('error_report')

def save_flax(model, path):
    """Implement the contract in the project guide; run the staged checker."""
    raise NotImplementedError('save_flax')

def load_flax(path):
    """Implement the contract in the project guide; run the staged checker."""
    raise NotImplementedError('load_flax')
