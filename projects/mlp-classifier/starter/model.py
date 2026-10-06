"""Implement each contract; do not edit the public tests to make them pass."""
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax


def make_model(seed=0, width=8):
    """Return an NNX module: Linear(2,width), tanh, Linear(width,1), (B,) logits."""
    raise NotImplementedError('Stage 1: construct and verify the model')


def objective(model, x, labels):
    """Mean stable binary CE; reject logits/label shape mismatch."""
    raise NotImplementedError('Stage 2: scalar loss and independent gradient evidence')


def train(seed, x, labels, rate=.03, steps=200, width=8):
    """Return (trained_model, pre_update_losses); compile updates, not evaluation."""
    raise NotImplementedError('Stage 3: reproducible training with Optax Adam')


def evaluate(model, x, labels):
    """Return loss, accuracy, count without modifying any model state."""
    raise NotImplementedError('Stage 3: isolated held-out metrics')
