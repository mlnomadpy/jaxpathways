"""Implement each contract; do not edit the public tests to make them pass."""
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax


def make_model(seed=0, width=8):
    """Return an NNX module: Linear(2,width), tanh, Linear(width,1), (B,) logits."""
    # Key APIs to use: `Classifier`
    # Step 1: Return `Classifier(seed, width)` to the caller.
    raise NotImplementedError('Stage 1: construct and verify the model')


def objective(model, x, labels):
    """Mean stable binary CE; reject logits/label shape mismatch."""
    # Key APIs to use: `model`, `contract`, `shape`, `jnp.mean`, `optax.sigmoid_binary_cross_entropy`
    # Step 1: Run `model` to compute `scores`.
    # Step 2: Guard input contract (`scores.shape != labels.shape`) and fail fast if violated.
    # Step 3: Return `jnp.mean(optax.sigmoid_binary_cross_entropy(scores, labels))` to the caller.
    raise NotImplementedError('Stage 2: scalar loss and independent gradient evidence')


def train(seed, x, labels, rate=.03, steps=200, width=8):
    """Return (trained_model, pre_update_losses); compile updates, not evaluation."""
    # Key APIs to use: `contract`, `make_model`, `state`, `nnx.Optimizer`, `optax.adam`
    # Step 1: Guard input contract (`steps < 1 or rate <= 0`) and fail fast if violated.
    # Step 2: Run `make_model` to compute `model`.
    # Step 3: Initialize the optimizer transformation and state (`optimizer`).
    # Step 4: Evaluate `losses` from the current inputs and state.
    # Step 5: Repeat the update loop over `range(steps)` steps:
    # Step 6: Inside block: Evaluate `_step(model, optimizer, x, labels)` and convert the result into Python scalar/collection `value`.
    raise NotImplementedError('Stage 3: reproducible training with Optax Adam')


def evaluate(model, x, labels):
    """Return loss, accuracy, count without modifying any model state."""
    # Key APIs to use: `model`, `objective`, `jnp.mean`
    # Step 1: Run `model` to compute `scores`.
    # Step 2: Evaluate `objective(model, x, labels)` and convert the result into Python scalar/collection `mean_loss`.
    # Step 3: Reduce across the target axis to summarize `accuracy`.
    # Step 4: Return `{'loss': mean_loss, 'accuracy': accuracy, 'count': len(labels)}` to the caller.
    raise NotImplementedError('Stage 3: isolated held-out metrics')
