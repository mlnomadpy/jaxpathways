"""Reference: inspectable NNX classifier audit on a synthetic XOR fixture."""
# Import flax (nnx) for this computation.
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax


# Define `Classifier` module / container with explicit state and forward pass:
class Classifier(nnx.Module):
    # Function `__init__(self, seed, width)` implementing this stage's computation:
    def __init__(self, seed=0, width=8):
        # Run `nnx.Rngs` to compute `rngs`.
        rngs = nnx.Rngs(seed)
        # Run `nnx.Linear` to compute `self.hidden`.
        self.hidden = nnx.Linear(2, width, rngs=rngs)
        # Run `nnx.Linear` to compute `self.out`.
        self.out = nnx.Linear(width, 1, rngs=rngs)

    # Function `__call__(self, x)` implementing this stage's computation:
    def __call__(self, x):
        # Return `self.out(nnx.tanh(self.hidden(x))).squeeze(-1)` to the caller.
        return self.out(nnx.tanh(self.hidden(x))).squeeze(-1)


# Function `make_model(seed, width)` implementing this stage's computation:
def make_model(seed=0, width=8):
    # Return `Classifier(seed, width)` to the caller.
    return Classifier(seed, width)


# Function `objective(model, x, labels)` implementing this stage's computation:
def objective(model, x, labels):
    # Run `model` to compute `scores`.
    scores = model(x)
    # Guard input contract (`scores.shape != labels.shape`) and fail fast if violated.
    if scores.shape != labels.shape:
        raise ValueError('scores and labels must both have shape (batch,)')
    # Return `jnp.mean(optax.sigmoid_binary_cross_entropy(scores, labels))` to the caller.
    return jnp.mean(optax.sigmoid_binary_cross_entropy(scores, labels))


# Define and JIT-compile `_step(model, optimizer, x, labels)` so XLA traces and fuses the operations:
@nnx.jit
# Function `_step(model, optimizer, x, labels)` implementing this stage's computation:
def _step(model, optimizer, x, labels):
    # Differentiate the objective to obtain `(value, grads)` via automatic differentiation.
    value, grads = nnx.value_and_grad(objective)(model, x, labels)
    # Apply the computed gradient updates to update the model parameters.
    optimizer.update(model, grads)
    # Return `value` to the caller.
    return value


# Function `train(seed, x, labels, rate, ...)` implementing this stage's computation:
def train(seed, x, labels, rate=.03, steps=200, width=8):
    # Guard input contract (`steps < 1 or rate <= 0`) and fail fast if violated.
    if steps < 1 or rate <= 0:
        raise ValueError('steps and rate must be positive')
    # Run `make_model` to compute `model`.
    model = make_model(seed, width)
    # Configure or step the Optax optimizer state (`optimizer`).
    optimizer = nnx.Optimizer(model, optax.adam(rate), wrt=nnx.Param)
    # Initialize list `losses` for the stage values.
    losses = []
    # Repeat the update loop over `range(steps)` steps:
    for _ in range(steps):
        # Evaluate `_step(model, optimizer, x, labels)` and convert the result into Python scalar/collection `value`.
        value = float(_step(model, optimizer, x, labels))
        # Guard input contract (`not np.isfinite(value)`) and fail fast if violated.
        if not np.isfinite(value):
            raise FloatingPointError('nonfinite pre-update training loss')
        # Append the current step result to `losses`.
        losses.append(value)
    # Return `(model, np.asarray(losses))` to the caller.
    return model, np.asarray(losses)


# Function `evaluate(model, x, labels)` implementing this stage's computation:
def evaluate(model, x, labels):
    # Run `model` to compute `scores`.
    scores = model(x)
    # Evaluate `objective(model, x, labels)` and convert the result into Python scalar/collection `mean_loss`.
    mean_loss = float(objective(model, x, labels))
    # Aggregate array values to compute `accuracy`.
    accuracy = float(jnp.mean((scores > 0) == labels))
    # Return `{'loss': mean_loss, 'accuracy': accuracy, 'count': len(labels)}` to the caller.
    return {'loss': mean_loss, 'accuracy': accuracy, 'count': len(labels)}
