"""Reference: inspectable NNX classifier audit on a synthetic XOR fixture."""
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax


class Classifier(nnx.Module):
    def __init__(self, seed=0, width=8):
        rngs = nnx.Rngs(seed)
        self.hidden = nnx.Linear(2, width, rngs=rngs)
        self.out = nnx.Linear(width, 1, rngs=rngs)

    def __call__(self, x):
        return self.out(nnx.tanh(self.hidden(x))).squeeze(-1)


def make_model(seed=0, width=8):
    return Classifier(seed, width)


def objective(model, x, labels):
    scores = model(x)
    if scores.shape != labels.shape:
        raise ValueError('scores and labels must both have shape (batch,)')
    return jnp.mean(optax.sigmoid_binary_cross_entropy(scores, labels))


@nnx.jit
def _step(model, optimizer, x, labels):
    value, grads = nnx.value_and_grad(objective)(model, x, labels)
    optimizer.update(model, grads)
    return value


def train(seed, x, labels, rate=.03, steps=200, width=8):
    if steps < 1 or rate <= 0:
        raise ValueError('steps and rate must be positive')
    model = make_model(seed, width)
    optimizer = nnx.Optimizer(model, optax.adam(rate), wrt=nnx.Param)
    losses = []
    for _ in range(steps):
        value = float(_step(model, optimizer, x, labels))
        if not np.isfinite(value):
            raise FloatingPointError('nonfinite pre-update training loss')
        losses.append(value)
    return model, np.asarray(losses)


def evaluate(model, x, labels):
    scores = model(x)
    mean_loss = float(objective(model, x, labels))
    accuracy = float(jnp.mean((scores > 0) == labels))
    return {'loss': mean_loss, 'accuracy': accuracy, 'count': len(labels)}
