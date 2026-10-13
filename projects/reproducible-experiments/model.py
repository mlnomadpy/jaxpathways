"""An intentionally small, inspectable Flax NNX model and synchronized update."""
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax


class Regressor(nnx.Module):
    def __init__(self, width, seed):
        rngs = nnx.Rngs(seed)
        self.hidden = nnx.Linear(2, width, rngs=rngs)
        self.out = nnx.Linear(width, 1, rngs=rngs)

    def __call__(self, x):
        return self.out(nnx.tanh(self.hidden(x))).squeeze(-1)


def objective(model, x, y):
    return jnp.mean((model(x) - y) ** 2)


@nnx.jit
def train_step(model, optimizer, x, y):
    grads = nnx.grad(objective)(model, x, y)
    optimizer.update(model, grads)
    norm = jnp.sqrt(sum(jnp.sum(a*a) for a in jax.tree.leaves(grads)))
    return jnp.stack([objective(model, x, y), norm])


def optimizer(model, rate):
    # No momentum: SGD has no persistent moment arrays in this teaching recipe.
    return nnx.Optimizer(model, optax.sgd(rate), wrt=nnx.Param)


def arrays(model):
    return {'hidden_kernel': np.asarray(model.hidden.kernel[...]), 'hidden_bias': np.asarray(model.hidden.bias[...]),
            'out_kernel': np.asarray(model.out.kernel[...]), 'out_bias': np.asarray(model.out.bias[...])}


def restore(path, width):
    model = Regressor(width, 0)
    with np.load(path, allow_pickle=False) as saved:
        for layer, attr, key in [(model.hidden, 'kernel', 'hidden_kernel'), (model.hidden, 'bias', 'hidden_bias'),
                                 (model.out, 'kernel', 'out_kernel'), (model.out, 'bias', 'out_bias')]:
            variable = getattr(layer, attr)
            value = saved[key]
            if value.shape != variable[...].shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('invalid model artifact')
            variable[...] = jnp.asarray(value)
    return model
