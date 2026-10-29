"""Reference for a count-weighted, replicated JAX multi-controller update."""
from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def local_numerator_and_gradient(weights, x, y, valid):
    def summed_loss(w):
        residual = x @ w - y
        return jnp.sum(jnp.where(valid, residual * residual, 0.0))
    return jax.value_and_grad(summed_loss)(weights)


@jax.jit
def global_transition(weights, momentum, gathered_gradient_sums, gathered_counts,
                      learning_rate=0.05, momentum_rate=0.8):
    count = jnp.sum(gathered_counts)
    gradient = jnp.sum(gathered_gradient_sums, axis=0) / jnp.maximum(count, 1)
    next_momentum = momentum_rate * momentum + gradient
    next_weights = weights - learning_rate * next_momentum
    return next_weights, next_momentum
