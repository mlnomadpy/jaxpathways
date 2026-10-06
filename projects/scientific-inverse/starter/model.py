"""Implement each contract after completing science-01 through science-04."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np


def simulate(rate, initials, steps=40, dt=0.05):
    """RK4 cooling ensemble, shape (B, steps+1), including initial state.

    Require a nonempty vector of initials, scalar rate, positive integer steps
    and positive dt. Preserve JAX differentiation with respect to rate.
    """
    raise NotImplementedError('Stage 1: explicit RK4 transition and scan')


def loss(log_rate, initials, observations, dt=0.05):
    """Mean squared observation error using rate=exp(log_rate).

    Reject observations not shaped (B, steps+1) with at least two times.
    """
    raise NotImplementedError('Stage 2: scalar differentiable observation loss')


def fit(initials, observations, dt=0.05, initial_rate=1.4, updates=250, learning_rate=0.4):
    """Return positive fitted rate and loss history including pre-update loss.

    Use exactly updates gradient descent updates in log-rate, return a history
    of length updates+1, and reject nonpositive fitting configuration values.
    """
    raise NotImplementedError('Stage 3: reproducible positive-parameter fit')


def evaluate(rate, initials, observations, dt=0.05):
    """Return mse, rmse, max_abs_error, trajectory_count, observation_count.

    Use observations only for measurement; do not fit or mutate inputs.
    Reject nonfinite/nonpositive rate.
    """
    raise NotImplementedError('Stage 3: isolated evaluation')
