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
    # Key APIs to use: `contract`, `jnp.ndim`, `advance`, `lax.scan`, `jnp.concatenate`
    # Step 1: Guard input contract (`initials.ndim != 1 or initials.size == 0`) and fail fast if violated.
    # Step 2: Guard input contract (`not isinstance(steps, int) or steps < 1 or dt <= 0`) and fail fast if violated.
    # Step 3: Guard input contract (`jnp.ndim(rate) != 0`) and fail fast if violated.
    # Step 4: Function `advance(values, unused)` implementing this stage's computation:
    # Step 5: Run a compiled sequential scan over the time/step axis (`(_, tail)`).
    # Step 6: Return `jnp.concatenate((initials[None, :], tail), axis=0).T` to the caller.
    raise NotImplementedError('Stage 1: explicit RK4 transition and scan')


def loss(log_rate, initials, observations, dt=0.05):
    """Mean squared observation error using rate=exp(log_rate).

    Reject observations not shaped (B, steps+1) with at least two times.
    """
    # Key APIs to use: `contract`, `shape`, `simulate`, `jnp.exp`, `jnp.mean`
    # Step 1: Guard input contract (`observations.ndim != 2 or observations.shape[0] != len(initials) or observations.shape[1] < 2`) and fail fast if violated.
    # Step 2: Run `simulate` to compute `predictions`.
    # Step 3: Return `jnp.mean((predictions - observations) ** 2)` to the caller.
    raise NotImplementedError('Stage 2: scalar differentiable observation loss')


def fit(initials, observations, dt=0.05, initial_rate=1.4, updates=250, learning_rate=0.4):
    """Return positive fitted rate and loss history including pre-update loss.

    Use exactly updates gradient descent updates in log-rate, return a history
    of length updates+1, and reject nonpositive fitting configuration values.
    """
    # Key APIs to use: `contract`, `jnp.log`, `jnp.asarray`, `loss`, `objective`
    # Step 1: Guard input contract (`initial_rate <= 0 or updates < 1 or learning_rate <= 0`) and fail fast if violated.
    # Step 2: Create device-backed JAX array `theta`.
    # Step 3: Evaluate `objective` from the current inputs and state.
    # Step 4: Evaluate `history` from the current inputs and state.
    # Step 5: Differentiate the objective to obtain gradients `update`.
    # Step 6: Repeat the update loop over `range(updates)` steps:
    raise NotImplementedError('Stage 3: reproducible positive-parameter fit')


def evaluate(rate, initials, observations, dt=0.05):
    """Return mse, rmse, max_abs_error, trajectory_count, observation_count.

    Use observations only for measurement
    do not fit or mutate inputs.
    Reject nonfinite/nonpositive rate.
    """
    # Key APIs to use: `contract`, `np.isfinite`, `loss`, `jnp.log`, `simulate`
    # Step 1: Guard input contract (`not np.isfinite(rate) or rate <= 0`) and fail fast if violated.
    # Step 2: Run `loss` to compute `value`.
    # Step 3: Run `simulate` to compute `predictions`.
    # Step 4: Evaluate `error` from the current inputs and state.
    # Step 5: Return `{'mse': float(value), 'rmse': float(jnp.sqrt(value)), 'max_abs_error': float(jnp.max(jnp.abs(error))), 'trajectory_count': int(len(initials)), 'observation_count': int(observations.size)}` to the caller.
    raise NotImplementedError('Stage 3: isolated evaluation')
