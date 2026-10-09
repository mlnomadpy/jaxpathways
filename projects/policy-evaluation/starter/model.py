"""Implement the four cumulative stages described in ../README.md.

Use the course CPU environment. Do not import the instructor reference or
hard-code its reported returns: checks use changed horizons, seeds and policies.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

class State(NamedTuple):
    position: jax.Array
    elapsed: jax.Array
    done: jax.Array

def reset(key):
    """Stage 1: uniform start at 0 or 1, zero elapsed, unfinished."""
    # Key APIs to use: `State`, `random.randint`, `jnp.int32`, `jnp.bool_`
    # Step 1: Return `State(jax.random.randint(key, (), 0, 2), jnp.int32(0), jnp.bool_(False))` to the caller.
    raise NotImplementedError('Implement explicit reset state')

def step(state, action, horizon=8):
    """Stage 1: return (State, reward, terminated event, truncated event).

    Binary actions move left/right
    boundaries are 0 and goal 3. Active moves
    pay -.01 except entering the goal pays 1. Done states absorb with no reward.
    """
    # Key APIs to use: `jnp.clip`, `jnp.where`, `active.astype`, `State`
    # Step 1: Evaluate `active` from the current inputs and state.
    # Step 2: Combine or mask array elements to form `proposed`.
    # Step 3: Combine or mask array elements to form `position`.
    # Step 4: Evaluate `elapsed` from the current inputs and state.
    # Step 5: Evaluate `terminated` from the current inputs and state.
    # Step 6: Evaluate `truncated` from the current inputs and state.
    raise NotImplementedError('Implement and enumerate transition rules')

def log_probability(theta, observations, actions):
    """Stage 2: theta contains one right-action logit for positions 0, 1, 2."""
    # Key APIs to use: `jnp.minimum`, `jnp.where`, `nn.log_sigmoid`
    # Step 1: Reduce across the target axis to summarize `logits`.
    # Step 2: Return `jnp.where(actions == 1, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits))` to the caller.
    raise NotImplementedError('Use stable log-sigmoid probabilities')

def rollout(theta, key, batch_size=256, horizon=8):
    """Stage 2: final State and time-major record dictionary.

    Fields: observation, action, reward, active, terminated, truncated, old_logp.
    Use fresh keys across reset, time, and environments. Do not auto-reset.
    """
    # Key APIs to use: `key`, `random.split`, `jax.vmap`, `body`, `subkeys`
    # Step 1: Split the PRNG key deterministically into independent subkeys (`(reset_key, action_key)`).
    # Step 2: Split the PRNG key deterministically into independent subkeys (`states`).
    # Step 3: Split the PRNG key deterministically into independent subkeys (`time_keys`).
    # Step 4: Function `body(states, time_key)` implementing this stage's computation:
    # Step 5: Run a compiled sequential scan over the time/step axis (`(final, records)`).
    # Step 6: Return `(final, records)` to the caller.
    raise NotImplementedError('Build vmap within scan')

def returns_to_go(rewards):
    """Stage 2: sum future rewards along time axis, preserving both axes."""
    # Key APIs to use: `jnp.cumsum`
    # Step 1: Return `jnp.cumsum(rewards[::-1], axis=0)[::-1]` to the caller.
    raise NotImplementedError('Implement reverse cumulative returns')

def policy_loss(theta, records, method='ppo', clip=.2):
    """Stage 3: negative live-action sum divided by episode count.

    REINFORCE uses log probability times frozen reward-to-go. PPO uses the
    minimum of ratio*target and clipped-ratio*target, with frozen old logp.
    """
    # Key APIs to use: `lax.stop_gradient`, `returns_to_go`, `log_probability`, `jnp.exp`, `jnp.minimum`
    # Step 1: Run `jax.lax.stop_gradient` to compute `advantage`.
    # Step 2: Run `jax.lax.stop_gradient` to compute `old_logp`.
    # Step 3: Run `log_probability` to compute `new_logp`.
    # Step 4: Branch on condition `method == 'reinforce'`:
    # Step 5: Return `-jnp.sum(jnp.where(records['active'], terms, 0.0)) / records['reward'].shape[1]` to the caller.
    raise NotImplementedError('Implement both objective branches')

def exact_return(theta, horizon=8):
    """Stage 3: exact finite-horizon expectation, averaged over starts 0 and 1."""
    # Key APIs to use: `nn.sigmoid`, `jnp.arange`, `jnp.maximum`, `backup`, `jnp.where`
    # Step 1: Run `jax.nn.sigmoid` to compute `p`.
    # Step 2: Create evenly spaced index values in `states`.
    # Step 3: Reduce across the target axis to summarize `left`.
    # Step 4: Evaluate `right` from the current inputs and state.
    # Step 5: Function `backup(_, values)` implementing this stage's computation:
    # Step 6: Allocate initialized array `values` with the specified shape and dtype.
    raise NotImplementedError('Use dynamic programming, not random samples')

def train(seed=0, updates=40, batch_size=256, horizon=8, method='ppo', epochs=3, learning_rate=.5):
    """Stage 4: (final theta, exact-return history including initial value).

    Start at zero logits. Collect fresh data for each update. Freeze each batch
    across its epochs. REINFORCE requires one epoch per fresh batch.
    """
    # Key APIs to use: `contract`, `or`, `in`, `_train`
    # Step 1: Guard input contract (`updates < 1 or batch_size < 1 or horizon < 1 or (epochs < 1) or (learning_rate <= 0)`) and fail fast if violated.
    # Step 2: Guard input contract (`method not in ('ppo', 'reinforce')`) and fail fast if violated.
    # Step 3: Guard input contract (`method == 'reinforce' and epochs != 1`) and fail fast if violated.
    # Step 4: Return `_train(seed, updates, batch_size, horizon, method, epochs, learning_rate)` to the caller.
    raise NotImplementedError('Implement complete rollout and update cycle')

def evaluate(theta, seeds, batch_size=512, horizon=8):
    """Stage 4: reject malformed theta, duplicate/fewer-than-two seeds or bad sizes.

    Return seeds, episode_returns (streams,batch), stream_means, mean and exact.
    Keep policy immutable and randomness separate from training.
    """
    # Key APIs to use: `jnp.asarray`, `contract`, `jnp.all`, `jnp.isfinite`, `or`
    # Step 1: Create device-backed JAX array `theta`.
    # Step 2: Guard input contract (`theta.shape != (3,) or not bool(jnp.all(jnp.isfinite(theta)))`) and fail fast if violated.
    # Step 3: Evaluate `seeds` and convert the result into Python scalar/collection `seeds`.
    # Step 4: Guard input contract (`len(seeds) < 2 or len(set(seeds)) != len(seeds) or batch_size < 1 or (horizon < 1)`) and fail fast if violated.
    # Step 5: Convert `returns` to a host NumPy array for inspection or verification.
    # Step 6: Return `{'seeds': list(seeds), 'episode_returns': returns, 'stream_means': returns.mean(1), 'mean': float(returns.mean()), 'exact': float(exact_return(theta, horizon))}` to the caller.
    raise NotImplementedError('Retain disaggregated held-out evaluation')
