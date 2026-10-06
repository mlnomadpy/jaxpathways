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
    raise NotImplementedError('Implement explicit reset state')

def step(state, action, horizon=8):
    """Stage 1: return (State, reward, terminated event, truncated event).

    Binary actions move left/right; boundaries are 0 and goal 3. Active moves
    pay -.01 except entering the goal pays 1. Done states absorb with no reward.
    """
    raise NotImplementedError('Implement and enumerate transition rules')

def log_probability(theta, observations, actions):
    """Stage 2: theta contains one right-action logit for positions 0, 1, 2."""
    raise NotImplementedError('Use stable log-sigmoid probabilities')

def rollout(theta, key, batch_size=256, horizon=8):
    """Stage 2: final State and time-major record dictionary.

    Fields: observation, action, reward, active, terminated, truncated, old_logp.
    Use fresh keys across reset, time, and environments. Do not auto-reset.
    """
    raise NotImplementedError('Build vmap within scan')

def returns_to_go(rewards):
    """Stage 2: sum future rewards along time axis, preserving both axes."""
    raise NotImplementedError('Implement reverse cumulative returns')

def policy_loss(theta, records, method='ppo', clip=.2):
    """Stage 3: negative live-action sum divided by episode count.

    REINFORCE uses log probability times frozen reward-to-go. PPO uses the
    minimum of ratio*target and clipped-ratio*target, with frozen old logp.
    """
    raise NotImplementedError('Implement both objective branches')

def exact_return(theta, horizon=8):
    """Stage 3: exact finite-horizon expectation, averaged over starts 0 and 1."""
    raise NotImplementedError('Use dynamic programming, not random samples')

def train(seed=0, updates=40, batch_size=256, horizon=8, method='ppo', epochs=3, learning_rate=.5):
    """Stage 4: (final theta, exact-return history including initial value).

    Start at zero logits. Collect fresh data for each update. Freeze each batch
    across its epochs. REINFORCE requires one epoch per fresh batch.
    """
    raise NotImplementedError('Implement complete rollout and update cycle')

def evaluate(theta, seeds, batch_size=512, horizon=8):
    """Stage 4: reject malformed theta, duplicate/fewer-than-two seeds or bad sizes.

    Return seeds, episode_returns (streams,batch), stream_means, mean and exact.
    Keep policy immutable and randomness separate from training.
    """
    raise NotImplementedError('Retain disaggregated held-out evaluation')
