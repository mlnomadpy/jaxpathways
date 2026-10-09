"""Write a functional environment: worked experiments and reference solutions. CPU checks."""

# Define state and the transition
# Step 1: Define state and the transition
"""Finite-horizon tabular policy training. CPU teaching environment, not a benchmark."""
# Import functools (partial) for this computation.
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


# Define `State` module / container with explicit state and forward pass:
class State(NamedTuple):
    # Evaluate `position` from the current inputs and state.
    position: jax.Array
    # Evaluate `elapsed` from the current inputs and state.
    elapsed: jax.Array
    # Evaluate `done` from the current inputs and state.
    done: jax.Array


# Function `reset(key)` implementing this stage's computation:
def reset(key):
    """Start uniformly at position 0 or 1; goal is position 3."""
    # Return `State(jax.random.randint(key, (), 0, 2), jnp.int32(0), jnp.bool_(False))` to the caller.
    return State(jax.random.randint(key, (), 0, 2), jnp.int32(0), jnp.bool_(False))


# Function `step(state, action, horizon)` implementing this stage's computation:
def step(state, action, horizon=8):
    """Action 0: left, 1: right. Caller supplies binary action and positive horizon.

    Returns next state, reward, termination event, truncation event. Finished
    states absorb without more rewards/events. No implicit reset or lost final observation.
    """
    # Evaluate `active` from the current inputs and state.
    active = ~state.done
    # Combine or mask array elements to form `proposed`.
    proposed = jnp.clip(state.position + 2 * action - 1, 0, 3)
    # Combine or mask array elements to form `position`.
    position = jnp.where(active, proposed, state.position)
    # Evaluate `elapsed` from the current inputs and state.
    elapsed = state.elapsed + active.astype(jnp.int32)
    # Evaluate `terminated` from the current inputs and state.
    terminated = active & (position == 3)
    # Evaluate `truncated` from the current inputs and state.
    truncated = active & ~terminated & (elapsed >= horizon)
    # Combine or mask array elements to form `reward`.
    reward = jnp.where(active, jnp.where(terminated, 1., -.01), 0.)
    # Return `(State(position, elapsed, state.done | terminated | truncated), reward, terminated, truncated)` to the caller.
    return State(position, elapsed, state.done | terminated | truncated), reward, terminated, truncated

# Check a complete episode
# Step 2 — Check a complete episode: The known two-action path produces one terminal reward and then...
s = State(jnp.int32(1), jnp.int32(0), jnp.bool_(False))
# Evaluate `(positions, rewards)` from the current inputs and state.
positions, rewards = [int(s.position)], []
# Iterate over `action` to step through the computation:
for action in [1, 1, 0, 0]:
    # Run `step` to compute `(s, reward, terminated, truncated)`.
    s, reward, terminated, truncated = step(s, jnp.int32(action))
    # Append the current step result to `positions`.
    # Append the current step result to `positions`.
    positions.append(int(s.position)); rewards.append(float(reward))
# Verify contract: `positions == [1, 2, 3, 3, 3]`.
assert positions == [1, 2, 3, 3, 3]
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.array(rewards), jnp.array([-.01, 1., 0., 0.]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert int(s.elapsed) == 2
# Print the observed values to compare against the expected result.
print('positions:', positions, 'rewards:', rewards)

# Complete runnable example (rl-01)
"""Finite-horizon tabular policy training. CPU teaching environment, not a benchmark."""
# Import functools (partial) for this computation.
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


# Define `State` module / container with explicit state and forward pass:
class State(NamedTuple):
    # Evaluate `position` from the current inputs and state.
    position: jax.Array
    # Evaluate `elapsed` from the current inputs and state.
    elapsed: jax.Array
    # Evaluate `done` from the current inputs and state.
    done: jax.Array


# Function `reset(key)` implementing this stage's computation:
def reset(key):
    """Start uniformly at position 0 or 1; goal is position 3."""
    # Return `State(jax.random.randint(key, (), 0, 2), jnp.int32(0), jnp.bool_(False))` to the caller.
    return State(jax.random.randint(key, (), 0, 2), jnp.int32(0), jnp.bool_(False))


# Function `step(state, action, horizon)` implementing this stage's computation:
def step(state, action, horizon=8):
    """Action 0: left, 1: right. Caller supplies binary action and positive horizon.

    Returns next state, reward, termination event, truncation event. Finished
    states absorb without more rewards/events. No implicit reset or lost final observation.
    """
    # Evaluate `active` from the current inputs and state.
    active = ~state.done
    # Combine or mask array elements to form `proposed`.
    proposed = jnp.clip(state.position + 2 * action - 1, 0, 3)
    # Combine or mask array elements to form `position`.
    position = jnp.where(active, proposed, state.position)
    # Evaluate `elapsed` from the current inputs and state.
    elapsed = state.elapsed + active.astype(jnp.int32)
    # Evaluate `terminated` from the current inputs and state.
    terminated = active & (position == 3)
    # Evaluate `truncated` from the current inputs and state.
    truncated = active & ~terminated & (elapsed >= horizon)
    # Combine or mask array elements to form `reward`.
    reward = jnp.where(active, jnp.where(terminated, 1., -.01), 0.)
    # Return `(State(position, elapsed, state.done | terminated | truncated), reward, terminated, truncated)` to the caller.
    return State(position, elapsed, state.done | terminated | truncated), reward, terminated, truncated

# Step 2 — Check a complete episode: The known two-action path produces one terminal reward and then...
s = State(jnp.int32(1), jnp.int32(0), jnp.bool_(False))
# Evaluate `(positions, rewards)` from the current inputs and state.
positions, rewards = [int(s.position)], []
# Iterate over `action` to step through the computation:
for action in [1, 1, 0, 0]:
    # Run `step` to compute `(s, reward, terminated, truncated)`.
    s, reward, terminated, truncated = step(s, jnp.int32(action))
    # Append the current step result to `positions`.
    # Append the current step result to `positions`.
    positions.append(int(s.position)); rewards.append(float(reward))
# Verify contract: `positions == [1, 2, 3, 3, 3]`.
assert positions == [1, 2, 3, 3, 3]
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.array(rewards), jnp.array([-.01, 1., 0., 0.]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert int(s.elapsed) == 2
# Print the observed values to compare against the expected result.
print('positions:', positions, 'rewards:', rewards)

# Figure data experiment
# Compute figure data for: One goal reward, then absorbing padding
# Combine or mask array elements to form `visual_data`.
visual_data = {'kind':'line','x':[1,2,3,4],'xlabel':'transition index','ylabel':'reward per transition','series':[{'label':'right, right, padded, padded','y':rewards}]}

# Experiment: A goal at the deadline
# Experiment — A goal at the deadline: The goal transition defines a completed task, so the two flags...
s = State(jnp.int32(2), jnp.int32(1), jnp.bool_(False))
# Run `step` to compute `(s, r, term, trunc)`.
s, r, term, trunc = step(s, jnp.int32(1), horizon=2)
# Verify contract: `bool(term) and (not bool(trunc)) and (float(r) == 1.0)`.
assert bool(term) and not bool(trunc) and float(r) == 1.
# Print the observed values to compare against the expected result.
print('deadline goal:', bool(term), bool(trunc))

# Experiment: The same key is the same start
# Experiment — The same key is the same start: This verifies the randomness contract, not a guarantee that...
# Create or split explicit PRNG key(s) (`key`) for reproducible randomness.
key = jax.random.key(7)
# Evaluate `(a, b)` from the current inputs and state.
a, b = reset(key), reset(key)
# Verify contract: `int(a.position) == int(b.position)`.
assert int(a.position) == int(b.position)
# Create or split explicit PRNG key(s) (`starts`) for reproducible randomness.
starts = jax.vmap(reset)(jax.random.split(key, 1000)).position
# Verify contract: `0.4 < float(starts.mean()) < 0.6`.
assert 0.4 < float(starts.mean()) < 0.6
# Print the observed values to compare against the expected result.
print('fraction starting at one:', float(starts.mean()))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Begin at position 0, choose left for three actions with horizon 3, and...
s = State(jnp.int32(0), jnp.int32(0), jnp.bool_(False))
# Evaluate `total` from the current inputs and state.
total = 0.
# Iterate over `i` to step through the computation:
for i in range(3):
    # Run `step` to compute `(s, r, term, trunc)`.
    s, r, term, trunc = step(s, jnp.int32(0), 3)
    # Accumulate the next contribution into `total`.
    total += float(r)
    # Verify contract: `not bool(term) and bool(trunc) == (i == 2)`.
    assert not bool(term) and bool(trunc) == (i == 2)
# Verify contract: `abs(total + 0.03) < 1e-06`.
assert abs(total + .03) < 1e-6
# Run `step` to compute `(s2, r, term, trunc)`.
s2, r, term, trunc = step(s, jnp.int32(1), 3)
# Verify contract: `int(s2.position) == 0 and int(s2.elapsed) == 3 and (float(r) == 0.0)`.
assert int(s2.position) == 0 and int(s2.elapsed) == 3 and float(r) == 0.

# Reference practice: Separate the two target meanings
# Separate the two target meanings (Transfer / diagnosis): The arithmetic exposes a modeling choice.
r, gamma, value = -.01, .9, 2.
# Evaluate `continuing_timeout` from the current inputs and state.
continuing_timeout = r + gamma * value
# Evaluate `terminal` from the current inputs and state.
terminal = r
# Evaluate `finite_horizon_timeout` from the current inputs and state.
finite_horizon_timeout = r
# Verify contract: `abs(continuing_timeout - 1.79) < 1e-07`.
assert abs(continuing_timeout - 1.79) < 1e-7
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert terminal == finite_horizon_timeout == -.01

# Reference practice: Enumerate all one-step moves
# Enumerate all one-step moves (Transfer / diagnosis): Enumeration checks boundary behavior separately from the...
# Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
compiled = jax.jit(step)
# Iterate over `position` to step through the computation:
for position in range(3):
    # Loop over `action` in `[0, 1]`:
    for action in [0, 1]:
        # Run `State` to compute `s`.
        s = State(jnp.int32(position), jnp.int32(0), jnp.bool_(False))
        # Run `compiled` to compute `(nxt, reward, term, trunc)`.
        nxt, reward, term, trunc = compiled(s, jnp.int32(action))
        # Run `min` to compute `expected`.
        expected = min(3, max(0, position + (1 if action else -1)))
        # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
        assert int(nxt.position) == expected
        # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
        assert bool(term) == (expected == 3)
        # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
        assert abs(float(reward) - (1. if expected == 3 else -.01)) < 1e-6
print("PASS: rl-01")
