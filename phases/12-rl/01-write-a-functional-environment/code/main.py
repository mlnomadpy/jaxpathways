"""Write a functional environment: worked experiments and reference solutions. CPU checks."""

# Define state and the transition
# Step 1: Define state and the transition
"""Finite-horizon tabular policy training. CPU teaching environment, not a benchmark."""
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


# Define `State` module / container with explicit state and forward pass:
class State(NamedTuple):
    position: jax.Array
    elapsed: jax.Array
    done: jax.Array


def reset(key):
    """Start uniformly at position 0 or 1; goal is position 3."""
    return State(jax.random.randint(key, (), 0, 2), jnp.int32(0), jnp.bool_(False))


def step(state, action, horizon=8):
    """Action 0: left, 1: right. Caller supplies binary action and positive horizon.

    Returns next state, reward, termination event, truncation event. Finished
    states absorb without more rewards/events. No implicit reset or lost final observation.
    """
    active = ~state.done
    proposed = jnp.clip(state.position + 2 * action - 1, 0, 3)
    position = jnp.where(active, proposed, state.position)
    elapsed = state.elapsed + active.astype(jnp.int32)
    terminated = active & (position == 3)
    truncated = active & ~terminated & (elapsed >= horizon)
    reward = jnp.where(active, jnp.where(terminated, 1., -.01), 0.)
    return State(position, elapsed, state.done | terminated | truncated), reward, terminated, truncated

# Check a complete episode
# Step 2 — Check a complete episode
s = State(jnp.int32(1), jnp.int32(0), jnp.bool_(False))
positions, rewards = [int(s.position)], []
for action in [1, 1, 0, 0]:
    s, reward, terminated, truncated = step(s, jnp.int32(action))
    positions.append(int(s.position))
    rewards.append(float(reward))
assert positions == [1, 2, 3, 3, 3]
assert jnp.allclose(jnp.array(rewards), jnp.array([-.01, 1., 0., 0.]))
assert int(s.elapsed) == 2
print('positions:', positions, 'rewards:', rewards)

# Step 3: Verify invariants on the completed state
assert jnp.allclose(jnp.array(rewards), jnp.array([-.01, 1., 0., 0.]))
assert int(s.elapsed) == 2

# Complete runnable example (rl-01)
"""Finite-horizon tabular policy training. CPU teaching environment, not a benchmark."""
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


# Define `State` module / container with explicit state and forward pass:
class State(NamedTuple):
    position: jax.Array
    elapsed: jax.Array
    done: jax.Array


def reset(key):
    """Start uniformly at position 0 or 1; goal is position 3."""
    return State(jax.random.randint(key, (), 0, 2), jnp.int32(0), jnp.bool_(False))


def step(state, action, horizon=8):
    """Action 0: left, 1: right. Caller supplies binary action and positive horizon.

    Returns next state, reward, termination event, truncation event. Finished
    states absorb without more rewards/events. No implicit reset or lost final observation.
    """
    active = ~state.done
    proposed = jnp.clip(state.position + 2 * action - 1, 0, 3)
    position = jnp.where(active, proposed, state.position)
    elapsed = state.elapsed + active.astype(jnp.int32)
    terminated = active & (position == 3)
    truncated = active & ~terminated & (elapsed >= horizon)
    reward = jnp.where(active, jnp.where(terminated, 1., -.01), 0.)
    return State(position, elapsed, state.done | terminated | truncated), reward, terminated, truncated

# Step 2 — Check a complete episode
s = State(jnp.int32(1), jnp.int32(0), jnp.bool_(False))
positions, rewards = [int(s.position)], []
for action in [1, 1, 0, 0]:
    s, reward, terminated, truncated = step(s, jnp.int32(action))
    positions.append(int(s.position))
    rewards.append(float(reward))
assert positions == [1, 2, 3, 3, 3]
assert jnp.allclose(jnp.array(rewards), jnp.array([-.01, 1., 0., 0.]))
assert int(s.elapsed) == 2
print('positions:', positions, 'rewards:', rewards)

# Figure data experiment
visual_data = {'kind':'line','x':[1,2,3,4],'xlabel':'transition index','ylabel':'reward per transition','series':[{'label':'right, right, padded, padded','y':rewards}]}

# Experiment: A goal at the deadline
# Experiment — A goal at the deadline
s = State(jnp.int32(2), jnp.int32(1), jnp.bool_(False))
s, r, term, trunc = step(s, jnp.int32(1), horizon=2)
assert bool(term) and not bool(trunc) and float(r) == 1.
print('deadline goal:', bool(term), bool(trunc))

# Experiment: The same key is the same start
# Experiment — The same key is the same start
# Create or split explicit PRNG key(s) (`key`) for reproducible randomness.
key = jax.random.key(7)
a, b = reset(key), reset(key)
assert int(a.position) == int(b.position)
# Create or split explicit PRNG key(s) (`starts`) for reproducible randomness.
starts = jax.vmap(reset)(jax.random.split(key, 1000)).position
assert 0.4 < float(starts.mean()) < 0.6
print('fraction starting at one:', float(starts.mean()))

# Reference solution. Try the exercise before reading this.
# Exercise solution
s = State(jnp.int32(0), jnp.int32(0), jnp.bool_(False))
total = 0.
for i in range(3):
    s, r, term, trunc = step(s, jnp.int32(0), 3)
    # Accumulate the next contribution into `total`.
    total += float(r)
    assert not bool(term) and bool(trunc) == (i == 2)
assert abs(total + .03) < 1e-6
s2, r, term, trunc = step(s, jnp.int32(1), 3)
assert int(s2.position) == 0 and int(s2.elapsed) == 3 and float(r) == 0.

# Reference practice: Separate the two target meanings
# Separate the two target meanings (Transfer / diagnosis): The arithmetic exposes a modeling choice.
r, gamma, value = -.01, .9, 2.
continuing_timeout = r + gamma * value
terminal = r
finite_horizon_timeout = r
assert abs(continuing_timeout - 1.79) < 1e-7
assert terminal == finite_horizon_timeout == -.01

# Reference practice: Enumerate all one-step moves
# Enumerate all one-step moves (Transfer / diagnosis)
# Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
compiled = jax.jit(step)
for position in range(3):
    for action in [0, 1]:
        s = State(jnp.int32(position), jnp.int32(0), jnp.bool_(False))
        nxt, reward, term, trunc = compiled(s, jnp.int32(action))
        expected = min(3, max(0, position + (1 if action else -1)))
        assert int(nxt.position) == expected
        assert bool(term) == (expected == 3)
        assert abs(float(reward) - (1. if expected == 3 else -.01)) < 1e-6
print("PASS: rl-01")
