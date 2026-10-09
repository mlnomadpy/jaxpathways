"""Batch environments and scan rollouts: worked experiments and reference solutions. CPU checks."""

# Reuse the verified environment
# Step 1: Reuse the verified environment
"""Finite-horizon tabular policy training. CPU teaching environment, not a benchmark."""
# Import functools (partial) for this computation.
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


# Define `State` module / container with explicit state and forward pass:
class State(NamedTuple):
    # Execute `position: jax.Array`
    position: jax.Array
    # Execute `elapsed: jax.Array`
    elapsed: jax.Array
    # Execute `done: jax.Array`
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
    # Compute `active` from `~state.done`
    active = ~state.done
    # Combine or mask array elements to form `proposed`.
    proposed = jnp.clip(state.position + 2 * action - 1, 0, 3)
    # Combine or mask array elements to form `position`.
    position = jnp.where(active, proposed, state.position)
    # Compute `elapsed` from `state.elapsed + active.astype(jnp.int32)`
    elapsed = state.elapsed + active.astype(jnp.int32)
    # Compute `terminated` from `active & (position == 3)`
    terminated = active & (position == 3)
    # Compute `truncated` from `active & ~terminated & (elapsed >= horizon)`
    truncated = active & ~terminated & (elapsed >= horizon)
    # Combine or mask array elements to form `reward`.
    reward = jnp.where(active, jnp.where(terminated, 1., -.01), 0.)
    # Return `(State(position, elapsed, state.done | terminated | truncated), reward, terminated, truncated)` to the caller.
    return State(position, elapsed, state.done | terminated | truncated), reward, terminated, truncated

# Collect a batch and compute targets
# Step 2 — Collect a batch and compute targets: scan emits time-major arrays; vmap keeps environments independent...
def log_probability(theta, observations, actions):
    # Reduce across the target axis to summarize `logits`.
    logits = theta[jnp.minimum(observations, 2)]
    # Return `jnp.where(actions == 1, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits))` to the caller.
    return jnp.where(actions == 1, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits))


# Define and JIT-compile `rollout(theta, key, batch_size, horizon)` so XLA traces and fuses the operations:
@partial(jax.jit, static_argnames=('batch_size', 'horizon'))
# Function `rollout(theta, key, batch_size, horizon)` implementing this stage's computation:
def rollout(theta, key, batch_size=256, horizon=8):
    # Create or split explicit PRNG key(s) (`(reset_key, action_key)`) for reproducible randomness.
    reset_key, action_key = jax.random.split(key)
    # Create or split explicit PRNG key(s) (`states`) for reproducible randomness.
    states = jax.vmap(reset)(jax.random.split(reset_key, batch_size))
    # Create or split explicit PRNG key(s) (`time_keys`) for reproducible randomness.
    time_keys = jax.random.split(action_key, horizon)
    # Function `body(states, time_key)` implementing this stage's computation:
    def body(states, time_key):
        # Compute `observations` from `states.position`
        observations = states.position
        # Split the PRNG key deterministically into independent subkeys (`keys`).
        keys = jax.random.split(time_key, batch_size)
        # Reduce across the target axis to summarize `probabilities`.
        probabilities = jax.nn.sigmoid(theta[jnp.minimum(observations, 2)])
        # Draw pseudorandom samples for `actions` using the explicit RNG state.
        actions = jax.vmap(jax.random.bernoulli)(keys, probabilities).astype(jnp.int32)
        # Vectorize across the batch dimension without a Python loop (`(next_states, rewards, terminated, truncated)`).
        next_states, rewards, terminated, truncated = jax.vmap(
            lambda s, a: step(s, a, horizon))(states, actions)
        # Evaluate `observation=observations, action=actions, reward=rewards, active=~states.done, terminated=terminated, truncated=truncated, old_logp=log_probability(theta, observations, actions)` and convert the result into Python scalar/collection `record`.
        record = dict(observation=observations, action=actions, reward=rewards,
                      active=~states.done, terminated=terminated, truncated=truncated,
                      old_logp=log_probability(theta, observations, actions))
        # Return `(next_states, record)` to the caller.
        return next_states, record
    # Run a compiled sequential scan over the time/step axis (`(final, records)`).
    final, records = jax.lax.scan(body, states, time_keys)
    # Return `(final, records)` to the caller.
    return final, records


# Function `returns_to_go(rewards)` implementing this stage's computation:
def returns_to_go(rewards):
    """Undiscounted finite-episode objective; rewards after done are already zero."""
    # Return `jnp.cumsum(rewards[::-1], axis=0)[::-1]` to the caller.
    return jnp.cumsum(rewards[::-1], axis=0)[::-1]

# Check shapes and episode boundaries
# Step 3 — Check shapes and episode boundaries: Exactly one ending event and zero padded rewards are stronger...
# Construct `theta` via `jnp.zeros(3)`
theta = jnp.zeros(3)
# Create or split explicit PRNG key(s) (`(final, records)`) for reproducible randomness.
final, records = rollout(theta, jax.random.key(5), batch_size=64, horizon=8)
# Check tensor shape invariant: `records['reward'].shape == (8`
assert records['reward'].shape == (8, 64)
# Assert invariant `bool(final.done.all())` holds
assert bool(final.done.all())
# Assert invariant `bool(jnp.all(jnp.sum(records['terminated'] | records['truncated']` holds
assert bool(jnp.all(jnp.sum(records['terminated'] | records['truncated'], axis=0) == 1))
# Assert invariant `bool(jnp.all(jnp.where(records['active']` holds
assert bool(jnp.all(jnp.where(records['active'], 0., records['reward']) == 0.))
# Reduce along axis=0 to compute `episode_returns`.
episode_returns = records['reward'].sum(axis=0)
# Print the observed values to compare against the expected result.
print('return mean:', float(episode_returns.mean()))
# Print diagnostic summary of the computed outputs.
print('active environments by transition:', records['active'].sum(axis=1))

# Complete runnable example (rl-02)
"""Finite-horizon tabular policy training. CPU teaching environment, not a benchmark."""
# Import functools (partial) for this computation.
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


# Define `State` module / container with explicit state and forward pass:
class State(NamedTuple):
    # Execute `position: jax.Array`
    position: jax.Array
    # Execute `elapsed: jax.Array`
    elapsed: jax.Array
    # Execute `done: jax.Array`
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
    # Compute `active` from `~state.done`
    active = ~state.done
    # Combine or mask array elements to form `proposed`.
    proposed = jnp.clip(state.position + 2 * action - 1, 0, 3)
    # Combine or mask array elements to form `position`.
    position = jnp.where(active, proposed, state.position)
    # Compute `elapsed` from `state.elapsed + active.astype(jnp.int32)`
    elapsed = state.elapsed + active.astype(jnp.int32)
    # Compute `terminated` from `active & (position == 3)`
    terminated = active & (position == 3)
    # Compute `truncated` from `active & ~terminated & (elapsed >= horizon)`
    truncated = active & ~terminated & (elapsed >= horizon)
    # Combine or mask array elements to form `reward`.
    reward = jnp.where(active, jnp.where(terminated, 1., -.01), 0.)
    # Return `(State(position, elapsed, state.done | terminated | truncated), reward, terminated, truncated)` to the caller.
    return State(position, elapsed, state.done | terminated | truncated), reward, terminated, truncated

# Step 2 — Collect a batch and compute targets: scan emits time-major arrays; vmap keeps environments independent...
def log_probability(theta, observations, actions):
    # Reduce across the target axis to summarize `logits`.
    logits = theta[jnp.minimum(observations, 2)]
    # Return `jnp.where(actions == 1, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits))` to the caller.
    return jnp.where(actions == 1, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits))


# Define and JIT-compile `rollout(theta, key, batch_size, horizon)` so XLA traces and fuses the operations:
@partial(jax.jit, static_argnames=('batch_size', 'horizon'))
# Function `rollout(theta, key, batch_size, horizon)` implementing this stage's computation:
def rollout(theta, key, batch_size=256, horizon=8):
    # Create or split explicit PRNG key(s) (`(reset_key, action_key)`) for reproducible randomness.
    reset_key, action_key = jax.random.split(key)
    # Create or split explicit PRNG key(s) (`states`) for reproducible randomness.
    states = jax.vmap(reset)(jax.random.split(reset_key, batch_size))
    # Create or split explicit PRNG key(s) (`time_keys`) for reproducible randomness.
    time_keys = jax.random.split(action_key, horizon)
    # Function `body(states, time_key)` implementing this stage's computation:
    def body(states, time_key):
        # Compute `observations` from `states.position`
        observations = states.position
        # Split the PRNG key deterministically into independent subkeys (`keys`).
        keys = jax.random.split(time_key, batch_size)
        # Reduce across the target axis to summarize `probabilities`.
        probabilities = jax.nn.sigmoid(theta[jnp.minimum(observations, 2)])
        # Draw pseudorandom samples for `actions` using the explicit RNG state.
        actions = jax.vmap(jax.random.bernoulli)(keys, probabilities).astype(jnp.int32)
        # Vectorize across the batch dimension without a Python loop (`(next_states, rewards, terminated, truncated)`).
        next_states, rewards, terminated, truncated = jax.vmap(
            lambda s, a: step(s, a, horizon))(states, actions)
        # Evaluate `observation=observations, action=actions, reward=rewards, active=~states.done, terminated=terminated, truncated=truncated, old_logp=log_probability(theta, observations, actions)` and convert the result into Python scalar/collection `record`.
        record = dict(observation=observations, action=actions, reward=rewards,
                      active=~states.done, terminated=terminated, truncated=truncated,
                      old_logp=log_probability(theta, observations, actions))
        # Return `(next_states, record)` to the caller.
        return next_states, record
    # Run a compiled sequential scan over the time/step axis (`(final, records)`).
    final, records = jax.lax.scan(body, states, time_keys)
    # Return `(final, records)` to the caller.
    return final, records


# Function `returns_to_go(rewards)` implementing this stage's computation:
def returns_to_go(rewards):
    """Undiscounted finite-episode objective; rewards after done are already zero."""
    # Return `jnp.cumsum(rewards[::-1], axis=0)[::-1]` to the caller.
    return jnp.cumsum(rewards[::-1], axis=0)[::-1]

# Step 3 — Check shapes and episode boundaries: Exactly one ending event and zero padded rewards are stronger...
# Construct `theta` via `jnp.zeros(3)`
theta = jnp.zeros(3)
# Create or split explicit PRNG key(s) (`(final, records)`) for reproducible randomness.
final, records = rollout(theta, jax.random.key(5), batch_size=64, horizon=8)
# Check tensor shape invariant: `records['reward'].shape == (8`
assert records['reward'].shape == (8, 64)
# Assert invariant `bool(final.done.all())` holds
assert bool(final.done.all())
# Assert invariant `bool(jnp.all(jnp.sum(records['terminated'] | records['truncated']` holds
assert bool(jnp.all(jnp.sum(records['terminated'] | records['truncated'], axis=0) == 1))
# Assert invariant `bool(jnp.all(jnp.where(records['active']` holds
assert bool(jnp.all(jnp.where(records['active'], 0., records['reward']) == 0.))
# Reduce along axis=0 to compute `episode_returns`.
episode_returns = records['reward'].sum(axis=0)
# Print the observed values to compare against the expected result.
print('return mean:', float(episode_returns.mean()))
# Print diagnostic summary of the computed outputs.
print('active environments by transition:', records['active'].sum(axis=1))

# Figure data experiment
# Compute figure data for: The live batch shrinks while the array shape stays fixed
# Reduce across the target axis to summarize `visual_data`.
visual_data = {'kind':'line','x':list(range(1,9)),'xlabel':'transition index','ylabel':'active environments before transition','series':[{'label':'uniform random policy','y':records['active'].sum(1).tolist()}]}

# Experiment: Replay the entire collector
# Experiment — Replay the entire collector: The random stream is part of the experiment state.
# Create or split explicit PRNG key(s) (`again`) for reproducible randomness.
again = rollout(theta, jax.random.key(5), 64, 8)[1]
# Iterate over `name` to step through the computation:
for name in records:
    # Assert invariant `jnp.array_equal(records[name], again[name])` holds
    assert jnp.array_equal(records[name], again[name])
# Create or split explicit PRNG key(s) (`changed`) for reproducible randomness.
changed = rollout(theta, jax.random.key(6), 64, 8)[1]
# Assert invariant `not jnp.array_equal(records['action']` holds
assert not jnp.array_equal(records['action'], changed['action'])

# Experiment: Reward-to-go by hand
# Experiment — Reward-to-go by hand: Zero padded rewards preserve the finite-episode sum, while the...
# Construct `r` via `jnp.array([[-.01],[-.01],[1.],[0.]])`
r = jnp.array([[-.01],[-.01],[1.],[0.]])
# Run `returns_to_go` to compute `g`.
g = returns_to_go(r)
# Assert that `jnp.allclose(g[:,0], jnp.array([.98,.99,1.,0.]))`.
assert jnp.allclose(g[:,0], jnp.array([.98,.99,1.,0.]))
# Print the observed values to compare against the expected result.
print('reward-to-go:', g[:,0])

# Reference solution. Try the exercise before reading this.
# Exercise solution: Change the batch to 7 environments and the horizon to 5.
# Create or split explicit PRNG key(s) (`(final7, r7)`) for reproducible randomness.
final7, r7 = rollout(theta, jax.random.key(17), 7, 5)
# Check tensor shape invariant: `r7['reward'].shape == (5`
assert r7['reward'].shape == (5,7)
# Assert invariant `bool(final7.done.all())` holds
assert bool(final7.done.all())
# Assert invariant `jnp.array_equal((r7['terminated'] | r7['truncated']).sum(0)` holds
assert jnp.array_equal((r7['terminated'] | r7['truncated']).sum(0), jnp.ones(7))
# Assert invariant `bool(jnp.all(jnp.where(r7['active']` holds
assert bool(jnp.all(jnp.where(r7['active'],0.,r7['reward']) == 0.))

# Reference practice: Catch a shifted mask
# Catch a shifted mask (Transfer / diagnosis): The broken mask removes the action responsible for success,...
s = State(jnp.int32(1),jnp.int32(0),jnp.bool_(False))
# Compute `correct, broken` from `0., 0.`
correct, broken = 0., 0.
# Iterate over `a` to step through the computation:
for a in [1,1]:
    # Compute `active` from `~s.done`
    active = ~s.done
    # Run `step` to compute `(s, reward, _, _)`.
    s, reward, _, _ = step(s,jnp.int32(a))
    # Accumulate the next contribution into `correct`.
    correct += float(reward*active)
    # Accumulate the next contribution into `broken`.
    broken += float(reward*(~s.done))
# Assert that `abs(correct-.99) < 1e-6`.
assert abs(correct-.99) < 1e-6
# Assert that `abs(broken+.01) < 1e-6`.
assert abs(broken+.01) < 1e-6

# Reference practice: Check a deterministic policy limit
# Check a deterministic policy limit (Transfer / diagnosis): The expected return is calculated from the initial position...
# Create or split explicit PRNG key(s) (`(_, easy)`) for reproducible randomness.
_, easy = rollout(jnp.full(3,100.),jax.random.key(42),23,8)
# Combine or mask array elements to form `expected`.
expected = jnp.where(easy['observation'][0] == 0,.98,.99)
# Assert that `jnp.allclose(easy['reward'].sum(0),expected,atol=1e-6)`.
assert jnp.allclose(easy['reward'].sum(0),expected,atol=1e-6)
# Assert invariant `jnp.array_equal(easy['active'].sum(0)` holds
assert jnp.array_equal(easy['active'].sum(0),3-easy['observation'][0])
print("PASS: rl-02")
