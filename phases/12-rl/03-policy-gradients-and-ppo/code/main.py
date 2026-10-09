"""Policy gradients and PPO: worked experiments and reference solutions. CPU checks."""

# Reuse the environment and collector
# Step 1: Reuse the environment and collector
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

# Function `log_probability(theta, observations, actions)` implementing this stage's computation:
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

# Add objectives and the repeated training loop
# Step 2 — Add objectives and the repeated training loop: Keep a frozen rollout per update, then collect again using a fresh...
# Define `policy_loss(theta, records, method, clip)` to evaluate the objective and its automatic derivatives:
def policy_loss(theta, records, method='ppo', clip=.2):
    """Normalize by episodes, not live transitions. Freeze behavior data/returns."""
    # Run `jax.lax.stop_gradient` to compute `advantage`.
    advantage = jax.lax.stop_gradient(returns_to_go(records['reward']))
    # Run `jax.lax.stop_gradient` to compute `old_logp`.
    old_logp = jax.lax.stop_gradient(records['old_logp'])
    # Run `log_probability` to compute `new_logp`.
    new_logp = log_probability(theta, records['observation'], records['action'])
    # Branch on condition `method == 'reinforce'`:
    if method == 'reinforce':
        terms = new_logp * advantage
    elif method == 'ppo':
        ratio = jnp.exp(new_logp - old_logp)
        terms = jnp.minimum(ratio * advantage, jnp.clip(ratio, 1-clip, 1+clip) * advantage)
    else:
        raise ValueError('method must be reinforce or ppo')
    # Return `-jnp.sum(jnp.where(records['active'], terms, 0.0)) / records['reward'].shape[1]` to the caller.
    return -jnp.sum(jnp.where(records['active'], terms, 0.)) / records['reward'].shape[1]


# Define and JIT-compile `exact_return(theta, horizon)` so XLA traces and fuses the operations:
@partial(jax.jit, static_argnames=('horizon',))
# Function `exact_return(theta, horizon)` implementing this stage's computation:
def exact_return(theta, horizon=8):
    """Dynamic programming over states, independent of sampled rollouts."""
    # Run `jax.nn.sigmoid` to compute `p`.
    p = jax.nn.sigmoid(theta)
    # Construct `states` via `jnp.arange(3)`
    states = jnp.arange(3)
    # Reduce across the target axis to summarize `left`.
    left = jnp.maximum(states-1, 0)
    # Compute `right` from `states+1`
    right = states+1
    # Function `backup(_, values)` implementing this stage's computation:
    def backup(_, values):
        # Compute `q_left` from `-.01 + values[left]`
        q_left = -.01 + values[left]
        # Combine or mask array elements to form `q_right`.
        q_right = jnp.where(right == 3, 1., -.01 + values[right])
        # Return `jnp.concatenate(((1 - p) * q_left + p * q_right, jnp.zeros(1)))` to the caller.
        return jnp.concatenate(((1-p)*q_left+p*q_right, jnp.zeros(1)))
    # Allocate initialized array `values` with the specified shape and dtype.
    values = jax.lax.fori_loop(0, horizon, backup, jnp.zeros(4))
    # Return `jnp.mean(values[:2])` to the caller.
    return jnp.mean(values[:2])


# Define and JIT-compile `_train(seed, updates, batch_size, horizon...)` so XLA traces and fuses the operations:
@partial(jax.jit, static_argnames=('updates', 'batch_size', 'horizon', 'method', 'epochs'))
# Function `_train(seed, updates, batch_size, horizon, ...)` implementing this stage's computation:
def _train(seed, updates, batch_size, horizon, method, epochs, learning_rate):
    # Create or split explicit PRNG key(s) (`key`) for reproducible randomness.
    key = jax.random.key(seed)
    # Construct `theta` via `jnp.zeros(3)`
    theta = jnp.zeros(3)
    # Define `update(carry, _)` to evaluate the objective and its automatic derivatives:
    def update(carry, _):
        # Compute `theta, key` from `carry`
        theta, key = carry
        # Create or split explicit PRNG key(s) (`(key, sample_key)`) for reproducible randomness.
        key, sample_key = jax.random.split(key)
        # Run `rollout` to compute `(_, records)`.
        _, records = rollout(theta, sample_key, batch_size, horizon)
        # Define `epoch(_, candidate)` to evaluate the objective and its automatic derivatives:
        def epoch(_, candidate):
            # Differentiate the objective to obtain `grad` via automatic differentiation.
            grad = jax.grad(policy_loss)(candidate, records, method)
            # Return `candidate - learning_rate * grad` to the caller.
            return candidate - learning_rate * grad
        # Run `jax.lax.fori_loop` to compute `theta`.
        theta = jax.lax.fori_loop(0, epochs, epoch, theta)
        # Return `((theta, key), exact_return(theta, horizon))` to the caller.
        return (theta, key), exact_return(theta, horizon)
    # Run a compiled sequential scan over the time/step axis (`((theta, _), history)`).
    (theta, _), history = jax.lax.scan(update, (theta, key), None, length=updates)
    # Return `(theta, jnp.concatenate((exact_return(jnp.zeros(3), horizon)[None], history)))` to the caller.
    return theta, jnp.concatenate((exact_return(jnp.zeros(3), horizon)[None], history))


# Function `train(seed, updates, batch_size, horizon, ...)` implementing this stage's computation:
def train(seed=0, updates=40, batch_size=256, horizon=8, method='ppo', epochs=3, learning_rate=.5):
    # Guard input contract (`updates < 1 or batch_size < 1 or horizon < 1 or (epochs < 1) or (learning_rate <= 0)`) and fail fast if violated.
    if updates < 1 or batch_size < 1 or horizon < 1 or epochs < 1 or learning_rate <= 0:
        raise ValueError('positive updates, batch_size, horizon, epochs and learning_rate required')
    # Guard input contract (`method not in ('ppo', 'reinforce')`) and fail fast if violated.
    if method not in ('ppo', 'reinforce'):
        raise ValueError('unknown method')
    # Guard input contract (`method == 'reinforce' and epochs != 1`) and fail fast if violated.
    if method == 'reinforce' and epochs != 1:
        raise ValueError('REINFORCE uses one update per fresh on-policy rollout')
    # Return `_train(seed, updates, batch_size, horizon, method, epochs, learning_rate)` to the caller.
    return _train(seed, updates, batch_size, horizon, method, epochs, learning_rate)

# Train and compare to a derivative reference
# Step 3 — Train and compare to a derivative reference: The seed-specific return is measured after actual interaction and...
theta_pg, history_pg = train(seed=0, method='reinforce', epochs=1)
# Run `train` to compute `(theta_ppo, history_ppo)`.
theta_ppo, history_ppo = train(seed=0, method='ppo', epochs=3)
# Assert invariant `history_pg[-1] > .9 and history_ppo[-1] > .9` holds
assert history_pg[-1] > .9 and history_ppo[-1] > .9
# Check numerical equivalence within tolerance: `jnp.allclose(history_pg[0], .466875, atol=1e-6)`
assert jnp.allclose(history_pg[0], .466875, atol=1e-6)
# Print the observed values to compare against the expected result.
print('REINFORCE initial/final exact return:', history_pg[0], history_pg[-1])
# Print diagnostic summary of the computed outputs.
print('PPO initial/final exact return:', history_ppo[0], history_ppo[-1])
# Compare a derivative of exact expectation with central differences.
z = jnp.array([-.2, .3, .7])
eps = 1e-3
# Construct `d` via `jnp.array([0., 1., 0.])`
d = jnp.array([0., 1., 0.])
# Compute `fd` from `(exact_return(z+eps*d)-exact_return(z-eps*d))/(2*eps)`
fd = (exact_return(z+eps*d)-exact_return(z-eps*d))/(2*eps)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(fd, jax.grad(exact_return)(z)[1], atol=1e-4)

# Complete runnable example (rl-03)
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

# Function `log_probability(theta, observations, actions)` implementing this stage's computation:
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

# Step 2 — Add objectives and the repeated training loop: Keep a frozen rollout per update, then collect again using a fresh...
# Define `policy_loss(theta, records, method, clip)` to evaluate the objective and its automatic derivatives:
def policy_loss(theta, records, method='ppo', clip=.2):
    """Normalize by episodes, not live transitions. Freeze behavior data/returns."""
    # Run `jax.lax.stop_gradient` to compute `advantage`.
    advantage = jax.lax.stop_gradient(returns_to_go(records['reward']))
    # Run `jax.lax.stop_gradient` to compute `old_logp`.
    old_logp = jax.lax.stop_gradient(records['old_logp'])
    # Run `log_probability` to compute `new_logp`.
    new_logp = log_probability(theta, records['observation'], records['action'])
    # Branch on condition `method == 'reinforce'`:
    if method == 'reinforce':
        terms = new_logp * advantage
    elif method == 'ppo':
        ratio = jnp.exp(new_logp - old_logp)
        terms = jnp.minimum(ratio * advantage, jnp.clip(ratio, 1-clip, 1+clip) * advantage)
    else:
        raise ValueError('method must be reinforce or ppo')
    # Return `-jnp.sum(jnp.where(records['active'], terms, 0.0)) / records['reward'].shape[1]` to the caller.
    return -jnp.sum(jnp.where(records['active'], terms, 0.)) / records['reward'].shape[1]


# Define and JIT-compile `exact_return(theta, horizon)` so XLA traces and fuses the operations:
@partial(jax.jit, static_argnames=('horizon',))
# Function `exact_return(theta, horizon)` implementing this stage's computation:
def exact_return(theta, horizon=8):
    """Dynamic programming over states, independent of sampled rollouts."""
    # Run `jax.nn.sigmoid` to compute `p`.
    p = jax.nn.sigmoid(theta)
    # Construct `states` via `jnp.arange(3)`
    states = jnp.arange(3)
    # Reduce across the target axis to summarize `left`.
    left = jnp.maximum(states-1, 0)
    # Compute `right` from `states+1`
    right = states+1
    # Function `backup(_, values)` implementing this stage's computation:
    def backup(_, values):
        # Compute `q_left` from `-.01 + values[left]`
        q_left = -.01 + values[left]
        # Combine or mask array elements to form `q_right`.
        q_right = jnp.where(right == 3, 1., -.01 + values[right])
        # Return `jnp.concatenate(((1 - p) * q_left + p * q_right, jnp.zeros(1)))` to the caller.
        return jnp.concatenate(((1-p)*q_left+p*q_right, jnp.zeros(1)))
    # Allocate initialized array `values` with the specified shape and dtype.
    values = jax.lax.fori_loop(0, horizon, backup, jnp.zeros(4))
    # Return `jnp.mean(values[:2])` to the caller.
    return jnp.mean(values[:2])


# Define and JIT-compile `_train(seed, updates, batch_size, horizon...)` so XLA traces and fuses the operations:
@partial(jax.jit, static_argnames=('updates', 'batch_size', 'horizon', 'method', 'epochs'))
# Function `_train(seed, updates, batch_size, horizon, ...)` implementing this stage's computation:
def _train(seed, updates, batch_size, horizon, method, epochs, learning_rate):
    # Create or split explicit PRNG key(s) (`key`) for reproducible randomness.
    key = jax.random.key(seed)
    # Construct `theta` via `jnp.zeros(3)`
    theta = jnp.zeros(3)
    # Define `update(carry, _)` to evaluate the objective and its automatic derivatives:
    def update(carry, _):
        # Compute `theta, key` from `carry`
        theta, key = carry
        # Create or split explicit PRNG key(s) (`(key, sample_key)`) for reproducible randomness.
        key, sample_key = jax.random.split(key)
        # Run `rollout` to compute `(_, records)`.
        _, records = rollout(theta, sample_key, batch_size, horizon)
        # Define `epoch(_, candidate)` to evaluate the objective and its automatic derivatives:
        def epoch(_, candidate):
            # Differentiate the objective to obtain `grad` via automatic differentiation.
            grad = jax.grad(policy_loss)(candidate, records, method)
            # Return `candidate - learning_rate * grad` to the caller.
            return candidate - learning_rate * grad
        # Run `jax.lax.fori_loop` to compute `theta`.
        theta = jax.lax.fori_loop(0, epochs, epoch, theta)
        # Return `((theta, key), exact_return(theta, horizon))` to the caller.
        return (theta, key), exact_return(theta, horizon)
    # Run a compiled sequential scan over the time/step axis (`((theta, _), history)`).
    (theta, _), history = jax.lax.scan(update, (theta, key), None, length=updates)
    # Return `(theta, jnp.concatenate((exact_return(jnp.zeros(3), horizon)[None], history)))` to the caller.
    return theta, jnp.concatenate((exact_return(jnp.zeros(3), horizon)[None], history))


# Function `train(seed, updates, batch_size, horizon, ...)` implementing this stage's computation:
def train(seed=0, updates=40, batch_size=256, horizon=8, method='ppo', epochs=3, learning_rate=.5):
    # Guard input contract (`updates < 1 or batch_size < 1 or horizon < 1 or (epochs < 1) or (learning_rate <= 0)`) and fail fast if violated.
    if updates < 1 or batch_size < 1 or horizon < 1 or epochs < 1 or learning_rate <= 0:
        raise ValueError('positive updates, batch_size, horizon, epochs and learning_rate required')
    # Guard input contract (`method not in ('ppo', 'reinforce')`) and fail fast if violated.
    if method not in ('ppo', 'reinforce'):
        raise ValueError('unknown method')
    # Guard input contract (`method == 'reinforce' and epochs != 1`) and fail fast if violated.
    if method == 'reinforce' and epochs != 1:
        raise ValueError('REINFORCE uses one update per fresh on-policy rollout')
    # Return `_train(seed, updates, batch_size, horizon, method, epochs, learning_rate)` to the caller.
    return _train(seed, updates, batch_size, horizon, method, epochs, learning_rate)

# Step 3 — Train and compare to a derivative reference: The seed-specific return is measured after actual interaction and...
theta_pg, history_pg = train(seed=0, method='reinforce', epochs=1)
# Run `train` to compute `(theta_ppo, history_ppo)`.
theta_ppo, history_ppo = train(seed=0, method='ppo', epochs=3)
# Assert invariant `history_pg[-1] > .9 and history_ppo[-1] > .9` holds
assert history_pg[-1] > .9 and history_ppo[-1] > .9
# Check numerical equivalence within tolerance: `jnp.allclose(history_pg[0], .466875, atol=1e-6)`
assert jnp.allclose(history_pg[0], .466875, atol=1e-6)
# Print the observed values to compare against the expected result.
print('REINFORCE initial/final exact return:', history_pg[0], history_pg[-1])
# Print diagnostic summary of the computed outputs.
print('PPO initial/final exact return:', history_ppo[0], history_ppo[-1])
# Compare a derivative of exact expectation with central differences.
z = jnp.array([-.2, .3, .7])
eps = 1e-3
# Construct `d` via `jnp.array([0., 1., 0.])`
d = jnp.array([0., 1., 0.])
# Compute `fd` from `(exact_return(z+eps*d)-exact_return(z-eps*d))/(2*eps)`
fd = (exact_return(z+eps*d)-exact_return(z-eps*d))/(2*eps)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(fd, jax.grad(exact_return)(z)[1], atol=1e-4)

# Figure data experiment
# Compute figure data for: Two sampled training procedures, one exact evaluator
# Combine or mask array elements to form `visual_data`.
visual_data = {'kind':'line','x':list(range(41)),'xlabel':'fresh rollout batches','ylabel':'exact expected episode return','series':[{'label':'REINFORCE: 1 epoch/batch','y':history_pg.tolist()},{'label':'clipped updates: 3 epochs/batch','y':history_ppo.tolist()}]}

# Experiment: Check clipping on both signs
# Experiment — Check clipping on both signs: The minimum prevents an update from gaining unlimited credit for...
# Construct `ratio` via `jnp.array([.6,1.,1.4])`
ratio = jnp.array([.6,1.,1.4])
# Reduce across the target axis to summarize `positive`.
positive = jnp.minimum(ratio,jnp.clip(ratio,.8,1.2))
# Reduce across the target axis to summarize `negative`.
negative = jnp.minimum(-ratio,-jnp.clip(ratio,.8,1.2))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(positive,jnp.array([.6,1.,1.2]))
# Check numerical equivalence within tolerance: `jnp.allclose(negative,jnp.array([-.8,-1.,-1.4]))`
assert jnp.allclose(negative,jnp.array([-.8,-1.,-1.4]))
# Print the observed values to compare against the expected result.
print('positive / negative targets:', positive, negative)

# Experiment: Compare sampled and exact gradients
# Experiment — Compare sampled and exact gradients: The large batch reduces Monte Carlo noise.
# Create or split explicit PRNG key(s) (`(_, batch)`) for reproducible randomness.
_, batch = rollout(z,jax.random.key(902),32768,8)
# Differentiate the objective to obtain `sampled` via automatic differentiation.
sampled = -jax.grad(policy_loss)(z,batch,'reinforce')
# Differentiate the objective to obtain `exact` via automatic differentiation.
exact = jax.grad(exact_return)(z)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(sampled,exact,atol=.012,rtol=.04)
# Print the observed values to compare against the expected result.
print('sampled / exact gradient:', sampled, exact)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Run REINFORCE with a different training seed and retain the initial...
other_theta, other_history = train(seed=11,method='reinforce',epochs=1)
# Assert invariant `other_history[-1] > other_history[0] + .35` holds
assert other_history[-1] > other_history[0] + .35
# Assert invariant `other_history[-1] > .9` holds
assert other_history[-1] > .9
# Assert invariant `not jnp.array_equal(other_theta,theta_pg)` holds
assert not jnp.array_equal(other_theta,theta_pg)
# Print the observed values to compare against the expected result.
print('changed seed final exact return:', other_history[-1])

# Reference practice: Inspect a frozen behavior reference
# Inspect a frozen behavior reference (Transfer / diagnosis): The candidate changes while the behavior policy remains the...
# Create or split explicit PRNG key(s) (`(_, b)`) for reproducible randomness.
_, b = rollout(jnp.zeros(3),jax.random.key(77),128,8)
# Compute `old` from `b['old_logp'].copy()`
old = b['old_logp'].copy()
# Construct `newlog` via `log_probability(jnp.ones(3),b['observation'],b['acti...`
newlog = log_probability(jnp.ones(3),b['observation'],b['action'])
# Run `jnp.exp` to compute `ratios`.
ratios = jnp.exp(newlog-old)
# Assert invariant `jnp.array_equal(old` holds
assert jnp.array_equal(old,b['old_logp'])
# Assert invariant `float(jnp.min(ratios)) < 1 < float(jnp.max(ratios))` holds
assert float(jnp.min(ratios)) < 1 < float(jnp.max(ratios))
# Differentiate the objective to obtain `old_grad` via automatic differentiation.
old_grad = jax.grad(lambda lp: policy_loss(jnp.ones(3),dict(b,old_logp=lp)))(old)
# Assert invariant `jnp.all(old_grad == 0)` holds
assert jnp.all(old_grad == 0)

# Reference practice: Change the time budget
# Change the time budget (Transfer / diagnosis): Changing the horizon changes the task itself.
# Compute `short` from `exact_return(jnp.full(3,100.),horizon=2)`
short = exact_return(jnp.full(3,100.),horizon=2)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(short,(-.02+.99)/2,atol=1e-6)
# Assert invariant `short < exact_return(jnp.full(3,100.),horizon=8)` holds
assert short < exact_return(jnp.full(3,100.),horizon=8)
print("PASS: rl-03")
