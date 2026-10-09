# Policy gradients and PPO

Phase 12: Reinforcement learning · about 110 minutes · CPU

## What you will be able to do

- Derive and implement the score-function policy gradient for complete episodes.
- Train with fresh stochastic rollouts and a fixed behavior policy per PPO batch.
- Explain both signs of the clipping objective and verify improvement against exact expected return.

## The problem

The collector produces experiences, but which parameter change makes useful actions more likely? We will train an actual stochastic policy from freshly sampled episodes, first with REINFORCE and then with repeated clipped policy updates. An exact calculation in this tiny environment will tell us whether the policy improved, independently of its noisy training rewards.

## The idea

A policy gradient increases the likelihood of actions with favorable estimated advantage. PPO compares a current policy with the rollout-time policy and clips part of the surrogate objective. The sign of the advantage determines which changes the clipping discourages.

## The clipped objective depends on the advantage sign

Let $r$ be the current-to-old action-probability ratio. With positive advantage, making $r$ much larger than the upper clipping threshold stops improving the clipped surrogate. With negative advantage, moving below the lower threshold has the corresponding saturation.

This does not impose a hard bound on the new policy probabilities. Shared parameters, multiple updates and other loss terms can still move the policy beyond the threshold. Inspect the ratio distribution and KL behavior rather than claiming clipping guarantees a trust region.

Keep old probabilities and advantage estimates fixed for the intended inner updates. The training-return curve reports the particular sampled runs; use the same evaluator and multiple seeds before concluding one procedure reliably performs better.

### Pause and reason

Is a ratio outside the clipping interval proof that PPO was implemented incorrectly?

<details><summary>Compare your reasoning</summary>

No. Clipping modifies the objective incentive, not a hard constraint on ratios. Check the actual surrogate calculation, fixed rollout quantities and update behavior.

</details>

## The derivative does not pass through the sampled action

Let $\theta$ denote the three position logits. We maximize expected episode return $J(\theta)$. The score-function identity gives the expectation below. Rewards before an action do not depend on that action, so reward-to-go is a valid credit signal. Our baseline is zero; we do not fit a critic or use generalized advantage estimation. We freeze sampled actions and targets with stop_gradient. Trying to differentiate through Bernoulli sampling is a different, inappropriate estimator for this exercise.

$$
\nabla J(\theta)=\mathbb{E}\!\left[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t\right]
$$

## Normalize by the object you promised to optimize

The objective is return per episode. We sum live action terms and divide by the number of episodes $B$, not the number of live transitions. Dividing by live transitions changes the batch-dependent scale as the policy learns to finish sooner. Each REINFORCE update uses a fresh on-policy rollout once. The learning rate is $0.5$ for this tabular fixture; that value is not a recommendation for neural agents.

## Freeze the policy that collected the batch

For PPO, store old log probabilities at collection time and keep them fixed through all $3$ optimization epochs. The ratio is computed by subtracting log probabilities before exponentiating. At the first epoch the ratio is $1$. It changes as the candidate policy changes; overwriting old probabilities each epoch would erase the reference point and defeat clipping. Our target $\hat A_t$ is reward-to-go with zero baseline, a noisy but valid first-epoch policy-gradient estimator.

$$
\rho_t(\theta)=\exp\!\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)\right)
$$

## The minimum matters for negative advantages

We maximize the minimum of the ordinary ratio-weighted target and its clipped alternative, with $\epsilon=0.2$. For a positive target $1$ and ratio $1.4$, the clipped objective is $1.2$: extra probability gets no extra reward. For a negative target $-1$ and ratio $0.6$, it is $-0.8$: reducing probability further gets no extra reward. The opposite directions remain penalized. Clipping a ratio inside every term without the minimum is not the same objective.

$$
L^{\mathrm{clip}}(\theta)=\frac1B\sum_{b,t}\min\!\left(\rho_{t,b}\hat A_{t,b},\operatorname{clip}(\rho_{t,b},1-\epsilon,1+\epsilon)\hat A_{t,b}\right)
$$

## Clipping is not a guaranteed trust region

The surrogate has flat regions for particular sampled actions; it does not constrain every probability or guarantee improvement in true return. Several parameters share data in a neural model, unseen states are absent, and multiple gradient steps can move the policy substantially. Our complete learning loop repeatedly collects fresh episodes and optimizes the frozen batch. It is a small clipped policy-gradient agent, not a production PPO stack: no critic, entropy bonus, observation normalization or minibatch scheduler is included.

## Calculate expected return without sampling

This environment has only three unfinished positions, so dynamic programming can average left and right outcomes exactly for each remaining step budget. Start from zero remaining reward, back up the transition expectation $8$ times, and average the values at initial positions $0$ and $1$. This evaluator does not use rollout samples. An always-left policy earns $-0.08$, the uniform policy earns $0.466875$, and always-right earns $0.985$. It is an unusually strong diagnostic available in a tiny fixture, not a scalable evaluator for arbitrary environments.

## Reuse the environment and collector

Create main.py with this block. Run python3 main.py in the course CPU environment.

```python
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
        # Evaluate `observations` from the current inputs and state.
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
```

These functions generate genuine trajectories. The policy update will not differentiate through their sampled actions.

## Add objectives and the repeated training loop

Append this block to main.py. Run python3 main.py in the course CPU environment.

```python
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
    # Initialize array `states` with explicit values and shape.
    states = jnp.arange(3)
    # Reduce across the target axis to summarize `left`.
    left = jnp.maximum(states-1, 0)
    # Evaluate `right` from the current inputs and state.
    right = states+1
    # Function `backup(_, values)` implementing this stage's computation:
    def backup(_, values):
        # Evaluate `q_left` from the current inputs and state.
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
    # Initialize array `theta` with explicit values and shape.
    theta = jnp.zeros(3)
    # Define `update(carry, _)` to evaluate the objective and its automatic derivatives:
    def update(carry, _):
        # Evaluate `(theta, key)` from the current inputs and state.
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
```

Keep a frozen rollout per update, then collect again using a fresh key. The exact evaluator is separate from the training targets.

## Train and compare to a derivative reference

Append this block to main.py. Run python3 main.py in the course CPU environment.

```python
# Step 3 — Train and compare to a derivative reference: The seed-specific return is measured after actual interaction and...
theta_pg, history_pg = train(seed=0, method='reinforce', epochs=1)
# Run `train` to compute `(theta_ppo, history_ppo)`.
theta_ppo, history_ppo = train(seed=0, method='ppo', epochs=3)
# Verify contract: `history_pg[-1] > 0.9 and history_ppo[-1] > 0.9`.
assert history_pg[-1] > .9 and history_ppo[-1] > .9
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(history_pg[0], .466875, atol=1e-6)
# Print the observed values to compare against the expected result.
print('REINFORCE initial/final exact return:', history_pg[0], history_pg[-1])
# Print diagnostic summary of the computed outputs.
print('PPO initial/final exact return:', history_ppo[0], history_ppo[-1])
# Compare a derivative of exact expectation with central differences.
z = jnp.array([-.2, .3, .7]); eps = 1e-3
# Initialize array `d` with explicit values and shape.
d = jnp.array([0., 1., 0.])
# Evaluate `fd` from the current inputs and state.
fd = (exact_return(z+eps*d)-exact_return(z-eps*d))/(2*eps)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(fd, jax.grad(exact_return)(z)[1], atol=1e-4)
```

The seed-specific return is measured after actual interaction and updates, while the derivative check uses deterministic expectations.

## Run the example

```python
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
        # Evaluate `observations` from the current inputs and state.
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
    # Initialize array `states` with explicit values and shape.
    states = jnp.arange(3)
    # Reduce across the target axis to summarize `left`.
    left = jnp.maximum(states-1, 0)
    # Evaluate `right` from the current inputs and state.
    right = states+1
    # Function `backup(_, values)` implementing this stage's computation:
    def backup(_, values):
        # Evaluate `q_left` from the current inputs and state.
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
    # Initialize array `theta` with explicit values and shape.
    theta = jnp.zeros(3)
    # Define `update(carry, _)` to evaluate the objective and its automatic derivatives:
    def update(carry, _):
        # Evaluate `(theta, key)` from the current inputs and state.
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
# Verify contract: `history_pg[-1] > 0.9 and history_ppo[-1] > 0.9`.
assert history_pg[-1] > .9 and history_ppo[-1] > .9
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(history_pg[0], .466875, atol=1e-6)
# Print the observed values to compare against the expected result.
print('REINFORCE initial/final exact return:', history_pg[0], history_pg[-1])
# Print diagnostic summary of the computed outputs.
print('PPO initial/final exact return:', history_ppo[0], history_ppo[-1])
# Compare a derivative of exact expectation with central differences.
z = jnp.array([-.2, .3, .7]); eps = 1e-3
# Initialize array `d` with explicit values and shape.
d = jnp.array([0., 1., 0.])
# Evaluate `fd` from the current inputs and state.
fd = (exact_return(z+eps*d)-exact_return(z-eps*d))/(2*eps)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(fd, jax.grad(exact_return)(z)[1], atol=1e-4)
```

Expected: The reference seed improves exact expected return from $0.466875$ to approximately $0.9522$ with REINFORCE and $0.9758$ with the clipped updates after $40$ rollout batches.

## Two sampled training procedures, one exact evaluator

**Predict:** Must the exact return rise at every update?

![Two sampled training procedures, one exact evaluator](../../phases/12-rl/03-policy-gradients-and-ppo/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis counts freshly collected rollout batches; zero is the untrained policy. Each later point is the exact expected finite-horizon return of the updated policy. Both lines begin at $0.466875$. In this seed the REINFORCE line ends near $0.9522$, while the clipped-update line ends near $0.9758$. The curves measure return, not the optimization surrogate.

### Connect it to the computation

The clipped method takes three gradient steps per collection, while REINFORCE takes one. This is therefore not a compute-matched algorithm comparison. The always-right ceiling in this environment is $0.985$; approaching it shows that the policy has learned this corridor task. Small non-monotonic changes are compatible with stochastic updates, and one seed cannot establish which method is better on other tasks.

```python
# Compute figure data for: Two sampled training procedures, one exact evaluator
# Combine or mask array elements to form `visual_data`.
visual_data = {'kind':'line','x':list(range(41)),'xlabel':'fresh rollout batches','ylabel':'exact expected episode return','series':[{'label':'REINFORCE: 1 epoch/batch','y':history_pg.tolist()},{'label':'clipped updates: 3 epochs/batch','y':history_ppo.tolist()}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:05:11.711219+00:00. JAX 0.9.2.

```text
REINFORCE initial/final exact return: 0.46687505 0.9521624
PPO initial/final exact return: 0.46687505 0.97578394
REINFORCE initial/final exact return: 0.46687505 0.9521624
PPO initial/final exact return: 0.46687505 0.97578394
positive / negative targets: [0.6 1.  1.2] [-0.8 -1.  -1.4]
sampled / exact gradient: [0.12067959 0.21948992 0.13326262] [0.12001554 0.21822323 0.13327113]
changed seed final exact return: 0.9501661
PASS: rl-03

```

## Check clipping on both signs

**Predict before running:** For which ratios does the clipped objective become flat?

```python
# Experiment — Check clipping on both signs: The minimum prevents an update from gaining unlimited credit for...
# Initialize array `ratio` with explicit values and shape.
ratio = jnp.array([.6,1.,1.4])
# Reduce across the target axis to summarize `positive`.
positive = jnp.minimum(ratio,jnp.clip(ratio,.8,1.2))
# Reduce across the target axis to summarize `negative`.
negative = jnp.minimum(-ratio,-jnp.clip(ratio,.8,1.2))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(positive,jnp.array([.6,1.,1.2]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(negative,jnp.array([-.8,-1.,-1.4]))
# Print the observed values to compare against the expected result.
print('positive / negative targets:', positive, negative)
```

**Expected:** Positive and negative targets flatten on opposite sides.

The minimum prevents an update from gaining unlimited credit for pushing a sampled action in an already beneficial direction.

## Compare sampled and exact gradients

**Predict before running:** Will a large batch reproduce the exact gradient perfectly?

```python
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
```

**Expected:** The gradients agree within the declared sampling tolerance, not bit for bit.

The large batch reduces Monte Carlo noise. The exact evaluator and the sampled score-function estimator compute the same objective by different paths.

## Make it yours

Run REINFORCE with a different training seed and retain the initial and final exact return. Verify that the final policy improves on the uniform baseline without requiring every intermediate point to improve.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `train(...)` — Call `train` with your updated parameters or inputs from this lesson's workspace.
- `jnp.array_equal(...)` — Call `jnp.array_equal` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Verify contract: `other_history[-1] > other_history[0] + 0.35`.
2. Verify that the output satisfies the expected shape, finite-value, or numerical contract.
3. Verify that the output satisfies the expected shape, finite-value, or numerical contract.
4. Print the observed values to compare against the expected result.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Run REINFORCE with a different training seed and retain the initial...
other_theta, other_history = train(...)  # TODO: compute other_theta, other_history
# Verify contract: `other_history[-1] > other_history[0] + 0.35`.
assert other_history[-1]  # TODO: complete assertion check
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert other_history[-1]  # TODO: complete assertion check
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not jnp.array_equal(other_theta,theta_pg)  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print('changed seed final exact return:', other_history[-1])
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Run REINFORCE with a different training seed and retain the initial...
other_theta, other_history = train(seed=11,method='reinforce',epochs=1)
# Verify contract: `other_history[-1] > other_history[0] + 0.35`.
assert other_history[-1] > other_history[0] + .35
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert other_history[-1] > .9
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not jnp.array_equal(other_theta,theta_pg)
# Print the observed values to compare against the expected result.
print('changed seed final exact return:', other_history[-1])
```

</details>

## Inspect a frozen behavior reference

**Transfer / diagnosis**

Collect at zero logits, then increase all logits to $1$. Verify that stored old log probabilities do not change and that live-action ratios include values on both sides of $1$.

<details><summary>Hint</summary>

Compute new probabilities from changed parameters, leaving the rollout dictionary untouched.

</details>

### How to write: Inspect a frozen behavior reference — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.zeros / jnp.ones(shape, dtype=...)` — Allocates a tensor of the given `shape` initialized with constants.
- `jax.grad(loss_fn)(params, ...)` — Transforms a scalar-output function into a function returning the gradient PyTree with the same structure as `params`.
- `jax.random.PRNGKey(seed) & jax.random.split(key)` — Manages explicit, stateless PRNG keys—always split a key before passing subkeys into independent random draws.

**Step-by-step implementation plan:**
1. Create or split explicit PRNG key(s) (`(_, b)`) for reproducible randomness.
2. Evaluate `old` from the current inputs and state.
3. Initialize array `newlog` with explicit values and shape.
4. Run `jnp.exp` to compute `ratios`.
5. Verify contract: `jnp.array_equal(old, b['old_logp'])`.

**Starter code scaffold (fill in the TODOs):**

```python
# Inspect a frozen behavior reference (Transfer / diagnosis): The candidate changes while the behavior policy remains the...
# Create or split explicit PRNG key(s) (`(_, b)`) for reproducible randomness.
_, b = rollout(...)  # TODO: compute _, b
# Evaluate `old` from the current inputs and state.
old = ...  # TODO: compute old
# Initialize array `newlog` with explicit values and shape.
newlog = log_probability(...)  # TODO: compute newlog
# Run `jnp.exp` to compute `ratios`.
ratios = jnp.exp(...)  # TODO: compute ratios
# Verify contract: `jnp.array_equal(old, b['old_logp'])`.
assert jnp.array_equal(old,b['old_logp'])  # TODO: complete assertion check
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert float(jnp.min(ratios))  # TODO: complete assertion check
# Differentiate the objective to obtain `old_grad` via automatic differentiation.
old_grad = jax.grad(...)  # TODO: compute old_grad
# Verify contract: `jnp.all(old_grad == 0)`.
assert jnp.all(old_grad  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Inspect a frozen behavior reference (Transfer / diagnosis): The candidate changes while the behavior policy remains the...
# Create or split explicit PRNG key(s) (`(_, b)`) for reproducible randomness.
_, b = rollout(jnp.zeros(3),jax.random.key(77),128,8)
# Evaluate `old` from the current inputs and state.
old = b['old_logp'].copy()
# Initialize array `newlog` with explicit values and shape.
newlog = log_probability(jnp.ones(3),b['observation'],b['action'])
# Run `jnp.exp` to compute `ratios`.
ratios = jnp.exp(newlog-old)
# Verify contract: `jnp.array_equal(old, b['old_logp'])`.
assert jnp.array_equal(old,b['old_logp'])
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert float(jnp.min(ratios)) < 1 < float(jnp.max(ratios))
# Differentiate the objective to obtain `old_grad` via automatic differentiation.
old_grad = jax.grad(lambda lp: policy_loss(jnp.ones(3),dict(b,old_logp=lp)))(old)
# Verify contract: `jnp.all(old_grad == 0)`.
assert jnp.all(old_grad == 0)
```

The candidate changes while the behavior policy remains the fixed denominator. A zero derivative through old probabilities enforces that distinction.

</details>

## Change the time budget

**Transfer / diagnosis**

For horizon $2$, compute the always-right expected return before running code. Only one initial position can reach the goal.

<details><summary>Hint</summary>

From position $0$, two moves earn $-0.02$. From position $1$, they earn $0.99$.

</details>

### How to write: Change the time budget — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.zeros / jnp.ones(shape, dtype=...)` — Allocates a tensor of the given `shape` initialized with constants.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Initialize array `short` with explicit values and shape.
2. Verify that the numerical values match the expected reference within tolerance.
3. Verify that the output satisfies the expected shape, finite-value, or numerical contract.

**Starter code scaffold (fill in the TODOs):**

```python
# Change the time budget (Transfer / diagnosis): Changing the horizon changes the task itself.
# Initialize array `short` with explicit values and shape.
short = exact_return(...)  # TODO: compute short
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(short,(-.02+.99)/2,atol=1e-6)  # TODO: complete assertion check
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert short  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Change the time budget (Transfer / diagnosis): Changing the horizon changes the task itself.
# Initialize array `short` with explicit values and shape.
short = exact_return(jnp.full(3,100.),horizon=2)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(short,(-.02+.99)/2,atol=1e-6)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert short < exact_return(jnp.full(3,100.),horizon=8)
```

Changing the horizon changes the task itself. A lower return need not indicate a broken gradient.

</details>

## Check your understanding

During multiple PPO epochs on one rollout batch, what must remain fixed?

1. Only the current candidate policy
2. Only the clipping threshold; old probabilities can be recomputed from the candidate
3. The collected actions, return targets and behavior-policy log probabilities

<details><summary>Answer and explanation</summary>

The collected actions, return targets and behavior-policy log probabilities

The probability ratio compares the changing candidate with the policy that actually generated the batch. Replacing its denominator destroys that comparison.

</details>

## Diagnose the result

If loss falls while exact return fails to improve, first inspect the sign of the loss, the active mask and the frozen behavior probabilities. A ratio stuck at one on every epoch suggests the old policy was overwritten. If returns improve only on collection samples, evaluate independently before changing the optimizer.

## Carry forward

- A complete policy update alternates new interaction data with optimization.
- Clipping modifies an objective; it does not prove monotonic improvement or constrain every policy probability.

## Keep your evidence

Keep the exact small-state policy-gradient comparison, actual training curves and frozen PPO reference quantities. Diagnose changed clipping signs and distinguish Monte Carlo variation from an incorrect objective.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX: explicit pseudorandom keys](https://docs.jax.dev/en/latest/random-numbers.html)
- [JAX scan: fixed-shape carry](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
- [Schulman et al.: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Gymnasium: termination and truncation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

