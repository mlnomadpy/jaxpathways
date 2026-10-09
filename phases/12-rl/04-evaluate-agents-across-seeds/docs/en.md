# Evaluate agents across seeds

Phase 12: Reinforcement learning · about 90 minutes · CPU

## What you will be able to do

- Separate training seeds, evaluation streams and individual episodes.
- Report per-agent means and variability with an explicit aggregation unit.
- Compare a trained policy to uniform and known deterministic baselines without mutating it.

## The problem

A single training run looks excellent. Would another random seed learn the same behavior, and did the evaluation quietly reuse the training data? We will separate randomness used to fit an agent from randomness used to test it, retain per-agent results and compare with policies whose behavior we already understand.

## The idea

A policy's result can vary because training produced a different policy and because evaluation sampled different episodes. A useful comparison records both sources instead of pooling every episode as though it came from one independent training run.

## Separate training randomness from evaluation randomness

Train several agents with distinct training seeds. Evaluate each on a declared set of held-out conditions. First summarize within-agent episode variability; then compare the results across independently trained agents.

When methods share evaluation conditions, paired comparisons can make differences easier to interpret. Preserve that pairing rather than mixing episode identities. A large number of episodes from one trained agent does not replace multiple training seeds.

The mean-return bars summarize the fixture. Inspect individual seed results before trusting a small average difference. If you add an uncertainty interval, state its unit of resampling and method rather than attaching an unexplained error bar.

### Pause and reason

Do a thousand episodes from one trained policy establish training-seed robustness?

<details><summary>Compare your reasoning</summary>

No. They improve knowledge of that policy's evaluation behavior. Training variability needs independently trained policies.

</details>

## Write the protocol before inspecting results

Fix the environment, horizon, training budget and evaluation procedure before seeing the held-out metrics. We use training seeds $(0,7,23,41,59)$ and evaluation seeds $(1000,1001,1002,1003)$. These seed identifiers are part of this protocol, not proof of statistical independence by themselves. The generator receives separate keys for fitting and testing. Do not change the learning rate after inspecting final test results and still call the same results held out.

## Name the experimental unit

Within an agent, averaging $2048$ evaluation episodes reduces uncertainty about that particular policy. Across five independently trained agents, differences reveal sensitivity to training randomness. A large episode count does not make the number of trained agents large. We retain all episode returns and stream means, then summarize the five agent means with their sample standard deviation. That standard deviation describes variability; it is not a confidence interval.

$$
\bar R_i=\frac1E\sum_{e=1}^E R_{i,e},\qquad s=\sqrt{\frac1{K-1}\sum_{i=1}^K(\bar R_i-\bar R)^2}
$$

## Pair the comparison deliberately

Using the same evaluation seed set across agents and baselines keeps the reset draws aligned and may reduce comparison noise. Actions still depend on the policy, so trajectories diverge. Shared evaluation randomness means the agent estimates are not wholly independent measurement errors. Preserve paired differences when comparing methods; do not pool every episode and pretend there were thousands of independent training runs. A broader study would use more training seeds, predeclared metrics and uncertainty estimates suited to its dependence structure.

## A baseline can catch a broken evaluator

The uniform policy has exact expected return $0.466875$. Always-left earns $-0.08$, because it never reaches the goal and receives eight costs. Always-right earns $0.985$, averaged over the two starts. If the sampled evaluator places these in the wrong order, inspect masks, reward sums and horizon before celebrating the learned policy. Test a changed horizon too: it should change the known returns in predictable ways.

## Evaluation must leave training state alone

The policy array is passed to the evaluator as a value. We copy it before evaluation and compare afterward. Evaluation creates its own keys and never calls an optimizer. In a neural model this contract also includes running statistics, dropout mode and any other mutable model state. The tabular example cannot certify that a future network implementation handles those correctly.

## Keep a report another learner can challenge

Retain source revision, package versions, policy parameters, train/evaluation seeds, horizon, step budget, raw episode returns and per-agent means. Add an exact-versus-sampled comparison, a failure diagnosis, and the limits: one small deterministic-transition environment, a randomized start distribution, stationary tabular policies, CPU execution and no hyperparameter search. High return here demonstrates a correct teaching harness, not general control capability.

## Assemble the verified learning system

Create main.py with this block. Run python3 main.py in the course CPU environment.

```python
# Step 1: Assemble the verified learning system
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

Use the same training protocol for every seed. Changing budgets between agents would confound the comparison.

## Add a read-only evaluator

Append this block to main.py. Run python3 main.py in the course CPU environment.

```python
# Step 2 — Add a read-only evaluator: The evaluator retains raw returns, reports its seed set and...
def evaluate(theta, seeds, batch_size=512, horizon=8):
    """One mean per evaluation stream; preserve raw episode returns too."""
    # Create device-backed JAX array `theta`.
    theta = jnp.asarray(theta)
    # Guard input contract (`theta.shape != (3,) or not bool(jnp.all(jnp.isfinite(theta)))`) and fail fast if violated.
    if theta.shape != (3,) or not bool(jnp.all(jnp.isfinite(theta))):
        raise ValueError('theta must contain three finite logits')
    # Evaluate `seeds` and convert the result into Python scalar/collection `seeds`.
    seeds = tuple(seeds)
    # Guard input contract (`len(seeds) < 2 or len(set(seeds)) != len(seeds) or batch_size < 1 or (horizon < 1)`) and fail fast if violated.
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or batch_size < 1 or horizon < 1:
        raise ValueError('provide at least two distinct evaluation seeds and positive sizes')
    # Create or split explicit PRNG key(s) (`returns`) for reproducible randomness.
    returns = np.stack([np.asarray(rollout(theta, jax.random.key(s), batch_size, horizon)[1]['reward'].sum(0)) for s in seeds])
    # Return `{'seeds': list(seeds), 'episode_returns': returns, 'stream_means': returns.mean(1), 'mean': float(returns.mean()), 'exact': float(exact_return(theta, horizon))}` to the caller.
    return {'seeds': list(seeds), 'episode_returns': returns, 'stream_means': returns.mean(1),
            'mean': float(returns.mean()), 'exact': float(exact_return(theta, horizon))}
```

The evaluator retains raw returns, reports its seed set and rejects malformed policies or repeated seed identifiers.

## Train several agents and retain every result

Append this block to main.py. Run python3 main.py in the course CPU environment.

```python
# Step 3 — Train several agents and retain every result: Each agent receives a separate training key; all use the declared...
training_seeds = [0,7,23,41,59]
# Evaluate `evaluation_seeds` from the current inputs and state.
evaluation_seeds = [1000,1001,1002,1003]
# Evaluate `(policies, histories, reports)` from the current inputs and state.
policies, histories, reports = [], [], []
# Iterate over `seed` to step through the computation:
for seed in training_seeds:
    # Run `train` to compute `(policy, history)`.
    policy, history = train(seed=seed)
    # Append the current step result to `policies`.
    # Append the current step result to `policies`.
    policies.append(policy); histories.append(history)
    # Append the current step result to `reports`.
    reports.append(evaluate(policy,evaluation_seeds))
# Initialize array `trained_means` with explicit values and shape.
trained_means = np.array([r['mean'] for r in reports])
# Initialize array `exact_means` with explicit values and shape.
exact_means = np.array([r['exact'] for r in reports])
# Initialize array `baseline` with explicit values and shape.
baseline = evaluate(jnp.zeros(3),evaluation_seeds)
# Verify contract: `np.all(trained_means > baseline['mean'] + 0.4)`.
assert np.all(trained_means > baseline['mean'] + .4)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert np.all(np.abs(trained_means-exact_means) < .03)
# Print the observed values to compare against the expected result.
print('training seeds:', training_seeds)
# Print diagnostic summary of the computed outputs.
print('held-out mean per agent:', trained_means)
# Print diagnostic summary of the computed outputs.
print('exact mean per agent:', exact_means)
# Print diagnostic summary of the computed outputs.
print('uniform baseline mean:', baseline['mean'])
# Print diagnostic summary of the computed outputs.
print('across-agent mean / sample sd:', trained_means.mean(),trained_means.std(ddof=1))
```

Each agent receives a separate training key; all use the declared held-out evaluation streams.

## Run the example

```python
# Complete runnable example (rl-04)
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

# Step 2 — Add a read-only evaluator: The evaluator retains raw returns, reports its seed set and...
def evaluate(theta, seeds, batch_size=512, horizon=8):
    """One mean per evaluation stream; preserve raw episode returns too."""
    # Create device-backed JAX array `theta`.
    theta = jnp.asarray(theta)
    # Guard input contract (`theta.shape != (3,) or not bool(jnp.all(jnp.isfinite(theta)))`) and fail fast if violated.
    if theta.shape != (3,) or not bool(jnp.all(jnp.isfinite(theta))):
        raise ValueError('theta must contain three finite logits')
    # Evaluate `seeds` and convert the result into Python scalar/collection `seeds`.
    seeds = tuple(seeds)
    # Guard input contract (`len(seeds) < 2 or len(set(seeds)) != len(seeds) or batch_size < 1 or (horizon < 1)`) and fail fast if violated.
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or batch_size < 1 or horizon < 1:
        raise ValueError('provide at least two distinct evaluation seeds and positive sizes')
    # Create or split explicit PRNG key(s) (`returns`) for reproducible randomness.
    returns = np.stack([np.asarray(rollout(theta, jax.random.key(s), batch_size, horizon)[1]['reward'].sum(0)) for s in seeds])
    # Return `{'seeds': list(seeds), 'episode_returns': returns, 'stream_means': returns.mean(1), 'mean': float(returns.mean()), 'exact': float(exact_return(theta, horizon))}` to the caller.
    return {'seeds': list(seeds), 'episode_returns': returns, 'stream_means': returns.mean(1),
            'mean': float(returns.mean()), 'exact': float(exact_return(theta, horizon))}

# Step 3 — Train several agents and retain every result: Each agent receives a separate training key; all use the declared...
training_seeds = [0,7,23,41,59]
# Evaluate `evaluation_seeds` from the current inputs and state.
evaluation_seeds = [1000,1001,1002,1003]
# Evaluate `(policies, histories, reports)` from the current inputs and state.
policies, histories, reports = [], [], []
# Iterate over `seed` to step through the computation:
for seed in training_seeds:
    # Run `train` to compute `(policy, history)`.
    policy, history = train(seed=seed)
    # Append the current step result to `policies`.
    # Append the current step result to `policies`.
    policies.append(policy); histories.append(history)
    # Append the current step result to `reports`.
    reports.append(evaluate(policy,evaluation_seeds))
# Initialize array `trained_means` with explicit values and shape.
trained_means = np.array([r['mean'] for r in reports])
# Initialize array `exact_means` with explicit values and shape.
exact_means = np.array([r['exact'] for r in reports])
# Initialize array `baseline` with explicit values and shape.
baseline = evaluate(jnp.zeros(3),evaluation_seeds)
# Verify contract: `np.all(trained_means > baseline['mean'] + 0.4)`.
assert np.all(trained_means > baseline['mean'] + .4)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert np.all(np.abs(trained_means-exact_means) < .03)
# Print the observed values to compare against the expected result.
print('training seeds:', training_seeds)
# Print diagnostic summary of the computed outputs.
print('held-out mean per agent:', trained_means)
# Print diagnostic summary of the computed outputs.
print('exact mean per agent:', exact_means)
# Print diagnostic summary of the computed outputs.
print('uniform baseline mean:', baseline['mean'])
# Print diagnostic summary of the computed outputs.
print('across-agent mean / sample sd:', trained_means.mean(),trained_means.std(ddof=1))
```

Expected: All five independently trained agents outperform the held-out uniform baseline. The printed per-agent values preserve variability instead of hiding it in one pooled number.

## Held-out returns for independently trained agents

**Predict:** Will measured and exact means coincide for every agent?

![Held-out returns for independently trained agents](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal categories identify the five training seeds. Each group compares sampled held-out episode return with that policy’s exact expectation. The third bar is the uniform-policy baseline, repeated as a shared reference. Taller bars mean better return in this fixed corridor task; they do not mean higher accuracy or greater statistical confidence. Seed $41$ has the lowest sampled mean, about $0.9694$, and exact mean, about $0.9711$. Seed $0$ is around $0.9760$ sampled and $0.9758$ exact. The visible spread is small compared with the roughly $0.51$ improvement over the uniform baseline.

### Connect it to the computation

The trained policies cluster close to the always-right limit of $0.985$, while the uniform baseline remains around $0.47$. Small gaps between sampled and exact bars are evaluation noise. Differences among exact bars reflect differences in the trained policies. The plot has no confidence intervals: five training runs and shared evaluation streams support a descriptive comparison here, not a broad claim about PPO.

```python
# Compute figure data for: Held-out returns for independently trained agents
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind':'bar','labels':[str(s) for s in training_seeds],'xlabel':'training seed','ylabel':'mean episode return','series':[{'label':'held-out sampled','y':trained_means.tolist()},{'label':'exact expectation','y':exact_means.tolist()},{'label':'uniform baseline (shared)','y':[baseline['mean']]*5}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:05:15.597026+00:00. JAX 0.9.2.

```text
training seeds: [0, 7, 23, 41, 59]
held-out mean per agent: [0.9759717  0.97472161 0.97537106 0.96944821 0.97455072]
exact mean per agent: [0.97578394 0.9749347  0.97562152 0.97113729 0.97404611]
uniform baseline mean: 0.46628910303115845
across-agent mean / sample sd: 0.9740126609802247 0.0026129212275767476
training seeds: [0, 7, 23, 41, 59]
held-out mean per agent: [0.9759717  0.97472161 0.97537106 0.96944821 0.97455072]
exact mean per agent: [0.97578394 0.9749347  0.97562152 0.97113729 0.97404611]
uniform baseline mean: 0.46628910303115845
across-agent mean / sample sd: 0.9740126609802247 0.0026129212275767476
left / random / right: -0.07999999821186066 0.46628910303115845 0.9850389361381531
paired improvements: [0.5096826  0.50843251 0.50908196 0.50315911 0.50826162]
mean improvement across trained agents: 0.5077235579490662
sample sd / independence-assuming SE: 0.09999999999999998 0.05773502691896257
PASS: rl-04

```

## Check baselines with known returns

**Predict before running:** Which policy should have the highest mean, and by how much?

```python
# Experiment — Check baselines with known returns: The baseline check can fail an evaluator even when the learned...
# Initialize array `left` with explicit values and shape.
left = evaluate(jnp.full(3,-100.),evaluation_seeds)
# Initialize array `right` with explicit values and shape.
right = evaluate(jnp.full(3,100.),evaluation_seeds)
# Verify contract: `abs(left['mean'] + 0.08) < 1e-06`.
assert abs(left['mean']+.08) < 1e-6
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert abs(right['mean']-.985) < .002
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert left['mean'] < baseline['mean'] < right['mean']
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert abs(baseline['mean']-.466875) < .04
# Print the observed values to compare against the expected result.
print('left / random / right:', left['mean'], baseline['mean'], right['mean'])
```

**Expected:** The means follow the known order and agree with the exact values within the declared sampling tolerances.

The baseline check can fail an evaluator even when the learned policy appears successful.

## Prove isolation and replay

**Predict before running:** Should evaluating a policy twice change either its parameters or the report?

```python
# Experiment — Prove isolation and replay: Repeatability is useful for debugging.
# Initialize array `before` with explicit values and shape.
before = np.array(policies[0],copy=True)
# Run `evaluate` to compute `a`.
a = evaluate(policies[0],evaluation_seeds)
# Run `evaluate` to compute `b`.
b = evaluate(policies[0],evaluation_seeds)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_array_equal(before,policies[0])
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_array_equal(a['episode_returns'],b['episode_returns'])
# Run `evaluate` to compute `c`.
c = evaluate(policies[0],[2000,2001,2002,2003])
# Verify contract: `not np.array_equal(a['episode_returns'], c['episode_returns'])`.
assert not np.array_equal(a['episode_returns'],c['episode_returns'])
```

**Expected:** The same stream reproduces the report; new streams change sampled episodes while preserving the policy.

Repeatability is useful for debugging. It does not convert a reused test stream into fresh evidence.

## Make it yours

Compute the paired held-out improvement over the uniform baseline for each trained agent. Report all $5$ differences and their mean, keeping the training run as the unit.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `x.sum(axis=...) / x.mean(axis=..., keepdims=...)` — Reduces values along the named `axis` (the axis you name is collapsed unless `keepdims=True`).

**Step-by-step implementation plan:**
1. Verify that the output tensor shape matches our prediction.
2. Verify that the output satisfies the expected shape, finite-value, or numerical contract.
3. Print the observed values to compare against the expected result.
4. Print diagnostic summary of the computed outputs.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Compute the paired held-out improvement over the uniform baseline for...
deltas = ...  # TODO: compute deltas
# Verify that the output tensor shape matches our prediction.
assert deltas.shape  # TODO: complete assertion check
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert np.all(deltas  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print('paired improvements:', deltas)
# Print diagnostic summary of the computed outputs.
print('mean improvement across trained agents:', deltas.mean())
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Compute the paired held-out improvement over the uniform baseline for...
deltas = trained_means - baseline['mean']
# Verify that the output tensor shape matches our prediction.
assert deltas.shape == (5,)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert np.all(deltas > .4)
# Print the observed values to compare against the expected result.
print('paired improvements:', deltas)
# Print diagnostic summary of the computed outputs.
print('mean improvement across trained agents:', deltas.mean())
```

</details>

## Expose an unequal-batch averaging bug

**Transfer / diagnosis**

One evaluation shard contains one episode with return $1$; another contains nine episodes with return $0$. Compare the unweighted mean of shard means with the true episode mean.

<details><summary>Hint</summary>

Weight each shard mean by its episode count.

</details>

### How to write: Expose an unequal-batch averaging bug — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `x.sum(axis=...) / x.mean(axis=..., keepdims=...)` — Reduces values along the named `axis` (the axis you name is collapsed unless `keepdims=True`).

**Step-by-step implementation plan:**
1. Initialize array `means` with explicit values and shape.
2. Aggregate array values to compute `wrong`.
3. Verify contract: `wrong == 0.5 and right == 0.1`.

**Starter code scaffold (fill in the TODOs):**

```python
# Expose an unequal-batch averaging bug (Transfer / diagnosis): Equal shard weighting changes the unit from episode to shard.
# Initialize array `means` with explicit values and shape.
means = np.array(...)  # TODO: compute means
# Aggregate array values to compute `wrong`.
wrong = means.mean(...)  # TODO: compute wrong
# Verify contract: `wrong == 0.5 and right == 0.1`.
assert wrong  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Expose an unequal-batch averaging bug (Transfer / diagnosis): Equal shard weighting changes the unit from episode to shard.
# Initialize array `means` with explicit values and shape.
means = np.array([1.,0.]); counts = np.array([1,9])
# Aggregate array values to compute `wrong`.
wrong = means.mean(); right = np.average(means,weights=counts)
# Verify contract: `wrong == 0.5 and right == 0.1`.
assert wrong == .5 and right == .1
```

Equal shard weighting changes the unit from episode to shard. Our equal-size streams avoid the discrepancy, but exported raw episode counts make the contract explicit.

</details>

## Distinguish spread from standard error

**Transfer / diagnosis**

For agent means $(0.8,0.9,1.0)$, compute sample standard deviation and the naive standard error assuming independent agent measurements. Explain why neither automatically describes our shared-stream protocol.

<details><summary>Hint</summary>

Use sample standard deviation with ddof=1; a standard error divides by the square root of the number of independent units.

</details>

### How to write: Distinguish spread from standard error — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `error(...)` — Call `error` with your updated parameters or inputs from this lesson's workspace.
- `np.array(...)` — Call `np.array` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Initialize array `values` with explicit values and shape.
2. Aggregate array values to compute `sd`.
3. Evaluate `naive_se` from the current inputs and state.
4. Verify contract: `abs(sd - 0.1) < 1e-12`.
5. Verify that the output satisfies the expected shape, finite-value, or numerical contract.

**Starter code scaffold (fill in the TODOs):**

```python
# Distinguish spread from standard error (Transfer / diagnosis): The numerical formula does not establish its sampling...
# Initialize array `values` with explicit values and shape.
values = np.array(...)  # TODO: compute values
# Aggregate array values to compute `sd`.
sd = values.std(...)  # TODO: compute sd
# Evaluate `naive_se` from the current inputs and state.
naive_se = ...  # TODO: compute naive_se
# Verify contract: `abs(sd - 0.1) < 1e-12`.
assert abs(sd-.1)  # TODO: complete assertion check
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert abs(naive_se-.0577350269)  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print('sample sd / independence-assuming SE:',sd,naive_se)
```

<details><summary>Reference solution and reasoning</summary>

```python
# Distinguish spread from standard error (Transfer / diagnosis): The numerical formula does not establish its sampling...
# Initialize array `values` with explicit values and shape.
values = np.array([.8,.9,1.])
# Aggregate array values to compute `sd`.
sd = values.std(ddof=1)
# Evaluate `naive_se` from the current inputs and state.
naive_se = sd/np.sqrt(3)
# Verify contract: `abs(sd - 0.1) < 1e-12`.
assert abs(sd-.1) < 1e-12
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert abs(naive_se-.0577350269) < 1e-9
# Print the observed values to compare against the expected result.
print('sample sd / independence-assuming SE:',sd,naive_se)
```

The numerical formula does not establish its sampling assumptions. Shared evaluation streams and only five training runs warrant a descriptive report, not a confident general claim.

</details>

## Check your understanding

What does evaluating one policy on many episodes fail to measure?

1. Its mean return on those sampled episodes
2. Sensitivity to training randomness across independently fitted policies
3. Variation in returns between the sampled episodes

<details><summary>Answer and explanation</summary>

Sensitivity to training randomness across independently fitted policies

More evaluation episodes estimate one policy more precisely. To study training variability, repeat training with different seeds and preserve those runs as separate units.

</details>

## Diagnose the result

If a result changes after evaluation, compare model state and random-key ownership before inspecting the optimizer. If a reported uncertainty is implausibly tiny, check whether episodes were incorrectly counted as independent training runs. If a baseline is unexpectedly strong, recompute its known return under the exact horizon and reward contract.

## Carry forward

- Separate randomness for fitting from randomness for evaluation.
- Report per-agent results, known baselines and explicit limits before making general claims.

## Keep your evidence

Keep multiple trained seeds, fixed held-out streams, exact versus sampled returns and failed trajectories. Explain uncertainty and demonstrate evaluation leaves the policy unchanged.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX: explicit pseudorandom keys](https://docs.jax.dev/en/latest/random-numbers.html)
- [JAX scan: fixed-shape carry](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
- [Schulman et al.: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Gymnasium: termination and truncation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

