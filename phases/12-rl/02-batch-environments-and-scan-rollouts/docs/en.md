# Batch environments and scan rollouts

Phase 12: Reinforcement learning · about 85 minutes · CPU

## What you will be able to do

- Build a scan over vmapped transitions with explicit key ownership.
- Reconstruct episode returns and reward-to-go from padded trajectories.
- Verify the live-transition mask and distinguish batch dimensions from time dimensions.

## The problem

One episode is easy to inspect, but an agent needs many experiences. How do we collect a batch without mixing time, environments or random streams? We will turn the previous transition into a fixed-shape rollout and prove that its padded rows cannot become training evidence.

## The idea

Batched rollouts put independent environments on one axis and time on another. Some environments finish earlier, so masks describe which transitions remain meaningful while arrays keep fixed shapes.

## Keep the array shape fixed while activity changes

Draw one environment per row and time across columns. Mark the transition that terminates an episode, then mark later padding separately. The terminal transition can still earn reward; masking it out too early loses part of the episode.

Carry the next state and activity information explicitly. If the implementation resets finished environments, distinguish that contract from absorbing padding instead of mixing them in one explanation.

The active-count curve decreases while the tensor shape stays fixed. It measures active environments, not shrinking allocation. Verify one environment independently, then permute the batch to check that unrelated trajectories stay independent.

### A terminal transition still counts

**Predict:** Should every cell after the first done flag be treated identically to the transition that caused done?

![A terminal transition still counts](../outputs/mechanism.svg)

*Conceptual / analytic teaching diagram; not a recorded benchmark.*

Each row is one environment. A terminating transition can still earn reward; later cells are padding under this absorbing-state contract. Tensor length stays fixed while activity changes. This is a conceptual rollout example, not an auto-reset implementation.

### Pause and reason

Should every cell after the first done flag be treated identically to the transition that caused done?

<details><summary>Compare your reasoning</summary>

No. The terminating transition can contain valid reward and learning information. Later padding follows the declared terminal policy and should not invent extra transitions.

</details>

## Give each axis one meaning

For $T=8$ and $B=64$, the reward array has shape $(8,64)$. Summing axis zero gives one return per episode. Summing axis one gives total reward at each time index across the batch, a different statistic. vmap maps reset and step over environments; scan threads the complete batch of states through time. The carry has the same structure, shapes and dtypes at every iteration.

## Split randomness where ownership changes

First split a reset key from an action key. Split reset keys across the $B$ environments. Split action keys across the $T$ time steps, then across environments. A scalar reset key broadcast to all environments would create correlated starts. An action key reused at every step would replay draws. Deterministic transitions do not need an extra key, so their signature stays simple.

## Store the behavior policy before changing it

Each position has a policy logit $\theta_s$. Applying the sigmoid gives the probability of moving right; the left probability is its complement. A sampled action is discrete, so we will not differentiate through action sampling. Instead, we store its log probability under the behavior policy. Stable log-sigmoid avoids taking a logarithm of a probability rounded to zero.

$$
\pi_\theta(1\mid s)=\sigma(\theta_s),\qquad \log\pi_\theta(a\mid s)=a\log\sigma(\theta_s)+(1-a)\log\sigma(-\theta_s)
$$

## A final transition is still a learning example

The active mask is computed from the state before step. Therefore the action that reaches the goal is active and retains its reward. Only subsequent rows are padding. Masking with the next state would delete the successful action itself. We store the final observation in the returned state and do not replace it with a fresh start. This collector handles one finite episode per environment, not an endless auto-reset stream.

## Credit each action with rewards that follow it

Reward-to-go sums rewards from the current time onward. For rewards $(-0.01,-0.01,1,0)$, targets are $(0.98,0.99,1,0)$. The first action can influence all later rewards; the last successful action is credited with the goal reward only. We reverse the time axis, take a cumulative sum, and reverse it back. There is no value bootstrap here because these are complete finite-horizon episodes.

$$
G_{t,b}=\sum_{u=t}^{T-1}r_{u,b}
$$

## Reuse the verified environment

Create main.py with this block. Run python3 main.py in the course CPU environment.

```python
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
```

Keep the state and ending contract identical to the previous lesson.

## Collect a batch and compute targets

Append this block to main.py. Run python3 main.py in the course CPU environment.

```python
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
```

scan emits time-major arrays; vmap keeps environments independent within each time step.

## Check shapes and episode boundaries

Append this block to main.py. Run python3 main.py in the course CPU environment.

```python
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
```

Exactly one ending event and zero padded rewards are stronger checks than a plausible mean return.

## Run the example

```python
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
```

Expected: The trajectory has the stated time and environment axes, exactly one ending event per episode, and zero reward on every padded row.

## The live batch shrinks while the array shape stays fixed

**Predict:** Should every row contain the same number of active environments?

![The live batch shrinks while the array shape stays fixed](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis is the transition index; the vertical axis counts environments active before that transition. The first value is $64$. The count never rises because we do not reset inside a rollout. Decreases show episodes reaching the goal; episodes still active on the final row will either finish or time out on that row. In the recorded seed, the first two counts are both $64$: even the nearer starting position needs two actions to reach the goal. The third count is $58$, and the last is $29$. The unchanged first pair is therefore expected, not a stalled loop.

### Connect it to the computation

This curve is not reward or policy quality: a fast-ending policy could succeed, fail or exploit a termination bug in a different task. Here the transition tests establish that early endings are goals. The fixed $(8,64)$ reward array retains later rows, but their active flags are false. Changing the seed can move individual counts while preserving these invariants.

```python
# Compute figure data for: The live batch shrinks while the array shape stays fixed
# Reduce across the target axis to summarize `visual_data`.
visual_data = {'kind':'line','x':list(range(1,9)),'xlabel':'transition index','ylabel':'active environments before transition','series':[{'label':'uniform random policy','y':records['active'].sum(1).tolist()}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:05:04.389780+00:00. JAX 0.9.2.

```text
return mean: 0.5848437547683716
active environments by transition: [64 64 58 53 47 44 39 29]
return mean: 0.5848437547683716
active environments by transition: [64 64 58 53 47 44 39 29]
reward-to-go: [0.98 0.99 1.   0.  ]
PASS: rl-02

```

## Replay the entire collector

**Predict before running:** Does replay require restoring only the environment state?

```python
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
```

**Expected:** Every field replays for the same configuration and key; another key changes actions.

The random stream is part of the experiment state. Replay in this environment is not a promise of identical outputs across arbitrary library versions or devices.

## Reward-to-go by hand

**Predict before running:** Predict the four targets for rewards $(-0.01,-0.01,1,0)$.

```python
# Experiment — Reward-to-go by hand: Zero padded rewards preserve the finite-episode sum, while the...
# Construct `r` via `jnp.array([[-.01],[-.01],[1.],[0.]])`
r = jnp.array([[-.01],[-.01],[1.],[0.]])
# Run `returns_to_go` to compute `g`.
g = returns_to_go(r)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(g[:,0], jnp.array([.98,.99,1.,0.]))
# Print the observed values to compare against the expected result.
print('reward-to-go:', g[:,0])
```

**Expected:** The targets are $(0.98,0.99,1,0)$.

Zero padded rewards preserve the finite-episode sum, while the active mask removes meaningless log probabilities.

## Make it yours

Change the batch to $7$ environments and the horizon to $5$. Verify array shapes, one ending event per episode and zero rewards after finishing.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.zeros / jnp.ones(shape, dtype=...)` — Allocates a tensor of the given `shape` initialized with constants.
- `x.sum(axis=...) / x.mean(axis=..., keepdims=...)` — Reduces values along the named `axis` (the axis you name is collapsed unless `keepdims=True`).
- `jax.random.PRNGKey(seed) & jax.random.split(key)` — Manages explicit, stateless PRNG keys—always split a key before passing subkeys into independent random draws.

**Step-by-step implementation plan:**
1. Create or split explicit PRNG key(s) (`(final7, r7)`) for reproducible randomness.
2. Check tensor shape invariant: `r7['reward'].shape == (5`
3. Assert invariant `bool(final7.done.all())` holds
4. Assert invariant `jnp.array_equal((r7['terminated'] | r7['truncated']).sum(0)` holds
5. Assert invariant `bool(jnp.all(jnp.where(r7['active']` holds

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Change the batch to 7 environments and the horizon to 5.
# Create or split explicit PRNG key(s) (`(final7, r7)`) for reproducible randomness.
final7, r7 = rollout(...)  # TODO: compute final7, r7
# Check tensor shape invariant: `r7['reward'].shape == (5`
assert r7['reward'].shape  # TODO: complete assertion check
# Assert invariant `bool(final7.done.all())` holds
assert bool(final7.done.all())  # TODO: complete assertion check
# Assert invariant `jnp.array_equal((r7['terminated'] | r7['truncated']).sum(0)` holds
assert jnp.array_equal((r7['terminated'] | r7['truncated']).sum(0), jnp.ones(7))  # TODO: complete assertion check
# Assert invariant `bool(jnp.all(jnp.where(r7['active']` holds
assert bool(jnp.all(jnp.where(r7['active'],0.,r7['reward'])  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
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
```

</details>

## Catch a shifted mask

**Transfer / diagnosis**

Construct a two-action success from position $1$. Compare masking by the pre-transition done flag with masking by the post-transition done flag.

<details><summary>Hint</summary>

Record the mask before calling step; the successful transition still belongs to the episode.

</details>

### How to write: Catch a shifted mask — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `mask(...)` — Call `mask` with your updated parameters or inputs from this lesson's workspace.
- `State(...)` — Call `State` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Compute `correct, broken` from `0., 0.`
2. Iterate over `a` to step through the computation:
3. Compute `active` from `~s.done`
4. Run `step` to compute `(s, reward, _, _)`.
5. Accumulate the next contribution into `correct`.

**Starter code scaffold (fill in the TODOs):**

```python
# Catch a shifted mask (Transfer / diagnosis): The broken mask removes the action responsible for success,...
s = State(...)  # TODO: compute s
# Compute `correct, broken` from `0., 0.`
correct, broken = ...  # TODO: compute correct, broken
# Iterate over `a` to step through the computation:
for a in [1,1]:
    # Compute `active` from `~s.done`
    active = ...  # TODO: compute active
    # Run `step` to compute `(s, reward, _, _)`.
    s, reward, _, _ = step(...)  # TODO: compute s, reward, _, _
    # Accumulate the next contribution into `correct`.
    correct += float(reward*active)
    # Accumulate the next contribution into `broken`.
    broken += float(reward*(~s.done))
# Check numerical equivalence within tolerance: `abs(correct-.99) < 1e-6`
assert abs(correct-.99)  # TODO: complete assertion check
# Check numerical equivalence within tolerance: `abs(broken+.01) < 1e-6`
assert abs(broken+.01)  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
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
# Check numerical equivalence within tolerance: `abs(correct-.99) < 1e-6`
assert abs(correct-.99) < 1e-6
# Check numerical equivalence within tolerance: `abs(broken+.01) < 1e-6`
assert abs(broken+.01) < 1e-6
```

The broken mask removes the action responsible for success, leaving only a cost.

</details>

## Check a deterministic policy limit

**Transfer / diagnosis**

Use very large positive logits and verify that every episode follows a shortest route.

<details><summary>Hint</summary>

The two start states need different numbers of actions. Compare each return against its own initial position.

</details>

### How to write: Check a deterministic policy limit — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.zeros / jnp.ones(shape, dtype=...)` — Allocates a tensor of the given `shape` initialized with constants.
- `x.sum(axis=...) / x.mean(axis=..., keepdims=...)` — Reduces values along the named `axis` (the axis you name is collapsed unless `keepdims=True`).
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.random.PRNGKey(seed) & jax.random.split(key)` — Manages explicit, stateless PRNG keys—always split a key before passing subkeys into independent random draws.

**Step-by-step implementation plan:**
1. Create or split explicit PRNG key(s) (`(_, easy)`) for reproducible randomness.
2. Combine or mask array elements to form `expected`.
3. Verify that the numerical values match the expected reference within tolerance.
4. Assert invariant `jnp.array_equal(easy['active'].sum(0)` holds

**Starter code scaffold (fill in the TODOs):**

```python
# Check a deterministic policy limit (Transfer / diagnosis): The expected return is calculated from the initial position...
# Create or split explicit PRNG key(s) (`(_, easy)`) for reproducible randomness.
_, easy = rollout(...)  # TODO: compute _, easy
# Combine or mask array elements to form `expected`.
expected = jnp.where(...)  # TODO: compute expected
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(easy['reward'].sum(0),expected,atol=1e-6)  # TODO: complete assertion check
# Assert invariant `jnp.array_equal(easy['active'].sum(0)` holds
assert jnp.array_equal(easy['active'].sum(0),3-easy['observation'][0])  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Check a deterministic policy limit (Transfer / diagnosis): The expected return is calculated from the initial position...
# Create or split explicit PRNG key(s) (`(_, easy)`) for reproducible randomness.
_, easy = rollout(jnp.full(3,100.),jax.random.key(42),23,8)
# Combine or mask array elements to form `expected`.
expected = jnp.where(easy['observation'][0] == 0,.98,.99)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(easy['reward'].sum(0),expected,atol=1e-6)
# Assert invariant `jnp.array_equal(easy['active'].sum(0)` holds
assert jnp.array_equal(easy['active'].sum(0),3-easy['observation'][0])
```

The expected return is calculated from the initial position rather than copied from the collector.

</details>

## Check your understanding

Which axis should be summed to obtain one return per environment from a time-major reward array?

1. The time axis
2. The environment axis
3. Both axes before averaging by live transitions

<details><summary>Answer and explanation</summary>

The time axis

Summing over time leaves one scalar for each episode. Averaging these episode returns gives the finite-episode performance objective.

</details>

## Diagnose the result

If reported return changes merely by padding an already finished trajectory, inspect reward masking. If success disappears from the objective, check whether active was recorded before step. If trajectories match too often, inspect broadcast or reused keys before increasing the batch.

## Carry forward

- vmap handles independent environments; scan handles the dependent time sequence.
- Store behavior log probabilities and live-transition masks with rewards.

## Keep your evidence

Keep reconstructed trajectories, randomized keys, padding masks and independent return calculations. Show how a changed key affects interaction without changing the environment rules.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX: explicit pseudorandom keys](https://docs.jax.dev/en/latest/random-numbers.html)
- [JAX scan: fixed-shape carry](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
- [Schulman et al.: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Gymnasium: termination and truncation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

