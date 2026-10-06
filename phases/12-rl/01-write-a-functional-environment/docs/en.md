# Write a functional environment

Phase 12: Reinforcement learning · about 75 minutes · CPU

## What you will be able to do

- Implement reset and step as pure functions with explicit state.
- Distinguish termination, time-limit truncation and an absorbing finished state.
- Check transitions independently before using rewards to train a policy.

## The problem

A learner agent reaches a goal, but its reported reward keeps increasing afterward. Is the policy improving, or is the environment paying it repeatedly? Before training anything, we will build a small environment whose entire transition table can be checked. You will leave with explicit rules for starting, moving, ending and resetting an episode.

## The idea

A functional environment receives state and action and returns the next state, reward and termination information. Define exactly which transition earns reward and what later terminal padding means. This keeps rollouts and return calculations consistent.

## Reward belongs to a transition

Suppose entering the goal earns one reward. The transition into the goal counts once. Later padded steps in the terminal state should not repeatedly earn the same reward unless that is explicitly the task's intended reward rule.

Draw separate arrows for an ordinary move, a goal-reaching move and terminal padding. A timeout is another condition whose treatment must be specified; it is not automatically the same learning signal as reaching a terminal goal.

The reward trace shows one event followed by absorbing padding. Read the event's time index against the transition convention. Then test actions after termination to ensure padding does not introduce extra rewards or unintended state changes.

### Pause and reason

Why can checking only the final state miss a reward bug?

<details><summary>Compare your reasoning</summary>

A rollout can end in the correct goal while counting its reward repeatedly during padding. Check transition rewards and accumulated return as well as final state.

</details>

## State contains what the next transition needs

Our state holds position, elapsed steps and a done flag. Observation is position: enough for the stationary policy used here, but not the full finite-horizon decision state. A policy that needs to reason about time remaining should also observe elapsed time. Carrying elapsed time in environment state prevents a time limit from depending on a hidden Python counter. NamedTuple fields are JAX pytree leaves, so the same function can run directly, under jit, or under vmap.

## Walk through the reward before naming the return

Start at position $1$, then move right twice. The first action reaches position $2$ and earns $-0.01$. The second reaches the goal and earns $1$. Total reward is $0.99$. Starting at position $0$ instead needs one extra move and earns $0.98$. Return means the sum of rewards in this lesson; we do not discount future rewards. Averaging over our two equally likely starts gives the always-right policy return $0.985$.

$$
G=\sum_{t=0}^{H-1}r_t,\qquad H=8
$$

## Ending is an event; finished is a state

Terminated marks the transition that reaches the goal. Truncated marks an active transition that reaches the time limit without reaching the goal. A simultaneous goal and time limit counts as termination here. The done flag remains true afterward, but subsequent padded transitions emit neither event and pay zero. This lets us collect a fixed-size array without inventing extra episodes. We never auto-reset inside step: the returned position is the final observation, not a replacement start.

## A timeout does not always mean a zero future value

Here the task itself asks for the return during a fixed budget of actions, so truncation ends our finite-horizon objective and its future value is zero. In a continuing task interrupted only by a collection limit, the final observation can still have future value. A bootstrapped target would then mask true termination, not every timeout. For reward $-0.01$, discount $0.9$ and estimated next value $2$, that continuing target is $1.79$; replacing it with $-0.01$ changes the learning problem.

$$
y=r+\gamma(1-\mathbb{1}_{\mathrm{terminated}})V(s_{\mathrm{next}})
$$

## Make the contract small enough to test

The compiled transition assumes a binary integer action, positive horizon, and states produced by reset or step. We test those rules outside the hot loop rather than performing Python assertions on traced values. Randomness enters reset through a supplied key. Reusing that key deliberately reproduces a start; independent starts require split keys. The transition itself is deterministic. This is a teaching task, not evidence of skill on a realistic control benchmark.

## Define state and the transition

Create main.py with this block. Run python3 main.py in the course CPU environment.

```python
"""Finite-horizon tabular policy training. CPU teaching environment, not a benchmark."""
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


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
```

The function returns all information needed by the next call. No hidden counter or reset changes the episode.

## Check a complete episode

Append this block to main.py. Run python3 main.py in the course CPU environment.

```python
s = State(jnp.int32(1), jnp.int32(0), jnp.bool_(False))
positions, rewards = [int(s.position)], []
for action in [1, 1, 0, 0]:
    s, reward, terminated, truncated = step(s, jnp.int32(action))
    positions.append(int(s.position)); rewards.append(float(reward))
assert positions == [1, 2, 3, 3, 3]
assert jnp.allclose(jnp.array(rewards), jnp.array([-.01, 1., 0., 0.]))
assert int(s.elapsed) == 2
print('positions:', positions, 'rewards:', rewards)

```

The known two-action path produces one terminal reward and then zero padding.

## Run the example

```python
"""Finite-horizon tabular policy training. CPU teaching environment, not a benchmark."""
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


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

s = State(jnp.int32(1), jnp.int32(0), jnp.bool_(False))
positions, rewards = [int(s.position)], []
for action in [1, 1, 0, 0]:
    s, reward, terminated, truncated = step(s, jnp.int32(action))
    positions.append(int(s.position)); rewards.append(float(reward))
assert positions == [1, 2, 3, 3, 3]
assert jnp.allclose(jnp.array(rewards), jnp.array([-.01, 1., 0., 0.]))
assert int(s.elapsed) == 2
print('positions:', positions, 'rewards:', rewards)

```

Expected: Positions are $(1,2,3,3,3)$. Rewards are $(-0.01,1,0,0)$; elapsed time stops at $2$.

## One goal reward, then absorbing padding

**Predict:** Will the reward stay at one after the goal?

![One goal reward, then absorbing padding](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis counts transitions, starting with the first action. The vertical axis is reward for that individual transition, not accumulated return. Moving right twice from position $1$ produces $-0.01$ followed by $1$. The final two points sit at zero because the episode has already ended.

### Connect it to the computation

The positions printed by the program are $(1,2,3,3,3)$. The sharp reward peak occurs only when entering the goal. Summing the four plotted rewards gives $0.99$, exactly the two-action return. A flat tail at $1$ would expose repeated terminal rewards; this zero tail demonstrates the absorbing-state contract, not learning.

```python
visual_data = {'kind':'line','x':[1,2,3,4],'xlabel':'transition index','ylabel':'reward per transition','series':[{'label':'right, right, padded, padded','y':rewards}]}
```

## Recorded reference execution

CPU run: 2026-10-06T23:02:28.838371+00:00. JAX 0.9.2.

```text
positions: [1, 2, 3, 3, 3] rewards: [-0.009999999776482582, 1.0, 0.0, 0.0]
positions: [1, 2, 3, 3, 3] rewards: [-0.009999999776482582, 1.0, 0.0, 0.0]
deadline goal: True False
fraction starting at one: 0.5020000338554382
PASS: rl-01

```

## A goal at the deadline

**Predict before running:** Which event wins if the final permitted action reaches the goal?

```python
s = State(jnp.int32(2), jnp.int32(1), jnp.bool_(False))
s, r, term, trunc = step(s, jnp.int32(1), horizon=2)
assert bool(term) and not bool(trunc) and float(r) == 1.
print('deadline goal:', bool(term), bool(trunc))
```

**Expected:** Termination is true; truncation is false.

The goal transition defines a completed task, so the two flags remain mutually exclusive.

## The same key is the same start

**Predict before running:** Will resetting twice with one key produce two independent trials?

```python
key = jax.random.key(7)
a, b = reset(key), reset(key)
assert int(a.position) == int(b.position)
starts = jax.vmap(reset)(jax.random.split(key, 1000)).position
assert 0.4 < float(starts.mean()) < 0.6
print('fraction starting at one:', float(starts.mean()))
```

**Expected:** Replay is exact; a split-key batch contains both starts.

This verifies the randomness contract, not a guarantee that every tiny batch is balanced.

## Make it yours

Begin at position $0$, choose left for three actions with horizon $3$, and then attempt one more action. Verify the event flags, total reward and absorbing state.

<details><summary>Reference solution</summary>

```python
s = State(jnp.int32(0), jnp.int32(0), jnp.bool_(False))
total = 0.
for i in range(3):
    s, r, term, trunc = step(s, jnp.int32(0), 3)
    total += float(r)
    assert not bool(term) and bool(trunc) == (i == 2)
assert abs(total + .03) < 1e-6
s2, r, term, trunc = step(s, jnp.int32(1), 3)
assert int(s2.position) == 0 and int(s2.elapsed) == 3 and float(r) == 0.
```

</details>

## Separate the two target meanings

**Transfer / diagnosis**

Compute the continuing-task bootstrap target for a timeout and a true terminal transition when $r=-0.01$, $\gamma=0.9$ and $V=2$. Compare with our finite-horizon objective.

<details><summary>Hint</summary>

Use the termination flag for the continuing task. The finite-horizon task assigns zero value after its own deadline.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
r, gamma, value = -.01, .9, 2.
continuing_timeout = r + gamma * value
terminal = r
finite_horizon_timeout = r
assert abs(continuing_timeout - 1.79) < 1e-7
assert terminal == finite_horizon_timeout == -.01
```

The arithmetic exposes a modeling choice. Copying a done mask from a different task can silently change the objective.

</details>

## Enumerate all one-step moves

**Transfer / diagnosis**

Check both actions at every unfinished position with a Python reference, then compare the jitted function.

<details><summary>Hint</summary>

Use ordinary min and max for the independent reference.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
compiled = jax.jit(step)
for position in range(3):
    for action in [0, 1]:
        s = State(jnp.int32(position), jnp.int32(0), jnp.bool_(False))
        nxt, reward, term, trunc = compiled(s, jnp.int32(action))
        expected = min(3, max(0, position + (1 if action else -1)))
        assert int(nxt.position) == expected
        assert bool(term) == (expected == 3)
        assert abs(float(reward) - (1. if expected == 3 else -.01)) < 1e-6
```

Enumeration checks boundary behavior separately from the vectorized implementation.

</details>

## Check your understanding

What should step emit when called on a finished state?

1. Another goal reward because the agent is still at the goal
2. A new random start and its first reward
3. The unchanged state, zero reward and no new ending event

<details><summary>Answer and explanation</summary>

The unchanged state, zero reward and no new ending event

Absorbing padding preserves one episode. Reset is a separate explicit operation, and ending flags mark events rather than every later padded row.

</details>

## Diagnose the result

If return exceeds the always-right value, inspect rewards after the first ending event. If a trajectory jumps from the goal to a start, look for auto-reset before storing the final observation. If all starts match, inspect whether keys were split before batching.

## Carry forward

- Environment state, observation and policy parameters are different objects.
- Explicit ending contracts make fixed-shape rollouts safe to interpret.

## Keep your evidence

Keep independent transition enumeration, reward/terminal checks and the absorbing-state diagnosis. Explain the state and observation contract.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [JAX: explicit pseudorandom keys](https://docs.jax.dev/en/latest/random-numbers.html)
- [JAX scan: fixed-shape carry](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
- [Schulman et al.: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Gymnasium: termination and truncation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

