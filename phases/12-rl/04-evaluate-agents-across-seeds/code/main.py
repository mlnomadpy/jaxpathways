"""Evaluate agents across seeds: worked experiments and reference solutions. CPU checks."""

# Assemble the verified learning system
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

def log_probability(theta, observations, actions):
    logits = theta[jnp.minimum(observations, 2)]
    return jnp.where(actions == 1, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits))


@partial(jax.jit, static_argnames=('batch_size', 'horizon'))
def rollout(theta, key, batch_size=256, horizon=8):
    reset_key, action_key = jax.random.split(key)
    states = jax.vmap(reset)(jax.random.split(reset_key, batch_size))
    time_keys = jax.random.split(action_key, horizon)
    def body(states, time_key):
        observations = states.position
        keys = jax.random.split(time_key, batch_size)
        probabilities = jax.nn.sigmoid(theta[jnp.minimum(observations, 2)])
        actions = jax.vmap(jax.random.bernoulli)(keys, probabilities).astype(jnp.int32)
        next_states, rewards, terminated, truncated = jax.vmap(
            lambda s, a: step(s, a, horizon))(states, actions)
        record = dict(observation=observations, action=actions, reward=rewards,
                      active=~states.done, terminated=terminated, truncated=truncated,
                      old_logp=log_probability(theta, observations, actions))
        return next_states, record
    final, records = jax.lax.scan(body, states, time_keys)
    return final, records


def returns_to_go(rewards):
    """Undiscounted finite-episode objective; rewards after done are already zero."""
    return jnp.cumsum(rewards[::-1], axis=0)[::-1]

def policy_loss(theta, records, method='ppo', clip=.2):
    """Normalize by episodes, not live transitions. Freeze behavior data/returns."""
    advantage = jax.lax.stop_gradient(returns_to_go(records['reward']))
    old_logp = jax.lax.stop_gradient(records['old_logp'])
    new_logp = log_probability(theta, records['observation'], records['action'])
    if method == 'reinforce':
        terms = new_logp * advantage
    elif method == 'ppo':
        ratio = jnp.exp(new_logp - old_logp)
        terms = jnp.minimum(ratio * advantage, jnp.clip(ratio, 1-clip, 1+clip) * advantage)
    else:
        raise ValueError('method must be reinforce or ppo')
    return -jnp.sum(jnp.where(records['active'], terms, 0.)) / records['reward'].shape[1]


@partial(jax.jit, static_argnames=('horizon',))
def exact_return(theta, horizon=8):
    """Dynamic programming over states, independent of sampled rollouts."""
    p = jax.nn.sigmoid(theta)
    states = jnp.arange(3)
    left = jnp.maximum(states-1, 0)
    right = states+1
    def backup(_, values):
        q_left = -.01 + values[left]
        q_right = jnp.where(right == 3, 1., -.01 + values[right])
        return jnp.concatenate(((1-p)*q_left+p*q_right, jnp.zeros(1)))
    values = jax.lax.fori_loop(0, horizon, backup, jnp.zeros(4))
    return jnp.mean(values[:2])


@partial(jax.jit, static_argnames=('updates', 'batch_size', 'horizon', 'method', 'epochs'))
def _train(seed, updates, batch_size, horizon, method, epochs, learning_rate):
    key = jax.random.key(seed)
    theta = jnp.zeros(3)
    def update(carry, _):
        theta, key = carry
        key, sample_key = jax.random.split(key)
        _, records = rollout(theta, sample_key, batch_size, horizon)
        def epoch(_, candidate):
            grad = jax.grad(policy_loss)(candidate, records, method)
            return candidate - learning_rate * grad
        theta = jax.lax.fori_loop(0, epochs, epoch, theta)
        return (theta, key), exact_return(theta, horizon)
    (theta, _), history = jax.lax.scan(update, (theta, key), None, length=updates)
    return theta, jnp.concatenate((exact_return(jnp.zeros(3), horizon)[None], history))


def train(seed=0, updates=40, batch_size=256, horizon=8, method='ppo', epochs=3, learning_rate=.5):
    if updates < 1 or batch_size < 1 or horizon < 1 or epochs < 1 or learning_rate <= 0:
        raise ValueError('positive updates, batch_size, horizon, epochs and learning_rate required')
    if method not in ('ppo', 'reinforce'):
        raise ValueError('unknown method')
    if method == 'reinforce' and epochs != 1:
        raise ValueError('REINFORCE uses one update per fresh on-policy rollout')
    return _train(seed, updates, batch_size, horizon, method, epochs, learning_rate)

# Add a read-only evaluator
def evaluate(theta, seeds, batch_size=512, horizon=8):
    """One mean per evaluation stream; preserve raw episode returns too."""
    theta = jnp.asarray(theta)
    if theta.shape != (3,) or not bool(jnp.all(jnp.isfinite(theta))):
        raise ValueError('theta must contain three finite logits')
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or batch_size < 1 or horizon < 1:
        raise ValueError('provide at least two distinct evaluation seeds and positive sizes')
    returns = np.stack([np.asarray(rollout(theta, jax.random.key(s), batch_size, horizon)[1]['reward'].sum(0)) for s in seeds])
    return {'seeds': list(seeds), 'episode_returns': returns, 'stream_means': returns.mean(1),
            'mean': float(returns.mean()), 'exact': float(exact_return(theta, horizon))}

# Train several agents and retain every result
training_seeds = [0,7,23,41,59]
evaluation_seeds = [1000,1001,1002,1003]
policies, histories, reports = [], [], []
for seed in training_seeds:
    policy, history = train(seed=seed)
    policies.append(policy); histories.append(history)
    reports.append(evaluate(policy,evaluation_seeds))
trained_means = np.array([r['mean'] for r in reports])
exact_means = np.array([r['exact'] for r in reports])
baseline = evaluate(jnp.zeros(3),evaluation_seeds)
assert np.all(trained_means > baseline['mean'] + .4)
assert np.all(np.abs(trained_means-exact_means) < .03)
print('training seeds:', training_seeds)
print('held-out mean per agent:', trained_means)
print('exact mean per agent:', exact_means)
print('uniform baseline mean:', baseline['mean'])
print('across-agent mean / sample sd:', trained_means.mean(),trained_means.std(ddof=1))


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

def log_probability(theta, observations, actions):
    logits = theta[jnp.minimum(observations, 2)]
    return jnp.where(actions == 1, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits))


@partial(jax.jit, static_argnames=('batch_size', 'horizon'))
def rollout(theta, key, batch_size=256, horizon=8):
    reset_key, action_key = jax.random.split(key)
    states = jax.vmap(reset)(jax.random.split(reset_key, batch_size))
    time_keys = jax.random.split(action_key, horizon)
    def body(states, time_key):
        observations = states.position
        keys = jax.random.split(time_key, batch_size)
        probabilities = jax.nn.sigmoid(theta[jnp.minimum(observations, 2)])
        actions = jax.vmap(jax.random.bernoulli)(keys, probabilities).astype(jnp.int32)
        next_states, rewards, terminated, truncated = jax.vmap(
            lambda s, a: step(s, a, horizon))(states, actions)
        record = dict(observation=observations, action=actions, reward=rewards,
                      active=~states.done, terminated=terminated, truncated=truncated,
                      old_logp=log_probability(theta, observations, actions))
        return next_states, record
    final, records = jax.lax.scan(body, states, time_keys)
    return final, records


def returns_to_go(rewards):
    """Undiscounted finite-episode objective; rewards after done are already zero."""
    return jnp.cumsum(rewards[::-1], axis=0)[::-1]

def policy_loss(theta, records, method='ppo', clip=.2):
    """Normalize by episodes, not live transitions. Freeze behavior data/returns."""
    advantage = jax.lax.stop_gradient(returns_to_go(records['reward']))
    old_logp = jax.lax.stop_gradient(records['old_logp'])
    new_logp = log_probability(theta, records['observation'], records['action'])
    if method == 'reinforce':
        terms = new_logp * advantage
    elif method == 'ppo':
        ratio = jnp.exp(new_logp - old_logp)
        terms = jnp.minimum(ratio * advantage, jnp.clip(ratio, 1-clip, 1+clip) * advantage)
    else:
        raise ValueError('method must be reinforce or ppo')
    return -jnp.sum(jnp.where(records['active'], terms, 0.)) / records['reward'].shape[1]


@partial(jax.jit, static_argnames=('horizon',))
def exact_return(theta, horizon=8):
    """Dynamic programming over states, independent of sampled rollouts."""
    p = jax.nn.sigmoid(theta)
    states = jnp.arange(3)
    left = jnp.maximum(states-1, 0)
    right = states+1
    def backup(_, values):
        q_left = -.01 + values[left]
        q_right = jnp.where(right == 3, 1., -.01 + values[right])
        return jnp.concatenate(((1-p)*q_left+p*q_right, jnp.zeros(1)))
    values = jax.lax.fori_loop(0, horizon, backup, jnp.zeros(4))
    return jnp.mean(values[:2])


@partial(jax.jit, static_argnames=('updates', 'batch_size', 'horizon', 'method', 'epochs'))
def _train(seed, updates, batch_size, horizon, method, epochs, learning_rate):
    key = jax.random.key(seed)
    theta = jnp.zeros(3)
    def update(carry, _):
        theta, key = carry
        key, sample_key = jax.random.split(key)
        _, records = rollout(theta, sample_key, batch_size, horizon)
        def epoch(_, candidate):
            grad = jax.grad(policy_loss)(candidate, records, method)
            return candidate - learning_rate * grad
        theta = jax.lax.fori_loop(0, epochs, epoch, theta)
        return (theta, key), exact_return(theta, horizon)
    (theta, _), history = jax.lax.scan(update, (theta, key), None, length=updates)
    return theta, jnp.concatenate((exact_return(jnp.zeros(3), horizon)[None], history))


def train(seed=0, updates=40, batch_size=256, horizon=8, method='ppo', epochs=3, learning_rate=.5):
    if updates < 1 or batch_size < 1 or horizon < 1 or epochs < 1 or learning_rate <= 0:
        raise ValueError('positive updates, batch_size, horizon, epochs and learning_rate required')
    if method not in ('ppo', 'reinforce'):
        raise ValueError('unknown method')
    if method == 'reinforce' and epochs != 1:
        raise ValueError('REINFORCE uses one update per fresh on-policy rollout')
    return _train(seed, updates, batch_size, horizon, method, epochs, learning_rate)

def evaluate(theta, seeds, batch_size=512, horizon=8):
    """One mean per evaluation stream; preserve raw episode returns too."""
    theta = jnp.asarray(theta)
    if theta.shape != (3,) or not bool(jnp.all(jnp.isfinite(theta))):
        raise ValueError('theta must contain three finite logits')
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or batch_size < 1 or horizon < 1:
        raise ValueError('provide at least two distinct evaluation seeds and positive sizes')
    returns = np.stack([np.asarray(rollout(theta, jax.random.key(s), batch_size, horizon)[1]['reward'].sum(0)) for s in seeds])
    return {'seeds': list(seeds), 'episode_returns': returns, 'stream_means': returns.mean(1),
            'mean': float(returns.mean()), 'exact': float(exact_return(theta, horizon))}

training_seeds = [0,7,23,41,59]
evaluation_seeds = [1000,1001,1002,1003]
policies, histories, reports = [], [], []
for seed in training_seeds:
    policy, history = train(seed=seed)
    policies.append(policy); histories.append(history)
    reports.append(evaluate(policy,evaluation_seeds))
trained_means = np.array([r['mean'] for r in reports])
exact_means = np.array([r['exact'] for r in reports])
baseline = evaluate(jnp.zeros(3),evaluation_seeds)
assert np.all(trained_means > baseline['mean'] + .4)
assert np.all(np.abs(trained_means-exact_means) < .03)
print('training seeds:', training_seeds)
print('held-out mean per agent:', trained_means)
print('exact mean per agent:', exact_means)
print('uniform baseline mean:', baseline['mean'])
print('across-agent mean / sample sd:', trained_means.mean(),trained_means.std(ddof=1))


# Figure data experiment
visual_data = {'kind':'bar','labels':[str(s) for s in training_seeds],'xlabel':'training seed','ylabel':'mean episode return','series':[{'label':'held-out sampled','y':trained_means.tolist()},{'label':'exact expectation','y':exact_means.tolist()},{'label':'uniform baseline (shared)','y':[baseline['mean']]*5}]}

# Experiment: Check baselines with known returns
left = evaluate(jnp.full(3,-100.),evaluation_seeds)
right = evaluate(jnp.full(3,100.),evaluation_seeds)
assert abs(left['mean']+.08) < 1e-6
assert abs(right['mean']-.985) < .002
assert left['mean'] < baseline['mean'] < right['mean']
assert abs(baseline['mean']-.466875) < .04
print('left / random / right:', left['mean'], baseline['mean'], right['mean'])

# Experiment: Prove isolation and replay
before = np.array(policies[0],copy=True)
a = evaluate(policies[0],evaluation_seeds)
b = evaluate(policies[0],evaluation_seeds)
np.testing.assert_array_equal(before,policies[0])
np.testing.assert_array_equal(a['episode_returns'],b['episode_returns'])
c = evaluate(policies[0],[2000,2001,2002,2003])
assert not np.array_equal(a['episode_returns'],c['episode_returns'])

# Reference solution. Try the exercise before reading this.
deltas = trained_means - baseline['mean']
assert deltas.shape == (5,)
assert np.all(deltas > .4)
print('paired improvements:', deltas)
print('mean improvement across trained agents:', deltas.mean())

# Reference practice: Expose an unequal-batch averaging bug
means = np.array([1.,0.]); counts = np.array([1,9])
wrong = means.mean(); right = np.average(means,weights=counts)
assert wrong == .5 and right == .1

# Reference practice: Distinguish spread from standard error
values = np.array([.8,.9,1.])
sd = values.std(ddof=1)
naive_se = sd/np.sqrt(3)
assert abs(sd-.1) < 1e-12
assert abs(naive_se-.0577350269) < 1e-9
print('sample sd / independence-assuming SE:',sd,naive_se)
print("PASS: rl-04")
