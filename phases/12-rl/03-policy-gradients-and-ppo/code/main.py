"""Policy gradients and PPO: worked experiments and reference solutions. CPU checks."""

# Reuse the environment and collector
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

# Add objectives and the repeated training loop
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

# Train and compare to a derivative reference
theta_pg, history_pg = train(seed=0, method='reinforce', epochs=1)
theta_ppo, history_ppo = train(seed=0, method='ppo', epochs=3)
assert history_pg[-1] > .9 and history_ppo[-1] > .9
assert jnp.allclose(history_pg[0], .466875, atol=1e-6)
print('REINFORCE initial/final exact return:', history_pg[0], history_pg[-1])
print('PPO initial/final exact return:', history_ppo[0], history_ppo[-1])
# Compare a derivative of exact expectation with central differences.
z = jnp.array([-.2, .3, .7]); eps = 1e-3
d = jnp.array([0., 1., 0.])
fd = (exact_return(z+eps*d)-exact_return(z-eps*d))/(2*eps)
assert jnp.allclose(fd, jax.grad(exact_return)(z)[1], atol=1e-4)


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

theta_pg, history_pg = train(seed=0, method='reinforce', epochs=1)
theta_ppo, history_ppo = train(seed=0, method='ppo', epochs=3)
assert history_pg[-1] > .9 and history_ppo[-1] > .9
assert jnp.allclose(history_pg[0], .466875, atol=1e-6)
print('REINFORCE initial/final exact return:', history_pg[0], history_pg[-1])
print('PPO initial/final exact return:', history_ppo[0], history_ppo[-1])
# Compare a derivative of exact expectation with central differences.
z = jnp.array([-.2, .3, .7]); eps = 1e-3
d = jnp.array([0., 1., 0.])
fd = (exact_return(z+eps*d)-exact_return(z-eps*d))/(2*eps)
assert jnp.allclose(fd, jax.grad(exact_return)(z)[1], atol=1e-4)


# Figure data experiment
visual_data = {'kind':'line','x':list(range(41)),'xlabel':'fresh rollout batches','ylabel':'exact expected episode return','series':[{'label':'REINFORCE: 1 epoch/batch','y':history_pg.tolist()},{'label':'clipped updates: 3 epochs/batch','y':history_ppo.tolist()}]}

# Experiment: Check clipping on both signs
ratio = jnp.array([.6,1.,1.4])
positive = jnp.minimum(ratio,jnp.clip(ratio,.8,1.2))
negative = jnp.minimum(-ratio,-jnp.clip(ratio,.8,1.2))
assert jnp.allclose(positive,jnp.array([.6,1.,1.2]))
assert jnp.allclose(negative,jnp.array([-.8,-1.,-1.4]))
print('positive / negative targets:', positive, negative)

# Experiment: Compare sampled and exact gradients
_, batch = rollout(z,jax.random.key(902),32768,8)
sampled = -jax.grad(policy_loss)(z,batch,'reinforce')
exact = jax.grad(exact_return)(z)
assert jnp.allclose(sampled,exact,atol=.012,rtol=.04)
print('sampled / exact gradient:', sampled, exact)

# Reference solution. Try the exercise before reading this.
other_theta, other_history = train(seed=11,method='reinforce',epochs=1)
assert other_history[-1] > other_history[0] + .35
assert other_history[-1] > .9
assert not jnp.array_equal(other_theta,theta_pg)
print('changed seed final exact return:', other_history[-1])

# Reference practice: Inspect a frozen behavior reference
_, b = rollout(jnp.zeros(3),jax.random.key(77),128,8)
old = b['old_logp'].copy()
newlog = log_probability(jnp.ones(3),b['observation'],b['action'])
ratios = jnp.exp(newlog-old)
assert jnp.array_equal(old,b['old_logp'])
assert float(jnp.min(ratios)) < 1 < float(jnp.max(ratios))
old_grad = jax.grad(lambda lp: policy_loss(jnp.ones(3),dict(b,old_logp=lp)))(old)
assert jnp.all(old_grad == 0)

# Reference practice: Change the time budget
short = exact_return(jnp.full(3,100.),horizon=2)
assert jnp.allclose(short,(-.02+.99)/2,atol=1e-6)
assert short < exact_return(jnp.full(3,100.),horizon=8)
print("PASS: rl-03")
