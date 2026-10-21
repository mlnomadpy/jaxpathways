"""Finite-horizon tabular policy training. CPU teaching environment, not a benchmark."""

# Import functools (partial) for this computation.
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


# Define `State` module / container with explicit state and forward pass:
class State(NamedTuple):
    # Execute `position: jax.Array`.
    position: jax.Array
    # Execute `elapsed: jax.Array`.
    elapsed: jax.Array
    # Execute `done: jax.Array`.
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
    # Compute `active` as `~state.done`.
    active = ~state.done
    # Combine or mask array elements to form `proposed`.
    proposed = jnp.clip(state.position + 2 * action - 1, 0, 3)
    # Combine or mask array elements to form `position`.
    position = jnp.where(active, proposed, state.position)
    # Compute `elapsed` as `state.elapsed + active.astype(jnp.int32)`.
    elapsed = state.elapsed + active.astype(jnp.int32)
    # Compute `terminated` as `active & (position == 3)`.
    terminated = active & (position == 3)
    # Compute `truncated` as `active & ~terminated & (elapsed >= horizon)`.
    truncated = active & ~terminated & (elapsed >= horizon)
    # Combine or mask array elements to form `reward`.
    reward = jnp.where(active, jnp.where(terminated, 1.0, -0.01), 0.0)
    # Return `(State(position, elapsed, state.done | terminated | truncated), reward, terminated, truncated)` to the caller.
    return (
        State(position, elapsed, state.done | terminated | truncated),
        reward,
        terminated,
        truncated,
    )


# Function `log_probability(theta, observations, actions)` implementing this stage's computation:
def log_probability(theta, observations, actions):
    # Reduce across the target axis to summarize `logits`.
    logits = theta[jnp.minimum(observations, 2)]
    # Return `jnp.where(actions == 1, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits))` to the caller.
    return jnp.where(
        actions == 1, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits)
    )


# Define and JIT-compile `rollout(theta, key, batch_size, horizon)` so XLA traces and fuses the operations:
@partial(jax.jit, static_argnames=("batch_size", "horizon"))
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
        # Compute `observations` as `states.position`.
        observations = states.position
        # Split the PRNG key deterministically into independent subkeys (`keys`).
        keys = jax.random.split(time_key, batch_size)
        # Reduce across the target axis to summarize `probabilities`.
        probabilities = jax.nn.sigmoid(theta[jnp.minimum(observations, 2)])
        # Draw pseudorandom samples for `actions` using the explicit RNG state.
        actions = jax.vmap(jax.random.bernoulli)(keys, probabilities).astype(jnp.int32)
        # Vectorize across the batch dimension without a Python loop (`(next_states, rewards, terminated, truncated)`).
        next_states, rewards, terminated, truncated = jax.vmap(
            lambda s, a: step(s, a, horizon)
        )(states, actions)
        # Evaluate `observation=observations, action=actions, reward=rewards, active=~states.done, terminated=terminated, truncated=truncated, old_logp=log_probability(theta, observations, actions)` and convert the result into Python scalar/collection `record`.
        record = dict(
            observation=observations,
            next_observation=next_states.position,
            next_elapsed=next_states.elapsed,
            action=actions,
            reward=rewards,
            active=~states.done,
            terminated=terminated,
            truncated=truncated,
            old_logp=log_probability(theta, observations, actions),
        )
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
def policy_loss(theta, records, method="ppo", clip=0.2):
    """Normalize by episodes, not live transitions. Freeze behavior data/returns."""
    # Run `jax.lax.stop_gradient` to compute `advantage`.
    advantage = jax.lax.stop_gradient(returns_to_go(records["reward"]))
    # Run `jax.lax.stop_gradient` to compute `old_logp`.
    old_logp = jax.lax.stop_gradient(records["old_logp"])
    # Run `log_probability` to compute `new_logp`.
    new_logp = log_probability(theta, records["observation"], records["action"])
    # Branch on condition `method == 'reinforce'`:
    if method == "reinforce":
        terms = new_logp * advantage
    elif method == "ppo":
        ratio = jnp.exp(new_logp - old_logp)
        terms = jnp.minimum(
            ratio * advantage, jnp.clip(ratio, 1 - clip, 1 + clip) * advantage
        )
    else:
        raise ValueError("method must be reinforce or ppo")
    # Return `-jnp.sum(jnp.where(records['active'], terms, 0.0)) / records['reward'].shape[1]` to the caller.
    return (
        -jnp.sum(jnp.where(records["active"], terms, 0.0)) / records["reward"].shape[1]
    )


# Define and JIT-compile `exact_return(theta, horizon)` so XLA traces and fuses the operations:
@partial(jax.jit, static_argnames=("horizon",))
# Function `exact_return(theta, horizon)` implementing this stage's computation:
def exact_return(theta, horizon=8):
    """Dynamic programming over states, independent of sampled rollouts."""
    # Run `jax.nn.sigmoid` to compute `p`.
    p = jax.nn.sigmoid(theta)
    # Initialize array `states` with explicit values and shape.
    states = jnp.arange(3)
    # Reduce across the target axis to summarize `left`.
    left = jnp.maximum(states - 1, 0)
    # Compute `right` as `states+1`.
    right = states + 1

    # Function `backup(_, values)` implementing this stage's computation:
    def backup(_, values):
        # Compute `q_left` as `-.01 + values[left]`.
        q_left = -0.01 + values[left]
        # Combine or mask array elements to form `q_right`.
        q_right = jnp.where(right == 3, 1.0, -0.01 + values[right])
        # Return `jnp.concatenate(((1 - p) * q_left + p * q_right, jnp.zeros(1)))` to the caller.
        return jnp.concatenate(((1 - p) * q_left + p * q_right, jnp.zeros(1)))

    # Allocate initialized array `values` with the specified shape and dtype.
    values = jax.lax.fori_loop(0, horizon, backup, jnp.zeros(4))
    # Return `jnp.mean(values[:2])` to the caller.
    return jnp.mean(values[:2])


# Define and JIT-compile `_train(seed, updates, batch_size, horizon...)` so XLA traces and fuses the operations:
@partial(
    jax.jit, static_argnames=("updates", "batch_size", "horizon", "method", "epochs")
)
# Function `_train(seed, updates, batch_size, horizon, ...)` implementing this stage's computation:
def _train(seed, updates, batch_size, horizon, method, epochs, learning_rate):
    # Create or split explicit PRNG key(s) (`key`) for reproducible randomness.
    key = jax.random.key(seed)
    # Initialize array `theta` with explicit values and shape.
    theta = jnp.zeros(3)

    # Define `update(carry, _)` to evaluate the objective and its automatic derivatives:
    def update(carry, _):
        # Compute `theta, key` as `carry`.
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
def train(
    seed=0,
    updates=40,
    batch_size=256,
    horizon=8,
    method="ppo",
    epochs=3,
    learning_rate=0.5,
):
    # Guard input contract (`updates < 1 or batch_size < 1 or horizon < 1 or (epochs < 1) or (learning_rate <= 0)`) and fail fast if violated.
    if updates < 1 or batch_size < 1 or horizon < 1 or epochs < 1 or learning_rate <= 0:
        raise ValueError(
            "positive updates, batch_size, horizon, epochs and learning_rate required"
        )
    # Guard input contract (`method not in ('ppo', 'reinforce')`) and fail fast if violated.
    if method not in ("ppo", "reinforce"):
        raise ValueError("unknown method")
    # Guard input contract (`method == 'reinforce' and epochs != 1`) and fail fast if violated.
    if method == "reinforce" and epochs != 1:
        raise ValueError("REINFORCE uses one update per fresh on-policy rollout")
    # Return `_train(seed, updates, batch_size, horizon, method, epochs, learning_rate)` to the caller.
    return _train(seed, updates, batch_size, horizon, method, epochs, learning_rate)


# Function `evaluate(theta, seeds, batch_size, horizon)` implementing this stage's computation:
def evaluate(theta, seeds, batch_size=512, horizon=8):
    """One mean per evaluation stream; preserve raw episode returns too."""
    # Create device-backed JAX array `theta`.
    theta = jnp.asarray(theta)
    # Guard input contract (`theta.shape != (3,) or not bool(jnp.all(jnp.isfinite(theta)))`) and fail fast if violated.
    if theta.shape != (3,) or not bool(jnp.all(jnp.isfinite(theta))):
        raise ValueError("theta must contain three finite logits")
    # Evaluate `seeds` and convert the result into Python scalar/collection `seeds`.
    seeds = tuple(seeds)
    # Guard input contract (`len(seeds) < 2 or len(set(seeds)) != len(seeds) or batch_size < 1 or (horizon < 1)`) and fail fast if violated.
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or batch_size < 1 or horizon < 1:
        raise ValueError(
            "provide at least two distinct evaluation seeds and positive sizes"
        )
    # Convert `returns` to a host NumPy array for inspection or verification.
    returns = np.stack(
        [
            np.asarray(
                rollout(theta, jax.random.key(s), batch_size, horizon)[1]["reward"].sum(
                    0
                )
            )
            for s in seeds
        ]
    )
    # Return `{'seeds': list(seeds), 'episode_returns': returns, 'stream_means': returns.mean(1), 'mean': float(returns.mean()), 'exact': float(exact_return(theta, horizon))}` to the caller.
    return {
        "seeds": list(seeds),
        "episode_returns": returns,
        "stream_means": returns.mean(1),
        "mean": float(returns.mean()),
        "exact": float(exact_return(theta, horizon)),
    }


# Stage 5 — fit a state-value table and estimate GAE from retained transitions.
def value_predict(value_table, observations):
    """Look up V(s) for corridor positions 0..3."""
    return value_table[jnp.clip(observations, 0, 3)]


def value_loss(value_table, records, targets):
    """Mean squared prediction error over live transitions only."""
    predictions = value_predict(value_table, records["observation"])
    active = jnp.asarray(records["active"], dtype=jnp.bool_)
    targets = jax.lax.stop_gradient(jnp.asarray(targets))
    mask = active.astype(predictions.dtype)
    count = jnp.sum(mask)
    # Sanitize targets before subtracting: NaN/Inf padding must not enter
    # residual arithmetic, where a later where could still poison gradients.
    safe_targets = jnp.where(active, targets, jnp.zeros_like(targets))
    residual = jnp.where(
        active, predictions - safe_targets, jnp.zeros_like(predictions)
    )
    return jnp.sum(jnp.square(residual)) / jnp.maximum(count, 1.0)


def fit_value_function(records, targets, updates=200, learning_rate=0.1):
    """Fit V(s) to return targets and retain initial plus per-update MSE."""
    targets = jnp.asarray(targets)
    observations = records["observation"]
    active = records["active"]
    if observations.shape != targets.shape or active.shape != targets.shape:
        raise ValueError(
            "observations, active mask, and targets must have the same shape"
        )
    if updates < 1 or learning_rate <= 0:
        raise ValueError("updates and learning_rate must be positive")
    if not bool(jnp.any(active)):
        raise ValueError("cannot fit a value function from an empty active mask")
    value_table = jnp.zeros((4,), dtype=jnp.float32)
    loss_fn = lambda table: value_loss(table, records, targets)
    initial_loss = loss_fn(value_table)

    def update(table, _):
        loss, grad = jax.value_and_grad(loss_fn)(table)
        table = table - learning_rate * grad
        return table, loss_fn(table)

    value_table, rest = jax.lax.scan(update, value_table, None, length=updates)
    return value_table, jnp.concatenate((initial_loss[None], rest))


def generalized_advantage_estimate(
    records, value_table, gamma=0.9, lam=0.95, bootstrap_on_truncation=False
):
    """Reverse-scan GAE with pre-reset observations and separate masks."""
    observations = records["observation"]
    next_observations = records["next_observation"]
    active = records["active"].astype(jnp.bool_)
    terminated = records["terminated"].astype(jnp.bool_)
    truncated = records["truncated"].astype(jnp.bool_)
    rewards = records["reward"]
    values = value_predict(value_table, observations)
    next_values = value_predict(value_table, next_observations)
    # At a true terminal, the next-state value is zero. At an external cutoff,
    # the retained final observation supplies the bootstrap. Finite-horizon
    # callers opt out explicitly through bootstrap_on_truncation=False.
    bootstrap_mask = ~(terminated | (truncated & ~jnp.asarray(bootstrap_on_truncation)))
    delta = rewards + gamma * bootstrap_mask.astype(values.dtype) * next_values - values
    delta = jnp.where(active, delta, 0.0)
    next_active = jnp.concatenate((active[1:], jnp.zeros_like(active[:1])), axis=0)
    trace_mask = active & ~(terminated | truncated) & next_active
    reverse_delta = delta[::-1]
    reverse_trace_mask = trace_mask[::-1]

    def scan_step(carry, xs):
        residual, continues = xs
        advantage = residual + gamma * lam * continues.astype(residual.dtype) * carry
        return advantage, advantage

    _, reverse_advantages = jax.lax.scan(
        scan_step,
        jnp.zeros_like(delta[0]) if delta.shape[0] else jnp.zeros(delta.shape[1:]),
        (reverse_delta, reverse_trace_mask),
    )
    return jnp.where(active, reverse_advantages[::-1], 0.0)
