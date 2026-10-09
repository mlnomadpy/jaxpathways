"""Reference fixed-grid scientific inverse problem; CPU float64 teaching fixture."""
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update('jax_enable_x64', True)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np


# Define `simulate(rate, initials, steps, dt)` to carry state across steps with `jax.lax.scan`:
def simulate(rate, initials, steps=40, dt=0.05):
    """Return batch-major states, including the initial observation."""
    # Guard input contract (`initials.ndim != 1 or initials.size == 0`) and fail fast if violated.
    if initials.ndim != 1 or initials.size == 0:
        raise ValueError('initials must be a nonempty vector')
    # Guard input contract (`not isinstance(steps, int) or steps < 1 or dt <= 0`) and fail fast if violated.
    if not isinstance(steps, int) or steps < 1 or dt <= 0:
        raise ValueError('steps must be a positive integer and dt must be positive')
    # Guard input contract (`jnp.ndim(rate) != 0`) and fail fast if violated.
    if jnp.ndim(rate) != 0:
        raise ValueError('rate must be scalar')
    # Function `advance(values, unused)` implementing this stage's computation:
    def advance(values, unused):
        # Compute `a` as `-rate*values`.
        a = -rate*values
        # Compute `b` as `-rate*(values+dt*a/2)`.
        b = -rate*(values+dt*a/2)
        # Compute `c` as `-rate*(values+dt*b/2)`.
        c = -rate*(values+dt*b/2)
        # Compute `d` as `-rate*(values+dt*c)`.
        d = -rate*(values+dt*c)
        # Compute `next_values` as `values+dt*(a+2*b+2*c+d)/6`.
        next_values = values+dt*(a+2*b+2*c+d)/6
        # Return `(next_values, next_values)` to the caller.
        return next_values, next_values
    # Run compiled structured control flow via `jax.lax` (`(_, tail)`).
    _, tail = jax.lax.scan(advance, initials, None, length=steps)
    # Return `jnp.concatenate((initials[None, :], tail), axis=0).T` to the caller.
    return jnp.concatenate((initials[None,:],tail),axis=0).T


# Function `loss(log_rate, initials, observations, dt)` implementing this stage's computation:
def loss(log_rate, initials, observations, dt=0.05):
    # Guard input contract (`observations.ndim != 2 or observations.shape[0] != len(initials) or observations.shape[1] < 2`) and fail fast if violated.
    if observations.ndim != 2 or observations.shape[0] != len(initials) or observations.shape[1] < 2:
        raise ValueError('observations must have shape (batch, steps+1) with at least two times')
    # Run `simulate` to compute `predictions`.
    predictions = simulate(jnp.exp(log_rate),initials,int(observations.shape[1]-1),dt)
    # Return `jnp.mean((predictions - observations) ** 2)` to the caller.
    return jnp.mean((predictions-observations)**2)


# Define `fit(initials, observations, dt, initial_rate...)` to evaluate the objective and its automatic derivatives:
def fit(initials, observations, dt=0.05, initial_rate=1.4, updates=250, learning_rate=0.4):
    # Guard input contract (`initial_rate <= 0 or updates < 1 or learning_rate <= 0`) and fail fast if violated.
    if initial_rate <= 0 or updates < 1 or learning_rate <= 0:
        raise ValueError('initial rate, update count and learning rate must be positive')
    # Create device-backed JAX array `theta`.
    theta = jnp.log(jnp.asarray(initial_rate,dtype=jnp.float64))
    # Compute `objective` as `lambda p: loss(p,initials,observations,dt)`.
    objective = lambda p: loss(p,initials,observations,dt)
    # Validate before entering the compiled update and retain a pre-update point.
    history = [float(objective(theta))]
    # Differentiate the objective to obtain `update` via automatic differentiation.
    update = jax.jit(lambda p:p-learning_rate*jax.grad(objective)(p))
    # Repeat the update loop over `range(updates)` steps:
    for _ in range(updates):
        # Run `update` to compute `theta`.
        theta=update(theta)
        # Append the current step result to `history`.
        history.append(float(objective(theta)))
    # Return `(float(jnp.exp(theta)), np.asarray(history))` to the caller.
    return float(jnp.exp(theta)),np.asarray(history)


# Function `evaluate(rate, initials, observations, dt)` implementing this stage's computation:
def evaluate(rate, initials, observations, dt=0.05):
    # Guard input contract (`not np.isfinite(rate) or rate <= 0`) and fail fast if violated.
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError('rate must be finite and positive')
    # Run `loss` to compute `value`.
    value=loss(jnp.log(rate),initials,observations,dt)
    # Run `simulate` to compute `predictions`.
    predictions=simulate(rate,initials,int(observations.shape[1]-1),dt)
    # Compute `error` as `predictions-observations`.
    error=predictions-observations
    # Return `{'mse': float(value), 'rmse': float(jnp.sqrt(value)), 'max_abs_error': float(jnp.max(jnp.abs(error))), 'trajectory_count': int(len(initials)), 'observation_count': int(observations.size)}` to the caller.
    return {'mse':float(value),'rmse':float(jnp.sqrt(value)),
            'max_abs_error':float(jnp.max(jnp.abs(error))),
            'trajectory_count':int(len(initials)),'observation_count':int(observations.size)}
