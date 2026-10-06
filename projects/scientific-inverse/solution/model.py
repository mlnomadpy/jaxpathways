"""Reference fixed-grid scientific inverse problem; CPU float64 teaching fixture."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np


def simulate(rate, initials, steps=40, dt=0.05):
    """Return batch-major states, including the initial observation."""
    if initials.ndim != 1 or initials.size == 0:
        raise ValueError('initials must be a nonempty vector')
    if not isinstance(steps, int) or steps < 1 or dt <= 0:
        raise ValueError('steps must be a positive integer and dt must be positive')
    if jnp.ndim(rate) != 0:
        raise ValueError('rate must be scalar')
    def advance(values, unused):
        a = -rate*values
        b = -rate*(values+dt*a/2)
        c = -rate*(values+dt*b/2)
        d = -rate*(values+dt*c)
        next_values = values+dt*(a+2*b+2*c+d)/6
        return next_values, next_values
    _, tail = jax.lax.scan(advance, initials, None, length=steps)
    return jnp.concatenate((initials[None,:],tail),axis=0).T


def loss(log_rate, initials, observations, dt=0.05):
    if observations.ndim != 2 or observations.shape[0] != len(initials) or observations.shape[1] < 2:
        raise ValueError('observations must have shape (batch, steps+1) with at least two times')
    predictions = simulate(jnp.exp(log_rate),initials,int(observations.shape[1]-1),dt)
    return jnp.mean((predictions-observations)**2)


def fit(initials, observations, dt=0.05, initial_rate=1.4, updates=250, learning_rate=0.4):
    if initial_rate <= 0 or updates < 1 or learning_rate <= 0:
        raise ValueError('initial rate, update count and learning rate must be positive')
    theta = jnp.log(jnp.asarray(initial_rate,dtype=jnp.float64))
    objective = lambda p: loss(p,initials,observations,dt)
    # Validate before entering the compiled update and retain a pre-update point.
    history = [float(objective(theta))]
    update = jax.jit(lambda p:p-learning_rate*jax.grad(objective)(p))
    for _ in range(updates):
        theta=update(theta)
        history.append(float(objective(theta)))
    return float(jnp.exp(theta)),np.asarray(history)


def evaluate(rate, initials, observations, dt=0.05):
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError('rate must be finite and positive')
    value=loss(jnp.log(rate),initials,observations,dt)
    predictions=simulate(rate,initials,int(observations.shape[1]-1),dt)
    error=predictions-observations
    return {'mse':float(value),'rmse':float(jnp.sqrt(value)),
            'max_abs_error':float(jnp.max(jnp.abs(error))),
            'trajectory_count':int(len(initials)),'observation_count':int(observations.size)}
