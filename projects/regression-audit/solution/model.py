"""Reference implementation for the CPU foundation capstone."""
# Import jax for this computation.
import jax
import jax.numpy as jnp

# Function `predict(params, x)` implementing this stage's computation:
def predict(params, x):
    # Return `params['weight'] * x + params['bias']` to the caller.
    return params['weight'] * x + params['bias']

# Function `loss(params, x, y)` implementing this stage's computation:
def loss(params, x, y):
    # Guard input contract (`x.shape != y.shape`) and fail fast if violated.
    if x.shape != y.shape:
        raise ValueError('Inputs and scalar targets must have matching shapes')
    # Return `jnp.mean((predict(params, x) - y) ** 2)` to the caller.
    return jnp.mean((predict(params, x) - y) ** 2)

# Define `gradient_check(params, x, y, h)` to evaluate the objective and its automatic derivatives:
def gradient_check(params, x, y, h=1e-2):
    # Differentiate the objective to obtain `automatic` via automatic differentiation.
    automatic = jax.grad(loss)(params, x, y)
    # Evaluate `finite` from the current inputs and state.
    finite = {}
    # Iterate over `key` to step through the computation:
    for key in params:
        # Evaluate `plus` from the current inputs and state.
        plus = {**params, key: params[key] + h}
        # Evaluate `minus` from the current inputs and state.
        minus = {**params, key: params[key] - h}
        # Evaluate `finite[key]` from the current inputs and state.
        finite[key] = (loss(plus, x, y) - loss(minus, x, y)) / (2*h)
    # Return `(automatic, finite)` to the caller.
    return automatic, finite

# Define `train(params, x, y, rate...)` to evaluate the objective and its automatic derivatives:
def train(params, x, y, rate=0.15, steps=200):
    # Guard input contract (`steps < 1`) and fail fast if violated.
    if steps < 1:
        raise ValueError('steps must be positive')
    # Define `step(p, _)` to evaluate the objective and its automatic derivatives:
    def step(p, _):
        # Differentiate the objective to obtain `(value, gradients)` via automatic differentiation.
        value, gradients = jax.value_and_grad(loss)(p, x, y)
        # Transform every leaf of the parameter PyTree (`updated`).
        updated = jax.tree.map(lambda v, g: v - rate * g, p, gradients)
        # Return `(updated, value)` to the caller.
        return updated, value
    # Return `jax.lax.scan(step, params, None, length=steps)` to the caller.
    return jax.lax.scan(step, params, None, length=steps)

# Function `report(params, x, y, rate, ...)` implementing this stage's computation:
def report(params, x, y, rate=0.15, steps=200):
    # Run `train` to compute `(fitted, history)`.
    fitted, history = train(params, x, y, rate, steps)
    # Initialize array `heldout_x` with explicit values and shape.
    heldout_x = jnp.array([-0.8, 0.2, 0.8])
    # Evaluate `heldout_y` from the current inputs and state.
    heldout_y = 2 * heldout_x + 1
    # Return `{'parameters': {k: float(v) for k, v in fitted.items()}, 'initial_loss': float(history[0]), 'training_loss': float(loss(fitted, x, y)), 'heldout_loss': float(loss(fitted, heldout_x, heldout_y)), 'finite': bool(jnp.all(jnp.isfinite(history))), 'steps': steps, 'learning_rate': rate, 'backend': jax.default_backend()}` to the caller.
    return {'parameters': {k: float(v) for k,v in fitted.items()},
            'initial_loss': float(history[0]), 'training_loss': float(loss(fitted,x,y)),
            'heldout_loss': float(loss(fitted,heldout_x,heldout_y)),
            'finite': bool(jnp.all(jnp.isfinite(history))), 'steps': steps,
            'learning_rate': rate, 'backend': jax.default_backend()}
