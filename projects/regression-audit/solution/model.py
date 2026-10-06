"""Reference implementation for the CPU foundation capstone."""
import jax
import jax.numpy as jnp

def predict(params, x):
    return params['weight'] * x + params['bias']

def loss(params, x, y):
    if x.shape != y.shape:
        raise ValueError('Inputs and scalar targets must have matching shapes')
    return jnp.mean((predict(params, x) - y) ** 2)

def gradient_check(params, x, y, h=1e-2):
    automatic = jax.grad(loss)(params, x, y)
    finite = {}
    for key in params:
        plus = {**params, key: params[key] + h}
        minus = {**params, key: params[key] - h}
        finite[key] = (loss(plus, x, y) - loss(minus, x, y)) / (2*h)
    return automatic, finite

def train(params, x, y, rate=0.15, steps=200):
    if steps < 1:
        raise ValueError('steps must be positive')
    def step(p, _):
        value, gradients = jax.value_and_grad(loss)(p, x, y)
        updated = jax.tree.map(lambda v, g: v - rate * g, p, gradients)
        return updated, value
    return jax.lax.scan(step, params, None, length=steps)

def report(params, x, y, rate=0.15, steps=200):
    fitted, history = train(params, x, y, rate, steps)
    heldout_x = jnp.array([-0.8, 0.2, 0.8])
    heldout_y = 2 * heldout_x + 1
    return {'parameters': {k: float(v) for k,v in fitted.items()},
            'initial_loss': float(history[0]), 'training_loss': float(loss(fitted,x,y)),
            'heldout_loss': float(loss(fitted,heldout_x,heldout_y)),
            'finite': bool(jnp.all(jnp.isfinite(history))), 'steps': steps,
            'learning_rate': rate, 'backend': jax.default_backend()}
