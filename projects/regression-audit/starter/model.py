"""Implement one stage at a time. Reference answers are in ../solution/model.py."""
def predict(params, x):
    # Step 1: Return `params['weight'] * x + params['bias']` to the caller.
    raise NotImplementedError('Stage 1: implement a linear prediction')

def loss(params, x, y):
    # Key APIs to use: `contract`, `jnp.mean`, `predict`
    # Step 1: Guard input contract (`x.shape != y.shape`) and fail fast if violated.
    # Step 2: Return `jnp.mean((predict(params, x) - y) ** 2)` to the caller.
    raise NotImplementedError('Stage 1: implement scalar MSE with shape checks')

def gradient_check(params, x, y, h=1e-2):
    # Key APIs to use: `jax.grad`, `loss`
    # Step 1: Differentiate the objective to obtain gradients `automatic`.
    # Step 2: Evaluate `finite` from the current inputs and state.
    # Step 3: Loop over `key` in `params`:
    # Step 4: Inside block: Evaluate `plus` from the current inputs and state.
    # Step 5: Inside block: Evaluate `minus` from the current inputs and state.
    # Step 6: Return `(automatic, finite)` to the caller.
    raise NotImplementedError('Stage 2: compare autodiff with independent differences')

def train(params, x, y, rate=0.15, steps=200):
    # Key APIs to use: `contract`, `step`, `jax.value_and_grad`, `PyTree`, `tree.map`
    # Step 1: Guard input contract (`steps < 1`) and fail fast if violated.
    # Step 2: Function `step(p, _)` implementing this stage's computation:
    # Step 3: Return `jax.lax.scan(step, params, None, length=steps)` to the caller.
    raise NotImplementedError('Stage 3: carry parameters through a compiled loop')

def report(params, x, y, rate=0.15, steps=200):
    # Key APIs to use: `train`, `jnp.array`, `fitted.items`, `loss`, `jnp.all`
    # Step 1: Run `train` to compute `(fitted, history)`.
    # Step 2: Create device-backed JAX array `heldout_x`.
    # Step 3: Evaluate `heldout_y` from the current inputs and state.
    # Step 4: Return `{'parameters': {k: float(v) for k, v in fitted.items()}, 'initial_loss': float(history[0]), 'training_loss': float(loss(fitted, x, y)), 'heldout_loss': float(loss(fitted, heldout_x, heldout_y)), 'finite': bool(jnp.all(jnp.isfinite(history))), 'steps': steps, 'learning_rate': rate, 'backend': jax.default_backend()}` to the caller.
    raise NotImplementedError('Stage 4: report fit, held-out error, and numerical health')
