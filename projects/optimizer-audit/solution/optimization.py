"""Small CPU optimization experiments with explicit metric and sampling contracts."""
# Import functools (partial) for this computation.
from functools import partial
from itertools import product
import numpy as np
import jax
import jax.numpy as jnp


# Function `check_xy(x, y)` implementing this stage's computation:
def check_xy(x, y):
    # Convert `(x, y)` to a host NumPy array for inspection or verification.
    x, y = np.asarray(x), np.asarray(y)
    # Guard input contract (`x.dtype != np.float32 or x.ndim != 2 or min(x.shape) < 1 or (y.dtype != np.float32) or (y.shape != (len(x),))`) and fail fast if violated.
    if x.dtype != np.float32 or x.ndim != 2 or min(x.shape) < 1 or y.dtype != np.float32 or y.shape != (len(x),):
        raise ValueError('float32 X (n,d), y (n,), n,d positive required')
    # Guard input contract (`not np.isfinite(x).all() or not np.isfinite(y).all()`) and fail fast if violated.
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite observations')
    # Return `(x, y)` to the caller.
    return x, y


# Function `check_weights(w, d)` implementing this stage's computation:
def check_weights(w, d):
    # Convert `w` to a host NumPy array for inspection or verification.
    w = np.asarray(w)
    # Guard input contract (`w.dtype != np.float32 or w.shape != (d,) or (not np.isfinite(w).all())`) and fail fast if violated.
    if w.dtype != np.float32 or w.shape != (d,) or not np.isfinite(w).all():
        raise ValueError('finite float32 weights (d,) required')
    # Return `w` to the caller.
    return w


# Function `fixture(seed, n, correlation, feature_ratio, ...)` implementing this stage's computation:
def fixture(seed=1, n=96, correlation=0.65, feature_ratio=25., noise=0.15):
    """Generated two-feature regression; disjoint seeds are held-out draws."""
    # Guard input contract (`type(n) != int or n < 4 or (not -1 < correlation < 1) or (feature_ratio <= 0) or (noise < 0)`) and fail fast if violated.
    if type(n) != int or n < 4 or not -1 < correlation < 1 or feature_ratio <= 0 or noise < 0:
        raise ValueError('invalid synthetic configuration')
    # Draw pseudorandom samples for `rng` using the explicit RNG state.
    rng = np.random.default_rng(seed)
    # Draw pseudorandom samples for `(u, v)` using the explicit RNG state.
    u, v = rng.normal(size=(2, n))
    # Cast or evaluate `x` in explicit floating-point precision.
    x = np.column_stack([u, feature_ratio * (correlation*u + np.sqrt(1-correlation**2)*v)]).astype(np.float32)
    # Convert `truth` to a host NumPy array for inspection or verification.
    truth = np.array([1.5, -1.0/feature_ratio], np.float32)
    # Cast or evaluate `y` in explicit floating-point precision.
    y = (x @ truth + noise * rng.normal(size=n)).astype(np.float32)
    # Return `(x, y, truth)` to the caller.
    return x, y, truth


# Function `objective(w, x, y, penalty)` implementing this stage's computation:
def objective(w, x, y, penalty=0.):
    """Half mean squared residual plus half penalty times squared coefficient norm."""
    # Return `0.5 * jnp.mean((x @ w - y) ** 2) + 0.5 * penalty * jnp.sum(w * w)` to the caller.
    return 0.5*jnp.mean((x @ w-y)**2) + 0.5*penalty*jnp.sum(w*w)


# Define and JIT-compile `_geometry(w, x, y, penalty)` so XLA traces and fuses the operations:
@jax.jit
# Function `_geometry(w, x, y, penalty)` implementing this stage's computation:
def _geometry(w, x, y, penalty):
    # Differentiate the objective to obtain `(value, gradient)` via automatic differentiation.
    value, gradient = jax.value_and_grad(objective)(w, x, y, penalty)
    # Compute exact directional derivative / Jacobian / Hessian (`hessian`).
    hessian = jax.hessian(objective)(w, x, y, penalty)
    # Return `(value, gradient, hessian)` to the caller.
    return value, gradient, hessian


# Function `geometry(w, x, y, penalty)` implementing this stage's computation:
def geometry(w, x, y, penalty=0.):
    # Run `check_xy` to compute `(x, y)`.
    # Run `check_weights` to compute `w`.
    x, y = check_xy(x, y)
    w = check_weights(w, x.shape[1])
    # Guard input contract (`not np.isfinite(penalty) or penalty < 0`) and fail fast if violated.
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError('nonnegative finite penalty required')
    # Cast or evaluate `(value, gradient, hessian)` in explicit floating-point precision.
    value, gradient, hessian = _geometry(w, x, y, np.float32(penalty))
    # Return `{'value': float(value), 'gradient': np.asarray(gradient), 'hessian': np.asarray(hessian)}` to the caller.
    return {'value':float(value), 'gradient':np.asarray(gradient), 'hessian':np.asarray(hessian)}


# Function `fit_scales(x)` implementing this stage's computation:
def fit_scales(x):
    """Training-only column RMS, no centering; zero columns retain scale one."""
    # Convert `x` to a host NumPy array for inspection or verification.
    x = np.asarray(x)
    # Allocate initialized array `` with the specified shape and dtype.
    check_xy(x, np.zeros(len(x), np.float32))
    # Reduce along axis=0 to compute `rms`.
    rms = np.sqrt(np.mean(x.astype(np.float64)**2, axis=0))
    # Return `np.where(rms > 0, rms, 1).astype(np.float32)` to the caller.
    return np.where(rms > 0, rms, 1).astype(np.float32)


# Function `apply_scales(x, scales)` implementing this stage's computation:
def apply_scales(x, scales):
    # Convert `x` to a host NumPy array for inspection or verification.
    # Convert `scales` to a host NumPy array for inspection or verification.
    x = np.asarray(x)
    scales = np.asarray(scales)
    # Allocate initialized array `` with the specified shape and dtype.
    check_xy(x, np.zeros(len(x), np.float32))
    # Guard input contract (`scales.dtype != np.float32 or scales.shape != (x.shape[1],) or (not np.isfinite(scales).all()) or np.any(scales <= 0)`) and fail fast if violated.
    if scales.dtype != np.float32 or scales.shape != (x.shape[1],) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('positive finite float32 feature scales required')
    # Return `(x / scales).astype(np.float32)` to the caller.
    return (x/scales).astype(np.float32)


# Define `noise_audit(w, x, y, batch_size...)` to evaluate the objective and its automatic derivatives:
def noise_audit(w, x, y, batch_size=2, penalty=0.):
    """Exactly enumerate ordered iid draws WITH replacement; never Monte Carlo here."""
    # Run `check_xy` to compute `(x, y)`.
    # Run `check_weights` to compute `w`.
    x, y = check_xy(x, y)
    w = check_weights(w, x.shape[1])
    # Guard input contract (`type(batch_size) != int or batch_size < 1 or len(x) ** batch_size > 65536 or (not np.isfinite(penalty)) or (penalty < 0)`) and fail fast if violated.
    if type(batch_size) != int or batch_size < 1 or len(x)**batch_size > 65536 or not np.isfinite(penalty) or penalty < 0:
        raise ValueError('invalid or excessive enumeration')
    # Differentiate the objective to obtain `per` via automatic differentiation.
    per = jax.vmap(jax.grad(lambda weights, row, target: objective(weights, row[None], target[None], penalty)), in_axes=(None,0,0))(w,x,y)
    # Convert `draws` to a host NumPy array for inspection or verification.
    draws = np.asarray(list(product(range(len(x)), repeat=batch_size)), np.int32)
    # Convert `gradients` to a host NumPy array for inspection or verification.
    gradients = np.asarray(per)[draws].mean(axis=1)
    # Reduce across the target axis to summarize `mean`.
    mean = gradients.astype(np.float64).mean(axis=0)
    # Compute `centered` as `gradients-mean`.
    centered = gradients-mean
    # Return `{'draws': draws, 'gradients': gradients, 'mean': mean, 'covariance': centered.T @ centered / len(draws), 'per_example': np.asarray(per)}` to the caller.
    return {'draws':draws,'gradients':gradients,'mean':mean,
            'covariance':centered.T@centered/len(draws),'per_example':np.asarray(per)}


# Function `batch_plan(seed, n, steps, batch_size)` implementing this stage's computation:
def batch_plan(seed, n, steps, batch_size):
    # Guard input contract (`any((type(v) != int or v < 1 for v in [n, steps, batch_size]))`) and fail fast if violated.
    if any(type(v) != int or v < 1 for v in [n,steps,batch_size]):
        raise ValueError('positive integer plan sizes required')
    # Return `np.random.default_rng(seed).integers(0, n, size=(steps, batch_size), dtype=np.int32)` to the caller.
    return np.random.default_rng(seed).integers(0,n,size=(steps,batch_size),dtype=np.int32)


# Function `rates(initial, steps, decay_at, factor)` implementing this stage's computation:
def rates(initial, steps, decay_at=None, factor=0.1):
    # Guard input contract (`type(steps) != int or steps < 1 or (not np.isfinite(initial)) or (initial <= 0) or (not np.isfinite(factor)) or (not 0 < factor <= 1)`) and fail fast if violated.
    if type(steps) != int or steps < 1 or not np.isfinite(initial) or initial <= 0 or not np.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError('invalid schedule')
    # Guard input contract (`decay_at is not None and (type(decay_at) != int or not 0 <= decay_at < steps)`) and fail fast if violated.
    if decay_at is not None and (type(decay_at) != int or not 0 <= decay_at < steps):
        raise ValueError('decay_at is a zero-based update index')
    # Cast or evaluate `values` in explicit floating-point precision.
    values = np.full(steps,initial,np.float32)
    # Branch on condition `decay_at is not None`:
    if decay_at is not None: values[decay_at:] *= np.float32(factor)
    # Guard input contract (`not np.isfinite(values).all() or np.any(values <= 0)`) and fail fast if violated.
    if not np.isfinite(values).all() or np.any(values<=0):
        raise ValueError('schedule outside float32 range')
    # Return `values` to the caller.
    return values


# Function `optimizer_step(state, gradient, rate, method, ...)` implementing this stage's computation:
def optimizer_step(state, gradient, rate, method='gd', clip_norm=jnp.inf):
    """Clip raw gradient first, then momentum/Adam, then learning-rate multiplication."""
    # Run `jnp.linalg.norm` to compute `norm`.
    norm = jnp.linalg.norm(gradient)
    # Reduce across the target axis to summarize `clipped`.
    clipped = gradient*jnp.minimum(1.,clip_norm/jnp.maximum(norm,1e-12))
    # Compute `count` as `state['step']+1`.
    count = state['step']+1
    # Branch on condition `method == 'gd'`:
    if method == 'gd':
        first, second, direction = state['first'], state['second'], clipped
    elif method == 'momentum':
        first = .85*state['first']+clipped
        second, direction = state['second'], first
    elif method == 'adam':
        first = .9*state['first']+.1*clipped
        second = .99*state['second']+.01*clipped**2
        direction = (first/(1-.9**count))/(jnp.sqrt(second/(1-.99**count))+1e-8)
    else:
        raise ValueError('unknown optimizer')
    # Construct dictionary `new` with the structured fields for this stage.
    new = {'weights':state['weights']-rate*direction,'first':first,'second':second,'step':count}
    # Return `(new, {'raw_norm': norm, 'clipped_norm': jnp.linalg.norm(clipped), 'update_norm': jnp.linalg.norm(rate * direction)})` to the caller.
    return new, {'raw_norm':norm,'clipped_norm':jnp.linalg.norm(clipped),'update_norm':jnp.linalg.norm(rate*direction)}


@partial(jax.jit, static_argnames=['method'])
# Function `_run(w, x, y, held_x, ...)` implementing this stage's computation:
def _run(w, x, y, held_x, held_y, indices, learning_rates, penalty, clip_norm, method):
    # Allocate initialized array `state` with the specified shape and dtype.
    state = {'weights':w,'first':jnp.zeros_like(w),'second':jnp.zeros_like(w),'step':jnp.int32(0)}
    # Function `advance(state, inputs)` implementing this stage's computation:
    def advance(state, inputs):
        # Compute `ids, rate` as `inputs`.
        ids, rate = inputs
        # Evaluate both scalar loss and parameter gradients in one pass (`(value, gradient)`).
        value, gradient = jax.value_and_grad(objective)(state['weights'],x[ids],y[ids],penalty)
        # Combine or mask array elements to form `(updated, diagnostics)`.
        updated, diagnostics = optimizer_step(state,gradient,rate,method,clip_norm)
        # Compute `weights` as `updated['weights']`.
        weights = updated['weights']
        # Perform matrix contraction / projection to compute `observed`.
        observed = {'weights':weights,'batch_objective_before':value,
                    'train_mse':jnp.mean((x@weights-y)**2),
                    'held_mse':jnp.mean((held_x@weights-held_y)**2),**diagnostics}
        # Return `(updated, observed)` to the caller.
        return updated, observed
    # Return `jax.lax.scan(advance, state, (indices, learning_rates))` to the caller.
    return jax.lax.scan(advance,state,(indices,learning_rates))


# Function `run(w, x, y, held_x, ...)` implementing this stage's computation:
def run(w, x, y, held_x, held_y, indices, learning_rates, method='gd', penalty=0., clip_norm=None):
    # Run `check_xy` to compute `(x, y)`.
    # Run `check_xy` to compute `(held_x, held_y)`.
    # Run `check_weights` to compute `w`.
    x,y=check_xy(x,y)
    held_x,held_y=check_xy(held_x,held_y)
    w=check_weights(w,x.shape[1])
    # Convert `ids` to a host NumPy array for inspection or verification.
    # Convert `lr` to a host NumPy array for inspection or verification.
    ids=np.asarray(indices)
    lr=np.asarray(learning_rates)
    # Guard input contract (`held_x.shape[1] != x.shape[1] or ids.dtype != np.int32 or ids.ndim != 2 or (min(ids.shape) < 1) or np.any((ids < 0) | (ids >= len(x)))`) and fail fast if violated.
    if held_x.shape[1]!=x.shape[1] or ids.dtype!=np.int32 or ids.ndim!=2 or min(ids.shape)<1 or np.any((ids<0)|(ids>=len(x))):
        raise ValueError('held feature width or sample plan invalid')
    # Guard input contract (`lr.dtype != np.float32 or lr.shape != (len(ids),) or (not np.isfinite(lr).all()) or np.any(lr <= 0)`) and fail fast if violated.
    if lr.dtype!=np.float32 or lr.shape!=(len(ids),) or not np.isfinite(lr).all() or np.any(lr<=0):
        raise ValueError('one positive finite float32 rate per update required')
    # Guard input contract (`method not in ['gd', 'momentum', 'adam'] or not np.isfinite(penalty) or penalty < 0`) and fail fast if violated.
    if method not in ['gd','momentum','adam'] or not np.isfinite(penalty) or penalty<0:
        raise ValueError('invalid optimizer or penalty')
    # Guard input contract (`clip_norm is not None and (not np.isfinite(clip_norm) or clip_norm <= 0)`) and fail fast if violated.
    if clip_norm is not None and (not np.isfinite(clip_norm) or clip_norm<=0):
        raise ValueError('positive finite clipping threshold required')
    # Cast or evaluate `(final, trace)` in explicit floating-point precision.
    final, trace=_run(w,x,y,held_x,held_y,ids,lr,np.float32(penalty),jnp.float32(jnp.inf if clip_norm is None else clip_norm),method)
    # Synchronize host execution until asynchronous device computation completes.
    jax.block_until_ready(final)
    # Return `(jax.tree.map(np.asarray, final), jax.tree.map(np.asarray, trace))` to the caller.
    return jax.tree.map(np.asarray, final),jax.tree.map(np.asarray,trace)


# Function `ridge_solution(x, y, penalty)` implementing this stage's computation:
def ridge_solution(x,y,penalty=0.):
    """Host float64 augmented least squares avoids forming a squared-condition system."""
    # Run `check_xy` to compute `(x, y)`.
    x,y=check_xy(x,y)
    # Guard input contract (`not np.isfinite(penalty) or penalty < 0`) and fail fast if violated.
    if not np.isfinite(penalty) or penalty<0:raise ValueError('nonnegative finite penalty required')
    # Construct an identity matrix `a`.
    a=np.vstack([x.astype(np.float64)/np.sqrt(len(x)),np.sqrt(penalty)*np.eye(x.shape[1])])
    # Allocate initialized array `b` with the specified shape and dtype.
    b=np.concatenate([y.astype(np.float64)/np.sqrt(len(x)),np.zeros(x.shape[1])])
    # Return `np.linalg.lstsq(a, b, rcond=None)[0]` to the caller.
    return np.linalg.lstsq(a,b,rcond=None)[0]
