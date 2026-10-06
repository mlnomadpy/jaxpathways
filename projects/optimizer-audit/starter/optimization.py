"""Small CPU optimization experiments with explicit metric and sampling contracts."""
from functools import partial
from itertools import product
import numpy as np
import jax
import jax.numpy as jnp


def check_xy(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.dtype != np.float32 or x.ndim != 2 or min(x.shape) < 1 or y.dtype != np.float32 or y.shape != (len(x),):
        raise ValueError('float32 X (n,d), y (n,), n,d positive required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite observations')
    return x, y


def check_weights(w, d):
    w = np.asarray(w)
    if w.dtype != np.float32 or w.shape != (d,) or not np.isfinite(w).all():
        raise ValueError('finite float32 weights (d,) required')
    return w


def fixture(seed=1, n=96, correlation=0.65, feature_ratio=25., noise=0.15):
    """Generated two-feature regression; disjoint seeds are held-out draws."""
    if type(n) != int or n < 4 or not -1 < correlation < 1 or feature_ratio <= 0 or noise < 0:
        raise ValueError('invalid synthetic configuration')
    rng = np.random.default_rng(seed)
    u, v = rng.normal(size=(2, n))
    x = np.column_stack([u, feature_ratio * (correlation*u + np.sqrt(1-correlation**2)*v)]).astype(np.float32)
    truth = np.array([1.5, -1.0/feature_ratio], np.float32)
    y = (x @ truth + noise * rng.normal(size=n)).astype(np.float32)
    return x, y, truth


def objective(w, x, y, penalty=0.):
    """TODO — Stage 1: implement half mean squared error plus half the declared L2 penalty."""
    raise NotImplementedError('Stage 1: implement half mean squared error plus half the declared L2 penalty.')


@jax.jit
def _geometry(w, x, y, penalty):
    value, gradient = jax.value_and_grad(objective)(w, x, y, penalty)
    hessian = jax.hessian(objective)(w, x, y, penalty)
    return value, gradient, hessian


def geometry(w, x, y, penalty=0.):
    x, y = check_xy(x, y); w = check_weights(w, x.shape[1])
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError('nonnegative finite penalty required')
    value, gradient, hessian = _geometry(w, x, y, np.float32(penalty))
    return {'value':float(value), 'gradient':np.asarray(gradient), 'hessian':np.asarray(hessian)}


def fit_scales(x):
    """TODO — Stage 2: fit column RMS on training inputs; protect a completely zero column."""
    raise NotImplementedError('Stage 2: fit column RMS on training inputs; protect a completely zero column.')


def apply_scales(x, scales):
    x = np.asarray(x); scales = np.asarray(scales)
    check_xy(x, np.zeros(len(x), np.float32))
    if scales.dtype != np.float32 or scales.shape != (x.shape[1],) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('positive finite float32 feature scales required')
    return (x/scales).astype(np.float32)


def noise_audit(w, x, y, batch_size=2, penalty=0.):
    """TODO — Stage 3: enumerate ordered batches with replacement and return exact gradient moments."""
    raise NotImplementedError('Stage 3: enumerate ordered batches with replacement and return exact gradient moments.')


def batch_plan(seed, n, steps, batch_size):
    if any(type(v) != int or v < 1 for v in [n,steps,batch_size]):
        raise ValueError('positive integer plan sizes required')
    return np.random.default_rng(seed).integers(0,n,size=(steps,batch_size),dtype=np.int32)


def rates(initial, steps, decay_at=None, factor=0.1):
    if type(steps) != int or steps < 1 or not np.isfinite(initial) or initial <= 0 or not np.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError('invalid schedule')
    if decay_at is not None and (type(decay_at) != int or not 0 <= decay_at < steps):
        raise ValueError('decay_at is a zero-based update index')
    values = np.full(steps,initial,np.float32)
    if decay_at is not None: values[decay_at:] *= np.float32(factor)
    if not np.isfinite(values).all() or np.any(values<=0): raise ValueError('schedule outside float32 range')
    return values


def optimizer_step(state, gradient, rate, method='gd', clip_norm=jnp.inf):
    """TODO — Stage 4: clip the raw gradient, then update the selected optimizer state and return diagnostics."""
    raise NotImplementedError('Stage 4: clip the raw gradient, then update the selected optimizer state and return diagnostics.')


@partial(jax.jit, static_argnames=['method'])
def _run(w, x, y, held_x, held_y, indices, learning_rates, penalty, clip_norm, method):
    state = {'weights':w,'first':jnp.zeros_like(w),'second':jnp.zeros_like(w),'step':jnp.int32(0)}
    def advance(state, inputs):
        ids, rate = inputs
        value, gradient = jax.value_and_grad(objective)(state['weights'],x[ids],y[ids],penalty)
        updated, diagnostics = optimizer_step(state,gradient,rate,method,clip_norm)
        weights = updated['weights']
        observed = {'weights':weights,'batch_objective_before':value,
                    'train_mse':jnp.mean((x@weights-y)**2),
                    'held_mse':jnp.mean((held_x@weights-held_y)**2),**diagnostics}
        return updated, observed
    return jax.lax.scan(advance,state,(indices,learning_rates))


def run(w, x, y, held_x, held_y, indices, learning_rates, method='gd', penalty=0., clip_norm=None):
    x,y=check_xy(x,y); held_x,held_y=check_xy(held_x,held_y); w=check_weights(w,x.shape[1])
    ids=np.asarray(indices); lr=np.asarray(learning_rates)
    if held_x.shape[1]!=x.shape[1] or ids.dtype!=np.int32 or ids.ndim!=2 or min(ids.shape)<1 or np.any((ids<0)|(ids>=len(x))):
        raise ValueError('held feature width or sample plan invalid')
    if lr.dtype!=np.float32 or lr.shape!=(len(ids),) or not np.isfinite(lr).all() or np.any(lr<=0):
        raise ValueError('one positive finite float32 rate per update required')
    if method not in ['gd','momentum','adam'] or not np.isfinite(penalty) or penalty<0:
        raise ValueError('invalid optimizer or penalty')
    if clip_norm is not None and (not np.isfinite(clip_norm) or clip_norm<=0): raise ValueError('positive finite clipping threshold required')
    final, trace=_run(w,x,y,held_x,held_y,ids,lr,np.float32(penalty),jnp.float32(jnp.inf if clip_norm is None else clip_norm),method)
    jax.block_until_ready(final)
    return jax.tree.map(np.asarray, final),jax.tree.map(np.asarray,trace)


def ridge_solution(x,y,penalty=0.):
    """TODO — Stage 5: solve the augmented least-squares problem, including rank-deficient zero penalty."""
    raise NotImplementedError('Stage 5: solve the augmented least-squares problem, including rank-deficient zero penalty.')
