"""Four connected foundations contracts, before the optimization capstone."""
# Import platform for this computation.
import platform
from functools import partial
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

# Function `environment_report()` implementing this stage's computation:
def environment_report():
    # Construct and reshape `x` into the target tensor dimensions.
    x=jnp.arange(6,dtype=jnp.float32).reshape(2,3)
    # Wrap with `jax.jit` (`y`) so XLA traces and compiles the function.
    y=jax.jit(lambda a:jnp.sum(a,axis=1))(x)
    # Synchronize host execution until asynchronous device computation completes.
    y.block_until_ready()
    # Return `{'python': platform.python_version(), 'jax': jax.__version__, 'numpy': np.__version__, 'backend': jax.default_backend(), 'devices': [str(d) for d in y.devices()], 'input_shape': list(x.shape), 'output_shape': list(y.shape), 'dtype': str(y.dtype), 'row_sums': np.asarray(y).tolist()}` to the caller.
    return {'python':platform.python_version(),'jax':jax.__version__,'numpy':np.__version__,
            'backend':jax.default_backend(),'devices':[str(d) for d in y.devices()],
            'input_shape':list(x.shape),'output_shape':list(y.shape),'dtype':str(y.dtype),
            'row_sums':np.asarray(y).tolist()}

# Function `normalize_columns(x, valid)` implementing this stage's computation:
def normalize_columns(x,valid):
    """Fit masked column statistics; preserve all rows and zero invalid outputs."""
    # Convert `x` to a host NumPy array for inspection or verification.
    # Convert `valid` to a host NumPy array for inspection or verification.
    x=np.asarray(x)
    valid=np.asarray(valid)
    # Guard input contract (`x.dtype != np.float32 or x.ndim != 2 or (not x.shape[1]) or (not np.isfinite(x).all())`) and fail fast if violated.
    if x.dtype!=np.float32 or x.ndim!=2 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('finite float32 observation-by-feature matrix required')
    # Guard input contract (`valid.dtype != np.bool_ or valid.shape != (len(x),) or (not np.any(valid))`) and fail fast if violated.
    if valid.dtype!=np.bool_ or valid.shape!=(len(x),) or not np.any(valid):
        raise ValueError('one boolean per row with at least one valid row required')
    # Create device-backed JAX array `data`.
    # Create device-backed JAX array `mask`.
    data=jnp.asarray(x)
    mask=jnp.asarray(valid)
    # Aggregate array values to compute `count`.
    count=jnp.sum(mask)
    # Reduce across the target axis to summarize `mean`.
    mean=jnp.sum(jnp.where(mask[:,None],data,0),axis=0)/count
    # Reduce across the target axis to summarize `variance`.
    variance=jnp.sum(jnp.where(mask[:,None],(data-mean)**2,0),axis=0)/count
    # Combine or mask array elements to form `scale`.
    scale=jnp.where(variance>0,jnp.sqrt(variance),1.)
    # Combine or mask array elements to form `normalized`.
    normalized=jnp.where(mask[:,None],(data-mean)/scale,0.)
    # Return `{'values': normalized, 'mean': mean, 'scale': scale, 'count': count}` to the caller.
    return {'values':normalized,'mean':mean,'scale':scale,'count':count}

# Function `per_example_loss(w, x, y)` implementing this stage's computation:
def per_example_loss(w,x,y):
    # Compute `residual` as `w[0]*x+w[1]-y`.
    residual=w[0]*x+w[1]-y
    # Return `0.5 * residual ** 2 + 0.1 * jnp.sum(w ** 2)` to the caller.
    return .5*residual**2+.1*jnp.sum(w**2)

# Differentiate the objective to obtain `_gradients` via automatic differentiation.
_gradients=jax.jit(jax.vmap(jax.grad(per_example_loss),in_axes=(None,0,0)))

# Define `batch_gradients(w, x, y)` to evaluate the objective and its automatic derivatives:
def batch_gradients(w,x,y):
    # Convert `w` to a host NumPy array for inspection or verification.
    # Convert `x` to a host NumPy array for inspection or verification.
    # Convert `y` to a host NumPy array for inspection or verification.
    w=np.asarray(w)
    x=np.asarray(x)
    y=np.asarray(y)
    # Guard input contract (`w.shape != (2,) or x.ndim != 1 or (not len(x)) or (y.shape != x.shape)`) and fail fast if violated.
    if w.shape!=(2,) or x.ndim!=1 or not len(x) or y.shape!=x.shape:
        raise ValueError('weights (2,), observations/targets matching nonempty vectors required')
    # Guard input contract (`any((a.dtype != np.float32 or not np.isfinite(a).all() for a in [w, x, y]))`) and fail fast if violated.
    if any(a.dtype!=np.float32 or not np.isfinite(a).all() for a in [w,x,y]):
        raise ValueError('finite float32 inputs required')
    # Return `_gradients(jnp.asarray(w), jnp.asarray(x), jnp.asarray(y))` to the caller.
    return _gradients(jnp.asarray(w),jnp.asarray(x),jnp.asarray(y))

# Function `initial_state(seed, batch)` implementing this stage's computation:
def initial_state(seed=0,batch=3):
    # Guard input contract (`type(batch) != int or batch < 1`) and fail fast if violated.
    if type(batch)!=int or batch<1:raise ValueError('positive particle count required')
    # Return `{'position': jnp.zeros((batch, 2), jnp.float32), 'velocity': jnp.zeros((batch, 2), jnp.float32), 'key': jax.random.PRNGKey(seed), 'step': jnp.int32(0)}` to the caller.
    return {'position':jnp.zeros((batch,2),jnp.float32),'velocity':jnp.zeros((batch,2),jnp.float32),
            'key':jax.random.PRNGKey(seed),'step':jnp.int32(0)}

# Function `transition(state, dt)` implementing this stage's computation:
def transition(state,dt=.05):
    # Create or split explicit PRNG key(s) (`(key, noise_key)`) for reproducible randomness.
    key,noise_key=jax.random.split(state['key'])
    # Sample deterministic random values into `noise` using an explicit PRNG key.
    noise=jax.random.normal(noise_key,state['position'].shape)
    # Compute `velocity` as `.9*state['velocity']+.1*noise`.
    velocity=.9*state['velocity']+.1*noise
    # Compute `position` as `state['position']+dt*velocity`.
    position=state['position']+dt*velocity
    # Construct dictionary `new` with the structured fields for this stage.
    new={'position':position,'velocity':velocity,'key':key,'step':state['step']+1}
    # Return `(new, position)` to the caller.
    return new,position

# Define and JIT-compile `_scan(state, steps, dt)` so XLA traces and fuses the operations:
@partial(jax.jit,static_argnames=['steps'])
# Function `_scan(state, steps, dt)` implementing this stage's computation:
def _scan(state,steps,dt):
    # Return `jax.lax.scan(lambda carry, _: transition(carry, dt), state, None, length=steps)` to the caller.
    return jax.lax.scan(lambda carry,_:transition(carry,dt),state,None,length=steps)

# Function `validate_state(state)` implementing this stage's computation:
def validate_state(state):
    # Guard input contract (`set(state) != {'position', 'velocity', 'key', 'step'}`) and fail fast if violated.
    if set(state)!={'position','velocity','key','step'}:raise ValueError('state fields changed')
    # Convert `p` to a host NumPy array for inspection or verification.
    # Convert `v` to a host NumPy array for inspection or verification.
    # Convert `k` to a host NumPy array for inspection or verification.
    # Convert `s` to a host NumPy array for inspection or verification.
    p=np.asarray(state['position'])
    v=np.asarray(state['velocity'])
    k=np.asarray(state['key'])
    s=np.asarray(state['step'])
    # Guard input contract (`p.ndim != 2 or p.shape[1] != 2 or (not len(p)) or (v.shape != p.shape) or (p.dtype != np.float32) or (v.dtype != np.float32)`) and fail fast if violated.
    if p.ndim!=2 or p.shape[1]!=2 or not len(p) or v.shape!=p.shape or p.dtype!=np.float32 or v.dtype!=np.float32:
        raise ValueError('position/velocity require matching float32 particle-by-coordinate arrays')
    # Guard input contract (`not np.isfinite(p).all() or not np.isfinite(v).all() or k.shape != (2,) or (k.dtype != np.uint32) or (s.shape != ()) or (s.dtype != np.int32) or (int(s) < 0)`) and fail fast if violated.
    if not np.isfinite(p).all() or not np.isfinite(v).all() or k.shape!=(2,) or k.dtype!=np.uint32 or s.shape!=() or s.dtype!=np.int32 or int(s)<0:
        raise ValueError('nonfinite state or invalid key/step')

# Function `simulate(state, steps, dt)` implementing this stage's computation:
def simulate(state,steps,dt=.05):
    # Run `validate_state` to perform the next check or state transition.
    validate_state(state)
    # Guard input contract (`type(steps) != int or steps < 1 or (not np.isfinite(dt)) or (dt <= 0)`) and fail fast if violated.
    if type(steps)!=int or steps<1 or not np.isfinite(dt) or dt<=0:raise ValueError('positive step count and time increment required')
    # Return `_scan(state, steps, jnp.float32(dt))` to the caller.
    return _scan(state,steps,jnp.float32(dt))

# Function `save_state(path, state)` implementing this stage's computation:
def save_state(path,state):
    # Run `validate_state` to perform the next check or state transition.
    validate_state(state)
    # Enter managed runtime/context scope for this block:
    # Convert `` to a host NumPy array for inspection or verification.
    with Path(path).open('xb') as stream:np.savez(stream,**{k:np.asarray(v) for k,v in state.items()})

# Function `load_state(path)` implementing this stage's computation:
def load_state(path):
    # Enter managed runtime/context scope for this block:
    # Execute `with np.load(path,allow_pickle=False) as arrays:state={k:arrays[k] for k in arra`.
    with np.load(path,allow_pickle=False) as arrays:state={k:arrays[k] for k in arrays.files}
    # Validate stored dtypes before JAX can downcast or wrap a malformed counter.
    validate_state(state)
    # Return `{k: jnp.asarray(v) for k, v in state.items()}` to the caller.
    return {k:jnp.asarray(v) for k,v in state.items()}
