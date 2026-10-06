"""Four connected foundations contracts, before the optimization capstone."""
import platform
from functools import partial
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

def environment_report():
    x=jnp.arange(6,dtype=jnp.float32).reshape(2,3)
    y=jax.jit(lambda a:jnp.sum(a,axis=1))(x)
    y.block_until_ready()
    return {'python':platform.python_version(),'jax':jax.__version__,'numpy':np.__version__,
            'backend':jax.default_backend(),'devices':[str(d) for d in y.devices()],
            'input_shape':list(x.shape),'output_shape':list(y.shape),'dtype':str(y.dtype),
            'row_sums':np.asarray(y).tolist()}

def normalize_columns(x,valid):
    """Fit masked column statistics; preserve all rows and zero invalid outputs."""
    x=np.asarray(x);valid=np.asarray(valid)
    if x.dtype!=np.float32 or x.ndim!=2 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('finite float32 observation-by-feature matrix required')
    if valid.dtype!=np.bool_ or valid.shape!=(len(x),) or not np.any(valid):
        raise ValueError('one boolean per row with at least one valid row required')
    data=jnp.asarray(x);mask=jnp.asarray(valid)
    count=jnp.sum(mask)
    mean=jnp.sum(jnp.where(mask[:,None],data,0),axis=0)/count
    variance=jnp.sum(jnp.where(mask[:,None],(data-mean)**2,0),axis=0)/count
    scale=jnp.where(variance>0,jnp.sqrt(variance),1.)
    normalized=jnp.where(mask[:,None],(data-mean)/scale,0.)
    return {'values':normalized,'mean':mean,'scale':scale,'count':count}

def per_example_loss(w,x,y):
    residual=w[0]*x+w[1]-y
    return .5*residual**2+.1*jnp.sum(w**2)

_gradients=jax.jit(jax.vmap(jax.grad(per_example_loss),in_axes=(None,0,0)))

def batch_gradients(w,x,y):
    w=np.asarray(w);x=np.asarray(x);y=np.asarray(y)
    if w.shape!=(2,) or x.ndim!=1 or not len(x) or y.shape!=x.shape:
        raise ValueError('weights (2,), observations/targets matching nonempty vectors required')
    if any(a.dtype!=np.float32 or not np.isfinite(a).all() for a in [w,x,y]):
        raise ValueError('finite float32 inputs required')
    return _gradients(jnp.asarray(w),jnp.asarray(x),jnp.asarray(y))

def initial_state(seed=0,batch=3):
    if type(batch)!=int or batch<1:raise ValueError('positive particle count required')
    return {'position':jnp.zeros((batch,2),jnp.float32),'velocity':jnp.zeros((batch,2),jnp.float32),
            'key':jax.random.PRNGKey(seed),'step':jnp.int32(0)}

def transition(state,dt=.05):
    key,noise_key=jax.random.split(state['key'])
    noise=jax.random.normal(noise_key,state['position'].shape)
    velocity=.9*state['velocity']+.1*noise
    position=state['position']+dt*velocity
    new={'position':position,'velocity':velocity,'key':key,'step':state['step']+1}
    return new,position

@partial(jax.jit,static_argnames=['steps'])
def _scan(state,steps,dt):
    return jax.lax.scan(lambda carry,_:transition(carry,dt),state,None,length=steps)

def validate_state(state):
    if set(state)!={'position','velocity','key','step'}:raise ValueError('state fields changed')
    p=np.asarray(state['position']);v=np.asarray(state['velocity']);k=np.asarray(state['key']);s=np.asarray(state['step'])
    if p.ndim!=2 or p.shape[1]!=2 or not len(p) or v.shape!=p.shape or p.dtype!=np.float32 or v.dtype!=np.float32:
        raise ValueError('position/velocity require matching float32 particle-by-coordinate arrays')
    if not np.isfinite(p).all() or not np.isfinite(v).all() or k.shape!=(2,) or k.dtype!=np.uint32 or s.shape!=() or s.dtype!=np.int32 or int(s)<0:
        raise ValueError('nonfinite state or invalid key/step')

def simulate(state,steps,dt=.05):
    validate_state(state)
    if type(steps)!=int or steps<1 or not np.isfinite(dt) or dt<=0:raise ValueError('positive step count and time increment required')
    return _scan(state,steps,jnp.float32(dt))

def save_state(path,state):
    validate_state(state)
    with Path(path).open('xb') as stream:np.savez(stream,**{k:np.asarray(v) for k,v in state.items()})

def load_state(path):
    with np.load(path,allow_pickle=False) as arrays:state={k:arrays[k] for k in arrays.files}
    # Validate stored dtypes before JAX can downcast or wrap a malformed counter.
    validate_state(state)
    return {k:jnp.asarray(v) for k,v in state.items()}
