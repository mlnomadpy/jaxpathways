"""A one-process, four-device training experiment with a complete sampler state."""
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


def initial(size, width, seed, mesh):
    if size < 1 or width < 1:
        raise ValueError('positive dataset size and feature width required')
    key, order_key = jax.random.split(jax.random.PRNGKey(seed))
    replicated = NamedSharding(mesh, P())
    return dict(weights=jax.device_put(np.zeros(width, np.float32), replicated),
                momentum=jax.device_put(np.zeros(width, np.float32), replicated),
                key=np.asarray(key), order=np.asarray(jax.random.permutation(order_key, size)),
                cursor=np.asarray(0, np.int32), step=np.asarray(0, np.int32))


def next_batch(state, x, y, batch_size, mesh):
    """Return NEW sampler state, IDs, and padded/masked placed inputs."""
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) != len(state['order']):
        raise ValueError('dataset must match the saved sampler and scalar-target contract')
    if x.dtype != np.float32 or y.dtype != np.float32 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite float32 inputs required')
    if not isinstance(batch_size, int) or batch_size < 1 or int(state['cursor']) not in range(len(x)+1):
        raise ValueError('positive integer batch and valid cursor required')
    if mesh.size != 4:
        raise ValueError('this exercise requires exactly four devices')
    result = dict(state)
    if int(result['cursor']) == len(x):
        key, order_key = jax.random.split(jnp.asarray(result['key']))
        result.update(key=np.asarray(key), order=np.asarray(jax.random.permutation(order_key, len(x))), cursor=np.asarray(0,np.int32))
    start = int(result['cursor']); ids = result['order'][start:start+batch_size]
    size = len(ids); padded = ((size+3)//4)*4
    xb = np.zeros((padded,x.shape[1]),np.float32); yb = np.zeros(padded,np.float32)
    mask = np.zeros(padded,np.float32)
    xb[:size],yb[:size],mask[:size] = x[ids],y[ids],1
    result['cursor'] = np.asarray(start+size,np.int32)
    arrays = (jax.device_put(xb,NamedSharding(mesh,P('data',None))),
              jax.device_put(yb,NamedSharding(mesh,P('data'))),
              jax.device_put(mask,NamedSharding(mesh,P('data'))))
    return result, ids.copy(), arrays


def make_step(mesh, *, compiled=True):
    if mesh.size != 4:
        raise ValueError('exactly four devices required')
    def local(x,y,valid,w,v):
        # Sum THEN divide by global count; local means fail on uneven partitions.
        count = jax.lax.psum(jnp.sum(valid), 'data')
        residual = x@w-y
        grad = jax.lax.psum(2*x.T@(residual*valid), 'data')/count
        loss = jax.lax.psum(jnp.sum(valid*residual**2), 'data')/count
        velocity = .8*v+grad
        return w-.03*velocity, velocity, loss, grad
    mapped = jax.shard_map(local, mesh=mesh,
        in_specs=(P('data',None),P('data'),P('data'),P(),P()),
        out_specs=(P(),P(),P(),P()))
    return jax.jit(mapped) if compiled else mapped


def transition(state,x,y,batch_size,step_fn,mesh):
    result,ids,arrays = next_batch(state,x,y,batch_size,mesh)
    w,v,loss,_ = step_fn(*arrays,state['weights'],state['momentum'])
    result.update(weights=w,momentum=v,step=np.asarray(int(state['step'])+1,np.int32))
    return result, loss, ids


def contract(x,y,batch_size,seed):
    digest = hashlib.sha256()
    for a in (x,y):
        digest.update(str(a.shape).encode()); digest.update(a.dtype.str.encode()); digest.update(a.tobytes())
    return dict(schema=1, dataset_sha256=digest.hexdigest(), batch_size=batch_size, seed=seed,
                learning_rate=.03,momentum=.8,feature_width=x.shape[1],dataset_size=len(x),devices=4)


def save(path,state,metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError('use a fresh checkpoint path')
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream,**{k:np.asarray(v) for k,v in state.items()},
                 metadata=np.asarray(json.dumps(metadata,sort_keys=True)))
    temporary.replace(path)


def restore(path,expected,mesh):
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files) != {'weights','momentum','key','order','cursor','step','metadata'}:
            raise ValueError('incomplete checkpoint state')
        if json.loads(str(archive['metadata'])) != expected:
            raise ValueError('checkpoint configuration/dataset mismatch')
        state = {k:archive[k].copy() for k in archive.files if k!='metadata'}
    size,width = expected['dataset_size'],expected['feature_width']
    if (state['weights'].shape!=(width,) or state['momentum'].shape!=(width,) or
        state['weights'].dtype!=np.float32 or state['momentum'].dtype!=np.float32 or
        not np.isfinite(state['weights']).all() or not np.isfinite(state['momentum']).all() or
        state['key'].shape!=(2,) or state['key'].dtype!=np.uint32 or
        state['order'].dtype!=np.int32 or not np.array_equal(np.sort(state['order']),np.arange(size)) or
        state['cursor'].shape!=() or state['cursor'].dtype!=np.int32 or
        state['step'].shape!=() or state['step'].dtype!=np.int32 or
        not 0<=int(state['cursor'])<=size or int(state['step'])<0):
        raise ValueError('checkpoint state shape, dtype or value contract failed')
    replicated=NamedSharding(mesh,P())
    for field in ('weights','momentum'):
        state[field]=jax.device_put(state[field],replicated)
    return state
