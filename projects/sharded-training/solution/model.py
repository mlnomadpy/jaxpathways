"""A one-process, four-device training experiment with a complete sampler state."""
# Import hashlib for this computation.
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


# Function `initial(size, width, seed, mesh)` implementing this stage's computation:
def initial(size, width, seed, mesh):
    # Guard input contract (`size < 1 or width < 1`) and fail fast if violated.
    if size < 1 or width < 1:
        raise ValueError('positive dataset size and feature width required')
    # Create or split explicit PRNG key(s) (`(key, order_key)`) for reproducible randomness.
    key, order_key = jax.random.split(jax.random.PRNGKey(seed))
    # Configure multi-device placement / sharding specification (`replicated`).
    replicated = NamedSharding(mesh, P())
    # Return `dict(weights=jax.device_put(np.zeros(width, np.float32), replicated), momentum=jax.device_put(np.zeros(width, np.float32), replicated), key=np.asarray(key), order=np.asarray(jax.random.permutation(order_key, size)), cursor=np.asarray(0, np.int32), step=np.asarray(0, np.int32))` to the caller.
    return dict(weights=jax.device_put(np.zeros(width, np.float32), replicated),
                momentum=jax.device_put(np.zeros(width, np.float32), replicated),
                key=np.asarray(key), order=np.asarray(jax.random.permutation(order_key, size)),
                cursor=np.asarray(0, np.int32), step=np.asarray(0, np.int32))


# Function `next_batch(state, x, y, batch_size, ...)` implementing this stage's computation:
def next_batch(state, x, y, batch_size, mesh):
    """Return NEW sampler state, IDs, and padded/masked placed inputs."""
    # Convert `(x, y)` to a host NumPy array for inspection or verification.
    x, y = np.asarray(x), np.asarray(y)
    # Guard input contract (`x.ndim != 2 or y.shape != (len(x),) or len(x) != len(state['order'])`) and fail fast if violated.
    if x.ndim != 2 or y.shape != (len(x),) or len(x) != len(state['order']):
        raise ValueError('dataset must match the saved sampler and scalar-target contract')
    # Guard input contract (`x.dtype != np.float32 or y.dtype != np.float32 or (not np.isfinite(x).all()) or (not np.isfinite(y).all())`) and fail fast if violated.
    if x.dtype != np.float32 or y.dtype != np.float32 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite float32 inputs required')
    # Guard input contract (`not isinstance(batch_size, int) or batch_size < 1 or int(state['cursor']) not in range(len(x) + 1)`) and fail fast if violated.
    if not isinstance(batch_size, int) or batch_size < 1 or int(state['cursor']) not in range(len(x)+1):
        raise ValueError('positive integer batch and valid cursor required')
    # Guard input contract (`mesh.size != 4`) and fail fast if violated.
    if mesh.size != 4:
        raise ValueError('this exercise requires exactly four devices')
    # Evaluate `state` and convert the result into Python scalar/collection `result`.
    result = dict(state)
    # Branch on condition `int(result['cursor']) == len(x)`:
    if int(result['cursor']) == len(x):
        key, order_key = jax.random.split(jnp.asarray(result['key']))
        result.update(key=np.asarray(key), order=np.asarray(jax.random.permutation(order_key, len(x))), cursor=np.asarray(0,np.int32))
    # Evaluate `result['cursor']` and convert the result into Python scalar/collection `start`.
    # Compute `start` as `int(result['cursor'])`.
    start = int(result['cursor'])
    ids = result['order'][start:start+batch_size]
    # Run `len` to compute `size`.
    # Compute `size` as `len(ids)`.
    size = len(ids)
    padded = ((size+3)//4)*4
    # Allocate initialized array `xb` with the specified shape and dtype.
    # Allocate initialized array `yb` with the specified shape and dtype.
    xb = np.zeros((padded,x.shape[1]),np.float32)
    yb = np.zeros(padded,np.float32)
    # Allocate initialized array `mask` with the specified shape and dtype.
    mask = np.zeros(padded,np.float32)
    # Compute `xb[:size],yb[:size],mask[:size]` as `x[ids],y[ids],1`.
    xb[:size],yb[:size],mask[:size] = x[ids],y[ids],1
    # Convert `result['cursor']` to a host NumPy array for inspection or verification.
    result['cursor'] = np.asarray(start+size,np.int32)
    # Place `arrays` explicitly onto the target JAX device.
    arrays = (jax.device_put(xb,NamedSharding(mesh,P('data',None))),
              jax.device_put(yb,NamedSharding(mesh,P('data'))),
              jax.device_put(mask,NamedSharding(mesh,P('data'))))
    # Return `(result, ids.copy(), arrays)` to the caller.
    return result, ids.copy(), arrays


# Function `make_step(mesh)` implementing this stage's computation:
def make_step(mesh, *, compiled=True):
    # Guard input contract (`mesh.size != 4`) and fail fast if violated.
    if mesh.size != 4:
        raise ValueError('exactly four devices required')
    # Function `local(x, y, valid, w, ...)` implementing this stage's computation:
    def local(x,y,valid,w,v):
        # Sum THEN divide by global count; local means fail on uneven partitions.
        count = jax.lax.psum(jnp.sum(valid), 'data')
        # Perform matrix contraction / projection to compute `residual`.
        residual = x@w-y
        # Perform matrix contraction / projection to compute `grad`.
        grad = jax.lax.psum(2*x.T@(residual*valid), 'data')/count
        # Aggregate array values to compute `loss`.
        loss = jax.lax.psum(jnp.sum(valid*residual**2), 'data')/count
        # Compute `velocity` as `.8*v+grad`.
        velocity = .8*v+grad
        # Return `(w - 0.03 * velocity, velocity, loss, grad)` to the caller.
        return w-.03*velocity, velocity, loss, grad
    # Configure multi-device placement / sharding specification (`mapped`).
    mapped = jax.shard_map(local, mesh=mesh,
        in_specs=(P('data',None),P('data'),P('data'),P(),P()),
        out_specs=(P(),P(),P(),P()))
    # Return `jax.jit(mapped) if compiled else mapped` to the caller.
    return jax.jit(mapped) if compiled else mapped


# Function `transition(state, x, y, batch_size, ...)` implementing this stage's computation:
def transition(state,x,y,batch_size,step_fn,mesh):
    # Run `next_batch` to compute `(result, ids, arrays)`.
    result,ids,arrays = next_batch(state,x,y,batch_size,mesh)
    # Run `step_fn` to compute `(w, v, loss, _)`.
    w,v,loss,_ = step_fn(*arrays,state['weights'],state['momentum'])
    # Convert `` to a host NumPy array for inspection or verification.
    result.update(weights=w,momentum=v,step=np.asarray(int(state['step'])+1,np.int32))
    # Return `(result, loss, ids)` to the caller.
    return result, loss, ids


# Function `contract(x, y, batch_size, seed)` implementing this stage's computation:
def contract(x,y,batch_size,seed):
    # Compute deterministic cryptographic digest `digest` for provenance verification.
    digest = hashlib.sha256()
    # Iterate over `a` to step through the computation:
    for a in (x,y):
        # Update state in place with the new values.
        # Update state in place with the new values.
        # Update state in place with the new values.
        digest.update(str(a.shape).encode())
        digest.update(a.dtype.str.encode())
        digest.update(a.tobytes())
    # Return `dict(schema=1, dataset_sha256=digest.hexdigest(), batch_size=batch_size, seed=seed, learning_rate=0.03, momentum=0.8, feature_width=x.shape[1], dataset_size=len(x), devices=4)` to the caller.
    return dict(schema=1, dataset_sha256=digest.hexdigest(), batch_size=batch_size, seed=seed,
                learning_rate=.03,momentum=.8,feature_width=x.shape[1],dataset_size=len(x),devices=4)


# Function `save(path, state, metadata)` implementing this stage's computation:
def save(path,state,metadata):
    # Read or serialize artifact data on disk (`path`).
    path = Path(path)
    # Guard input contract (`path.exists()`) and fail fast if violated.
    if path.exists():
        raise FileExistsError('use a fresh checkpoint path')
    # Run `path.with_suffix` to compute `temporary`.
    temporary = path.with_suffix('.tmp')
    # Enter `temporary.open('wb')` context block:
    with temporary.open('wb') as stream:
        # Convert `` to a host NumPy array for inspection or verification.
        np.savez(stream,**{k:np.asarray(v) for k,v in state.items()},
                 metadata=np.asarray(json.dumps(metadata,sort_keys=True)))
    # Run `temporary.replace` to perform the next check or state transition.
    temporary.replace(path)


# Function `restore(path, expected, mesh)` implementing this stage's computation:
def restore(path,expected,mesh):
    # Enter `np.load(path, allow_pickle=False)` context block:
    with np.load(path,allow_pickle=False) as archive:
        # Guard input contract (`set(archive.files) != {'weights', 'momentum', 'key', 'order', 'cursor', 'step', 'metadata'}`) and fail fast if violated.
        if set(archive.files) != {'weights','momentum','key','order','cursor','step','metadata'}:
            raise ValueError('incomplete checkpoint state')
        # Guard input contract (`json.loads(str(archive['metadata'])) != expected`) and fail fast if violated.
        if json.loads(str(archive['metadata'])) != expected:
            raise ValueError('checkpoint configuration/dataset mismatch')
        # Construct dictionary `state` with the structured fields for this stage.
        state = {k:archive[k].copy() for k in archive.files if k!='metadata'}
    # Compute `size,width` as `expected['dataset_size'],expected['feature_width']`.
    size,width = expected['dataset_size'],expected['feature_width']
    # Guard input contract (`state['weights'].shape != (width,) or state['momentum'].shape != (width,) or state['weights'].dtype != np.float32 or (state['momentum'].dtype != np.float32) or (not np.isfinite(state['weights']).all()) or (not np.isfinite(state['momentum']).all()) or (state['key'].shape != (2,)) or (state['key'].dtype != np.uint32) or (state['order'].dtype != np.int32) or (not np.array_equal(np.sort(state['order']), np.arange(size))) or (state['cursor'].shape != ()) or (state['cursor'].dtype != np.int32) or (state['step'].shape != ()) or (state['step'].dtype != np.int32) or (not 0 <= int(state['cursor']) <= size) or (int(state['step']) < 0)`) and fail fast if violated.
    if (state['weights'].shape!=(width,) or state['momentum'].shape!=(width,) or
        state['weights'].dtype!=np.float32 or state['momentum'].dtype!=np.float32 or
        not np.isfinite(state['weights']).all() or not np.isfinite(state['momentum']).all() or
        state['key'].shape!=(2,) or state['key'].dtype!=np.uint32 or
        state['order'].dtype!=np.int32 or not np.array_equal(np.sort(state['order']),np.arange(size)) or
        state['cursor'].shape!=() or state['cursor'].dtype!=np.int32 or
        state['step'].shape!=() or state['step'].dtype!=np.int32 or
        not 0<=int(state['cursor'])<=size or int(state['step'])<0):
        raise ValueError('checkpoint state shape, dtype or value contract failed')
    # Configure multi-device placement / sharding specification (`replicated`).
    replicated=NamedSharding(mesh,P())
    # Loop over `field` in `('weights', 'momentum')`:
    for field in ('weights','momentum'):
        # Place `state[field]` explicitly onto the target JAX device.
        state[field]=jax.device_put(state[field],replicated)
    # Return `state` to the caller.
    return state
