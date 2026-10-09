"""Implement each contract using the README stages. No hidden solution import."""
def initial(size,width,seed,mesh):
    """Zero replicated model/momentum; split key; permutation; int32 cursor/step."""
    # Key APIs to use: `contract`, `key`, `random.split`, `random.PRNGKey`, `specification`
    # Step 1: Guard input contract (`size < 1 or width < 1`) and fail fast if violated.
    # Step 2: Initialize explicit deterministic PRNG key `(key, order_key)`.
    # Step 3: Define device mesh or sharding layout for `replicated`.
    # Step 4: Return `dict(weights=jax.device_put(np.zeros(width, np.float32), replicated), momentum=jax.device_put(np.zeros(width, np.float32), replicated), key=np.asarray(key), order=np.asarray(jax.random.permutation(order_key, size)), cursor=np.asarray(0, np.int32), step=np.asarray(0, np.int32))` to the caller.
    raise NotImplementedError('stage 1: complete explicit state')
def next_batch(state,x,y,batch_size,mesh):
    """Pure transition, epoch reshuffle, tail padding and valid mask, four-way placement."""
    # Key APIs to use: `np.asarray`, `contract`, `or`, `np.isfinite`, `all`
    # Step 1: Convert `(x, y)` to a host NumPy array for inspection or verification.
    # Step 2: Guard input contract (`x.ndim != 2 or y.shape != (len(x),) or len(x) != len(state['order'])`) and fail fast if violated.
    # Step 3: Guard input contract (`x.dtype != np.float32 or y.dtype != np.float32 or (not np.isfinite(x).all()) or (not np.isfinite(y).all())`) and fail fast if violated.
    # Step 4: Guard input contract (`not isinstance(batch_size, int) or batch_size < 1 or int(state['cursor']) not in range(len(x) + 1)`) and fail fast if violated.
    # Step 5: Guard input contract (`mesh.size != 4`) and fail fast if violated.
    # Step 6: Evaluate `state` and convert the result into Python scalar/collection `result`.
    raise NotImplementedError('stage 1: sampler and shape contract')
def make_step(mesh,*,compiled=True):
    """Return real shard_map global count-weighted momentum update, optionally jitted."""
    # Key APIs to use: `contract`, `local`, `lax.psum`, `jnp.sum`, `specification`
    # Step 1: Guard input contract (`mesh.size != 4`) and fail fast if violated.
    # Step 2: Function `local(x, y, valid, w, ...)` implementing this stage's computation:
    # Step 3: Define device mesh or sharding layout for `mapped`.
    # Step 4: Return `jax.jit(mapped) if compiled else mapped` to the caller.
    raise NotImplementedError('stage 2: collective sums and global denominator')
def transition(state,x,y,batch_size,step_fn,mesh):
    """Combine sampler and update; return new state, pre-update loss, selected IDs."""
    # Key APIs to use: `next_batch`, `step_fn`, `result.update`, `np.asarray`
    # Step 1: Run `next_batch` to compute `(result, ids, arrays)`.
    # Step 2: Run `step_fn` to compute `(w, v, loss, _)`.
    # Step 3: Convert `` to a host NumPy array for inspection or verification.
    # Step 4: Return `(result, loss, ids)` to the caller.
    raise NotImplementedError('stage 2: functional update order')
def contract(x,y,batch_size,seed):
    """Fingerprint dataset bytes, training configuration and schema."""
    # Key APIs to use: `hashlib.sha256`, `in`, `digest.update`, `encode`, `str.encode`
    # Step 1: Compute deterministic cryptographic digest `digest` for provenance verification.
    # Step 2: Loop over `a` in `(x, y)`:
    # Step 3: Inside block: Update state in place with the new values.
    # Step 4: Inside block: Update state in place with the new values.
    # Step 5: Return `dict(schema=1, dataset_sha256=digest.hexdigest(), batch_size=batch_size, seed=seed, learning_rate=0.03, momentum=0.8, feature_width=x.shape[1], dataset_size=len(x), devices=4)` to the caller.
    raise NotImplementedError('stage 3: restore compatibility contract')
def save(path,state,metadata):
    """Durably write every state field plus metadata to a fresh NPZ checkpoint."""
    # Key APIs to use: `disk`, `Path`, `contract`, `path.exists`, `FileExistsError`
    # Step 1: Read or serialize artifact data on disk (`path`).
    # Step 2: Guard input contract (`path.exists()`) and fail fast if violated.
    # Step 3: Run `path.with_suffix` to compute `temporary`.
    # Step 4: Enter managed runtime/context scope for this block:
    # Step 5: Inside block: Convert `` to a host NumPy array for inspection or verification.
    # Step 6: Run `temporary.replace` to perform the next check or state transition.
    raise NotImplementedError('stage 3: full-state persistence')
def restore(path,expected,mesh):
    """Reject missing/corrupt/incompatible state and restore actual placement."""
    # Key APIs to use: `np.load`, `contract`, `json.loads`, `copy`, `or`
    # Step 1: Enter managed runtime/context scope for this block:
    # Step 2: Inside block: Guard input contract (`set(archive.files) != {'weights', 'momentum', 'key', 'order', 'cursor', 'step', 'metadata'}`) and fail fast if violated.
    # Step 3: Inside block: Guard input contract (`json.loads(str(archive['metadata'])) != expected`) and fail fast if violated.
    # Step 4: Evaluate `(size, width)` from the current inputs and state.
    # Step 5: Guard input contract (`state['weights'].shape != (width,) or state['momentum'].shape != (width,) or state['weights'].dtype != np.float32 or (state['momentum'].dtype != np.float32) or (not np.isfinite(state['weights']).all()) or (not np.isfinite(state['momentum']).all()) or (state['key'].shape != (2,)) or (state['key'].dtype != np.uint32) or (state['order'].dtype != np.int32) or (not np.array_equal(np.sort(state['order']), np.arange(size))) or (state['cursor'].shape != ()) or (state['cursor'].dtype != np.int32) or (state['step'].shape != ()) or (state['step'].dtype != np.int32) or (not 0 <= int(state['cursor']) <= size) or (int(state['step']) < 0)`) and fail fast if violated.
    # Step 6: Define device mesh or sharding layout for `replicated`.
    raise NotImplementedError('stage 3: validated restore')
