"""Implement each contract using the README stages. No hidden solution import."""
def initial(size,width,seed,mesh):
    """Zero replicated model/momentum; split key; permutation; int32 cursor/step."""
    raise NotImplementedError('stage 1: complete explicit state')
def next_batch(state,x,y,batch_size,mesh):
    """Pure transition, epoch reshuffle, tail padding and valid mask, four-way placement."""
    raise NotImplementedError('stage 1: sampler and shape contract')
def make_step(mesh,*,compiled=True):
    """Return real shard_map global count-weighted momentum update, optionally jitted."""
    raise NotImplementedError('stage 2: collective sums and global denominator')
def transition(state,x,y,batch_size,step_fn,mesh):
    """Combine sampler and update; return new state, pre-update loss, selected IDs."""
    raise NotImplementedError('stage 2: functional update order')
def contract(x,y,batch_size,seed):
    """Fingerprint dataset bytes, training configuration and schema."""
    raise NotImplementedError('stage 3: restore compatibility contract')
def save(path,state,metadata):
    """Durably write every state field plus metadata to a fresh NPZ checkpoint."""
    raise NotImplementedError('stage 3: full-state persistence')
def restore(path,expected,mesh):
    """Reject missing/corrupt/incompatible state and restore actual placement."""
    raise NotImplementedError('stage 3: validated restore')
