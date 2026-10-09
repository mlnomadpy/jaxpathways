"""Connected, inspectable CPU image harness; synthetic fixture, no vision benchmark."""
# Import pathlib (Path) for this computation.
from pathlib import Path
import hashlib
import json
import os
import tempfile
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import export

# Rearrange tensor axes to match the required layout for `PREPROCESS`.
PREPROCESS=dict(version='image-v1',height=8,width=8,channels=1,source_channels=[1,3],
                resize='nearest floor-index',rgb_weights=[.2126,.7152,.0722],
                normalization='2 * gray_in_0_1 - 1',layout='NHWC',exif='transpose on file decode')
# Initialize list `CLASSES` for the stage values.
CLASSES=['vertical','horizontal','cross']


# Function `digest_json(value)` implementing this stage's computation:
def digest_json(value):
    # Return `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()` to the caller.
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


# Function `digest_arrays()` implementing this stage's computation:
def digest_arrays(*arrays):
    # Compute deterministic cryptographic digest `h` for provenance verification.
    h=hashlib.sha256()
    # Iterate over `value` to step through the computation:
    for value in arrays:
        # Run `np.ascontiguousarray` to compute `a`.
        # Compute `a` as `np.ascontiguousarray(value)`.
        a=np.ascontiguousarray(value)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    # Return `h.hexdigest()` to the caller.
    return h.hexdigest()


# Function `make_dataset(seed, count, split, corruption)` implementing this stage's computation:
def make_dataset(seed=11,count=96,split='train',corruption=False):
    # Guard input contract (`count < 3`) and fail fast if violated.
    if count<3:raise ValueError('at least three images required')
    # Draw pseudorandom samples for `rng` using the explicit RNG state.
    # Create evenly spaced index values in `labels`.
    rng=np.random.default_rng(seed)
    labels=np.arange(count,dtype=np.int32)%3
    # Run `np.empty` to compute `images`.
    images=np.empty((count,8,8,1),np.uint8)
    # Iterate over `(i, label)` to step through the computation:
    for i,label in enumerate(labels):
        # Draw pseudorandom samples for `canvas` using the explicit RNG state.
        # Draw pseudorandom samples for `(row, col)` using the explicit RNG state.
        canvas=rng.normal(18,5,(8,8))
        row,col=rng.integers(1,6,size=2)
        # Branch on condition `label in (0, 2)`:
        if label in (0,2):canvas[:,col:col+2]+=rng.uniform(170,210)
        # Branch on condition `label in (1, 2)`:
        if label in (1,2):canvas[row:row+2,:]+=rng.uniform(170,210)
        # Combine or mask array elements to form `images[i, :, :, 0]`.
        images[i,:,:,0]=np.clip(canvas,0,255).astype(np.uint8)
    # Branch on condition `corruption`:
    if corruption:
        # Apply a separate random stream after clean generation: pairs remain aligned.
        altered=images.astype(np.float32)
        altered[:,:,2:6,:]=18
        altered+=np.random.default_rng(seed+5000).normal(0,28,altered.shape)
        images=np.clip(altered,0,255).astype(np.uint8)
    # Return `dict(images=images, labels=labels, ids=[f'{split}-{seed}-{i:04d}' for i in range(count)], groups=[f'{split}-synthetic-{seed}-{i:04d}' for i in range(count)], provenance={'kind': 'synthetic teaching fixture', 'seed': seed, 'split': split, 'corruption': corruption, 'license': 'original generated fixture; no external photographs'})` to the caller.
    return dict(images=images,labels=labels,ids=[f'{split}-{seed}-{i:04d}' for i in range(count)],
                groups=[f'{split}-synthetic-{seed}-{i:04d}' for i in range(count)],
                provenance={'kind':'synthetic teaching fixture','seed':seed,'split':split,'corruption':corruption,'license':'original generated fixture; no external photographs'})


# Function `dataset_hash(data)` implementing this stage's computation:
def dataset_hash(data):
    # Return `digest_json(dict(arrays=digest_arrays(data['images'], data['labels']), ids=data['ids'], groups=data['groups'], provenance=data.get('provenance', {})))` to the caller.
    return digest_json(dict(arrays=digest_arrays(data['images'],data['labels']),ids=data['ids'],groups=data['groups'],provenance=data.get('provenance',{})))


# Function `validate_splits(training, heldout)` implementing this stage's computation:
def validate_splits(training,heldout):
    # Iterate over `data` to step through the computation:
    for data in (training,heldout):
        # Run `len` to compute `n`.
        n=len(data['images'])
        # Guard input contract (`np.asarray(data['labels']).shape != (n,) or len(data['ids']) != n or len(data['groups']) != n`) and fail fast if violated.
        if np.asarray(data['labels']).shape!=(n,) or len(data['ids'])!=n or len(data['groups'])!=n:raise ValueError('dataset shape contract')
        # Guard input contract (`len(set(data['ids'])) != n`) and fail fast if violated.
        if len(set(data['ids']))!=n:raise ValueError('duplicate image IDs')
        # Guard input contract (`not np.issubdtype(np.asarray(data['labels']).dtype, np.integer) or np.any((data['labels'] < 0) | (data['labels'] >= 3))`) and fail fast if violated.
        if not np.issubdtype(np.asarray(data['labels']).dtype,np.integer) or np.any((data['labels']<0)|(data['labels']>=3)):raise ValueError('class labels required')
    # Guard input contract (`set(training['ids']) & set(heldout['ids']) or set(training['groups']) & set(heldout['groups'])`) and fail fast if violated.
    if set(training['ids'])&set(heldout['ids']) or set(training['groups'])&set(heldout['groups']):raise ValueError('split leakage')
    # Construct dictionary `train_pixels` with the structured fields for this stage.
    train_pixels={digest_arrays(image) for image in training['images']}
    # Guard input contract (`train_pixels & {digest_arrays(image) for image in heldout['images']}`) and fail fast if violated.
    if train_pixels & {digest_arrays(image) for image in heldout['images']}:raise ValueError('duplicate image content across splits')


# Function `preprocess(images, layout, input_range)` implementing this stage's computation:
def preprocess(images,layout='NHWC',input_range='uint8'):
    # Convert `a` to a host NumPy array for inspection or verification.
    a=np.asarray(images)
    # Guard input contract (`a.ndim != 4 or not len(a)`) and fail fast if violated.
    if a.ndim!=4 or not len(a):raise ValueError('nonempty rank-four image batch required')
    # Branch on condition `layout == 'NCHW'`:
    if layout=='NCHW':a=np.transpose(a,(0,2,3,1))
    elif layout!='NHWC':raise ValueError('unsupported layout')
    # Guard input contract (`a.shape[-1] not in (1, 3) or min(a.shape[1:3]) < 1`) and fail fast if violated.
    if a.shape[-1] not in (1,3) or min(a.shape[1:3])<1:raise ValueError('grayscale/RGB channels and positive size required')
    # Branch on condition `input_range == 'uint8'`:
    if input_range=='uint8':
        if a.dtype!=np.uint8:raise ValueError('uint8 input contract requires uint8 dtype')
        a=a.astype(np.float32)/255.
    elif input_range=='unit_float':
        if not np.issubdtype(a.dtype,np.floating) or not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError('finite unit-range floating input required')
        a=a.astype(np.float32)
    else:raise ValueError('unsupported input range')
    # Initialize array `rows` with explicit values and shape.
    rows=(np.arange(8)*a.shape[1]//8)
    cols=(np.arange(8)*a.shape[2]//8)
    # Compute `a` as `a[:,rows][:,:,cols]`.
    a=a[:,rows][:,:,cols]
    # Branch on condition `a.shape[-1] == 3`:
    if a.shape[-1]==3:a=np.sum(a*np.array(PREPROCESS['rgb_weights'],np.float32),axis=-1,keepdims=True)
    # Return `np.ascontiguousarray(2 * a - 1, dtype=np.float32)` to the caller.
    return np.ascontiguousarray(2*a-1,dtype=np.float32)


# Function `read_image(path)` implementing this stage's computation:
def read_image(path):
    """External PNG/JPEG/etc through Pillow, explicit EXIF transpose and RGB conversion."""
    # Import PIL (Image, ImageOps) for this computation.
    from PIL import Image,ImageOps
    # Enter `Image.open(path)` context block:
    with Image.open(path) as image:
        # Guard input contract (`image.width * image.height > 16000000`) and fail fast if violated.
        if image.width*image.height>16_000_000:raise ValueError('image exceeds teaching decoder pixel limit')
        # Guard input contract (`getattr(image, 'n_frames', 1) != 1`) and fail fast if violated.
        if getattr(image,'n_frames',1)!=1:raise ValueError('animated images are not supported')
        # Return `np.asarray(ImageOps.exif_transpose(image).convert('RGB'), dtype=np.uint8)[None]` to the caller.
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB'),dtype=np.uint8)[None]


# Function `load_external_manifest(path)` implementing this stage's computation:
def load_external_manifest(path):
    """Load user-owned labeled images; relative paths stay inside manifest directory.

    JSON: split, source, license, items [{path,id,group,label}]. This loads supplied
    files only and never downloads data. Split related subjects/sources before use.
    """
    # Run `path.parent.resolve` to compute `root`.
    path=Path(path)
    manifest=json.loads(path.read_text())
    root=path.parent.resolve()
    # Guard input contract (`not all((manifest.get(k) for k in ('split', 'source', 'license', 'items')))`) and fail fast if violated.
    if not all(manifest.get(k) for k in ('split','source','license','items')):raise ValueError('data provenance and items required')
    # Initialize list `images` for the stage values.
    images=[]
    labels=[]
    ids=[]
    groups=[]
    file_hashes=[]
    # Loop over `item` in `manifest['items']`:
    for item in manifest['items']:
        # Evaluate the compound expression for `image_path`.
        image_path=(root/item['path']).resolve()
        # Guard input contract (`not image_path.is_relative_to(root)`) and fail fast if violated.
        if not image_path.is_relative_to(root):raise ValueError('image outside manifest directory')
        # Guard input contract (`item['label'] not in CLASSES`) and fail fast if violated.
        if item['label'] not in CLASSES:raise ValueError('unknown image label')
        # Guard input contract (`not item.get('id') or not item.get('group')`) and fail fast if violated.
        if not item.get('id') or not item.get('group'):raise ValueError('image and source-group IDs required')
        # Run `read_image` to compute `raw`.
        raw=read_image(image_path)
        # Create evenly spaced index values in `rows`.
        # Create evenly spaced index values in `cols`.
        rows=np.arange(8)*raw.shape[1]//8
        cols=np.arange(8)*raw.shape[2]//8
        # Append the current step result to `images`.
        # Append the current step result to `images`.
        images.append(raw[0][rows][:,cols])
        labels.append(CLASSES.index(item['label']))
        # Append the current step result to `ids`.
        # Append the current step result to `ids`.
        # Compute deterministic cryptographic digest `` for provenance verification.
        ids.append(item['id'])
        groups.append(item['group'])
        file_hashes.append(hashlib.sha256(image_path.read_bytes()).hexdigest())
    # Guard input contract (`len(set(ids)) != len(ids)`) and fail fast if violated.
    if len(set(ids))!=len(ids):raise ValueError('duplicate image IDs')
    # Return `dict(images=np.stack(images), labels=np.asarray(labels, np.int32), ids=ids, groups=groups, provenance=dict(kind='user-supplied image files', split=manifest['split'], source=manifest['source'], license=manifest['license'], file_hashes=file_hashes))` to the caller.
    return dict(images=np.stack(images),labels=np.asarray(labels,np.int32),ids=ids,groups=groups,
                provenance=dict(kind='user-supplied image files',split=manifest['split'],source=manifest['source'],license=manifest['license'],file_hashes=file_hashes))


# Function `init_params(seed)` implementing this stage's computation:
def init_params(seed=0):
    # Initialize explicit deterministic PRNG key `(a, b)`.
    a,b=jax.random.split(jax.random.PRNGKey(seed))
    # Return `dict(conv=jax.random.normal(a, (3, 3, 1, 6)) * 0.3, conv_bias=jnp.zeros(6), head=jax.random.normal(b, (6, 3)) * 0.2, head_bias=jnp.zeros(3))` to the caller.
    return dict(conv=jax.random.normal(a,(3,3,1,6))*.3,conv_bias=jnp.zeros(6),
                head=jax.random.normal(b,(6,3))*.2,head_bias=jnp.zeros(3))


# Function `forward(params, x, compute, return_hidden)` implementing this stage's computation:
def forward(params,x,compute='float32',return_hidden=False):
    # Cast or evaluate `dtype` in explicit floating-point precision.
    dtype={'float32':jnp.float32,'float16':jnp.float16,'bfloat16':jnp.bfloat16}[compute]
    # Create device-backed JAX array `conv`.
    conv=jax.lax.conv_general_dilated(jnp.asarray(x,dtype),params['conv'].astype(dtype),(1,1),'VALID',dimension_numbers=('NHWC','HWIO','NHWC'),preferred_element_type=jnp.float32)
    # Run `jax.nn.relu` to compute `hidden`.
    hidden=jax.nn.relu(conv+params['conv_bias'])
    # Reduce across the target axis to summarize `pooled`.
    pooled=hidden.mean(axis=(1,2))
    # Cast or evaluate `logits` in explicit floating-point precision.
    logits=jax.lax.dot_general(pooled.astype(dtype),params['head'].astype(dtype),(((1,),(0,)),((),())),preferred_element_type=jnp.float32)+params['head_bias']
    # Return `(logits, pooled) if return_hidden else logits` to the caller.
    return (logits,pooled) if return_hidden else logits


# Function `objective(params, x, labels)` implementing this stage's computation:
def objective(params,x,labels):
    # Create device-backed JAX array `labels`.
    labels=jnp.asarray(labels)
    # Guard input contract (`labels.ndim != 1 or labels.shape[0] != x.shape[0] or (not jnp.issubdtype(labels.dtype, jnp.integer))`) and fail fast if violated.
    if labels.ndim!=1 or labels.shape[0]!=x.shape[0] or not jnp.issubdtype(labels.dtype,jnp.integer):raise ValueError('one integer label per image required')
    # Run `forward` to compute `logits`.
    logits=forward(params,x)
    # Return `-jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), labels[:, None], axis=1))` to the caller.
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits),labels[:,None],axis=1))


# Function `config(batch_size, learning_rate, momentum)` implementing this stage's computation:
def config(batch_size=12,learning_rate=.08,momentum=.9):
    # Guard input contract (`not isinstance(batch_size, int) or batch_size < 1 or (not 0 < learning_rate < 1) or (not 0 <= momentum < 1)`) and fail fast if violated.
    if not isinstance(batch_size,int) or batch_size<1 or not 0<learning_rate<1 or not 0<=momentum<1:raise ValueError('invalid optimizer/input configuration')
    # Return `dict(batch_size=batch_size, learning_rate=learning_rate, momentum=momentum)` to the caller.
    return dict(batch_size=batch_size,learning_rate=learning_rate,momentum=momentum)


# Function `start(data, seed, settings)` implementing this stage's computation:
def start(data,seed=0,settings=None):
    # Run `config` to compute `settings`.
    # Run `init_params` to compute `params`.
    settings=config(**(settings or {}))
    params=init_params(seed)
    # Initialize explicit deterministic PRNG key `(key, order_key)`.
    key,order_key=jax.random.split(jax.random.PRNGKey(seed+100))
    # Convert `order` to a host NumPy array for inspection or verification.
    order=np.asarray(jax.random.permutation(order_key,len(data['images'])))
    # Return `dict(params=params, velocity=jax.tree.map(jnp.zeros_like, params), key=key, order=order, position=0, epoch=0, step=0, seed=seed, data_hash=dataset_hash(data), settings=settings)` to the caller.
    return dict(params=params,velocity=jax.tree.map(jnp.zeros_like,params),key=key,order=order,
                position=0,epoch=0,step=0,seed=seed,data_hash=dataset_hash(data),settings=settings)


@jax.jit
# Function `_update(params, velocity, key, x, ...)` implementing this stage's computation:
def _update(params,velocity,key,x,y,learning_rate,momentum):
    # Split the PRNG key deterministically into independent subkeys (`(key, noise_key, flip_key)`).
    key,noise_key,flip_key=jax.random.split(key,3)
    # Draw pseudorandom samples for `flips` using the explicit RNG state.
    flips=jax.random.bernoulli(flip_key,.5,(len(x),1,1,1))
    # Combine or mask array elements to form `augmented`.
    augmented=jnp.where(flips,x[:,:,::-1,:],x)
    # Draw pseudorandom samples for `augmented` using the explicit RNG state.
    augmented=jnp.clip(augmented+.035*jax.random.normal(noise_key,x.shape),-1,1)
    # Evaluate both scalar loss and parameter gradients in one pass (`(loss, grad)`).
    loss,grad=jax.value_and_grad(objective)(params,augmented,y)
    # Apply leaf-wise transformation across the PyTree to produce `velocity`.
    velocity=jax.tree.map(lambda v,g:momentum*v+g,velocity,grad)
    # Apply leaf-wise transformation across the PyTree to produce `params`.
    params=jax.tree.map(lambda p,v:p-learning_rate*v,params,velocity)
    # Return `(params, velocity, key, loss, augmented)` to the caller.
    return params,velocity,key,loss,augmented


# Function `advance(state, data, updates)` implementing this stage's computation:
def advance(state,data,updates=1):
    # Guard input contract (`dataset_hash(data) != state['data_hash']`) and fail fast if violated.
    if dataset_hash(data)!=state['data_hash']:raise ValueError('dataset changed since checkpoint')
    # Guard input contract (`not isinstance(updates, int) or updates < 1`) and fail fast if violated.
    if not isinstance(updates,int) or updates<1:raise ValueError('positive update count required')
    # Evaluate `state` and convert the result into Python scalar/collection `s`.
    # Run `len` to compute `n`.
    # Compute `s` as `dict(state)`.
    s=dict(state)
    records=[]
    n=len(data['images'])
    batch=s['settings']['batch_size']
    # Guard input contract (`batch > n`) and fail fast if violated.
    if batch>n:raise ValueError('batch exceeds dataset')
    # Run `preprocess` to compute `all_x`.
    all_x=preprocess(data['images'])
    # Repeat the update loop over `range(updates)` steps:
    for _ in range(updates):
        # Branch on condition `s['position'] + batch > n`:
        if s['position']+batch>n:
            s['key'],k=jax.random.split(s['key'])
            s['order']=np.asarray(jax.random.permutation(k,n))
            s['position']=0
            s['epoch']+=1
        # Compute `indices` as `s['order'][s['position']:s['position']+batch]`.
        indices=s['order'][s['position']:s['position']+batch]
        # Create device-backed JAX array `(p, v, k, loss, aug)`.
        p,v,k,loss,aug=_update(s['params'],s['velocity'],s['key'],jnp.asarray(all_x[indices]),jnp.asarray(data['labels'][indices]),s['settings']['learning_rate'],s['settings']['momentum'])
        # Synchronize host execution until asynchronous device computation completes.
        # Synchronize host execution until asynchronous device computation completes.
        jax.block_until_ready((p,v,k,loss,aug))
        s.update(params=p,velocity=v,key=k,position=s['position']+batch,step=s['step']+1)
        # Append the current step result to `records`.
        records.append(dict(step=s['step'],loss=float(loss),ids=[data['ids'][i] for i in indices],augmented_hash=digest_arrays(aug)))
    # Return `(s, records)` to the caller.
    return s,records


# Function `evaluate(params, data, batch_size, compute)` implementing this stage's computation:
def evaluate(params,data,batch_size=17,compute='float32'):
    # Guard input contract (`batch_size < 1 or not len(data['images'])`) and fail fast if violated.
    if batch_size<1 or not len(data['images']):raise ValueError('positive evaluation batch and nonempty data required')
    # Run `preprocess` to compute `x`.
    # Convert `y` to a host NumPy array for inspection or verification.
    x=preprocess(data['images'])
    y=np.asarray(data['labels'])
    # Guard input contract (`y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= 3))`) and fail fast if violated.
    if y.shape!=(len(x),) or not np.issubdtype(y.dtype,np.integer) or np.any((y<0)|(y>=3)):raise ValueError('valid integer vector labels required')
    # Create device-backed JAX array `logits`.
    logits=np.concatenate([np.asarray(forward(params,jnp.asarray(x[i:i+batch_size]),compute)) for i in range(0,len(x),batch_size)])
    # Return `metrics(logits, y, data['ids'])` to the caller.
    return metrics(logits,y,data['ids'])


# Function `metrics(logits, labels, ids)` implementing this stage's computation:
def metrics(logits,labels,ids):
    # Convert `logits` to a host NumPy array for inspection or verification.
    # Convert `labels` to a host NumPy array for inspection or verification.
    logits=np.asarray(logits,dtype=np.float64)
    labels=np.asarray(labels)
    # Guard input contract (`logits.ndim != 2 or logits.shape[1] != 3 or len(logits) == 0 or (not np.isfinite(logits).all()) or (labels.shape != (len(logits),)) or (not np.issubdtype(labels.dtype, np.integer)) or np.any((labels < 0) | (labels >= 3)) or (len(ids) != len(logits))`) and fail fast if violated.
    if logits.ndim!=2 or logits.shape[1]!=3 or len(logits)==0 or not np.isfinite(logits).all() or labels.shape!=(len(logits),) or not np.issubdtype(labels.dtype,np.integer) or np.any((labels<0)|(labels>=3)) or len(ids)!=len(logits):raise ValueError('invalid evaluation arrays')
    # Reduce across the target axis to summarize `shifted`.
    shifted=logits-logits.max(1,keepdims=True)
    # Create evenly spaced index values in `losses`.
    losses=np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(labels)),labels]
    # Run `np.argmax` to compute `predicted`.
    # Allocate initialized array `confusion` with the specified shape and dtype.
    predicted=np.argmax(logits,1)
    confusion=np.zeros((3,3),int)
    # Run `np.add.at` to perform the next check or state transition.
    np.add.at(confusion,(labels,predicted),1)
    # Return `dict(count=len(labels), loss=float(losses.mean()), loss_sum=float(losses.sum()), correct=int((predicted == labels).sum()), accuracy=float((predicted == labels).mean()), confusion=confusion.tolist(), error_ids=[ids[i] for i in np.flatnonzero(predicted != labels)], logits=logits, predictions=predicted)` to the caller.
    return dict(count=len(labels),loss=float(losses.mean()),loss_sum=float(losses.sum()),correct=int((predicted==labels).sum()),
                accuracy=float((predicted==labels).mean()),confusion=confusion.tolist(),error_ids=[ids[i] for i in np.flatnonzero(predicted!=labels)],logits=logits,predictions=predicted)


# Function `save_checkpoint(path, state)` implementing this stage's computation:
def save_checkpoint(path,state):
    # Compute `path` as `Path(path)`.
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Convert `arrays` to a host NumPy array for inspection or verification.
    arrays={**{'p_'+k:np.asarray(v) for k,v in state['params'].items()},**{'v_'+k:np.asarray(v) for k,v in state['velocity'].items()},'key':np.asarray(state['key']),'order':np.asarray(state['order'])}
    # Construct dictionary `meta` with the structured fields for this stage.
    meta={k:state[k] for k in ('step','position','epoch','seed','data_hash','settings')}
    meta.update(version=1,jax_version=jax.__version__,numpy_version=np.__version__,preprocess=PREPROCESS,arrays_hash=digest_arrays(*[arrays[k] for k in sorted(arrays)]))
    # Run `digest_json` to compute `meta['metadata_hash']`.
    meta['metadata_hash']=digest_json(meta)
    # Run `tempfile.mkstemp` to compute `(fd, name)`.
    fd,name=tempfile.mkstemp(dir=path.parent,suffix='.npz')
    try:
        with os.fdopen(fd,'wb') as out:
            np.savez(out,**arrays,metadata=np.array(json.dumps(meta,sort_keys=True)))
            out.flush()
            os.fsync(out.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


# Function `restore_checkpoint(path, data, settings)` implementing this stage's computation:
def restore_checkpoint(path,data,settings=None):
    # Enter managed runtime/context scope for this block:
    with np.load(path,allow_pickle=False) as bundle:
        # Read or serialize artifact data on disk (`meta`).
        arrays={k:bundle[k].copy() for k in bundle.files if k!='metadata'}
        meta=json.loads(str(bundle['metadata']))
    # Run `meta.pop` to compute `metadata_hash`.
    metadata_hash=meta.pop('metadata_hash',None)
    # Guard input contract (`metadata_hash != digest_json(meta)`) and fail fast if violated.
    if metadata_hash!=digest_json(meta):raise ValueError('checkpoint metadata integrity mismatch')
    # Guard input contract (`meta['jax_version'] != jax.__version__ or meta['numpy_version'] != np.__version__`) and fail fast if violated.
    if meta['jax_version']!=jax.__version__ or meta['numpy_version']!=np.__version__:raise ValueError('checkpoint runtime version mismatch')
    # Guard input contract (`meta['version'] != 1 or meta['preprocess'] != PREPROCESS or meta['data_hash'] != dataset_hash(data) or (meta['settings'] != config(**settings or {}))`) and fail fast if violated.
    if meta['version']!=1 or meta['preprocess']!=PREPROCESS or meta['data_hash']!=dataset_hash(data) or meta['settings']!=config(**(settings or {})):raise ValueError('checkpoint data/preprocess/config mismatch')
    # Guard input contract (`meta['arrays_hash'] != digest_arrays(*[arrays[k] for k in sorted(arrays)])`) and fail fast if violated.
    if meta['arrays_hash']!=digest_arrays(*[arrays[k] for k in sorted(arrays)]):raise ValueError('checkpoint integrity mismatch')
    # Guard input contract (`arrays['key'].shape != (2,) or arrays['key'].dtype != np.uint32`) and fail fast if violated.
    if arrays['key'].shape!=(2,) or arrays['key'].dtype!=np.uint32:raise ValueError('invalid random key')
    # Guard input contract (`sorted(arrays['order'].tolist()) != list(range(len(data['images']))) or not 0 <= meta['position'] <= len(data['images']) or meta['step'] < 0 or (meta['epoch'] < 0)`) and fail fast if violated.
    if sorted(arrays['order'].tolist())!=list(range(len(data['images']))) or not 0<=meta['position']<=len(data['images']) or meta['step']<0 or meta['epoch']<0:raise ValueError('invalid sampler state')
    # Create device-backed JAX array `params`.
    # Create device-backed JAX array `velocity`.
    params={k: jnp.asarray(arrays['p_'+k]) for k in init_params()}
    velocity={k:jnp.asarray(arrays['v_'+k]) for k in params}
    # Loop over `(k, template)` in `init_params().items()`:
    for k,template in init_params().items():
        # Guard input contract (`params[k].shape != template.shape or velocity[k].shape != template.shape or (not np.isfinite(params[k]).all()) or (not np.isfinite(velocity[k]).all())`) and fail fast if violated.
        if params[k].shape!=template.shape or velocity[k].shape!=template.shape or not np.isfinite(params[k]).all() or not np.isfinite(velocity[k]).all():raise ValueError('invalid parameter or optimizer state')
    # Return `dict(params=params, velocity=velocity, key=jnp.asarray(arrays['key']), order=arrays['order'], **{k: meta[k] for k in ('step', 'position', 'epoch', 'seed', 'data_hash', 'settings')})` to the caller.
    return dict(params=params,velocity=velocity,key=jnp.asarray(arrays['key']),order=arrays['order'],**{k:meta[k] for k in ('step','position','epoch','seed','data_hash','settings')})


# Function `patches(x)` implementing this stage's computation:
def patches(x):
    # Return `np.stack([x[:, i:i + 6, j:j + 6, :] for i in range(3) for j in range(3)], axis=-2).reshape(len(x), 6, 6, 9)` to the caller.
    return np.stack([x[:,i:i+6,j:j+6,:] for i in range(3) for j in range(3)],axis=-2).reshape(len(x),6,6,9)


# Function `quantize(values, scale)` implementing this stage's computation:
def quantize(values,scale):
    # Return `np.clip(np.rint(np.asarray(values) / scale), -127, 127).astype(np.int8)` to the caller.
    return np.clip(np.rint(np.asarray(values)/scale),-127,127).astype(np.int8)


# Function `calibrate(params, data, percentile)` implementing this stage's computation:
def calibrate(params,data,percentile=99.):
    # Guard input contract (`not 0 < percentile <= 100`) and fail fast if violated.
    if not 0<percentile<=100:raise ValueError('invalid calibration percentile')
    # Run `preprocess` to compute `x`.
    # Create device-backed JAX array `(_, hidden)`.
    x=preprocess(data['images'])
    _,hidden=forward(params,jnp.asarray(x),return_hidden=True)
    # Run `max` to compute `input_scale`.
    input_scale=max(float(np.percentile(np.abs(x),percentile))/127,1e-8)
    # Run `max` to compute `hidden_scale`.
    hidden_scale=max(float(np.percentile(np.abs(hidden),percentile))/127,1e-8)
    # Convert `conv` to a host NumPy array for inspection or verification.
    # Convert `head` to a host NumPy array for inspection or verification.
    conv=np.asarray(params['conv']).reshape(9,6)
    head=np.asarray(params['head'])
    # Reduce across the target axis to summarize `cs`.
    # Reduce across the target axis to summarize `hs`.
    cs=np.maximum(np.max(np.abs(conv),axis=0)/127,1e-8)
    hs=np.maximum(np.max(np.abs(head),axis=0)/127,1e-8)
    # Return `dict(conv=quantize(conv, cs), conv_scale=cs, conv_bias=np.asarray(params['conv_bias']), head=quantize(head, hs), head_scale=hs, head_bias=np.asarray(params['head_bias']), input_scale=np.array(input_scale, np.float32), hidden_scale=np.array(hidden_scale, np.float32), calibration_hash=np.array(dataset_hash(data)), percentile=np.array(percentile))` to the caller.
    return dict(conv=quantize(conv,cs),conv_scale=cs,conv_bias=np.asarray(params['conv_bias']),head=quantize(head,hs),head_scale=hs,head_bias=np.asarray(params['head_bias']),input_scale=np.array(input_scale,np.float32),hidden_scale=np.array(hidden_scale,np.float32),calibration_hash=np.array(dataset_hash(data)),percentile=np.array(percentile))


# Function `integer_forward(quantized, x, accumulator)` implementing this stage's computation:
def integer_forward(quantized,x,accumulator='int32'):
    # Guard input contract (`accumulator != 'int32'`) and fail fast if violated.
    if accumulator!='int32':raise ValueError('only int32 accumulation is supported')
    # Convert `raw` to a host NumPy array for inspection or verification.
    # Run `quantize` to compute `ix`.
    q=quantized
    raw=np.asarray(x)
    ix=quantize(raw,q['input_scale'])
    # Perform matrix contraction / projection to compute `acc`.
    acc=patches(ix).astype(np.int32) @ q['conv'].astype(np.int32)
    # Cast or evaluate `hidden` in explicit floating-point precision.
    hidden=np.maximum(acc.astype(np.float32)*(q['input_scale']*q['conv_scale'])+q['conv_bias'],0).mean(axis=(1,2))
    # Run `quantize` to compute `ih`.
    ih=quantize(hidden,q['hidden_scale'])
    # Cast or evaluate `logits` in explicit floating-point precision.
    logits=(ih.astype(np.int32)@q['head'].astype(np.int32)).astype(np.float32)*(q['hidden_scale']*q['head_scale'])+q['head_bias']
    # Return `(logits, dict(input_clipped=int((np.abs(raw) > 127 * q['input_scale']).sum()), input_count=raw.size, hidden_clipped=int((np.abs(hidden) > 127 * q['hidden_scale']).sum()), hidden_count=hidden.size))` to the caller.
    return logits,dict(input_clipped=int((np.abs(raw)>127*q['input_scale']).sum()),input_count=raw.size,hidden_clipped=int((np.abs(hidden)>127*q['hidden_scale']).sum()),hidden_count=hidden.size)


# Function `export_model(folder, state, quantized)` implementing this stage's computation:
def export_model(folder,state,quantized):
    # Read or serialize artifact data on disk (`folder`).
    folder=Path(folder)
    # Guard input contract (`folder.exists()`) and fail fast if violated.
    if folder.exists():raise ValueError('export destination must be new')
    # Run `folder.mkdir` to perform the next check or state transition.
    folder.mkdir(parents=True)
    # Loop over `batch` in `(1, 4)`:
    for batch in (1,4):
        # Cast or evaluate `artifact` in explicit floating-point precision.
        artifact=export.export(jax.jit(lambda x:forward(state['params'],x)))(jax.ShapeDtypeStruct((batch,8,8,1),jnp.float32))
        # Write the serialized artifact payload to disk.
        (folder/f'model-{batch}.jax').write_bytes(artifact.serialize())
    # Run `np.savez` to perform the next check or state transition.
    np.savez(folder/'integer.npz',**quantized)
    # Convert `` to a host NumPy array for inspection or verification.
    np.savez(folder/'float-params.npz',**{k:np.asarray(v) for k,v in state['params'].items()})
    # Initialize list `names` for the stage values.
    names=['model-1.jax','model-4.jax','integer.npz','float-params.npz']
    # Compute deterministic cryptographic digest `manifest` for provenance verification.
    manifest=dict(version=1,preprocess=PREPROCESS,classes=CLASSES,batches=[1,4],training_data_hash=state['data_hash'],training_settings=state['settings'],calibration_data_hash=str(quantized['calibration_hash']),calibration_percentile=float(quantized['percentile']),completed_steps=state['step'],jax_version=jax.__version__,validated_platform='cpu',integer_accumulator='int32',files={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in names})
    # Compute `(folder/'manifest.json').write_text(json.dumps(manifest,indent` as `2,sort_keys=True)+'\n')`.
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    # Return `manifest` to the caller.
    return manifest


# Function `load_model(folder, expected_preprocess)` implementing this stage's computation:
def load_model(folder,expected_preprocess=PREPROCESS):
    # Read or serialize artifact data on disk (`meta`).
    folder=Path(folder)
    meta=json.loads((folder/'manifest.json').read_text())
    # Guard input contract (`meta['version'] != 1 or meta['preprocess'] != expected_preprocess or meta['classes'] != CLASSES or (meta['batches'] != [1, 4]) or (meta['integer_accumulator'] != 'int32')`) and fail fast if violated.
    if meta['version']!=1 or meta['preprocess']!=expected_preprocess or meta['classes']!=CLASSES or meta['batches']!=[1,4] or meta['integer_accumulator']!='int32':raise ValueError('incompatible artifact contract')
    # Initialize list `names` for the stage values.
    names=['model-1.jax','model-4.jax','integer.npz','float-params.npz']
    # Guard input contract (`set(meta['files']) != set(names)`) and fail fast if violated.
    if set(meta['files'])!=set(names):raise ValueError('unexpected artifact files')
    # Loop over `name` in `names`:
    for name in names:
        # Guard input contract (`hashlib.sha256((folder / name).read_bytes()).hexdigest() != meta['files'][name]`) and fail fast if violated.
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=meta['files'][name]:raise ValueError('artifact integrity mismatch')
    # Construct dictionary `models` with the structured fields for this stage.
    models={b:export.deserialize((folder/f'model-{b}.jax').read_bytes()) for b in (1,4)}
    # Enter managed runtime/context scope for this block:
    # Execute `with np.load(folder/'integer.npz',allow_pickle=False) as f:q={k:f[k].copy() for `.
    with np.load(folder/'integer.npz',allow_pickle=False) as f:q={k:f[k].copy() for k in f.files}
    # Return `dict(models=models, quantized=q, manifest=meta)` to the caller.
    return dict(models=models,quantized=q,manifest=meta)


# Function `infer(runtime, images, layout, input_range, ...)` implementing this stage's computation:
def infer(runtime,images,layout='NHWC',input_range='uint8',precision='float32'):
    # Run `preprocess` to compute `x`.
    x=preprocess(images,layout,input_range)
    # Branch on condition `precision == 'int8'`:
    if precision=='int8':return integer_forward(runtime['quantized'],x)[0]
    # Guard input contract (`precision != 'float32'`) and fail fast if violated.
    if precision!='float32':raise ValueError('runtime supports float32 exported graph or int8 reference')
    # Initialize list `outputs` for the stage values.
    outputs=[]
    position=0
    while position<len(x):
        batch=4 if len(x)-position>=4 else 1
        result=runtime['models'][batch].call(jnp.asarray(x[position:position+batch]))
        result.block_until_ready()
        outputs.append(np.asarray(result))
        position+=batch
    # Return `np.concatenate(outputs)` to the caller.
    return np.concatenate(outputs)


# Function `benchmark(runtime, images, repeats, precision)` implementing this stage's computation:
def benchmark(runtime,images,repeats=12,precision='float32'):
    # Guard input contract (`repeats < 3`) and fail fast if violated.
    if repeats<3:raise ValueError('at least three timing repetitions required')
    # Record execution timing or profiler trace in `start`.
    # Record execution timing or profiler trace in `warmup`.
    start=time.perf_counter()
    infer(runtime,images,precision=precision)
    warmup=time.perf_counter()-start
    # Initialize list `samples` for the stage values.
    samples=[]
    # Repeat the update loop over `range(repeats)` steps:
    for _ in range(repeats):
        # Record execution timing or profiler trace in `start`.
        # Record execution timing or profiler trace in ``.
        start=time.perf_counter()
        infer(runtime,images,precision=precision)
        samples.append(time.perf_counter()-start)
    # Return `dict(precision=precision, backend=jax.default_backend(), device=str(jax.devices()[0]), batch=len(images), repeats=repeats, warmup_s=warmup, samples_s=samples, median_ms=1000 * float(np.median(samples)), p95_ms=1000 * float(np.percentile(samples, 95)), images_per_s=len(images) * repeats / sum(samples), scope='host preprocessing + synchronous inference; excludes file decoding')` to the caller.
    return dict(precision=precision,backend=jax.default_backend(),device=str(jax.devices()[0]),batch=len(images),repeats=repeats,warmup_s=warmup,samples_s=samples,median_ms=1000*float(np.median(samples)),p95_ms=1000*float(np.percentile(samples,95)),images_per_s=len(images)*repeats/sum(samples),scope='host preprocessing + synchronous inference; excludes file decoding')


# Function `select_runtime(candidate_folder, canary_images, expected_logits)` implementing this stage's computation:
def select_runtime(candidate_folder,canary_images,expected_logits):
    """Validate a candidate before the caller swaps its active runtime reference."""
    # Run `load_model` to compute `candidate`.
    candidate=load_model(candidate_folder)
    # Run `infer` to compute `actual`.
    actual=infer(candidate,canary_images)
    # Guard input contract (`np.shape(actual) != np.shape(expected_logits) or not np.allclose(actual, expected_logits, rtol=2e-05, atol=2e-05)`) and fail fast if violated.
    if np.shape(actual)!=np.shape(expected_logits) or not np.allclose(actual,expected_logits,rtol=2e-5,atol=2e-5):
        raise ValueError('candidate failed declared canary parity')
    # Return `candidate` to the caller.
    return candidate
