"""Image harness starter: deterministic fixture helpers are supplied; implement the lifecycle.
Do not import the instructor reference. Stages are cumulative and inspect changed conditions.
"""
"""Connected, inspectable CPU image harness; synthetic fixture, no vision benchmark."""
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

PREPROCESS=dict(version='image-v1',height=8,width=8,channels=1,source_channels=[1,3],
                resize='nearest floor-index',rgb_weights=[.2126,.7152,.0722],
                normalization='2 * gray_in_0_1 - 1',layout='NHWC',exif='transpose on file decode')
CLASSES=['vertical','horizontal','cross']



def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()

def digest_arrays(*arrays):
    h=hashlib.sha256()
    for value in arrays:
        a=np.ascontiguousarray(value)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()

def make_dataset(seed=11,count=96,split='train',corruption=False):
    if count<3:raise ValueError('at least three images required')
    rng=np.random.default_rng(seed)
    labels=np.arange(count,dtype=np.int32)%3
    images=np.empty((count,8,8,1),np.uint8)
    for i,label in enumerate(labels):
        canvas=rng.normal(18,5,(8,8))
        row,col=rng.integers(1,6,size=2)
        if label in (0,2):canvas[:,col:col+2]+=rng.uniform(170,210)
        if label in (1,2):canvas[row:row+2,:]+=rng.uniform(170,210)
        images[i,:,:,0]=np.clip(canvas,0,255).astype(np.uint8)
    if corruption:
        # Apply a separate random stream after clean generation: pairs remain aligned.
        altered=images.astype(np.float32)
        altered[:,:,2:6,:]=18
        altered+=np.random.default_rng(seed+5000).normal(0,28,altered.shape)
        images=np.clip(altered,0,255).astype(np.uint8)
    return dict(images=images,labels=labels,ids=[f'{split}-{seed}-{i:04d}' for i in range(count)],
                groups=[f'{split}-synthetic-{seed}-{i:04d}' for i in range(count)],
                provenance={'kind':'synthetic teaching fixture','seed':seed,'split':split,'corruption':corruption,'license':'original generated fixture; no external photographs'})

def dataset_hash(data):
    return digest_json(dict(arrays=digest_arrays(data['images'],data['labels']),ids=data['ids'],groups=data['groups'],provenance=data.get('provenance',{})))

def validate_splits(training,heldout):
    'Implement the contract described in README.md and the stage checks.'
    # Key APIs to use: `in`, `contract`, `np.asarray`, `np.issubdtype`, `np.any`
    # Step 1: Loop over `data` in `(training, heldout)`:
    # Step 2: Inside block: Run `len` to compute `n`.
    # Step 3: Inside block: Guard input contract (`np.asarray(data['labels']).shape != (n,) or len(data['ids']) != n or len(data['groups']) != n`) and fail fast if violated.
    # Step 4: Guard input contract (`set(training['ids']) & set(heldout['ids']) or set(training['groups']) & set(heldout['groups'])`) and fail fast if violated.
    # Step 5: Evaluate `train_pixels` from the current inputs and state.
    # Step 6: Guard input contract (`train_pixels & {digest_arrays(image) for image in heldout['images']}`) and fail fast if violated.
    raise NotImplementedError('Implement validate_splits')

def preprocess(images,layout='NHWC',input_range='uint8'):
    'Convert declared layouts/ranges to (B,8,8,1) float32.'
    # Key APIs to use: `np.asarray`, `contract`, `np.transpose`, `in`, `min`
    # Step 1: Convert `a` to a host NumPy array for inspection or verification.
    # Step 2: Guard input contract (`a.ndim != 4 or not len(a)`) and fail fast if violated.
    # Step 3: Branch on condition `layout == 'NCHW'`:
    # Step 4: Guard input contract (`a.shape[-1] not in (1, 3) or min(a.shape[1:3]) < 1`) and fail fast if violated.
    # Step 5: Branch on condition `input_range == 'uint8'`:
    # Step 6: Create evenly spaced index values in `rows`.
    raise NotImplementedError('Implement preprocess')

def read_image(path):
    'External PNG/JPEG/etc through Pillow, explicit EXIF transpose and RGB conversion.'
    # Key APIs to use: `PIL`, `Image.open`, `contract`, `getattr`, `np.asarray`
    # Step 1: Import required JAX, NumPy, and standard-library modules.
    # Step 2: Enter managed runtime/context scope for this block:
    # Step 3: Inside block: Guard input contract (`image.width * image.height > 16000000`) and fail fast if violated.
    # Step 4: Inside block: Guard input contract (`getattr(image, 'n_frames', 1) != 1`) and fail fast if violated.
    raise NotImplementedError('Implement read_image')

def load_external_manifest(path):
    'Load user-owned labeled images; relative paths stay inside manifest directory.\n\nJSON: split, source, license, items [{path,id,group,label}]. This loads supplied\nfiles only and never downloads data. Split related subjects/sources before use.'
    # Key APIs to use: `disk`, `Path`, `json.loads`, `path.read_text`, `parent.resolve`
    # Step 1: Read or serialize artifact data on disk (`path`).
    # Step 2: Read or serialize artifact data on disk (`manifest`).
    # Step 3: Run `path.parent.resolve` to compute `root`.
    # Step 4: Guard input contract (`not all((manifest.get(k) for k in ('split', 'source', 'license', 'items')))`) and fail fast if violated.
    # Step 5: Evaluate `images` from the current inputs and state.
    # Step 6: Evaluate `labels` from the current inputs and state.
    raise NotImplementedError('Implement load_external_manifest')

def init_params(seed=0):
    'Implement the contract described in README.md and the stage checks.'
    # Key APIs to use: `random.split`, `random.PRNGKey`, `random.normal`, `jnp.zeros`
    # Step 1: Initialize explicit deterministic PRNG key `(a, b)`.
    # Step 2: Return `dict(conv=jax.random.normal(a, (3, 3, 1, 6)) * 0.3, conv_bias=jnp.zeros(6), head=jax.random.normal(b, (6, 3)) * 0.2, head_bias=jnp.zeros(3))` to the caller.
    raise NotImplementedError('Implement init_params')

def forward(params,x,compute='float32',return_hidden=False):
    'Valid 3x3 convolution, ReLU, spatial mean and class logits.'
    # Key APIs to use: `lax.conv_general_dilated`, `jnp.asarray`, `astype`, `nn.relu`, `hidden.mean`
    # Step 1: Cast or evaluate `dtype` in explicit floating-point precision.
    # Step 2: Create device-backed JAX array `conv`.
    # Step 3: Run `jax.nn.relu` to compute `hidden`.
    # Step 4: Reduce across the target axis to summarize `pooled`.
    # Step 5: Cast or evaluate `logits` in explicit floating-point precision.
    # Step 6: Return `(logits, pooled) if return_hidden else logits` to the caller.
    raise NotImplementedError('Implement forward')

def objective(params,x,labels):
    'Implement the contract described in README.md and the stage checks.'
    # Key APIs to use: `jnp.asarray`, `contract`, `or`, `jnp.issubdtype`, `forward`
    # Step 1: Create device-backed JAX array `labels`.
    # Step 2: Guard input contract (`labels.ndim != 1 or labels.shape[0] != x.shape[0] or (not jnp.issubdtype(labels.dtype, jnp.integer))`) and fail fast if violated.
    # Step 3: Run `forward` to compute `logits`.
    # Step 4: Return `-jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), labels[:, None], axis=1))` to the caller.
    raise NotImplementedError('Implement objective')

def config(batch_size=12,learning_rate=.08,momentum=.9):
    if not isinstance(batch_size,int) or batch_size<1 or not 0<learning_rate<1 or not 0<=momentum<1:raise ValueError('invalid optimizer/input configuration')
    return dict(batch_size=batch_size,learning_rate=learning_rate,momentum=momentum)

def start(data,seed=0,settings=None):
    'Initialize parameters, momentum, key, permutation and data cursor.'
    # Key APIs to use: `config`, `init_params`, `random.split`, `random.PRNGKey`, `np.asarray`
    # Step 1: Run `config` to compute `settings`.
    # Step 2: Run `init_params` to compute `params`.
    # Step 3: Initialize explicit deterministic PRNG key `(key, order_key)`.
    # Step 4: Convert `order` to a host NumPy array for inspection or verification.
    # Step 5: Return `dict(params=params, velocity=jax.tree.map(jnp.zeros_like, params), key=key, order=order, position=0, epoch=0, step=0, seed=seed, data_hash=dataset_hash(data), settings=settings)` to the caller.
    raise NotImplementedError('Implement start')

def _update(params,velocity,key,x,y,learning_rate,momentum):
    'Implement the contract described in README.md and the stage checks.'
    # Key APIs to use: `subkeys`, `random.split`, `random.bernoulli`, `jnp.where`, `jnp.clip`
    # Step 1: Split the PRNG key deterministically into independent subkeys (`(key, noise_key, flip_key)`).
    # Step 2: Draw pseudorandom samples for `flips` using the explicit RNG state.
    # Step 3: Combine or mask array elements to form `augmented`.
    # Step 4: Draw pseudorandom samples for `augmented` using the explicit RNG state.
    # Step 5: Evaluate both scalar loss and parameter gradients in one pass (`(loss, grad)`).
    # Step 6: Apply leaf-wise transformation across the PyTree to produce `velocity`.
    raise NotImplementedError('Implement _update')

def advance(state,data,updates=1):
    'Complete actual augmented minibatch updates and preserve next-input state.'
    # Key APIs to use: `contract`, `dataset_hash`, `preprocess`, `random.split`, `np.asarray`
    # Step 1: Guard input contract (`dataset_hash(data) != state['data_hash']`) and fail fast if violated.
    # Step 2: Guard input contract (`not isinstance(updates, int) or updates < 1`) and fail fast if violated.
    # Step 3: Evaluate `state` and convert the result into Python scalar/collection `s`.
    # Step 4: Evaluate `records` from the current inputs and state.
    # Step 5: Run `len` to compute `n`.
    # Step 6: Evaluate `batch` from the current inputs and state.
    raise NotImplementedError('Implement advance')

def evaluate(params,data,batch_size=17,compute='float32'):
    'Implement the contract described in README.md and the stage checks.'
    # Key APIs to use: `contract`, `preprocess`, `np.asarray`, `np.issubdtype`, `np.any`
    # Step 1: Guard input contract (`batch_size < 1 or not len(data['images'])`) and fail fast if violated.
    # Step 2: Run `preprocess` to compute `x`.
    # Step 3: Convert `y` to a host NumPy array for inspection or verification.
    # Step 4: Guard input contract (`y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= 3))`) and fail fast if violated.
    # Step 5: Create device-backed JAX array `logits`.
    # Step 6: Return `metrics(logits, y, data['ids'])` to the caller.
    raise NotImplementedError('Implement evaluate')

def metrics(logits,labels,ids):
    'Implement the contract described in README.md and the stage checks.'
    # Key APIs to use: `np.asarray`, `contract`, `or`, `np.isfinite`, `all`
    # Step 1: Convert `logits` to a host NumPy array for inspection or verification.
    # Step 2: Convert `labels` to a host NumPy array for inspection or verification.
    # Step 3: Guard input contract (`logits.ndim != 2 or logits.shape[1] != 3 or len(logits) == 0 or (not np.isfinite(logits).all()) or (labels.shape != (len(logits),)) or (not np.issubdtype(labels.dtype, np.integer)) or np.any((labels < 0) | (labels >= 3)) or (len(ids) != len(logits))`) and fail fast if violated.
    # Step 4: Reduce across the target axis to summarize `shifted`.
    # Step 5: Create evenly spaced index values in `losses`.
    # Step 6: Run `np.argmax` to compute `predicted`.
    raise NotImplementedError('Implement metrics')

def save_checkpoint(path,state):
    'Implement the contract described in README.md and the stage checks.'
    # Key APIs to use: `disk`, `Path`, `parent.mkdir`, `np.asarray`, `items`
    # Step 1: Read or serialize artifact data on disk (`path`).
    # Step 2: Execute the next step of the computation.
    # Step 3: Convert `arrays` to a host NumPy array for inspection or verification.
    # Step 4: Evaluate `meta` from the current inputs and state.
    # Step 5: Execute the next step of the computation.
    # Step 6: Run `digest_json` to compute `meta['metadata_hash']`.
    raise NotImplementedError('Implement save_checkpoint')

def restore_checkpoint(path,data,settings=None):
    'Implement the contract described in README.md and the stage checks.'
    # Key APIs to use: `np.load`, `disk`, `copy`, `json.loads`, `meta.pop`
    # Step 1: Enter managed runtime/context scope for this block:
    # Step 2: Inside block: Evaluate `arrays` from the current inputs and state.
    # Step 3: Inside block: Read or serialize artifact data on disk (`meta`).
    # Step 4: Run `meta.pop` to compute `metadata_hash`.
    # Step 5: Guard input contract (`metadata_hash != digest_json(meta)`) and fail fast if violated.
    # Step 6: Guard input contract (`meta['jax_version'] != jax.__version__ or meta['numpy_version'] != np.__version__`) and fail fast if violated.
    raise NotImplementedError('Implement restore_checkpoint')

def patches(x):
    'Implement the contract described in README.md and the stage checks.'
    # Key APIs to use: `np.stack`, `reshape`
    # Step 1: Return `np.stack([x[:, i:i + 6, j:j + 6, :] for i in range(3) for j in range(3)], axis=-2).reshape(len(x), 6, 6, 9)` to the caller.
    raise NotImplementedError('Implement patches')

def quantize(values,scale):
    'Implement the contract described in README.md and the stage checks.'
    # Key APIs to use: `np.clip`, `np.rint`, `np.asarray`, `astype`
    # Step 1: Return `np.clip(np.rint(np.asarray(values) / scale), -127, 127).astype(np.int8)` to the caller.
    raise NotImplementedError('Implement quantize')

def calibrate(params,data,percentile=99.):
    'Fit per-output-channel weight scales and input/pooled activation scales on training data.'
    # Key APIs to use: `contract`, `preprocess`, `forward`, `jnp.asarray`, `max`
    # Step 1: Guard input contract (`not 0 < percentile <= 100`) and fail fast if violated.
    # Step 2: Run `preprocess` to compute `x`.
    # Step 3: Create device-backed JAX array `(_, hidden)`.
    # Step 4: Run `max` to compute `input_scale`.
    # Step 5: Run `max` to compute `hidden_scale`.
    # Step 6: Convert `conv` to a host NumPy array for inspection or verification.
    raise NotImplementedError('Implement calibrate')

def integer_forward(quantized,x,accumulator='int32'):
    'Perform int8 products with int32 accumulation; report activation clipping.'
    # Key APIs to use: `contract`, `np.asarray`, `quantize`, `patches`, `astype`
    # Step 1: Guard input contract (`accumulator != 'int32'`) and fail fast if violated.
    # Step 2: Evaluate `q` from the current inputs and state.
    # Step 3: Convert `raw` to a host NumPy array for inspection or verification.
    # Step 4: Run `quantize` to compute `ix`.
    # Step 5: Perform matrix contraction / projection to compute `acc`.
    # Step 6: Cast or evaluate `hidden` in explicit floating-point precision.
    raise NotImplementedError('Implement integer_forward')

def export_model(folder,state,quantized):
    'Serialize actual JAX graphs for batch sizes 1 and 4 plus integer data and preprocessing contract.'
    # Key APIs to use: `disk`, `Path`, `contract`, `folder.exists`, `folder.mkdir`
    # Step 1: Read or serialize artifact data on disk (`folder`).
    # Step 2: Guard input contract (`folder.exists()`) and fail fast if violated.
    # Step 3: Run `folder.mkdir` to perform the next check or state transition.
    # Step 4: Loop over `batch` in `(1, 4)`:
    # Step 5: Inside block: Cast or evaluate `artifact` in explicit floating-point precision.
    # Step 6: Inside block: Execute the next step of the computation.
    raise NotImplementedError('Implement export_model')

def load_model(folder,expected_preprocess=PREPROCESS):
    'Verify all hashes/contracts and deserialize JAX computations.'
    # Key APIs to use: `disk`, `Path`, `json.loads`, `read_text`, `contract`
    # Step 1: Read or serialize artifact data on disk (`folder`).
    # Step 2: Read or serialize artifact data on disk (`meta`).
    # Step 3: Guard input contract (`meta['version'] != 1 or meta['preprocess'] != expected_preprocess or meta['classes'] != CLASSES or (meta['batches'] != [1, 4]) or (meta['integer_accumulator'] != 'int32')`) and fail fast if violated.
    # Step 4: Evaluate `names` from the current inputs and state.
    # Step 5: Guard input contract (`set(meta['files']) != set(names)`) and fail fast if violated.
    # Step 6: Loop over `name` in `names`:
    raise NotImplementedError('Implement load_model')

def infer(runtime,images,layout='NHWC',input_range='uint8',precision='float32'):
    'Validate/preprocess raw images and call the selected runtime.'
    # Key APIs to use: `preprocess`, `integer_forward`, `contract`, `call`, `jnp.asarray`
    # Step 1: Run `preprocess` to compute `x`.
    # Step 2: Branch on condition `precision == 'int8'`:
    # Step 3: Guard input contract (`precision != 'float32'`) and fail fast if violated.
    # Step 4: Evaluate `outputs` from the current inputs and state.
    # Step 5: Evaluate `position` from the current inputs and state.
    # Step 6: Return `np.concatenate(outputs)` to the caller.
    raise NotImplementedError('Implement infer')

def benchmark(runtime,images,repeats=12,precision='float32'):
    'Warm up and repeatedly synchronize measured inference.'
    # Key APIs to use: `contract`, `time.perf_counter`, `infer`, `samples.append`, `jax.default_backend`
    # Step 1: Guard input contract (`repeats < 3`) and fail fast if violated.
    # Step 2: Record execution timing or profiler trace in `start`.
    # Step 3: Execute the next step of the computation.
    # Step 4: Record execution timing or profiler trace in `warmup`.
    # Step 5: Evaluate `samples` from the current inputs and state.
    # Step 6: Repeat the update loop over `range(repeats)` steps:
    raise NotImplementedError('Implement benchmark')

def select_runtime(candidate_folder,canary_images,expected_logits):
    'Validate a candidate before the caller swaps its active runtime reference.'
    # Key APIs to use: `load_model`, `infer`, `contract`, `np.shape`, `np.allclose`
    # Step 1: Run `load_model` to compute `candidate`.
    # Step 2: Run `infer` to compute `actual`.
    # Step 3: Guard input contract (`np.shape(actual) != np.shape(expected_logits) or not np.allclose(actual, expected_logits, rtol=2e-05, atol=2e-05)`) and fail fast if violated.
    # Step 4: Return `candidate` to the caller.
    raise NotImplementedError('Implement select_runtime')
