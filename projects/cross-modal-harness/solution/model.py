"""Inspectable image/text retrieval lifecycle; synthetic fixture, CPU reference."""
# Import pathlib (Path) for this computation.
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import export

# Construct dictionary `TRAINING` with the structured fields for this stage.
TRAINING = {'optimizer': 'momentum', 'learning_rate': 0.015, 'momentum': 0.85, 'temperature': 0.2, 'normalization_floor': 1e-6}

# Function `implementation_hash()` implementing this stage's computation:
def implementation_hash():
    # Return `hashlib.sha256(Path(__file__).read_bytes()).hexdigest()` to the caller.
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# Construct dictionary `CONTRACT` with the structured fields for this stage.
CONTRACT = {'schema': 1, 'image_shape': [8, 8], 'image_range': [0., 1.],
            'vocabulary': ['vertical', 'horizontal', 'thin', 'thick'],
            'embedding_width': 4, 'preprocessing': 'grayscale-f32/bag-v1'}

# Function `digest(value)` implementing this stage's computation:
def digest(value):
    # Return `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()` to the caller.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

# Function `text_features(captions)` implementing this stage's computation:
def text_features(captions):
    """Exactly one orientation and one thickness; reject missing/ambiguous concepts."""
    # Initialize list `result` for the stage values.
    result = []
    # Iterate over `caption` to step through the computation:
    for caption in captions:
        # Guard input contract (`not isinstance(caption, str)`) and fail fast if violated.
        if not isinstance(caption, str):
            raise ValueError('caption must be text')
        # Trace or lower the function to inspect its compiler representation (`words`).
        words = caption.lower().split()
        # Guard input contract (`len(words) != 2 or len(set(words)) != 2`) and fail fast if violated.
        if len(words) != 2 or len(set(words)) != 2:
            raise ValueError('caption needs exactly two different concepts')
        # Guard input contract (`sum((w in words for w in ['vertical', 'horizontal'])) != 1 or sum((w in words for w in ['thin', 'thick'])) != 1`) and fail fast if violated.
        if sum(w in words for w in ['vertical', 'horizontal']) != 1 or sum(w in words for w in ['thin', 'thick']) != 1:
            raise ValueError('caption requires one orientation and one thickness')
        # Guard input contract (`any((w not in CONTRACT['vocabulary'] for w in words))`) and fail fast if violated.
        if any(w not in CONTRACT['vocabulary'] for w in words):
            raise ValueError('unknown concept')
        # Append the current step result to `result`.
        result.append([float(w in words) for w in CONTRACT['vocabulary']])
    # Guard input contract (`not result`) and fail fast if violated.
    if not result:
        raise ValueError('empty caption batch')
    # Return `np.asarray(result, np.float32)` to the caller.
    return np.asarray(result, np.float32)

# Function `image_features(images)` implementing this stage's computation:
def image_features(images):
    # Convert `x` to a host NumPy array for inspection or verification.
    x = np.asarray(images)
    # Guard input contract (`x.dtype != np.float32 or x.ndim != 3 or x.shape[1:] != (8, 8) or (not len(x))`) and fail fast if violated.
    if x.dtype != np.float32 or x.ndim != 3 or x.shape[1:] != (8, 8) or not len(x):
        raise ValueError('images require nonempty float32 (batch,8,8)')
    # Guard input contract (`not np.isfinite(x).all() or np.min(x) < 0 or np.max(x) > 1`) and fail fast if violated.
    if not np.isfinite(x).all() or np.min(x) < 0 or np.max(x) > 1:
        raise ValueError('image range must be finite [0,1]')
    # Return `x.reshape(len(x), 64)` to the caller.
    return x.reshape(len(x), 64)

# Function `fixture(seed, per_class, shift)` implementing this stage's computation:
def fixture(seed=0, per_class=12, shift=0):
    """Four semantic classes, multiple noisy paired observations per class."""
    # Guard input contract (`per_class < 1 or shift not in (-1, 0, 1)`) and fail fast if violated.
    if per_class < 1 or shift not in (-1, 0, 1):
        raise ValueError('invalid fixture configuration')
    # Draw pseudorandom samples for `rng` using the explicit RNG state.
    rng = np.random.default_rng(seed)
    # Initialize list `images, captions, labels` for the stage values.
    images, captions, labels = [], [], []
    # Iterate over `label` to step through the computation:
    for label in range(4):
        # Compute `vertical, thick` as `label < 2, label % 2 == 1`.
        vertical, thick = label < 2, label % 2 == 1
        # Evaluate the compound expression for `caption`.
        caption = ('vertical' if vertical else 'horizontal') + ' ' + ('thick' if thick else 'thin')
        # Repeat the update loop over `range(per_class)` steps:
        for _ in range(per_class):
            # Allocate initialized array `x` with the specified shape and dtype.
            x = np.zeros((8, 8), np.float32)
            # Evaluate the compound expression for `start, stop`.
            start, stop = (2, 6) if thick else (3, 4)
            # Compute `start, stop` as `start + shift, stop + shift`.
            start, stop = start + shift, stop + shift
            # Branch on condition `vertical`:
            if vertical: x[:, start:stop] = 1
            else:
                x[start:stop, :] = 1
            # Cast or evaluate `x` in explicit floating-point precision.
            x = np.clip(x + rng.normal(0, 0.045, x.shape), 0, 1).astype(np.float32)
            # Append the current step result to `images`.
            # Append the current step result to `images`.
            # Append the current step result to `images`.
            images.append(x)
            captions.append(caption)
            labels.append(label)
    # Return `{'images': np.stack(images), 'captions': captions, 'labels': np.asarray(labels, np.int32), 'ids': [f'synthetic:{seed}:{shift}:{i}' for i in range(4 * per_class)], 'provenance': {'kind': 'synthetic-bars', 'seed': seed, 'shift': shift, 'per_class': per_class}}` to the caller.
    return {'images': np.stack(images), 'captions': captions, 'labels': np.asarray(labels, np.int32),
            'ids': [f'synthetic:{seed}:{shift}:{i}' for i in range(4 * per_class)],
            'provenance': {'kind': 'synthetic-bars', 'seed': seed, 'shift': shift, 'per_class': per_class}}

# Function `dataset_hash(data)` implementing this stage's computation:
def dataset_hash(data):
    # Run `image_features` to compute `x`.
    # Run `text_features` to compute `t`.
    x = image_features(data['images'])
    t = text_features(data['captions'])
    # Convert `labels` to a host NumPy array for inspection or verification.
    labels = np.asarray(data['labels'])
    # Guard input contract (`labels.shape != (len(x),) or labels.dtype != np.int32 or np.any((labels < 0) | (labels >= 4))`) and fail fast if violated.
    if labels.shape != (len(x),) or labels.dtype != np.int32 or np.any((labels < 0) | (labels >= 4)):
        raise ValueError('invalid semantic labels')
    # Guard input contract (`len(t) != len(x) or len(data['ids']) != len(x) or len(set(data['ids'])) != len(x)`) and fail fast if violated.
    if len(t) != len(x) or len(data['ids']) != len(x) or len(set(data['ids'])) != len(x):
        raise ValueError('pair count or ID contract')
    # Return `hashlib.sha256(x.tobytes() + t.tobytes() + labels.tobytes() + digest(data['ids']).encode()).hexdigest()` to the caller.
    return hashlib.sha256(x.tobytes() + t.tobytes() + labels.tobytes() + digest(data['ids']).encode()).hexdigest()

# Function `check_splits(train, held)` implementing this stage's computation:
def check_splits(train, held):
    # Run `dataset_hash` to perform the next check or state transition.
    # Run `dataset_hash` to perform the next check or state transition.
    dataset_hash(train)
    dataset_hash(held)
    # Guard input contract (`set(train['ids']) & set(held['ids']) or set(train.get('groups', [])) & set(held.get('groups', []))`) and fail fast if violated.
    if set(train['ids']) & set(held['ids']) or set(train.get('groups',[])) & set(held.get('groups',[])):
        raise ValueError('pair IDs leak across splits')

# Function `init_params(seed)` implementing this stage's computation:
def init_params(seed=0):
    # Initialize explicit deterministic PRNG key `keys`.
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    # Return `{'image': jax.random.normal(keys[0], (64, 4)) * 0.1, 'text': jax.random.normal(keys[1], (4, 4)) * 0.1}` to the caller.
    return {'image': jax.random.normal(keys[0], (64, 4)) * 0.1,
            'text': jax.random.normal(keys[1], (4, 4)) * 0.1}

# Function `normalize(x)` implementing this stage's computation:
def normalize(x):
    # Floor squared norm before sqrt so the derivative is finite at a zero vector.
    return x / jnp.sqrt(jnp.maximum(jnp.sum(x*x, axis=-1, keepdims=True), 1e-12))

# Function `embeddings(params, x, t)` implementing this stage's computation:
def embeddings(params, x, t):
    # Return `(normalize(x @ params['image']), normalize(t @ params['text']))` to the caller.
    return normalize(x @ params['image']), normalize(t @ params['text'])

# Function `objective(params, x, t, labels)` implementing this stage's computation:
def objective(params, x, t, labels):
    # Run `embeddings` to compute `(zi, zt)`.
    zi, zt = embeddings(params, x, t)
    # Perform matrix contraction / projection to compute `scores`.
    scores = zi @ zt.T / TRAINING['temperature']
    # Compute `positive` as `labels[:, None] == labels[None, :]`.
    positive = labels[:, None] == labels[None, :]
    # All captions/images of the same semantic class are positives, not false negatives.
    row = jax.scipy.special.logsumexp(scores, axis=1) - jax.scipy.special.logsumexp(jnp.where(positive, scores, -jnp.inf), axis=1)
    # Evaluate numerically stable log-space cross-entropy/likelihood (`col`).
    col = jax.scipy.special.logsumexp(scores, axis=0) - jax.scipy.special.logsumexp(jnp.where(positive, scores, -jnp.inf), axis=0)
    # Return `(jnp.mean(row) + jnp.mean(col)) / 2` to the caller.
    return (jnp.mean(row) + jnp.mean(col)) / 2

@jax.jit
# Function `_update(params, momentum, x, t, ...)` implementing this stage's computation:
def _update(params, momentum, x, t, labels):
    # Evaluate both scalar loss and parameter gradients in one pass (`(value, grad)`).
    value, grad = jax.value_and_grad(objective)(params, x, t, labels)
    # Apply leaf-wise transformation across the PyTree to produce `velocity`.
    velocity = jax.tree.map(lambda v, g: TRAINING['momentum'] * v + g, momentum, grad)
    # Apply leaf-wise transformation across the PyTree to produce `params`.
    params = jax.tree.map(lambda p, v: p - TRAINING['learning_rate'] * v, params, velocity)
    # Return `(params, velocity, value)` to the caller.
    return params, velocity, value

# Function `initial_state(data, seed, batch_size)` implementing this stage's computation:
def initial_state(data, seed=0, batch_size=16):
    # Guard input contract (`batch_size < 4 or batch_size > len(data['ids'])`) and fail fast if violated.
    if batch_size < 4 or batch_size > len(data['ids']):
        raise ValueError('batch size outside fixture')
    # Run `init_params` to compute `params`.
    params = init_params(seed)
    # Initialize explicit deterministic PRNG key `(key, shuffle)`.
    key, shuffle = jax.random.split(jax.random.PRNGKey(seed + 123))
    # Return `{'params': params, 'momentum': jax.tree.map(jnp.zeros_like, params), 'key': key, 'order': np.asarray(jax.random.permutation(shuffle, len(data['ids']))), 'cursor': 0, 'step': 0, 'data_hash': dataset_hash(data), 'batch_size': batch_size}` to the caller.
    return {'params': params, 'momentum': jax.tree.map(jnp.zeros_like, params), 'key': key,
            'order': np.asarray(jax.random.permutation(shuffle, len(data['ids']))),
            'cursor': 0, 'step': 0, 'data_hash': dataset_hash(data), 'batch_size': batch_size}

# Function `transition(state, data)` implementing this stage's computation:
def transition(state, data):
    # Guard input contract (`state['data_hash'] != dataset_hash(data)`) and fail fast if violated.
    if state['data_hash'] != dataset_hash(data):
        raise ValueError('data identity changed')
    # Compute `cursor, order, key` as `state['cursor'], state['order'], state['key']`.
    cursor, order, key = state['cursor'], state['order'], state['key']
    # Branch on condition `cursor == len(order)`:
    if cursor == len(order):
        key, shuffle = jax.random.split(key)
        order = np.asarray(jax.random.permutation(shuffle, len(order)))
        cursor = 0
    # Compute `chosen` as `order[cursor:cursor + state['batch_size']]`.
    chosen = order[cursor:cursor + state['batch_size']]
    # Run `image_features` to compute `x`.
    # Run `text_features` to compute `t`.
    x = image_features(data['images'][chosen])
    t = text_features([data['captions'][i] for i in chosen])
    # Create device-backed JAX array `(params, momentum, loss)`.
    params, momentum, loss = _update(state['params'], state['momentum'], jnp.asarray(x), jnp.asarray(t), jnp.asarray(data['labels'][chosen]))
    # Synchronize host execution until asynchronous device computation completes.
    loss.block_until_ready()
    # Construct dictionary `new` with the structured fields for this stage.
    new = {**state, 'params': params, 'momentum': momentum, 'key': key, 'order': order,
           'cursor': cursor + len(chosen), 'step': state['step'] + 1}
    # Return `(new, {'loss': float(loss), 'ids': [data['ids'][i] for i in chosen], 'count': len(chosen)})` to the caller.
    return new, {'loss': float(loss), 'ids': [data['ids'][i] for i in chosen], 'count': len(chosen)}

# Function `train(data, seed, steps)` implementing this stage's computation:
def train(data, seed=0, steps=80):
    # Run `initial_state` to compute `state`.
    state = initial_state(data, seed)
    # Initialize list `history` for the stage values.
    history = []
    # Repeat the update loop over `range(steps)` steps:
    for _ in range(steps):
        # Run `transition` to compute `(state, result)`.
        state, result = transition(state, data)
        # Append the current step result to `history`.
        history.append(result['loss'])
    # Return `(state, history)` to the caller.
    return state, history

# Function `evaluate(params, data)` implementing this stage's computation:
def evaluate(params, data):
    # Compute `x, t` as `image_features(data['images']), text_features(data['captions'])`.
    x, t = image_features(data['images']), text_features(data['captions'])
    # Create device-backed JAX array `(zi, zt)`.
    zi, zt = embeddings(params, jnp.asarray(x), jnp.asarray(t))
    # Convert `scores` to a host NumPy array for inspection or verification.
    scores = np.asarray(zi @ zt.T)
    # Compute `labels` as `data['labels']`.
    labels = data['labels']
    # Compute `pi, pt` as `labels[scores.argmax(axis=1)], labels[scores.argmax(axis=0)]`.
    pi, pt = labels[scores.argmax(axis=1)], labels[scores.argmax(axis=0)]
    # Return `{'count': len(labels), 'image_to_text_correct': int(np.sum(pi == labels)), 'text_to_image_correct': int(np.sum(pt == labels)), 'scores': scores, 'image_predictions': pi, 'text_predictions': pt}` to the caller.
    return {'count': len(labels), 'image_to_text_correct': int(np.sum(pi == labels)),
            'text_to_image_correct': int(np.sum(pt == labels)),
            'scores': scores, 'image_predictions': pi, 'text_predictions': pt}

# Function `_host_state(state)` implementing this stage's computation:
def _host_state(state):
    # Return `{'params': {k: np.asarray(v).tolist() for k, v in state['params'].items()}, 'momentum': {k: np.asarray(v).tolist() for k, v in state['momentum'].items()}, 'key': np.asarray(state['key']).tolist(), 'order': np.asarray(state['order']).tolist(), **{k: state[k] for k in ['cursor', 'step', 'data_hash', 'batch_size']}}` to the caller.
    return {'params': {k: np.asarray(v).tolist() for k,v in state['params'].items()},
            'momentum': {k: np.asarray(v).tolist() for k,v in state['momentum'].items()},
            'key': np.asarray(state['key']).tolist(), 'order': np.asarray(state['order']).tolist(),
            **{k: state[k] for k in ['cursor', 'step', 'data_hash', 'batch_size']}}

# Function `save_checkpoint(folder, state)` implementing this stage's computation:
def save_checkpoint(folder, state):
    # Compute `folder` as `Path(folder)`.
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    # Construct dictionary `payload` with the structured fields for this stage.
    payload = {'contract': CONTRACT, 'training': TRAINING, 'implementation_hash': implementation_hash(), 'state': _host_state(state)}
    # Compute `(folder/'checkpoint.json').write_text(json.dumps({'payload': payload, 'sha256': digest(payload)}, allow_nan` as `False))`.
    (folder/'checkpoint.json').write_text(json.dumps({'payload': payload, 'sha256': digest(payload)}, allow_nan=False))

# Function `load_checkpoint(folder, data, batch_size)` implementing this stage's computation:
def load_checkpoint(folder, data, batch_size=16):
    # Read or serialize artifact data on disk (`record`).
    record = json.loads((Path(folder)/'checkpoint.json').read_text())
    # Compute `p` as `record['payload']`.
    p = record['payload']
    s = p['state']
    # Guard input contract (`digest(p) != record['sha256'] or p['contract'] != CONTRACT or p.get('training') != TRAINING or (p.get('implementation_hash') != implementation_hash()) or (s['data_hash'] != dataset_hash(data)) or (s['batch_size'] != batch_size)`) and fail fast if violated.
    if digest(p) != record['sha256'] or p['contract'] != CONTRACT or p.get('training') != TRAINING or p.get('implementation_hash') != implementation_hash() or s['data_hash'] != dataset_hash(data) or s['batch_size'] != batch_size:
        raise ValueError('checkpoint identity, configuration or checksum mismatch')
    # Guard input contract (`sorted(s['order']) != list(range(len(data['ids']))) or not 0 <= s['cursor'] <= len(s['order']) or s['step'] < 0`) and fail fast if violated.
    if sorted(s['order']) != list(range(len(data['ids']))) or not 0 <= s['cursor'] <= len(s['order']) or s['step'] < 0:
        raise ValueError('invalid iterator state')
    # Guard input contract (`len(s['key']) != 2`) and fail fast if violated.
    if len(s['key']) != 2:
        raise ValueError('invalid random state')
    # Loop over `group` in `['params', 'momentum']`:
    for group in ['params', 'momentum']:
        # Loop over `(k, shape)` in `[('image', (64, 4)), ('text', (4, 4))]`:
        for k, shape in [('image', (64,4)), ('text', (4,4))]:
            # Convert `v` to a host NumPy array for inspection or verification.
            v = np.asarray(s[group][k], np.float32)
            # Guard input contract (`v.shape != shape or not np.isfinite(v).all()`) and fail fast if violated.
            if v.shape != shape or not np.isfinite(v).all():
                raise ValueError('invalid parameter state')
            # Create device-backed JAX array `s[group][k]`.
            s[group][k] = jnp.asarray(v)
    # Create device-backed JAX array `s['key']`.
    # Convert `s['order']` to a host NumPy array for inspection or verification.
    s['key'] = jnp.asarray(s['key'], jnp.uint32)
    s['order'] = np.asarray(s['order'], np.int32)
    # Return `s` to the caller.
    return s

# Function `quantize_weights(w)` implementing this stage's computation:
def quantize_weights(w):
    # Convert `w` to a host NumPy array for inspection or verification.
    w = np.asarray(w, np.float32)
    # Cast or evaluate `scale` in explicit floating-point precision.
    scale = np.maximum(np.max(np.abs(w), axis=0, keepdims=True) / 127, np.float32(1e-8))
    # Return `(np.clip(np.rint(w / scale), -127, 127).astype(np.int8), scale.astype(np.float32))` to the caller.
    return np.clip(np.rint(w / scale), -127, 127).astype(np.int8), scale.astype(np.float32)

# Function `calibrate(params, train_data)` implementing this stage's computation:
def calibrate(params, train_data):
    # Return `{'data_hash': dataset_hash(train_data), 'input_scales': {'image': float(np.max(np.abs(image_features(train_data['images']))) / 127), 'text': float(np.max(np.abs(text_features(train_data['captions']))) / 127)}}` to the caller.
    return {'data_hash': dataset_hash(train_data),
            'input_scales': {'image': float(np.max(np.abs(image_features(train_data['images']))) / 127),
                             'text': float(np.max(np.abs(text_features(train_data['captions']))) / 127)}}

# Function `encoder(params, branch, policy, calibration)` implementing this stage's computation:
def encoder(params, branch, policy, calibration):
    # Guard input contract (`branch not in ['image', 'text'] or policy not in ['fp32', 'w8a32', 'w8a8']`) and fail fast if violated.
    if branch not in ['image', 'text'] or policy not in ['fp32','w8a32','w8a8']:
        raise ValueError('unsupported branch or precision')
    # Create device-backed JAX array `w`.
    # Run `quantize_weights` to compute `(qw, scale)`.
    w = jnp.asarray(params[branch])
    qw, scale = quantize_weights(w)
    # Branch on condition `policy == 'fp32'`:
    if policy == 'fp32': return lambda x: normalize(x @ w)
    # Branch on condition `policy == 'w8a32'`:
    if policy == 'w8a32': return lambda x: normalize(x @ (jnp.asarray(qw, jnp.float32) * jnp.asarray(scale)))
    # Compute `activation_scale` as `calibration['input_scales'][branch]`.
    activation_scale = calibration['input_scales'][branch]
    # Guard input contract (`not np.isfinite(activation_scale) or activation_scale <= 0`) and fail fast if violated.
    if not np.isfinite(activation_scale) or activation_scale <= 0:
        raise ValueError('invalid calibration')
    # Function `run(x)` implementing this stage's computation:
    def run(x):
        # Combine or mask array elements to form `qx`.
        qx = jnp.clip(jnp.rint(x / activation_scale), -127, 127).astype(jnp.int8)
        # Create device-backed JAX array `accumulation`.
        accumulation = qx.astype(jnp.int32) @ jnp.asarray(qw, jnp.int32)
        # Return `normalize(accumulation.astype(jnp.float32) * activation_scale * jnp.asarray(scale))` to the caller.
        return normalize(accumulation.astype(jnp.float32) * activation_scale * jnp.asarray(scale))
    # Return `run` to the caller.
    return run

# Function `export_artifact(folder, params, calibration, policy)` implementing this stage's computation:
def export_artifact(folder, params, calibration, policy='fp32'):
    # Compute `folder` as `Path(folder)`.
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    # Construct dictionary `files` with the structured fields for this stage.
    files = {}
    # Loop over `(branch, width)` in `[('image', 64), ('text', 4)]`:
    for branch, width in [('image',64),('text',4)]:
        # Run `encoder` to compute `fn`.
        fn = encoder(params, branch, policy, calibration)
        # Loop over `batch` in `[1, 8]`:
        for batch in [1,8]:
            # Cast or evaluate `spec` in explicit floating-point precision.
            spec = jax.ShapeDtypeStruct((batch, width), jnp.float32)
            # Compile and trace the function with XLA (`payload`).
            payload = export.export(jax.jit(fn))(spec).serialize()
            # Compute `name` as `f'{branch}-{batch}.jax'`.
            name = f'{branch}-{batch}.jax'
            # Write the serialized artifact payload to disk.
            (folder/name).write_bytes(payload)
            # Compute deterministic cryptographic digest `files[name]` for provenance verification.
            files[name] = hashlib.sha256(payload).hexdigest()
    # Convert `metadata` to a host NumPy array for inspection or verification.
    metadata = {'contract':CONTRACT,'policy':policy,'calibration':calibration,'files':files,
                'parameter_hash':digest({k:np.asarray(v).tolist() for k,v in params.items()}),
                'runtime':'JAX CPU export; int32 accumulation for w8a8; no native-int8 speed claim'}
    # Compute `(folder/'manifest.json').write_text(json.dumps({'payload':metadata,'sha256':digest(metadata)},allow_nan` as `False))`.
    (folder/'manifest.json').write_text(json.dumps({'payload':metadata,'sha256':digest(metadata)},allow_nan=False))

# Function `load_artifact(folder)` implementing this stage's computation:
def load_artifact(folder):
    # Compute `folder` as `Path(folder)`.
    folder = Path(folder)
    record = json.loads((folder/'manifest.json').read_text())
    meta = record['payload']
    # Construct dictionary `names` with the structured fields for this stage.
    names = {f'{branch}-{batch}.jax' for branch in ['image','text'] for batch in [1,8]}
    # Guard input contract (`record['sha256'] != digest(meta) or meta['contract'] != CONTRACT or set(meta['files']) != names`) and fail fast if violated.
    if record['sha256'] != digest(meta) or meta['contract'] != CONTRACT or set(meta['files']) != names:
        raise ValueError('artifact metadata mismatch')
    # Construct dictionary `loaded` with the structured fields for this stage.
    loaded = {}
    # Loop over `(name, checksum)` in `meta['files'].items()`:
    for name, checksum in meta['files'].items():
        # Evaluate the compound expression for `payload`.
        payload = (folder/name).read_bytes()
        # Guard input contract (`hashlib.sha256(payload).hexdigest() != checksum`) and fail fast if violated.
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError('artifact checksum mismatch')
        # Run `export.deserialize` to compute `loaded[name]`.
        loaded[name] = export.deserialize(payload)
    # Return `{'meta': meta, 'models': loaded}` to the caller.
    return {'meta':meta,'models':loaded}

# Function `infer(artifact)` implementing this stage's computation:
def infer(artifact, *, images=None, captions=None):
    # Guard input contract (`(images is None) == (captions is None)`) and fail fast if violated.
    if (images is None) == (captions is None):
        raise ValueError('supply exactly one modality per embedding request')
    # Compute `branch` as `'image' if images is not None else 'text'`.
    branch = 'image' if images is not None else 'text'
    # Run `image_features` to compute `x`.
    x = image_features(images) if images is not None else text_features(captions)
    # Compute `name` as `f'{branch}-{len(x)}.jax'`.
    name = f'{branch}-{len(x)}.jax'
    # Guard input contract (`name not in artifact['models']`) and fail fast if violated.
    if name not in artifact['models']:
        raise ValueError('supported batch sizes are 1 and 8')
    # Create device-backed JAX array `result`.
    result = artifact['models'][name].call(jnp.asarray(x))
    # Synchronize host execution until asynchronous device computation completes.
    result.block_until_ready()
    # Return `np.asarray(result)` to the caller.
    return np.asarray(result)

# Function `measure(artifact, images, repeats)` implementing this stage's computation:
def measure(artifact, images, repeats=30):
    # Guard input contract (`repeats < 2`) and fail fast if violated.
    if repeats < 2:
        raise ValueError('need repeated observations')
    # Run `infer` to perform the next check or state transition.
    infer(artifact, images=images)
    # Initialize list `samples` for the stage values.
    samples=[]
    # Repeat the update loop over `range(repeats)` steps:
    for _ in range(repeats):
        # Record execution timing or profiler trace in `start`.
        # Record execution timing or profiler trace in ``.
        start=time.perf_counter()
        infer(artifact,images=images)
        samples.append((time.perf_counter()-start)*1000)
    # Return `{'milliseconds': samples, 'p50_ms': float(np.median(samples)), 'p95_ms': float(np.percentile(samples, 95)), 'batch': len(images), 'scope': 'resident NumPy inputs through validation/placement/exported inference/host output; no file decode or network'}` to the caller.
    return {'milliseconds':samples,'p50_ms':float(np.median(samples)),'p95_ms':float(np.percentile(samples,95)),
            'batch':len(images),'scope':'resident NumPy inputs through validation/placement/exported inference/host output; no file decode or network'}

# Function `activate(registry, candidate, evaluation_images, evaluation_labels, ...)` implementing this stage's computation:
def activate(registry, candidate, evaluation_images, evaluation_labels, minimum_accuracy=0.75):
    """Measured fixture quality gate before an atomic local pointer change."""
    # Guard input contract (`not 0 <= minimum_accuracy <= 1`) and fail fast if violated.
    if not 0 <= minimum_accuracy <= 1:
        raise ValueError('invalid quality threshold')
    # Run `load_artifact` to compute `artifact`.
    artifact = load_artifact(candidate)
    # Guard input contract (`len(evaluation_images) != 8 or np.asarray(evaluation_labels).shape != (8,)`) and fail fast if violated.
    if len(evaluation_images) != 8 or np.asarray(evaluation_labels).shape != (8,):
        raise ValueError('quality gate needs eight declared labeled examples')
    # Initialize list `bank` for the stage values.
    bank = ['vertical thin','vertical thick','horizontal thin','horizontal thick'] * 2
    # Run `infer` to compute `zi`.
    # Run `infer` to compute `zt`.
    zi = infer(artifact, images=evaluation_images)
    zt = infer(artifact, captions=bank)
    # Create evenly spaced index values in `predicted`.
    predicted = np.arange(8)[(zi @ zt.T).argmax(axis=1)] % 4
    # Reduce across the target axis to summarize `score`.
    score = float(np.mean(predicted == evaluation_labels))
    # Guard input contract (`score < minimum_accuracy`) and fail fast if violated.
    if score < minimum_accuracy:
        raise ValueError('candidate failed measured retrieval gate')
    # Compute `registry` as `Path(registry)`.
    registry=Path(registry)
    registry.mkdir(parents=True,exist_ok=True)
    # Read or serialize artifact data on disk (`record`).
    record={'folder':str(Path(candidate).resolve()),'manifest_sha256':hashlib.sha256((Path(candidate)/'manifest.json').read_bytes()).hexdigest(),'gate_accuracy':score}
    # Compute `temporary` as `registry/'ACTIVE.tmp'`.
    temporary=registry/'ACTIVE.tmp'
    temporary.write_text(json.dumps(record))
    temporary.replace(registry/'ACTIVE.json')
    # Return `record` to the caller.
    return record

# Function `load_pairs(manifest_path, split)` implementing this stage's computation:
def load_pairs(manifest_path, split):
    """Read explicitly licensed local .npy image/caption pairs; no downloads."""
    # Read or serialize artifact data on disk (`manifest`).
    path=Path(manifest_path).resolve()
    manifest=json.loads(path.read_text())
    # Guard input contract (`split not in ['train', 'held'] or not manifest.get('provenance') or (not manifest.get('license'))`) and fail fast if violated.
    if split not in ['train','held'] or not manifest.get('provenance') or not manifest.get('license'):
        raise ValueError('split, provenance and rights statement required')
    # Run `manifest.get` to compute `records`.
    records=manifest.get('records',[])
    # Guard input contract (`not records or len({r['id'] for r in records}) != len(records)`) and fail fast if violated.
    if not records or len({r['id'] for r in records})!=len(records):raise ValueError('duplicate or missing pair IDs')
    # Construct dictionary `train_groups` with the structured fields for this stage.
    train_groups={r['group'] for r in records if r['split']=='train'}
    # Construct dictionary `held_groups` with the structured fields for this stage.
    held_groups={r['group'] for r in records if r['split']=='held'}
    # Guard input contract (`train_groups & held_groups`) and fail fast if violated.
    if train_groups & held_groups:raise ValueError('source groups leak across splits')
    # Initialize list `rows` for the stage values.
    rows=[r for r in records if r['split']==split]
    # Guard input contract (`not rows`) and fail fast if violated.
    if not rows:raise ValueError('empty split')
    # Initialize list `images` for the stage values.
    images=[]
    captions=[]
    labels=[]
    # Loop over `r` in `rows`:
    for r in rows:
        # Evaluate the compound expression for `image_path`.
        image_path=(path.parent/r['image']).resolve()
        # Guard input contract (`not image_path.is_relative_to(path.parent)`) and fail fast if violated.
        if not image_path.is_relative_to(path.parent):raise ValueError('image must be inside the dataset folder')
        # Run `np.load` to compute `pixels`.
        pixels=np.load(image_path,allow_pickle=False)
        # Run `image_features` to perform the next check or state transition.
        # Run `text_features` to compute `feat`.
        image_features(pixels[None,...])
        feat=text_features([r['caption']])[0]
        # Evaluate the compound expression for `label`.
        label=(0 if feat[0] else 2)+(1 if feat[3] else 0)
        # Append the current step result to `images`.
        # Append the current step result to `images`.
        # Append the current step result to `images`.
        images.append(pixels)
        captions.append(r['caption'])
        labels.append(label)
    # Convert `data` to a host NumPy array for inspection or verification.
    data={'images':np.stack(images),'captions':captions,'labels':np.asarray(labels,np.int32),
          'ids':[r['id'] for r in rows], 'groups':[r['group'] for r in rows],
          'provenance':{'source':manifest['provenance'],'license':manifest['license'],'split':split}}
    # Run `dataset_hash` to perform the next check or state transition.
    dataset_hash(data)
    # Return `data` to the caller.
    return data
