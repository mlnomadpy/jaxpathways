"""Inspectable image/text retrieval lifecycle; synthetic fixture, CPU reference."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import export
TRAINING = {'optimizer': 'momentum', 'learning_rate': 0.015, 'momentum': 0.85, 'temperature': 0.2, 'normalization_floor': 1e-06}

def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CONTRACT = {'schema': 1, 'image_shape': [8, 8], 'image_range': [0.0, 1.0], 'vocabulary': ['vertical', 'horizontal', 'thin', 'thick'], 'embedding_width': 4, 'preprocessing': 'grayscale-f32/bag-v1'}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def text_features(captions):
    """Exactly one orientation and one thickness; reject missing/ambiguous concepts."""
    raise NotImplementedError('Implement text_features')

def image_features(images):
    """Implement the contract from the project guide."""
    raise NotImplementedError('Implement image_features')

def fixture(seed=0, per_class=12, shift=0):
    """Four semantic classes, multiple noisy paired observations per class."""
    if per_class < 1 or shift not in (-1, 0, 1):
        raise ValueError('invalid fixture configuration')
    rng = np.random.default_rng(seed)
    images, captions, labels = ([], [], [])
    for label in range(4):
        vertical, thick = (label < 2, label % 2 == 1)
        caption = ('vertical' if vertical else 'horizontal') + ' ' + ('thick' if thick else 'thin')
        for _ in range(per_class):
            x = np.zeros((8, 8), np.float32)
            start, stop = (2, 6) if thick else (3, 4)
            start, stop = (start + shift, stop + shift)
            if vertical:
                x[:, start:stop] = 1
            else:
                x[start:stop, :] = 1
            x = np.clip(x + rng.normal(0, 0.045, x.shape), 0, 1).astype(np.float32)
            images.append(x)
            captions.append(caption)
            labels.append(label)
    return {'images': np.stack(images), 'captions': captions, 'labels': np.asarray(labels, np.int32), 'ids': [f'synthetic:{seed}:{shift}:{i}' for i in range(4 * per_class)], 'provenance': {'kind': 'synthetic-bars', 'seed': seed, 'shift': shift, 'per_class': per_class}}

def dataset_hash(data):
    x = image_features(data['images'])
    t = text_features(data['captions'])
    labels = np.asarray(data['labels'])
    if labels.shape != (len(x),) or labels.dtype != np.int32 or np.any((labels < 0) | (labels >= 4)):
        raise ValueError('invalid semantic labels')
    if len(t) != len(x) or len(data['ids']) != len(x) or len(set(data['ids'])) != len(x):
        raise ValueError('pair count or ID contract')
    return hashlib.sha256(x.tobytes() + t.tobytes() + labels.tobytes() + digest(data['ids']).encode()).hexdigest()

def check_splits(train, held):
    dataset_hash(train)
    dataset_hash(held)
    if set(train['ids']) & set(held['ids']) or set(train.get('groups', [])) & set(held.get('groups', [])):
        raise ValueError('pair IDs leak across splits')

def init_params(seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    return {'image': jax.random.normal(keys[0], (64, 4)) * 0.1, 'text': jax.random.normal(keys[1], (4, 4)) * 0.1}

def normalize(x):
    return x / jnp.sqrt(jnp.maximum(jnp.sum(x * x, axis=-1, keepdims=True), 1e-12))

def embeddings(params, x, t):
    return (normalize(x @ params['image']), normalize(t @ params['text']))

def objective(params, x, t, labels):
    """Implement the contract from the project guide."""
    raise NotImplementedError('Implement objective')

@jax.jit
def _update(params, momentum, x, t, labels):
    value, grad = jax.value_and_grad(objective)(params, x, t, labels)
    velocity = jax.tree.map(lambda v, g: TRAINING['momentum'] * v + g, momentum, grad)
    params = jax.tree.map(lambda p, v: p - TRAINING['learning_rate'] * v, params, velocity)
    return (params, velocity, value)

def initial_state(data, seed=0, batch_size=16):
    if batch_size < 4 or batch_size > len(data['ids']):
        raise ValueError('batch size outside fixture')
    params = init_params(seed)
    key, shuffle = jax.random.split(jax.random.PRNGKey(seed + 123))
    return {'params': params, 'momentum': jax.tree.map(jnp.zeros_like, params), 'key': key, 'order': np.asarray(jax.random.permutation(shuffle, len(data['ids']))), 'cursor': 0, 'step': 0, 'data_hash': dataset_hash(data), 'batch_size': batch_size}

def transition(state, data):
    """Implement the contract from the project guide."""
    raise NotImplementedError('Implement transition')

def train(data, seed=0, steps=80):
    state = initial_state(data, seed)
    history = []
    for _ in range(steps):
        state, result = transition(state, data)
        history.append(result['loss'])
    return (state, history)

def evaluate(params, data):
    x, t = (image_features(data['images']), text_features(data['captions']))
    zi, zt = embeddings(params, jnp.asarray(x), jnp.asarray(t))
    scores = np.asarray(zi @ zt.T)
    labels = data['labels']
    pi, pt = (labels[scores.argmax(axis=1)], labels[scores.argmax(axis=0)])
    return {'count': len(labels), 'image_to_text_correct': int(np.sum(pi == labels)), 'text_to_image_correct': int(np.sum(pt == labels)), 'scores': scores, 'image_predictions': pi, 'text_predictions': pt}

def _host_state(state):
    return {'params': {k: np.asarray(v).tolist() for k, v in state['params'].items()}, 'momentum': {k: np.asarray(v).tolist() for k, v in state['momentum'].items()}, 'key': np.asarray(state['key']).tolist(), 'order': np.asarray(state['order']).tolist(), **{k: state[k] for k in ['cursor', 'step', 'data_hash', 'batch_size']}}

def save_checkpoint(folder, state):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    payload = {'contract': CONTRACT, 'training': TRAINING, 'implementation_hash': implementation_hash(), 'state': _host_state(state)}
    (folder / 'checkpoint.json').write_text(json.dumps({'payload': payload, 'sha256': digest(payload)}, allow_nan=False))

def load_checkpoint(folder, data, batch_size=16):
    """Implement the contract from the project guide."""
    raise NotImplementedError('Implement load_checkpoint')

def quantize_weights(w):
    w = np.asarray(w, np.float32)
    scale = np.maximum(np.max(np.abs(w), axis=0, keepdims=True) / 127, np.float32(1e-08))
    return (np.clip(np.rint(w / scale), -127, 127).astype(np.int8), scale.astype(np.float32))

def calibrate(params, train_data):
    return {'data_hash': dataset_hash(train_data), 'input_scales': {'image': float(np.max(np.abs(image_features(train_data['images']))) / 127), 'text': float(np.max(np.abs(text_features(train_data['captions']))) / 127)}}

def encoder(params, branch, policy, calibration):
    """Implement the contract from the project guide."""
    raise NotImplementedError('Implement encoder')

def export_artifact(folder, params, calibration, policy='fp32'):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    files = {}
    for branch, width in [('image', 64), ('text', 4)]:
        fn = encoder(params, branch, policy, calibration)
        for batch in [1, 8]:
            spec = jax.ShapeDtypeStruct((batch, width), jnp.float32)
            payload = export.export(jax.jit(fn))(spec).serialize()
            name = f'{branch}-{batch}.jax'
            (folder / name).write_bytes(payload)
            files[name] = hashlib.sha256(payload).hexdigest()
    metadata = {'contract': CONTRACT, 'policy': policy, 'calibration': calibration, 'files': files, 'parameter_hash': digest({k: np.asarray(v).tolist() for k, v in params.items()}), 'runtime': 'JAX CPU export; int32 accumulation for w8a8; no native-int8 speed claim'}
    (folder / 'manifest.json').write_text(json.dumps({'payload': metadata, 'sha256': digest(metadata)}, allow_nan=False))

def load_artifact(folder):
    folder = Path(folder)
    record = json.loads((folder / 'manifest.json').read_text())
    meta = record['payload']
    names = {f'{branch}-{batch}.jax' for branch in ['image', 'text'] for batch in [1, 8]}
    if record['sha256'] != digest(meta) or meta['contract'] != CONTRACT or set(meta['files']) != names:
        raise ValueError('artifact metadata mismatch')
    loaded = {}
    for name, checksum in meta['files'].items():
        payload = (folder / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError('artifact checksum mismatch')
        loaded[name] = export.deserialize(payload)
    return {'meta': meta, 'models': loaded}

def infer(artifact, *, images=None, captions=None):
    """Implement the contract from the project guide."""
    raise NotImplementedError('Implement infer')

def measure(artifact, images, repeats=30):
    if repeats < 2:
        raise ValueError('need repeated observations')
    infer(artifact, images=images)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        infer(artifact, images=images)
        samples.append((time.perf_counter() - start) * 1000)
    return {'milliseconds': samples, 'p50_ms': float(np.median(samples)), 'p95_ms': float(np.percentile(samples, 95)), 'batch': len(images), 'scope': 'resident NumPy inputs through validation/placement/exported inference/host output; no file decode or network'}

def activate(registry, candidate, evaluation_images, evaluation_labels, minimum_accuracy=0.75):
    """Measured fixture quality gate before an atomic local pointer change."""
    raise NotImplementedError('Implement activate')

def load_pairs(manifest_path, split):
    """Read explicitly licensed local .npy image/caption pairs; no downloads."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if split not in ['train', 'held'] or not manifest.get('provenance') or (not manifest.get('license')):
        raise ValueError('split, provenance and rights statement required')
    records = manifest.get('records', [])
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate or missing pair IDs')
    train_groups = {r['group'] for r in records if r['split'] == 'train'}
    held_groups = {r['group'] for r in records if r['split'] == 'held'}
    if train_groups & held_groups:
        raise ValueError('source groups leak across splits')
    rows = [r for r in records if r['split'] == split]
    if not rows:
        raise ValueError('empty split')
    images = []
    captions = []
    labels = []
    for r in rows:
        image_path = (path.parent / r['image']).resolve()
        if not image_path.is_relative_to(path.parent):
            raise ValueError('image must be inside the dataset folder')
        pixels = np.load(image_path, allow_pickle=False)
        image_features(pixels[None, ...])
        feat = text_features([r['caption']])[0]
        label = (0 if feat[0] else 2) + (1 if feat[3] else 0)
        images.append(pixels)
        captions.append(r['caption'])
        labels.append(label)
    data = {'images': np.stack(images), 'captions': captions, 'labels': np.asarray(labels, np.int32), 'ids': [r['id'] for r in rows], 'groups': [r['group'] for r in rows], 'provenance': {'source': manifest['provenance'], 'license': manifest['license'], 'split': split}}
    dataset_hash(data)
    return data
