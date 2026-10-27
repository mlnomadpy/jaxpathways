import hashlib
import itertools
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

project = Path('/Users/tahabsn/Documents/ChatGPT/jaxpathways/projects/audio-model-300m')
sys.path.insert(0, str(project))
import ctc


def objective_numpy(logits, target, valid):
    values = np.asarray(logits, np.float64)[:valid]
    probs = np.exp(values - values.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    probability = 0.0
    for path in itertools.product(range(3), repeat=valid):
        collapsed = [x for i, x in enumerate(path) if x != 0 and (i == 0 or x != path[i - 1])]
        if collapsed == target:
            probability += np.prod([probs[i, x] for i, x in enumerate(path)])
    return float(-np.log(probability)) if probability else float('inf')


rng = np.random.default_rng(7291)
cases = []
for valid in (1, 2, 3, 4, 5):
    for target in ([], [1], [2], [1, 1], [1, 2], [2, 1, 2]):
        logits = rng.normal(size=(6, 3)).astype(np.float32)
        labels = np.array([target + [0] * (3 - len(target))], np.int32)
        fn = lambda z: ctc.ctc_loss(z[None], jnp.array([valid]), labels, jnp.array([len(target)]))[0]
        actual = float(fn(jnp.asarray(logits)))
        expected = objective_numpy(logits, target, valid)
        if np.isfinite(expected):
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
            gradient = np.asarray(jax.grad(fn)(jnp.asarray(logits)))
            assert np.isfinite(gradient).all()
            np.testing.assert_array_equal(gradient[valid:], 0)
            errors = []
            for index in ((0, 0), (valid - 1, 2)):
                plus = logits.astype(np.float64); minus = plus.copy()
                plus[index] += 1e-4; minus[index] -= 1e-4
                fd = (objective_numpy(plus, target, valid) - objective_numpy(minus, target, valid)) / 2e-4
                errors.append(abs(fd - float(gradient[index])))
            assert max(errors) < 2e-6
            cases.append(dict(valid=valid, target=target, loss_error=abs(actual - expected), max_gradient_error=max(errors)))
        else:
            assert np.isposinf(actual)
            cases.append(dict(valid=valid, target=target, impossible=True))

edges = []
for valid, width in ((0, 1), (0, 0), (2, 0)):
    logits = np.zeros((3, 3), np.float32)
    try:
        actual = float(ctc.ctc_loss(logits[None], jnp.array([valid]), np.zeros((1, width), np.int32), jnp.array([0]))[0])
        edges.append(dict(valid=valid, label_width=width, actual=actual, expected=objective_numpy(logits, [], valid)))
    except Exception as exc:
        edges.append(dict(valid=valid, label_width=width, error=type(exc).__name__, message=str(exc)))

result = dict(source_sha256=hashlib.sha256((project / 'ctc.py').read_bytes()).hexdigest(), cases=cases, edge_cases=edges,
              scope='Independent enumeration of every three-symbol path, positive valid frame lengths, repeated and empty targets, finite differences and padded-frame gradients. Edge probes are reported separately.')
Path('/private/tmp/jaxpathways-c04-root-ctc-review/result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
