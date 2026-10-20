"""Export a computation and verify its serving contract: worked experiments and reference solutions. CPU checks."""

# Define an inference-only computation
# Step 1 — Define an inference-only computation: The probe returns [3.1,-0.45].
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import export
weights = jnp.array([[1., -2.], [.5, 1.], [-1., .25]], dtype=jnp.float32)
bias = jnp.array([.1, -.2], dtype=jnp.float32)
# Define and JIT-compile `inference(x)` so XLA traces and fuses the operations:
@jax.jit
def inference(x):
    return x @ weights + bias

# Create device-backed JAX array ``.
np.testing.assert_allclose(inference(jnp.array([[1.,2.,-1.]], jnp.float32)), [[3.1,-.45]], atol=1e-6)

# Declare the fixed input signature
# Step 2 — Declare the fixed input signature: The export has one input of shape (1,3).
signature = jax.ShapeDtypeStruct((1, 3), jnp.float32)
artifact = export.export(inference)(signature)

# Check tensor shape invariant: `artifact.in_avals[0].shape == (1,3)`
assert artifact.in_avals[0].shape == (1,3)

# Write bytes, reload and compare
# Step 3 — Write bytes, reload and compare: The output agrees with hand arithmetic.
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "dense.jaxexport"
    path.write_bytes(artifact.serialize())
    restored = export.deserialize(path.read_bytes())
    # Transfer the device result to the host for an independent NumPy comparison.
    sample = np.array([[1., 2., -1.]], np.float32)
    # Transfer the device result to the host for an independent NumPy comparison.
    actual = np.asarray(restored.call(sample))
    np.testing.assert_allclose(actual, [[3.1, -.45]], atol=1e-6)
    print("Verified serialized bytes:", path.stat().st_size)

# Export a computation and verify its serving contract: Export packages a computation for a supported runtime.
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import export
weights = jnp.array([[1., -2.], [.5, 1.], [-1., .25]], dtype=jnp.float32)
bias = jnp.array([.1, -.2], dtype=jnp.float32)
# Define and JIT-compile `inference(x)` so XLA traces and fuses the operations:
@jax.jit
def inference(x):
    return x @ weights + bias
signature = jax.ShapeDtypeStruct((1, 3), jnp.float32)
artifact = export.export(inference)(signature)
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "dense.jaxexport"
    path.write_bytes(artifact.serialize())
    restored = export.deserialize(path.read_bytes())
    # Transfer the device result to the host for an independent NumPy comparison.
    sample = np.array([[1., 2., -1.]], np.float32)
    # Transfer the device result to the host for an independent NumPy comparison.
    actual = np.asarray(restored.call(sample))
    np.testing.assert_allclose(actual, [[3.1, -.45]], atol=1e-6)
    print("Verified serialized bytes:", path.stat().st_size)

# Figure data experiment
# Transfer the device result to the host for an independent NumPy comparison.
direct = np.asarray(inference(sample))
visual_data = {'kind': 'bar', 'labels': ['score 0', 'score 1'], 'ylabel': 'output score', 'series': [{'label': 'direct', 'y': direct[0].tolist()}, {'label': 'restored artifact', 'y': actual[0].tolist()}]}

# Experiment: Reject the wrong batch
# Experiment — Reject the wrong batch: A Python function being shape-generic does not make its fixed...
try:
    restored.call(np.ones((2, 3), np.float32))
except (ValueError, TypeError) as error:
    print("Rejected incompatible shape:", type(error).__name__)
else:
    raise AssertionError("Fixed batch-one contract was not enforced")

# Experiment: Load only the exported bytes in a fresh interpreter
# Experiment — Load only the exported bytes in a fresh interpreter: The checked boundary is artifact bytes → installed JAX CPU...
import json
import os
import subprocess
import sys
# Transfer the device result to the host for an independent NumPy comparison.
worker = """import json,sys
from pathlib import Path
import numpy as np
from jax import export
loaded = export.deserialize(Path(sys.argv[1]).read_bytes())
request = np.asarray(json.load(sys.stdin)['features'], dtype=np.float32)
print(json.dumps({'scores': np.asarray(loaded.call(request)).tolist()}, allow_nan=False))
"""
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory() as directory:
    exported_path = Path(directory) / 'dense.jaxexport'
    exported_path.write_bytes(artifact.serialize())
    completed = subprocess.run([sys.executable, '-c', worker, str(exported_path)], input=json.dumps({'features': [[1.,2.,-1.]]}), text=True, capture_output=True, check=True, env=dict(os.environ, JAX_PLATFORMS='cpu'), timeout=60)
    worker_scores = json.loads(completed.stdout)['scores']
np.testing.assert_allclose(worker_scores, [[3.1, -.45]], atol=1e-6)
print('Fresh interpreter loaded bytes and matched the known scores.')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Verify a new input [-2, 0, 3] against an independent NumPy expression.
changed = np.array([[-2., 0., 3.]], np.float32)
# Transfer the device result to the host for an independent NumPy comparison.
expected = changed @ np.asarray(weights) + np.asarray(bias)
np.testing.assert_allclose(restored.call(changed), expected, atol=1e-6)
print("Changed exported request verified")

# Reference practice: Validate a request before inference
# Validate a request before inference (Transfer): Validation belongs before the runtime call.
def validate_request(values):
    # Transfer the device result to the host for an independent NumPy comparison.
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (1, 3) or not np.isfinite(array).all():
        raise ValueError("expected finite float32 [1, 3]")
    return array
for invalid in ([[1., 2.]], [[1., float("nan"), 3.]]):
    try:
        validate_request(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid input accepted")
np.testing.assert_allclose(restored.call(validate_request([[1.,2.,-1.]])), [[3.1,-.45]], atol=1e-6)

# Reference practice: Prove that request validation precedes exported inference
# Prove that request validation precedes exported inference (Transfer / diagnosis): The wrapper tests ordering as well as acceptance.
inference_calls = [0]
def checked_prediction(values):
    array = validate_request(values)
    # Accumulate the next contribution into `inference_calls[0]`.
    inference_calls[0] += 1
    return np.asarray(restored.call(array))
for invalid in ([[1., 2.]], [[1., float('inf'), 3.]]):
    try:
        checked_prediction(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid request reached inference')
assert inference_calls[0] == 0
np.testing.assert_allclose(checked_prediction([[1.,2.,-1.]]), [[3.1,-.45]], atol=1e-6)
assert inference_calls[0] == 1
print('Invalid requests rejected before the one valid inference call.')
print("PASS: deployment-03")
