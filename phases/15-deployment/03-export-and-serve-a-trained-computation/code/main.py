"""Export a computation and verify its serving contract: worked experiments and reference solutions. CPU checks."""

# Define an inference-only computation
# Step 1 — Define an inference-only computation: The probe returns [3.1,-0.45].
# Import tempfile for this computation.
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import export
# Construct `weights` via `jnp.array([[1., -2.], [.5, 1.], [-1., .25]], dtype=j...`
weights = jnp.array([[1., -2.], [.5, 1.], [-1., .25]], dtype=jnp.float32)
# Construct `bias` via `jnp.array([.1, -.2], dtype=jnp.float32)`
bias = jnp.array([.1, -.2], dtype=jnp.float32)
# Define and JIT-compile `inference(x)` so XLA traces and fuses the operations:
@jax.jit
# Function `inference(x)` implementing this stage's computation:
def inference(x):
    # Return `x @ weights + bias` to the caller.
    return x @ weights + bias

# Create device-backed JAX array ``.
np.testing.assert_allclose(inference(jnp.array([[1.,2.,-1.]], jnp.float32)), [[3.1,-.45]], atol=1e-6)

# Declare the fixed input signature
# Step 2 — Declare the fixed input signature: The export has one input of shape (1,3).
signature = jax.ShapeDtypeStruct((1, 3), jnp.float32)
# Run `export.export` to compute `artifact`.
artifact = export.export(inference)(signature)

# Check tensor shape invariant: `artifact.in_avals[0].shape == (1,3)`
assert artifact.in_avals[0].shape == (1,3)

# Write bytes, reload and compare
# Step 3 — Write bytes, reload and compare: The output agrees with hand arithmetic.
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory() as folder:
    # Read or serialize artifact data on disk (`path`).
    path = Path(folder) / "dense.jaxexport"
    # Run `path.write_bytes` to perform the next check or state transition.
    path.write_bytes(artifact.serialize())
    # Run `export.deserialize` to compute `restored`.
    restored = export.deserialize(path.read_bytes())
    # Convert `sample` to a host NumPy array for inspection or verification.
    sample = np.array([[1., 2., -1.]], np.float32)
    # Convert `actual` to a host NumPy array for inspection or verification.
    actual = np.asarray(restored.call(sample))
    # Check numerical equivalence within tolerance: `np.testing.assert_allclose(actual, [[3.1, -.45]], atol=1e-6)`
    np.testing.assert_allclose(actual, [[3.1, -.45]], atol=1e-6)
    # Print diagnostic summary of the computed outputs.
    print("Verified serialized bytes:", path.stat().st_size)

# Export a computation and verify its serving contract: Export packages a computation for a supported runtime.
# Import tempfile for this computation.
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import export
# Construct `weights` via `jnp.array([[1., -2.], [.5, 1.], [-1., .25]], dtype=j...`
weights = jnp.array([[1., -2.], [.5, 1.], [-1., .25]], dtype=jnp.float32)
# Construct `bias` via `jnp.array([.1, -.2], dtype=jnp.float32)`
bias = jnp.array([.1, -.2], dtype=jnp.float32)
# Define and JIT-compile `inference(x)` so XLA traces and fuses the operations:
@jax.jit
# Function `inference(x)` implementing this stage's computation:
def inference(x):
    # Return `x @ weights + bias` to the caller.
    return x @ weights + bias
# Cast or evaluate `signature` in explicit floating-point precision.
signature = jax.ShapeDtypeStruct((1, 3), jnp.float32)
# Run `export.export` to compute `artifact`.
artifact = export.export(inference)(signature)
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory() as folder:
    # Read or serialize artifact data on disk (`path`).
    path = Path(folder) / "dense.jaxexport"
    # Run `path.write_bytes` to perform the next check or state transition.
    path.write_bytes(artifact.serialize())
    # Run `export.deserialize` to compute `restored`.
    restored = export.deserialize(path.read_bytes())
    # Convert `sample` to a host NumPy array for inspection or verification.
    sample = np.array([[1., 2., -1.]], np.float32)
    # Convert `actual` to a host NumPy array for inspection or verification.
    actual = np.asarray(restored.call(sample))
    # Check numerical equivalence within tolerance: `np.testing.assert_allclose(actual, [[3.1, -.45]], atol=1e-6)`
    np.testing.assert_allclose(actual, [[3.1, -.45]], atol=1e-6)
    # Print diagnostic summary of the computed outputs.
    print("Verified serialized bytes:", path.stat().st_size)

# Figure data experiment
# Compute figure data for: Serialization preserves the declared computation
# Convert `direct` to a host NumPy array for inspection or verification.
direct = np.asarray(inference(sample))
# Compute `visual_data` from `{'kind': 'bar', 'labels': ['score 0', 'score 1'], 'y...`
visual_data = {'kind': 'bar', 'labels': ['score 0', 'score 1'], 'ylabel': 'output score', 'series': [{'label': 'direct', 'y': direct[0].tolist()}, {'label': 'restored artifact', 'y': actual[0].tolist()}]}

# Experiment: Reject the wrong batch
# Experiment — Reject the wrong batch: A Python function being shape-generic does not make its fixed...
# Run the boundary check and catch the expected exception:
try:
    restored.call(np.ones((2, 3), np.float32))
except (ValueError, TypeError) as error:
    print("Rejected incompatible shape:", type(error).__name__)
else:
    raise AssertionError("Fixed batch-one contract was not enforced")

# Experiment: Load only the exported bytes in a fresh interpreter
# Experiment — Load only the exported bytes in a fresh interpreter: The checked boundary is artifact bytes → installed JAX CPU...
# Import json for this computation.
import json
import os
import subprocess
import sys
# Convert `worker` to a host NumPy array for inspection or verification.
worker = """import json,sys
from pathlib import Path
import numpy as np
from jax import export
loaded = export.deserialize(Path(sys.argv[1]).read_bytes())
request = np.asarray(json.load(sys.stdin)['features'], dtype=np.float32)
print(json.dumps({'scores': np.asarray(loaded.call(request)).tolist()}, allow_nan=False))
"""
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory() as directory:
    # Read or serialize artifact data on disk (`exported_path`).
    exported_path = Path(directory) / 'dense.jaxexport'
    # Run `exported_path.write_bytes` to perform the next check or state transition.
    exported_path.write_bytes(artifact.serialize())
    # Configure environment variable before initializing the runtime.
    completed = subprocess.run([sys.executable, '-c', worker, str(exported_path)], input=json.dumps({'features': [[1.,2.,-1.]]}), text=True, capture_output=True, check=True, env=dict(os.environ, JAX_PLATFORMS='cpu'), timeout=60)
    # Read or serialize artifact data on disk (`worker_scores`).
    worker_scores = json.loads(completed.stdout)['scores']
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(worker_scores, [[3.1, -.45]], atol=1e-6)`
np.testing.assert_allclose(worker_scores, [[3.1, -.45]], atol=1e-6)
# Print the observed values to compare against the expected result.
print('Fresh interpreter loaded bytes and matched the known scores.')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Verify a new input [-2, 0, 3] against an independent NumPy expression.
# Compute `changed` from `np.array([[-2., 0., 3.]], np.float32)`
changed = np.array([[-2., 0., 3.]], np.float32)
# Convert `expected` to a host NumPy array for inspection or verification.
expected = changed @ np.asarray(weights) + np.asarray(bias)
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(restored.call(changed), expected, atol...`
np.testing.assert_allclose(restored.call(changed), expected, atol=1e-6)
# Print the observed values to compare against the expected result.
print("Changed exported request verified")

# Reference practice: Validate a request before inference
# Validate a request before inference (Transfer): Validation belongs before the runtime call.
def validate_request(values):
    # Convert `array` to a host NumPy array for inspection or verification.
    array = np.asarray(values, dtype=np.float32)
    # Guard input contract (`array.shape != (1, 3) or not np.isfinite(array).all()`) and fail fast if violated.
    if array.shape != (1, 3) or not np.isfinite(array).all():
        raise ValueError("expected finite float32 [1, 3]")
    # Return `array` to the caller.
    return array
# Iterate over `invalid` to step through the computation:
for invalid in ([[1., 2.]], [[1., float("nan"), 3.]]):
    try:
        validate_request(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid input accepted")
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(restored.call(validate_request([[1.,2....`
np.testing.assert_allclose(restored.call(validate_request([[1.,2.,-1.]])), [[3.1,-.45]], atol=1e-6)

# Reference practice: Prove that request validation precedes exported inference
# Prove that request validation precedes exported inference (Transfer / diagnosis): The wrapper tests ordering as well as acceptance.
inference_calls = [0]
# Function `checked_prediction(values)` implementing this stage's computation:
def checked_prediction(values):
    # Run `validate_request` to compute `array`.
    array = validate_request(values)
    # Accumulate the next contribution into `inference_calls[0]`.
    inference_calls[0] += 1
    # Return `np.asarray(restored.call(array))` to the caller.
    return np.asarray(restored.call(array))
# Iterate over `invalid` to step through the computation:
for invalid in ([[1., 2.]], [[1., float('inf'), 3.]]):
    try: checked_prediction(invalid)
    except ValueError: pass
    else: raise AssertionError('invalid request reached inference')
# Assert invariant `inference_calls[0] == 0` holds
assert inference_calls[0] == 0
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(checked_prediction([[1.,2.,-1.]]), [[3...`
np.testing.assert_allclose(checked_prediction([[1.,2.,-1.]]), [[3.1,-.45]], atol=1e-6)
# Assert invariant `inference_calls[0] == 1` holds
assert inference_calls[0] == 1
# Print the observed values to compare against the expected result.
print('Invalid requests rejected before the one valid inference call.')
print("PASS: deployment-03")
