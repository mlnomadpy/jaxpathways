"""Move your experiment to a TPU: worked experiments and reference solutions. CPU checks."""

# Choose the requested platform
# Step 1 — Choose the requested platform: Keep the requested backend, observed output placement and...
# Import os for this computation.
import os
import json
import platform
import numpy as np
import jax
import jax.numpy as jnp

# Place inputs and compare a completed prediction
# Step 2 — Place inputs and compare a completed prediction: Keep the requested backend, observed output placement and...
expected = os.environ.get('COURSE_EXPECT_PLATFORM', 'cpu')
# Guard input contract (`expected not in {'cpu', 'tpu'}`) and fail fast if violated.
if expected not in {'cpu', 'tpu'}:
    raise ValueError('COURSE_EXPECT_PLATFORM must be cpu or tpu')
# Selecting a requested backend must fail when it is unavailable.
devices = jax.devices(expected)
# Verify contract: `devices and all((d.platform == expected for d in devices))`.
assert devices and all(d.platform == expected for d in devices)
# Construct and reshape `x_host` into the target tensor dimensions.
x_host = np.arange(24, dtype=np.float32).reshape(8, 3) / 8
# Initialize array `w_host` with explicit values and shape.
w_host = np.array([0.5, -0.25, 1.0], dtype=np.float32)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(x_host, devices[0])
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(w_host, devices[0])
# Wrap with `jax.jit` (`predict`) so XLA traces and compiles the function.
predict = jax.jit(lambda a, b: a @ b + jnp.float32(0.125))
# Run `predict` to compute `y`.
y = predict(x, w)
# Synchronize host execution until asynchronous device computation completes.
y.block_until_ready()
# Cast or evaluate `reference` in explicit floating-point precision.
reference = x_host @ w_host + np.float32(0.125)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(y), reference, rtol=1e-5, atol=1e-5)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert {d.platform for d in y.devices()} == {expected}
# Convert `report` to a host NumPy array for inspection or verification.
report = dict(expected=expected, output_devices=[str(d) for d in y.devices()],
              platform=next(iter(y.devices())).platform, jax=jax.__version__,
              python=platform.python_version(), shape=list(y.shape), dtype=str(y.dtype),
              max_absolute_error=float(np.max(np.abs(np.asarray(y)-reference))),
              process_index=jax.process_index(), process_count=jax.process_count())
# Print diagnostic summary of the computed outputs.
print(json.dumps(report, indent=2))
# Print diagnostic summary of the computed outputs.
print('Completed prediction:', np.asarray(y).tolist())

# Step 1 — Choose the requested platform: Keep the requested backend, observed output placement and...
# Import os for this computation.
import os
import json
import platform
import numpy as np
import jax
import jax.numpy as jnp

# Step 2 — Place inputs and compare a completed prediction: Keep the requested backend, observed output placement and...
expected = os.environ.get('COURSE_EXPECT_PLATFORM', 'cpu')
# Guard input contract (`expected not in {'cpu', 'tpu'}`) and fail fast if violated.
if expected not in {'cpu', 'tpu'}:
    raise ValueError('COURSE_EXPECT_PLATFORM must be cpu or tpu')
# Selecting a requested backend must fail when it is unavailable.
devices = jax.devices(expected)
# Verify contract: `devices and all((d.platform == expected for d in devices))`.
assert devices and all(d.platform == expected for d in devices)
# Construct and reshape `x_host` into the target tensor dimensions.
x_host = np.arange(24, dtype=np.float32).reshape(8, 3) / 8
# Initialize array `w_host` with explicit values and shape.
w_host = np.array([0.5, -0.25, 1.0], dtype=np.float32)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(x_host, devices[0])
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(w_host, devices[0])
# Wrap with `jax.jit` (`predict`) so XLA traces and compiles the function.
predict = jax.jit(lambda a, b: a @ b + jnp.float32(0.125))
# Run `predict` to compute `y`.
y = predict(x, w)
# Synchronize host execution until asynchronous device computation completes.
y.block_until_ready()
# Cast or evaluate `reference` in explicit floating-point precision.
reference = x_host @ w_host + np.float32(0.125)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(y), reference, rtol=1e-5, atol=1e-5)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert {d.platform for d in y.devices()} == {expected}
# Convert `report` to a host NumPy array for inspection or verification.
report = dict(expected=expected, output_devices=[str(d) for d in y.devices()],
              platform=next(iter(y.devices())).platform, jax=jax.__version__,
              python=platform.python_version(), shape=list(y.shape), dtype=str(y.dtype),
              max_absolute_error=float(np.max(np.abs(np.asarray(y)-reference))),
              process_index=jax.process_index(), process_count=jax.process_count())
# Print diagnostic summary of the computed outputs.
print(json.dumps(report, indent=2))
# Print diagnostic summary of the computed outputs.
print('Completed prediction:', np.asarray(y).tolist())

# Figure data experiment
# Compute figure data for: The same prediction checked two ways
# Convert `visual_data` to a host NumPy array for inspection or verification.
visual_data={'kind':'line','x':list(range(8)),'xlabel':'observation row','ylabel':'prediction','series':[{'label':'JAX: '+expected,'y':np.asarray(y).tolist()},{'label':'NumPy reference','y':reference.tolist()}]}

# Experiment: Reject a mismatched report
# Experiment — Reject a mismatched report: Numerical agreement and requested placement are separate...
def verify_report(record, required):
    # Guard input contract (`record['platform'] != required or record['expected'] != required`) and fail fast if violated.
    if record['platform'] != required or record['expected'] != required:
        raise ValueError('requested and observed platform disagree')
    # Guard input contract (`record['shape'] != [8] or record['dtype'] != 'float32'`) and fail fast if violated.
    if record['shape'] != [8] or record['dtype'] != 'float32':
        raise ValueError('prediction contract changed')
    # Return `True` to the caller.
    return True
# Verify contract: `verify_report(report, expected)`.
assert verify_report(report, expected)
# Evaluate `wrong` from the current inputs and state.
wrong = {**report, 'platform': 'cpu' if expected == 'tpu' else 'tpu'}
# Run the boundary check and catch the expected exception:
try:
    verify_report(wrong, expected)
except ValueError:
    print('Mismatched platform report rejected.')
else:
    raise AssertionError('a mismatched report was accepted')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Predict the effect of replacing the bias by minus one quarter.
# Wrap with `jax.jit` (`changed_predict`) so XLA traces and compiles the function.
changed_predict = jax.jit(lambda a, b: a @ b - jnp.float32(0.25))
# Run `changed_predict` to compute `changed`.
changed = changed_predict(x, w)
# Synchronize host execution until asynchronous device computation completes.
changed.block_until_ready()
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(changed), reference - 0.375, atol=1e-5, rtol=1e-5)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(changed), x_host @ w_host - 0.25, atol=1e-5, rtol=1e-5)
# Verify contract: `{d.platform for d in changed.devices()} == {expected}`.
assert {d.platform for d in changed.devices()} == {expected}
# Print the observed values to compare against the expected result.
print('Changed bias, first prediction:', float(changed[0]))

# Reference practice: Test row order without changing the model
# Test row order without changing the model (Transfer): Reversing rows should reverse predictions because each...
reversed_x = jax.device_put(x_host[::-1].copy(), devices[0])
# Run `predict` to compute `reversed_y`.
reversed_y = predict(reversed_x, w)
# Synchronize host execution until asynchronous device computation completes.
reversed_y.block_until_ready()
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(reversed_y), reference[::-1], atol=1e-5, rtol=1e-5)
# Verify contract: `{d.platform for d in reversed_y.devices()} == {expected}`.
assert {d.platform for d in reversed_y.devices()} == {expected}
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert np.isclose(float(reversed_y[0]), 3.625)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert np.isclose(float(reversed_y[-1]), 0.34375)
# Print the observed values to compare against the expected result.
print('Row reversal preserved values and requested placement.')
print("PASS: welcome-03")
