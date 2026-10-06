"""Move your experiment to a TPU: worked experiments and reference solutions. CPU checks."""

# Choose the requested platform
import os
import json
import platform
import numpy as np
import jax
import jax.numpy as jnp

# Place inputs and compare a completed prediction
expected = os.environ.get('COURSE_EXPECT_PLATFORM', 'cpu')
if expected not in {'cpu', 'tpu'}:
    raise ValueError('COURSE_EXPECT_PLATFORM must be cpu or tpu')
# Selecting a requested backend must fail when it is unavailable.
devices = jax.devices(expected)
assert devices and all(d.platform == expected for d in devices)
x_host = np.arange(24, dtype=np.float32).reshape(8, 3) / 8
w_host = np.array([0.5, -0.25, 1.0], dtype=np.float32)
x = jax.device_put(x_host, devices[0])
w = jax.device_put(w_host, devices[0])
predict = jax.jit(lambda a, b: a @ b + jnp.float32(0.125))
y = predict(x, w)
y.block_until_ready()
reference = x_host @ w_host + np.float32(0.125)
np.testing.assert_allclose(np.asarray(y), reference, rtol=1e-5, atol=1e-5)
assert {d.platform for d in y.devices()} == {expected}
report = dict(expected=expected, output_devices=[str(d) for d in y.devices()],
              platform=next(iter(y.devices())).platform, jax=jax.__version__,
              python=platform.python_version(), shape=list(y.shape), dtype=str(y.dtype),
              max_absolute_error=float(np.max(np.abs(np.asarray(y)-reference))),
              process_index=jax.process_index(), process_count=jax.process_count())
print(json.dumps(report, indent=2))
print('Completed prediction:', np.asarray(y).tolist())


import os
import json
import platform
import numpy as np
import jax
import jax.numpy as jnp

expected = os.environ.get('COURSE_EXPECT_PLATFORM', 'cpu')
if expected not in {'cpu', 'tpu'}:
    raise ValueError('COURSE_EXPECT_PLATFORM must be cpu or tpu')
# Selecting a requested backend must fail when it is unavailable.
devices = jax.devices(expected)
assert devices and all(d.platform == expected for d in devices)
x_host = np.arange(24, dtype=np.float32).reshape(8, 3) / 8
w_host = np.array([0.5, -0.25, 1.0], dtype=np.float32)
x = jax.device_put(x_host, devices[0])
w = jax.device_put(w_host, devices[0])
predict = jax.jit(lambda a, b: a @ b + jnp.float32(0.125))
y = predict(x, w)
y.block_until_ready()
reference = x_host @ w_host + np.float32(0.125)
np.testing.assert_allclose(np.asarray(y), reference, rtol=1e-5, atol=1e-5)
assert {d.platform for d in y.devices()} == {expected}
report = dict(expected=expected, output_devices=[str(d) for d in y.devices()],
              platform=next(iter(y.devices())).platform, jax=jax.__version__,
              python=platform.python_version(), shape=list(y.shape), dtype=str(y.dtype),
              max_absolute_error=float(np.max(np.abs(np.asarray(y)-reference))),
              process_index=jax.process_index(), process_count=jax.process_count())
print(json.dumps(report, indent=2))
print('Completed prediction:', np.asarray(y).tolist())


# Figure data experiment
visual_data={'kind':'line','x':list(range(8)),'xlabel':'observation row','ylabel':'prediction','series':[{'label':'JAX: '+expected,'y':np.asarray(y).tolist()},{'label':'NumPy reference','y':reference.tolist()}]}

# Experiment: Reject a mismatched report
def verify_report(record, required):
    if record['platform'] != required or record['expected'] != required:
        raise ValueError('requested and observed platform disagree')
    if record['shape'] != [8] or record['dtype'] != 'float32':
        raise ValueError('prediction contract changed')
    return True
assert verify_report(report, expected)
wrong = {**report, 'platform': 'cpu' if expected == 'tpu' else 'tpu'}
try:
    verify_report(wrong, expected)
except ValueError:
    print('Mismatched platform report rejected.')
else:
    raise AssertionError('a mismatched report was accepted')

# Reference solution. Try the exercise before reading this.
changed_predict = jax.jit(lambda a, b: a @ b - jnp.float32(0.25))
changed = changed_predict(x, w)
changed.block_until_ready()
np.testing.assert_allclose(np.asarray(changed), reference - 0.375, atol=1e-5, rtol=1e-5)
np.testing.assert_allclose(np.asarray(changed), x_host @ w_host - 0.25, atol=1e-5, rtol=1e-5)
assert {d.platform for d in changed.devices()} == {expected}
print('Changed bias, first prediction:', float(changed[0]))

# Reference practice: Test row order without changing the model
reversed_x = jax.device_put(x_host[::-1].copy(), devices[0])
reversed_y = predict(reversed_x, w)
reversed_y.block_until_ready()
np.testing.assert_allclose(np.asarray(reversed_y), reference[::-1], atol=1e-5, rtol=1e-5)
assert {d.platform for d in reversed_y.devices()} == {expected}
assert np.isclose(float(reversed_y[0]), 3.625)
assert np.isclose(float(reversed_y[-1]), 0.34375)
print('Row reversal preserved values and requested placement.')
print("PASS: welcome-03")
