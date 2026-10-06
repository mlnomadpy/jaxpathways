"""Diagnose recompilation and host synchronization: worked experiments and reference solutions. CPU checks."""

# Prepare a trace ledger
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
cpu = jax.devices("cpu")[0]
trace_events = []
def score(x, gain):
    # Authoring diagnostic: this Python effect happens during tracing.
    # It is NOT model state and does not count executable compilations.
    trace_events.append({"shape": list(x.shape), "dtype": str(x.dtype)})
    return jnp.sum((x * gain)**2)
compiled_score = jax.jit(score)
x8 = jax.device_put(np.arange(8, dtype=np.float32), cpu)
x12 = jax.device_put(np.arange(12, dtype=np.float32), cpu)
jax.block_until_ready((x8, x12))


# Record call signatures and observed traces
call_rows = []
def observed_call(label, x, gain):
    before = len(trace_events)
    result = compiled_score(x, jax.device_put(np.float32(gain), cpu))
    result.block_until_ready()
    expected = np.sum((np.asarray(x) * np.float32(gain))**2)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-5, atol=2e-5)
    call_rows.append({"label": label, "shape": list(x.shape), "dtype": str(x.dtype),
                      "observed_trace_delta": len(trace_events) - before,
                      "result": float(result)})
    return result


# Run the controlled call sequence
observed_call("initial signature", x8, 1.)
observed_call("same signature changed data", x8 + 1., 1.)
observed_call("same signature changed dynamic gain", x8, 2.)
observed_call("different input shape", x12, 1.)
print(json.dumps({"jax": jax.__version__, "backend": str(cpu),
                  "calls": call_rows, "python_trace_events": trace_events}, indent=2))
assert call_rows[0]["result"] == 140.
assert call_rows[2]["result"] == 560.
assert call_rows[3]["result"] == 506.


import json
import time
import numpy as np
import jax
import jax.numpy as jnp
cpu = jax.devices("cpu")[0]
trace_events = []
def score(x, gain):
    # Authoring diagnostic: this Python effect happens during tracing.
    # It is NOT model state and does not count executable compilations.
    trace_events.append({"shape": list(x.shape), "dtype": str(x.dtype)})
    return jnp.sum((x * gain)**2)
compiled_score = jax.jit(score)
x8 = jax.device_put(np.arange(8, dtype=np.float32), cpu)
x12 = jax.device_put(np.arange(12, dtype=np.float32), cpu)
jax.block_until_ready((x8, x12))

call_rows = []
def observed_call(label, x, gain):
    before = len(trace_events)
    result = compiled_score(x, jax.device_put(np.float32(gain), cpu))
    result.block_until_ready()
    expected = np.sum((np.asarray(x) * np.float32(gain))**2)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-5, atol=2e-5)
    call_rows.append({"label": label, "shape": list(x.shape), "dtype": str(x.dtype),
                      "observed_trace_delta": len(trace_events) - before,
                      "result": float(result)})
    return result

observed_call("initial signature", x8, 1.)
observed_call("same signature changed data", x8 + 1., 1.)
observed_call("same signature changed dynamic gain", x8, 2.)
observed_call("different input shape", x12, 1.)
print(json.dumps({"jax": jax.__version__, "backend": str(cpu),
                  "calls": call_rows, "python_trace_events": trace_events}, indent=2))
assert call_rows[0]["result"] == 140.
assert call_rows[2]["result"] == 560.
assert call_rows[3]["result"] == 506.


# Figure data experiment
visual_data = {'kind': 'bar', 'labels': ['initial', 'new values', 'new gain', 'new shape'], 'ylabel': 'observed Python trace events', 'series': [{'label': 'trace delta', 'y': [r['observed_trace_delta'] for r in call_rows]}]}

# Experiment: Contrast static branch metadata with a dynamic gain
branch_events = []
def branch_score(x, square):
    branch_events.append({"shape": list(x.shape), "square": square})
    return jnp.sum(x**2) if square else jnp.sum(x)
static_score = jax.jit(branch_score, static_argnames=("square",))
for square in [True, True, False]:
    out = static_score(x8, square=square)
    np.testing.assert_allclose(np.asarray(out), 140. if square else 28.)
print("Static branch trace observations:", branch_events)


# Experiment: Keep a metric on device until the reporting boundary
update = jax.jit(lambda value: 0.9 * value + 1.)
initial = jax.device_put(np.float32(0.), cpu)
update(initial).block_until_ready()
def metric_run(read_each):
    value = initial
    observed = []
    start = time.perf_counter()
    for _ in range(6):
        value = update(value)
        if read_each:
            observed.append(float(value))
    value.block_until_ready()
    elapsed = time.perf_counter() - start
    return value, elapsed, observed
read_result, read_seconds, metrics = metric_run(True)
final_result, final_seconds, _ = metric_run(False)
reference_value = 10. * (1. - 0.9**6)
np.testing.assert_allclose(np.asarray(read_result), reference_value, rtol=2e-5)
np.testing.assert_allclose(np.asarray(final_result), reference_value, rtol=2e-5)
assert len(metrics) == 6
print("Per-step host reads seconds:", read_seconds)
print("Final-only wait seconds:", final_seconds)


# Reference solution. Try the exercise before reading this.
x16 = jax.device_put(np.arange(16, dtype=np.float32), cpu)
observed_call("new length sixteen", x16, 1.)
assert call_rows[-1]["result"] == sum(i*i for i in range(16))
print("Added ledger row:", call_rows[-1])


# Reference practice: Pad a final batch while preserving the objective
masked_score = jax.jit(lambda values, mask: jnp.sum(jnp.where(mask, (values + 1.)**2, 0.)))
five = np.arange(5, dtype=np.float32)
padded = jax.device_put(np.pad(five, (0, 3)), cpu)
mask = jax.device_put(np.arange(8) < 5, cpu)
masked = masked_score(padded, mask)
np.testing.assert_allclose(np.asarray(masked), sum((i+1)**2 for i in range(5)))
wrong = jnp.sum((padded + 1.)**2)
assert float(wrong) == float(masked) + 3.
print("Masked versus unmasked padded objective:", float(masked), float(wrong))


# Reference practice: Reproduce and repair runtime Python branching
def broken_branch(values, gain):
    if gain > 1.:
        return jnp.sum(values**2)
    return jnp.sum(values)
try:
    jax.jit(broken_branch)(x8, jax.device_put(np.float32(2.), cpu))
except jax.errors.TracerBoolConversionError:
    print("Observed runtime-Python-branch tracing failure")
else:
    raise AssertionError("runtime branch unexpectedly accepted")
repaired = jax.jit(lambda values, gain: jax.lax.cond(gain > 1., lambda v:jnp.sum(v**2), lambda v:jnp.sum(v), values))
for gain, expected in [(0.5, 28.), (2., 140.)]:
    result = repaired(x8, jax.device_put(np.float32(gain), cpu))
    np.testing.assert_allclose(np.asarray(result), expected)

print("PASS: performance-02")
