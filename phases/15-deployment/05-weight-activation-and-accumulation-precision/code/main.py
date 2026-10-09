"""Choose weight, activation and accumulation precision: worked experiments and reference solutions. CPU checks."""

# Write the full-precision reference
# Step 1 — Write the full-precision reference: The top-left output is -1.437.
# Import numpy for this computation.
import numpy as np
import jax.numpy as jnp
# Compute `x` from `np.array([[.3, -1.2, 2.1], [1.1, .2, -.8]], np.float32)`
x = np.array([[.3, -1.2, 2.1], [1.1, .2, -.8]], np.float32)
# Compute `w` from `np.array([[.11, -1.8], [.7, .2], [-.3, 2.4]], np.flo...`
w = np.array([[.11, -1.8], [.7, .2], [-.3, 2.4]], np.float32)
# Perform matrix / vector contraction (`@`) to compute `reference`.
reference = x @ w

# Check numerical equivalence within tolerance: `np.testing.assert_allclose(reference[0,0], -1.437, atol=1e-6)`
np.testing.assert_allclose(reference[0,0], -1.437, atol=1e-6)

# Round inputs and weights, then accumulate explicitly
# Step 2 — Round inputs and weights, then accumulate explicitly: The printed errors isolate the consequence of representational...
def mixed_forward(x, w, dtype):
    # Demonstrate rounding inputs/weights with explicit float32 accumulation.
    a = jnp.asarray(x, dtype).astype(jnp.float32)
    # Create device-backed JAX array `b`.
    b = jnp.asarray(w, dtype).astype(jnp.float32)
    # Return `np.asarray(a @ b)` to the caller.
    return np.asarray(a @ b)
# Iterate over `dtype` to step through the computation:
for dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
    # Run `mixed_forward` to compute `y`.
    y = mixed_forward(x, w, dtype)
    # Aggregate array values to compute `error`.
    error = float(np.max(np.abs(y-reference)))
    # Confirm that all computed values remain finite (no NaN or Inf).
    assert np.isfinite(y).all() and error < .05
    # Print diagnostic summary of the computed outputs.
    print(str(dtype), "max absolute error", error)

# Construct signed integer codes and channel scales
# Step 3 — Construct signed integer codes and channel scales: An all-zero channel uses scale 1 and zero codes, so reconstruction...
def quantize_weights(weights, bits):
    # Guard input contract (`bits not in (4, 8)`) and fail fast if violated.
    if bits not in (4, 8):
        raise ValueError("use signed 4 or 8 bit symmetric quantization")
    # Compute `limit` from `2 ** (bits - 1) - 1`
    limit = 2 ** (bits - 1) - 1
    # Reduce along axis=0 to compute `maxima`.
    maxima = np.max(np.abs(weights), axis=0, keepdims=True)
    # Cast or evaluate `scale` in explicit floating-point precision.
    scale = np.where(maxima == 0, 1., maxima / limit).astype(np.float32)
    # Combine or mask array elements to form `q`.
    q = np.clip(np.rint(weights / scale), -limit, limit).astype(np.int8)
    # Return `(q, scale)` to the caller.
    return q, scale

# Allocate initialized array `(zero_codes, zero_scales)` with the specified shape and dtype.
zero_codes, zero_scales = quantize_weights(np.zeros((3,2),np.float32), 8)
# Allocate initialized array `` with the specified shape and dtype.
np.testing.assert_array_equal(zero_codes, np.zeros((3,2),np.int8))
# Allocate initialized array `` with the specified shape and dtype.
np.testing.assert_array_equal(zero_scales, np.ones((1,2),np.float32))

# Check reconstructed outputs against a bound
# Step 4 — Check reconstructed outputs against a bound: Every error must fit its stated bound.
# Iterate over `bits` to step through the computation:
for bits in (8, 4):
    # Run `quantize_weights` to compute `(q, scale)`.
    q, scale = quantize_weights(w, bits)
    # Cast or evaluate `reconstructed` in explicit floating-point precision.
    reconstructed = q.astype(np.float32) * scale
    # Check numerical equivalence within tolerance: `np.all(np.abs(reconstructed-w) <= scale / 2 + 1e-6)`
    assert np.all(np.abs(reconstructed-w) <= scale / 2 + 1e-6)
    # Independent error bound: |x deltaW| <= |x| |deltaW|.
    error = np.abs(x @ reconstructed-reference)
    # Perform matrix contraction / projection to compute `bound`.
    bound = np.abs(x) @ np.broadcast_to(scale / 2, w.shape)
    # Assert invariant `np.all(error <= bound + 1e-6)` holds
    assert np.all(error <= bound + 1e-6)
    # Print diagnostic summary of the computed outputs.
    print("W%dA32 simulated output error" % bits, float(error.max()))

# Choose weight, activation and accumulation precision: A precision policy specifies how weights, activations,...
# Import numpy for this computation.
import numpy as np
import jax.numpy as jnp
# Compute `x` from `np.array([[.3, -1.2, 2.1], [1.1, .2, -.8]], np.float32)`
x = np.array([[.3, -1.2, 2.1], [1.1, .2, -.8]], np.float32)
# Compute `w` from `np.array([[.11, -1.8], [.7, .2], [-.3, 2.4]], np.flo...`
w = np.array([[.11, -1.8], [.7, .2], [-.3, 2.4]], np.float32)
# Perform matrix / vector contraction (`@`) to compute `reference`.
reference = x @ w
# Function `mixed_forward(x, w, dtype)` implementing this stage's computation:
def mixed_forward(x, w, dtype):
    # Demonstrate rounding inputs/weights with explicit float32 accumulation.
    a = jnp.asarray(x, dtype).astype(jnp.float32)
    # Create device-backed JAX array `b`.
    b = jnp.asarray(w, dtype).astype(jnp.float32)
    # Return `np.asarray(a @ b)` to the caller.
    return np.asarray(a @ b)
# Iterate over `dtype` to step through the computation:
for dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
    # Run `mixed_forward` to compute `y`.
    y = mixed_forward(x, w, dtype)
    # Aggregate array values to compute `error`.
    error = float(np.max(np.abs(y-reference)))
    # Confirm that all computed values remain finite (no NaN or Inf).
    assert np.isfinite(y).all() and error < .05
    # Print diagnostic summary of the computed outputs.
    print(str(dtype), "max absolute error", error)

# Function `quantize_weights(weights, bits)` implementing this stage's computation:
def quantize_weights(weights, bits):
    # Guard input contract (`bits not in (4, 8)`) and fail fast if violated.
    if bits not in (4, 8):
        raise ValueError("use signed 4 or 8 bit symmetric quantization")
    # Compute `limit` from `2 ** (bits - 1) - 1`
    limit = 2 ** (bits - 1) - 1
    # Reduce along axis=0 to compute `maxima`.
    maxima = np.max(np.abs(weights), axis=0, keepdims=True)
    # Cast or evaluate `scale` in explicit floating-point precision.
    scale = np.where(maxima == 0, 1., maxima / limit).astype(np.float32)
    # Combine or mask array elements to form `q`.
    q = np.clip(np.rint(weights / scale), -limit, limit).astype(np.int8)
    # Return `(q, scale)` to the caller.
    return q, scale
# Iterate over `bits` to step through the computation:
for bits in (8, 4):
    # Run `quantize_weights` to compute `(q, scale)`.
    q, scale = quantize_weights(w, bits)
    # Cast or evaluate `reconstructed` in explicit floating-point precision.
    reconstructed = q.astype(np.float32) * scale
    # Check numerical equivalence within tolerance: `np.all(np.abs(reconstructed-w) <= scale / 2 + 1e-6)`
    assert np.all(np.abs(reconstructed-w) <= scale / 2 + 1e-6)
    # Independent error bound: |x deltaW| <= |x| |deltaW|.
    error = np.abs(x @ reconstructed-reference)
    # Perform matrix contraction / projection to compute `bound`.
    bound = np.abs(x) @ np.broadcast_to(scale / 2, w.shape)
    # Assert invariant `np.all(error <= bound + 1e-6)` holds
    assert np.all(error <= bound + 1e-6)
    # Print diagnostic summary of the computed outputs.
    print("W%dA32 simulated output error" % bits, float(error.max()))

# Figure data experiment
# Compute figure data for: Precision choices trade representation for output error
# Compute `names` from `[]`
names = []
# Compute `errors` from `[]`
errors = []
# Loop over `(name, dtype)` in `[('FP32', jnp.float32), ('FP16 → FP32', jnp.float16), ('BF16 → FP32', jnp.bfloat16)]`:
for name, dtype in [('FP32', jnp.float32), ('FP16 → FP32', jnp.float16), ('BF16 → FP32', jnp.bfloat16)]:
    # Append the current step result to `names`.
    names.append(name)
    # Reduce across the target axis to summarize ``.
    errors.append(float(np.max(np.abs(mixed_forward(x, w, dtype) - reference))))
# Loop over `bits_plot` in `(8, 4)`:
for bits_plot in (8, 4):
    # Run `quantize_weights` to compute `(q_plot, s_plot)`.
    q_plot, s_plot = quantize_weights(w, bits_plot)
    # Append the current step result to `names`.
    names.append('W' + str(bits_plot) + ' A32')
    # Cast or evaluate `` in explicit floating-point precision.
    errors.append(float(np.max(np.abs(x @ (q_plot.astype(np.float32) * s_plot) - reference))))
# Compute `visual_data` from `{'kind': 'bar', 'labels': names, 'ylabel': 'maximum ...`
visual_data = {'kind': 'bar', 'labels': names, 'ylabel': 'maximum absolute output error', 'series': [{'label': 'CPU reference error', 'y': errors}]}
# Deliberately exact grid values make an independent hand calculation possible.
a_sx, a_zx = .25, -3
# Convert `a_sw` to a host NumPy array for inspection or verification.
a_sw = np.array([.5, .25], np.float64)
# Convert `a_qx` to a host NumPy array for inspection or verification.
a_qx = np.array([[-3, 1, 5]], np.int8)
# Convert `a_qw` to a host NumPy array for inspection or verification.
a_qw = np.array([[2, -1], [-1, 2], [1, 1]], np.int8)
# Convert `a_bias` to a host NumPy array for inspection or verification.
a_bias = np.array([.125, -.0625], np.float64)
# Run `np.rint` to compute `a_bias_codes`.
a_bias_codes = np.rint(a_bias / (a_sx * a_sw)).astype(np.int64)
# Check the wide result before representing it with an INT32 accumulator.
a_wide = (a_qx.astype(np.int64) - a_zx) @ a_qw.astype(np.int64) + a_bias_codes
# Assert invariant `np.all((a_wide >= np.iinfo(np.int32).min) & (a_wide <= np.iinfo(n...` holds
assert np.all((a_wide >= np.iinfo(np.int32).min) & (a_wide <= np.iinfo(np.int32).max))
# Run `a_wide.astype` to compute `a_acc`.
a_acc = a_wide.astype(np.int32)
# Execute `np.testing.assert_array_equal(a_acc, [[5, 15]])`
np.testing.assert_array_equal(a_acc, [[5, 15]])
# Run `a_acc.astype` to compute `a_real`.
a_real = a_acc.astype(np.float64) * a_sx * a_sw
# Independent real-valued input/weights, written without decoding the codes.
a_reference = np.array([[0., 1., 2.]]) @ np.array([[1., -.25], [-.5, .5], [.5, .25]]) + a_bias
# Execute `np.testing.assert_array_equal(a_real, [[.625, .9375]])`
np.testing.assert_array_equal(a_real, [[.625, .9375]])
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(a_real, a_reference, atol=1e-12, rtol=0)`
np.testing.assert_allclose(a_real, a_reference, atol=1e-12, rtol=0)
# Function `output_grid(real, scale, zero_point)` implementing this stage's computation:
def output_grid(real, scale, zero_point):
    # Guard input contract (`not np.isfinite(scale) or scale <= 0`) and fail fast if violated.
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('positive finite output scale required')
    # Guard input contract (`type(zero_point) is not int or not -128 <= zero_point <= 127`) and fail fast if violated.
    if type(zero_point) is not int or not -128 <= zero_point <= 127:
        raise ValueError('integer INT8 zero point required')
    # Run `np.rint` to compute `unbounded`.
    unbounded = np.rint(real / scale) + zero_point
    # Compute `clipped` from `(unbounded < -128) | (unbounded > 127)`
    clipped = (unbounded < -128) | (unbounded > 127)
    # Combine or mask array elements to form `codes`.
    codes = np.clip(unbounded, -128, 127).astype(np.int8)
    # Return `(codes, (codes.astype(np.float64) - zero_point) * scale, clipped)` to the caller.
    return codes, (codes.astype(np.float64) - zero_point) * scale, clipped

# Run `output_grid` to compute `(a_codes, a_restored, a_clipped)`.
a_codes, a_restored, a_clipped = output_grid(a_real, .0625, -8)
# Execute `np.testing.assert_array_equal(a_codes, [[2, 7]])`
np.testing.assert_array_equal(a_codes, [[2, 7]])
# Assert invariant `not a_clipped.any()` holds
assert not a_clipped.any()
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(a_restored, a_reference, atol=1e-12, r...`
np.testing.assert_allclose(a_restored, a_reference, atol=1e-12, rtol=0)
# Perform matrix contraction / projection to compute `a_wrong`.
a_wrong = (a_qx.astype(np.int64) @ a_qw.astype(np.int64) + a_bias_codes) * a_sx * a_sw
# Execute `np.testing.assert_array_equal(a_wrong, [[-.125, .5625]])`
np.testing.assert_array_equal(a_wrong, [[-.125, .5625]])
# Run `output_grid` to compute `(_, a_saturated, a_small_clipped)`.
_, a_saturated, a_small_clipped = output_grid(a_real, 1./256, -8)
# Assert invariant `a_small_clipped.all()` holds
assert a_small_clipped.all()
# Execute `np.testing.assert_array_equal(a_saturated, [[135./256, 135./`
np.testing.assert_array_equal(a_saturated, [[135./256, 135./256]])
# Print diagnostic summary of the computed outputs.
print('Affine accumulators:', a_acc.tolist(), 'output codes:', a_codes.tolist())
# Print diagnostic summary of the computed outputs.
print('Correct / missing zero point / clipped:', a_restored.tolist(), a_wrong.tolist(), a_saturated.tolist())

# Compute `first_panel` from `visual_data`
first_panel = visual_data
# Combine or mask array elements to form `visual_data`.
visual_data = {'panels': [first_panel, {'kind': 'bar', 'labels': ['output 0', 'output 1'], 'ylabel': 'output value', 'series': [
    {'label': 'correct affine / float reference', 'y': a_restored[0].tolist()},
    {'label': 'input zero point omitted', 'y': a_wrong[0].tolist()},
    {'label': 'narrow output grid clips', 'y': a_saturated[0].tolist()}]}]}

# Experiment: Calibrate an activation range, then shift it
# Experiment — Calibrate an activation range, then shift it: Do not recalibrate on the test set to hide this error.
# Compute `calibration` from `np.linspace(-1., 1., 101, dtype=np.float32)`
calibration = np.linspace(-1., 1., 101, dtype=np.float32)
# Aggregate array values to compute `activation_scale`.
activation_scale = float(np.max(np.abs(calibration))) / 127
# Compute `held_out` from `np.array([.2, .9, 4.], np.float32)`
held_out = np.array([.2, .9, 4.], np.float32)
# Combine or mask array elements to form `activation_codes`.
activation_codes = np.clip(np.rint(held_out / activation_scale), -127, 127).astype(np.int8)
# Cast or evaluate `restored_activations` in explicit floating-point precision.
restored_activations = activation_codes.astype(np.float32) * activation_scale
# Run `np.count_nonzero` to compute `clipped`.
clipped = np.count_nonzero(np.abs(held_out) > 127 * activation_scale)
# Check numerical equivalence within tolerance: `clipped == 1 and abs(float(restored_activations[-1])-4.) > 2.9`
assert clipped == 1 and abs(float(restored_activations[-1])-4.) > 2.9
# Print the observed values to compare against the expected result.
print("Clipped values:", clipped, "reconstructed:", restored_activations)

# Experiment: Include zero points, bias and the output grid
# Deliberately exact grid values make an independent hand calculation possible.
a_sx, a_zx = .25, -3
# Compute `a_sw` from `np.array([.5, .25], np.float64)`
a_sw = np.array([.5, .25], np.float64)
# Compute `a_qx` from `np.array([[-3, 1, 5]], np.int8)`
a_qx = np.array([[-3, 1, 5]], np.int8)
# Compute `a_qw` from `np.array([[2, -1], [-1, 2], [1, 1]], np.int8)`
a_qw = np.array([[2, -1], [-1, 2], [1, 1]], np.int8)
# Compute `a_bias` from `np.array([.125, -.0625], np.float64)`
a_bias = np.array([.125, -.0625], np.float64)
# Run `np.rint` to compute `a_bias_codes`.
a_bias_codes = np.rint(a_bias / (a_sx * a_sw)).astype(np.int64)
# Check the wide result before representing it with an INT32 accumulator.
a_wide = (a_qx.astype(np.int64) - a_zx) @ a_qw.astype(np.int64) + a_bias_codes
# Assert invariant `np.all((a_wide >= np.iinfo(np.int32).min) & (a_wide <= np.iinfo(n...` holds
assert np.all((a_wide >= np.iinfo(np.int32).min) & (a_wide <= np.iinfo(np.int32).max))
# Run `a_wide.astype` to compute `a_acc`.
a_acc = a_wide.astype(np.int32)
# Execute `np.testing.assert_array_equal(a_acc, [[5, 15]])`
np.testing.assert_array_equal(a_acc, [[5, 15]])
# Run `a_acc.astype` to compute `a_real`.
a_real = a_acc.astype(np.float64) * a_sx * a_sw
# Independent real-valued input/weights, written without decoding the codes.
a_reference = np.array([[0., 1., 2.]]) @ np.array([[1., -.25], [-.5, .5], [.5, .25]]) + a_bias
# Execute `np.testing.assert_array_equal(a_real, [[.625, .9375]])`
np.testing.assert_array_equal(a_real, [[.625, .9375]])
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(a_real, a_reference, atol=1e-12, rtol=0)`
np.testing.assert_allclose(a_real, a_reference, atol=1e-12, rtol=0)
# Function `output_grid(real, scale, zero_point)` implementing this stage's computation:
def output_grid(real, scale, zero_point):
    # Guard input contract (`not np.isfinite(scale) or scale <= 0`) and fail fast if violated.
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('positive finite output scale required')
    # Guard input contract (`type(zero_point) is not int or not -128 <= zero_point <= 127`) and fail fast if violated.
    if type(zero_point) is not int or not -128 <= zero_point <= 127:
        raise ValueError('integer INT8 zero point required')
    # Run `np.rint` to compute `unbounded`.
    unbounded = np.rint(real / scale) + zero_point
    # Compute `clipped` from `(unbounded < -128) | (unbounded > 127)`
    clipped = (unbounded < -128) | (unbounded > 127)
    # Combine or mask array elements to form `codes`.
    codes = np.clip(unbounded, -128, 127).astype(np.int8)
    # Return `(codes, (codes.astype(np.float64) - zero_point) * scale, clipped)` to the caller.
    return codes, (codes.astype(np.float64) - zero_point) * scale, clipped

# Run `output_grid` to compute `(a_codes, a_restored, a_clipped)`.
a_codes, a_restored, a_clipped = output_grid(a_real, .0625, -8)
# Execute `np.testing.assert_array_equal(a_codes, [[2, 7]])`
np.testing.assert_array_equal(a_codes, [[2, 7]])
# Assert invariant `not a_clipped.any()` holds
assert not a_clipped.any()
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(a_restored, a_reference, atol=1e-12, r...`
np.testing.assert_allclose(a_restored, a_reference, atol=1e-12, rtol=0)
# Perform matrix contraction / projection to compute `a_wrong`.
a_wrong = (a_qx.astype(np.int64) @ a_qw.astype(np.int64) + a_bias_codes) * a_sx * a_sw
# Execute `np.testing.assert_array_equal(a_wrong, [[-.125, .5625]])`
np.testing.assert_array_equal(a_wrong, [[-.125, .5625]])
# Run `output_grid` to compute `(_, a_saturated, a_small_clipped)`.
_, a_saturated, a_small_clipped = output_grid(a_real, 1./256, -8)
# Assert invariant `a_small_clipped.all()` holds
assert a_small_clipped.all()
# Execute `np.testing.assert_array_equal(a_saturated, [[135./256, 135./`
np.testing.assert_array_equal(a_saturated, [[135./256, 135./256]])
# Print diagnostic summary of the computed outputs.
print('Affine accumulators:', a_acc.tolist(), 'output codes:', a_codes.tolist())
# Print diagnostic summary of the computed outputs.
print('Correct / missing zero point / clipped:', a_restored.tolist(), a_wrong.tolist(), a_saturated.tolist())

# Experiment: Give small and large output channels different grids
# Experiment — Give small and large output channels different grids: The global maximum determines a coarse grid for every channel.
# Compute `heterogeneous` from `np.array([[.01, 10.], [-.02, -5.], [.03, 2.]], np.fl...`
heterogeneous = np.array([[.01, 10.], [-.02, -5.], [.03, 2.]], np.float32)
# Run `quantize_weights` to compute `(channel_codes, channel_scales)`.
channel_codes, channel_scales = quantize_weights(heterogeneous, 8)
# Cast or evaluate `per_channel` in explicit floating-point precision.
per_channel = channel_codes.astype(np.float32) * channel_scales
# Aggregate array values to compute `shared_scale`.
shared_scale = np.max(np.abs(heterogeneous)) / 127
# Cast or evaluate `shared` in explicit floating-point precision.
shared = np.clip(np.rint(heterogeneous/shared_scale), -127, 127).astype(np.int8).astype(np.float32)*shared_scale
# Assert invariant `np.all(shared[:,0] == 0)` holds
assert np.all(shared[:,0] == 0)
# Check numerical equivalence within tolerance: `np.max(np.abs(per_channel[:,0]-heterogeneous[:,0])) < .001`
assert np.max(np.abs(per_channel[:,0]-heterogeneous[:,0])) < .001
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(shared[:,0], [0.,0.,0.], atol=0)`
np.testing.assert_allclose(shared[:,0], [0.,0.,0.], atol=0)
# Print the observed values to compare against the expected result.
print('Small-channel max error; shared / per-channel:', np.max(np.abs(shared[:,0]-heterogeneous[:,0])), np.max(np.abs(per_channel[:,0]-heterogeneous[:,0])))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Add a zero-valued output channel and verify that its quantization...
with_zero = np.column_stack((w, np.zeros(3, np.float32)))
# Run `quantize_weights` to compute `(q, scale)`.
q, scale = quantize_weights(with_zero, 8)
# Confirm that all computed values remain finite (no NaN or Inf).
assert np.isfinite(scale).all()
# Execute `np.testing.assert_array_equal(q[:, -1], 0)`
np.testing.assert_array_equal(q[:, -1], 0)
# Execute `np.testing.assert_array_equal((q * scale)[:, -1], 0)`
np.testing.assert_array_equal((q * scale)[:, -1], 0)
# Print the observed values to compare against the expected result.
print("Zero channel handled")

# Reference practice: Compute W8A8 with a wide accumulator
# Compute W8A8 with a wide accumulator (Transfer): This dense fixture uses zero points of zero and no bias.
# Aggregate array values to compute `sx`.
sx = float(np.max(np.abs(x))) / 127
# Combine or mask array elements to form `qx`.
qx = np.clip(np.rint(x / sx), -127, 127).astype(np.int8)
# Run `quantize_weights` to compute `(qw, sw)`.
qw, sw = quantize_weights(w, 8)
# Perform matrix / vector contraction (`@`) to compute `accumulator`.
accumulator = qx.astype(np.int32) @ qw.astype(np.int32)
# Cast or evaluate `y_integer` in explicit floating-point precision.
y_integer = accumulator.astype(np.float32) * sx * sw
# Assert invariant `accumulator.dtype == np.int32` holds
assert accumulator.dtype == np.int32
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(y_integer, reference, atol=.06, rtol=0)`
np.testing.assert_allclose(y_integer, reference, atol=.06, rtol=0)
# Print the observed values to compare against the expected result.
print("W8A8 fixture error", float(np.max(np.abs(y_integer-reference))))

# Reference practice: Change the code origin without changing the values
# Change the code origin without changing the values (Transfer): Integer codes are coordinates on a grid.
shifted_zx = 17
# Run `a_qx.astype` to compute `shifted_codes_wide`.
shifted_codes_wide = a_qx.astype(np.int64) + (shifted_zx - a_zx)
# Assert invariant `np.all((-128 <= shifted_codes_wide) & (shifted_codes_wide <= 127))` holds
assert np.all((-128 <= shifted_codes_wide) & (shifted_codes_wide <= 127))
# Run `shifted_codes_wide.astype` to compute `shifted_codes`.
shifted_codes = shifted_codes_wide.astype(np.int8)
# Perform matrix contraction / projection to compute `shifted_acc`.
shifted_acc = (shifted_codes.astype(np.int64) - shifted_zx) @ a_qw.astype(np.int64) + a_bias_codes
# Execute `np.testing.assert_array_equal(shifted_acc, a_acc)`
np.testing.assert_array_equal(shifted_acc, a_acc)
# Run `np.full` to compute `zero_codes`.
zero_codes = np.full((1, 3), shifted_zx, np.int8)
# Perform matrix contraction / projection to compute `zero_result`.
zero_result = ((zero_codes.astype(np.int64) - shifted_zx) @ a_qw.astype(np.int64) + a_bias_codes) * a_sx * a_sw
# Execute `np.testing.assert_array_equal(zero_result, a_bias[None, :])`
np.testing.assert_array_equal(zero_result, a_bias[None, :])
# Iterate over `bad_scale` to step through the computation:
for bad_scale in (0., -1., float('nan')):
    try: output_grid(a_real, bad_scale, -8)
    except ValueError: pass
    else: raise AssertionError('invalid scale accepted')
# Print the observed values to compare against the expected result.
print('Code-origin invariance and bias-only zero input verified.')

# Reference practice: Account for actual low-bit storage
# Account for actual low-bit storage (Transfer / diagnosis): The simulation uses 14 bytes for its codes and scales, not...
q4, s4 = quantize_weights(w, 4)
# Compute `float_weight_bytes` from `w.nbytes`
float_weight_bytes = w.nbytes
# Compute `simulated_weight_and_scale_bytes` from `q4.nbytes + s4.nbytes`
simulated_weight_and_scale_bytes = q4.nbytes + s4.nbytes
# Compute `ideal_packed_weight_and_scale_bytes` from `(w.size + 1)//2 + s4.nbytes`
ideal_packed_weight_and_scale_bytes = (w.size + 1)//2 + s4.nbytes
# Assert invariant `(float_weight_bytes, simulated_weight_and_scale_bytes, ideal_pack...` holds
assert (float_weight_bytes, simulated_weight_and_scale_bytes, ideal_packed_weight_and_scale_bytes) == (24,14,11)
# Print the observed values to compare against the expected result.
print('Weight-plus-scale subtotals:', float_weight_bytes, simulated_weight_and_scale_bytes, ideal_packed_weight_and_scale_bytes)
print("PASS: deployment-05")
