"""Choose weight, activation and accumulation precision: worked experiments and reference solutions. CPU checks."""



import numpy as np
import jax.numpy as jnp
x = np.array([[.3, -1.2, 2.1], [1.1, .2, -.8]], np.float32)
w = np.array([[.11, -1.8], [.7, .2], [-.3, 2.4]], np.float32)
reference = x @ w
def mixed_forward(x, w, dtype):
    # Demonstrate rounding inputs/weights with explicit float32 accumulation.
    a = jnp.asarray(x, dtype).astype(jnp.float32)
    b = jnp.asarray(w, dtype).astype(jnp.float32)
    return np.asarray(a @ b)
for dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
    y = mixed_forward(x, w, dtype)
    error = float(np.max(np.abs(y-reference)))
    assert np.isfinite(y).all() and error < .05
    print(str(dtype), "max absolute error", error)

def quantize_weights(weights, bits):
    if bits not in (4, 8):
        raise ValueError("use signed 4 or 8 bit symmetric quantization")
    limit = 2 ** (bits - 1) - 1
    maxima = np.max(np.abs(weights), axis=0, keepdims=True)
    scale = np.where(maxima == 0, 1., maxima / limit).astype(np.float32)
    q = np.clip(np.rint(weights / scale), -limit, limit).astype(np.int8)
    return q, scale
for bits in (8, 4):
    q, scale = quantize_weights(w, bits)
    reconstructed = q.astype(np.float32) * scale
    assert np.all(np.abs(reconstructed-w) <= scale / 2 + 1e-6)
    # Independent error bound: |x deltaW| <= |x| |deltaW|.
    error = np.abs(x @ reconstructed-reference)
    bound = np.abs(x) @ np.broadcast_to(scale / 2, w.shape)
    assert np.all(error <= bound + 1e-6)
    print("W%dA32 simulated output error" % bits, float(error.max()))


# Figure data experiment
names = []
errors = []
for name, dtype in [('FP32', jnp.float32), ('FP16 → FP32', jnp.float16), ('BF16 → FP32', jnp.bfloat16)]:
    names.append(name)
    errors.append(float(np.max(np.abs(mixed_forward(x, w, dtype) - reference))))
for bits_plot in (8, 4):
    q_plot, s_plot = quantize_weights(w, bits_plot)
    names.append('W' + str(bits_plot) + ' A32')
    errors.append(float(np.max(np.abs(x @ (q_plot.astype(np.float32) * s_plot) - reference))))
visual_data = {'kind': 'bar', 'labels': names, 'ylabel': 'maximum absolute output error', 'series': [{'label': 'CPU reference error', 'y': errors}]}
# Deliberately exact grid values make an independent hand calculation possible.
a_sx, a_zx = .25, -3
a_sw = np.array([.5, .25], np.float64)
a_qx = np.array([[-3, 1, 5]], np.int8)
a_qw = np.array([[2, -1], [-1, 2], [1, 1]], np.int8)
a_bias = np.array([.125, -.0625], np.float64)
a_bias_codes = np.rint(a_bias / (a_sx * a_sw)).astype(np.int64)
# Check the wide result before representing it with an INT32 accumulator.
a_wide = (a_qx.astype(np.int64) - a_zx) @ a_qw.astype(np.int64) + a_bias_codes
assert np.all((a_wide >= np.iinfo(np.int32).min) & (a_wide <= np.iinfo(np.int32).max))
a_acc = a_wide.astype(np.int32)
np.testing.assert_array_equal(a_acc, [[5, 15]])
a_real = a_acc.astype(np.float64) * a_sx * a_sw
# Independent real-valued input/weights, written without decoding the codes.
a_reference = np.array([[0., 1., 2.]]) @ np.array([[1., -.25], [-.5, .5], [.5, .25]]) + a_bias
np.testing.assert_array_equal(a_real, [[.625, .9375]])
np.testing.assert_allclose(a_real, a_reference, atol=1e-12, rtol=0)
def output_grid(real, scale, zero_point):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('positive finite output scale required')
    if type(zero_point) is not int or not -128 <= zero_point <= 127:
        raise ValueError('integer INT8 zero point required')
    unbounded = np.rint(real / scale) + zero_point
    clipped = (unbounded < -128) | (unbounded > 127)
    codes = np.clip(unbounded, -128, 127).astype(np.int8)
    return codes, (codes.astype(np.float64) - zero_point) * scale, clipped

a_codes, a_restored, a_clipped = output_grid(a_real, .0625, -8)
np.testing.assert_array_equal(a_codes, [[2, 7]])
assert not a_clipped.any()
np.testing.assert_allclose(a_restored, a_reference, atol=1e-12, rtol=0)
a_wrong = (a_qx.astype(np.int64) @ a_qw.astype(np.int64) + a_bias_codes) * a_sx * a_sw
np.testing.assert_array_equal(a_wrong, [[-.125, .5625]])
_, a_saturated, a_small_clipped = output_grid(a_real, 1./256, -8)
assert a_small_clipped.all()
np.testing.assert_array_equal(a_saturated, [[135./256, 135./256]])
print('Affine accumulators:', a_acc.tolist(), 'output codes:', a_codes.tolist())
print('Correct / missing zero point / clipped:', a_restored.tolist(), a_wrong.tolist(), a_saturated.tolist())

first_panel = visual_data
visual_data = {'panels': [first_panel, {'kind': 'bar', 'labels': ['output 0', 'output 1'], 'ylabel': 'output value', 'series': [
    {'label': 'correct affine / float reference', 'y': a_restored[0].tolist()},
    {'label': 'input zero point omitted', 'y': a_wrong[0].tolist()},
    {'label': 'narrow output grid clips', 'y': a_saturated[0].tolist()}]}]}


# Experiment: Calibrate an activation range, then shift it
calibration = np.linspace(-1., 1., 101, dtype=np.float32)
activation_scale = float(np.max(np.abs(calibration))) / 127
held_out = np.array([.2, .9, 4.], np.float32)
activation_codes = np.clip(np.rint(held_out / activation_scale), -127, 127).astype(np.int8)
restored_activations = activation_codes.astype(np.float32) * activation_scale
clipped = np.count_nonzero(np.abs(held_out) > 127 * activation_scale)
assert clipped == 1 and abs(float(restored_activations[-1])-4.) > 2.9
print("Clipped values:", clipped, "reconstructed:", restored_activations)


# Experiment: Include zero points, bias and the output grid
# Deliberately exact grid values make an independent hand calculation possible.
a_sx, a_zx = .25, -3
a_sw = np.array([.5, .25], np.float64)
a_qx = np.array([[-3, 1, 5]], np.int8)
a_qw = np.array([[2, -1], [-1, 2], [1, 1]], np.int8)
a_bias = np.array([.125, -.0625], np.float64)
a_bias_codes = np.rint(a_bias / (a_sx * a_sw)).astype(np.int64)
# Check the wide result before representing it with an INT32 accumulator.
a_wide = (a_qx.astype(np.int64) - a_zx) @ a_qw.astype(np.int64) + a_bias_codes
assert np.all((a_wide >= np.iinfo(np.int32).min) & (a_wide <= np.iinfo(np.int32).max))
a_acc = a_wide.astype(np.int32)
np.testing.assert_array_equal(a_acc, [[5, 15]])
a_real = a_acc.astype(np.float64) * a_sx * a_sw
# Independent real-valued input/weights, written without decoding the codes.
a_reference = np.array([[0., 1., 2.]]) @ np.array([[1., -.25], [-.5, .5], [.5, .25]]) + a_bias
np.testing.assert_array_equal(a_real, [[.625, .9375]])
np.testing.assert_allclose(a_real, a_reference, atol=1e-12, rtol=0)
def output_grid(real, scale, zero_point):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('positive finite output scale required')
    if type(zero_point) is not int or not -128 <= zero_point <= 127:
        raise ValueError('integer INT8 zero point required')
    unbounded = np.rint(real / scale) + zero_point
    clipped = (unbounded < -128) | (unbounded > 127)
    codes = np.clip(unbounded, -128, 127).astype(np.int8)
    return codes, (codes.astype(np.float64) - zero_point) * scale, clipped

a_codes, a_restored, a_clipped = output_grid(a_real, .0625, -8)
np.testing.assert_array_equal(a_codes, [[2, 7]])
assert not a_clipped.any()
np.testing.assert_allclose(a_restored, a_reference, atol=1e-12, rtol=0)
a_wrong = (a_qx.astype(np.int64) @ a_qw.astype(np.int64) + a_bias_codes) * a_sx * a_sw
np.testing.assert_array_equal(a_wrong, [[-.125, .5625]])
_, a_saturated, a_small_clipped = output_grid(a_real, 1./256, -8)
assert a_small_clipped.all()
np.testing.assert_array_equal(a_saturated, [[135./256, 135./256]])
print('Affine accumulators:', a_acc.tolist(), 'output codes:', a_codes.tolist())
print('Correct / missing zero point / clipped:', a_restored.tolist(), a_wrong.tolist(), a_saturated.tolist())


# Reference solution. Try the exercise before reading this.
with_zero = np.column_stack((w, np.zeros(3, np.float32)))
q, scale = quantize_weights(with_zero, 8)
assert np.isfinite(scale).all()
np.testing.assert_array_equal(q[:, -1], 0)
np.testing.assert_array_equal((q * scale)[:, -1], 0)
print("Zero channel handled")

# Reference practice: Compute W8A8 with a wide accumulator
sx = float(np.max(np.abs(x))) / 127
qx = np.clip(np.rint(x / sx), -127, 127).astype(np.int8)
qw, sw = quantize_weights(w, 8)
accumulator = qx.astype(np.int32) @ qw.astype(np.int32)
y_integer = accumulator.astype(np.float32) * sx * sw
assert accumulator.dtype == np.int32
np.testing.assert_allclose(y_integer, reference, atol=.06, rtol=0)
print("W8A8 fixture error", float(np.max(np.abs(y_integer-reference))))


# Reference practice: Change the code origin without changing the values
shifted_zx = 17
shifted_codes_wide = a_qx.astype(np.int64) + (shifted_zx - a_zx)
assert np.all((-128 <= shifted_codes_wide) & (shifted_codes_wide <= 127))
shifted_codes = shifted_codes_wide.astype(np.int8)
shifted_acc = (shifted_codes.astype(np.int64) - shifted_zx) @ a_qw.astype(np.int64) + a_bias_codes
np.testing.assert_array_equal(shifted_acc, a_acc)
zero_codes = np.full((1, 3), shifted_zx, np.int8)
zero_result = ((zero_codes.astype(np.int64) - shifted_zx) @ a_qw.astype(np.int64) + a_bias_codes) * a_sx * a_sw
np.testing.assert_array_equal(zero_result, a_bias[None, :])
for bad_scale in (0., -1., float('nan')):
    try: output_grid(a_real, bad_scale, -8)
    except ValueError: pass
    else: raise AssertionError('invalid scale accepted')
print('Code-origin invariance and bias-only zero input verified.')
print("PASS: deployment-05")
