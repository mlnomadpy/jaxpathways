"""Move models between JAX, Keras, TensorFlow and PyTorch: worked experiments and reference solutions. CPU checks."""

# Write the source function and its known input
import numpy as np
import jax
import jax.numpy as jnp
# Same dense layer, different parameter layouts.
x = np.array([[1., 2., -1.], [-2., 0., 3.]], np.float32)
keras_kernel = np.array([[1., -2.], [.5, 1.], [-1., .25]], np.float32)
bias = np.array([.1, -.2], np.float32)

np.testing.assert_allclose(x[0] @ keras_kernel + bias, [3.1, -.45], atol=1e-6)


# Map the stored tensor into explicit JAX parameters
torch_weight = keras_kernel.T.copy()  # torch Linear: output, input
params = {"kernel": jnp.asarray(torch_weight.T), "bias": jnp.asarray(bias)}
def predict(params, inputs):
    return inputs @ params["kernel"] + params["bias"]

np.testing.assert_allclose(predict(params, jnp.zeros((2,3))), np.tile(bias,(2,1)), atol=1e-6)


# Verify values before extending the architecture
actual = np.asarray(jax.jit(predict)(params, jnp.asarray(x)))
# A hand-computed first row detects a shared layout mistake.
np.testing.assert_allclose(actual[0], [3.1, -.45], atol=1e-6)
np.testing.assert_allclose(actual, x @ keras_kernel + bias, atol=1e-6)
print("Matched dense outputs:", actual)


import numpy as np
import jax
import jax.numpy as jnp
# Same dense layer, different parameter layouts.
x = np.array([[1., 2., -1.], [-2., 0., 3.]], np.float32)
keras_kernel = np.array([[1., -2.], [.5, 1.], [-1., .25]], np.float32)
bias = np.array([.1, -.2], np.float32)
torch_weight = keras_kernel.T.copy()  # torch Linear: output, input
params = {"kernel": jnp.asarray(torch_weight.T), "bias": jnp.asarray(bias)}
def predict(params, inputs):
    return inputs @ params["kernel"] + params["bias"]
actual = np.asarray(jax.jit(predict)(params, jnp.asarray(x)))
# A hand-computed first row detects a shared layout mistake.
np.testing.assert_allclose(actual[0], [3.1, -.45], atol=1e-6)
np.testing.assert_allclose(actual, x @ keras_kernel + bias, atol=1e-6)
print("Matched dense outputs:", actual)


# Figure data experiment
visual_data = {'kind': 'panels', 'panels': [{'kind': 'heatmap', 'title': 'Input × output layout', 'values': keras_kernel.tolist(), 'rows': ['input 0', 'input 1', 'input 2'], 'columns': ['output 0', 'output 1'], 'unit': 'weight'}, {'kind': 'heatmap', 'title': 'Output × input layout', 'values': torch_weight.tolist(), 'rows': ['output 0', 'output 1'], 'columns': ['input 0', 'input 1', 'input 2'], 'unit': 'weight'}]}

# Experiment: A square matrix hides a transpose
square = np.array([[1., 2.], [3., 4.]], np.float32)
z = np.array([[2., -1.]], np.float32)
assert (z @ square).shape == (z @ square.T).shape
assert not np.allclose(z @ square, z @ square.T)
print("Equal shapes, unequal outputs")


# Experiment: Reorder features without changing tensor dimensions
permutation = [2, 1, 0]
reordered = x[:, permutation]
wrong_order = np.asarray(predict(params, reordered))
np.testing.assert_allclose(wrong_order[0], [-.9, 4.05], atol=1e-6)
assert not np.allclose(wrong_order, actual)
repaired_params = {'kernel': params['kernel'][permutation, :], 'bias': params['bias']}
np.testing.assert_allclose(predict(repaired_params, reordered), actual, atol=1e-6)
print('Feature order failed, then matched after aligning kernel rows.')


# Reference solution. Try the exercise before reading this.
changed = np.arange(12, dtype=np.float32).reshape(4, 3) / 4
expected = np.stack([sum(row[i] * keras_kernel[i] for i in range(3)) + bias for row in changed])
np.testing.assert_allclose(predict(params, changed), expected, atol=2e-6)
print("Changed batch verified")

# Reference practice: Detect a preprocessing mismatch
pixels = np.array([[255., 128., 0.]], np.float32)
source = predict(params, pixels / 255.)
wrong = predict(params, pixels)
assert not np.allclose(source, wrong)
np.testing.assert_allclose(source, predict(params, pixels / 255.), atol=1e-6)


# Reference practice: Map a second layer and locate an omitted activation
second_kernel = np.array([[2.], [-1.]], np.float32)
second_bias = np.array([.3], np.float32)
hidden = np.maximum(np.asarray(predict(params, x)), 0.)
bridged_scores = hidden @ second_kernel + second_bias
np.testing.assert_allclose(bridged_scores[:, 0], [6.5, -4.25], atol=2e-6)
omitted_relu = np.asarray(predict(params, x)) @ second_kernel + second_bias
assert not np.allclose(omitted_relu, bridged_scores)
positive_probe = np.array([[1., 2.]], np.float32)
np.testing.assert_array_equal(np.maximum(positive_probe, 0), positive_probe)
print('Two-layer reference scores:', bridged_scores[:, 0])

print("PASS: deployment-01")
