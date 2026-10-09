"""Move models between JAX, Keras, TensorFlow and PyTorch: worked experiments and reference solutions. CPU checks."""

# Write the source function and its known input
# Step 1 — Write the source function and its known input: The arrays define a batch of shape (2,3), a kernel of shape (3,2),...
# Import numpy for this computation.
import numpy as np
import jax
import jax.numpy as jnp
# Same dense layer, different parameter layouts.
x = np.array([[1., 2., -1.], [-2., 0., 3.]], np.float32)
# Compute `keras_kernel` from `np.array([[1., -2.], [.5, 1.], [-1., .25]], np.float32)`
keras_kernel = np.array([[1., -2.], [.5, 1.], [-1., .25]], np.float32)
# Compute `bias` from `np.array([.1, -.2], np.float32)`
bias = np.array([.1, -.2], np.float32)

# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(x[0] @ keras_kernel + bias, [3.1, -.45], atol=1e-6)

# Map the stored tensor into explicit JAX parameters
# Step 2 — Map the stored tensor into explicit JAX parameters: The zero-input check returns the bias twice.
torch_weight = keras_kernel.T.copy()  # torch Linear: output, input
# Create device-backed JAX array `params`.
params = {"kernel": jnp.asarray(torch_weight.T), "bias": jnp.asarray(bias)}
# Function `predict(params, inputs)` implementing this stage's computation:
def predict(params, inputs):
    # Return `inputs @ params['kernel'] + params['bias']` to the caller.
    return inputs @ params["kernel"] + params["bias"]

# Allocate initialized array `` with the specified shape and dtype.
np.testing.assert_allclose(predict(params, jnp.zeros((2,3))), np.tile(bias,(2,1)), atol=1e-6)

# Verify values before extending the architecture
# Step 3 — Verify values before extending the architecture: The first assertion checks hand arithmetic; the second compares...
# Wrap with `jax.jit` (`actual`) so XLA traces and compiles the function.
actual = np.asarray(jax.jit(predict)(params, jnp.asarray(x)))
# A hand-computed first row detects a shared layout mistake.
np.testing.assert_allclose(actual[0], [3.1, -.45], atol=1e-6)
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(actual, x @ keras_kernel + bias, atol=1e-6)
# Print the observed values to compare against the expected result.
print("Matched dense outputs:", actual)

# Move models between JAX, Keras, TensorFlow and PyTorch: Cross-framework conversion must preserve the meaning of inputs,...
# Import numpy for this computation.
import numpy as np
import jax
import jax.numpy as jnp
# Same dense layer, different parameter layouts.
x = np.array([[1., 2., -1.], [-2., 0., 3.]], np.float32)
# Compute `keras_kernel` from `np.array([[1., -2.], [.5, 1.], [-1., .25]], np.float32)`
keras_kernel = np.array([[1., -2.], [.5, 1.], [-1., .25]], np.float32)
# Compute `bias` from `np.array([.1, -.2], np.float32)`
bias = np.array([.1, -.2], np.float32)
torch_weight = keras_kernel.T.copy()  # torch Linear: output, input
# Create device-backed JAX array `params`.
params = {"kernel": jnp.asarray(torch_weight.T), "bias": jnp.asarray(bias)}
# Function `predict(params, inputs)` implementing this stage's computation:
def predict(params, inputs):
    # Return `inputs @ params['kernel'] + params['bias']` to the caller.
    return inputs @ params["kernel"] + params["bias"]
# Wrap with `jax.jit` (`actual`) so XLA traces and compiles the function.
actual = np.asarray(jax.jit(predict)(params, jnp.asarray(x)))
# A hand-computed first row detects a shared layout mistake.
np.testing.assert_allclose(actual[0], [3.1, -.45], atol=1e-6)
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(actual, x @ keras_kernel + bias, atol=1e-6)
# Print the observed values to compare against the expected result.
print("Matched dense outputs:", actual)

# Figure data experiment
# Compute figure data for: A transpose changes storage layout, not the intended model
# Compute `visual_data` from `{'kind': 'panels', 'panels': [{'kind': 'heatmap', 't...`
visual_data = {'kind': 'panels', 'panels': [{'kind': 'heatmap', 'title': 'Input × output layout', 'values': keras_kernel.tolist(), 'rows': ['input 0', 'input 1', 'input 2'], 'columns': ['output 0', 'output 1'], 'unit': 'weight'}, {'kind': 'heatmap', 'title': 'Output × input layout', 'values': torch_weight.tolist(), 'rows': ['output 0', 'output 1'], 'columns': ['input 0', 'input 1', 'input 2'], 'unit': 'weight'}]}

# Experiment: A square matrix hides a transpose
# Experiment — A square matrix hides a transpose: Check asymmetric values and independently known outputs, not...
# Compute `square` from `np.array([[1., 2.], [3., 4.]], np.float32)`
square = np.array([[1., 2.], [3., 4.]], np.float32)
# Compute `z` from `np.array([[2., -1.]], np.float32)`
z = np.array([[2., -1.]], np.float32)
# Check tensor shape invariant: `(z @ square).shape == (z @ square.T).shape`
assert (z @ square).shape == (z @ square.T).shape
# Assert that `not np.allclose(z @ square, z @ square.T)`.
assert not np.allclose(z @ square, z @ square.T)
# Print the observed values to compare against the expected result.
print("Equal shapes, unequal outputs")

# Experiment: Reorder features without changing tensor dimensions
# Experiment — Reorder features without changing tensor dimensions: The repair changes the pairing of features and coefficients.
permutation = [2, 1, 0]
# Compute `reordered` from `x[:, permutation]`
reordered = x[:, permutation]
# Convert `wrong_order` to a host NumPy array for inspection or verification.
wrong_order = np.asarray(predict(params, reordered))
# Compute `np.testing.assert_allclose(wrong_order[0], [-.9, 4.05], atol` as `1e-6)`.
np.testing.assert_allclose(wrong_order[0], [-.9, 4.05], atol=1e-6)
# Assert that `not np.allclose(wrong_order, actual)`.
assert not np.allclose(wrong_order, actual)
# Compute `repaired_params` from `{'kernel': params['kernel'][permutation, :], 'bias':...`
repaired_params = {'kernel': params['kernel'][permutation, :], 'bias': params['bias']}
# Compute `np.testing.assert_allclose(predict(repaired_params, reordered), actual, atol` as `1e-6)`.
np.testing.assert_allclose(predict(repaired_params, reordered), actual, atol=1e-6)
# Print the observed values to compare against the expected result.
print('Feature order failed, then matched after aligning kernel rows.')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Add a batch with four rows and preserve the same parameters.
# Construct and reshape `changed` into the target tensor dimensions.
changed = np.arange(12, dtype=np.float32).reshape(4, 3) / 4
# Combine or mask array elements to form `expected`.
expected = np.stack([sum(row[i] * keras_kernel[i] for i in range(3)) + bias for row in changed])
# Compute `np.testing.assert_allclose(predict(params, changed), expected, atol` as `2e-6)`.
np.testing.assert_allclose(predict(params, changed), expected, atol=2e-6)
# Print the observed values to compare against the expected result.
print("Changed batch verified")

# Reference practice: Detect a preprocessing mismatch
# Detect a preprocessing mismatch (Transfer): Input normalization belongs in the deployment contract.
# Compute `pixels` from `np.array([[255., 128., 0.]], np.float32)`
pixels = np.array([[255., 128., 0.]], np.float32)
# Run `predict` to compute `source`.
source = predict(params, pixels / 255.)
# Run `predict` to compute `wrong`.
wrong = predict(params, pixels)
# Assert that `not np.allclose(source, wrong)`.
assert not np.allclose(source, wrong)
# Compute `np.testing.assert_allclose(source, predict(params, pixels / 255.), atol` as `1e-6)`.
np.testing.assert_allclose(source, predict(params, pixels / 255.), atol=1e-6)

# Reference practice: Map a second layer and locate an omitted activation
# Map a second layer and locate an omitted activation (Transfer / diagnosis): The first dense layer agrees, but the activation is the...
# Compute `second_kernel` from `np.array([[2.], [-1.]], np.float32)`
second_kernel = np.array([[2.], [-1.]], np.float32)
# Compute `second_bias` from `np.array([.3], np.float32)`
second_bias = np.array([.3], np.float32)
# Convert `hidden` to a host NumPy array for inspection or verification.
hidden = np.maximum(np.asarray(predict(params, x)), 0.)
# Perform matrix contraction / projection to compute `bridged_scores`.
bridged_scores = hidden @ second_kernel + second_bias
# Compute `np.testing.assert_allclose(bridged_scores[:, 0], [6.5, -4.25], atol` as `2e-6)`.
np.testing.assert_allclose(bridged_scores[:, 0], [6.5, -4.25], atol=2e-6)
# Convert `omitted_relu` to a host NumPy array for inspection or verification.
omitted_relu = np.asarray(predict(params, x)) @ second_kernel + second_bias
# Assert that `not np.allclose(omitted_relu, bridged_scores)`.
assert not np.allclose(omitted_relu, bridged_scores)
# Compute `positive_probe` from `np.array([[1., 2.]], np.float32)`
positive_probe = np.array([[1., 2.]], np.float32)
# Execute `np.testing.assert_array_equal(np.maximum(positive_probe, 0), positive_probe)`.
np.testing.assert_array_equal(np.maximum(positive_probe, 0), positive_probe)
# Print the observed values to compare against the expected result.
print('Two-layer reference scores:', bridged_scores[:, 0])
print("PASS: deployment-01")
