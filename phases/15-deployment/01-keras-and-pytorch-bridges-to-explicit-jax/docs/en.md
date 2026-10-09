# Move models between JAX, Keras, TensorFlow and PyTorch

Phase 15: Deployment, interoperability & edge AI · about 110 minutes · CPU

## What you will be able to do

- Map Dense and Linear weight axes and verify a known output.
- Separate architecture, weights, non-trainable state, optimizer state and preprocessing.
- Run a backend-neutral Keras model and explain the boundaries of tensor exchange and export.

## The problem

You have a dense layer in PyTorch and the same model idea in Keras. Can you reuse the weights in JAX? We’ll start with one small layer, work out a prediction by hand, and check the weight axes. From there, you’ll learn which parts of a model can transfer directly and which need separate checks.

## The idea

Cross-framework conversion must preserve the meaning of inputs, parameters and outputs. A layout transpose can be necessary, but architecture, preprocessing and arithmetic conventions must agree too. Start by writing the function each framework computes.

## Map the function, not just the weight file

For a dense layer, one framework may store weights as input-by-output while another stores output-by-input. The transpose can align the intended multiplication. If both dimensions happen to match, an incorrect mapping may still have a legal shape.

Use unequal dimensions and distinguishable values, then compare intermediate results on the same input. Check bias placement, activation and normalization conventions instead of inferring equivalence from a successful load.

The transpose heatmaps show storage correspondence. They do not demonstrate conversion of an arbitrary Keras, TensorFlow or PyTorch architecture. A supported converter needs an explicit model-family contract, including tokenizer or image/audio processor identity where relevant.

### Pause and reason

Why is a square weight matrix a weak first conversion test?

<details><summary>Compare your reasoning</summary>

The wrong orientation can still have the expected shape. Unequal dimensions and independently known values make layout mistakes easier to detect.

</details>

## A parameter file is not a model

Our inputs have shape $[\mathrm{batch}, 3]$. Keras Dense uses a $[3, 2]$ kernel; PyTorch Linear stores $[2, 3]$ weights and multiplies by their transpose. Bias has shape $[2]$. For $x=[1, 2, -1]$, the first output is $1+1+1+0.1=3.1$. Verify a hand-derived row before comparing whole arrays. Convolution kernels have additional layout conventions; do not apply this dense transpose rule to them.

## Use Keras as a shared model vocabulary

Keras $3$ can execute models on JAX, TensorFlow or PyTorch. Select KERAS_BACKEND before importing keras, in a fresh process. Use `keras.layers` and `keras.ops` in portable model code; a call to tf.* inside a custom layer is a TensorFlow dependency. This does not port arbitrary Python side effects or framework-specific training loops. A .keras archive preserves Keras configuration and weights; an inference export has a different job.

## Explicit JAX state is a design decision

Pass parameters and non-trainable state explicitly; return updated state and keep random keys visible. A matched inference result does not show that dropout randomness, BatchNorm moving statistics or optimizer slots were transferred. Start in inference mode with `training=False.` Compare gradients and an update separately before claiming training equivalence. Reinitializing the optimizer is a new training run, not an exact resume.

## Choose the bridge for the destination

NumPy exchange makes a clear host copy and usually breaks autodiff across the boundary. DLPack exchanges supported tensor buffers between frameworks; sharing memory is not model conversion or cross-framework autodiff, and external mutation can violate JAX assumptions. JAX export serializes a staged computation for compatible consumers. jax2tf is a separate, version-sensitive TensorFlow bridge. Keras export can target supported inference formats; inspect the destination operators and signatures. ONNX is a model interchange format, while ONNX Runtime execution providers choose supported execution paths. PyTorch/XLA runs PyTorch through XLA; it does not turn a PyTorch model into an explicit JAX function.

## Run the optional real-framework lab

The file labs/keras_backends.py beside this lesson builds the same Dense model, fixes its weights, compares known outputs, and inspects mixed precision policy. From the extracted course workspace root, use an isolated environment with Keras and the backend you want. Run the same file in a new process for each backend; --backend selects it before importing Keras. Tested locally with Keras 3.14.1 on the installed JAX and PyTorch backends. The same companion also passed on TensorFlow 2.21.0 with Keras 3.15.1 in the separate edge-lab environment. These optional backend checks are recorded separately from the core CPU receipt.

**Optional Keras environment — macOS/Linux, workspace root**

```sh
# Run optional keras environment — macos/linux, workspace root using the course Python environment
python3 -m venv .venv-interop
.venv-interop/bin/python -m pip install "keras==3.14.1" "jax==0.9.2" "torch==2.11.0"
```

**Expected:** An isolated environment with the tested Keras/JAX/PyTorch versions. Use .venv-interop/bin/python instead of python3 in the commands below; on Windows use .venv-interop\Scripts\python.exe. TensorFlow is a separate optional backend: use the edge lab environment instructions and record its installed versions.

**Optional lab — run from the course workspace root after installing Keras and the selected backend**

```sh
# Run optional lab — run from the course workspace root after installing keras and the selected backend using the course Python environment
python3 phases/15-deployment/01-keras-and-pytorch-bridges-to-explicit-jax/labs/keras_backends.py --backend jax
python3 phases/15-deployment/01-keras-and-pytorch-bridges-to-explicit-jax/labs/keras_backends.py --backend torch
python3 phases/15-deployment/01-keras-and-pytorch-bridges-to-explicit-jax/labs/keras_backends.py --backend tensorflow
```

**Expected:** Each installed backend prints PASS and measured precision errors. A missing backend must be installed in its isolated environment before running that command.

## Write the same dense layer in both layouts

We’ll call the input matrix $X$, the Keras-style kernel $W$, and the PyTorch-style weight $W_{\mathrm{torch}}$. The math agrees when $W=W_{\mathrm{torch}}^\mathsf{T}$. The transpose changes how weights are arranged, not what the layer means. Bias $b$ is added to every output row. Check the known first row before trying a larger model.

$$
Y=XW+b=XW_{\mathrm{torch}}^\mathsf{T}+b
$$

## A valid shape can still mean the wrong feature

Imagine that the source reads temperature, pressure and humidity in that order. The target reads humidity, pressure and temperature. Both receive three numbers, so a shape assertion passes. But the first coefficient now multiplies the wrong measurement. Write down feature names and units next to each input column before mapping a weight tensor.

For the worked first row, the two scores are $1+1+1+0.1=3.1$ and $-2+2-0.25-0.2=-0.45$. Swapping the first and last features gives $-0.9$ and $4.05$. These large differences are not floating-point noise. Follow the first mismatch: raw values, normalized values, first layer, activation, then output labels. This order prevents you from adjusting a numerical tolerance to conceal a different function.

### Pause and reason

If the target swaps input features and also swaps the matching rows of its kernel, should its output change?

<details><summary>Compare your reasoning</summary>

No. Both changes restore each feature–coefficient pairing. Swapping only the values or only the rows changes the function. This is a semantic alignment check, separate from changing the storage layout from input-by-output to output-by-input.

</details>

## Use a small bridge before a complete model

Keep the dense example as a test you can reason through by hand. Then add one operation at a time: a nonlinearity, normalization, and a second layer. Record an intermediate tensor at each boundary. For a convolution, declare both image layout and kernel layout; for attention, declare token order, head order, masks, and how query, key and value weights are packed.

The optional Keras lab runs real backends in separate processes. The default companion below uses NumPy and JAX to expose the mapping arithmetic, so its success alone does not certify TensorFlow, PyTorch, or a complete pretrained checkpoint. Carry a table of source names, target names, shapes, axis permutations and test inputs into the model-specific conversion lesson.

## Write the source function and its known input

Create main.py in your lesson workspace and run it with the active course Python environment. Keep feature order fixed while you name the input, kernel and bias.

```python
# Step 1 — Write the source function and its known input: The arrays define a batch of shape (2,3), a kernel of shape (3,2),...
# Import numpy for this computation.
import numpy as np
import jax
import jax.numpy as jnp
# Same dense layer, different parameter layouts.
x = np.array([[1., 2., -1.], [-2., 0., 3.]], np.float32)
# Initialize array `keras_kernel` with explicit values and shape.
keras_kernel = np.array([[1., -2.], [.5, 1.], [-1., .25]], np.float32)
# Initialize array `bias` with explicit values and shape.
bias = np.array([.1, -.2], np.float32)

# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(x[0] @ keras_kernel + bias, [3.1, -.45], atol=1e-6)
```

The arrays define a batch of shape $(2,3)$, a kernel of shape $(3,2)$, and a bias of shape $(2,)$. The first known output is $[3.1,-0.45]$.

## Map the stored tensor into explicit JAX parameters

Append this block to the same main.py and rerun the whole file. Transpose the stored PyTorch-style weights back into input-by-output order, then define the pure prediction function.

```python
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
```

The zero-input check returns the bias twice. If it does not, inspect bias broadcasting before comparing full predictions.

## Verify values before extending the architecture

Append this block to the same main.py and rerun the whole file. Run both independent checks and inspect the two output rows.

```python
# Step 3 — Verify values before extending the architecture: The first assertion checks hand arithmetic; the second compares...
# Wrap with `jax.jit` (`actual`) so XLA traces and compiles the function.
actual = np.asarray(jax.jit(predict)(params, jnp.asarray(x)))
# A hand-computed first row detects a shared layout mistake.
np.testing.assert_allclose(actual[0], [3.1, -.45], atol=1e-6)
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(actual, x @ keras_kernel + bias, atol=1e-6)
# Print the observed values to compare against the expected result.
print("Matched dense outputs:", actual)
```

The first assertion checks hand arithmetic; the second compares the entire batch with the source-layout calculation. Continue with the feature-order and second-layer experiments below.

## Run the example

```python
# Move models between JAX, Keras, TensorFlow and PyTorch: Cross-framework conversion must preserve the meaning of inputs,...
# Import numpy for this computation.
import numpy as np
import jax
import jax.numpy as jnp
# Same dense layer, different parameter layouts.
x = np.array([[1., 2., -1.], [-2., 0., 3.]], np.float32)
# Initialize array `keras_kernel` with explicit values and shape.
keras_kernel = np.array([[1., -2.], [.5, 1.], [-1., .25]], np.float32)
# Initialize array `bias` with explicit values and shape.
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
```

Expected: Two rows match; the first is $[3.1, -0.45]$ within absolute tolerance $10^{-6}$.

## A transpose changes storage layout, not the intended model

**Predict:** Which axis names swap between the two parameter layouts?

![A transpose changes storage layout, not the intended model](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The upper matrix has input features as rows and output features as columns. The lower matrix swaps those axes. The cell values are weights, and the two panels contain the same six numbers in transposed positions.

Track the weight $-1$: it moves from input $2$, output $0$ in the upper panel to output $0$, input $2$ in the lower panel. It still connects the same input feature to the same output feature. The layout changes from shape $(3,2)$ to $(2,3)$.

### Connect it to the computation

With the first convention, a row batch computes $XW+b$. With the transposed storage convention, the equivalent calculation uses $X\widetilde W^{\mathsf T}+b$, where $\widetilde W=W^{\mathsf T}$. Transposing storage without adapting the computation would change the meaning or produce a shape error.

The picture explains this layout contract using an explicit numerical reference. It does not demonstrate that every layer or serialized model can be transferred between frameworks by one transpose. Check activations, bias, preprocessing, and layer-specific conventions against an independent output reference as well.

```python
# Compute figure data for: A transpose changes storage layout, not the intended model
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'panels', 'panels': [{'kind': 'heatmap', 'title': 'Input × output layout', 'values': keras_kernel.tolist(), 'rows': ['input 0', 'input 1', 'input 2'], 'columns': ['output 0', 'output 1'], 'unit': 'weight'}, {'kind': 'heatmap', 'title': 'Output × input layout', 'values': torch_weight.tolist(), 'rows': ['output 0', 'output 1'], 'columns': ['input 0', 'input 1', 'input 2'], 'unit': 'weight'}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:05:59.032313+00:00. JAX 0.9.2.

```text
Matched dense outputs: [[ 3.1  -0.45]
 [-4.9   4.55]]
Matched dense outputs: [[ 3.1  -0.45]
 [-4.9   4.55]]
Equal shapes, unequal outputs
Feature order failed, then matched after aligning kernel rows.
Changed batch verified
Two-layer reference scores: [ 6.5  -4.25]
PASS: deployment-01

```

## A square matrix hides a transpose

**Predict before running:** Will a shape check catch wrong axes when input and output dimensions are equal?

```python
# Experiment — A square matrix hides a transpose: Check asymmetric values and independently known outputs, not...
# Initialize array `square` with explicit values and shape.
square = np.array([[1., 2.], [3., 4.]], np.float32)
# Initialize array `z` with explicit values and shape.
z = np.array([[2., -1.]], np.float32)
# Verify that the output tensor shape matches our prediction.
assert (z @ square).shape == (z @ square.T).shape
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not np.allclose(z @ square, z @ square.T)
# Print the observed values to compare against the expected result.
print("Equal shapes, unequal outputs")
```

**Expected:** Both outputs have the same shape but different values.

Check asymmetric values and independently known outputs, not just tensor dimensions.

## Reorder features without changing tensor dimensions

**Predict before running:** Predict both scores if the first and last input features are exchanged while the weights stay fixed.

```python
# Experiment — Reorder features without changing tensor dimensions: The repair changes the pairing of features and coefficients.
permutation = [2, 1, 0]
# Evaluate `reordered` from the current inputs and state.
reordered = x[:, permutation]
# Convert `wrong_order` to a host NumPy array for inspection or verification.
wrong_order = np.asarray(predict(params, reordered))
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(wrong_order[0], [-.9, 4.05], atol=1e-6)
# Verify that the numerical values match the expected reference within tolerance.
assert not np.allclose(wrong_order, actual)
# Evaluate `repaired_params` from the current inputs and state.
repaired_params = {'kernel': params['kernel'][permutation, :], 'bias': params['bias']}
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(predict(repaired_params, reordered), actual, atol=1e-6)
# Print the observed values to compare against the expected result.
print('Feature order failed, then matched after aligning kernel rows.')
```

**Expected:** The wrong first row is $[-0.9,4.05]$; permuting the kernel rows restores both original observations.

The repair changes the pairing of features and coefficients. It does not change the model represented by those pairings. Repeat with asymmetric feature values; repeated values can hide a wrong order.

## Make it yours

Add a batch with four rows and preserve the same parameters. Verify each row independently using a feature-by-feature sum.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `array.reshape(new_shape)` — Reorganizes tensor axes without changing the total element count (`array.size`).
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Construct and reshape `changed` into the target tensor dimensions.
2. Combine or mask array elements to form `expected`.
3. Verify that computed values match the expected reference within numerical tolerance.
4. Print the observed values to compare against the expected result.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Add a batch with four rows and preserve the same parameters.
# Construct and reshape `changed` into the target tensor dimensions.
changed = np.arange(...)  # TODO: compute changed
# Combine or mask array elements to form `expected`.
expected = np.stack(...)  # TODO: compute expected
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(predict(params, changed), expected, atol = ...  # TODO: compute np.testing.assert_allclose(predict(params, changed), expected, atol
# Print the observed values to compare against the expected result.
print("Changed batch verified")
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Add a batch with four rows and preserve the same parameters.
# Construct and reshape `changed` into the target tensor dimensions.
changed = np.arange(12, dtype=np.float32).reshape(4, 3) / 4
# Combine or mask array elements to form `expected`.
expected = np.stack([sum(row[i] * keras_kernel[i] for i in range(3)) + bias for row in changed])
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(predict(params, changed), expected, atol=2e-6)
# Print the observed values to compare against the expected result.
print("Changed batch verified")
```

</details>

## Detect a preprocessing mismatch

**Transfer**

Suppose the source model expects inputs divided by $255$, while the target receives raw pixels. Demonstrate why correct weight transfer still fails.

<details><summary>Hint</summary>

Compute both predictions from the same raw input: once with the source normalization and once without it. Matching weights cannot cancel an input-scale mismatch.

</details>

### How to write: Detect a preprocessing mismatch — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Initialize array `pixels` with explicit values and shape.
2. Run `predict` to compute `source`.
3. Run `predict` to compute `wrong`.
4. Verify that the numerical values match the expected reference within tolerance.
5. Verify that computed values match the expected reference within numerical tolerance.

**Starter code scaffold (fill in the TODOs):**

```python
# Detect a preprocessing mismatch (Transfer): Input normalization belongs in the deployment contract.
# Initialize array `pixels` with explicit values and shape.
pixels = np.array(...)  # TODO: compute pixels
# Run `predict` to compute `source`.
source = predict(...)  # TODO: compute source
# Run `predict` to compute `wrong`.
wrong = predict(...)  # TODO: compute wrong
# Verify that the numerical values match the expected reference within tolerance.
assert not np.allclose(source, wrong)  # TODO: complete assertion check
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(source, predict(params, pixels / 255.), atol=1e-6)
```

<details><summary>Reference solution and reasoning</summary>

```python
# Detect a preprocessing mismatch (Transfer): Input normalization belongs in the deployment contract.
# Initialize array `pixels` with explicit values and shape.
pixels = np.array([[255., 128., 0.]], np.float32)
# Run `predict` to compute `source`.
source = predict(params, pixels / 255.)
# Run `predict` to compute `wrong`.
wrong = predict(params, pixels)
# Verify that the numerical values match the expected reference within tolerance.
assert not np.allclose(source, wrong)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(source, predict(params, pixels / 255.), atol=1e-6)
```

Input normalization belongs in the deployment contract. Matching weights cannot compensate for a different input distribution.

</details>

## Map a second layer and locate an omitted activation

**Transfer / diagnosis**

Extend the dense mapping with ReLU and a second dense layer. Use independently computed final scores to catch a converter that forgets ReLU. Then show why an all-positive probe would be a weak test.

<details><summary>Hint</summary>

Compute both first-layer rows by hand, replace negative entries by zero, and apply the new output weights.

</details>

### How to write: Map a second layer and locate an omitted activation — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Initialize array `second_kernel` with explicit values and shape.
2. Initialize array `second_bias` with explicit values and shape.
3. Convert `hidden` to a host NumPy array for inspection or verification.
4. Perform matrix contraction / projection to compute `bridged_scores`.
5. Verify that computed values match the expected reference within numerical tolerance.

**Starter code scaffold (fill in the TODOs):**

```python
# Map a second layer and locate an omitted activation (Transfer / diagnosis): The first dense layer agrees, but the activation is the...
# Initialize array `second_kernel` with explicit values and shape.
second_kernel = np.array(...)  # TODO: compute second_kernel
# Initialize array `second_bias` with explicit values and shape.
second_bias = np.array(...)  # TODO: compute second_bias
# Convert `hidden` to a host NumPy array for inspection or verification.
hidden = np.maximum(...)  # TODO: compute hidden
# Perform matrix contraction / projection to compute `bridged_scores`.
bridged_scores = ...  # TODO: compute bridged_scores
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(bridged_scores[:, 0], [6.5, -4.25], atol=2e-6)
# Convert `omitted_relu` to a host NumPy array for inspection or verification.
omitted_relu = np.asarray(...)  # TODO: compute omitted_relu
# Verify that the numerical values match the expected reference within tolerance.
assert not np.allclose(omitted_relu, bridged_scores)  # TODO: complete assertion check
# Initialize array `positive_probe` with explicit values and shape.
positive_probe = np.array(...)  # TODO: compute positive_probe
# Reduce across the target axis to summarize ``.
np.testing.assert_array_equal(np.maximum(positive_probe, 0), positive_probe)
# Print the observed values to compare against the expected result.
print('Two-layer reference scores:', bridged_scores[:, 0])
```

<details><summary>Reference solution and reasoning</summary>

```python
# Map a second layer and locate an omitted activation (Transfer / diagnosis): The first dense layer agrees, but the activation is the...
# Initialize array `second_kernel` with explicit values and shape.
second_kernel = np.array([[2.], [-1.]], np.float32)
# Initialize array `second_bias` with explicit values and shape.
second_bias = np.array([.3], np.float32)
# Convert `hidden` to a host NumPy array for inspection or verification.
hidden = np.maximum(np.asarray(predict(params, x)), 0.)
# Perform matrix contraction / projection to compute `bridged_scores`.
bridged_scores = hidden @ second_kernel + second_bias
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(bridged_scores[:, 0], [6.5, -4.25], atol=2e-6)
# Convert `omitted_relu` to a host NumPy array for inspection or verification.
omitted_relu = np.asarray(predict(params, x)) @ second_kernel + second_bias
# Verify that the numerical values match the expected reference within tolerance.
assert not np.allclose(omitted_relu, bridged_scores)
# Initialize array `positive_probe` with explicit values and shape.
positive_probe = np.array([[1., 2.]], np.float32)
# Reduce across the target axis to summarize ``.
np.testing.assert_array_equal(np.maximum(positive_probe, 0), positive_probe)
# Print the observed values to compare against the expected result.
print('Two-layer reference scores:', bridged_scores[:, 0])
```

The first dense layer agrees, but the activation is the first divergent boundary. The final reference scores are $6.5$ and $-4.25$. A probe containing only positive hidden values would make an omitted ReLU invisible.

</details>

## Check your understanding

The Dense predictions match after transposing weights. What has been established?

1. The tested inference mapping matches for these inputs.
2. Optimizer and random state are now equivalent.
3. Any runtime can execute the model.

<details><summary>Answer and explanation</summary>

The tested inference mapping matches for these inputs.

The comparison establishes bounded numerical equivalence. Training state, other operators, shapes and target runtime support each need their own evidence.

</details>

## Diagnose the result

If output shapes fail, write the contraction axes. If shapes match but values drift, check layout, preprocessing, bias, inference mode and dtype in that order. A missing TensorFlow import in the optional lab means that backend is not installed; it is not a failure of the JAX core exercise.

## Keep your evidence

Keep the layout map, two changed-input comparisons, runtime versions and optional backend-lab output. Identify which model state and preprocessing are not covered by a dense-layer match.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [Keras multi-backend migration](https://keras.io/guides/migrating_to_keras_3/)
- [Keras inference export](https://keras.io/api/models/model_saving_apis/export/)
- [JAX DLPack contract](https://docs.jax.dev/en/latest/_autosummary/jax.dlpack.from_dlpack.html)

