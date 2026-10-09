# Deploy at the edge: conversion, budgets and device checks

Phase 15: Deployment, interoperability & edge AI · about 110 minutes · CPU

## What you will be able to do

- Trace source model through conversion, runtime and device execution.
- Measure a complete batch-one request separately from first-call overhead.
- Define calibration, held-out quality and device acceptance checks.
- Identify fallback operators and memory costs before claiming edge readiness.

## The problem

A model works on your computer, and now you want it to run near the user—perhaps on a phone or an embedded board. Let’s trace the whole request, from raw input to usable output. We’ll check conversion, measure a batch-one CPU request, and write down what still needs to be measured on the real device.

## The idea

An edge deployment includes input decoding, preprocessing, transfers, inference and output handling under device constraints. Choose a target and an explicit request contract before claiming that an exported model is ready for that device.

## Measure the device path the user actually experiences

A converted operator may run on the intended accelerator while another falls back to the CPU. The end-to-end request can then pay transfer and scheduling costs absent from an isolated kernel measurement. Inspect actual runtime placement and supported operators.

Memory also has several parts: weights, activations, temporary workspace and runtime overhead. A smaller weight file does not guarantee the whole request fits the device's working-memory budget.

The CPU request trace is a useful reference for the interface. It is distinct from phone, browser, NPU or TPU performance. Repeat the same correctness cases on the named target, then measure cold and warm behavior with its runtime and power/thermal conditions recorded when relevant.

### Model speed is one part of request speed

**Predict:** If inference becomes twice as fast, does the whole request become twice as fast?

![Model speed is one part of request speed](../../phases/15-deployment/06-edge-ai-conversion-and-device-validation/outputs/mechanism.svg)

*Architecture and dataflow mechanism diagram.*

Read the two columns as a hypothetical serial request before and after halving only inference. Preprocessing and transfer/response work remain unchanged. Add down each column: the total moves from $24$ to $20$ milliseconds, a $1.2$-fold improvement. These times are analytic teaching values; the separate plot contains actual local CPU request measurements.

### Pause and reason

What should you verify after conversion before interpreting a faster timing?

<details><summary>Compare your reasoning</summary>

Verify output behavior on the same inputs, operator placement/fallbacks and the complete request boundary. A faster but different computation is not a qualified optimization.

</details>

## Choose a deployment lane

For a phone or Linux edge device, LiteRT is one runtime option; ONNX Runtime offers execution providers for supported targets. Apple deployments may use Core ML after a supported conversion; PyTorch models may use ExecuTorch. Microcontrollers require a runtime and operator set designed for constrained memory, such as LiteRT for Microcontrollers. These are alternatives, not interchangeable file formats. Write down OS, device model, runtime version, operators, input shapes, precision policy and accelerator path before converting.

## Follow one concrete TensorFlow-to-LiteRT route

The optional labs/tensorflow_litert.py companion builds the known dense model in TensorFlow, converts float32 and fully integer variants, inspects input/output scale and zero point, quantizes requests and dequantizes outputs, then checks held-out predictions. Representative calibration inputs are preprocessed floats in the exact source-model input domain. Setting integer-only supported ops and int8 input/output requests a strict integer graph; unsupported conversion should fail rather than silently satisfy the contract with float fallback. The lab uses TensorFlow’s interpreter interface as a desktop compatibility check; it is not a device delegate test. Desktop conversion was verified with Python 3.12.13, TensorFlow 2.21.0 and Keras 3.15.1 on macOS arm64. The optional requirements file pins these tested top-level dependencies; other platforms still need a compatible installation.

**Optional TensorFlow environment — macOS/Linux, from the workspace root**

```sh
# Run optional tensorflow environment — macos/linux, from the workspace root using the course Python environment
python3.12 -m venv .venv-tensorflow
.venv-tensorflow/bin/python -m pip install -r phases/15-deployment/06-edge-ai-conversion-and-device-validation/labs/requirements-tensorflow.txt
.venv-tensorflow/bin/python phases/15-deployment/06-edge-ai-conversion-and-device-validation/labs/tensorflow_litert.py --output edge-artifacts
.venv-tensorflow/bin/python -m pip freeze > edge-artifacts/requirements-tested.txt
```

**Expected:** A compatible TensorFlow installation writes two .tflite files and report.json after numerical checks. Desktop float32 and INT8 conversion passed separately from the core CPU receipt; device validation remains unperformed. On Windows use .venv-tensorflow\Scripts\python.exe in place of .venv-tensorflow/bin/python. If no compatible TensorFlow wheel exists, choose a supported Python/platform combination from the official installation guide.

## Keep preprocessing attached to the model

Our sample uses three sensor features in the range $0$ to $255$ and divides by $255$ before inference. The JSON request exercises that boundary. A camera model also needs color order, resize/crop behavior and normalization; tokenizer-based models need vocabulary, special tokens and sequence rules. Keep preprocessing versioned with weights and output labels. Run the same raw golden inputs through the source pipeline and the device pipeline.

## Budget more than weights

Peak memory includes activations, tensors held by the runtime, workspace, preprocessing buffers and, for autoregressive models, KV cache. A compressed file may expand on load. Batch one often minimizes interaction latency but may reduce throughput. Include cold process startup, model load and runtime initialization in a separate device cold-start measurement; our first-request timing starts after imports and model setup, so it does not cover them. Measure power or energy with an actual target-device tool and record thermal state.

## Measure the request the user experiences

The timer includes JSON decoding, normalization, host/device placement, inference completion and response encoding. Thirty warm requests produce an instructional p50 and p95, not a production tail-latency estimate. A device report needs more samples, realistic input variation and sustained runs to reveal thermal throttling. Never reuse a kernel-only time as end-to-end latency. Verify the numerical output before interpreting any timings.

## Device acceptance is an experiment

Choose quality, memory, startup and latency thresholds before comparing candidates. Record runtime/delegate logs and partitioning so unsupported operators cannot quietly fall back to CPU. Compare float32, FP16 and calibrated INT8 where that device supports them. Retain the original artifact, hashes, versions, input/output metadata, calibration provenance and failed cases. A conversion failure, regression on rare inputs or memory-budget failure is a result to diagnose. Leave device fields unmeasured until tested on the named hardware.

## Declare the raw sensor contract before normalizing

This exercise uses one row of three numeric sensor values in the range $[0,255]$. Numeric strings, booleans, missing values, extra fields and out-of-range readings are rejected. That is a deliberately chosen protocol for this fixture, not a rule for every sensor. If a physical device uses a different unit or range, change and version the contract before changing the model.

Check the raw values before converting them to an array. Otherwise a convenient cast may turn a string or boolean into a plausible number, hiding a caller's mistake. After validation, divide by $255$ exactly once. Keep that conversion with the model release so the desktop reference and device application see the same numeric inputs.

## Spend a latency budget across the whole request

Suppose decoding and preprocessing take $12$ milliseconds, transfers and response handling take $4$, and inference takes $8$. The request takes $24$ milliseconds if these stages run serially. Halving inference time reduces the total to $20$, not $12$. The model sped up by a factor of two; the request sped up by only $24/20=1.2$.

These are illustrative stage times, not observations from the course device. On a real target, instrument the same request boundary, check whether stages overlap, and record warmup, temperature, power mode and operator fallback. If transfers dominate, a smaller model may help less than reducing transfers. Compare those hypotheses with traces before choosing an intervention.

$$
\mathrm{speedup}=\frac{t_{\mathrm{other}}+t_{\mathrm{model}}}{t_{\mathrm{other}}+t_{\mathrm{model}}/s}
$$

## Keep a known inference function

Create main.py in your lesson workspace and run it with the active course Python environment. Define the fixed dense model and check the parameter shapes without invoking the compiled function.

```python
# Step 1 — Keep a known inference function: A row has three sensor features and the result has two scores.
# Import json for this computation.
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
# Construct `weights` via `jnp.array([[1., -2.], [.5, 1.], [-1., .25]], jnp.flo...`
weights = jnp.array([[1., -2.], [.5, 1.], [-1., .25]], jnp.float32)
# Construct `bias` via `jnp.array([.1, -.2], jnp.float32)`
bias = jnp.array([.1, -.2], jnp.float32)
# Define and JIT-compile `infer(features)` so XLA traces and fuses the operations:
@jax.jit
# Function `infer(features)` implementing this stage's computation:
def infer(features):
    # Return `features @ weights + bias` to the caller.
    return features @ weights + bias

# Check tensor shape invariant: `weights.shape == (3, 2) and bias.shape == (2,)`
assert weights.shape == (3, 2) and bias.shape == (2,)
```

A row has three sensor features and the result has two scores. Leave the compiled function uncalled so the later first-request measurement can include initialization.

## Build the raw-request boundary

Append this block to the same main.py and rerun the whole file. Validate the raw JSON protocol, normalize once, wait for inference, and encode the result.

```python
# Step 2 — Build the raw-request boundary: This probe checks decoding and raw values without running inference.
payload = json.dumps({"features": [[255., 128., 0.]]})
# Function `validate_sensor_payload(payload)` implementing this stage's computation:
def validate_sensor_payload(payload):
    # Read or serialize artifact data on disk (`document`).
    document = json.loads(payload)
    # Guard input contract (`not isinstance(document, dict) or set(document) != {'features'}`) and fail fast if violated.
    if not isinstance(document, dict) or set(document) != {'features'}:
        raise ValueError('expected features object')
    # Compute `rows` from `document['features']`
    rows = document['features']
    # Guard input contract (`not isinstance(rows, list) or len(rows) != 1 or (not isinstance(rows[0], list)) or (len(rows[0]) != 3)`) and fail fast if violated.
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list) or len(rows[0]) != 3:
        raise ValueError('expected one row of three sensor values')
    # Guard input contract (`any((type(value) not in (int, float) or not 0 <= value <= 255 or (not np.isfinite(value)) for value in rows[0]))`) and fail fast if violated.
    if any(type(value) not in (int, float) or not 0 <= value <= 255 or not np.isfinite(value) for value in rows[0]):
        raise ValueError('expected numeric sensor values in the range 0 through 255')
    # Return `np.asarray(rows, dtype=np.float32)` to the caller.
    return np.asarray(rows, dtype=np.float32)
# Function `request(payload)` implementing this stage's computation:
def request(payload):
    # Run `validate_sensor_payload` to compute `decoded`.
    decoded = validate_sensor_payload(payload)
    # Compute `features` from `decoded / 255.`
    features = decoded / 255.
    # Synchronize host execution until asynchronous device computation completes.
    output = np.asarray(infer(jnp.asarray(features)).block_until_ready())
    # Return `json.dumps({'scores': output.tolist()})` to the caller.
    return json.dumps({"scores": output.tolist()})

# Execute `np.testing.assert_array_equal(validate_sensor_payload(payloa`
np.testing.assert_array_equal(validate_sensor_payload(payload), [[255., 128., 0.]])
```

This probe checks decoding and raw values without running inference. Keep normalization inside request so it happens exactly once.

## Measure cold and warm requests separately

Append this block to the same main.py and rerun the whole file. Time the first request, then retain every warm observation.

```python
# Step 3 — Measure cold and warm requests separately: The first request can include compilation and initialization.
start = time.perf_counter()
# Run `request` to compute `first_response`.
first_response = request(payload)
# Record execution timing or profiler trace in `first_ms`.
first_ms = (time.perf_counter() - start) * 1000
# Compute `samples` from `[]`
samples = []
# Repeat the update loop over `range(30)` steps:
for _ in range(30):
    # Record execution timing or profiler trace in `start`.
    start = time.perf_counter()
    # Run `request` to compute `response`.
    response = request(payload)
    # Record execution timing or profiler trace in ``.
    samples.append((time.perf_counter() - start) * 1000)
```

The first request can include compilation and initialization. The line plot contains only the subsequent warm requests; do not describe its first point as the cold request.

## Validate outputs and describe measurement scope

Append this block to the same main.py and rerun the whole file. Compare the response with independent NumPy arithmetic and write an honest measurement report.

```python
# Step 4 — Validate outputs and describe measurement scope: The report records local CPU timings and explicitly leaves...
# Compute `expected` from `(np.array([[255., 128., 0.]], np.float32) / 255.) @ ...`
expected = (np.array([[255., 128., 0.]], np.float32) / 255.) @ np.asarray(weights) + np.asarray(bias)
# Compute `np.testing.assert_allclose(json.loads(response)["scores"], expected, atol` as `1e-6)`.
np.testing.assert_allclose(json.loads(response)["scores"], expected, atol=1e-6)
# Compute `report` from `{"runtime": "JAX CPU instructional proxy", "jax": ja...`
report = {"runtime": "JAX CPU instructional proxy", "jax": jax.__version__,
          "first_request_ms": first_ms, "warm_samples_ms": samples,
          "p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95)),
          "boundary": "JSON decode + normalize + transfer + infer + wait + encode",
          "edge_device_validated": False}
# Assert invariant `len(samples) == 30 and all(t >= 0 for t in samples)` holds
assert len(samples) == 30 and all(t >= 0 for t in samples)
# Print the observed values to compare against the expected result.
print(json.dumps(report, indent=2))

# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(json.loads(request(json.dumps({'features':[[0,0,0]]})))['scores'], np.asarray(bias)[None,:], atol=1e-6)
```

The report records local CPU timings and explicitly leaves edge-device qualification false. Carry the same protocol to a named device before replacing that field.

## Run the example

```python
# Deploy at the edge: conversion, budgets and device checks: An edge deployment includes input decoding, preprocessing,...
# Import json for this computation.
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
# Construct `weights` via `jnp.array([[1., -2.], [.5, 1.], [-1., .25]], jnp.flo...`
weights = jnp.array([[1., -2.], [.5, 1.], [-1., .25]], jnp.float32)
# Construct `bias` via `jnp.array([.1, -.2], jnp.float32)`
bias = jnp.array([.1, -.2], jnp.float32)
# Define and JIT-compile `infer(features)` so XLA traces and fuses the operations:
@jax.jit
# Function `infer(features)` implementing this stage's computation:
def infer(features):
    # Return `features @ weights + bias` to the caller.
    return features @ weights + bias
# Read or serialize artifact data on disk (`payload`).
payload = json.dumps({"features": [[255., 128., 0.]]})
# Function `validate_sensor_payload(payload)` implementing this stage's computation:
def validate_sensor_payload(payload):
    # Read or serialize artifact data on disk (`document`).
    document = json.loads(payload)
    # Guard input contract (`not isinstance(document, dict) or set(document) != {'features'}`) and fail fast if violated.
    if not isinstance(document, dict) or set(document) != {'features'}:
        raise ValueError('expected features object')
    # Compute `rows` from `document['features']`
    rows = document['features']
    # Guard input contract (`not isinstance(rows, list) or len(rows) != 1 or (not isinstance(rows[0], list)) or (len(rows[0]) != 3)`) and fail fast if violated.
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list) or len(rows[0]) != 3:
        raise ValueError('expected one row of three sensor values')
    # Guard input contract (`any((type(value) not in (int, float) or not 0 <= value <= 255 or (not np.isfinite(value)) for value in rows[0]))`) and fail fast if violated.
    if any(type(value) not in (int, float) or not 0 <= value <= 255 or not np.isfinite(value) for value in rows[0]):
        raise ValueError('expected numeric sensor values in the range 0 through 255')
    # Return `np.asarray(rows, dtype=np.float32)` to the caller.
    return np.asarray(rows, dtype=np.float32)
# Function `request(payload)` implementing this stage's computation:
def request(payload):
    # Run `validate_sensor_payload` to compute `decoded`.
    decoded = validate_sensor_payload(payload)
    # Compute `features` from `decoded / 255.`
    features = decoded / 255.
    # Synchronize host execution until asynchronous device computation completes.
    output = np.asarray(infer(jnp.asarray(features)).block_until_ready())
    # Return `json.dumps({'scores': output.tolist()})` to the caller.
    return json.dumps({"scores": output.tolist()})
# Record execution timing or profiler trace in `start`.
start = time.perf_counter()
# Run `request` to compute `first_response`.
first_response = request(payload)
# Record execution timing or profiler trace in `first_ms`.
first_ms = (time.perf_counter() - start) * 1000
# Compute `samples` from `[]`
samples = []
# Repeat the update loop over `range(30)` steps:
for _ in range(30):
    # Record execution timing or profiler trace in `start`.
    start = time.perf_counter()
    # Run `request` to compute `response`.
    response = request(payload)
    # Record execution timing or profiler trace in ``.
    samples.append((time.perf_counter() - start) * 1000)
# Convert `expected` to a host NumPy array for inspection or verification.
expected = (np.array([[255., 128., 0.]], np.float32) / 255.) @ np.asarray(weights) + np.asarray(bias)
# Compute `np.testing.assert_allclose(json.loads(response)["scores"], expected, atol` as `1e-6)`.
np.testing.assert_allclose(json.loads(response)["scores"], expected, atol=1e-6)
# Compute `report` from `{"runtime": "JAX CPU instructional proxy", "jax": ja...`
report = {"runtime": "JAX CPU instructional proxy", "jax": jax.__version__,
          "first_request_ms": first_ms, "warm_samples_ms": samples,
          "p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95)),
          "boundary": "JSON decode + normalize + transfer + infer + wait + encode",
          "edge_device_validated": False}
# Assert invariant `len(samples) == 30 and all(t >= 0 for t in samples)` holds
assert len(samples) == 30 and all(t >= 0 for t in samples)
# Print diagnostic summary of the computed outputs.
print(json.dumps(report, indent=2))
```

Expected: A JSON report with $30$ measured warm requests, first-request time, p50/p95 and `edge_device_validated=false`. Timings depend on this CPU; no device speedup is asserted.

## An end-to-end boundary includes more than inference

**Predict:** Which parts of a request are included in these samples?

![An end-to-end boundary includes more than inference](../../phases/15-deployment/06-edge-ai-conversion-and-device-validation/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis counts warm requests, and the vertical axis measures end-to-end latency in milliseconds. The solid line joins the measured request times. The dashed horizontal line is the empirical $95$th percentile of those same $30$ samples; it is a summary threshold, not a second implementation.

Read a solid point above the dashed line as a request slower than that percentile threshold. The line is not the maximum, so a few points can lie above it. The first cold request is reported separately and is not the first point of this warm-only plot.

### Connect it to the computation

Each duration includes JSON decoding, normalization, transfer, inference, waiting for completion, and encoding the response. That boundary explains why the chart should not be read as pure model execution time. An early high warm point or later spike shows variation in the whole request; this plot alone cannot identify its cause.

The percentile is computed from a small sample, with interpolation between ordered values. It does not guarantee that future requests will meet the threshold. Compare the sample distribution, median, and tail across repeated runs on the target device. These are local CPU proxy measurements, not evidence of mobile or edge-device latency.

```python
# Compute figure data for: An end-to-end boundary includes more than inference
# Compute `visual_data` from `{'kind': 'line', 'x': list(range(1, len(samples) + 1...`
visual_data = {'kind': 'line', 'x': list(range(1, len(samples) + 1)), 'xlabel': 'warm request number', 'ylabel': 'end-to-end milliseconds', 'series': [{'label': 'local CPU request', 'y': samples}, {'label': 'recorded p95', 'y': [report['p95_ms']] * len(samples)}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:06:19.415261+00:00. JAX 0.9.2.

```text
{
  "runtime": "JAX CPU instructional proxy",
  "jax": "0.9.2",
  "first_request_ms": 16.428582835942507,
  "warm_samples_ms": [
    0.19670836627483368,
    0.08245836943387985,
    0.061042141169309616,
    0.054915901273489,
    0.05250005051493645,
    0.04837522283196449,
    0.04566693678498268,
    0.04462478682398796,
    0.044582877308130264,
    0.042750034481287,
    0.04204222932457924,
    0.04208320751786232,
    0.043625012040138245,
    0.04095863550901413,
    0.041291117668151855,
    0.040499959141016006,
    0.040333718061447144,
    0.040667131543159485,
    0.04016607999801636,
    0.03870902583003044,
    0.03933301195502281,
    0.038875266909599304,
    0.039041973650455475,
    0.040333252400159836,
    0.03920821473002434,
    0.04075001925230026,
    0.03970786929130554,
    0.04329206421971321,
    0.042667146772146225,
    0.03958307206630707
  ],
  "p50_ms": 0.04166667349636555,
  "p95_ms": 0.07282106671482319,
  "boundary": "JSON decode + normalize + transfer + infer + wait + encode",
  "edge_device_validated": false
}
{
  "runtime": "JAX CPU instructional proxy",
  "jax": "0.9.2",
  "first_request_ms": 12.682375032454729,
  "warm_samples_ms": [
    0.17187464982271194,
    0.07224967703223228,
    0.05766702815890312,
    0.052374787628650665,
    0.050582922995090485,
    0.04545878618955612,
    0.043958891183137894,
    0.04316680133342743,
    0.041665975004434586,
    0.041750259697437286,
    0.04054093733429909,
    0.040207989513874054,
    0.0427919439971447,
    0.039833132177591324,
    0.042667146772146225,
    0.04041707143187523,
    0.040040817111730576,
    0.0399160198867321,
    0.03979215398430824,
    0.03929203376173973,
    0.040374696254730225,
    0.03933301195502281,
    0.04020892083644867,
    0.03937492147088051,
    0.04054233431816101,
    0.0387909822165966,
    0.03916677087545395,
    0.03916723653674126,
    0.038499943912029266,
    0.038499943912029266
  ],
  "p50_ms": 0.04039588384330273,
  "p95_ms": 0.06568748503923412,
  "boundary": "JSON decode + normalize + transfer + infer + wait + encode",
  "edge_device_validated": false
}
Preprocessing mismatch max error 380.5019836425781
Seven malformed or out-of-domain requests rejected; zero input returns the bias.
Changed end-to-end request verified
Not device-validated; missing: device, runtime_version, delegate, quality_metric, p95_ms, peak_memory_bytes, cold_start_ms, fallback_operators
Illustrative request before/after/lower-bound (ms): 24.0 20.0 16.0
PASS: deployment-06

```

## A normalization mismatch survives conversion

**Predict before running:** Will raw sensor values and normalized values produce the same scores?

```python
# Experiment — A normalization mismatch survives conversion: Conversion checks must start at the raw input boundary when...
# Compute `raw` from `np.array([[255., 128., 0.]], np.float32)`
raw = np.array([[255., 128., 0.]], np.float32)
# Convert `wrong` to a host NumPy array for inspection or verification.
wrong = np.asarray(infer(raw))
# Convert `right` to a host NumPy array for inspection or verification.
right = np.asarray(infer(raw / 255.))
# Assert that `not np.allclose(wrong, right)`.
assert not np.allclose(wrong, right)
# Print the observed values to compare against the expected result.
print("Preprocessing mismatch max error", float(np.max(np.abs(wrong-right))))
```

**Expected:** The mismatch is large despite identical weights.

Conversion checks must start at the raw input boundary when preprocessing is part of the product.

## Reject raw inputs before normalization hides their meaning

**Predict before running:** Would an array cast distinguish a numeric string, a boolean and a legitimate sensor reading?

```python
# Experiment — Reject raw inputs before normalization hides their meaning: The zero-input oracle checks that preprocessing does not add a...
invalid_payloads=[{'features':[['255',128,0]]},{'features':[[True,128,0]]},{'features':[[256,128,0]]},{'features':[[-1,128,0]]},{'features':[[0,1]]},{'features':[[0,1,float('nan')]]},{'features':[[0,1,2]],'extra':1}]
# Iterate over `invalid` to step through the computation:
for invalid in invalid_payloads:
    try:
        request(json.dumps(invalid))
    except ValueError:
        pass
    else:
        raise AssertionError('invalid sensor payload accepted')
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(json.loads(request(json.dumps({'features':[[0,0,0]]})))['scores'],np.asarray(bias)[None,:],atol=1e-6)
# Print the observed values to compare against the expected result.
print('Seven malformed or out-of-domain requests rejected; zero input returns the bias.')
```

**Expected:** All seven invalid payloads fail. A valid zero-valued row produces the two bias scores.

The zero-input oracle checks that preprocessing does not add a hidden offset. Rejection tests check input meaning before the model sees an apparently valid float array.

## Make it yours

Send a second raw input $[0, 255, 128]$ through the full request, and verify its independent normalized NumPy result.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Compute `changed_raw` from `np.array([[0., 255., 128.]], np.float32)`
2. Read or serialize artifact data on disk (`changed_response`).
3. Convert `expected_changed` to a host NumPy array for inspection or verification.
4. Compute `np.testing.assert_allclose(changed_response["scores"], expected_changed, atol` as `1e-6)`.
5. Print the observed values to compare against the expected result.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Send a second raw input [0, 255, 128] through the full request, and...
# Compute `changed_raw` from `np.array([[0., 255., 128.]], np.float32)`
changed_raw = np.array(...)  # TODO: compute changed_raw
# Read or serialize artifact data on disk (`changed_response`).
changed_response = json.loads(...)  # TODO: compute changed_response
# Convert `expected_changed` to a host NumPy array for inspection or verification.
expected_changed = ...  # TODO: compute expected_changed
# Compute `np.testing.assert_allclose(changed_response["scores"], expected_changed, atol` as `1e-6)`.
np.testing.assert_allclose(changed_response["scores"], expected_changed, atol = ...  # TODO: compute np.testing.assert_allclose(changed_response["scores"], expected_changed, atol
# Print the observed values to compare against the expected result.
print("Changed end-to-end request verified")
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Send a second raw input [0, 255, 128] through the full request, and...
# Compute `changed_raw` from `np.array([[0., 255., 128.]], np.float32)`
changed_raw = np.array([[0., 255., 128.]], np.float32)
# Read or serialize artifact data on disk (`changed_response`).
changed_response = json.loads(request(json.dumps({"features": changed_raw.tolist()})))
# Convert `expected_changed` to a host NumPy array for inspection or verification.
expected_changed = (changed_raw / 255.) @ np.asarray(weights) + np.asarray(bias)
# Compute `np.testing.assert_allclose(changed_response["scores"], expected_changed, atol` as `1e-6)`.
np.testing.assert_allclose(changed_response["scores"], expected_changed, atol=1e-6)
# Print the observed values to compare against the expected result.
print("Changed end-to-end request verified")
```

</details>

## Make incomplete deployment evidence explicit

**Transfer**

Create a device report template with required hardware and measurement fields. Reject it as ready while those fields are missing.

<details><summary>Hint</summary>

List the hardware, runtime, delegate and measurement fields before testing completeness. A missing value is different from a measured zero or an explicitly unsupported capability.

</details>

### How to write: Make incomplete deployment evidence explicit — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `explicit(...)` — Call `explicit` with your updated parameters or inputs from this lesson's workspace.
- `device_report.items(...)` — Call `device_report.items` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Compute `missing` from `[key for key, value in device_report.items() if valu...`
2. Assert invariant `missing and "device" in missing` holds
3. Print the observed values to compare against the expected result.

**Starter code scaffold (fill in the TODOs):**

```python
# Make incomplete deployment evidence explicit (Transfer): Empty evidence stays empty.
device_report = ...  # TODO: compute device_report
                 "quality_metric": None, "p95_ms": None, "peak_memory_bytes": None,
                 "cold_start_ms": None, "fallback_operators": None}
# Compute `missing` from `[key for key, value in device_report.items() if valu...`
missing = ...  # TODO: compute missing
# Assert invariant `missing and "device" in missing` holds
assert missing  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print("Not device-validated; missing:", ", ".join(missing))
```

<details><summary>Reference solution and reasoning</summary>

```python
# Make incomplete deployment evidence explicit (Transfer): Empty evidence stays empty.
device_report = {"device": None, "runtime_version": None, "delegate": None,
                 "quality_metric": None, "p95_ms": None, "peak_memory_bytes": None,
                 "cold_start_ms": None, "fallback_operators": None}
# Compute `missing` from `[key for key, value in device_report.items() if valu...`
missing = [key for key, value in device_report.items() if value is None]
# Assert invariant `missing and "device" in missing` holds
assert missing and "device" in missing
# Print the observed values to compare against the expected result.
print("Not device-validated; missing:", ", ".join(missing))
```

Empty evidence stays empty. Desktop conversion and simulation cannot populate measurements for hardware that was never used.

</details>

## Decide whether a model speedup meets the request budget

**Transfer / diagnosis**

Use the illustrative serial times above. A $19$-millisecond deadline is declared before optimization. Does halving model time meet it? What is the best possible total if model time tends to zero?

<details><summary>Hint</summary>

Keep the fixed work unchanged. Compare the total request with the deadline, not the model-only duration.

</details>

### How to write: Decide whether a model speedup meets the request budget — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Compute `before_ms` from `fixed_ms+model_ms`
2. Compute `after_ms` from `fixed_ms+model_ms/2`
3. Execute `np.testing.assert_allclose([before_ms,after_ms,before_ms/aft`
4. Assert invariant `after_ms>deadline_ms and fixed_ms==16.` holds
5. Print the observed values to compare against the expected result.

**Starter code scaffold (fill in the TODOs):**

```python
# Decide whether a model speedup meets the request budget (Transfer / diagnosis): Halving model time leaves a 20-millisecond request, so it...
fixed_ms = ...  # TODO: compute fixed_ms
model_ms = ...  # TODO: compute model_ms
deadline_ms = ...  # TODO: compute deadline_ms
# Compute `before_ms` from `fixed_ms+model_ms`
before_ms = ...  # TODO: compute before_ms
# Compute `after_ms` from `fixed_ms+model_ms/2`
after_ms = ...  # TODO: compute after_ms
# Execute `np.testing.assert_allclose([before_ms,after_ms,before_ms/aft`
np.testing.assert_allclose([before_ms,after_ms,before_ms/after_ms],[24.,20.,1.2])
# Assert invariant `after_ms>deadline_ms and fixed_ms==16.` holds
assert after_ms  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print('Illustrative request before/after/lower-bound (ms):',before_ms,after_ms,fixed_ms)
```

<details><summary>Reference solution and reasoning</summary>

```python
# Decide whether a model speedup meets the request budget (Transfer / diagnosis): Halving model time leaves a 20-millisecond request, so it...
fixed_ms=12.+4.
model_ms=8.
deadline_ms=19.
# Compute `before_ms` from `fixed_ms+model_ms`
before_ms=fixed_ms+model_ms
# Compute `after_ms` from `fixed_ms+model_ms/2`
after_ms=fixed_ms+model_ms/2
# Execute `np.testing.assert_allclose([before_ms,after_ms,before_ms/aft`
np.testing.assert_allclose([before_ms,after_ms,before_ms/after_ms],[24.,20.,1.2])
# Assert invariant `after_ms>deadline_ms and fixed_ms==16.` holds
assert after_ms>deadline_ms and fixed_ms==16.
# Print the observed values to compare against the expected result.
print('Illustrative request before/after/lower-bound (ms):',before_ms,after_ms,fixed_ms)
```

Halving model time leaves a $20$-millisecond request, so it misses the stated deadline. The serial fixed work gives a $16$-millisecond lower bound even with zero model time. Replace these illustrative values with measurements before making a device decision.

</details>

## Check your understanding

A LiteRT file converts successfully. What must happen before claiming NPU latency?

1. Run it on the named device, verify execution placement and measure the stated request boundary.
2. Divide CPU latency by the weight compression ratio.
3. Assume every operator uses the NPU.

<details><summary>Answer and explanation</summary>

Run it on the named device, verify execution placement and measure the stated request boundary.

Conversion proves a converter accepted the graph. Device execution, fallback behavior and end-to-end latency require direct measurement.

</details>

## Diagnose the result

If INT8 quality drops, compare float conversion first, then verify calibration preprocessing, ranges and quantization metadata. If device latency disappoints, inspect fallback partitions, copies, request preprocessing and thermal conditions. If startup dominates, report it separately from warm inference and test the actual loading strategy.

## Keep your evidence

Keep the CPU request report, preprocessing equivalence checks, chosen runtime/operator/precision matrix and an unfilled target-device report until actual device measurements exist. For the optional converter, retain the artifact, quantization metadata and held-out comparison.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [LiteRT integer conversion tutorial](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant)
- [LiteRT float16 conversion](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_float16_quant)
- [ONNX Runtime quantization and providers](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ExecuTorch edge runtime](https://docs.pytorch.org/executorch/stable/index.html)
- [Core ML Tools overview](https://apple.github.io/coremltools/docs-guides/source/overview-coremltools.html)
- [LiteRT for Microcontrollers](https://developers.google.com/edge/litert/microcontrollers/overview)
- [TensorFlow installation and supported platforms](https://www.tensorflow.org/install/pip)

