# Deploy at the edge: conversion, budgets and device checks

Phase 15: Deployment, interoperability & edge AI · about 75 minutes · CPU

## What you will be able to do

- Trace source model through conversion, runtime and device execution.
- Measure a complete batch-one request separately from first-call overhead.
- Define calibration, held-out quality and device acceptance checks.
- Identify fallback operators and memory costs before claiming edge readiness.

## The problem

A model works on your computer, and now you want it to run near the user—perhaps on a phone or an embedded board. Let’s trace the whole request, from raw input to usable output. We’ll check conversion, measure a batch-one CPU request, and write down what still needs to be measured on the real device.

## The idea

Edge AI runs inference near the data source. A mobile app with CPU/GPU/NPU delegates, a Linux board, and a microcontroller have different runtimes and operator budgets. Choose the device and execution path first, then prove the exported computation fits. This CPU lesson teaches the measurement boundary; it does not emulate an NPU, battery draw or microcontroller memory.

## Choose a deployment lane

For a phone or Linux edge device, LiteRT is one runtime option; ONNX Runtime offers execution providers for supported targets. Apple deployments may use Core ML after a supported conversion; PyTorch models may use ExecuTorch. Microcontrollers require a runtime and operator set designed for constrained memory, such as LiteRT for Microcontrollers. These are alternatives, not interchangeable file formats. Write down OS, device model, runtime version, operators, input shapes, precision policy and accelerator path before converting.

## Follow one concrete TensorFlow-to-LiteRT route

The optional labs/tensorflow_litert.py companion builds the known dense model in TensorFlow, converts float32 and fully integer variants, inspects input/output scale and zero point, quantizes requests and dequantizes outputs, then checks held-out predictions. Representative calibration inputs are preprocessed floats in the exact source-model input domain. Setting integer-only supported ops and int8 input/output requests a strict integer graph; unsupported conversion should fail rather than silently satisfy the contract with float fallback. The lab uses TensorFlow’s interpreter interface as a desktop compatibility check; it is not a device delegate test. Desktop conversion was verified with Python 3.12.13, TensorFlow 2.21.0 and Keras 3.15.1 on macOS arm64. The optional requirements file pins these tested top-level dependencies; other platforms still need a compatible installation.

**Optional TensorFlow environment — macOS/Linux, from the workspace root**

```sh
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

## Run the example

```python
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
weights = jnp.array([[1., -2.], [.5, 1.], [-1., .25]], jnp.float32)
bias = jnp.array([.1, -.2], jnp.float32)
@jax.jit
def infer(features):
    return features @ weights + bias
payload = json.dumps({"features": [[255., 128., 0.]]})
def request(payload):
    decoded = np.asarray(json.loads(payload)["features"], dtype=np.float32)
    if decoded.shape != (1, 3) or not np.isfinite(decoded).all():
        raise ValueError("expected finite [1, 3] sensor features")
    features = decoded / 255.
    output = np.asarray(infer(jnp.asarray(features)).block_until_ready())
    return json.dumps({"scores": output.tolist()})
start = time.perf_counter()
first_response = request(payload)
first_ms = (time.perf_counter() - start) * 1000
samples = []
for _ in range(30):
    start = time.perf_counter()
    response = request(payload)
    samples.append((time.perf_counter() - start) * 1000)
expected = (np.array([[255., 128., 0.]], np.float32) / 255.) @ np.asarray(weights) + np.asarray(bias)
np.testing.assert_allclose(json.loads(response)["scores"], expected, atol=1e-6)
report = {"runtime": "JAX CPU instructional proxy", "jax": jax.__version__,
          "first_request_ms": first_ms, "warm_samples_ms": samples,
          "p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95)),
          "boundary": "JSON decode + normalize + transfer + infer + wait + encode",
          "edge_device_validated": False}
assert len(samples) == 30 and all(t >= 0 for t in samples)
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
visual_data = {'kind': 'line', 'x': list(range(1, len(samples) + 1)), 'xlabel': 'warm request number', 'ylabel': 'end-to-end milliseconds', 'series': [{'label': 'local CPU request', 'y': samples}, {'label': 'recorded p95', 'y': [report['p95_ms']] * len(samples)}]}
```

## Recorded reference execution

CPU run: 2026-10-06T01:26:25.160375+00:00. JAX 0.9.2.

```text
{
  "runtime": "JAX CPU instructional proxy",
  "jax": "0.9.2",
  "first_request_ms": 10.835832916200161,
  "warm_samples_ms": [
    0.13049994595348835,
    0.056416960433125496,
    0.04233303479850292,
    0.03800005652010441,
    0.03429199568927288,
    0.03337510861456394,
    0.031084055081009865,
    0.0299159437417984,
    0.02883397974073887,
    0.028916867449879646,
    0.02929195761680603,
    0.028374837711453438,
    0.02683396451175213,
    0.02674991264939308,
    0.026250025257468224,
    0.026082852855324745,
    0.026708003133535385,
    0.026332912966609,
    0.02654106356203556,
    0.026167137548327446,
    0.026332912966609,
    0.026625115424394608,
    0.02687494270503521,
    0.027207890525460243,
    0.02629193477332592,
    0.02733292058110237,
    0.026209047064185143,
    0.02591707743704319,
    0.02574990503489971,
    0.02537504769861698
  ],
  "p50_ms": 0.02685445360839367,
  "p95_ms": 0.050079193897545296,
  "boundary": "JSON decode + normalize + transfer + infer + wait + encode",
  "edge_device_validated": false
}
Preprocessing mismatch max error 380.5019836425781
Changed end-to-end request verified
Not device-validated; missing: device, runtime_version, delegate, quality_metric, p95_ms, peak_memory_bytes, cold_start_ms, fallback_operators
PASS: deployment-06

```

## A normalization mismatch survives conversion

**Predict before running:** Will raw sensor values and normalized values produce the same scores?

```python
raw = np.array([[255., 128., 0.]], np.float32)
wrong = np.asarray(infer(raw))
right = np.asarray(infer(raw / 255.))
assert not np.allclose(wrong, right)
print("Preprocessing mismatch max error", float(np.max(np.abs(wrong-right))))

```

**Expected:** The mismatch is large despite identical weights.

Conversion checks must start at the raw input boundary when preprocessing is part of the product.

## Make it yours

Send a second raw input $[0, 255, 128]$ through the full request, and verify its independent normalized NumPy result.

<details><summary>Reference solution</summary>

```python
changed_raw = np.array([[0., 255., 128.]], np.float32)
changed_response = json.loads(request(json.dumps({"features": changed_raw.tolist()})))
expected_changed = (changed_raw / 255.) @ np.asarray(weights) + np.asarray(bias)
np.testing.assert_allclose(changed_response["scores"], expected_changed, atol=1e-6)
print("Changed end-to-end request verified")
```

</details>

## Make incomplete deployment evidence explicit

**Transfer**

Create a device report template with required hardware and measurement fields. Reject it as ready while those fields are missing.

<details><summary>Hint</summary>

List the hardware, runtime, delegate and measurement fields before testing completeness. A missing value is different from a measured zero or an explicitly unsupported capability.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
device_report = {"device": None, "runtime_version": None, "delegate": None,
                 "quality_metric": None, "p95_ms": None, "peak_memory_bytes": None,
                 "cold_start_ms": None, "fallback_operators": None}
missing = [key for key, value in device_report.items() if value is None]
assert missing and "device" in missing
print("Not device-validated; missing:", ", ".join(missing))

```

Empty evidence stays empty. Desktop conversion and simulation cannot populate measurements for hardware that was never used.

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

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [LiteRT integer conversion tutorial](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant)
- [LiteRT float16 conversion](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_float16_quant)
- [ONNX Runtime quantization and providers](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ExecuTorch edge runtime](https://docs.pytorch.org/executorch/stable/index.html)
- [Core ML Tools overview](https://apple.github.io/coremltools/docs-guides/source/overview-coremltools.html)
- [LiteRT for Microcontrollers](https://developers.google.com/edge/litert/microcontrollers/overview)
- [TensorFlow installation and supported platforms](https://www.tensorflow.org/install/pip)

