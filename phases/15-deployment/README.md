# Phase 15: Deployment, interoperability & edge AI

Specializations.

Adapt a trained model, verify framework and export contracts, compare precision policies and measure local inference before planning server or edge capacity. Package a model service in a reproducible container and verify readiness, artifact identity and request contracts.

Preserve checkpoint, data and calibration provenance. Compare held-out inputs before and after conversion; separate local request timing from queue simulation and target-device evidence.

**Prerequisites:** 06: Data & checkpoint recovery; 08: Performance diagnosis.

**Hardware:** CPU for core labs; optional framework environments and target devices for conversion and edge validation.

## Study guide: Does the exported service preserve the model’s complete contract?

A deployed model begins with a raw request and ends with a response, not with a weight file alone. Preprocessing, parameter layout, precision, serialization and runtime execution must preserve the meaning of that request across each boundary.

Carry the same examples through adaptation, framework conversion, export and inference so the first changed result is easy to locate. Then ask what storage savings, queue calculations or edge budgets actually imply for this runtime. Keep numerical fidelity, held-out task behavior and measured service work as separate parts of a deployment decision.

### Check your starting point

Weights match after conversion, but predictions differ. What besides weights must be compared?

<details><summary>Compare your reasoning</summary>

Compare input normalization, feature/channel order, operation semantics, non-trainable state, output labels, dtype and shape. Start with a hand-computed asymmetric example to expose layout mistakes.

</details>

Review: [Move models between JAX, Keras, TensorFlow and PyTorch](01-keras-and-pytorch-bridges-to-explicit-jax/docs/en.md).

### Build in stages

1. **Choose the model and preserve its meaning.** Adapt the model under a fixed held-out protocol before freezing its release. Compare framework layouts, preprocessing and state with independent predictions. Convert the actual PyTorch model to Flax and locate an injected operation mismatch numerically. Keep a candidate decision table, feature/parameter mapping table, and the first divergent operation from a deliberately broken conversion.

   Lessons: [Adapt a pretrained model and choose a post-training objective](02-adapt-a-pretrained-model-and-choose-a-post-training-objective/docs/en.md) · [Move models between JAX, Keras, TensorFlow and PyTorch](01-keras-and-pytorch-bridges-to-explicit-jax/docs/en.md) · [Convert PyTorch weights to Flax and locate numerical errors](08-pytorch-to-flax-numerical-parity/docs/en.md).

2. **Freeze the artifact and numeric policy.** Export and load in a fresh process, reject malformed inputs, then track weight, activation, accumulator, bias and output formats. Separate clipping from rounding and retain held-out conversion checks. Include a fresh-interpreter export receipt, raw-request rejection cases, clipping counts and actual versus ideal packed byte counts.

   Lessons: [Export a computation and verify its serving contract](03-export-and-serve-a-trained-computation/docs/en.md) · [Choose weight, activation and accumulation precision](05-weight-activation-and-accumulation-precision/docs/en.md).

3. **Deliver to a stated runtime.** Package and qualify the selected service, relate queue assumptions to measured service work, then examine the additional conversion and device budgets for an edge target. Leave actual device fields unmeasured until tested. Keep separate artifact, process, image and device receipts. Explain an interpolated percentile and a burst deadline failure before choosing a capacity intervention.

   Lessons: [Containerize a model service and verify its boundary](07-containerize-a-model-service/docs/en.md) · [Inference capacity, batching, and autoscaling](04-inference-capacity-batching-and-autoscaling/docs/en.md) · [Deploy at the edge: conversion, budgets and device checks](06-edge-ai-conversion-and-device-validation/docs/en.md).

### Try a changed condition

Frozen calibration maps [-1, 1] to signed integers. New inputs include 3. Explain why a small weight-file size says little about prediction quality or speed.

<details><summary>Compare an approach</summary>

The out-of-range activation must follow the declared clipping or fallback policy. Track saturation by channel and compare the same preprocessed input with the float reference. Storage savings do not show native integer execution. Measure actual runtime placement and latency after numerical and quality gates pass.

</details>

**Symptom:** A smaller model file produces no lower request latency.

**Check next:** Inspect load-time expansion, fallback operators, transfers and preprocessing on the named runtime. File size is not a kernel or latency measurement.

### Decide what is ready

Use deployment-audit and engineering-release to assemble one release dossier: candidate selection and retention criteria; a named feature and parameter mapping; source/export and layerwise parity; full precision policy and calibration identity; raw-request validation; artifact and process receipts; measured request samples; and a queue/deadline analysis. Mark container and target-device evidence unmeasured unless you actually ran those environments. Another learner should be able to reproduce each claimed check from the recorded inputs and command. For framework transfer, run the optional bounded saved-decoder parity lab: map QKV/head layouts, masks, positions and tied embeddings, locate both deliberate attention failures, and compare a nonuniform VJP after fresh-process reload.

### Further work

Actual edge-device delegates, thermal/energy budgets, packed low-bit kernels and production serving loads remain unqualified. The revised sequence moves adaptation before export and containerization before service capacity planning.

## Lesson sequence

### 15.01 Adapt a pretrained model and choose a post-training objective

[Read the lesson](02-adapt-a-pretrained-model-and-choose-a-post-training-objective/docs/en.md) · [Run the code](02-adapt-a-pretrained-model-and-choose-a-post-training-objective/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Pretrain and reload an artifact with recorded data and checkpoint hashes Compare supervised adaptation and teacher distillation from the same starting weights

**Evidence:** Source/checkpoint hashes, independent loss and gradient, disjoint evaluation results and teacher-failure diagnosis. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** Why should the two adaptation branches be compared using a common held-out criterion rather than their own training losses?

### 15.02 Move models between JAX, Keras, TensorFlow and PyTorch

[Read the lesson](01-keras-and-pytorch-bridges-to-explicit-jax/docs/en.md) · [Run the code](01-keras-and-pytorch-bridges-to-explicit-jax/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Verify parameter layouts and inference semantics before switching frameworks.

**Evidence:** Keep the layout map, two changed-input comparisons, runtime versions and optional backend-lab output. Identify which model state and preprocessing are not covered by a dense-layer match.

**Checkpoint:** The Dense predictions match after transposing weights. What has been established?

### 15.03 Convert PyTorch weights to Flax and locate numerical errors

[Read the lesson](08-pytorch-to-flax-numerical-parity/docs/en.md) · [Run the code](08-pytorch-to-flax-numerical-parity/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Map actual PyTorch parameters to Flax NNX and localize the first layer or gradient mismatch, then transfer to the optional small saved decoder.

**Evidence:** Keep MLP normalization/activation parity, first-divergence evidence, the optional QKV/head layout and mask canaries, nonuniform VJP, processor/position identity and fresh-process reload results.

**Checkpoint:** The first linear layer agrees, but LayerNorm is the first mismatch. What should you inspect next?

### 15.04 Export a computation and verify its serving contract

[Read the lesson](03-export-and-serve-a-trained-computation/docs/en.md) · [Run the code](03-export-and-serve-a-trained-computation/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Round-trip an exported inference function and reject incompatible requests.

**Evidence:** Keep the export/reload code, version/platform manifest, known-output and changed-input checks, rejected-shape result, and the planned serving request boundary.

**Checkpoint:** What does a successful local export round trip prove?

### 15.05 Choose weight, activation and accumulation precision

[Read the lesson](05-weight-activation-and-accumulation-precision/docs/en.md) · [Run the code](05-weight-activation-and-accumulation-precision/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Compare floating-point policies, weight-only quantization and calibrated integer activations.

**Evidence:** Keep a precision table listing weight, activation, accumulator and output dtypes; max error and clipping counts on changed inputs; calibration provenance; and storage estimates that include scales and packing.

**Checkpoint:** What must a deployment policy specify beyond “INT8 weights”?

### 15.06 Containerize a model service and verify its boundary

[Read the lesson](07-containerize-a-model-service/docs/en.md) · [Run the code](07-containerize-a-model-service/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Package an inference service and verify model identity, request validation and readiness across process and container boundaries.

**Evidence:** Keep the artifact digest, valid and rejected requests, fresh-process output and, when Docker is available, the actual image identity, runtime user and container qualification receipt.

**Checkpoint:** The image built successfully. What should happen before serving traffic?

### 15.07 Inference capacity, batching, and autoscaling

[Read the lesson](04-inference-capacity-batching-and-autoscaling/docs/en.md) · [Run the code](04-inference-capacity-batching-and-autoscaling/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Measure completed inference after warming each batch shape Report batch latency and example throughput with correct units

**Evidence:** Warm timing samples, latency/throughput units, independent queue timeline and labeled simulation/replica assumptions. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** A batch of eight finishes in four milliseconds. Which statement follows for a fully occupied serial worker?

### 15.08 Deploy at the edge: conversion, budgets and device checks

[Read the lesson](06-edge-ai-conversion-and-device-validation/docs/en.md) · [Run the code](06-edge-ai-conversion-and-device-validation/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Define an edge runtime contract and measure an end-to-end batch-one request.

**Evidence:** Keep the CPU request report, preprocessing equivalence checks, chosen runtime/operator/precision matrix and an unfilled target-device report until actual device measurements exist. For the optional converter, retain the artifact, quantization metadata and held-out comparison.

**Checkpoint:** A LiteRT file converts successfully. What must happen before claiming NPU latency?

## Phase project

Verify an inference artifact and capacity hypothesis.

**Demonstrate:** Verify an exported prediction contract, report weight/activation precision error, and distinguish CPU request measurements from target-device evidence.

Project status: implemented staged practice · [Open source](../../projects/deployment-audit/README.md). Copy `projects/deployment-audit/starter/model.py` to `projects/deployment-audit/my_model.py` and write your code in `projects/deployment-audit/my_model.py`. Run `python3 projects/deployment-audit/tests/check.py --implementation projects/deployment-audit/my_model.py --stage 1` from the top-level folder to verify all 3 stages.

Additional project: [Ship a tracked and containerized model release](../../projects/engineering-release/README.md).
Additional project: [Convert PyTorch weights and audit Flax architecture parity](../../projects/weight-conversion/README.md).

[Primary documentation](https://docs.jax.dev/en/latest/).
