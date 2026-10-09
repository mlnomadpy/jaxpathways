# Audio: from waveforms to an edge sound classifier

Runnable CPU harness: audio-harness. Open project.html?id=audio-harness or projects/audio-harness/README.md in the course workspace.

Build a sound-event system that preserves timing, handles silence and measures the full signal-processing path.

A complete CPU waveform-to-inference harness is runnable: trained spectral classifier, full fresh-process recovery, fixed evaluation, calibrated precision and complete exported artifacts. PCM WAV ingestion is available; natural-recording quality and target-edge performance remain unmeasured.

## Before you start

Shared foundations, neural-network training, explicit recovery and deployment contracts. The project supplies a connected waveform/STFT tutorial and lifecycle exercise.

## Your target artifact

Classify short sound events from fixed windows and explain what happens with silence, noise and a changed recording source.

## Figures to explain

Inspect the actual held-out error as waveform and spectrogram with a shared seconds axis. Color is log(1 + window-normalized power), not probability. Read confusion by true-class rows and predicted-class columns, and explain the high-tone/background error.

## 1. Define the data contract

Begin with what one example means. Keep stable IDs and split before fitting normalization, tokenization or augmentation. A small synthetic fixture makes bugs visible; a separate documented dataset is needed to study generalization.

Record sample rate, channel count, clip duration and amplitude convention. Split by speaker, recording session or source as appropriate before making overlapping windows.

**Keep:** A split manifest, provenance/license note, preprocessing version and an example inspection.

**Check:** Assert that train and held-out IDs are disjoint. Change one preprocessing setting and make the contract check reject the old artifact.

**Available preparation lessons:** arrays-01, arrays-02, recovery-01

**Runnable project:** audio-harness · stages 1

## 2. Build a transparent baseline

First build a model small enough to inspect. Predict intermediate shapes and check a forward pass using a separate calculation. Your baseline provides a reference for every later optimization.

First compare an energy or simple feature baseline with a small spectrogram classifier. Specify window size, hop, frequency bins and log floor. Test silence without producing nonfinite values.

**Keep:** A shape diagram, baseline configuration and an independently checked forward pass.

**Check:** Test a singleton batch and a changed batch size. Reject malformed targets before broadcasting can silently change the objective.

**Available preparation lessons:** networks-01, networks-02

**Runnable project:** audio-harness · stages 1, 2

## 3. Make each update explainable

Record whether a reported loss belongs to the parameters before or after an update. Keep validation outside the training transition and save the configuration with the random seed. A falling loss shows fitting; it is verified separately from usefulness on new examples.

Keep window labels and valid-frame masks aligned. Apply noise or gain augmentation only in training, and verify that padding does not become a class cue.

**Keep:** Training and validation curves, a parameter-gradient check and a record of one failed run.

**Check:** Overfit a tiny batch, compare one gradient with an independent finite difference and replay the same seed.

**Available preparation lessons:** optimization-02, optimization-04, networks-03, networks-05, recovery-06

**Runnable project:** audio-harness · stages 2

## 4. Prove that training can resume

Weights alone cannot reconstruct an interrupted experiment. Save optimizer memory, random state, completed step and input iterator position together with the data and package contract. Compare the next several updates after restore.

Save clip/window position and augmentation state. After restore, compare clip IDs, crop offsets and feature arrays before comparing losses.

**Keep:** A checkpoint manifest and uninterrupted-versus-restored sample IDs, losses and state.

**Check:** Interrupt at a known boundary. Restore into a fresh process and compare several subsequent batches; reject a changed dataset or optimizer configuration.

**Available preparation lessons:** recovery-02, recovery-03, recovery-04

**Runnable project:** audio-harness · stages 2

## 5. Measure quality and inspect errors

Freeze evaluation inputs before tuning. Aggregate sums and counts instead of averaging unequal batch means. Show representative failures alongside a metric so a reader can see what the model gets wrong.

Measure per-class recall and false positives on a fixed silence/background set. Aggregate at the intended unit—window or clip—and state how window decisions become a clip decision.

**Keep:** A baseline comparison, counts, error examples and a written interpretation of each figure.

**Check:** Evaluate one set as a whole and in uneven batches. Confirm that evaluation changes neither parameters nor optimizer state.

**Available preparation lessons:** networks-03, networks-04, optimization-10

**Runnable project:** audio-harness · stages 2

## 6. Choose weights, activations and accumulation separately

Start from a float32 reference. Weight storage, activation computation, accumulation and optimizer memory are separate decisions. Integer quantization also needs scales, zero-point policy, calibration data and runtime support. A smaller file is not evidence of faster inference.

Calibrate across quiet and loud representative signals. Compare feature-extraction precision separately from model precision; inspect clipping and rare-event recall.

**Keep:** A table of precision policies, artifact bytes, measured quality change and target-runtime latency.

**Check:** Calibrate on training or calibration data, keep final evaluation separate, and report unsupported kernels explicitly. Compare the same examples across policies.

**Available preparation lessons:** transformers-03, deployment-05

**Runnable project:** audio-harness · stages 3

## 7. Verify the deployment boundary

Package preprocessing with the trained computation. Check exported outputs on fixed normal and boundary inputs before serving them. Framework conversion and copying weights between model definitions need explicit shape, layout and dtype contracts.

Package resampling and spectrogram settings with weights. Compare features and logits stage by stage to locate a mismatch; an exported model alone does not define microphone input behavior. Use the weight-conversion project to audit an actual PyTorch/Flax dense–normalization–activation architecture. Extend its layerwise checks for this modality rather than assuming that its MLP mapping covers all operators.

**Keep:** A versioned artifact with signatures, preprocessing assets and source-versus-runtime parity results.

**Check:** Load in a fresh process. Test singleton and supported changed shapes, malformed input and a deliberately mismatched preprocessing version.

**Available preparation lessons:** deployment-01, deployment-03, deployment-06, deployment-07, deployment-08

**Runnable project:** audio-harness · stages 3

## 8. Measure inference and rehearse failure

Define the workload before timing it: input sizes, batching, concurrency and device. Warm up, synchronize device work and distinguish startup from steady execution. Test a failed load and a rollback to the last verified artifact.

Measure capture-window delay, preprocessing and inference. Use a stated hop size and real-time factor definition; verify dropped-window and missing-device behavior on actual hardware.

**Keep:** A workload report with latency units, throughput denominator, memory observations and a recovery/rollback note.

**Check:** Repeat the benchmark with a stated number of runs. On edge hardware, include preprocessing and transfer time and record the actual device; desktop proxies remain labeled proxies.

**Available preparation lessons:** performance-01, performance-02, operations-06, operations-08

**Runnable project:** audio-harness · stages 4

## Write a precision policy, not just a dtype

Record weights, activations, accumulation, optimizer state and any inference cache separately. Keep a float32 reference and compare the same held-out examples before choosing a lower-precision artifact.

- Float32: establish the reference output and quality first.
- Bfloat16 or float16: test numerical range and reduction behavior on the actual backend; retain sensitive reductions or optimizer state in higher precision where needed.
- Int8: document calibration examples, scale granularity, clipping and activation policy. Weight-only and weight-and-activation quantization are different experiments.
- Int4 or float8 extensions: require a supported runtime and hardware, declared packing/scaling rules and new quality measurements. No general compatibility or speedup is assumed.

For every supported policy keep artifact size, task quality, output error, latency and the actual device. Record unsupported policies as unsupported.

## Cross-framework work begins with parity

Use the existing Keras/JAX and TensorFlow conversion labs as preparation. A model definition, a saved checkpoint and an inference artifact have different contracts. Moving weights does not automatically move optimizer state, preprocessing or training behavior.

- Pin both environments and choose a conversion path supported by those versions.
- Map parameter names, kernel layouts, input shapes and dtypes explicitly.
- Compare intermediate tensors and final outputs on fixed inputs, including boundary cases.
- Compare task quality after export and again after quantization; measure the exported runtime itself.

PyTorch, Keras, TensorFlow and JAX interoperability is tested per model and operation set. A successful tiny example is verified separately from that every model can round-trip between them.

## Grow the harness after the small version works

Share run configuration, metrics, checkpoint metadata and artifact packaging. Keep modality-specific preprocessing, objectives and decoding behind explicit interfaces. Extend one working image system first, then text, audio and paired retrieval.



MaxText is an advanced reference for language-model training and supported multimodal workflows. Study its configuration, input pipeline, checkpointing, sharding and precision choices. Pin a release or commit in a separate compatible environment before adapting code; the course CPU environment does not validate a MaxText installation.

## Carry engineering evidence across the lifecycle

Use the engineering-release project after the modality harness. Track the same data, preprocessing, model and evaluation identity from experiment through service.

- MLflow: compare runs with the same metric definition and preserve data/code lineage.
- Containers: record base/image/model digests, non-root runtime, mounted artifacts and readiness.
- MLOps: validate data contracts, automate repeatable evaluation, inspect slices and monitor delayed quality.
- LLMOps: version prompts, retrieval and tools; retain calibrated evaluation cases and redacted traces.
- ModelOps: assign ownership, bind approval to a specific bundle and rehearse rollback and retirement.

MLflow tracking and Docker are concrete tools; MLOps, LLMOps and ModelOps describe responsibilities around them. Read the linked operations lessons before treating a registry alias or a passing local check as a deployed system.

## References

- [MaxText: advanced language-model training reference](https://github.com/AI-Hypercomputer/maxtext)
- [MaxText multimodal tutorial](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/multimodal.html)
- [MaxDiffusion: image-generation extension reference](https://github.com/AI-Hypercomputer/maxdiffusion)
