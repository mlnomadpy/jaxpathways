# Cross-modal: connect images and text

Runnable CPU harness: cross-modal-harness. Open project.html?id=cross-modal-harness or projects/cross-modal-harness/README.md in the course workspace.

Build a retrieval system that checks alignment between modalities before optimizing similarity search.

A connected CPU image–text retrieval harness is runnable: both encoders train, complete state restores in a fresh process, calibrated precision policies export and reload, and a measured release gate supports rollback. The synthetic four-concept fixture does not establish natural-language or real-image retrieval quality.

## Before you start

Shared foundations, neural networks, recovery and deployment contracts. The connected project teaches its own pairing, multi-positive alignment and retrieval mechanism; attention-based and generative multimodal systems are extensions.

## Your target artifact

Train and deploy two encoders that retrieve matching synthetic images and literal captions; explain the observed failure when image geometry shifts.

## Figures to explain

Read the actual training curve, paired image and clean similarity matrix, then inspect shifted-image confusion counts. Clean retrieval is 20/20 while one-pixel shifted retrieval is 5/20; explain the fixed pixel coordinates behind that failure.

## 1. Define the data contract

Begin with what one example means. Keep stable IDs and split before fitting normalization, tokenization or augmentation. A small synthetic fixture makes bugs visible; a separate documented dataset is needed to study generalization.

Keep pair and group IDs through every shuffle. Split related images and captions together; record the policy for multiple valid captions or near-duplicate images.

**Keep:** A split manifest, provenance/license note, preprocessing version and an example inspection.

**Check:** Assert that train and held-out IDs are disjoint. Change one preprocessing setting and make the contract check reject the old artifact.

**Available preparation lessons:** arrays-01, arrays-02, recovery-01

**Runnable project:** cross-modal-harness · stages 1

## 2. Build a transparent baseline

First build a model small enough to inspect. Predict intermediate shapes and check a forward pass using a separate calculation. Your baseline provides a reference for every later optimization.

Start with two small encoders and a declared embedding normalization rule. Trace both batch axes in the similarity matrix and compare one dot product independently.

**Keep:** A shape diagram, baseline configuration and an independently checked forward pass.

**Check:** Test a singleton batch and a changed batch size. Reject malformed targets before broadcasting can silently change the objective.

**Available preparation lessons:** networks-01, networks-02

**Runnable project:** cross-modal-harness · stages 2

## 3. Make each update explainable

Record whether a reported loss belongs to the parameters before or after an update. Keep validation outside the training transition and save the configuration with the random seed. A falling loss shows fitting; it does not establish usefulness on new examples.

Use a stable contrastive objective with an explicit temperature and positive-pair mask. Test aligned versus deliberately permuted pairs; duplicate positives must not silently become negatives. The training-methods project supplies separate objective labs for these methods; follow each phase prerequisite before attempting them. Integrating every objective into this modality harness remains an extension.

**Keep:** Training and validation curves, a parameter-gradient check and a record of one failed run.

**Check:** Overfit a tiny batch, compare one gradient with an independent finite difference and replay the same seed.

**Available preparation lessons:** optimization-02, optimization-04, networks-03, networks-05, recovery-06, pretraining-03, posttraining-02

**Runnable project:** cross-modal-harness · stages 3

## 4. Prove that training can resume

Weights alone cannot reconstruct an interrupted experiment. Save optimizer memory, random state, completed step and input iterator position together with the data and package contract. Compare the next several updates after restore.

Save both encoders, optimizer state and the paired sampler position together. Reject a checkpoint that restores only one side of the learned embedding space.

**Keep:** A checkpoint manifest and uninterrupted-versus-restored sample IDs, losses and state.

**Check:** Interrupt at a known boundary. Restore into a fresh process and compare several subsequent batches; reject a changed dataset or optimizer configuration.

**Available preparation lessons:** recovery-02, recovery-03, recovery-04

**Runnable project:** cross-modal-harness · stages 3

## 5. Measure quality and inspect errors

Freeze evaluation inputs before tuning. Aggregate sums and counts instead of averaging unequal batch means. Show representative failures alongside a metric so a reader can see what the model gets wrong.

Evaluate image-to-text and text-to-image retrieval on a frozen candidate set. Report recall at declared ranks, candidate count and multiple-positive handling; changing the candidate pool changes the task.

**Keep:** A baseline comparison, counts, error examples and a written interpretation of each figure.

**Check:** Evaluate one set as a whole and in uneven batches. Confirm that evaluation changes neither parameters nor optimizer state.

**Available preparation lessons:** networks-03, networks-04, optimization-10

**Runnable project:** cross-modal-harness · stages 3

## 6. Choose weights, activations and accumulation separately

Start from a float32 reference. Weight storage, activation computation, accumulation and optimizer memory are separate decisions. Integer quantization also needs scales, zero-point policy, calibration data and runtime support. A smaller file is not evidence of faster inference.

Compare encoder precision and stored embedding precision separately. Track rank changes and recall, not only elementwise embedding error.

**Keep:** A table of precision policies, artifact bytes, measured quality change and target-runtime latency.

**Check:** Calibrate on training or calibration data, keep final evaluation separate, and report unsupported kernels explicitly. Compare the same examples across policies.

**Available preparation lessons:** transformers-03, deployment-05

**Runnable project:** cross-modal-harness · stages 4

## 7. Verify the deployment boundary

Package preprocessing with the trained computation. Check exported outputs on fixed normal and boundary inputs before serving them. Framework conversion and copying weights between model definitions need explicit shape, layout and dtype contracts.

Version the two encoders, preprocessing assets and search index together. Refuse an index built by an incompatible encoder version. Use the weight-conversion project to audit an actual PyTorch/Flax dense–normalization–activation architecture. Extend its layerwise checks for this modality rather than assuming that its MLP mapping covers all operators.

**Keep:** A versioned artifact with signatures, preprocessing assets and source-versus-runtime parity results.

**Check:** Load in a fresh process. Test singleton and supported changed shapes, malformed input and a deliberately mismatched preprocessing version.

**Available preparation lessons:** deployment-01, deployment-03, deployment-06, deployment-07, deployment-08

**Runnable project:** cross-modal-harness · stages 4

## 8. Measure inference and rehearse failure

Define the workload before timing it: input sizes, batching, concurrency and device. Warm up, synchronize device work and distinguish startup from steady execution. Test a failed load and a rollback to the last verified artifact.

Measure query encoding and search separately, then end-to-end response time. Test missing or corrupted inputs and rebuild/rollback of an index. Generative multimodal models are a later extension.

**Keep:** A workload report with latency units, throughput denominator, memory observations and a recovery/rollback note.

**Check:** Repeat the benchmark with a stated number of runs. On edge hardware, include preprocessing and transfer time and record the actual device; desktop proxies remain labeled proxies.

**Available preparation lessons:** performance-01, performance-02, operations-06, operations-08

**Runnable project:** cross-modal-harness · stages 5

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

PyTorch, Keras, TensorFlow and JAX interoperability is tested per model and operation set. A successful tiny example does not establish that every model can round-trip between them.

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
