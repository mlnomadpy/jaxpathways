# Tooling for complete model projects

Status: planned curriculum additions, 2026-10-06. Existing JAX, MLflow, Docker and framework labs retain their recorded scope. W&B and the other extensions below have not been installed, connected to an account or executed by this planning change.

Choose tools for a specific experiment or release problem. The recommended core stays small: JAX/Flax/Optax, the existing input/checkpoint libraries, **one** tracking backend, a versioned evaluation runner, a container and a tested Hub package. Advanced alternatives are optional labs with a concrete comparison, not a list of mandatory services.

## Weights & Biases alongside MLflow

Teach W&B as a supported experiment-tracking option, with an equivalent exercise to the existing MLflow lab. The [W&B SDK](https://github.com/wandb/wandb) supports run configuration/metric logging and the [product documentation source](https://github.com/wandb/docs) covers artifacts, tables and run management. [W&B Sweeps](https://github.com/wandb/sweeps) supplies the hyperparameter-search component. Pin versions and verify the selected APIs during implementation; hosted documentation redirected unsuccessfully during this planning review, so the official repositories were used as primary references.

| Planned lab | Learner work | Evidence required |
| --- | --- | --- |
| Runs and meaningful axes | Record config, phase/parent, data/model IDs, valid-token/pair/audio-second counters, train and validation metrics | Match a local canonical metric record; show why microsteps, applied updates and consumed tokens differ |
| Artifact lineage | Track a data/processor manifest, selected checkpoint, evaluation and release bundle | Retrieve exact artifact versions and compare hashes; a mutable “best” alias is not the recorded identity |
| Media and evaluation tables | Inspect a bounded set of generation errors, retrieved pairs, image results or permitted audio examples | Case IDs, input/prediction/reference relationships and systematic failures; no cherry-picked gallery as the sole evidence |
| Offline and interrupted logging | Buffer the permitted experiment locally; test recovery and optional later synchronization | No lost/relabelled optimizer steps or duplicate public results; explicit account/sync action only when requested |
| Bounded sweeps | Compare rate, schedule, augmentation, objective weight or adapter rank on a pilot | Fixed development objective, equal resource accounting, trial/concurrency caps, seeds and retained failed trials |

For JAX, log summarized values outside compiled numerical functions and profile synchronization overhead. Aggregate counts correctly across workers; normally one designated process writes global metrics, with rank-tagged diagnostics when needed. Do not call a tracking SDK inside `jit` or serialize a huge parameter tree every step.

Keep a small proposed tracking interface—start run, log aggregated metrics, record artifact reference, close run—with JSONL/local files as the canonical fallback and separate MLflow/W&B adapters. It is not yet implemented. Test adapter consistency on the same **recorded** numerical run, not on two differently seeded trainings. Specify retry IDs and monotonic logging counters; a tracker retry must never reapply an optimizer update.

A tracker outage may allow training to continue under a tested local-buffer policy, but missing experiment artifacts block release promotion. Checkpoint durability remains the checkpoint system's responsibility. Choose MLflow or W&B as the primary tracker for a run; dual logging is an optional parity exercise, not a production requirement. Reuse the existing MLflow registry/release examples and show how equivalent immutable evidence is recorded with a different backend.

## Other tools worth teaching

| Responsibility | Tool and primary reference | Why it is in the course / boundary |
| --- | --- | --- |
| Reproducible input | [Grain](https://github.com/google/grain) | Existing JAX input/replay path; extend to real shards and sampling. A data reader does not resolve licensing or split leakage. |
| Data/pipeline versioning | [DVC](https://doc.dvc.org/user-guide) | Optional lab for reproducing preprocessing outputs and versioned large-file references. Keep hashes/manifests usable without DVC; it is not a full cluster supervisor. |
| Durable training state | [Orbax](https://github.com/google/orbax) | Existing save/restore path; extend sharding and failure drills. A tracker artifact upload does not substitute for accepted full-state recovery. |
| Experiment comparison | [MLflow](https://mlflow.org/docs/latest/ml/tracking/) or W&B | One default backend; track comparable runs and retrieve actual artifacts. Neither automatically establishes model quality. |
| Search alternatives | [Optuna](https://github.com/optuna/optuna) or W&B Sweeps | Optional comparison of bounded search/pruning on development data. Do not run both for the same introductory experiment. |
| Language benchmarks | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Causal likelihood/generation task adapter with pinned prompts and protocols. Validate the native or converted model path. |
| Embedding benchmarks | [MTEB](https://github.com/embeddings-benchmark/mteb) | Task-specific representation evaluation; retain complete task coverage and model/prompt/processor identity. |
| Multimodal benchmarks | [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | Optional grounded-generation evaluation once supported model and task adapters exist; preserve media/decoding contracts. |
| Metric implementations | [Hugging Face Evaluate](https://github.com/huggingface/evaluate), [torch-fidelity](https://github.com/toshas/torch-fidelity) | Reference metrics with pinned preprocessing and independent tiny oracles. Cross-framework evaluation is allowed; it does not require switching JAX training. |
| Speech experiments | [S3PRL](https://github.com/s3prl/s3prl) | Speech representation/probe reference and evaluation protocols. Porting a recipe is separate from claiming backend compatibility. |
| Generative pipelines | [Diffusers](https://github.com/huggingface/diffusers), [AudioCraft](https://github.com/facebookresearch/audiocraft) | Architecture/sampler/codec references and optional interoperability labs. Record model/component licenses and exact versions; a custom JAX artifact will not load automatically. |
| LLM serving | [vLLM supported-model registry](https://docs.vllm.ai/en/latest/models/supported_models/) | Optional runtime comparison only after architecture/conversion support is verified; do not advertise it for every custom decoder. |
| Service observability | [OpenTelemetry](https://opentelemetry.io/docs/what-is-opentelemetry/) and [Prometheus](https://github.com/prometheus/prometheus) | Trace request boundaries and expose operational counters/histograms. Connect model/container IDs; these metrics do not replace model evaluation. |
| Release and distribution | [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/guides/upload), [Safetensors](https://huggingface.co/docs/safetensors/index), Docker | Publish tested weights/processors/consumer interfaces and run a reproducible service. Upload and deployment remain separate steps. |

The course already uses Flax, Optax, Keras/TensorFlow/PyTorch interoperability and profiling. Keep them in their existing lessons and add only the model-specific gaps. MaxText remains an advanced JAX systems reference for a pinned supported workload, not a required framework for every modality. Apple/native edge runtimes, orchestration platforms and additional experiment trackers belong in later target-specific electives if a real project requires them.

## Tool selection exercise

Give the learner a constrained experiment: one dataset revision, two candidate configurations, a fixed development metric and a limited trial budget. They must choose a minimal tool stack, identify the authoritative checkpoint and metric records, reproduce one result after interruption, retrieve a selected artifact and explain what would break if a hosted service were unavailable.

Review the reasoning and observed recovery, not how many product dashboards were configured. Keep hosted credentials out of examples; core correctness and local tracking should work without a paid service. Document actual package/platform compatibility during implementation and provide distinct optional environments when framework dependencies conflict.
