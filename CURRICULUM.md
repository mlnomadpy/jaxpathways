# JAX Pathways curriculum

JAX from first principles to working systems.

Learn the shared foundation, build complete systems, then specialize. Notebooks are runnable companions to lessons. The course manifest is [curriculum/course.json](curriculum/course.json); authored content lives in each lesson folder; this document is generated with `npm run curriculum:docs`.

## Start here

If you know basic Python, begin at phase 00. If you already use NumPy, use the foundations route to translate that experience into transformed functions. Math is introduced when needed, with optimization providing the common bridge to applications.

A CPU is sufficient for the early lessons. TPU and multi-device exercises declare their hardware explicitly.

## The course

### [00: Setup & first steps](phases/00-welcome/README.md)

Start here. Start after: Basic Python; no previous JAX experience.

- Set up your learning workspace (lesson and exercise available)
- Meet your arrays and devices (lesson and exercise available)
- Practice with four virtual CPU devices (lesson and exercise available)
- Move your experiment to a TPU (lesson and exercise available)

**Build:** A reproducible environment report.

**Checkpoint:** Explain which device ran your code and reproduce the result in a fresh environment.

### [01: Arrays & pure functions](phases/01-arrays/README.md)

Shared foundations. Start after: 00: Setup & first steps.

- From NumPy to jax.numpy (lesson and exercise available)
- Shapes, broadcasting, and dtypes (lesson and exercise available)
- Pure functions and explicit inputs (lesson and exercise available)
- Immutable updates and indexing (lesson and exercise available)

**Build:** A tested array-processing function.

**Checkpoint:** Predict output shapes and explain why a pure function is easier to transform.

### [02: JAX transformations](phases/02-transforms/README.md)

Shared foundations. Start after: 01: Arrays & pure functions.

- Your first gradient (lesson and exercise available)
- Losses and value_and_grad (lesson and exercise available)
- Batch a function with vmap (lesson and exercise available)
- Compile a function with jit (lesson and exercise available)
- Tracing, static arguments, and recompilation (lesson and exercise available)

**Build:** A compiled batch of gradients.

**Checkpoint:** Compose grad, vmap, and jit; explain what each transformation changes.

### [03: State, randomness & control flow](phases/03-state/README.md)

Shared foundations. Start after: 02: JAX transformations.

- Random keys without surprises (lesson and exercise available)
- Pytrees and structured parameters (lesson and exercise available)
- Compiled loops with lax.scan (lesson and exercise available)
- Branches with lax.cond (lesson and exercise available)

**Build:** A deterministic batched simulation.

**Checkpoint:** Replay the same experiment with saved state and keys; explain the update order.

### [04: Math & optimization](phases/04-optimization/README.md)

Shared foundations. Start after: 03: State, randomness & control flow.

- Vectors, norms, and projections (lesson and exercise available)
- Linear algebra and loss intuition (lesson and exercise available)
- Least squares, rank, and conditioning (lesson and exercise available)
- The chain rule, Jacobians, and directions (lesson and exercise available)
- Gradient checking and numerical accuracy (lesson and exercise available)
- Curvature, Hessians, and learning rates (lesson and exercise available)
- Write gradient descent yourself (lesson and exercise available)
- Probability, likelihood, and stable losses (lesson and exercise available)
- Regularization and honest validation (lesson and exercise available)
- Minibatches, expectation, and gradient noise (lesson and exercise available)
- Optimize with Optax (lesson and exercise available)
- Adam, clipping, and learning-rate schedules (lesson and exercise available)

**Build:** A fitted regression model.

**Checkpoint:** Explain prediction geometry and loss assumptions; verify derivatives; diagnose conditioning, curvature and gradient noise; justify an optimizer using held-out evidence.

### [05: Neural network training](phases/05-networks/README.md)

Training systems. Start after: 04: Math & optimization.

- Build a tiny multilayer perceptron (lesson and exercise available)
- Model and state with Flax NNX (lesson and exercise available)
- A compiled train and evaluation step (lesson and exercise available)
- Train a CNN on a small dataset (lesson and exercise available)
- Debug unstable learning (lesson and exercise available)

**Build:** Build and audit a neural classifier.

**Checkpoint:** Keep train and evaluation behavior separate and reproduce metrics from a saved configuration.

### [06: Data & checkpoint recovery](phases/06-recovery/README.md)

Training systems. Start after: 05: Neural network training.

- Design an input pipeline (lesson and exercise available)
- Load, batch, and checkpoint order with Grain (lesson and exercise available)
- Save model, optimizer, and random state (lesson and exercise available)
- Recover data position and resume (lesson and exercise available)
- Asynchronous checkpoints and failure boundaries (lesson and exercise available)
- Track experiments and lineage with MLflow (lesson and exercise available)

**Build:** An interrupted run restored with its full state.

**Checkpoint:** Compare resumed and uninterrupted runs under a stated determinism tolerance.

### [07: Transformers & language models](phases/07-transformers/README.md)

Training systems. Start after: 06: Data & checkpoint recovery.

- Attention from small pieces (lesson and exercise available)
- Masking, tokenization, and sequence packing (lesson and exercise available)
- A Transformer block and mixed precision (lesson and exercise available)
- Train, checkpoint, and generate (lesson and exercise available)

**Build:** A checkpointed causal language model.

**Checkpoint:** Verify masking, sequence handling, restoration, and generation before comparing speed.

### [08: Performance diagnosis](phases/08-performance/README.md)

Training systems. Start after: 05: Neural network training.

- Benchmark asynchronous work correctly (lesson and exercise available)
- Diagnose recompilation and host synchronization (lesson and exercise available)
- Read an XProf trace (lesson and exercise available)
- Reason with HLO and the roofline model (lesson and exercise available)
- Memory, rematerialization, and optimization tradeoffs (lesson and exercise available)

**Build:** A trace-backed before-and-after performance report.

**Checkpoint:** Separate compilation, execution, input stalls, and synchronization before claiming a speedup.

### [09: Distributed training](phases/09-distributed/README.md)

Specializations. Start after: 06: Data & checkpoint recovery; 08: Performance diagnosis.

- Arrays, meshes, and sharding (lesson and exercise available)
- A sharded training step (lesson and exercise available)
- Communication-efficient algorithms (lesson and exercise available)
- Resilient distributed training (lesson and exercise available)

**Build:** A resilient sharded training experiment.

**Checkpoint:** Explain partitioning, communication costs, and recovery behavior with measured evidence.

### [10: Scientific computing](phases/10-science/README.md)

Specializations. Start after: 04: Math & optimization.

- A functional physical simulation (lesson and exercise available)
- Vectorize trajectories and scan time (lesson and exercise available)
- Differentiate through a solver (lesson and exercise available)
- Fit parameters and scale the simulation (lesson and exercise available)

**Build:** A differentiable inverse problem.

**Checkpoint:** Recover a known physical parameter and report error, stability, and gradient checks.

### [11: Probabilistic modeling](phases/11-probability/README.md)

Specializations. Start after: 04: Math & optimization.

- Distributions and sampling (lesson and exercise available)
- Build a Bayesian regression (lesson and exercise available)
- Hamiltonian Monte Carlo and sampler diagnostics (lesson and exercise available)
- Variational inference and model checking (lesson and exercise available)

**Build:** A Bayesian model with diagnostic evidence.

**Checkpoint:** Report sampler diagnostics and distinguish uncertainty from predictive error.

### [12: Reinforcement learning](phases/12-rl/README.md)

Specializations. Start after: 05: Neural network training.

- Write a functional environment (lesson and exercise available)
- Batch environments and scan rollouts (lesson and exercise available)
- Policy gradients and PPO (lesson and exercise available)
- Evaluate agents across seeds (lesson and exercise available)

**Build:** An evaluated policy in a functional environment.

**Checkpoint:** Compare a trained policy to a baseline across several seeds and report variability.

### [13: Pallas kernels](phases/13-kernels/README.md)

Specializations. Start after: 09: Distributed training.

- Pallas grids and BlockSpecs (lesson and exercise available)
- A first TPU kernel (lesson and exercise available)
- Tiling, memory, and pipelining (lesson and exercise available)
- Iterate with correctness and performance evidence (lesson and exercise available)

**Build:** A checked Pallas kernel with an explicit target qualification plan.

**Checkpoint:** Compare against a reference across shapes and justify each optimization with evidence.

### [14: Autodiff & JAX internals](phases/14-internals/README.md)

Specializations. Start after: 04: Math & optimization.

- Read your first jaxpr (lesson and exercise available)
- JVPs, VJPs, and higher-order derivatives (lesson and exercise available)
- Custom derivative rules (lesson and exercise available)
- Lowering, compilation, and a tiny transformation (lesson and exercise available)

**Build:** Audit derivatives and a tiny transformation.

**Checkpoint:** Explain tracing and verify the derivative rule numerically.

### [15: Deployment, interoperability & edge AI](phases/15-deployment/README.md)

Specializations. Start after: 06: Data & checkpoint recovery; 08: Performance diagnosis.

- Adapt a pretrained model and choose a post-training objective (lesson and exercise available)
- Move models between JAX, Keras, TensorFlow and PyTorch (lesson and exercise available)
- Convert PyTorch weights to Flax and locate numerical errors (lesson and exercise available)
- Export a computation and verify its serving contract (lesson and exercise available)
- Choose weight, activation and accumulation precision (lesson and exercise available)
- Containerize a model service and verify its boundary (lesson and exercise available)
- Inference capacity, batching, and autoscaling (lesson and exercise available)
- Deploy at the edge: conversion, budgets and device checks (lesson and exercise available)

**Build:** Verify an inference artifact and capacity hypothesis.

**Checkpoint:** Verify an exported prediction contract, report weight/activation precision error, and distinguish CPU request measurements from target-device evidence.

### [16: Workload operations](phases/16-operations/README.md)

Specializations. Start after: 09: Distributed training.

- Accelerator jobs and runtime lifecycle (lesson and exercise available)
- Health, logs, and workload observability (lesson and exercise available)
- Failure recovery and operational runbooks (lesson and exercise available)
- Changes, rollback, and artifact provenance (lesson and exercise available)
- Capacity, utilization, and operating cost (lesson and exercise available)
- MLOps: data contracts, CI gates and monitoring (lesson and exercise available)
- LLMOps: version prompts, evaluate behavior and trace requests (lesson and exercise available)
- ModelOps: own, approve, roll back and retire a model (lesson and exercise available)

**Build:** An observable, restartable local workload.

**Checkpoint:** Diagnose a failed or stalled job, restore it correctly, and justify its resource plan from recorded measurements.

### [17: Self-supervised pretraining: masked and contrastive learning](phases/17-pretraining/README.md)

Training systems. Start after: 07: Transformers & language models.

- Masked language modeling: predict hidden tokens (lesson and exercise available)
- Masked image modeling: reconstruct missing patches (lesson and exercise available)
- Contrastive learning: views, positives and negatives (lesson and exercise available)

**Build:** An audited pretraining and adaptation objective toolkit.

**Checkpoint:** Complete stages 1–3 of training-methods with independent loss references, real training curves, changed masks or pairs and explicit synthetic-data limits.

### [18: Post-training: SFT, LoRA, reward models and RLHF](phases/18-posttraining/README.md)

Training systems. Start after: 17: Self-supervised pretraining: masked and contrastive learning; 12: Reinforcement learning.

- Supervised fine-tuning with response-only token loss (lesson and exercise available)
- LoRA: adapt, save and merge low-rank updates (lesson and exercise available)
- Learn a reward model from pairwise preferences (lesson and exercise available)
- RLHF mechanics: a frozen reward and PPO policy updates (lesson and exercise available)
- Direct preference optimization and reference-corrected margins (lesson and exercise available)

**Build:** An audited pretraining and adaptation objective toolkit.

**Checkpoint:** Complete stages 4–8 of training-methods. Retain shifted token-mask checks, frozen-base/merge evidence, preference oracles, signed PPO clipping and fixed-reference DPO comparisons.

## Choose the work you want to do

### Training & model development

Move from a loss and a handwritten update to neural networks, Transformers, reproducible data, and recoverable training.

**Capabilities:** Design models and optimization loops; Evaluate quality and debug training; Checkpoint and reproduce complete runs.

**Pathways:** Train neural networks; Learn TPUs and Google Cloud; Learn by interacting.

**Roles:** ML & training engineer; RL engineer.

**Artifact:** A trained, evaluated model with reproducible state recovery.

### Inference & deployment

Verify model exports across frameworks, compare weight and activation precision, and measure serving and edge workloads.

**Capabilities:** Framework interoperability and verified exports; Weight and activation precision policies; Edge runtime contracts and serving measurements.

**Pathways:** Ship and adapt JAX workloads.

**Roles:** Inference & deployment engineer.

**Artifact:** A verified inference artifact and a measured serving plan.

### Operations & accelerator systems

Operate jobs, trace bottlenecks, recover from failures, shard across devices, and optimize kernels when the evidence calls for it.

**Capabilities:** Monitor jobs and test recovery; Profile and plan accelerator capacity; Scale workloads and optimize bottlenecks.

**Pathways:** Operate JAX workloads; Make it fast. Make it scale..

**Roles:** JAX / accelerator operations engineer; Accelerator & performance engineer.

**Artifact:** An observable workload, a recovery runbook, and a performance report.

### Research & scientific computing

Differentiate simulations, model uncertainty, inspect transformed programs, and validate methods with numerical evidence.

**Capabilities:** Build differentiable simulations; Fit parameters and model uncertainty; Check derivatives and compare methods.

**Pathways:** Simulate and discover; Model uncertainty; Understand JAX itself.

**Roles:** Scientific ML engineer; Research engineer.

**Artifact:** A validated simulation, inverse problem, or research experiment.

## Career routes

### ML & training engineer

Turn a dataset and model idea into a reproducible experiment. You define the objective, diagnose learning, keep held-out evaluation isolated and prove a saved run can continue correctly.

**Work:** Build and evaluate neural networks; Design data pipelines and resumable training; Debug losses, state, and model quality; Track comparable training runs, data lineage and artifacts with MLflow..

**Skills:** Flax NNX and explicit state; Optax and train/eval steps; Grain, Orbax, and XProf; MLflow experiment provenance; Masked and contrastive pretraining; response-only SFT, LoRA and preference optimization; numerical architecture parity.

**Portfolio:** A run dossier: data split, configuration, learning curves, error examples and a restart comparison.

**Demonstrate:** Demonstrate that your model trains, evaluates, resumes correctly, and produces reproducible results.

**Milestones:**
1. Build the shared foundation: A reproducible regression fit with checked gradients and explicit random state.
2. Train and evaluate a small model: A reproducible classifier
3. Prove full-state recovery: An interrupted run restored with its full state
4. Diagnose and improve training speed: A trace-backed before-and-after performance report

### Research engineer

Turn a proposed method into an experiment that can support or refute a claim. Your work connects mathematical assumptions, a simple baseline, independent checks and carefully bounded conclusions.

**Work:** Prototype differentiable algorithms; Investigate uncertainty and inference; Run controlled experiments and numerical checks; Keep experiment provenance and independent evaluation definitions..

**Skills:** JVPs, VJPs, and custom derivatives; jaxpr and program inspection; NumPyro and sampler diagnostics; MLflow experiment provenance.

**Portfolio:** A reproducibility report with a baseline, ablation, numerical checks and unresolved limitations.

**Demonstrate:** Explain the assumptions of your method and support a conclusion with correctness checks and an appropriate baseline.

**Milestones:**
1. Build the shared foundation: A reproducible regression fit with checked gradients and explicit random state.
2. Inspect and verify a differentiable method: A custom derivative with an explained jaxpr

### Scientific ML engineer

Connect observations to a numerical or physical model. You check derivatives and solver behavior, fit parameters and explain how noise, conditioning and model assumptions affect the answer.

**Work:** Build and batch numerical simulations; Differentiate through dynamics and solvers; Recover parameters from observed trajectories; Version simulation inputs, solver settings and inferred-parameter artifacts..

**Skills:** scan and batched simulation; Solver sensitivities; optional Diffrax extension; Numerical accuracy and inverse problems; MLflow experiment provenance.

**Portfolio:** An inverse-problem report with reference solutions, sensitivity checks and a held-out observation test.

**Demonstrate:** Verify parameter recovery, gradient accuracy, and simulation stability on a stated test problem.

**Milestones:**
1. Build the shared foundation: A reproducible regression fit with checked gradients and explicit random state.
2. Recover a hidden physical parameter: A differentiable inverse problem

### RL engineer

Build experiments in which actions change later observations. You keep environment state, random sampling and policy updates reproducible, and evaluate improvement across seeds and conditions.

**Work:** Build reproducible environments and rollouts; Implement and diagnose policy updates; Evaluate agents across independent seeds; Track environment, policy and seed identities across evaluations..

**Skills:** Vectorized environments and PRNG; scan-based rollouts; Policy gradients and PPO; MLflow experiment provenance.

**Portfolio:** A policy-learning report with fixed evaluation conditions, seed variation and trajectory-level failure analysis.

**Demonstrate:** Show that the evaluation measures policy improvement rather than a lucky seed or a changed environment.

**Milestones:**
1. Build the shared foundation: A reproducible regression fit with checked gradients and explicit random state.
2. Build a policy model: A reproducible classifier
3. Evaluate a learning agent: An evaluated policy in a functional environment

### Accelerator & performance engineer

Measure where a workload spends time and memory, then improve the limiting operation while preserving its numerical contract. Explain when an optimization helps and when it does not.

**Work:** Diagnose compute, input, and communication bottlenecks; Partition workloads across devices; Optimize custom kernels when justified; Preserve runtime/image identity and workload contracts in benchmark evidence..

**Skills:** XProf, HLO, and roofline reasoning; Meshes, sharding, and distributed recovery; Pallas and reference correctness checks; MLflow experiment provenance.

**Portfolio:** A correctness-backed benchmark with synchronized latency, throughput, memory and scaling limits.

**Demonstrate:** Preserve correctness, separate compile time from execution, and justify the improvement using workload measurements.

**Milestones:**
1. Build the shared foundation: A reproducible regression fit with checked gradients and explicit random state.
2. Build a reference workload: A reproducible classifier
3. Make it recoverable: An interrupted run restored with its full state
4. Find its measured bottleneck: A trace-backed before-and-after performance report
5. Partition it across devices: A resilient sharded training experiment
6. Optimize a checked kernel: A measured custom kernel with correctness checks

### Inference & deployment engineer

Package a trained model so another process or device can use it reliably. Own preprocessing, exported predictions, precision choices and measurements under a defined request workload.

**Work:** Adapt and export trained computations; Measure serving latency and throughput; Plan batching, capacity, and rollout behavior; Validate precision tradeoffs and execution on target edge devices; Package non-root containers, verify readiness and preserve preprocessing/model/image identity..

**Skills:** JAX, Keras, TensorFlow and PyTorch interoperability; JAX export and artifact verification; Weight, activation and accumulation precision; Edge deployment and capacity analysis; Containerized delivery and release evidence.

**Portfolio:** A deployment dossier with parity tests, precision tradeoffs, target-runtime measurements and rollback criteria.

**Demonstrate:** Reproduce exported predictions and defend the serving plan under a clearly stated workload.

**Milestones:**
1. Build the shared foundation: A reproducible regression fit with checked gradients and explicit random state.
2. Produce a trained artifact: A reproducible classifier
3. Reproduce its saved state: An interrupted run restored with its full state
4. Measure a reference workload: A trace-backed before-and-after performance report
5. Verify serving and capacity: An inference artifact and capacity plan

### JAX / accelerator operations engineer

Make model workloads observable and recoverable. Track progress and resource use, diagnose failed or stalled runs and rehearse recovery before relying on it.

**Work:** Manage accelerator jobs and runtime lifecycle; Monitor throughput, stalls, and checkpoint health; Test failure recovery and plan capacity; Own MLOps release gates, LLMOps trace/evaluation versions and ModelOps approvals, rollback and retirement..

**Skills:** Distributed JAX and workload configuration; Grain, Orbax, and recovery protocols; Profiling, observability, and operational runbooks; Containerized delivery and release evidence.

**Portfolio:** An incident runbook with a deliberate interruption, recovery evidence and a measured resource budget.

**Demonstrate:** Detect and diagnose a controlled failure, restore full workload state, and explain operational tradeoffs.

**Milestones:**
1. Build the shared foundation: A reproducible regression fit with checked gradients and explicit random state.
2. Run a reference job: A reproducible classifier
3. Verify a recovery point: An interrupted run restored with its full state
4. Observe and diagnose workload behavior: A trace-backed before-and-after performance report
5. Operate a sharded job: A resilient sharded training experiment
6. Rehearse failure and recovery: An observable local job, a deliberate worker failure and a tested restore/rollback runbook; accelerator transfer needs a separate target run.

## Goal-based pathways

Each pathway references the same canonical phases. It adds a final project and synthesis assessment, rather than duplicating content.

### Start with JAX

Learn how arrays, derivatives and explicit state fit together by building and checking a regression experiment. Use the deeper math lessons when you need to explain why an update works or fails.

**For:** Python learners who can write functions and use lists; NumPy experience helps but is not required.

**First artifact:** A regression implementation with an independent gradient check and held-out predictions.

**Study advice:** Begin with setup, arrays, transformations and explicit state, completing one foundation-toolkit stage per phase. In Math & optimization, first complete loss intuition, gradient checking, handwritten descent and Optax; then build the regression project. Return for the deeper math sequence and optimizer-audit to explain conditioning, curvature, noisy updates and regularization. Predict each output before running it, then test your explanation with a changed input.

**Engineering extensions (follow each lesson prerequisite):** recovery-06. See [the engineering release project](projects/engineering-release/README.md).

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization.

**Final project:** A fitted regression model.

**Assessment:** Show a tested loss, numerically checked gradients, and a reproducible fit.

**Evidence to collect:**
- A shape contract and independent loss and gradient calculations
- A reproducible fit, held-out predictions and a changed-data experiment
- A diagnosis of a deliberately wrong gradient or broadcasting rule

### Train neural networks

Build a classifier, preserve the complete training run through interruption, and verify its exported inference. Study sequence models alongside the image route, then choose a modality harness to connect the full lifecycle.

**For:** Learners who can trace shapes, differentiate a scalar loss and carry state through an update.

**First artifact:** A CPU classifier with stable loss, replay checks and count-weighted evaluation.

**Study advice:** Build the small classifier first and keep a fixed evaluation split. Add optimizer, random and data-position state before rehearsing interruption. For the image capstone, follow its linked performance and deployment prerequisites before exporting; the project lists the exact lessons. Study attention, masking and generation for the text branch rather than treating image classification as evidence of language-model capability. After the causal Transformer, study masked/contrastive pretraining, the RL/PPO prerequisite and post-training methods. Use training-methods for objective evidence and weight-conversion for actual PyTorch/Flax parity.

**Engineering extensions (follow each lesson prerequisite):** recovery-06, deployment-07, operations-06. See [the engineering release project](projects/engineering-release/README.md).

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 05: Neural network training → 06: Data & checkpoint recovery → 07: Transformers & language models → 17: Self-supervised pretraining: masked and contrastive learning → 12: Reinforcement learning → 18: Post-training: SFT, LoRA, reward models and RLHF → 08: Performance diagnosis → 15: Deployment, interoperability & edge AI.

**Final project:** A recoverable image classifier with verified exported inference.

**Assessment:** Complete the eight image lifecycle stages and changed-condition synthesis: independent CNN checks, full-state recovery, frozen error analysis, precision, export and measured inference. Transformer-specific evidence belongs to the complementary text harness.

**Evidence to collect:**
- Training and evaluation splits, seeded configuration and learning curves
- A saved full state whose next update matches an uninterrupted run
- Predictions from the restored model and examples explaining its failures

### Simulate and discover

Simulate physical dynamics, check numerical and derivative accuracy, and recover unknown parameters with held-out evidence. Build a reusable CPU inverse-problem audit before extending to adaptive solvers or accelerators.

**For:** Scientists and engineers with a domain model to test; calculus is introduced in the shared math lessons.

**First artifact:** A cooling simulator with a convergence table and an independent sensitivity check.

**Study advice:** Work from the cooling model through batched trajectories, sensitivities and parameter fitting. At each step compare against the analytic solution before changing the physical rate or time grid. Use the deeper linear algebra and curvature lessons when parameters are hard to identify. Finish the inverse-problem project, then attempt the two-compartment assessment with its independent conservation check.

**Engineering extensions (follow each lesson prerequisite):** recovery-06. See [the engineering release project](projects/engineering-release/README.md).

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 10: Scientific computing.

**Final project:** Audit a differentiable scientific inverse problem.

**Assessment:** Demonstrate the staged project contracts, then complete the changed-condition synthesis review draft with independent calculations, observed outputs and failure diagnosis. Public reference checks do not supply an independent reviewer.

**Evidence to collect:**
- Convergence against the analytic solution at multiple rates and time grids
- Solver sensitivities checked by independent finite differences
- Parameter recovery, held-out errors and a diagnosis of numerical bias or non-identifiability

### Model uncertainty

Express assumptions as probability models, verify a known posterior, then implement HMC and variational inference with explicit diagnostics. Use predictive checks to distinguish a sampler problem from a model that does not explain the data.

**For:** Learners who want to move beyond a point prediction and explain assumptions and uncertainty.

**First artifact:** An exact Bayesian regression posterior checked against an independent host calculation.

**Study advice:** Start with distributions and the exact regression posterior so you have a reference for approximate inference. Run multiple HMC chains and deliberately inspect an unstable sampler before studying variational inference. Compare latent uncertainty with uncertainty in a future observation. Complete the Bayesian project with a changed prior and held-out data before moving to an optional NumPyro model.

**Engineering extensions (follow each lesson prerequisite):** recovery-06. See [the engineering release project](projects/engineering-release/README.md).

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 11: Probabilistic modeling.

**Final project:** Audit a Bayesian regression model.

**Assessment:** Defend the prior and likelihood, verify posterior and sampler calculations independently, distinguish latent and observation uncertainty, and diagnose changed-condition predictive failures. The synthesis remains a public review draft.

**Evidence to collect:**
- Prior, likelihood and posterior calculations checked against an independent reference
- Multiple seeded chains with trace, acceptance and between-chain diagnostics
- Predictive intervals separating latent and observation variance, including a model-mismatch example

### Learn by interacting

Train a policy from actual environment interaction, verify its objective against an independent small-state reference, and compare multiple seeded agents on fixed held-out evaluation streams.

**For:** Learners interested in sequential decisions who can already implement and evaluate a neural network.

**First artifact:** A tested environment transition table and reproducible batched trajectories.

**Study advice:** Test one environment transition before batching it. Reconstruct a trajectory by hand, then compare the policy gradient with the small-state enumeration reference. Train multiple agents with different seeds and freeze the evaluation policy and random streams. Use the policy project to separate an implementation bug, Monte Carlo variation and an algorithm that learns poorly.

**Engineering extensions (follow each lesson prerequisite):** recovery-06, operations-06. See [the engineering release project](projects/engineering-release/README.md).

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 05: Neural network training → 12: Reinforcement learning.

**Final project:** Train and audit a policy from interaction.

**Assessment:** Demonstrate the staged project contracts, then complete the changed-condition synthesis review draft with independent calculations, observed outputs and failure diagnosis. Public reference checks do not supply an independent reviewer.

**Evidence to collect:**
- Verified transitions, terminal masks and reconstructed return targets
- A policy-gradient comparison against independent state enumeration
- Learning curves and fixed held-out evaluation for several trained seeds, with failed trajectories explained

### Make it fast. Make it scale.

Profile a correct workload, preserve its global objective while partitioning arrays, and audit actual Pallas kernels before requesting target measurements. CPU semantics are available; accelerator speed and multi-host claims require their own receipts.

**For:** Engineers comfortable with training loops who want to reason about memory, communication and throughput.

**First artifact:** A synchronized timing report and a checked array placement on logical CPU devices.

**Study advice:** Keep one correct reference workload throughout this path. Measure completed work, inspect its trace and relate the bottleneck to memory traffic or computation. Verify partitioning and collectives on logical devices, then audit a Pallas kernel against independent references. Run the separate guarded TPU measurement only on supported target hardware before claiming a device or system speedup.

**Engineering extensions (follow each lesson prerequisite):** deployment-07, operations-06. See [the engineering release project](projects/engineering-release/README.md).

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 05: Neural network training → 06: Data & checkpoint recovery → 08: Performance diagnosis → 09: Distributed training → 13: Pallas kernels.

**Final project:** A checked kernel and a measurement-backed optimization argument.

**Assessment:** Defend the profile, global objective, partitioning and kernel correctness; quantify a whole-workload opportunity. CPU checks establish semantics. Actual accelerator execution and performance remain a separate required qualification for device claims.

**Evidence to collect:**
- Reference agreement across more than one shape and parameter setting
- Synchronized compile and execution measurements with a trace of the bottleneck
- Array partitioning, collective communication and a before/after comparison on the stated hardware

### Understand JAX itself

Trace a function into jaxpr, verify forward and reverse sensitivities, write exact custom derivative rules, and compile a small transformation whose limits you can explain.

**For:** Curious JAX users who can already use grad and jit and want to diagnose transformation behavior.

**First artifact:** An annotated jaxpr that explains a computation and a changed input shape.

**Study advice:** Start by connecting each jaxpr primitive to the function that produced it. Revisit directional derivatives and the chain rule before studying JVPs, VJPs and custom rules. Keep an independent derivative reference beside every transformation. Finish the derivative audit, then extend the tiny interpreter in the synthesis assessment and explain its unsupported cases.

**Engineering extensions (follow each lesson prerequisite):** deployment-07. See [the engineering release project](projects/engineering-release/README.md).

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 14: Autodiff & JAX internals.

**Final project:** Audit derivatives and a tiny transformation.

**Assessment:** Connect a jaxpr to your function and numerically verify the derivative rule.

**Evidence to collect:**
- An annotated jaxpr linking primitives and shapes to the original function
- Directional and adjoint derivative checks on changed inputs
- A custom derivative or transformation with a deliberate unsupported case and explicit rejection

### Learn TPUs and Google Cloud

Build and recover an experiment on CPU, then learn GCP provisioning, TPU generations, training and inference memory, precision policies and XProf profiling. Connect measured error and real target traces to model engineering.

**For:** Model builders who understand minibatches and want to connect sequence models to accelerator engineering.

**First artifact:** A CPU run dossier with configuration, environment, events and a checkpoint that reproduces an uninterrupted run.

**Study advice:** Start with the TPU and Google Cloud practical guide: rehearse recovery locally, understand project access and resource lifetimes, then collect actual TPU evidence when you have approved access. Continue with the TPU performance guide: compare generations, estimate training and cache memory, measure BF16/INT8 error and diagnose a warmed profile. Then build the tiny CPU Transformer and verify causal masking and held-out token loss. Restore the complete state in a fresh process, then compare real cache policies and exported prefill/decode. Follow the distributed and deployment bridge lessons before completing the harness. Use the executable TPU and multi-controller labs for later target qualification; CPU results cannot establish accelerator performance. Follow the LLMOps extension for prompt, retrieval and tool versions, evaluation cases and traces; cache throughput and application quality answer different questions. After the causal Transformer, study masked/contrastive pretraining, the RL/PPO prerequisite and post-training methods. Use training-methods for objective evidence and weight-conversion for actual PyTorch/Flax parity.

**Engineering extensions (follow each lesson prerequisite):** recovery-06, deployment-07, operations-07, performance-03, performance-04. See [the engineering release project](projects/engineering-release/README.md).

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 05: Neural network training → 06: Data & checkpoint recovery → 07: Transformers & language models → 17: Self-supervised pretraining: masked and contrastive learning → 12: Reinforcement learning → 18: Post-training: SFT, LoRA, reward models and RLHF → 08: Performance diagnosis → 09: Distributed training → 15: Deployment, interoperability & edge AI → 16: Workload operations.

**Final project:** Causal Transformer training-system capstone.

**Assessment:** Train and recover a byte Transformer, verify cache precision and exported inference, then defend the measurements. CPU synthesis review draft is available; physical TPU and multi-controller qualification remain separate.

**Evidence to collect:**
- Independent causal-attention and valid-token loss checks on disjoint data
- Fresh-process recovery of parameters, both Adam moments, dropout keys and document position
- FP32/BF16/INT8 cache parity, stored bytes and numerical-error comparisons
- Six serialized prefill/decode endpoints with fresh-process reload and artifact rejection
- A genuine profile and synchronized CPU measurements, plus separate actual-target qualification before any TPU claim
- Cloud resource identity, resolved environment, actual backend events, retrieved checkpoints and verified resource cleanup for any claimed TPU run
- Generation and topology choice, explicit weight/activation/accumulator/cache policy, memory budget and one tested profile hypothesis with retained quality evidence

### Ship and adapt JAX workloads

Move trained computations into a measured inference service. Check framework/precision parity, package a non-root Docker image, verify readiness and connect evaluation, model identity and rollback evidence.

**For:** Model builders who want to package reliable inference on servers or edge devices.

**First artifact:** An exported computation checked against its source on the same inputs.

**Study advice:** Begin with one saved model and a fixed set of parity inputs. Trace preprocessing, parameter layout and output semantics across each framework boundary. Separate training precision, exported precision and the target runtime before benchmarking. Use the precision and edge lessons to build a validation checklist; measure a real target device before claiming device latency or memory savings.

**Engineering extensions (follow each lesson prerequisite):** recovery-06, deployment-07, operations-06, operations-07, operations-08. See [the engineering release project](projects/engineering-release/README.md).

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 05: Neural network training → 06: Data & checkpoint recovery → 08: Performance diagnosis → 15: Deployment, interoperability & edge AI.

**Final project:** Verify an inference artifact and capacity hypothesis.

**Assessment:** Audit adaptation, exported precision artifacts and a validated local request boundary. Derive a capacity hypothesis from measured execution and explicitly simulated arrivals; network serving and target-device measurements remain extensions.

**Evidence to collect:**
- Saved-model and exported-runtime parity on a fixed input contract
- Weight, activation and accumulation policies with accuracy and range checks
- Completed inference timings and a workload-based capacity argument, identifying unmeasured target runtimes

### Operate JAX workloads

Operate reproducible model workloads from tracked experiments to containerized inference. Practice failure recovery, MLOps release gates, LLMOps evaluation and tracing, and ModelOps ownership, approval and rollback.

**For:** Engineers interested in reproducibility, failure recovery and operating accelerator workloads.

**First artifact:** An interrupted CPU run whose next data batches, losses and full state match uninterrupted execution.

**Study advice:** First reproduce the next update after a local interruption. Then inspect workload traces and placement so an incident has interpretable signals. Work through the operations project’s process, event, restore, release and capacity stages. Keep a runbook naming the accepted checkpoint and rollback decision; validate an actual cluster separately before claiming accelerator operations experience. After the deployment bridge, complete engineering-release to connect MLflow, a verified container and application/release policy. Keep local Docker evidence separate from cloud and accelerator qualification.

**Engineering extensions (follow each lesson prerequisite):** recovery-06, deployment-07, operations-06, operations-07, operations-08. See [the engineering release project](projects/engineering-release/README.md).

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 05: Neural network training → 06: Data & checkpoint recovery → 08: Performance diagnosis → 09: Distributed training → 15: Deployment, interoperability & edge AI → 16: Workload operations.

**Final project:** An observable, restartable local workload.

**Assessment:** Diagnose real local subprocess failures and timeouts, restore the accepted complete state, reject a degraded artifact and defend rollback and capacity decisions. Accelerator and cluster operation require additional target evidence.

**Evidence to collect:**
- Versioned configuration, structured progress signals and an incident timeline
- A controlled failure followed by a verified restore of the next update
- An accepted artifact, rollback criteria and capacity calculations separating measurements from assumptions

## Modality project guides

These guides connect the full lifecycle and identify available harnesses separately from planned work.

- [Image: from pixels to a deployed classifier](public/guides/image.md): Build an image classification system whose preprocessing, errors and runtime behavior you can inspect.
- [Text: from tokens to reproducible generation](public/guides/text.md): Build a small causal language-model system with trustworthy masking, recovery and decoding.
- [Audio: from waveforms to an edge sound classifier](public/guides/audio.md): Build a sound-event system that preserves timing, handles silence and measures the full signal-processing path.
- [Cross-modal: connect images and text](public/guides/cross-modal.md): Build a retrieval system that checks alignment between modalities before optimizing similarity search.

## Where the three notebook topics belong

- Notebook 1: Flax NNX + Optax training system → 05: Neural network training. Source pending.
- Notebook 2: Grain + Orbax reproducible training → 06: Data & checkpoint recovery. Source pending.
- Notebook 3: XProf performance → 08: Performance diagnosis. Source pending.

Their source files have not been imported. Once available, split them into lesson companions and phase integration labs without reproducing the foundation in each notebook.

## Completion means evidence

Reading a lesson, solving its exercise, passing a checkpoint, and completing the pathway project are different milestones. No route awards readiness or a certificate without reviewed artifacts and a synthesis assessment. Each pathway identifies its implemented project and synthesis review draft where available. A brief without a linked implementation remains planned; public checks do not supply independent review.
