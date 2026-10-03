# JAX Pathways curriculum

JAX from first principles to working systems.

Learn the shared foundation, build complete systems, then specialize. Notebooks are runnable companions to lessons. The source of truth is [site/curriculum.json](site/curriculum.json); this document is generated with `npm run curriculum:docs`.

## Start here

If you know basic Python, begin at phase 00. If you already use NumPy, use the foundations route to translate that experience into transformed functions. Math is introduced when needed, with optimization providing the common bridge to applications.

A CPU is sufficient for the early lessons. TPU and multi-device exercises declare their hardware explicitly.

## The course

### [00: Setup & first steps](phases/00-welcome/README.md)

Start here. Start after: Basic Python; no previous JAX experience.

- Set up your learning workspace
- Meet your arrays and devices
- Move your experiment to a TPU

**Build:** A reproducible environment report.

**Checkpoint:** Explain which device ran your code and reproduce the result in a fresh environment.

### [01: Arrays & pure functions](phases/01-arrays/README.md)

Shared foundations. Start after: 00: Setup & first steps.

- From NumPy to jax.numpy
- Shapes, broadcasting, and dtypes
- Pure functions and explicit inputs
- Immutable updates and indexing

**Build:** A tested array-processing function.

**Checkpoint:** Predict output shapes and explain why a pure function is easier to transform.

### [02: JAX transformations](phases/02-transforms/README.md)

Shared foundations. Start after: 01: Arrays & pure functions.

- Your first gradient (sample available)
- Losses and value_and_grad
- Batch a function with vmap
- Compile a function with jit
- Tracing, static arguments, and recompilation

**Build:** A compiled batch of gradients.

**Checkpoint:** Compose grad, vmap, and jit; explain what each transformation changes.

### [03: State, randomness & control flow](phases/03-state/README.md)

Shared foundations. Start after: 02: JAX transformations.

- Random keys without surprises
- Pytrees and structured parameters
- Compiled loops with lax.scan
- Branches with lax.cond

**Build:** A deterministic batched simulation.

**Checkpoint:** Replay the same experiment with saved state and keys; explain the update order.

### [04: Math & optimization](phases/04-optimization/README.md)

Shared foundations. Start after: 03: State, randomness & control flow.

- Linear algebra and loss intuition
- Gradient checking and numerical accuracy
- Write gradient descent yourself
- Optimize with Optax

**Build:** A fitted regression model.

**Checkpoint:** Explain the loss, check a gradient numerically, and justify one optimization choice.

### [05: Neural network training](phases/05-networks/README.md)

Training systems. Start after: 04: Math & optimization.

- Build a tiny multilayer perceptron
- Model and state with Flax NNX
- A compiled train and evaluation step
- Train a CNN on a small dataset
- Debug unstable learning

**Build:** A reproducible classifier.

**Checkpoint:** Keep train and evaluation behavior separate and reproduce metrics from a saved configuration.

### [06: Data & checkpoint recovery](phases/06-recovery/README.md)

Training systems. Start after: 05: Neural network training.

- Design an input pipeline
- Load, batch, and prefetch with Grain
- Save model, optimizer, and random state
- Recover data position and resume
- Asynchronous checkpoints and failure boundaries

**Build:** An interrupted run restored with its full state.

**Checkpoint:** Compare resumed and uninterrupted runs under a stated determinism tolerance.

### [07: Transformers & language models](phases/07-transformers/README.md)

Training systems. Start after: 06: Data & checkpoint recovery.

- Attention from small pieces
- Masking, tokenization, and sequence packing
- A Transformer block and mixed precision
- Train, checkpoint, and generate

**Build:** A checkpointed causal language model.

**Checkpoint:** Verify masking, sequence handling, restoration, and generation before comparing speed.

### [08: Performance diagnosis](phases/08-performance/README.md)

Training systems. Start after: 05: Neural network training.

- Benchmark asynchronous work correctly
- Diagnose recompilation and host synchronization
- Read an XProf trace
- Reason with HLO and the roofline model
- Memory, rematerialization, and optimization tradeoffs

**Build:** A trace-backed before-and-after performance report.

**Checkpoint:** Separate compilation, execution, input stalls, and synchronization before claiming a speedup.

### [09: Distributed training](phases/09-distributed/README.md)

Specializations. Start after: 06: Data & checkpoint recovery; 08: Performance diagnosis.

- Arrays, meshes, and sharding
- A sharded training step
- Communication-efficient algorithms
- Resilient distributed training

**Build:** A resilient sharded training experiment.

**Checkpoint:** Explain partitioning, communication costs, and recovery behavior with measured evidence.

### [10: Scientific computing](phases/10-science/README.md)

Specializations. Start after: 04: Math & optimization.

- A functional physical simulation
- Vectorize trajectories and scan time
- Differentiate through a solver
- Fit parameters and scale the simulation

**Build:** A differentiable inverse problem.

**Checkpoint:** Recover a known physical parameter and report error, stability, and gradient checks.

### [11: Probabilistic modeling](phases/11-probability/README.md)

Specializations. Start after: 04: Math & optimization.

- Distributions and sampling
- Build a Bayesian regression
- Hamiltonian Monte Carlo and sampler diagnostics
- Variational inference and model checking

**Build:** A Bayesian model with diagnostic evidence.

**Checkpoint:** Report sampler diagnostics and distinguish uncertainty from predictive error.

### [12: Reinforcement learning](phases/12-rl/README.md)

Specializations. Start after: 05: Neural network training.

- Write a functional environment
- Batch environments and scan rollouts
- Policy gradients and PPO
- Evaluate agents across seeds

**Build:** An evaluated policy in a functional environment.

**Checkpoint:** Compare a trained policy to a baseline across several seeds and report variability.

### [13: Pallas kernels](phases/13-kernels/README.md)

Specializations. Start after: 09: Distributed training.

- Pallas grids and BlockSpecs
- A first TPU kernel
- Tiling, memory, and pipelining
- Iterate with correctness and performance evidence

**Build:** A measured custom kernel with correctness checks.

**Checkpoint:** Compare against a reference across shapes and justify each optimization with evidence.

### [14: Autodiff & JAX internals](phases/14-internals/README.md)

Specializations. Start after: 04: Math & optimization.

- Read your first jaxpr
- JVPs, VJPs, and higher-order derivatives
- Custom derivative rules
- Lowering, compilation, and a tiny transformation

**Build:** A custom derivative with an explained jaxpr.

**Checkpoint:** Explain tracing and verify the derivative rule numerically.

### [15: Inference & model adaptation](phases/15-deployment/README.md)

Specializations. Start after: 06: Data & checkpoint recovery; 08: Performance diagnosis.

- Keras and PyTorch bridges to explicit JAX
- Adapt a pretrained model and choose a post-training objective
- Export and serve a trained computation
- Inference capacity, batching, and autoscaling

**Build:** An inference artifact and capacity plan.

**Checkpoint:** Measure latency and throughput under a stated workload and test the exported artifact.

### [16: Workload operations](phases/16-operations/README.md)

Specializations. Start after: 09: Distributed training.

- Accelerator jobs and runtime lifecycle
- Health, logs, and workload observability
- Failure recovery and operational runbooks
- Changes, rollback, and artifact provenance
- Capacity, utilization, and operating cost

**Build:** An observable, restartable accelerator job with a tested recovery runbook.

**Checkpoint:** Diagnose a failed or stalled job, restore it correctly, and justify its resource plan from recorded measurements.

## Choose the work you want to do

### Training & model development

Move from a loss and a handwritten update to neural networks, Transformers, reproducible data, and recoverable training.

**Capabilities:** Design models and optimization loops; Evaluate quality and debug training; Checkpoint and reproduce complete runs.

**Pathways:** Train neural networks; Build a TPU training system; Learn by interacting.

**Roles:** ML & training engineer; RL engineer.

**Artifact:** A trained, evaluated model with reproducible state recovery.

### Inference & deployment

Adapt models, export computations, verify predictions, and plan serving capacity using measured latency and throughput.

**Capabilities:** Adapt and export trained models; Verify inference correctness; Plan batching, latency, and capacity.

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

Turn a model idea into a tested training system, from the first update to recovery after an interruption.

**Work:** Build and evaluate neural networks; Design data pipelines and resumable training; Debug losses, state, and model quality.

**Skills:** Flax NNX and explicit state; Optax and train/eval steps; Grain, Orbax, and XProf.

**Portfolio:** A seeded training system with held-out evaluation, complete checkpoint recovery, and a performance report.

**Demonstrate:** Demonstrate that your model trains, evaluates, resumes correctly, and produces reproducible results.

### Research engineer

Implement a method, inspect its gradients, and test whether the evidence supports the idea.

**Work:** Prototype differentiable algorithms; Investigate uncertainty and inference; Run controlled experiments and numerical checks.

**Skills:** JVPs, VJPs, and custom derivatives; jaxpr and program inspection; NumPyro and sampler diagnostics.

**Portfolio:** A small research implementation with checked gradients or posterior diagnostics, baselines, and reproducible results.

**Demonstrate:** Explain the assumptions of your method and support a conclusion with correctness checks and an appropriate baseline.

### Scientific ML engineer

Use differentiable computing to connect physical models, observations, and learned components.

**Work:** Build and batch numerical simulations; Differentiate through dynamics and solvers; Recover parameters from observed trajectories.

**Skills:** scan and batched simulation; Diffrax and differentiable solvers; Numerical accuracy and inverse problems.

**Portfolio:** A differentiable inverse problem that recovers a physical parameter and predicts held-out trajectories.

**Demonstrate:** Verify parameter recovery, gradient accuracy, and simulation stability on a stated test problem.

### RL engineer

Connect functional environments, batched experience, and policy updates into a repeatable learning experiment.

**Work:** Build reproducible environments and rollouts; Implement and diagnose policy updates; Evaluate agents across independent seeds.

**Skills:** Vectorized environments and PRNG; scan-based rollouts; Policy gradients and PPO.

**Portfolio:** A trained policy compared with a simple baseline across multiple seeds, with reproducible learning curves.

**Demonstrate:** Show that the evaluation measures policy improvement rather than a lucky seed or a changed environment.

### Accelerator & performance engineer

Explain where time and memory go, then improve a workload with evidence rather than guesses.

**Work:** Diagnose compute, input, and communication bottlenecks; Partition workloads across devices; Optimize custom kernels when justified.

**Skills:** XProf, HLO, and roofline reasoning; Meshes, sharding, and distributed recovery; Pallas and reference correctness checks.

**Portfolio:** A sharded workload or custom kernel with trace-backed analysis, correctness checks, and a measured speed comparison.

**Demonstrate:** Preserve correctness, separate compile time from execution, and justify the improvement using workload measurements.

### Inference & deployment engineer

Bridge experiments and deployed workloads, with tested predictions and a capacity plan grounded in measurements.

**Work:** Adapt and export trained computations; Measure serving latency and throughput; Plan batching, capacity, and rollout behavior.

**Skills:** Framework and state bridges; JAX export and artifact verification; Batching and capacity analysis.

**Portfolio:** A verified inference artifact with a workload definition, latency/throughput measurements, and a batching plan.

**Demonstrate:** Reproduce exported predictions and defend the serving plan under a clearly stated workload.

### JAX / accelerator operations engineer

Make accelerator jobs reproducible, diagnose failures, and plan resources for dependable operation.

**Work:** Manage accelerator jobs and runtime lifecycle; Monitor throughput, stalls, and checkpoint health; Test failure recovery and plan capacity.

**Skills:** Distributed JAX and workload configuration; Grain, Orbax, and recovery protocols; Profiling, observability, and operational runbooks.

**Portfolio:** A restartable accelerator workload with logs, a tested recovery runbook, and a measured resource plan.

**Demonstrate:** Detect and diagnose a controlled failure, restore full workload state, and explain operational tradeoffs.

## Goal-based pathways

Each pathway references the same canonical phases. It adds a final project and synthesis assessment, rather than duplicating content.

### Start with JAX

Turn familiar Python into functions you can differentiate, batch, and compile.

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization.

**Final project:** A fitted regression model.

**Assessment:** Show a tested loss, numerically checked gradients, and a reproducible fit.

### Train neural networks

Go from a handwritten training loop to models you can save, evaluate, and extend.

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 05: Neural network training → 06: Data & checkpoint recovery → 07: Transformers & language models.

**Final project:** A checkpointed classifier or tiny language model.

**Assessment:** Demonstrate evaluation, restore behavior, and predictions from a saved artifact.

### Simulate and discover

Differentiate through numerical models and fit them to observations.

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 10: Scientific computing.

**Final project:** A scientific inverse problem.

**Assessment:** Recover a known parameter and validate predictions on unseen trajectories.

### Model uncertainty

Make predictions that describe what you know—and what you do not.

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 11: Probabilistic modeling.

**Final project:** A checked Bayesian regression.

**Assessment:** Submit posterior predictive checks and sampler or approximation diagnostics.

### Learn by interacting

Build environments and train agents with batches of simulated experience.

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 05: Neural network training → 12: Reinforcement learning.

**Final project:** An evaluated policy.

**Assessment:** Compare trained and baseline returns across independent seeds.

### Make it fast. Make it scale.

Measure what matters, then move from one device to many.

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 05: Neural network training → 06: Data & checkpoint recovery → 08: Performance diagnosis → 09: Distributed training → 13: Pallas kernels.

**Final project:** A measured sharded workload.

**Assessment:** Submit correctness checks, a trace, partitioning decisions, and a performance comparison.

### Understand JAX itself

Look beneath the transformations and learn how a Python function becomes a program.

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 14: Autodiff & JAX internals.

**Final project:** An explained transformation and custom derivative.

**Assessment:** Connect a jaxpr to your function and numerically verify the derivative rule.

### Build a TPU training system

Connect the foundations to reproducible training, failure recovery, and measured optimization.

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 05: Neural network training → 06: Data & checkpoint recovery → 07: Transformers & language models → 08: Performance diagnosis.

**Final project:** From notebook to training system.

**Assessment:** Interrupt, restore, and profile a seeded training run; explain correctness and performance evidence.

### Ship and adapt JAX workloads

Move an experiment into a usable system, or bring your existing NumPy and framework experience along.

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 05: Neural network training → 06: Data & checkpoint recovery → 08: Performance diagnosis → 15: Inference & model adaptation.

**Final project:** A served model with a capacity plan.

**Assessment:** Verify exported predictions and justify batching and capacity using measured workload results.

### Operate JAX workloads

Connect repeatable jobs, data and checkpoint recovery, observability, and accelerator capacity into an operational workflow.

**Follow:** 00: Setup & first steps → 01: Arrays & pure functions → 02: JAX transformations → 03: State, randomness & control flow → 04: Math & optimization → 05: Neural network training → 06: Data & checkpoint recovery → 08: Performance diagnosis → 09: Distributed training → 16: Workload operations.

**Final project:** Operate and recover an accelerator job.

**Assessment:** Run a controlled failure drill, identify it from workload signals, restore full state, and explain the capacity and rollback plan.

## Where the three notebook topics belong

- Notebook 1: Flax NNX + Optax training system → 05: Neural network training. Source pending.
- Notebook 2: Grain + Orbax reproducible training → 06: Data & checkpoint recovery. Source pending.
- Notebook 3: XProf performance → 08: Performance diagnosis. Source pending.

Their source files have not been imported. Once available, split them into lesson companions and phase integration labs without reproducing the foundation in each notebook.

## Completion means evidence

Reading a lesson, solving its exercise, passing a checkpoint, and completing the pathway project are different milestones. No route awards readiness or a certificate without reviewed artifacts and a synthesis assessment. All projects and assessments in this revision are planned briefs.
