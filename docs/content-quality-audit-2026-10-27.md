# Content quality audit, October 27, 2026

Audited **edition 0.17.1, PR #71, source `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`**. Findings refer to that cumulative review candidate, not only the published site and not every uncommitted local edit. The existing dirty workspace was preserved.

The course has a substantial foundation, especially Math & optimization, Scientific computing and Autodiff & JAX internals. Many examples have independent references, changed-input controls, useful figure explanations and honest limits. It is not yet ready to describe every advanced chapter as complete. Several displayed equations teach a different concept from the accompanying code; a few required exercise results cannot be produced from their stated inputs. Advanced chapters need guided handoffs from bounded mechanics to saved models and observed service/device behavior.

## Coverage and meaning of the verdicts

- **104 / 104 lessons and 22 / 22 chapter study guides** have a recorded verdict below, with source paths, content hashes and issue mappings in [audit.json](evidence/content-quality-audit-2026-10-27/audit.json).
- **57 usable drafts**: no blocking teaching defect identified in this review; optional next steps remain. This is not expert/novice acceptance.
- **29 need content repair**: specific math, claim, code/exercise or answer inconsistency.
- **18 need depth**: bounded examples are generally honest, but the intended skill needs a guided transfer or richer exercise.
- Review covered each lesson's objective, main code, primary exercise/solution, practice prompts and figure guidance, plus displayed math and all chapter guides. Full narrative close reading was selective; this is not a claim that every sentence or proof was independently verified.
- Exact small CPU function probes and independent NumPy/analytic counterexamples substantiate the highest-risk findings. Existing course execution receipts were not replaced by a new full training run. No physical TPU, cloud resource or model-quality campaign was launched.
- The project references below concern lesson handoffs. This pass does not independently qualify every project implementation or the planned 300M model lifecycles.

These editorial verdicts do **not** alter learner progress, maturity labels or the human-review registry. [#57](https://github.com/mlnomadpy/jaxpathways/issues/57) still covers genuine expert/novice review; [#29](https://github.com/mlnomadpy/jaxpathways/issues/29) covers full production lifecycles; [#37](https://github.com/mlnomadpy/jaxpathways/issues/37) covers the remaining manual physical-TPU workflow invocation. Reader qualification remains [#56](https://github.com/mlnomadpy/jaxpathways/issues/56). None substitutes for fixing the content findings here.

## Fix order and GitHub backlog

Open issues were checked before creation to avoid duplicating the previous implementation/QA backlog. Five P1 issues address correctness. Thirteen P2 issues address exercise evidence, scaffolding and transfer depth. Some lessons map to multiple focused issues; they are counted once in the verdict totals.

| Priority | Finding | Type | Lessons |
| --- | --- | --- | --- |
| P1 | [#74 [P1][Content] Align displayed equations with the actual lessons and qualify replay/timing claims](https://github.com/mlnomadpy/jaxpathways/issues/74) | confirmed-correction | welcome-02, arrays-03, arrays-04, transforms-04, transforms-05, state-04, networks-02, networks-03, recovery-01, recovery-02, recovery-03, recovery-04, recovery-05, transformers-03, performance-01, performance-02, performance-03, kernels-04, operations-07, operations-08, tpu-04, tpu-02, tpu-05, tpu-06 |
| P1 | [#75 [P1][Content] Make the attention permutation exercise solvable from its stated inputs](https://github.com/mlnomadpy/jaxpathways/issues/75) | confirmed-correction | transformers-01 |
| P1 | [#76 [P1][Content] Align value-fitting targets with the discount used by the GAE example](https://github.com/mlnomadpy/jaxpathways/issues/76) | confirmed-correction | rl-05 |
| P1 | [#77 [P1][Content] Make TPU readiness, launch and cleanup instructions agree with the tested scope](https://github.com/mlnomadpy/jaxpathways/issues/77) | confirmed-correction | welcome-cpu, welcome-03, tpu-01, tpu-04, tpu-02, tpu-05 |
| P1 | [#78 [P1][Content] Report the actual storage and operand bytes for each precision policy](https://github.com/mlnomadpy/jaxpathways/issues/78) | confirmed-correction | tpu-06 |
| P2 | [#79 [P2][Content] Make worked solutions produce the evidence their exercises require](https://github.com/mlnomadpy/jaxpathways/issues/79) | confirmed-correction | recovery-06, deployment-07, transforms-04, tpu-06 |
| P2 | [#80 [P2][Content] Teach neural-network failure localization with an actual multilayer model](https://github.com/mlnomadpy/jaxpathways/issues/80) | curriculum-extension | networks-05, networks-03 |
| P2 | [#81 [P2][Content] Use a stable profiling workload and teach a real timeline interpretation task](https://github.com/mlnomadpy/jaxpathways/issues/81) | mixed-correction-and-extension | performance-03, tpu-06 |
| P2 | [#82 [P2][Content] Connect model sharding to a per-device memory-fit exercise](https://github.com/mlnomadpy/jaxpathways/issues/82) | mixed-correction-and-extension | distributed-01, distributed-02, distributed-03, distributed-04, tpu-01, tpu-03 |
| P2 | [#83 [P2][Content] Extend framework parity from Dense/MLP layers to attention and sequence semantics](https://github.com/mlnomadpy/jaxpathways/issues/83) | curriculum-extension | deployment-01, deployment-08, transformers-03, transformers-04 |
| P2 | [#84 [P2][Content] Bridge masked and contrastive mechanics to held-out saved-encoder exercises](https://github.com/mlnomadpy/jaxpathways/issues/84) | curriculum-extension | pretraining-01, pretraining-02, pretraining-03 |
| P2 | [#85 [P2][Content] Provide a guided sequence-model handoff for SFT, reward learning and DPO](https://github.com/mlnomadpy/jaxpathways/issues/85) | curriculum-extension | transformers-04, deployment-02, posttraining-01, posttraining-03, posttraining-04, posttraining-05 |
| P2 | [#86 [P2][Content] Make the LoRA adapter artifact enforce the base identity taught in prose](https://github.com/mlnomadpy/jaxpathways/issues/86) | confirmed-correction | posttraining-02 |
| P2 | [#87 [P2][Content] Connect MLOps and LLMOps gates to actual prediction and trace evidence](https://github.com/mlnomadpy/jaxpathways/issues/87) | curriculum-extension | operations-06, operations-07 |
| P2 | [#88 [P2][Content] Demonstrate behavioral rollback and model retirement in the release lessons](https://github.com/mlnomadpy/jaxpathways/issues/88) | curriculum-extension | operations-04, operations-08 |
| P2 | [#89 [P2][Content] Stage long operations and recovery listings around the concept being learned](https://github.com/mlnomadpy/jaxpathways/issues/89) | curriculum-extension | operations-01, operations-02, operations-03, operations-04, operations-05, recovery-03, recovery-04 |
| P2 | [#90 [P2][Content] Add concept-specific diagrams where current plots do not show the mechanism](https://github.com/mlnomadpy/jaxpathways/issues/90) | curriculum-extension | tpu-01, networks-02, transforms-05, rl-05, kernels-03, transforms-02 |
| P2 | [#91 [P2][Content] Expose lesson-level prerequisites and a clear CPU-first setup route](https://github.com/mlnomadpy/jaxpathways/issues/91) | curriculum-extension | welcome-cpu, welcome-03, posttraining-04, rl-01, rl-02, rl-03, rl-04, rl-05 |

Recommended sequence: correct equations and impossible/mismatched learning targets; repair required exercise evidence; connect advanced objectives to saved-model and per-device behavior; then improve scaffolding, diagrams and prerequisite handoffs. Keep small mechanics examples as foundations rather than replacing them with unexplained large programs.

## Chapter assessment

| Chapter | Verdict | What works | Highest-value next step |
| --- | --- | --- | --- |
| Setup & first steps (4) | Needs repair | The saved-file/interpreter workflow is concrete and useful. | Correct the reduction math and blanket TPU milestone; make the advanced device branch easier to defer. |
| Arrays & pure functions (4) | Needs repair | Preprocessing, broadcasting, constant features and pure-call examples support changed inputs. | Fix two equations attached to unrelated concepts and retain the good shape contracts. |
| JAX transformations (5) | Needs repair | Analytic/numerical/autodiff references and trace ledgers are substantive. | Repair JIT/specialization equations and finish the independent gradient exercise; improve signature visualization. |
| State, randomness & control flow (4) | Needs repair | Key ownership, pytrees, scan and conditional result contracts are useful. | Replace the conditional section's unrelated pytree equation. |
| Math & optimization (12) | Usable draft | The strongest broad sequence: hand derivations, independent expected results, changed inputs and plot interpretation. | Add a cumulative unseen optimizer decision task and uncertainty-aware model selection as extensions, then seek novice/expert review. |
| Neural network training (5) | Needs repair and depth | A real tiny XOR MLP, independent losses and convolution checks are useful. | Align NNX and binary-loss math; make debugging use actual multilayer failure localization and add stateful evaluation transfer. |
| Data & checkpoint recovery (6) | Needs repair | Actual Grain/Orbax replay, future IDs and next-transition comparisons are strong. | Repair checkpoint/iterator/publication equations, finish the MLflow exercise and stage repeated helper code. |
| Transformers & language models (4) | Needs repair | Nonuniform attention references, packing masks and checkpoint continuation are concrete. | Fix the impossible zero-query permutation demand, clarify normalization parameters and guide the natural-text handoff. |
| Performance diagnosis (5) | Needs repair | Synchronization, trace events, modeled-versus-measured limits and remat controls are well scoped. | Correct timing equations and give a labeled actual timeline interpretation task. |
| Distributed training (4) | Needs repair and depth | Actual shards, unequal-count weighted gradients, collectives and resume evidence teach genuine mechanisms. | Connect the opening 2D model axis to a bounded parameter-sharded step and per-device memory ledger. |
| Scientific computing (4) | Usable draft | Discrete/continuous references, solver bias, identifiability and held-out starts are unusually careful. | Make the adaptive-solver extension a learner checklist; later add uncertainty and coupled-state transfer. |
| Probabilistic modeling (4) | Usable draft | Density/probability, conjugate arithmetic, HMC controls and VI family error are clearly separated. | Use one shared posterior for HMC/VI transfer, add joint sample views and an autocorrelation/ESS diagnosis task. |
| Reinforcement learning (5) | Needs repair | Termination versus truncation, randomness units and exact small-task references are valuable. | Align fitted value targets with GAE discount; later connect critic advantages into a scoped actor update. |
| Pallas kernels (4) | Needs repair | Real interpreted Pallas, tail controls and explicit target refusal avoid fabricated performance claims. | Relocate Amdahl math and add a buffer-lifetime pipeline diagram; physical performance remains a separate evidence claim. |
| Autodiff & JAX internals (4) | Usable draft | Typed jaxpr, independent derivatives, wrong-but-linear custom rules and a restricted interpreter are substantive. | Improve residual-flow and stepped environment-table explanations before extending the interpreter. |
| Deployment, interoperability & edge AI (8) | Needs repair and depth | Fresh-process export/request checks, quantization arithmetic and actual optional-framework parity are useful. | Finish altered-artifact rejection evidence, add sequence-model conversion and a worked edge-runtime interpretation task. |
| Workload operations (8) | Needs repair and depth | Real supervisors, process-isolated recovery and content-bound approval are good engineering foundations. | Correct unrelated math, reduce repeated scaffolding and connect evaluations, rollback and retirement to observed model behavior. |
| Self-supervised pretraining: masked and contrastive learning (3) | Needs depth | Masked targets, denominators, patch oracles and symmetric contrastive gradients explain bounded objectives honestly. | Add guided saved-encoder transfer with held-out groups, varying masks and representation probes using existing projects. |
| Post-training: SFT, LoRA, reward models and RLHF (5) | Needs depth | Response masks, LoRA mechanics and preference-objective counterexamples are useful; failed PPO quality is honestly retained. | Enforce adapter base identity and build a guided saved-decoder route with sequence log-probabilities, held-out prompts and retention. |
| TPU Setup: Provisioning & Runtime Verification (2) | Needs repair | Provisioning alternatives and synchronized arithmetic receipts give a useful start. | Separate requested-target readiness from CPU arithmetic success, qualify device/chip notation and align cleanup with provisioning ownership. |
| TPU Workflows: Launching Jobs & Checkpointed Experiments (2) | Needs repair | Checkpoint/control hashes and the source-reviewed target matrix are valuable. | Fix warmup/count math and preemption guarantees; teach staging as source-bound classification with explicit prerequisites. |
| TPU Systems: Generations, Memory, Precision & XProf (2) | Needs repair and depth | Newer prose correctly separates output dtype, native quantized execution, training state and inference cache. | Correct policy byte counts, prove per-device fit and use stable profiling inputs with a real timeline interpretation task. |

## Lesson-by-lesson review

### Setup & first steps

**Chapter verdict: Needs repair.** The saved-file/interpreter workflow is concrete and useful. Correct the reduction math and blanket TPU milestone; make the advanced device branch easier to defer.

#### [welcome-01: Set up your learning workspace](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/00-welcome/01-set-up-your-learning-workspace/lesson.json)

**Usable draft**. Concrete saved-file setup, OS-specific troubleshooting and a changed-sum task.

**Gap / next step:** First plot introduces squares although the starter only computes a sum; keep it optional and tie the first visual to the task.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [welcome-02: Meet your arrays and devices](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/00-welcome/02-meet-your-arrays-and-devices/lesson.json)

**Needs repair**. Axis examples, column-sum transfer and figure explanations are concrete.

**Gap / next step:** Guided reduction section displays an unrelated affine transform equation and switches from rows 1..6 to the code's 0..5 without a comparison table.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74)

#### [welcome-cpu: Practice with four virtual CPU devices](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/00-welcome/03-virtual-cpu-devices/lesson.json)

**Needs depth**. Correctly distinguishes logical placement from accelerator evidence.

**Gap / next step:** Already labeled optional; add an explicit CPU-first continuation and minimal placement walkthrough before requiring Mesh, PartitionSpec, jit and shard concepts.

**Backlog:** [#77](https://github.com/mlnomadpy/jaxpathways/issues/77), [#91](https://github.com/mlnomadpy/jaxpathways/issues/91)

#### [welcome-03: Move your experiment to a TPU](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/00-welcome/03-move-your-experiment-to-a-tpu/lesson.json)

**Needs depth**. Separates numeric, backend and physical-TPU evidence honestly.

**Gap / next step:** Cloud provisioning and qualification workflow overwhelms an arange-level first chapter; provide a clear optional exit and bridge to the later TPU chapters.

**Backlog:** [#77](https://github.com/mlnomadpy/jaxpathways/issues/77), [#91](https://github.com/mlnomadpy/jaxpathways/issues/91)

### Arrays & pure functions

**Chapter verdict: Needs repair.** Preprocessing, broadcasting, constant features and pure-call examples support changed inputs. Fix two equations attached to unrelated concepts and retain the good shape contracts.

#### [arrays-01: From NumPy to jax.numpy](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/01-arrays/01-from-numpy-to-jax-numpy/lesson.json)

**Usable draft**. Training-only preprocessing, held-out reuse, constant-feature policy and changed-batch tests teach useful transfer.

**Gap / next step:** Retain the strong small numerical examples; test learner understanding of epsilon versus replacement-scale policies.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [arrays-02: Shapes, broadcasting, and dtypes](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/01-arrays/02-shapes-broadcasting-and-dtypes/lesson.json)

**Usable draft**. Asymmetric shapes and the scalar-loss broadcast trap are well explained and exercised.

**Gap / next step:** Improve typography of the 100,000,000 precision example and keep feature-versus-row contracts explicit.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [arrays-03: Pure functions and explicit inputs](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/01-arrays/03-pure-functions-and-explicit-inputs/lesson.json)

**Needs repair**. Explicit parameter and mutable-container alias examples are useful.

**Gap / next step:** Guided purity section displays an indexed-update equation that does not describe predict(params,x).

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74)

#### [arrays-04: Immutable updates and indexing](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/01-arrays/04-immutable-updates-and-indexing/lesson.json)

**Needs repair**. Value semantics, repeated-index accumulation and fixed-shape masking are useful.

**Gap / next step:** Guided immutable-update section displays a matrix multiplication/einsum equation unrelated to indexed set/add.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74)

### JAX transformations

**Chapter verdict: Needs repair.** Analytic/numerical/autodiff references and trace ledgers are substantive. Repair JIT/specialization equations and finish the independent gradient exercise; improve signature visualization.

#### [first-gradient: Your first gradient](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/02-transforms/01-your-first-gradient/lesson.json)

**Usable draft**. Small derivative and chain-rule references connect slopes to code.

**Gap / next step:** Retain scalar-output and nonsmooth-domain distinctions; next review should test transfer rather than recall.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [transforms-02: Losses and value_and_grad](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/02-transforms/02-losses-and-value-and-grad/lesson.json)

**Usable draft**. Derives the MSE gradient and step-size overshoot independently.

**Gap / next step:** Loss and derivative share one plot axis despite different units; separate panels would reduce a misconception the caption currently has to undo.

**Backlog:** [#90](https://github.com/mlnomadpy/jaxpathways/issues/90)

#### [transforms-03: Batch a function with vmap](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/02-transforms/03-batch-a-function-with-vmap/lesson.json)

**Usable draft**. Shared versus mapped axes, loop reference and mean-gradient transfer are substantive.

**Gap / next step:** Add a concrete non-default out_axes example if extending beyond the present scope.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [transforms-04: Compile a function with jit](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/02-transforms/04-compile-a-function-with-jit/lesson.json)

**Needs repair**. Synchronizes actual latency samples and separates correctness from speed.

**Gap / next step:** Guided jit section displays a scan recurrence; primary exercise solution omits the requested manual matrix-gradient check even though an experiment contains it.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74), [#79](https://github.com/mlnomadpy/jaxpathways/issues/79)

#### [transforms-05: Tracing, static arguments, and recompilation](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/02-transforms/05-tracing-static-arguments-and-recompilation/lesson.json)

**Needs repair**. Static configuration and specific traced-branch failures are useful.

**Gap / next step:** Guided specialization section displays an SGD equation; the main figure shows reduction outputs rather than the trace/specialization mechanism.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74), [#90](https://github.com/mlnomadpy/jaxpathways/issues/90)

### State, randomness & control flow

**Chapter verdict: Needs repair.** Key ownership, pytrees, scan and conditional result contracts are useful. Replace the conditional section's unrelated pytree equation.

#### [state-01: Random keys without surprises](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/03-state/01-random-keys-without-surprises/lesson.json)

**Usable draft**. Replay, continuation-key ownership and identity-derived noise are distinct.

**Gap / next step:** Replace generic state equation with the actual key-splitting transition; add explicit stream/epoch tags before multi-host reuse.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [state-02: Pytrees and structured parameters](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/03-state/02-pytrees-and-structured-parameters/lesson.json)

**Usable draft**. Hand-derived gradient leaves and nested-tree checks are aligned.

**Gap / next step:** Add custom-container registration as a clearly optional next step, rather than broadening basic claims.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [state-03: Compiled loops with lax.scan](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/03-state/03-compiled-loops-with-lax-scan/lesson.json)

**Usable draft**. Exact recurrence, sensitivity and growing-carry failure form a coherent lesson.

**Gap / next step:** Retain the distinction between final carry and stacked outputs; transfer to a multi-leaf carry is the natural extension.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [state-04: Branches with lax.cond](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/03-state/04-branches-with-lax-cond/lesson.json)

**Needs repair**. Boundary values and batched conditional caveats are well described.

**Gap / next step:** Guided cond section displays a pytree flatten/map equation instead of a piecewise rule and branch result contract.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74)

### Math & optimization

**Chapter verdict: Usable draft.** The strongest broad sequence: hand derivations, independent expected results, changed inputs and plot interpretation. Add a cumulative unseen optimizer decision task and uncertainty-aware model selection as extensions, then seek novice/expert review.

#### [optimization-05: Vectors, norms, and projections](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/04-optimization/05-vectors-norms-and-projections/lesson.json)

**Usable draft**. Projection, residual orthogonality, rescaling and zero-direction checks align.

**Gap / next step:** Extend to a non-axis direction before general subspace projection; the eager float guard is explicitly scoped.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [optimization-01: Linear algebra and loss intuition](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/04-optimization/01-linear-algebra-and-loss-intuition/lesson.json)

**Usable draft**. Rows-to-predictions contract, bias shift and independent gradients are coherent.

**Gap / next step:** Reduce repetition of the same broadcast trap from arrays-02 while retaining the cross-link.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [optimization-06: Least squares, rank, and conditioning](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/04-optimization/06-least-squares-rank-and-conditioning/lesson.json)

**Usable draft**. Rank, nonuniqueness, sensitivity and solver use are distinguished.

**Gap / next step:** Add a singular-value geometry panel if extending the existing target-versus-fit figure.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [optimization-07: The chain rule, Jacobians, and directions](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/04-optimization/07-chain-rule-jacobians-and-directional-derivatives/lesson.json)

**Usable draft**. Hand Jacobian, JVP/VJP duality and changed-point tests support transfer.

**Gap / next step:** Keep rectangular-Jacobian practice prominent so square fixtures do not hide transpose errors.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [optimization-02: Gradient checking and numerical accuracy](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/04-optimization/02-gradient-checking-and-numerical-accuracy/lesson.json)

**Usable draft**. Explains the special quadratic cancellation curve and nonsmooth counterexample accurately.

**Gap / next step:** A second nonlinear finite-difference curve would let learners observe truncation error rather than only read about it.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [optimization-08: Curvature, Hessians, and learning rates](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/04-optimization/08-curvature-hessians-and-learning-rates/lesson.json)

**Usable draft**. Curvature bound, reparameterization and saddle counterexample are substantive.

**Gap / next step:** The coordinate history explains dynamics well; an optional rotated quadratic would test geometry beyond a diagonal Hessian.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [optimization-03: Write gradient descent yourself](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/04-optimization/03-write-gradient-descent-yourself/lesson.json)

**Usable draft**. Hand recurrence, unstable-rate failure and false-convergence counterexample align.

**Gap / next step:** Retain trajectory indexing and distinguish stopping evidence from a tiny update.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [optimization-09: Probability, likelihood, and stable losses](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/04-optimization/09-probability-likelihood-and-stable-losses/lesson.json)

**Usable draft**. Bernoulli likelihood, stable logit loss and gradient reference align.

**Gap / next step:** Add a calibration/reliability exercise when broader probability interpretation is introduced.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [optimization-10: Regularization and honest validation](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/04-optimization/10-regularization-and-validation/lesson.json)

**Usable draft**. Correctly separates objective, training, validation and intercept penalty.

**Gap / next step:** The constructed split favors a known slope; add a noisy multi-split exercise before generalizing model-selection behavior.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [optimization-11: Minibatches, expectation, and gradient noise](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/04-optimization/11-minibatches-expectation-and-gradient-noise/lesson.json)

**Usable draft**. Exact gradient variance enumeration and count-weighted microbatches are strong.

**Gap / next step:** Add the finite-population correction as an optional comparison to ordinary epoch shuffling.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [optimization-04: Optimize with Optax](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/04-optimization/04-optimize-with-optax/lesson.json)

**Usable draft**. Separates gradient/update/state and catches reset and double-subtraction errors.

**Gap / next step:** Bridge the manual loop directly into the compiled NNX transition taught next.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [optimization-12: Adam, clipping, and learning-rate schedules](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/04-optimization/12-adam-clipping-and-learning-rate-schedules/lesson.json)

**Usable draft**. Adam moment arithmetic, clipping order, schedule clock and plotted rebounds are well explained.

**Gap / next step:** Add a final unseen decision exercise with more than one data scale before claiming optimizer-selection competence.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

### Neural network training

**Chapter verdict: Needs repair and depth.** A real tiny XOR MLP, independent losses and convolution checks are useful. Align NNX and binary-loss math; make debugging use actual multilayer failure localization and add stateful evaluation transfer.

#### [networks-01: Build a tiny multilayer perceptron](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/05-networks/01-build-a-tiny-multilayer-perceptron/lesson.json)

**Usable draft**. A real tiny XOR training run, independent BCE and changed-coordinate transfer are meaningful.

**Gap / next step:** Teach an explicit train/validation/test distinction when extending this bounded fixed-split task.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [networks-02: Model and state with Flax NNX](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/05-networks/02-model-and-state-with-flax-nnx/lesson.json)

**Needs repair**. NNX forward parity and independent-clone state checks are useful.

**Gap / next step:** The equation uses a three-part split while the shown call returns two parts; the leaf-count figure uses anonymous indices instead of module paths.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74), [#90](https://github.com/mlnomadpy/jaxpathways/issues/90)

#### [networks-03: A compiled train and evaluation step](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/05-networks/03-a-compiled-train-and-evaluation-step/lesson.json)

**Needs repair**. Evaluation checks frozen parameter bytes and optimizer count.

**Gap / next step:** The displayed formula is multiclass one-hot CE, while the implementation trains one binary logit with sigmoid BCE. Add a stateful train/eval example as a later extension.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74), [#80](https://github.com/mlnomadpy/jaxpathways/issues/80)

#### [networks-04: Train a CNN on a small dataset](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/05-networks/04-train-a-cnn-on-a-small-dataset/lesson.json)

**Usable draft**. Hand convolution patch, held-out confusion and missing-class recall transfer align.

**Gap / next step:** Add filter/activation illustrations before scaling beyond the simple stripe task.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [networks-05: Debug unstable learning](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/05-networks/05-debug-unstable-learning/lesson.json)

**Needs depth**. Scalar quadratic instability is accurately derived and checked.

**Gap / next step:** The neural-network debugging chapter returns to the same scalar regression instead of demonstrating a multi-layer failure, localization and repair.

**Backlog:** [#80](https://github.com/mlnomadpy/jaxpathways/issues/80)

### Data & checkpoint recovery

**Chapter verdict: Needs repair.** Actual Grain/Orbax replay, future IDs and next-transition comparisons are strong. Repair checkpoint/iterator/publication equations, finish the MLflow exercise and stage repeated helper code.

#### [recovery-01: Design an input pipeline](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/06-recovery/01-design-an-input-pipeline/lesson.json)

**Needs repair**. Membership/order/alignment, remainder masks and scope of timing are clear.

**Gap / next step:** Its checkpoint equation implies replay from an incomplete state tuple without data position, data/config identity or matching execution assumptions.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74)

#### [recovery-02: Load, batch, and checkpoint order with Grain](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/06-recovery/02-load-batch-and-prefetch-with-grain/lesson.json)

**Needs repair**. Grain next-read identity and epoch-boundary transfer are meaningful.

**Gap / next step:** The guided iterator section displays a momentum update, unrelated to restoring the next record.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74)

#### [recovery-03: Save model, optimizer, and random state](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/06-recovery/03-save-model-optimizer-and-random-state/lesson.json)

**Needs repair**. Actual Orbax save/restore compares complete state and the next transition.

**Gap / next step:** The hash equation compares serialization bytes to an undefined hash of a deserialized object; teach semantic state equality and artifact integrity separately.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74), [#89](https://github.com/mlnomadpy/jaxpathways/issues/89)

#### [recovery-04: Recover data position and resume](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/06-recovery/04-recover-data-position-and-resume/lesson.json)

**Needs repair**. Replays every future ID and update through an epoch boundary.

**Gap / next step:** The modulo-position equation is not the shuffled Grain iterator implemented here; a 128-line listing repeats low-level helpers before the central resume story.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74), [#89](https://github.com/mlnomadpy/jaxpathways/issues/89)

#### [recovery-05: Asynchronous checkpoints and failure boundaries](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/06-recovery/05-asynchronous-checkpoints-and-failure-boundaries/lesson.json)

**Needs repair**. Requested/completed/published states and controlled interrupted publication are well separated.

**Gap / next step:** Its multi-host timing equation equates commit to the last write and then places rename after commit; this contradicts the lesson's accepted-publication boundary.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74)

#### [recovery-06: Track experiments and lineage with MLflow](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/06-recovery/06-track-experiments-and-lineage-with-mlflow/lesson.json)

**Needs repair**. Real MLflow retrieval and controlled run selection are stronger than dashboard-only examples.

**Gap / next step:** The exercise asks for a third model artifact and a misleading training-only selection, but the solution only computes zero-model validation error and logs no third run or artifact.

**Backlog:** [#79](https://github.com/mlnomadpy/jaxpathways/issues/79)

### Transformers & language models

**Chapter verdict: Needs repair.** Nonuniform attention references, packing masks and checkpoint continuation are concrete. Fix the impossible zero-query permutation demand, clarify normalization parameters and guide the natural-text handoff.

#### [transformers-01: Attention from small pieces](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/07-transformers/01-attention-from-small-pieces/lesson.json)

**Needs repair**. Attention axes and value-translation/NumPy controls are strong.

**Gap / next step:** The primary permutation exercise demands a changed output after value-only permutation, but the main zero-query fixture is invariant. The solution silently uses q_new/actual from a later experiment.

**Backlog:** [#75](https://github.com/mlnomadpy/jaxpathways/issues/75)

#### [transformers-02: Masking, tokenization, and sequence packing](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/07-transformers/02-masking-tokenization-and-sequence-packing/lesson.json)

**Usable draft**. Separates context eligibility, shifted targets, packed segments and all-masked padding.

**Gap / next step:** Add a concrete packed-token and position-ID diagram when joining the optional natural-text workflow.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [transformers-03: A Transformer block and mixed precision](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/07-transformers/03-a-transformer-block-and-mixed-precision/lesson.json)

**Needs repair**. Residual, causal-perturbation and projection-gradient checks are meaningful.

**Gap / next step:** The displayed LN includes learned gamma/beta absent from layer_norm; clarify fixed versus affine normalization and show the precision policy on an asymmetric input.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74), [#83](https://github.com/mlnomadpy/jaxpathways/issues/83)

#### [transformers-04: Train, checkpoint, and generate](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/07-transformers/04-train-checkpoint-and-generate/lesson.json)

**Usable draft**. Connected train/checkpoint/continuation and schema failure cases are honest about the periodic data.

**Gap / next step:** The main exercise repeats a known cycle; make a guided handoff to the existing natural-text project the chapter's substantive transfer task.

**Backlog:** [#83](https://github.com/mlnomadpy/jaxpathways/issues/83), [#85](https://github.com/mlnomadpy/jaxpathways/issues/85)

### Performance diagnosis

**Chapter verdict: Needs repair.** Synchronization, trace events, modeled-versus-measured limits and remat controls are well scoped. Correct timing equations and give a labeled actual timeline interpretation task.

#### [performance-01: Benchmark asynchronous work correctly](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/08-performance/01-benchmark-asynchronous-work-correctly/lesson.json)

**Needs repair**. Explicit completion boundaries and warm distributions are sound.

**Gap / next step:** The exact equation first=compile+warm overstates what was isolated; first-call timing also includes tracing/lowering/dispatch and differs from a warm median.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74)

#### [performance-02: Diagnose recompilation and host synchronization](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/08-performance/02-diagnose-recompilation-and-host-synchronization/lesson.json)

**Needs repair**. Actual trace-event ledger separates value and signature changes.

**Gap / next step:** Guided recompilation section displays a roofline equation unrelated to its tracing experiment.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74)

#### [performance-03: Read an XProf trace](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/08-performance/03-read-an-xprof-trace/lesson.json)

**Needs depth**. Captures actual named trace events and distinguishes nested host durations.

**Gap / next step:** The additive step equation lacks overlap/other-work assumptions; provide a labeled retained XProf timeline and ask learners to identify an actual event/critical path rather than only host annotation totals.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74), [#81](https://github.com/mlnomadpy/jaxpathways/issues/81)

#### [performance-04: Reason with HLO and the roofline model](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/08-performance/04-reason-with-hlo-and-the-roofline-model/lesson.json)

**Usable draft**. Work/byte contracts, hypothetical roofline and measured timing are kept separate.

**Gap / next step:** Extend with one real backend ceiling only when evidence is available; do not replace honest hypothetical labeling.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [performance-05: Memory, rematerialization, and optimization tradeoffs](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/08-performance/05-memory-rematerialization-and-optimization-tradeoffs/lesson.json)

**Usable draft**. Independent reverse recurrence and distinct residual/memory views avoid false remat claims.

**Gap / next step:** A byte-lifetime diagram and representative measured-memory extension would strengthen interpretation.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

### Distributed training

**Chapter verdict: Needs repair and depth.** Actual shards, unequal-count weighted gradients, collectives and resume evidence teach genuine mechanisms. Connect the opening 2D model axis to a bounded parameter-sharded step and per-device memory ledger.

#### [distributed-01: Arrays, meshes, and sharding](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/09-distributed/01-arrays-meshes-and-sharding/lesson.json)

**Needs repair**. Local shard values, row/column ownership and masked uneven means are well tested.

**Gap / next step:** The opening equation uses an unexplained 2D data/model mesh while every executable placement is a 1D mesh.

**Backlog:** [#82](https://github.com/mlnomadpy/jaxpathways/issues/82)

#### [distributed-02: A sharded training step](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/09-distributed/02-a-sharded-training-step/lesson.json)

**Usable draft**. Independent global gradient and weighted unequal batches teach the key denominator.

**Gap / next step:** Add a parameter-sharded continuation to connect data sharding with the chapter's model-axis notation.

**Backlog:** [#82](https://github.com/mlnomadpy/jaxpathways/issues/82)

#### [distributed-03: Communication-efficient algorithms](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/09-distributed/03-communication-efficient-algorithms/lesson.json)

**Usable draft**. Actual collectives, local result shape and whole-cycle ring accounting align.

**Gap / next step:** Use a chunk-flow diagram and extend the optimizer-slice experiment into an explicitly scoped model-sharded step.

**Backlog:** [#82](https://github.com/mlnomadpy/jaxpathways/issues/82)

#### [distributed-04: Resilient distributed training](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/09-distributed/04-resilient-distributed-training/lesson.json)

**Usable draft**. Next IDs, full-state comparison and delayed wrong-key divergence are useful.

**Gap / next step:** Keep multi-controller ownership and physical-worker failure evidence separate; connect the existing local two-controller project at the exact handoff.

**Backlog:** [#82](https://github.com/mlnomadpy/jaxpathways/issues/82)

### Scientific computing

**Chapter verdict: Usable draft.** Discrete/continuous references, solver bias, identifiability and held-out starts are unusually careful. Make the adaptive-solver extension a learner checklist; later add uncertainty and coupled-state transfer.

#### [science-01: A functional physical simulation](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/10-science/01-a-functional-physical-simulation/lesson.json)

**Usable draft**. Units, discrete/continuous oracles, fixed-horizon refinement and stability are clear.

**Gap / next step:** Add a second physical system only as an extension; the present scalar decay scope is honest.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [science-02: Vectorize trajectories and scan time](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/10-science/02-vectorize-trajectories-and-scan-time/lesson.json)

**Usable draft**. RK4 polynomial, batch/time layouts and linear scaling invariants are substantive.

**Gap / next step:** A coupled-state example is the next transfer beyond independent scalar ensembles.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [science-03: Differentiate through a solver](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/10-science/03-differentiate-through-a-solver/lesson.json)

**Usable draft**. Discrete sensitivity, continuous bias and finite-difference checks are distinguished.

**Gap / next step:** Move the adaptive-solver handoff into a short concrete learner checklist rather than leaving it as an optional API reference.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [science-04: Fit parameters and scale the simulation](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/10-science/04-fit-parameters-and-scale-the-simulation/lesson.json)

**Usable draft**. Constraint parameterization, identifiability, numerical bias and held-out starts are strong.

**Gap / next step:** Retain noisy profile checks; add uncertainty interpretation before presenting real physical parameter confidence.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

### Probabilistic modeling

**Chapter verdict: Usable draft.** Density/probability, conjugate arithmetic, HMC controls and VI family error are clearly separated. Use one shared posterior for HMC/VI transfer, add joint sample views and an autocorrelation/ESS diagnosis task.

#### [probability-01: Distributions and sampling](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/11-probability/01-distributions-and-sampling/lesson.json)

**Usable draft**. Density versus probability, standard error, mixture log-space and replay are distinct.

**Gap / next step:** Add a small histogram/density integration panel to complement the analytic standard-error plot.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [probability-02: Build a Bayesian regression](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/11-probability/02-build-a-bayesian-regression/lesson.json)

**Usable draft**. Conjugate arithmetic and latent/observation uncertainty are correctly separated.

**Gap / next step:** Carry this exact posterior directly into a learner HMC/VI comparison rather than switching examples without a handoff.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [probability-03: Hamiltonian Monte Carlo and sampler diagnostics](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/11-probability/03-hamiltonian-monte-carlo-and-sampler-diagnostics/lesson.json)

**Usable draft**. Actual HMC, reversibility, energy and classical-R-hat limitations are clear.

**Gap / next step:** Add an effective-sample-size/autocorrelation failure task and a joint-sample plot; connect the existing NumPyro extension.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [probability-04: Variational inference and model checking](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/11-probability/04-variational-inference-and-model-checking/lesson.json)

**Usable draft**. Analytic diagonal-VI optimum separates family error from optimizer error.

**Gap / next step:** Add a joint ellipse/sample view and apply posterior-predictive checks to the fitted regression from probability-02.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

### Reinforcement learning

**Chapter verdict: Needs repair.** Termination versus truncation, randomness units and exact small-task references are valuable. Align fitted value targets with GAE discount; later connect critic advantages into a scoped actor update.

#### [rl-01: Write a functional environment](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/12-rl/01-write-a-functional-environment/lesson.json)

**Usable draft**. Transition reward, termination/truncation and final observation are explicitly separated.

**Gap / next step:** Add a small corridor-state diagram and show remaining-time state before nonstationary policies.

**Backlog:** [#91](https://github.com/mlnomadpy/jaxpathways/issues/91)

#### [rl-02: Batch environments and scan rollouts](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/12-rl/02-batch-environments-and-scan-rollouts/lesson.json)

**Usable draft**. Padded masks, ownership keys and reward-to-go connect environment and learner.

**Gap / next step:** Use a time-by-environment mask diagram to replace repeated collector boilerplate.

**Backlog:** [#91](https://github.com/mlnomadpy/jaxpathways/issues/91)

#### [rl-03: Policy gradients and PPO](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/12-rl/03-policy-gradients-and-ppo/lesson.json)

**Usable draft**. Actual policy training, frozen behavior batch and exact expected-return reference are substantive.

**Gap / next step:** Connect the later critic/GAE into an actor update and add an equal-gradient-budget comparison rather than only a disclosed unequal-budget chart.

**Backlog:** [#91](https://github.com/mlnomadpy/jaxpathways/issues/91)

#### [rl-04: Evaluate agents across seeds](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/12-rl/04-evaluate-agents-across-seeds/lesson.json)

**Usable draft**. Training/evaluation randomness and per-agent versus episode units are clearly separated.

**Gap / next step:** Add uncertainty over paired agent-level differences when extending descriptive results to algorithm comparison.

**Backlog:** [#91](https://github.com/mlnomadpy/jaxpathways/issues/91)

#### [rl-05: Learn value functions and generalized advantage estimation](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/12-rl/05-learn-values-and-gae/lesson.json)

**Needs repair**. Separate bootstrap/trace masks and independent reset-boundary arithmetic are strong.

**Gap / next step:** The value table is fitted to undiscounted return targets, then used in GAE at gamma=.9 without an explicit objective change; align or explain this mismatch and supply a consistent actor-critic handoff.

**Backlog:** [#76](https://github.com/mlnomadpy/jaxpathways/issues/76), [#90](https://github.com/mlnomadpy/jaxpathways/issues/90), [#91](https://github.com/mlnomadpy/jaxpathways/issues/91)

### Pallas kernels

**Chapter verdict: Needs repair.** Real interpreted Pallas, tail controls and explicit target refusal avoid fabricated performance claims. Relocate Amdahl math and add a buffer-lifetime pipeline diagram; physical performance remains a separate evidence claim.

#### [kernels-01: Pallas grids and BlockSpecs](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/13-kernels/01-pallas-grids-and-blockspecs/lesson.json)

**Usable draft**. Real interpreted Pallas, tile ownership and complete tail checks align.

**Gap / next step:** The best next visual is explicit padding outside logical bounds, not another latency claim.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [kernels-02: A first TPU kernel](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/13-kernels/02-a-first-tpu-kernel/lesson.json)

**Usable draft**. Bias BlockSpec, represented-input reference and explicit target refusal are sound.

**Gap / next step:** Provide a load/compute/store memory-space diagram beside the target-compatible wrapper.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [kernels-03: Tiling, memory, and pipelining](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/13-kernels/03-tiling-memory-and-pipelining/lesson.json)

**Usable draft**. Actual buffered pipeline semantics, race detection and modeled storage limits are explicit.

**Gap / next step:** Add a buffer-lifetime timeline for prologue/steady state/drain so learners can reason about scheduling visually.

**Backlog:** [#90](https://github.com/mlnomadpy/jaxpathways/issues/90)

#### [kernels-04: Iterate with correctness and performance evidence](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/13-kernels/04-iterate-with-correctness-and-performance-evidence/lesson.json)

**Needs repair**. Separates implementation error from precision loss and guards target timings.

**Gap / next step:** The precision-reference section displays Amdahl's speedup formula without introducing its variables; move it to the end-to-end opportunity practice.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74)

### Autodiff & JAX internals

**Chapter verdict: Usable draft.** Typed jaxpr, independent derivatives, wrong-but-linear custom rules and a restricted interpreter are substantive. Improve residual-flow and stepped environment-table explanations before extending the interpreter.

#### [internals-01: Read your first jaxpr](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/14-internals/01-read-your-first-jaxpr/lesson.json)

**Usable draft**. Typed jaxpr, concrete intermediates and captured-constant checks align.

**Gap / next step:** Keep the illustrative printout version-qualified and explicitly link to the earlier tracing lesson.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [internals-02: JVPs, VJPs, and higher-order derivatives](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/14-internals/02-jvps-vjps-and-higher-order-derivatives/lesson.json)

**Usable draft**. Independent Jacobian/Hessian and Gauss-Newton distinction are rigorous for the fixture.

**Gap / next step:** Add one rectangular worked example before relying on square-space bar panels.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [internals-03: Custom derivative rules](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/14-internals/03-custom-derivative-rules/lesson.json)

**Usable draft**. Custom JVP/VJP contracts include wrong-but-linear rules, extremes and higher-order checks.

**Gap / next step:** Use a residual/cotangent flow diagram to make custom reverse-pass tuple structure easier to learn.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [internals-04: Lowering, compilation, and a tiny transformation](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/14-internals/04-lowering-compilation-and-a-tiny-transformation/lesson.json)

**Usable draft**. A real restricted interpreter, supported-rule boundaries and compiled comparisons teach actual composition.

**Gap / next step:** Add a single stepped environment-table example before the full interpreter listing.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

### Deployment, interoperability & edge AI

**Chapter verdict: Needs repair and depth.** Fresh-process export/request checks, quantization arithmetic and actual optional-framework parity are useful. Finish altered-artifact rejection evidence, add sequence-model conversion and a worked edge-runtime interpretation task.

#### [deployment-02: Adapt a pretrained model and choose a post-training objective](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/15-deployment/02-adapt-a-pretrained-model-and-choose-a-post-training-objective/lesson.json)

**Usable draft**. Actually pretrained/reloaded logistic weights, matched adaptation branches and retention budget are explicit.

**Gap / next step:** Follow the synthetic objective lesson with a saved encoder/decoder transfer task using the existing pilots.

**Backlog:** [#85](https://github.com/mlnomadpy/jaxpathways/issues/85)

#### [deployment-01: Move models between JAX, Keras, TensorFlow and PyTorch](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/15-deployment/01-keras-and-pytorch-bridges-to-explicit-jax/lesson.json)

**Needs depth**. Dense layout parity and optional real Keras backend lab are clearly scoped.

**Gap / next step:** The promised cross-framework competence stops at Dense; add normalization/attention/processor semantics and a model-level parity exercise.

**Backlog:** [#83](https://github.com/mlnomadpy/jaxpathways/issues/83)

#### [deployment-08: Convert PyTorch weights to Flax and locate numerical errors](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/15-deployment/08-pytorch-to-flax-numerical-parity/lesson.json)

**Needs depth**. Actual PyTorch-to-NNX MLP conversion localizes epsilon errors and tests gradients.

**Gap / next step:** Add fused QKV/head-axis/tied-embedding conversion using the Transformer chapter; dense/LayerNorm parity is preparation for that missing bridge.

**Backlog:** [#83](https://github.com/mlnomadpy/jaxpathways/issues/83)

#### [deployment-03: Export a computation and verify its serving contract](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/15-deployment/03-export-and-serve-a-trained-computation/lesson.json)

**Usable draft**. Reloads serialized bytes and includes a fresh-process experiment and request checks.

**Gap / next step:** Add an optional polymorphic signature and version/platform rejection task before a serving project.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [deployment-05: Choose weight, activation and accumulation precision](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/15-deployment/05-weight-activation-and-accumulation-precision/lesson.json)

**Usable draft**. Per-channel scales, affine zero points, integer bias and output clipping are explained with independent arithmetic.

**Gap / next step:** Add weight-packing and a target-kernel/quality follow-on separately; the present simulation labeling is appropriate.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [deployment-07: Containerize a model service and verify its boundary](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/15-deployment/07-containerize-a-model-service/lesson.json)

**Needs repair**. Fresh-process schema/hash/request checks and Docker distinction are honest.

**Gap / next step:** The changed-artifact exercise solution only compares two digests; it never sends the altered artifact through the rejection boundary it asks to verify.

**Backlog:** [#79](https://github.com/mlnomadpy/jaxpathways/issues/79)

#### [deployment-04: Inference capacity, batching, and autoscaling](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/15-deployment/04-inference-capacity-batching-and-autoscaling/lesson.json)

**Usable draft**. Measured compute latency and hypothetical serial queues are separated with correct units.

**Gap / next step:** Add an executed bounded HTTP/load scenario in the connected service project before claiming autoscaling practice.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

#### [deployment-06: Deploy at the edge: conversion, budgets and device checks](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/15-deployment/06-edge-ai-conversion-and-device-validation/lesson.json)

**Needs depth**. Raw-input validation, request timing and optional LiteRT conversion are concrete.

**Gap / next step:** Device acceptance remains a report template and local CPU proxy. Add a worked target-runtime/delegate interpretation task using retained or newly authorized device evidence.

**Backlog:** Optional extension noted; no blocking issue found in this audit.

### Workload operations

**Chapter verdict: Needs repair and depth.** Real supervisors, process-isolated recovery and content-bound approval are good engineering foundations. Correct unrelated math, reduce repeated scaffolding and connect evaluations, rollback and retirement to observed model behavior.

#### [operations-01: Accelerator jobs and runtime lifecycle](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/01-accelerator-jobs-and-runtime-lifecycle/lesson.json)

**Needs depth**. A real supervisor tests completion, runtime mismatch, timeout and reaping.

**Gap / next step:** A 192-line embedded worker dominates a learner's first operations lesson; stage the lifecycle and provide common support code separately.

**Backlog:** [#89](https://github.com/mlnomadpy/jaxpathways/issues/89)

#### [operations-02: Health, logs, and workload observability](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/02-health-logs-and-workload-observability/lesson.json)

**Usable draft**. Structured logs, count/time throughput and checkpoint freshness have real worker evidence.

**Gap / next step:** Reduce repeated worker listing and add a log-to-diagnostic worksheet using a bounded stream.

**Backlog:** [#89](https://github.com/mlnomadpy/jaxpathways/issues/89)

#### [operations-03: Failure recovery and operational runbooks](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/03-failure-recovery-and-operational-runbooks/lesson.json)

**Usable draft**. Actual fresh-process recovery and next-update hashes substantiate the runbook.

**Gap / next step:** Add two distinct failure traces requiring different learner diagnoses rather than always the same fail-after update.

**Backlog:** [#89](https://github.com/mlnomadpy/jaxpathways/issues/89)

#### [operations-04: Changes, rollback, and artifact provenance](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/04-changes-rollback-and-artifact-provenance/lesson.json)

**Needs depth**. Content identity and failed-selection isolation are correctly scoped.

**Gap / next step:** Both accepted releases differ only in metadata, so rollback never demonstrates restored model behavior; add a changed-model consumer and observed output rollback.

**Backlog:** [#88](https://github.com/mlnomadpy/jaxpathways/issues/88), [#89](https://github.com/mlnomadpy/jaxpathways/issues/89)

#### [operations-05: Capacity, utilization, and operating cost](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/05-capacity-utilization-and-operating-cost/lesson.json)

**Usable draft**. Measures process lifetime and keeps hypothetical price/headroom assumptions explicit.

**Gap / next step:** Pair duration distributions with burst/retry cases; three tiny runs cannot establish tail capacity.

**Backlog:** [#89](https://github.com/mlnomadpy/jaxpathways/issues/89)

#### [operations-06: MLOps: data contracts, CI gates and monitoring](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/06-mlops-data-contracts-and-release-pipelines/lesson.json)

**Needs depth**. Group leakage, weighted slice gates and drift-versus-quality distinction are useful.

**Gap / next step:** The release inputs are supplied errors, not a connected evaluated model. Add a learner task computing slice evidence from reloaded predictions.

**Backlog:** [#87](https://github.com/mlnomadpy/jaxpathways/issues/87)

#### [operations-07: LLMOps: version prompts, evaluate behavior and trace requests](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/07-llmops-prompts-evaluation-and-traces/lesson.json)

**Needs repair**. Actual MLflow spans and missing-response/critical-case bookkeeping are useful.

**Gap / next step:** The guided formula claims P(reproduce)=1 from a container digest and is unrelated to LLMOps. Nine answerable fixtures are duplicates, and no prompt/model/retrieval stage is executed.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74), [#87](https://github.com/mlnomadpy/jaxpathways/issues/87)

#### [operations-08: ModelOps: own, approve, roll back and retire a model](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/08-modelops-ownership-approval-and-retirement/lesson.json)

**Needs repair**. Bundle-bound approval, expiry and side-effect-free rejection are concrete.

**Gap / next step:** The guided formula is unrelated cosine retrieval; retirement is in the title but not implemented or exercised. Add a retirement/revocation/rollback-consumer scenario.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74), [#88](https://github.com/mlnomadpy/jaxpathways/issues/88)

### Self-supervised pretraining: masked and contrastive learning

**Chapter verdict: Needs depth.** Masked targets, denominators, patch oracles and symmetric contrastive gradients explain bounded objectives honestly. Add guided saved-encoder transfer with held-out groups, varying masks and representation probes using existing projects.

#### [pretraining-01: Masked language modeling: predict hidden tokens](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/17-pretraining/01-mlm/lesson.json)

**Needs depth**. Clean/corrupted targets, selected-token denominator and logit-gradient audit are sound.

**Gap / next step:** The three repeated-token sequences and changed mask do not test document representation; connect the existing saved-encoder pilot with a learner-controlled mask/data/objective stage.

**Backlog:** [#84](https://github.com/mlnomadpy/jaxpathways/issues/84)

#### [pretraining-02: Masked image modeling: reconstruct missing patches](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/17-pretraining/02-mim/lesson.json)

**Needs depth**. Patch-order oracle, visible-only information and ambiguity counterexample are useful.

**Gap / next step:** A fixed visible-patch linear decoder is not a reusable mask-aware encoder. Add variable-mask transfer and an encoder/probe handoff using a saved small model.

**Backlog:** [#84](https://github.com/mlnomadpy/jaxpathways/issues/84)

#### [pretraining-03: Contrastive learning: views, positives and negatives](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/17-pretraining/03-contrastive/lesson.json)

**Needs depth**. Symmetric pairing, false negatives, temperature and encoder-gradient checks are substantive.

**Gap / next step:** Retrieval is checked only on four training pairs. Add held-out source groups, multiple-positive labels and a downstream frozen-encoder task.

**Backlog:** [#84](https://github.com/mlnomadpy/jaxpathways/issues/84)

### Post-training: SFT, LoRA, reward models and RLHF

**Chapter verdict: Needs depth.** Response masks, LoRA mechanics and preference-objective counterexamples are useful; failed PPO quality is honestly retained. Enforce adapter base identity and build a guided saved-decoder route with sequence log-probabilities, held-out prompts and retention.

#### [posttraining-01: Supervised fine-tuning with response-only token loss](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/18-posttraining/01-sft/lesson.json)

**Needs depth**. Response-target shifting and unequal-length gradient audit are well explained.

**Gap / next step:** The transition table cannot use earlier prompt context; turn the existing saved-decoder/causal-pilot handoff into a required transfer exercise.

**Backlog:** [#85](https://github.com/mlnomadpy/jaxpathways/issues/85)

#### [posttraining-02: LoRA: adapt, save and merge low-rank updates](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/18-posttraining/02-lora/lesson.json)

**Needs depth**. Real base fitting, first-gradient asymmetry, nonunit scaling and merge checks align.

**Gap / next step:** The saved adapter omits the base hash/module identity taught in prose and has no wrong-base rejection exercise.

**Backlog:** [#86](https://github.com/mlnomadpy/jaxpathways/issues/86)

#### [posttraining-03: Learn a reward model from pairwise preferences](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/18-posttraining/03-reward/lesson.json)

**Needs depth**. Stable pair loss, offset/scale identifiability and disagreement are explicit.

**Gap / next step:** All ranking evidence uses three training comparisons; add held-out prompts and disagreement slices linked to the saved reward-model stage.

**Backlog:** [#85](https://github.com/mlnomadpy/jaxpathways/issues/85)

#### [posttraining-04: RLHF mechanics: a frozen reward and PPO policy updates](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/18-posttraining/04-rlhf/lesson.json)

**Usable draft**. Actual sampled bandit PPO, frozen reward/reference and exact regularized optimum teach bounded mechanics.

**Gap / next step:** The saved-decoder appendix exists; add an explicit changed-condition handoff while preserving its failed quality result.

**Backlog:** [#85](https://github.com/mlnomadpy/jaxpathways/issues/85), [#91](https://github.com/mlnomadpy/jaxpathways/issues/91)

#### [posttraining-05: Direct preference optimization and reference-corrected margins](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/18-posttraining/05-dpo/lesson.json)

**Needs depth**. Reference cancellation, gradient and falling-chosen-probability counterexample are strong.

**Gap / next step:** Extend the action table to summed response-token likelihoods on unequal-length saved-decoder pairs with padding and prompt masks.

**Backlog:** [#85](https://github.com/mlnomadpy/jaxpathways/issues/85)

### TPU Setup: Provisioning & Runtime Verification

**Chapter verdict: Needs repair.** Provisioning alternatives and synchronized arithmetic receipts give a useful start. Separate requested-target readiness from CPU arithmetic success, qualify device/chip notation and align cleanup with provisioning ownership.

#### [tpu-01: Select a TPU slice and provision your Cloud TPU VM](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/19-tpu/01-get-a-tpu-running-and-verify-the-runtime/lesson.json)

**Needs repair**. Concrete provisioning alternatives, topology catalog and CPU arithmetic checks are present.

**Gap / next step:** An unqualified equation equates device count with chip count, and the first memory-fit task uses aggregate HBM without a placement counterexample. Main expected-backend check defaults to the backend actually observed.

**Backlog:** [#77](https://github.com/mlnomadpy/jaxpathways/issues/77), [#82](https://github.com/mlnomadpy/jaxpathways/issues/82), [#90](https://github.com/mlnomadpy/jaxpathways/issues/90)

#### [tpu-04: Bootstrap .venv and verify a TPU runtime receipt](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/19-tpu/02-bootstrap-venv-and-verify-tpu-runtime-receipt/lesson.json)

**Needs repair**. Closed-form arithmetic and saved runtime evidence are useful.

**Gap / next step:** verified=True checks arithmetic only and can be true on CPU; distinguish arithmetic pass from requested TPU/device readiness. The guided equation concerns optimizer memory, not runtime verification.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74), [#77](https://github.com/mlnomadpy/jaxpathways/issues/77)

### TPU Workflows: Launching Jobs & Checkpointed Experiments

**Chapter verdict: Needs repair.** Checkpoint/control hashes and the source-reviewed target matrix are valuable. Fix warmup/count math and preemption guarantees; teach staging as source-bound classification with explicit prerequisites.

#### [tpu-02: Stage and launch checkpointed TPU jobs](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/20-tpu-workflows/01-stage-and-launch-checkpointed-tpu-jobs/lesson.json)

**Needs repair**. Retains momentum/key/config and verifies a stopped-and-resumed trajectory.

**Gap / next step:** The objective promises preemption can never corrupt latest state from temp/fsync/replace, beyond the controlled single-writer local test; the steady-step equation has an inconsistent S-1 denominator.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74), [#77](https://github.com/mlnomadpy/jaxpathways/issues/77)

#### [tpu-05: Review scripts before TPU staging](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/20-tpu-workflows/02-run-course-scripts-and-multihost-tpu-slices/lesson.json)

**Needs repair**. Rejects unreviewed and CPU-only staging and distinguishes discovery from training.

**Gap / next step:** The global-gradient mean equation lacks equal-count/axis assumptions; topology exercise implies --worker=all is the only launch mechanism. Add a real classification task rather than catalog arithmetic alone.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74), [#77](https://github.com/mlnomadpy/jaxpathways/issues/77)

### TPU Systems: Generations, Memory, Precision & XProf

**Chapter verdict: Needs repair and depth.** Newer prose correctly separates output dtype, native quantized execution, training state and inference cache. Correct policy byte counts, prove per-device fit and use stable profiling inputs with a real timeline interpretation task.

#### [tpu-03: TPU generations and HBM memory budgeting](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/21-tpu-systems/01-tpu-generations-and-hbm-memory-budgeting/lesson.json)

**Needs depth**. Correctly distinguishes named training-state tensors, inference KV cache and aggregate fit limits in the newer prose.

**Gap / next step:** Exercises still ask aggregate fit calculations without making learners prove per-device placement. Add a replicated-versus-sharded ledger and inference/training-generation decision cases.

**Backlog:** [#82](https://github.com/mlnomadpy/jaxpathways/issues/82)

#### [tpu-06: bfloat16, int8 precision, and XProf profiling on TPU](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/21-tpu-systems/02-bfloat16-int8-precision-and-xprof-profiling/lesson.json)

**Needs repair**. Result dtype versus internal accumulation and dequantization versus native INT8 are explicitly distinguished.

**Gap / next step:** Every precision-policy row reports INT8 weight bytes even for BF16 operands. Repeated unnormalized dot profiling overflows for sufficiently many steps, and the guided formula is unrelated token cost.

**Backlog:** [#74](https://github.com/mlnomadpy/jaxpathways/issues/74), [#78](https://github.com/mlnomadpy/jaxpathways/issues/78), [#79](https://github.com/mlnomadpy/jaxpathways/issues/79), [#81](https://github.com/mlnomadpy/jaxpathways/issues/81)

## Reproduced teaching counterexamples

Retained evidence is intentionally small and contains no private data or weights:

- [Independent counterexamples](evidence/content-quality-audit-2026-10-27/counterexamples-20261027.json): uniform-query permutation invariance, discounted-value mismatch, precision byte arithmetic and repeated-dot growth.
- [Exact CPU function probe](evidence/content-quality-audit-2026-10-27/runtime-probe-20261027.json): JAX 0.9.2 on CPU. Only relevant imports/function definitions were extracted; no lesson training loop was executed. TPU readiness helpers return success for CPU arithmetic, all policy rows inherit INT8 byte counts, and the repeated BF16 dot becomes nonfinite at iteration five.
- Every lesson's SHA-256 and all chapter guide contents are retained in the machine-readable inventory. GitHub issue bodies are retained under the packet's `issues/` directory, with a creation manifest.

The [reproduction script](evidence/content-quality-audit-2026-10-27/reproduce.py) reruns only the small CPU counterexamples against content whose hashes match the audited source. Use the reviewed course Python environment with NumPy and JAX, then run:

```sh
JAX_PLATFORMS=cpu python docs/evidence/content-quality-audit-2026-10-27/reproduce.py --source-root /path/to/audited-checkout
```

The checkout must contain source revision `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. [The retained rerun](evidence/content-quality-audit-2026-10-27/verification/runtime-probe.json) passed on CPU with JAX 0.9.2. [Inventory validation](evidence/content-quality-audit-2026-10-27/validation.json) confirms all 104 lesson hashes against Git, all 22 chapter entries and all 18 issue mappings. [GitHub readback](evidence/content-quality-audit-2026-10-27/github-readback.json) confirms the created issues were visible as open issues with source and acceptance criteria.

The readiness observations do not imply that forcing `JAX_PLATFORMS=tpu` silently runs on CPU. The defect is ambiguous receipt/overall-contract semantics and missing expected-topology checks. The RL finding does not prohibit approximate critics; it asks that the value objective and mismatch be explicit. The profiling finding does not say the default four iterations overflow; the fifth does.

## Review recommendation

Review the P1 corrections first, then pilot one beginner route and one saved-model route with learners. Ask them to predict a changed result, interpret its figure and explain a failure without reading the solution. Preserve failed model-quality evidence and qualify every claim by its actual source, workload and target. Do not upgrade maturity merely because a script or schema check passes.
