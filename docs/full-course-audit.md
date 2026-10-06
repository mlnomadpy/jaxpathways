# Full course audit — 2026-10-05

> Current planning: [full curriculum audit — 2026-10-06](full-course-audit-2026-10-06.md) and [course expansion plan](course-roadmap-2026-10-06.md). This document preserves an earlier delivery snapshot.

> Historical baseline before the course completion work. Counts and open implementation items below describe the earlier audit, not the current course. See [course-completion-plan.md](course-completion-plan.md), [pathways-course-upgrade.md](pathways-course-upgrade.md), and regenerated [content-audit.md](content-audit.md) for current status.

The course has a substantial CPU foundation and useful numerical teaching examples. Its main weakness is the distance between those examples and independently completed, end-to-end learner work. More standalone lessons or more prose alone will not close that gap. The next milestone should be a complete, testable learner journey from setup to a trained, recovered, evaluated and deployed small model.

## Scope and evidence

Inspected all 83 curriculum entries, all 50 authored source bundles and current CPU hashes, all 10 routes, both implemented projects and their checkers, both synthesis drafts, the reader/navigation and tutor flows, artifact generation/validation, optional deployment-lab records, and CI workflows. Editorial review examined lesson objectives, sections, experiments and exercise progression, with closer source reading of training, recovery, Transformer, deployment and assessment material. The previous plot audit covers every authored figure.

This is a course-level audit, not a claim that every mathematical statement and every possible runtime input has been independently verified. Live browser and mobile layout, fresh-machine setup, external reference link availability, real accelerators/edge devices, and actual learner comprehension were not validated in this audit. CPU receipts are instructor-reference evidence; they are not learner assessments.

## Current state

- **50 of 83 entries are authored (60.2%); 33 remain planned.** Five specialist phases have no authored lessons at all.
- **Two implemented projects**: regression audit and classifier audit. Seventeen phase project descriptions do not represent seventeen implementations.
- **One of ten route capstones is marked authored.** The classifier project exists, but the broader models route capstone remains planned because its recovery/language-model outcome is not fully covered.
- **Two synthesis assessment drafts**, neither an independently validated credential. The classifier draft explicitly excludes the later recovery and Transformer capabilities.
- **All 50 current source hashes match CPU receipts.** Every authored lesson has a figure, explanation and executed notebook companion. This is a strength to retain.
- Authored lesson estimates total **3,710 minutes (61 hours 50 minutes)** before unspecified project/review time. These are author estimates, not measured learner completion times.

## Priority findings

### F01 · High — aggregate route coverage hides specialist gaps

The probability and science routes each contain 28 authored lessons out of 33, but every one of those authored lessons is shared foundation material: each specialist phase is 0/4. RL, kernels and operations are likewise wholly unwritten. The site already labels roadmap entries and drafts, and home cards explicitly say their counts include foundations. Those counts are accurate. The remaining problem is that learners must infer which specialist capabilities are available from the combined total; the specialist phase can still be entirely unwritten.

**Improve:** show shared-foundation coverage, specialist coverage, implemented project and assessment availability separately. Label unavailable outcomes at the point of route selection. Keep Careers discoverable while describing the actual deliverable currently attainable.

**Acceptance:** a learner choosing science can see “foundation available; scientific lessons and project planned” before beginning, without interpreting a large shared lesson count as a mostly finished science course.

**Evidence:** `learning-paths/science.json`, `learning-paths/probability.json`, `curriculum/course.json`, `src/data/home.json`, `src/components/home/WorkDirections.astro`.

### F02 · High — the route sequence is broader than the learner needs for a first result

Every route inherits full phases. The foundations route includes 28 authored lessons and 29 hours 50 minutes of estimated work, including all twelve math/optimization lessons and the early logical-device lab. There is no first-class distinction between essential, optional enrichment, hardware-specific and specialist prerequisites. Several prerequisite phases contain planned entries, while the reader simply skips planned lessons in authored navigation. There are no direct authored-to-planned lesson prerequisites, so this is an ambiguity in the phase contract, not a demonstrated hard navigation block.

**Improve:** define a short CPU start-to-regression route with explicit lesson-level prerequisites. Offer geometry, Jacobians, conditioning and advanced optimizer material at relevant decision points, while preserving a complete math-study path. Make logical-device/TPU work optional for CPU beginners. Validate skipped optional material versus genuinely missing prerequisites.

**Acceptance:** a new learner reaches one independently checked regression artifact on a clearly specified essential route; deeper lessons remain easy to discover without being an unexplained gate.

**Evidence:** `learning-paths/foundations.json`, `curriculum/course.json`, `src/scripts/reader.js` (authored filtering and prerequisite display), `skills/create-jax-course/references/pedagogy-and-ux.md`.

### F03 · High — integration projects do not cover the advertised later outcomes

The available projects end at regression training and a synthetic classifier audit. Recovery, Transformers, profiling, sharding, deployment and operations have project descriptions or worked examples, but no corresponding staged learner project with starter, tests and review rubric. The models assessment explicitly covers the classifier only. The final “View the phase project” reader action currently returns to the phase catalog; it does not necessarily open an implemented project.

**Improve:** build one continuing model project with stages for frozen data, training, evaluation, full-state recovery, export, precision and deployment. Reuse the same artifact across lessons. Make phase transitions point to the actual project stage or clearly identify a planned project.

**Acceptance:** a learner-owned implementation survives a changed fixture, interruption and restore, exported prediction check and documented deployment test; each stage has inspectable evidence and a diagnostic failure.

**Evidence:** `curriculum/projects.json`, both `projects/*/project.json`, `learning-paths/models.json`, `src/scripts/reader.js` (phase-project action).

### F04 · High — synthetic correctness fixtures need a bridge to realistic modeling

XOR corners, a deterministic linear sign rule, synthetic image stripes, and the three-character periodic language model are excellent small correctness fixtures. They are explicitly scoped and should remain. They do not teach a complete real-data workflow: provenance, acquisition, split strategy, train-only preprocessing, imperfect labels, baseline comparison, task metrics and error analysis. The periodic language model even repeats contexts between train and held-out strings; its text correctly acknowledges that limitation.

**Improve:** add one small, reproducibly packaged real-data classifier and one small corpus extension with a simple baseline. Start from the controlled fixture, then deliberately introduce the problems it avoided. Establish the dataset license/provenance and fixed split before training. Avoid selecting the held-out set to make scores look good.

**Acceptance:** retain dataset identity, baseline, frozen splits, preprocessing state, per-example errors and results across multiple seeds. Explain why an improvement matters for the task.

**Evidence:** `networks-01`, `networks-03`, `networks-04`, `transformers-04`, `projects/mlp-classifier/tests/check.py`.

### F05 · High — practice and assessment coverage are weaker than the lesson count suggests

Nine lessons repeat their main exercise verbatim as a practice prompt: all five network lessons and all four recovery lessons. Each authored lesson has one multiple-choice checkpoint; 34 correct answers are in the second position and 16 in the first, with none later. The same explanation is returned regardless of the selected distractor. Public worked scripts and notebook cells include solutions. Those are useful references, but they cannot establish independent mastery. The old regression project and foundations synthesis do not substantially assess the eight added math lessons (regularization, likelihood, gradient noise, Jacobians and optimizer decisions).

**Improve:** replace duplicate tasks with changed-condition transfer; add prediction, implement, diagnose and explain checkpoints at meaningful boundaries; use misconception-specific feedback. Separate learner starter notebooks from executed instructor notebooks. Extend the math synthesis and add recovery/deployment assessments. Assess source reasoning and changed cases, rather than treating option-order randomization as a quality fix.

**Acceptance:** each core capability maps to a genuinely distinct learner task and rubric; a learner cannot satisfy transfer merely by rerunning the shown solution. Track attempts, executable evidence and review separately, preserving the existing honest progress labels.

**Evidence:** canonical lesson exercise/practice fields; `src/scripts/reader.js` checkpoint handler; `scripts/lesson-artifacts.mjs`; `projects/regression-audit/project.json`; `assessments/foundations.md`.

### F06 · High — recovery, performance and edge claims need end-to-end evidence

The course has real CPU recovery checks and optional real-framework/LiteRT conversion labs. The two optional lab file hashes match their recorded receipt. However, asynchronous failure boundaries, profiling, memory tradeoffs, communication and multi-host recovery remain planned. The edge lesson’s core latency figure measures a local CPU request proxy; its optional conversion record explicitly says no target device was validated. Do not erase those distinctions by describing all of deployment or scale-out as complete.

**Improve:** prioritize a persistent recovery project, one trace-backed performance lesson, and one actual device deployment report. Separate CPU illustration, desktop runtime conversion and device measurements in the learner’s evidence. Extend precision work from small dense parity to activation calibration and held-out task quality for the chosen trained artifact.

**Acceptance:** a saved profiler trace supports the claimed bottleneck; a separate-process failure drill tests recovery; a named target-device run records runtime/delegate, output parity, cold/warm latency and memory. If hardware is unavailable, keep that stage clearly unvalidated.

**Evidence:** planned `recovery-05`, `performance-03`–`05`, `distributed-03`–`04`; `docs/deployment-lab-validation.json`; `deployment-05` and `deployment-06` sources/labs.

### F07 · Medium — execution is reproducible for the author but still demanding for a beginner

The beginner ZIP, explicit OS commands and setup help are useful. Later practice asks the learner to manage a checkout, pinned Python environment, Jupyter installation, reference notebooks and sometimes a separate framework environment. There is no evidence of fresh novice setup tests on each OS. Four deployment lessons also have only one experiment and one practice task with no incremental buildSteps; this is a signal to review scaffolding, not proof of poor teaching based on counts.

**Improve:** provide a starter workspace with one documented run/check command per stage, retain an offline path, and test fresh-machine setup on the supported OS/runtime combinations. Offer an optional hosted notebook launch only after its environment and state/restart behavior are validated. Separate read/setup/run/practice effort estimates using observed learner sessions.

**Acceptance:** a novice can install, edit, execute and interpret one changed result without an instructor repairing the environment silently. Record the blockers and support instructions.

**Evidence:** `src/lib/practice-content.js`, `requirements-cpu.txt`, `welcome-01`, deployment source structure. Timing estimates are not calibrated by observed learners.

### F08 · Medium — the tutor promise has two substantially different access paths

The skill-enabled-agent path has a workspace, four skills and a persistent `LEARNING.md` plan. A browser chatbot path is present, but it is inside a disclosure and starts by attaching the first lesson. Browser progress and tutor files do not synchronize; that is correctly documented. The npx command installs skills into compatible agents, not every chatbot. No automated learner-dialogue evaluation demonstrates that tutors consistently avoid solution leakage or detect misconceptions.

**Improve:** give browser chat and coding-agent learners equally clear entry choices. Export a portable tutor packet with lesson context, learner attempt and progress, plus an explicit reconciliation step. Test the tutor on scripted wrong answers, broken code, planned lessons, resume, and requests to reveal solutions early.

**Acceptance:** both paths can start, resume and change routes without silently losing progress; tutor claims match current source/receipt status and preserve learner-owned attempts.

**Evidence:** `src/components/home/TutorStart.astro`, `skills/start-learning/SKILL.md`, `skills/learn-jax/SKILL.md`, `src/lib/learner-records.js`.

### F09 · Medium — lesson presentation can still overload the learner

The reader supports outlines, saved positions, optional setup and collapsed assembled code. However, it presents objectives, problem, idea, all conceptual sections, a long figure walkthrough, then the build/run steps. Some learners must absorb a large explanation before their first action. Nine duplicated exercise prompts add length without new learning. Assessment Markdown uses code-style math instead of the shared KaTeX narrative renderer, so the math experience is inconsistent outside lessons. These are source-level findings; a live usability test is still required.

**Improve:** keep one immediate learner question and first action near the top, interleave each figure/explanation with its corresponding code step, and preserve a full-reading mode. Use explicit “predict → run → explain → change” milestones with buttons and links rather than more forms. Provide short symbol definitions and misconception feedback. Extend mathematical typography to assessment prose while keeping code literal.

**Acceptance:** test keyboard and narrow-screen use, screen-reader figure text, equation overflow and return-to-place behavior with real users; measure whether a learner can locate the next action and explain the plot before copying code.

**Evidence:** `src/lib/lesson-template.js`, `src/lib/lesson-content.js`, `src/lib/assessment-content.js`, both assessment Markdown files.

### F10 · Medium — quality automation verifies structure better than teaching outcomes

The structural audit marks a lesson “developed draft” when prescribed fields exist. Its report hardcodes zero authored pathway exams, while two review drafts exist separately; those are different categories but the summary does not inventory the distinction. Historical quality prose and recommendations retain earlier coverage numbers and now-completed work. Tests enforce artifacts and bounded numerical behavior, not learning transfer. The CPU workflow runs build/check before fresh smoke execution and only invokes the regression project through `test:project`; the classifier checker is not included in that command. Optional framework receipts are not automatically refreshed by the core workflow.

**Improve:** derive a current course-quality dashboard from manifests and record draft/reviewed assessment states separately. Move old counts into a clearly historical section. Run both project checkers in CI, and follow fresh execution with regeneration and fidelity checks when validating new output artifacts. Add an explicit optional-backend matrix and retain independent reviewer/learner records separately from machine passes.

**Acceptance:** changing a lesson or optional lab invalidates the appropriate evidence; both project references pass and broken starters fail as intended; displayed current coverage comes from manifests, not hand-maintained historical prose. A passing build must not be labeled pedagogical approval.

**Evidence:** `scripts/audit-content.py`, `docs/content-quality.md`, `.github/workflows/course-check.yml`, `package.json`, `scripts/smoke-lessons.py`, optional deployment receipt.

## Coverage by phase

Numbers count authored source availability only. A full row does not imply that its learning outcome has an assessed integration project.

| Phase | Authored / planned total | Main improvement | Evidence required |
|---|---:|---|---|
| Setup & first steps | 3/4 | Make a CPU-only first success an explicit complete path; keep logical devices and the unfinished TPU bridge optional. | Fresh-machine walkthrough on each supported OS; first independent edit and explanation. |
| Arrays & pure functions | 4/4 | Connect isolated transformations into one reusable preprocessing module, retaining training-only statistics. | Unequal axes, constant feature, unseen batch and malformed input checked by a learner-owned script. |
| JAX transformations | 5/5 | Use one running problem to compose grad, vmap and jit, then ask for a transfer without showing the solution. | Independent vectorized/loop and eager/compiled comparisons with a changed batch and objective. |
| State, randomness & control flow | 4/4 | Combine keys, pytree state, scan and branching in one small persistent simulator. | A restart from a saved state reproduces the next random event and transition. |
| Math & optimization | 12/12 | Retain the mathematical depth but separate essential training prerequisites from optional deeper study; expand the synthesis assessment to match the eight added lessons. | A learner justifies validation, regularization, batch aggregation and optimizer-state decisions on a changed multifeature problem. |
| Neural network training | 5/5 | Move from synthetic shape/correctness fixtures to a fixed, documented real dataset with a baseline and failure analysis. | Dataset provenance, frozen splits, train-only preprocessing, baseline, error examples and multiseed report. |
| Data & checkpoint recovery | 4/5 | Make recovery a runnable staged project; complete asynchronous/failure-boundary instruction. | Interrupt a separate process after known boundaries, resume, compare state and data position, and reject incomplete snapshots. |
| Transformers & language models | 4/4 | Keep the periodic fixture as a wiring check, then add a small corpus project that needs contextual prediction. | Documented corpus split, simple baseline, eligible-token loss, persistent checkpoint, generation policy and failure examples. |
| Performance diagnosis | 2/5 | Complete profiling, memory and before/after measurement rather than stopping at timers and trace counters. | A saved trace with annotated bottleneck, correctness-preserving change, repeated measurements and named hardware. |
| Distributed training | 2/4 | Complete communication and recovery; provide actual multi-device evidence before promising scaling outcomes. | Global numerical reference, placement explanation, communication evidence and a measured scaling/recovery report. |
| Scientific computing | 0/4 | Author the first complete CPU inverse-problem sequence; all four specialist lessons are missing. | Known parameter recovery, conservation/stability checks, independent gradients and unseen trajectories. |
| Probabilistic modeling | 0/4 | Author a Bayesian regression sequence and diagnostics; all four specialist lessons are missing. | Prior/posterior predictive checks, chain diagnostics and uncertainty interpretation. |
| Reinforcement learning | 0/4 | Author environment, rollout, policy-gradient and evaluation lessons as one project; all four are missing. | Known transition tests, baseline policy and evaluated returns across independent seeds. |
| Pallas kernels | 0/4 | Keep the route a roadmap until kernel lessons and supported accelerator evidence exist. | Reference parity across shapes/dtypes followed by reproducible timing on named hardware. |
| Autodiff & JAX internals | 1/4 | Build out JVP/VJP, custom derivatives and lowering from the existing jaxpr lesson; reuse rather than duplicate the math-phase introduction. | A custom rule checked independently, including its domain boundaries and a changed input. |
| Deployment, interoperability & edge AI | 4/6 | Connect the framework and precision labs to a trained artifact, executable serving workload and real device evaluation. | Cross-runtime parity, calibration/accuracy report, supported operators, cold/warm latency, memory and target-device record. |
| Workload operations | 0/5 | Author one operational vertical slice before presenting five separate job capabilities; all five lessons are missing. | Versioned job, telemetry, controlled incident, verified recovery/rollback and measured resource report. |

## Route coverage and attainable outcomes

The hours below sum declared minutes for authored lessons, excluding project/review time and missing material. Shared lessons appear in multiple routes; these hours must not be added together as unique course duration. No route with a planned capstone should be described as fully completed just because its available lessons were read.

| Route | Authored / total | Declared available hours | Route capstone | Missing lesson IDs |
|---|---:|---:|---|---|
| Start with JAX | 28/29 | 29.8 | authored | welcome-03 |
| Understand JAX itself | 29/33 | 31.3 | planned | welcome-03, internals-02, internals-03, internals-04 |
| Train neural networks | 41/43 | 49.8 | planned | welcome-03, recovery-05 |
| Operate JAX workloads | 41/53 | 48.8 | planned | welcome-03, recovery-05, performance-03, performance-04, performance-05, distributed-03, distributed-04, operations-01, operations-02, operations-03, operations-04, operations-05 |
| Model uncertainty | 28/33 | 29.8 | planned | welcome-03, probability-01, probability-02, probability-03, probability-04 |
| Learn by interacting | 33/38 | 37.4 | planned | welcome-03, rl-01, rl-02, rl-03, rl-04 |
| Make it fast. Make it scale. | 41/52 | 48.8 | planned | welcome-03, recovery-05, performance-03, performance-04, performance-05, distributed-03, distributed-04, kernels-01, kernels-02, kernels-03, kernels-04 |
| Simulate and discover | 28/33 | 29.8 | planned | welcome-03, science-01, science-02, science-03, science-04 |
| Ship and adapt JAX workloads | 43/50 | 51.0 | planned | welcome-03, recovery-05, performance-03, performance-04, performance-05, deployment-02, deployment-04 |
| Build a TPU training system | 43/48 | 52.5 | planned | welcome-03, recovery-05, performance-03, performance-04, performance-05 |

## Recommended implementation order

1. **Make the current offer clear.** Split foundation/specialist coverage, mark optional hardware work, point project transitions at actual projects, replace the nine duplicate prompts, and refresh the audit summary. Completion: route and project labels agree with manifests, and each changed task has distinct reasoning.
2. **Finish one CPU learner journey.** Define essential lesson-level prerequisites; ship learner starters and a running real-data classifier project that connects preprocessing, model, evaluation and saved artifacts. Completion: a learner can reproduce a changed fixture and explain baseline/error results without using the reference implementation.
3. **Make recovery and assessment real.** Extend that project to persistent full-state checkpoints and a separate-process interruption drill; expand math and classifier synthesis to assess the promised capabilities. Completion: changed-condition evidence and a review rubric cover each outcome, not just the original fixture.
4. **Connect training to deployment.** Carry the same trained model through export, framework parity and precision/calibration into one measured runtime/device. Add profiling and a retained before/after report. Completion: numerical quality and resource measurements refer to the same versioned artifact and workload.
5. **Expand one specialist route at a time.** Finish its lessons, project, figures and assessment as a unit. Start a science/probability/RL/kernel/operations route only when its end artifact and validation resources are defined. Keep unsupported hardware stages planned.
6. **Validate with learners and an independent technical reviewer.** Run a small novice and intermediate pilot; record setup blockers, incorrect explanations, unaided transfer, completion time and rubric disagreements. Use this evidence to revise the sequence and pacing rather than inventing quality scores.

This is a prioritized backlog, not a promise of calendar duration. Hardware work, reviewer access and learner recruitment are dependencies to resolve before scheduling those stages.

## Every authored lesson: next improvement

These are editorial recommendations, not claims that each current example is numerically broken. The existing fixtures and independent checks should be retained while adding the missing transfer or integration.

| Lesson | Specific next improvement |
|---|---|
| [welcome-01: Set up your learning workspace](../phases/00-welcome/01-set-up-your-learning-workspace/lesson.json) | Observe a novice installing and editing the first script; report OS-specific failures and time to first independent result. |
| [welcome-02: Meet your arrays and devices](../phases/00-welcome/02-meet-your-arrays-and-devices/lesson.json) | Add a no-code-before-prediction axis exercise with an unfamiliar rectangular array. |
| [welcome-cpu: Practice with four virtual CPU devices](../phases/00-welcome/03-virtual-cpu-devices/lesson.json) | Offer as an optional placement lab rather than a required early detour for CPU model learners. |
| [arrays-01: From NumPy to jax.numpy](../phases/01-arrays/01-from-numpy-to-jax-numpy/lesson.json) | Continue the same preprocessing artifact into later array lessons instead of resetting the example. |
| [arrays-02: Shapes, broadcasting, and dtypes](../phases/01-arrays/02-shapes-broadcasting-and-dtypes/lesson.json) | Turn the shape and dtype cases into an independent preprocessing contract exercise. |
| [arrays-03: Pure functions and explicit inputs](../phases/01-arrays/03-pure-functions-and-explicit-inputs/lesson.json) | Carry the explicit function interface into the next project and test caller-state preservation. |
| [arrays-04: Immutable updates and indexing](../phases/01-arrays/04-immutable-updates-and-indexing/lesson.json) | Add a small debugging task combining masking, update semantics and invalid-index policy. |
| [first-gradient: Your first gradient](../phases/02-transforms/01-your-first-gradient/lesson.json) | Add a short placement check and a learner-generated explanation of a slope on a new curve. |
| [transforms-02: Losses and value_and_grad](../phases/02-transforms/02-losses-and-value-and-grad/lesson.json) | Reuse the regression fixture across transformations and later optimization to reduce repeated introductions. |
| [transforms-03: Batch a function with vmap](../phases/02-transforms/03-batch-a-function-with-vmap/lesson.json) | Add a composed transformation task whose axes differ from the worked example. |
| [transforms-04: Compile a function with jit](../phases/02-transforms/04-compile-a-function-with-jit/lesson.json) | Tie timing practice to a saved report with a measurement boundary and repeated samples. |
| [transforms-05: Tracing, static arguments, and recompilation](../phases/02-transforms/05-tracing-static-arguments-and-recompilation/lesson.json) | Turn trace diagnosis into a small executable bug-and-repair lab with retained output. |
| [state-01: Random keys without surprises](../phases/03-state/01-random-keys-without-surprises/lesson.json) | Use learner-authored random-state continuation in the phase integration artifact. |
| [state-02: Pytrees and structured parameters](../phases/03-state/02-pytrees-and-structured-parameters/lesson.json) | Extend structured parameters into the same running model rather than another isolated tree. |
| [state-03: Compiled loops with lax.scan](../phases/03-state/03-compiled-loops-with-lax-scan/lesson.json) | Join scan history and saved carry to the phase simulator and test a resumed suffix. |
| [state-04: Branches with lax.cond](../phases/03-state/04-branches-with-lax-cond/lesson.json) | Combine branching with that simulator and check behavior around a changed boundary. |
| [optimization-05: Vectors, norms, and projections](../phases/04-optimization/05-vectors-norms-and-projections/lesson.json) | Offer geometry as a just-in-time bridge with a new-vector reasoning check. |
| [optimization-01: Linear algebra and loss intuition](../phases/04-optimization/01-linear-algebra-and-loss-intuition/lesson.json) | Connect residual accounting to the project data contract and learner-written validation. |
| [optimization-06: Least squares, rank, and conditioning](../phases/04-optimization/06-least-squares-rank-and-conditioning/lesson.json) | Assess changed-rank and ill-conditioned examples in the synthesis, with prediction error versus parameter sensitivity. |
| [optimization-07: The chain rule, Jacobians, and directions](../phases/04-optimization/07-chain-rule-jacobians-and-directional-derivatives/lesson.json) | Require a directional sensitivity explanation in synthesis; align with the future internals JVP/VJP lesson. |
| [optimization-02: Gradient checking and numerical accuracy](../phases/04-optimization/02-gradient-checking-and-numerical-accuracy/lesson.json) | Have the learner produce and explain their own finite-difference error plot on a nonquadratic function. |
| [optimization-08: Curvature, Hessians, and learning rates](../phases/04-optimization/08-curvature-hessians-and-learning-rates/lesson.json) | Add curvature-based rate justification to the project decision report. |
| [optimization-03: Write gradient descent yourself](../phases/04-optimization/03-write-gradient-descent-yourself/lesson.json) | Make a learner diagnose a changed objective before revealing the stable rate. |
| [optimization-09: Probability, likelihood, and stable losses](../phases/04-optimization/09-probability-likelihood-and-stable-losses/lesson.json) | Extend the assessment beyond regression to a stable classification likelihood and its assumptions. |
| [optimization-10: Regularization and honest validation](../phases/04-optimization/10-regularization-and-validation/lesson.json) | Make validation-based model selection a full project stage with an untouched final evaluation set. |
| [optimization-11: Minibatches, expectation, and gradient noise](../phases/04-optimization/11-minibatches-expectation-and-gradient-noise/lesson.json) | Assess unequal microbatch aggregation and a deliberately biased sampler in the project. |
| [optimization-04: Optimize with Optax](../phases/04-optimization/04-optimize-with-optax/lesson.json) | Use the project checkpoint to test a real resumed optimizer transition. |
| [optimization-12: Adam, clipping, and learning-rate schedules](../phases/04-optimization/12-adam-clipping-and-learning-rate-schedules/lesson.json) | Assess clipping order, schedule-clock recovery and optimizer comparison at a fixed budget; these exceed the old regression rubric. |
| [networks-01: Build a tiny multilayer perceptron](../phases/05-networks/01-build-a-tiny-multilayer-perceptron/lesson.json) | Replace the duplicated main/practice prompt with a new transfer task; retain XOR as a numerical fixture. |
| [networks-02: Model and state with Flax NNX](../phases/05-networks/02-model-and-state-with-flax-nnx/lesson.json) | Replace duplicate practice and add a concrete nontrainable-state task in the running model. |
| [networks-03: A compiled train and evaluation step](../phases/05-networks/03-a-compiled-train-and-evaluation-step/lesson.json) | Replace duplicate practice and connect train/eval contracts to a real frozen dataset split. |
| [networks-04: Train a CNN on a small dataset](../phases/05-networks/04-train-a-cnn-on-a-small-dataset/lesson.json) | Replace duplicate practice; add real image error analysis and a simple non-neural baseline. |
| [networks-05: Debug unstable learning](../phases/05-networks/05-debug-unstable-learning/lesson.json) | Replace duplicate practice; diagnose a failure in the actual NNX model rather than only a scalar quadratic. |
| [recovery-01: Design an input pipeline](../phases/06-recovery/01-design-an-input-pipeline/lesson.json) | Replace duplicate practice; add file-backed data validation and a reproducible dataset fingerprint. |
| [recovery-02: Load, batch, and checkpoint order with Grain](../phases/06-recovery/02-load-batch-and-prefetch-with-grain/lesson.json) | Replace duplicate practice; persist and resume an iterator in a separate process. |
| [recovery-03: Save model, optimizer, and random state](../phases/06-recovery/03-save-model-optimizer-and-random-state/lesson.json) | Replace duplicate practice; supply a persistent checkpoint workspace and partial-write failure exercise. |
| [recovery-04: Recover data position and resume](../phases/06-recovery/04-recover-data-position-and-resume/lesson.json) | Replace duplicate practice and turn the existing full-state example into a staged recovery project. |
| [transformers-01: Attention from small pieces](../phases/07-transformers/01-attention-from-small-pieces/lesson.json) | Add a transfer from one head to multiple heads with independently checked shape and normalization contracts. |
| [transformers-02: Masking, tokenization, and sequence packing](../phases/07-transformers/02-masking-tokenization-and-sequence-packing/lesson.json) | Connect padding/packing to an actual eligible-token training-loss reduction, rather than only mask illustrations. |
| [transformers-03: A Transformer block and mixed precision](../phases/07-transformers/03-a-transformer-block-and-mixed-precision/lesson.json) | Use varied input rows in a second experiment and carry the precision policy into a complete training run. |
| [transformers-04: Train, checkpoint, and generate](../phases/07-transformers/04-train-checkpoint-and-generate/lesson.json) | Provide a learner starter with persistent artifacts and a small corpus baseline beyond the periodic three-token fixture. |
| [performance-01: Benchmark asynchronous work correctly](../phases/08-performance/01-benchmark-asynchronous-work-correctly/lesson.json) | Create a retained benchmark report and repeatability protocol, then use it in the profiling project. |
| [performance-02: Diagnose recompilation and host synchronization](../phases/08-performance/02-diagnose-recompilation-and-host-synchronization/lesson.json) | Connect Python trace counts to an actual profiler view without equating tracing with compilation. |
| [distributed-01: Arrays, meshes, and sharding](../phases/09-distributed/01-arrays-meshes-and-sharding/lesson.json) | Add a supported physical-device reproduction and separate it from logical-CPU placement evidence. |
| [distributed-02: A sharded training step](../phases/09-distributed/02-a-sharded-training-step/lesson.json) | Move from a single checked update to a resumable multi-step project with measured communication. |
| [internals-01: Read your first jaxpr](../phases/14-internals/01-read-your-first-jaxpr/lesson.json) | Make this material discoverable where the foundations assessment requires make_jaxpr; author the remaining internals sequence. |
| [deployment-01: Move models between JAX, Keras, TensorFlow and PyTorch](../phases/15-deployment/01-keras-and-pytorch-bridges-to-explicit-jax/lesson.json) | Add a multi-layer, nontrainable-state transfer task and a reproducible environment matrix for the optional backends. |
| [deployment-03: Export a computation and verify its serving contract](../phases/15-deployment/03-export-and-serve-a-trained-computation/lesson.json) | Wrap the exported artifact in an actual request handler with signature, malformed-input and version checks. |
| [deployment-05: Choose weight, activation and accumulation precision](../phases/15-deployment/05-weight-activation-and-accumulation-precision/lesson.json) | Add activation calibration and held-out quality comparisons to the running trained model; separate simulated and native kernel evidence. |
| [deployment-06: Deploy at the edge: conversion, budgets and device checks](../phases/15-deployment/06-edge-ai-conversion-and-device-validation/lesson.json) | Run the converted model on a named real target device and retain conversion, correctness and resource measurements. |

## Every planned lesson

These entries have no authored lesson body. The corresponding phase acceptance criteria above define the required project evidence; the earlier visual audit specifies their figures. No new body or execution was fabricated for this report.

| Phase | Missing lesson |
|---|---|
| welcome | welcome-03: Move your experiment to a TPU |
| recovery | recovery-05: Asynchronous checkpoints and failure boundaries |
| performance | performance-03: Read an XProf trace |
| performance | performance-04: Reason with HLO and the roofline model |
| performance | performance-05: Memory, rematerialization, and optimization tradeoffs |
| distributed | distributed-03: Communication-efficient algorithms |
| distributed | distributed-04: Resilient distributed training |
| science | science-01: A functional physical simulation |
| science | science-02: Vectorize trajectories and scan time |
| science | science-03: Differentiate through a solver |
| science | science-04: Fit parameters and scale the simulation |
| probability | probability-01: Distributions and sampling |
| probability | probability-02: Build a Bayesian regression |
| probability | probability-03: Hamiltonian Monte Carlo and sampler diagnostics |
| probability | probability-04: Variational inference and model checking |
| rl | rl-01: Write a functional environment |
| rl | rl-02: Batch environments and scan rollouts |
| rl | rl-03: Policy gradients and PPO |
| rl | rl-04: Evaluate agents across seeds |
| kernels | kernels-01: Pallas grids and BlockSpecs |
| kernels | kernels-02: A first TPU kernel |
| kernels | kernels-03: Tiling, memory, and pipelining |
| kernels | kernels-04: Iterate with correctness and performance evidence |
| internals | internals-02: JVPs, VJPs, and higher-order derivatives |
| internals | internals-03: Custom derivative rules |
| internals | internals-04: Lowering, compilation, and a tiny transformation |
| deployment | deployment-02: Adapt a pretrained model and choose a post-training objective |
| deployment | deployment-04: Inference capacity, batching, and autoscaling |
| operations | operations-01: Accelerator jobs and runtime lifecycle |
| operations | operations-02: Health, logs, and workload observability |
| operations | operations-03: Failure recovery and operational runbooks |
| operations | operations-04: Changes, rollback, and artifact provenance |
| operations | operations-05: Capacity, utilization, and operating cost |

## Validation record

Current source hashes were compared with all 50 stored CPU records; all match. The previous turn freshly ran all 50 scripts and 50 notebook companions. This audit does not relabel that evidence as a new 100-run experiment. Both optional deployment-lab source hashes also match their local validation receipt; no backend or target-device rerun was performed here.

Fresh audit checks and project results are recorded below after completion. Live browser/mobile layout, external links, clean OS installations, accelerator/edge performance, independent review and learner comprehension remain outside the verified evidence.

Fresh checks completed during this audit:

- `npm run build`: passed; 13 static Astro pages and distribution artifacts regenerated.
- `npm run check`: passed; 51 automated tests, Astro/lint/format checks, curriculum validation, EPUB/bundle fidelity and 729 local-reference checks.
- `npm run test:project`: regression reference passed all four stages.
- `python3 projects/mlp-classifier/tests/check.py --stage 3 --implementation solution`: classifier reference passed all three cumulative stages.
- Audit inventory check: 83 unique entries, 17 phases, 10 routes, current hashes for all 50 authored lessons, and valid local report links.

No curriculum, exercise, project or UI remediation was implemented as part of this audit. The new deliverables are this report and `full-course-audit.json`; the recommendations above remain a backlog.
