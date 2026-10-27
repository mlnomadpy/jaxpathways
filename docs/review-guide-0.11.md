# Review the 0.11.0 JAX Pathways candidate

Candidate edition: **0.11.0**. This is an author review packet, not an acceptance report. Keep the candidate pending until a reviewer records findings against the exact source revision.

Use the local preview at [127.0.0.1:4187](http://127.0.0.1:4187) when it is running. The representative page links below use its lesson routes; these are candidate preview URLs, not published pages. The canonical lesson JSON and project sources are linked too, so a reviewer can compare the rendered task with its implementation contract.

The course contains **104 lessons across 22 phases**. Its canonical lessons now define **303 learner starter contracts** across lesson exercises and practices. A starter contract is an authored prompt for independent learner work; it is not automatically graded merely because it exists. In the refreshed candidate, check that learner-only downloads contain these contracts separately from the worked reference scripts and solution reveals. The generated Python and notebook companions have been refreshed and checked against the authored contracts.

## Verification available for review

Local checks passed: 148 automated tests; all 104 lesson scripts and their 104 notebook executions; all 31 registered project references; 11 assessment references; 26 desktop/mobile browser tests; and root plus GitHub Pages subpath checks for pages, links, maturity labels, SEO and structured data. The optional NumPyro sampler has a separate CPU receipt. These checks do not replace independent mathematical, teaching or native-reader acceptance. Detailed retained results are in [the verification record](evidence/course-review-0.11/verification.json). No 0.11 hosted CI result or production deployment is claimed.

## Review in this order

| Review stop | Lesson and permanent page | Canonical source | What to test |
| --- | --- | --- | --- |
| Arrays | [Shapes, broadcasting, and dtypes](http://127.0.0.1:4187/lesson-arrays-02.html) | [`phases/01-arrays/02-shapes-broadcasting-and-dtypes/lesson.json`](../phases/01-arrays/02-shapes-broadcasting-and-dtypes/lesson.json) | Predict row-offset shape and values before execution. Then compare the learner starter with the worked reference. |
| Transformations | [Batch a function with `vmap`](http://127.0.0.1:4187/lesson-transforms-03.html) | [`phases/02-transforms/03-batch-a-function-with-vmap/lesson.json`](../phases/02-transforms/03-batch-a-function-with-vmap/lesson.json) | Check one per-example gradient against a hand calculation and the batch mean against the mean of individual gradients. |
| Optimization | [Probability, likelihood, and stable losses](http://127.0.0.1:4187/lesson-optimization-09.html) | [`phases/04-optimization/09-probability-likelihood-and-stable-losses/lesson.json`](../phases/04-optimization/09-probability-likelihood-and-stable-losses/lesson.json) | Check finite binary losses, explicit reduction, and whether the changed probability case tests the stated reasoning. |
| Reinforcement learning | [Learn value functions and generalized advantage estimation](http://127.0.0.1:4187/lesson-rl-05.html) | [`phases/12-rl/05-learn-values-and-gae/lesson.json`](../phases/12-rl/05-learn-values-and-gae/lesson.json) | Run the learned tabular critic, inspect its retained masked-MSE history, and hand-check reverse GAE at terminal and time-limit boundaries. |
| RL project | [Policy evaluation project](../projects/policy-evaluation/README.md) | [`projects/policy-evaluation/tests/check.py`](../projects/policy-evaluation/tests/check.py), [`starter/model.py`](../projects/policy-evaluation/starter/model.py), [`solution/model.py`](../projects/policy-evaluation/solution/model.py) | Work through stage 5 using a learner copy, then compare the independent oracle and reference results. The project remains a finite CPU corridor task. |
| Probability | [Hamiltonian Monte Carlo and sampler diagnostics](http://127.0.0.1:4187/lesson-probability-03.html) | [`phases/11-probability/03-hamiltonian-monte-carlo-and-sampler-diagnostics/lesson.json`](../phases/11-probability/03-hamiltonian-monte-carlo-and-sampler-diagnostics/lesson.json) | Trace the proposal, acceptance rule, changed seed, and diagnostic limits into the optional project. |
| Optional NumPyro project | [Bayesian regression project](../projects/bayesian-regression/README.md) | [`projects/bayesian-regression/tests/check_numpyro.py`](../projects/bayesian-regression/tests/check_numpyro.py), [`reference/numpyro/numpyro-raw-chains.npz`](../projects/bayesian-regression/reference/numpyro/numpyro-raw-chains.npz) | Review the raw-chain check against the exact posterior and changed cases. The required candidate API is `regression_model` plus `run_nuts`; five helper exercises in the starter are optional and are not graded by that checker. This uses a separate project environment and CPU reference. |
| Runtime versus target | [TPU runtime receipt](http://127.0.0.1:4187/lesson-tpu-04.html) and [precision/XProf](http://127.0.0.1:4187/lesson-tpu-06.html) | [`phases/19-tpu/02-bootstrap-venv-and-verify-tpu-runtime-receipt/lesson.json`](../phases/19-tpu/02-bootstrap-venv-and-verify-tpu-runtime-receipt/lesson.json), [`phases/21-tpu-systems/02-bfloat16-int8-precision-and-xprof-profiling/lesson.json`](../phases/21-tpu-systems/02-bfloat16-int8-precision-and-xprof-profiling/lesson.json) | A CPU run may validate the local arithmetic and trace workflow. Check that the report names the observed backend and does not present CPU results as TPU qualification or performance evidence. |
| Book and answers | [Book reader](http://127.0.0.1:4187/book.html) and [standalone reader](http://127.0.0.1:4187/downloads/jax-foundations.html) | [`src/pages/book.astro`](../src/pages/book.astro), [`src/pages/downloads/jax-foundations.astro`](../src/pages/downloads/jax-foundations.astro) | Attempt the checkpoint before revealing the reasoning. Compare answer feedback and solution placement across reader, standalone download, and worked references. |

For RL stage 5, start with a learner copy and use the cumulative checker after attempting the exercise:

```sh
cp projects/policy-evaluation/starter/model.py projects/policy-evaluation/my_model.py
python3 projects/policy-evaluation/tests/check.py --stage all --implementation projects/policy-evaluation/my_model.py
```

The reference implementation is available after the attempt:

```sh
python3 projects/policy-evaluation/tests/check.py --stage all --implementation solution
```

The RL extension learns a four-state tabular critic from sampled returns and computes GAE with separate bootstrap and trace masks. At a continuing-task cutoff it bootstraps from the stored final pre-reset observation while stopping the trace across the reset. The policy-evaluation project itself defines a finite eight-action objective and sets truncation bootstrap off for that horizon. The added critic and GAE are not connected to a new actor update; the project does not claim continuous-control performance or general actor-critic acceptance.

For the optional NumPyro stage, inspect that the checker reconstructs diagnostics and posterior checks from retained raw chains rather than trusting the submitted loss or report. It checks `regression_model` and `run_nuts`; `modern_diagnostics`, `exact_posterior_numpy`, `audit_gaussian_case`, `posterior_predictive_discrepancy`, and `run_reference_workflow` are five further study exercises, not acceptance requirements. Do not infer general convergence or real-data calibration from the synthetic fixture.

## Projects, assessments, and career handoffs

Review the changed-condition handoff at the end of each runnable project manual. Have the learner state a prediction, run the changed case, identify the independent check, and explain what the checker leaves untested. The complete project inventory and reference commands remain in the [0.10 review guide](review-guide-2026-10-21.md); this 0.11 review sequence adds the RL stage-5 and optional raw-chain checks above. A passed reference suite is evidence about its declared fixtures, not about the learner’s independent understanding.

Attempt the [RL synthesis assessment](../assessments/rl.md) and [probability synthesis assessment](../assessments/probability.md) before reading their reviewer notes. For broader coverage, use [`curriculum/assessments.json`](../curriculum/assessments.json) and the separate `assessments/*-reviewer.md` files; discussion assessments do not acquire an automatic score from the project tests.

Inspect all **10 learning paths** and **7 career scenarios**. For each path, check whether its prerequisites lead to a project artifact that supports the stated outcome. For careers, read each `roles[].workExample` in [`curriculum/course.json`](../curriculum/course.json) and ask a working practitioner whether the task reflects real responsibilities. Duration and job-readiness claims still need external calibration.

## What remains open

Thirteen phases remain marked **in development**: Transformers, Distributed training, Scientific computing, Probabilistic modeling, Reinforcement learning, Pallas kernels, Deployment, Workload operations, Pretraining, Post-training, TPU setup, TPU workflows, and TPU systems. The ten larger model and retrieval capstones remain **planned** specifications. Native attention and low-bit kernel evidence, broader target hardware qualification, and real edge-device measurements remain open. The current bounded CPU and hardware receipts keep their own scope. No specialist review, novice pilot, or human acceptance is recorded by this author packet.

The complete 0.11 EPUB passes official EPUBCheck 5.4.0 with zero errors and zero warnings after 14 MathML conformance repairs. This is package and markup conformance evidence. Apple Books, a second EPUB reader, screen-reader speech and printed pagination still need actual reading sessions. The desktop was locked during the native review attempt. See the [book qualification matrix](book-compatibility.md).

Record actual findings in the [blank 0.11 review notes](review-notes-0.11.md). Leave every decision blank until a reviewer has inspected the material and identifies the revision reviewed.
