<!-- jaxpathways-content-audit:2026-10-27:semantic-equations -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P1. Finding type: confirmed-correction.

## Teaching problem

The rendered math is syntactically valid but often belongs to a different concept, or contradicts the adjacent implementation. Passing KaTeX and Python checks does not establish semantic agreement.

| Lesson | Observed problem | Required alignment |
| --- | --- | --- |
| welcome-02 | Guided reductions display an affine transform. | Row/column reduction with the exact input and axes. |
| arrays-03 | Purity displays an indexed replacement equation. | Pure input/state/output contract. |
| arrays-04 | Immutable indexing displays matrix multiplication. | Functional indexed replacement and preservation of other entries. |
| transforms-04 | JIT section displays a scan recurrence. | The actual compiled loss/gradient calculation. |
| transforms-05 | Specialization displays an SGD update. | Trace signatures, static values and cache behavior. |
| state-04 | cond displays pytree flatten/map. | Branch result agreement and predicate semantics. |
| networks-02 | Equation uses `(G, Theta, S) = split(M)`; code uses unfiltered two-result `nnx.split(model)`. | GraphDef plus State, or explicitly filtered states and matching code. |
| networks-03 | Multiclass one-hot cross entropy above one-logit binary sigmoid BCE code. | Binary loss with matching target/logit shapes. |
| recovery-01 | Restore of `(parameters, momentum, key, step)` implies uninterrupted equality. | Include data cursor/order, configuration/data identity and the matching execution assumptions. |
| recovery-02 | Iterator restore displays momentum SGD. | Next-read identity and iterator state. |
| recovery-03 | SHA of serialized state is equated with an undefined SHA of a deserialized object. | Separate artifact-byte integrity, semantic leaf equality and next-transition replay. |
| recovery-04 | Modulo cursor formula accompanies an opaque shuffled Grain iterator. | Describe the implemented iterator state, or explicitly scope modulo arithmetic to a different sequential example. |
| recovery-05 | `T_commit = max(T_write[h]) < T_rename`. | Define timestamps versus durations; accepted publication occurs only after required completion/publication checks. The local fixture does not qualify a multihost commit protocol. |
| transformers-03 | LayerNorm formula has learned gamma/beta absent from the function. | Fixed normalization versus affine LayerNorm, with declared parameterization. |
| performance-01 | Exact `T_first = T_compile + T_warm`. | First-call work includes tracing/lowering/dispatch; a warm median is a separate measurement, not an exactly isolated component. |
| performance-02 | Retracing displays a roofline formula. | Signature-to-trace event relationship. |
| performance-03 | Exact additive input + compute + sync despite nested/overlapping interval caveats. | Disjoint serial assumptions or critical-path/interval accounting, including other work. |
| kernels-04 | Precision-reference section displays Amdahl's formula. | Move it to the end-to-end opportunity task; explain its variables there. |
| operations-07 | Container digest plus `P(reproduce)=1` in LLMOps. | Trace identities and explicit reproducibility conditions. An image digest alone cannot guarantee reproduction. |
| operations-08 | Approval section displays cosine similarity. | Bundle-bound approval, expiry and policy identity. |
| tpu-04 | Runtime verification displays optimizer-state memory. | Expected versus observed backend/topology and arithmetic checks. |
| tpu-02 | `(T_total(S)-T_compile(1))/(S-1)` despite a discarded warmup followed by S timed updates. | Use actual logged update intervals/counts and distinguish warmup from isolated compilation. |
| tpu-05 | Unweighted local-gradient mean divided by P, without local-count/axis assumptions. | Equal valid-count assumptions or global numerator/count reduction, with collective participation defined. |
| tpu-06 | BF16 dot section displays token-cost arithmetic. | Operand/result/storage policy and the error being compared. |

The prose often already includes useful caveats. Preserve those and remove the contradictions rather than deleting the math. [Flax's graph API](https://flax.readthedocs.io/en/v0.12.7/api_reference/flax.nnx/graph.html) documents split/merge results and explicit filters.

## Acceptance criteria

- [ ] Review each row against its exact code, input shapes, units and learning objective; replace or relocate the equation and define every symbol.
- [ ] Include a hand-worked case for each distinct corrected concept and a changed-condition check where the current equation changes the expected result.
- [ ] Keep checkpoint integrity, numerical equality and reproducibility assumptions separate. Remove the probability-one reproducibility claim.
- [ ] Render the corrections in lesson, chapter and downloadable book outputs and check that the surrounding explanation reads the displayed expression accurately.
- [ ] Have a content reviewer assess semantic agreement; do not treat a substring/KaTeX smoke test as teaching verification.

## Audited source

- [welcome-02](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/00-welcome/02-meet-your-arrays-and-devices/lesson.json)
- [arrays-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/01-arrays/03-pure-functions-and-explicit-inputs/lesson.json)
- [arrays-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/01-arrays/04-immutable-updates-and-indexing/lesson.json)
- [transforms-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/02-transforms/04-compile-a-function-with-jit/lesson.json)
- [transforms-05](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/02-transforms/05-tracing-static-arguments-and-recompilation/lesson.json)
- [state-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/03-state/04-branches-with-lax-cond/lesson.json)
- [networks-02](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/05-networks/02-model-and-state-with-flax-nnx/lesson.json)
- [networks-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/05-networks/03-a-compiled-train-and-evaluation-step/lesson.json)
- [recovery-01](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/06-recovery/01-design-an-input-pipeline/lesson.json)
- [recovery-02](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/06-recovery/02-load-batch-and-prefetch-with-grain/lesson.json)
- [recovery-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/06-recovery/03-save-model-optimizer-and-random-state/lesson.json)
- [recovery-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/06-recovery/04-recover-data-position-and-resume/lesson.json)
- [recovery-05](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/06-recovery/05-asynchronous-checkpoints-and-failure-boundaries/lesson.json)
- [transformers-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/07-transformers/03-a-transformer-block-and-mixed-precision/lesson.json)
- [performance-01](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/08-performance/01-benchmark-asynchronous-work-correctly/lesson.json)
- [performance-02](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/08-performance/02-diagnose-recompilation-and-host-synchronization/lesson.json)
- [performance-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/08-performance/03-read-an-xprof-trace/lesson.json)
- [kernels-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/13-kernels/04-iterate-with-correctness-and-performance-evidence/lesson.json)
- [operations-07](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/07-llmops-prompts-evaluation-and-traces/lesson.json)
- [operations-08](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/08-modelops-ownership-approval-and-retirement/lesson.json)
- [tpu-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/19-tpu/02-bootstrap-venv-and-verify-tpu-runtime-receipt/lesson.json)
- [tpu-02](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/20-tpu-workflows/01-stage-and-launch-checkpointed-tpu-jobs/lesson.json)
- [tpu-05](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/20-tpu-workflows/02-run-course-scripts-and-multihost-tpu-slices/lesson.json)
- [tpu-06](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/21-tpu-systems/02-bfloat16-int8-precision-and-xprof-profiling/lesson.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
