<!-- jaxpathways-content-audit:2026-10-27:profiler-teaching -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P2. Finding type: mixed-correction-and-extension.

## Teaching problem

The course emits real traces and correctly warns about some measurement boundaries, but the learner is not guided through a labeled retained timeline to identify an actual event and its critical-path implication.

`tpu-06.capture_warmed_trace` repeatedly squares a 256x256 matrix of BF16 ones. For entry magnitudes, log2 grows as 8,24,56,120,248. The exact JIT dot/result-cast operation is finite for four iterations and nonfinite at iteration five. The default four iterations happen to survive; increasing the public steps parameter to five changes the computation to overflow without a finite guard.

Its quiz also overstates a warmed trace as measuring only steady TPU execution and collective communication. The shown loop has no collective, and traces can include host work and profiler overhead. Keep the distinction between CPU rehearsal and a target trace.

## Acceptance criteria

- [ ] Choose a bounded stable workload, justify its scale and check finite outputs before interpreting the trace.
- [ ] Separate warmup, profiled work and unprofiled timing evidence. Do not infer communication from a trace with no collective.
- [ ] Provide a retained, version/backend-labeled trace excerpt with corresponding event names and a readable timeline figure.
- [ ] Ask learners to identify nested/overlapping spans, synchronization and the actual bottleneck, then predict and check one change.
- [ ] Correct the quiz, explanation and distractor feedback to match what the supplied trace can establish.

## Audited source

- [performance-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/08-performance/03-read-an-xprof-trace/lesson.json)
- [tpu-06](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/21-tpu-systems/02-bfloat16-int8-precision-and-xprof-profiling/lesson.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
