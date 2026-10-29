# Profiler lesson P2#81 repair notes

## Scope

- Updated `phases/08-performance/03-read-an-xprof-trace/lesson.json` only within the performance-03 lesson: actual retained timeline interpretation, data-backed figure and timeline descriptions, an intervention prediction/check, and a learner exercise with nested-interval negative control.
- Added `phases/08-performance/03-read-an-xprof-trace/code/capture_retained_cpu_timeline_p2_81.py` and its compact numeric output `phases/08-performance/03-read-an-xprof-trace/data/retained_cpu_timeline_p2_81.json`.
- The source capture SHA-256 recorded in the excerpt is `bdc6894458ea8cb8f33aeebf76fd24974c5bae6190e88af0c706f3f5c8e4335c`.
- No P1 timing-math changes were made. The TPU lesson and generated lesson/visualization files were not edited.

## Evidence

Executed:

```sh
/private/tmp/jaxpathways-0170-clean-learner-env/bin/python phases/08-performance/03-read-an-xprof-trace/code/capture_retained_cpu_timeline_p2_81.py
```

Environment: Python 3.14.3, JAX 0.9.2, CPU backend, `TFRT_CPU_0`. The `(16,8) @ (8,8)` float32 `jit(tanh(a @ b))` result passed a finite independent NumPy comparison. The exact specialization was warmed and synchronized before recording two steps. The retained JSON contains Perfetto timestamp and duration excerpts in microseconds, source hash, runtime/backend labels, and capture scope. It contains no XPlane/Perfetto binary or absolute temporary path.

First step event excerpt:

| Event | Start (us) | Duration (us) | End (us) |
| --- | ---: | ---: | ---: |
| `learner_step` | 122 | 2665 | 2787 |
| `input_wait` | 125 | 2520 | 2645 |
| `model_and_wait` | 2668 | 119 | 2787 |

The child intervals are ordered and disjoint: union `2639 us`; outer remainder `26 us`. The negative control adding the parent and both children yields `5304 us`, which double-counts nested elapsed time. With the sleep request changed from 2 ms to 0.5 ms in the same trace, `input_wait` measured 642 us, a drop of 1878 us. `model_and_wait` measured 79 us in that second step versus 119 us in the first. This confirms the predicted direction for the injected host delay only.

## Verification and packaging

- The capture command above completed successfully and wrote the retained numeric JSON.
- Parsed the lesson and excerpt JSON, checked the capture-script source hash, confirmed six retained named events, checked each `start_us + duration_us == end_us`, and confirmed the recorded nested-sum result is 5304 us.
- Generated the full lesson script through `scripts/lesson-artifacts.mjs` and ran it with the learner Python environment from (a) an empty temporary working directory and (b) the course workspace. Both runs ended `PASS: performance-03`. The empty-directory run confirms the figure data does not require the repository excerpt or an absolute path; the workspace run confirms the embedded records match the retained excerpt.
- The mechanism table uses actual first-step CPU events. Do not substitute the renderer's conceptual `timeline` panel, which draws a generic device-work lane. The ordinary measured-duration figure comes from embedded event records checked against the fixture when available.
- Package the uniquely named capture helper and numeric JSON with the lesson source so the documented capture command and source-evidence link work from the course root. The standalone generated exercise remains runnable without that asset because it embeds the six small measured event records. Regenerate ordinary figure/visual outputs from the lesson source when assembling the broader content repair; do not commit generated outputs or release-maturity metadata for this repair.
- Runtime interpretation caveat: the custom regions are host spans; `model_and_wait` includes dispatch and result readiness and does not isolate a device kernel. The sleep is synthetic, and profiling can perturb short measurements. A separate synchronized benchmark is required for any optimization claim.
