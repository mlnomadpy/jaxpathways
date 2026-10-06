# Text harness authoring evidence

## Delivered and executed

The project at projects/text-harness is a connected, actual CPU training-system capstone. It includes a learner starter, independent cumulative stage checks, reference implementation, connected runner, project manifest, explanatory README, actual plotted output, complete checkpoint, six serialized inference endpoints and a real profiler trace.

Its one-block causal Transformer has a UTF-8 byte tokenizer, 259 vocabulary items, 12 input positions, width 24, two heads and a 48-wide feed-forward block. The task is a synthetic reversal-string fixture, not a pretrained language model. Training uses 96 strings; the fixed held-out set has 32 different identities, groups and string contents. The local external-document manifest loader checks actual UTF-8 bytes, content hashes, provenance fields and cross-split leakage. No external corpus was downloaded or evaluated.

Executed locally on 2026-10-05 with Python 3.14.3, JAX 0.9.2, NumPy 2.4.4 and TFRT_CPU_0:

```sh
python3 projects/text-harness/tests/check.py --implementation solution --stage all
python3 projects/text-harness/run.py --implementation solution
```

All four cumulative stages passed. The untouched starter fails immediately with the intended NotImplementedError: Implement make_corpus. This is a learner exercise, not an already passing starter.

## Independent evidence and boundaries

- Stage 1 checks UTF-8 bytes, split/content provenance, an independent NumPy attention row, independent masked log-softmax loss and causal invariance.
- Stage 2 compares an automatic derivative with a central difference; restores full Adam moments, parameters, dropout RNG, shuffled order, cursor, epoch and step in a fresh Python process; compares four updates across an epoch boundary; rejects changed data/configuration; verifies non-mutating, count-weighted held-out evaluation; and trains a changed initialization.
- Stage 3 implements real stored FP32/BF16/INT8 K/V arrays, per-token/per-head scale metadata, incremental one-token K/V append, unchanged old prefix and context rejection. Sequential decode matches full causal recomputation under each same policy. Independent cache rounding and NumPy INT64 dot references check the INT8/INT32 arithmetic.
- Stage 4 actually serializes and reloads six prefill/decode endpoints, executes them in another Python process, rejects corrupted artifacts, retains raw generated token IDs and performs synchronized CPU measurements.
- The connected runner records a real current-run XPlane/JSON/Perfetto trace. The receipt contains 23,630 trace events and five actual regions each for training updates, prefill and cached decode. Trace annotations include Python boundaries; they are not claims of pure kernel duration.

The final reference trains 240 updates and recovers after update 120. Public fresh-process recovery is separately exercised at the earlier epoch boundary. Final held-out NLL is 1.07082093 over 276 valid targets, with 0.601449 token accuracy and 2.917774 perplexity. The training-frequency baseline's held-out NLL is 2.35228644. A changed initialization trained for 120 steps produces NLL 1.02388727; this is one additional seed, not a stability theorem.

The first three fixed held-out generation demonstrations are correct. A separate diagnostic prompt abc| generates cbc rather than the desired cba. The report and README retain that failure and distinguish it from the held-out aggregate.

Actual allocated cache bytes, including scale arrays, are 2496 FP32, 1344 BF16 and 768 INT8. Cache-only maximum valid logit differences against FP32 are 0.03854144 for BF16 and 0.19518447 for INT8. These are numerical errors and allocated array bytes, not accuracy differences or peak process memory.

The combined policies use rounded BF16 dense operands or dynamic W8A8 dense operands with actual INT32 accumulation. Embeddings, layer normalization and attention remain FP32. No native integer-kernel speedup is claimed. The current run measures FP32 warm medians of approximately 0.185 ms prefill and 0.156 ms cached decode. These are local fixed-prefix single-step measurements with synchronization, not a service throughput or long-stream result. All samples and first-call times remain in outputs/report.json.

Both actual PNG plots were visually inspected: axes, labels, legends and corresponding explanations describe the measured training trajectory, one actual causal attention head, allocated cache bytes and isolated cache approximation error. The final checkpoint held-out line is explicitly described as one final-model comparison, not a trajectory. Attention is explained as information routing rather than correctness probability.

After final execution, source_sha256 and runner_sha256 match their actual files; all six export hashes match current artifact bytes; every reported current-run profiler path exists. Superseded profiler output was removed. The source-bound receipt is projects/text-harness/outputs/report.json.

## Primary API verification

The installed JAX export and profiler calls execute successfully in the checks and connected runner. [JAX export documentation](https://docs.jax.dev/en/latest/export/export.html) and [JAX profiling documentation](https://docs.jax.dev/en/latest/profiling.html) were checked. Serialized computations still need a compatible runtime; the local execution does not establish portability to every JAX version or target.

## Root integration mapping

No shared registries or generators were edited by this agent.

1. Register projects/text-harness/project.json in curriculum/projects.json and text-harness in course.projectIds.
2. Set the tpu pathway capstone projectId to text-harness and authored status, with wording such as “Causal Transformer training-system capstone.” Preserve the separate physical TPU/multi-controller qualification requirement.
3. Connect the text modality track to projectIds [text-harness]. Its available evidence is connected CPU byte-Transformer training, full recovery, fixed token evaluation, real KV-cache precision, serialized inference and profiler measurements. Keep its firstLessonId transformers-01 and describe external natural-language/physical target work as extension.
4. Map lifecycle stages: data and model → stage 1; train, recover and evaluate → stage 2; precision → stage 3; export and operate → stage 4.
5. Add this synthesis assessment entry, adapting only registry-required schema fields:

```json
{
  "id": "tpu",
  "projectId": "text-harness",
  "title": "Training-system synthesis",
  "status": "review-draft",
  "scope": "project-synthesis",
  "source": "assessments/tpu.md",
  "url": "assessments/tpu.html",
  "pathwayId": "tpu",
  "phaseId": "transformers"
}
```

The assessment and reviewer notes distinguish CPU systems acceptance from physical target acceptance. They include changed-data/configuration failure drills, multibyte transfer, unequal-batch aggregation, changed initialization, cache overflow, artifact corruption and target qualification.

README commands directly reference the existing welcome-03 platform check and distributed-04 recovery lab, with their actual environment variables and full-workspace requirements. Those physical TPU/distributed commands were not run during this delivery. No provisioning, paid deployment, device throughput, pretrained model quality or production-service claim is made.

Root remains responsible for registry integration, rendered assessment/project routes, downloadable project packaging, whole-corpus build/check and script/notebook smoke. Preserve checkpoint/export/profile artifacts in the project evidence or distribute them through the established packaging policy.
