# Pallas kernel authoring evidence — 2026-10-05

## Scope and evidence status

Four substantive canonical lessons are authored at the existing `phases/13-kernels` paths. They execute actual Pallas operations under CPU interpretation, including the actual TPU `emit_pipeline` API under explicit CPU simulation. They include staged builds, independent references, two experiments and two distinct transfer practices each, KaTeX prose, failure diagnosis, and four fresh interpreted figures.

**No real TPU lowering, execution, latency or speedup was validated.** A genuine TPU-only runner is implemented and refuses CPU fallback. This is a material remaining hardware qualification, not a placeholder teaching section or a simulated performance claim.

Applied skill: `/Users/tahabsn/.agents/skills/pallas-kernels/SKILL.md`. Current official grids/BlockSpec, TPU details, pipelining and InterpretParams documentation plus installed JAX source signatures were inspected. The code uses tested JAX 0.9.2 names, including `pltpu.CompilerParams` rather than the older `TPUCompilerParams` example name.

| Lesson     | Minutes | Concrete result                                                                                                                                     |
| ---------- | ------: | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| kernels-01 |      95 | Real Pallas grid/BlockSpec affine kernel, ownership map, explicit padding/cropping, full-array NumPy checks and floor-grid failure diagnosis        |
| kernels-02 |     110 | Real fused bias/ReLU Pallas body, broadcast window, precision contract and guarded TPU-only branch; CPU interpreted semantics executed              |
| kernels-03 |     120 | Actual synchronous/buffered `emit_pipeline`, race-enabled CPU TPU-memory simulation, multiple input buffer counts, tile/padding/footprint tradeoffs |
| kernels-04 |     120 | Shape/dtype correctness matrix, separate representation gaps, target-only synchronized benchmark protocol and end-to-end opportunity reasoning      |

## Root integration instructions

For `kernels-01` through `kernels-04`, preserve existing IDs/titles/paths and sequential prerequisites, set `status: authored`, copy the canonical source exercise, and replace placeholder objective/evidence/check metadata below. Authored means teaching/reference material is available; it must not imply target validation.

| ID         | Objective                                                                                                            | Evidence                                                                                                                                                 | Check                                                                                                |
| ---------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| kernels-01 | Map a matrix operation to actual Pallas grid and BlockSpecs with explicit boundary and precision contracts.          | Independent full-array comparisons, ownership map, tail fixtures and a floor-grid omission diagnosis.                                                    | Explain block coordinates versus element offsets and why partial logical tiles still need ownership. |
| kernels-02 | Implement and test a target-oriented fused kernel while keeping CPU interpretation distinct from real TPU execution. | Broadcast/boundary checks, represented-input precision oracle and a tested no-fallback TPU guard; target execution remains separately unverified.        | State what CPU interpretation establishes and what still requires actual target lowering/execution.  |
| kernels-03 | Execute buffered Pallas pipeline semantics and explain tile, padding and local-storage tradeoffs.                    | Synchronous/buffered CPU simulation agreement, two/three input slots, unchanged output buffering, correctness across tiles and modeled footprint counts. | Explain why fewer programs or extra buffering does not establish a target speedup.                   |
| kernels-04 | Build a correctness matrix and an executable target-only measurement protocol tied to a complete workload.           | Shape/dtype reference checks, separate precision gaps, guarded target benchmark and a bounded end-to-end speedup argument.                               | Distinguish implementation error, representation error and missing target performance evidence.      |

Recommended phase metadata:

- `projectId`: `kernel-audit`
- `description`: `Write actual Pallas grids and buffered pipelines, verify boundaries and precision on CPU, and prepare guarded target measurements before claiming an optimization win.`
- `learningAdvice`: `Begin with a profiled operation and independent oracle. Complete CPU semantics first; actual TPU compilation, correctness and performance require the separate target runner on supported hardware.`
- `hardware`: `CPU interpretation/simulation reference; real TPU target lab requires separately validated hardware`

Add `projects/kernel-audit/project.json` to the project registry. Its required lessons explicitly include profiling/sharding preparation and all four kernel lessons. The project's status is authored with CPU semantics available; its hardware text and README preserve unverified target execution.

Add the assessment registry entry:

```json
{
  "id": "scale",
  "projectId": "kernel-audit",
  "title": "Profiling, partitioning and kernel synthesis",
  "status": "review-draft",
  "scope": "project-synthesis",
  "source": "assessments/scale.md",
  "url": "assessments/scale.html"
}
```

The scale route should connect the authored project and review-draft assessment, remove stale statements that all custom-kernel material is missing, and retain explicit target evidence limits. Suggested description: `Profile a correct workload, preserve its global objective while partitioning arrays, and audit actual Pallas kernels before requesting target measurements. CPU semantics are available; accelerator speed and multi-host claims require their own receipts.` The assessment spans profiling, uneven partition reduction, global objective, candidate selection and local/end-to-end decisions. It explicitly leaves target qualification unresolved for CPU-only submissions.

## Actual execution and checks

Environment: Python 3.14, JAX 0.9.2, NumPy 2.4.4, actual backend CPU. Kernel compute dtype is float32 with explicit bfloat16 support through promoted arithmetic and one final output cast. No dependency changes.

1. Executed each complete example, experiments, main solution, transfer practices and visual code sequentially in fresh CPU Python processes. Four lessons passed after a real buffering-contract issue was identified and repaired. Staged build blocks concatenate to complete code.
2. The first pipeline probe failed because plain `interpret=True` lacked TPU layout metadata. The working explicit simulation uses `pltpu.InterpretParams(detect_races=True)` and `jax.sharding.AbstractMesh` with an `AbstractDevice` describing a simulated TPU v5 lite layout. The actual backend remains CPU, and both the source and printed output say so.
3. A three-output-buffer probe failed with the installed emitter's explicit restriction. The implementation now keeps two output buffers and compares two or three input buffers. Teaching, footprint formulas and tests reflect this observed contract. No failed run was relabeled a pass.
4. `python3 projects/kernel-audit/tests/check.py --implementation solution --stage all` passed all three cumulative stages: actual Pallas grids, full/tail/singleton fixtures, float32/bfloat16, independent represented-input references, fused bias axes/ReLU boundaries, actual synchronous and buffered pipeline simulation, input buffering and rejected shape/dtype/mode/buffer contracts.
5. The starter failed intentionally at its first `NotImplementedError` and cannot be counted as complete learner work.
6. Both `python3 projects/kernel-audit/target_tpu.py --kernel fused --block 8 128` and the analogous pipeline command refused the CPU environment with exit status two and no target receipt. The runner checks real backend before allocation, passes non-interpret target mode, synchronizes, verifies a NumPy-based oracle, then records actual device/configuration/source hash/compilation and timing samples only on success. Target branches remain unexecuted here.
7. `python3 projects/kernel-audit/tests/check-scale-reference.py` passed in a fresh four-logical-CPU process. It independently verifies a seeded global regression gradient, one update, actual logical-device placement, an uneven three/five count-weighted repair and conditional speedup arithmetic. It is not network or accelerator performance evidence.
8. Rendered all new lesson non-code math/prose through site math helpers: 71, 73, 74 and 73 fields passed. Final figure prose was updated to describe observed values and read again.
9. Regenerated all four figure assets with `renderer.worker` in fresh CPU processes; after a visual simplification, the final fourth figure was rerun again. Source hashes are fresh. Every PNG was inspected for readable labels, honest units and correct narrative. The all-zero implementation-error plot was removed in favor of the informative precision-gap chart; the exact oracle passes remain explicit in code and prose.

Representative observed results:

- Grid fixture `(5,11)`: nine programs cover a padded `(6,12)` buffer. Logical endpoints `-3.75` and `7.05000019`. A floor grid would miss 23 logical cells.
- Fused `(9,129)` CPU result: zero maximum error, final value `2.400000095`; about 48.58% of outputs are zero. Exact cancellation fixtures and represented bfloat16 inputs pass.
- Pipeline `(17,257)`: synchronous/buffered endpoints `-1.75 / 2.25`; all tested schedules agree. Tile program counts `9 / 6 / 4`, padded elements `9216 / 12288 / 16384`, modeled double-buffered data bytes `24576 / 49152 / 98304`.
- Three input slots with two output slots change the modeled data storage to `32768 / 65536 / 131072` bytes, an increase of one third, not one half.
- Eight shape/dtype audit cases have zero implementation error against the represented-input oracle. bfloat16 gaps to original float32 data are approximately `0.0008624 / 0.0252171 / 0.0307884 / 0.0292206`; these are numerical precision differences, not hardware measurements.
- Explicit rounding-order counterexample: early rounding `1.0`, one final rounding `1.0078125`.
- Scale-reference global gradient `[0.58924425, -0.23621893, -0.34260178]`; wrong unweighted local means `[0.5557211, -0.14050466, -0.29906958]`. Count weighting recovers the correct gradient. Conditional twofold-local speedup at 30% cost is `1.17647`, with zero-local-cost bound `1.42857`.

## Remaining integration and qualification

No shared manifests, generators or existing lesson sources were changed by this agent. Root must register and regenerate the course, execute the full script/notebook smoke suite and complete integration checks. The assessment is a public review draft. No browser, real TPU/GPU, multi-host, independent learner or external expert validation was performed.

The target branch is implemented using current inspected APIs and conservative layout contracts, but only real target execution can establish lowering support and performance. No interpretation timing is reported as a kernel benchmark; no simulated device identity is recorded as actual hardware. Root completion claims should preserve this boundary.
