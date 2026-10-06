# Phase 13: Pallas kernels

Specializations.

Write actual Pallas grids and buffered pipelines, verify boundaries and precision on CPU, and prepare guarded target measurements before claiming an optimization win.

Begin with a profiled operation and independent oracle. Complete CPU semantics first; actual TPU compilation, correctness and performance require the separate target runner on supported hardware.

**Prerequisites:** 09: Distributed training.

**Hardware:** CPU interpretation/simulation reference; TPU target lab requires separately validated hardware.

## Study guide: Does a custom kernel earn its complexity?

A lower-level implementation first owes you the same numerical contract. Use hostile shapes, independent references and explicit target limits before attempting a speed claim.

### Check your starting point

A matrix dimension is not divisible by the tile size. What must the last tile do?

<details><summary>Compare your reasoning</summary>

Mask or pad out-of-range loads and stores under a declared policy. Valid elements still need exactly one correct output; fewer launched programs do not excuse missing boundary elements.

</details>

Review: [Pallas grids and BlockSpecs](01-pallas-grids-and-blockspecs/docs/en.md).

### Build in stages

1. **Prove indexing and arithmetic.** Map each grid coordinate to input/output slices, test rectangular and boundary shapes and reject unsupported layouts or dtypes before launch.

   Lessons: [Pallas grids and BlockSpecs](01-pallas-grids-and-blockspecs/docs/en.md) · [A first TPU kernel](02-a-first-tpu-kernel/docs/en.md).

2. **Measure only a supported target.** Check pipeline semantics and buffer assumptions, compare with a strong compiled baseline and make an end-to-end decision from the operation’s measured share.

   Lessons: [Tiling, memory, and pipelining](03-tiling-memory-and-pipelining/docs/en.md) · [Iterate with correctness and performance evidence](04-iterate-with-correctness-and-performance-evidence/docs/en.md).

### Try a changed condition

Interpret mode passes all numerical tests. Can you use its timings to choose the fastest TPU tile?

<details><summary>Compare an approach</summary>

No. Interpretation supports bounded semantic checks. Compile and measure supported candidates on the named TPU with equivalent work and synchronization, including numerical verification there.

</details>

**Symptom:** A kernel passes square inputs but corrupts rectangular ones.

**Check next:** Inspect operand-specific index maps and boundary masks; square dimensions can hide an axis swap.

### Decide what is ready

Use kernel-audit. Retain independent numerical checks, rejected unsupported contracts and a target measurement plan. Keep hardware timings unmeasured until actually executed.

### Further work

Actual TPU/GPU compilation, race diagnosis and target tuning remain unqualified; CPU interpretation cannot close these gaps.

## Lesson sequence

### 13.01 Pallas grids and BlockSpecs

[Read the lesson](01-pallas-grids-and-blockspecs/docs/en.md) · [Run the code](01-pallas-grids-and-blockspecs/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Implement an actual Pallas Ref-writing kernel with grid and BlockSpecs. Derive block coordinates and distinguish them from element offsets.

**Evidence:** Independent full-array comparisons, ownership map, tail fixtures and a floor-grid omission diagnosis. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** For block shape (2,4), what does a BlockSpec index map returning (i,j) select?

### 13.02 A first TPU kernel

[Read the lesson](02-a-first-tpu-kernel/docs/en.md) · [Run the code](02-a-first-tpu-kernel/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Implement a real TPU-targeted Pallas launch with a broadcast BlockSpec. Specify target layout and float32 compute/output-cast policies.

**Evidence:** Broadcast/boundary checks, represented-input precision oracle and a tested no-fallback TPU guard; target execution remains separately unverified. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** What does a successful CPU interpret run establish for this target-oriented kernel?

### 13.03 Tiling, memory, and pipelining

[Read the lesson](03-tiling-memory-and-pipelining/docs/en.md) · [Run the code](03-tiling-memory-and-pipelining/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Execute actual emit_pipeline semantics under explicit CPU TPU-layout simulation. Compare synchronous and buffered schedules against an independent oracle.

**Evidence:** Synchronous/buffered CPU simulation agreement, two/three input slots, unchanged output buffering, correctness across tiles and modeled footprint counts. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** The three-buffer simulation is correct and has fewer grid programs after a tile change. Can you conclude the TPU version is faster?

### 13.04 Iterate with correctness and performance evidence

[Read the lesson](04-iterate-with-correctness-and-performance-evidence/docs/en.md) · [Run the code](04-iterate-with-correctness-and-performance-evidence/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Build a representative shape/dtype correctness matrix for an actual Pallas pipeline. Separate kernel implementation error from input/output precision differences.

**Evidence:** Shape/dtype reference checks, separate precision gaps, guarded target benchmark and a bounded end-to-end speedup argument. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** The kernel matches the represented-input oracle exactly but differs from the original float32 dataset in bfloat16. What should the report say?

## Phase project

A checked Pallas kernel with an explicit target qualification plan.

**Demonstrate:** Compare against a reference across shapes and justify each optimization with evidence.

Project status: implemented staged practice · [Open source](../../projects/kernel-audit/README.md).



[Primary documentation](https://docs.jax.dev/en/latest/pallas/index.html).
