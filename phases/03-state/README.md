# Phase 03: State, randomness & control flow

Shared foundations.

Make randomness, parameters and loop state explicit. Learn how to repeat an experiment without accidentally reusing randomness or mutating a caller’s state.

Record the state before and after a transition and explain every changed leaf.

**Prerequisites:** 02: JAX transformations.

**Hardware:** CPU.

## Study guide: Have you saved everything needed to determine the next step?

A program’s state includes random keys, counters and structured values as well as model parameters. Explicit ownership lets you reproduce a transition, run a batch safely, and later recover an interrupted experiment.

### Check your starting point

Two simulated examples receive the same random key and run the same sampling function. Should you expect independently allocated draws?

<details><summary>Compare your reasoning</summary>

No. The same key and operation replay the same draw in the tested environment. Split or derive distinct keys at the ownership boundary; unequal observed samples alone do not prove independence.

</details>

Review: [Random keys without surprises](01-random-keys-without-surprises/docs/en.md).

### Build in stages

1. **Name and structure the state.** Draw the key ownership tree and identify parameter, optimizer and metadata leaves. Check both tree structure and leaf values.

   Lessons: [Random keys without surprises](01-random-keys-without-surprises/docs/en.md) · [Pytrees and structured parameters](02-pytrees-and-structured-parameters/docs/en.md).

2. **Audit repeated and conditional transitions.** Implement a recurrence with fixed carry shapes. Save per-step observations as outputs, compare with a Python loop, then test both conditional branches and their boundary.

   Lessons: [Compiled loops with lax.scan](03-compiled-loops-with-lax-scan/docs/en.md) · [Branches with lax.cond](04-branches-with-lax-cond/docs/en.md).

### Try a changed condition

Resume a simulation halfway through with the saved position but a newly created key. What must you compare to determine whether it is a true continuation?

<details><summary>Compare an approach</summary>

Compare the next draw and next state, then the remaining trajectory, against an uninterrupted reference. Restore the original key and counter as well as position. A similar final average is not a continuation check.

</details>

**Symptom:** A compiled loop fails when its history grows each step.

**Check next:** Keep carry structure, shapes and dtypes fixed. Return the per-step history through scan outputs.

### Decide what is ready

Use foundation-toolkit stage 4. Save and reload state in a fresh process and verify the next transition and full remaining trajectory, not only the final mean.

### Further work

Local deterministic state does not establish concurrent-writer or multi-host recovery; the recovery and operations phases extend these boundaries.

## Lesson sequence

### 03.01 Random keys without surprises

[Read the lesson](01-random-keys-without-surprises/docs/en.md) · [Run the code](01-random-keys-without-surprises/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Split keys and show reproducibility without accidental key reuse.

**Evidence:** Keep a key-ownership tree, two-step replay including the final key, the per-ID reorder check, and the broken/repaired repeated-noise loop.

**Checkpoint:** What happens if the same key is used twice for the same random operation?

### 03.02 Pytrees and structured parameters

[Read the lesson](02-pytrees-and-structured-parameters/docs/en.md) · [Run the code](02-pytrees-and-structured-parameters/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Represent parameters and state in a nested structure and transform its leaves.

**Evidence:** Keep the parameter/gradient tree diagram, hand-derived leaves and update, flatten/unflatten round trip, nested-scale derivative and missing-key repair.

**Checkpoint:** What structure does grad(loss)(params) have?

### 03.03 Compiled loops with lax.scan

[Read the lesson](03-compiled-loops-with-lax-scan/docs/en.md) · [Run the code](03-compiled-loops-with-lax-scan/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Replace a fixed-length Python loop with scan and compare final states.

**Evidence:** Keep the labeled carry/output transition, Python-reference trajectory, polynomial derivative, compiled value check and growing-carry failure with a fixed-shape repair.

**Checkpoint:** What must remain consistent across scan steps?

### 03.04 Branches with lax.cond

[Read the lesson](04-branches-with-lax-cond/docs/en.md) · [Run the code](04-branches-with-lax-cond/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Implement a data-dependent branch that works inside a compiled function.

**Evidence:** Keep the branch-contract diagram, eager/jit failure and repair, boundary cases, batched values and off-boundary derivatives, and the shape-mismatch diagnosis.

**Checkpoint:** Can one cond branch return a scalar and the other a length-three array?

## Phase project

A deterministic batched simulation.

**Demonstrate:** Replay the same experiment with saved state and keys; explain the update order.

Project status: implemented staged practice · [Open source](../../projects/foundation-toolkit/README.md). Use stages 4 for this phase.



[Primary documentation](https://docs.jax.dev/en/latest/beginner_guide.html).
