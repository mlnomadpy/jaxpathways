# Phase 02: JAX transformations

Shared foundations.

Begin with a scalar function, then differentiate it, batch it and compile it. Compare each transformed result with a simple reference so you can explain what changed.

Keep eager and transformed outputs plus a deliberately broken assumption.

**Prerequisites:** 01: Arrays & pure functions.

**Hardware:** CPU.

## Study guide: What changes when you differentiate, batch or compile a function?

Build one trustworthy scalar calculation before composing transformations. Differentiation changes the quantity, batching adds an axis, and compilation changes execution. Keeping these roles distinct makes later training failures easier to locate.

### Check your starting point

For a squared residual, is the gradient of the average loss the same shape as the stack of per-example gradients?

<details><summary>Compare your reasoning</summary>

No. A stack retains the examples axis; the gradient of the average reduces that axis. Their mean agrees for the same equally weighted differentiable objective.

</details>

Review: [Losses and value_and_grad](02-losses-and-value-and-grad/docs/en.md).

### Build in stages

1. **Establish the derivative.** Predict derivatives at several inputs and compare with an analytic calculation. Check that the loss is a scalar before differentiating.

   Lessons: [Your first gradient](01-your-first-gradient/docs/en.md) · [Losses and value_and_grad](02-losses-and-value-and-grad/docs/en.md).

2. **Compose without changing the question.** Compare a Python loop, mapped gradients and their compiled version on the same inputs. Change numerical values separately from shapes and static configuration; record tracing and timing as different observations.

   Lessons: [Batch a function with vmap](03-batch-a-function-with-vmap/docs/en.md) · [Compile a function with jit](04-compile-a-function-with-jit/docs/en.md) · [Tracing, static arguments, and recompilation](05-tracing-static-arguments-and-recompilation/docs/en.md).

### Try a changed condition

One minibatch has two examples and another has six. How should you combine their mean gradients?

<details><summary>Compare an approach</summary>

Weight each mean gradient by its example count, add the weighted gradients, then divide by eight. An unweighted average gives examples in the smaller batch too much influence. Verify against a gradient over the concatenated batch.

</details>

**Symptom:** A Python conditional works eagerly but raises a tracer error under jit.

**Check next:** Identify the data-dependent truth test. Use a JAX control-flow operation with compatible branch outputs; do not mark changing array data static to suppress the symptom.

### Decide what is ready

Use foundation-toolkit stage 3. Keep independent derivatives, batched/compiled parity and one changed signature. Explain why a faster warm call does not measure first-call cost.

### Further work

Transformation combinations involving nondifferentiable boundaries require further examples; passing smooth fixtures does not validate arbitrary gradients.

## Lesson sequence

### 02.01 Your first gradient

[Read the lesson](01-your-first-gradient/docs/en.md) · [Run the code](01-your-first-gradient/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Explain local sensitivity and the chain rule, then compare analytic, numerical, and automatic derivatives.

**Evidence:** Keep predictions and analytic derivations at multiple inputs, the finite-difference step/error sweep with dtype, both argnums checks, and the scalar-objective repair. Explain why a zero gradient need not mark a minimum.

**Checkpoint:** At x = 0, jax.grad(lambda x: x ** 3)(0.0) returns zero. What can you conclude?

### 02.02 Losses and value_and_grad

[Read the lesson](02-losses-and-value-and-grad/docs/en.md) · [Run the code](02-losses-and-value-and-grad/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Derive and audit a scalar loss, its parameter gradient, and stable versus unstable updates.

**Evidence:** Keep handwritten loss/gradient arithmetic, checks at three weights, small/large-rate update results, auxiliary residuals, and mean-versus-sum batch-replication results. Explain the update-scale change.

**Checkpoint:** You duplicate every example and keep the same learning rate. What happens to a sum-loss SGD update compared with the original batch?

### 02.03 Batch a function with vmap

[Read the lesson](03-batch-a-function-with-vmap/docs/en.md) · [Run the code](03-batch-a-function-with-vmap/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Specify batching axes and connect per-example gradients to a reduced batch objective.

**Evidence:** Keep the shape contract, loop/vmap/matrix equivalence, row/column layout checks, per-example gradient array, and mean-gradient identity under two target sets. Explain what each gradient shape means.

**Checkpoint:** For B examples and D shared parameters, what shapes do per-example loss gradients and the mean-loss gradient have?

### 02.04 Compile a function with jit

[Read the lesson](04-compile-a-function-with-jit/docs/en.md) · [Run the code](04-compile-a-function-with-jit/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Validate compiled loss/gradient values and produce a correctly synchronized latency report.

**Evidence:** Keep the independent loss/gradient calculation, backend/shapes/dtype, first-call interval, warmup/repeat policy, median/range samples, and synchronization of every output leaf. State what the benchmark excludes.

**Checkpoint:** Why wait with block_until_ready when timing a computation?

### 02.05 Tracing, static arguments, and recompilation

[Read the lesson](05-tracing-static-arguments-and-recompilation/docs/en.md) · [Run the code](05-tracing-static-arguments-and-recompilation/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Explain the staging boundary and repair configuration versus runtime-control-flow mistakes.

**Evidence:** Keep the small jaxpr observation, the expected tracer Boolean error, checks of both repaired branches, all static reduction modes including invalid-mode rejection, and shape/value-change cases. Do not infer compilation counts from timing alone.

**Checkpoint:** A frequently changing runtime scalar controls a branch. What is the most appropriate first repair for a traced Python if?

## Phase project

A compiled batch of gradients.

**Demonstrate:** Compose grad, vmap, and jit; explain what each transformation changes.

Project status: implemented staged practice · [Open source](../../projects/foundation-toolkit/README.md). Use stages 3 for this phase. Copy `projects/foundation-toolkit/starter/toolkit.py` to `projects/foundation-toolkit/my_toolkit.py` and write your code in `projects/foundation-toolkit/my_toolkit.py`. Run `python3 projects/foundation-toolkit/tests/check.py --implementation projects/foundation-toolkit/my_toolkit.py --stage 3` from the top-level folder to verify stage 3.



[Primary documentation](https://docs.jax.dev/en/latest/beginner_guide.html).
