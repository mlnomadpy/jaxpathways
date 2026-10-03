# Phase 02: JAX transformations

Shared foundations.

**Prerequisites:** 01: Arrays & pure functions.

**Hardware:** CPU.

## Lesson sequence

### 02.01 Your first gradient

Status: authored sample.

**Learn and build:** Differentiate x², predict the derivative of x³, and check the result.

**Evidence:** A reproducible experiment for “Your first gradient”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Your first gradient”, identify one failure case, and show how you verified the fix.

### 02.02 Losses and value_and_grad

Status: planned brief.

**Learn and build:** Compute a scalar loss and its parameter gradients in one call.

**Evidence:** A reproducible experiment for “Losses and value_and_grad”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Losses and value_and_grad”, identify one failure case, and show how you verified the fix.

### 02.03 Batch a function with vmap

Status: planned brief.

**Learn and build:** Replace an explicit example loop with vmap and check equivalence.

**Evidence:** A reproducible experiment for “Batch a function with vmap”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Batch a function with vmap”, identify one failure case, and show how you verified the fix.

### 02.04 Compile a function with jit

Status: planned brief.

**Learn and build:** Compare first-call and repeated-call timing with correct synchronization.

**Evidence:** A reproducible experiment for “Compile a function with jit”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Compile a function with jit”, identify one failure case, and show how you verified the fix.

### 02.05 Tracing, static arguments, and recompilation

Status: planned brief.

**Learn and build:** Vary shapes and configuration and explain when a new compiled program is needed.

**Evidence:** A reproducible experiment for “Tracing, static arguments, and recompilation”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Tracing, static arguments, and recompilation”, identify one failure case, and show how you verified the fix.

## Phase project

A compiled batch of gradients.

**Demonstrate:** Compose grad, vmap, and jit; explain what each transformation changes.

Project status: planned brief.

[Primary documentation](https://docs.jax.dev/en/latest/beginner_guide.html).
