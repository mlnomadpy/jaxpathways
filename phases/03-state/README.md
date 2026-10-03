# Phase 03: State, randomness & control flow

Shared foundations.

**Prerequisites:** 02: JAX transformations.

**Hardware:** CPU.

## Lesson sequence

### 03.01 Random keys without surprises

Status: planned brief.

**Learn and build:** Split keys and show reproducibility without accidental key reuse.

**Evidence:** A reproducible experiment for “Random keys without surprises”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Random keys without surprises”, identify one failure case, and show how you verified the fix.

### 03.02 Pytrees and structured parameters

Status: planned brief.

**Learn and build:** Represent parameters and state in a nested structure and transform its leaves.

**Evidence:** A reproducible experiment for “Pytrees and structured parameters”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Pytrees and structured parameters”, identify one failure case, and show how you verified the fix.

### 03.03 Compiled loops with lax.scan

Status: planned brief.

**Learn and build:** Replace a fixed-length Python loop with scan and compare final states.

**Evidence:** A reproducible experiment for “Compiled loops with lax.scan”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Compiled loops with lax.scan”, identify one failure case, and show how you verified the fix.

### 03.04 Branches with lax.cond

Status: planned brief.

**Learn and build:** Implement a data-dependent branch that works inside a compiled function.

**Evidence:** A reproducible experiment for “Branches with lax.cond”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Branches with lax.cond”, identify one failure case, and show how you verified the fix.

## Phase project

A deterministic batched simulation.

**Demonstrate:** Replay the same experiment with saved state and keys; explain the update order.

Project status: planned brief.

[Primary documentation](https://docs.jax.dev/en/latest/beginner_guide.html).
