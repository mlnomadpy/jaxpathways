# Phase 07: Transformers & language models

Training systems.

**Prerequisites:** 06: Data & checkpoint recovery.

**Hardware:** CPU for small blocks; TPU or GPU for training experiments.

## Lesson sequence

### 07.01 Attention from small pieces

Status: planned brief.

**Learn and build:** Implement scaled dot-product attention and test its output shape.

**Evidence:** A reproducible experiment for “Attention from small pieces”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Attention from small pieces”, identify one failure case, and show how you verified the fix.

### 07.02 Masking, tokenization, and sequence packing

Status: planned brief.

**Learn and build:** Construct a causal mask and prevent packed sequences from attending across boundaries.

**Evidence:** A reproducible experiment for “Masking, tokenization, and sequence packing”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Masking, tokenization, and sequence packing”, identify one failure case, and show how you verified the fix.

### 07.03 A Transformer block and mixed precision

Status: planned brief.

**Learn and build:** Build a residual attention block and compare numerical behavior across precision choices.

**Evidence:** A reproducible experiment for “A Transformer block and mixed precision”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “A Transformer block and mixed precision”, identify one failure case, and show how you verified the fix.

### 07.04 Train, checkpoint, and generate

Status: planned brief.

**Learn and build:** Train a tiny causal model, restore it, and generate from a fixed seed.

**Evidence:** A reproducible experiment for “Train, checkpoint, and generate”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Train, checkpoint, and generate”, identify one failure case, and show how you verified the fix.

## Phase project

A checkpointed causal language model.

**Demonstrate:** Verify masking, sequence handling, restoration, and generation before comparing speed.

Project status: planned brief.

[Primary documentation](https://flax.readthedocs.io/).
