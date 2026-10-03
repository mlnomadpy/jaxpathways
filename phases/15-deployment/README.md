# Phase 15: Inference & model adaptation

Specializations.

**Prerequisites:** 06: Data & checkpoint recovery; 08: Performance diagnosis.

**Hardware:** CPU for bridges; deployment-specific hardware for serving.

## Lesson sequence

### 15.01 Keras and PyTorch bridges to explicit JAX

Status: planned brief.

**Learn and build:** Map familiar model and optimizer state to explicit JAX concepts; distinguish the PyTorch/XLA route.

**Evidence:** A reproducible experiment for “Keras and PyTorch bridges to explicit JAX”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Keras and PyTorch bridges to explicit JAX”, identify one failure case, and show how you verified the fix.

### 15.02 Adapt a pretrained model and choose a post-training objective

Status: planned brief.

**Learn and build:** Separate supervised adaptation from preference or RL objectives and specify an evaluation plan.

**Evidence:** A reproducible experiment for “Adapt a pretrained model and choose a post-training objective”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Adapt a pretrained model and choose a post-training objective”, identify one failure case, and show how you verified the fix.

### 15.03 Export and serve a trained computation

Status: planned brief.

**Learn and build:** Export a computation and verify predictions in the serving environment.

**Evidence:** A reproducible experiment for “Export and serve a trained computation”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Export and serve a trained computation”, identify one failure case, and show how you verified the fix.

### 15.04 Inference capacity, batching, and autoscaling

Status: planned brief.

**Learn and build:** Measure latency and throughput across batch sizes and derive a capacity plan.

**Evidence:** A reproducible experiment for “Inference capacity, batching, and autoscaling”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Inference capacity, batching, and autoscaling”, identify one failure case, and show how you verified the fix.

## Phase project

An inference artifact and capacity plan.

**Demonstrate:** Measure latency and throughput under a stated workload and test the exported artifact.

Project status: planned brief.

[Primary documentation](https://docs.jax.dev/en/latest/).
