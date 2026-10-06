# Reviewer notes: scale synthesis

This assessment spans profiling, partitioning and kernels. Do not accept a kernel-only report as proof of the full route outcome.

For local mean gradients \(g_1,g_2\) from counts three and five, the global mean is \((3g_1+5g_2)/8\). For a constructed scalar example \(g_1=2\), \(g_2=-1\), it is \(0.125\), while the unweighted mean is \(0.5\). Require an actual changed-partition workload check in addition to this arithmetic.

For candidate time fraction \(p\) and local speedup \(s\), the conditional end-to-end speedup is \(1/((1-p)+p/s)\). With \(p=0.3\), \(s=2\), it is approximately \(1.17647\). Removing the local cost entirely gives an upper bound of approximately \(1.42857\), under the assumption that the remaining work is unchanged. These are analytic estimates, not measured outcomes.

A represented-input reference must use the stored input dtype, the declared compute dtype and the final output cast. A bfloat16 result can match that oracle exactly while differing from the original float32 dataset. Require both meanings to be explained before accepting a tolerance change.

The kernel project models float32 data-buffer bytes as \((2n_{\text{input buffers}}+2)B_mB_n\times4\): two input arrays with the selected number of slots, one output with two slots. Installed JAX 0.9.2 rejects more than two output slots. This model omits semaphores, compiler scratch, layout padding and register use; do not call it measured VMEM consumption.

Inspect the real-target receipt for an actual TPU backend/device and `interpret=False` execution. The target runner refuses CPU and produces no successful receipt on its refusal path. Simulated TPU layout metadata is not an actual device identification. CPU simulation can establish numerical semantics and some synchronization checks, but no target lowering or speed claim.

Timing evidence must compare equivalent completed work and include the selected padding/cropping boundary. Warmup and compilation are separate. The ninetieth percentile is not the maximum. A local speedup should be re-evaluated inside the complete step; changing the partitioning, batch size or numerical objective can invalidate the original comparison.

Accept a well-supported choice to retain ordinary JAX. If no actual target is available, accept completed CPU preparation only and mark target qualification unresolved. Neither an authored lab nor passing its public fixtures implies independent hardware or expert validation.

The independent public partition reference is executable in a fresh CPU process:

```sh
python3 projects/kernel-audit/tests/check-scale-reference.py
```

It checks a fixed seeded global gradient, the three/five count-weighted repair, one update and actual placement over four logical CPU devices. This is a numerical reference, not a network benchmark or a substitute for the learner’s trace and changed-workload evidence.
