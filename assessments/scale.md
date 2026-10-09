# Scale synthesis: decide whether a custom kernel improves a sharded workload

**Scope:** numerical equivalence, profiling, partitioning, and a bounded Pallas kernel optimization decision building on the performance, distributed, and kernel lessons plus the [kernel audit project](../project.html?id=kernel-audit).

Keep CPU logical-device and interpretation evidence separate from real accelerator evidence. A CPU-only submission can complete the numerical and decision-preparation tasks; it must leave target performance qualification explicitly unresolved. An implemented target runner is not an executed target result.

## Task 1: freeze a workload and its reference

Use a regression step with feature matrix \(X\), weights \(w\), targets \(y\) and the global mean objective

\[
L(w)=\frac1N\lVert Xw-y\rVert^2,\qquad \nabla L(w)=\frac2N X^\mathsf{T}(Xw-y).
\]

Choose dimensions that produce both even and uneven logical batch partitions; record a fixed NumPy seed and data-generation rule. Derive an independent host loss/gradient reference. Verify one update and a short fixed update sequence before timing. State compute, parameter and output precision separately.

Freeze this workload before comparing implementations. Add an elementwise affine or bias/ReLU stage only if it represents a named part of the workload, rather than inventing extra work to make a custom kernel appear relevant.

**Keep:** shape/objective contract, frozen inputs, independent gradient/update checks and precision policy.

## Task 2: explain partitioning and communication

Use the distributed lessons to partition examples while preserving the global objective. State which arrays are sharded and which are replicated; draw local shapes and explain any communication. Compare the sharded update with the host reference, then change the partition or example order and repeat.

Construct an uneven partition with counts three and five. Demonstrate why an unweighted average of local mean gradients is generally wrong, and derive the count-weighted repair. Do not infer network performance from logical CPU devices. If using actual multi-host hardware, record topology and process/device counts independently.

**Keep:** partition diagram, placement observations, numerical comparison and a reproduced aggregation bug with repair.

## Task 3: profile the complete step before selecting a kernel

Measure compilation separately from warm execution, synchronize completed work, and collect repeated samples. Save a trace for the complete step using the profiling lesson's supported procedure. Identify whether time appears in local arithmetic, input staging, resharding, communication, or another region; distinguish measured regions from inferred causes.

Suppose a trace attributes \(30\%\) of the step to the candidate local operation. Derive the conditional end-to-end speedup if it becomes twice as fast and the upper bound if its cost vanished. Explain why a faster local kernel can leave most of the full step unchanged.

If the trace provides no persuasive local bottleneck, a reasoned decision to keep ordinary JAX is an acceptable outcome. A custom kernel should answer an observed problem rather than serve as a mandatory decoration.

**Keep:** raw samples, warmup/compilation boundary, trace, bottleneck interpretation and conditional opportunity calculation.

## Task 4: test a candidate across actual contracts

Use the project kernel relevant to the chosen operation. Add frozen fixtures of shapes \((15,255)\), \((16,256)\), \((17,257)\) and a singleton. Check both float32 and bfloat16 against a represented-input oracle, with the precision gap to original float32 data reported separately. Verify logical output shape, tails, activation boundaries when applicable, and one input permutation invariant.

Compare two tile shapes and two input-buffer counts, keeping two output slots under the tested JAX emitter contract. Record program counts, padding and a clearly labeled model of data-buffer storage. Execute CPU interpretation/simulation to validate semantics, including explicit rejection of unsupported modes and dtypes.

On available real TPU hardware, use the target-only runner to obtain completed target correctness and performance receipts. Compare the same logical-input-to-logical-output boundary with the strong jitted baseline. Record target failures as failures; never retry silently in simulation. Report all selected configurations, including regressions.

**Keep:** numerical matrix, resource model, supported-contract checks, and actual target receipts or an explicit unverified-target statement.

## Task 5: make an optimization decision

Create a figure that relates the actual observations to the decision. For CPU-only work, plot correctness/precision or configuration counts and label their scope. For target work, plot median and distribution information from saved timing samples, with axes and units. Explain representative values, tied/zero results, uncertainty and omitted costs.

If a target candidate wins locally, remeasure the complete sharded step and compare with the conditional estimate. Keep the same correctness check and global objective. Explain whether communication or data movement now dominates, and whether compilation or maintenance costs change the practical value.

Conclude with **keep**, **revise**, **use the baseline**, or **await target evidence**, justified by the actual data. No particular speedup is required; a defensible decision is.

## Reviewer decision

- **Accept mathematics:** Independent loss, gradient and update references survive changed partitions and uneven counts. **Revise:** only final loss decreases or local means are averaged without counts.
- **Accept profiling:** Compilation, warm execution and synchronization are explicit; the trace supports the stated bottleneck. **Revise:** dispatch time, logical-device timing or IR length is presented as accelerator throughput.
- **Accept kernel correctness:** Actual Pallas semantics, tails, precision policy, representative shapes and unsupported contracts are tested. **Revise:** only a divisible fixture passes, padding leaks into output or mixed precision is left ambiguous.
- **Accept target qualification:** The actual device, non-interpret execution, numerical result, baseline and timing boundary are documented. **Revise:** a CPU simulation is called TPU execution. Without hardware, mark this criterion unresolved.
- **Accept decision:** Local and end-to-end evidence support the recommendation and remaining uncertainty is visible. **Revise:** only favorable configurations are shown or a performance claim exceeds its hardware/workload evidence.

Save your source code, environment specification, data-generation rules, verification checks, trace files, figure data, timing records, and decision report in your portfolio.
