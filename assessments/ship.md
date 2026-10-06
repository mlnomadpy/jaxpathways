# Shipping synthesis: justify a model release

**Scope:** review draft after the six deployment lessons and deployment-audit project. This assessment covers adaptation, artifact identity, precision, actual local runtime behavior and an explicitly hypothetical service plan. A CPU release audit does not establish a cloud service or edge device deployment.

Create a release folder and a report. Keep original and adapted checkpoints, manifests, exact commands, independent checks, timing samples and plots. State which evidence was measured and which results came from simulation.

## Task 1: compare post-training objectives

Actually train a base model, save it, hash it and reload it. A reproducibly pretrained small synthetic model is acceptable when clearly labeled. Random weights are not a pretrained artifact.

From identical reloaded weights compare supervised fine-tuning with a teacher-based objective on the same adaptation inputs. State where labels and teacher outputs come from. Retain source, adaptation, calibration, validation and final evaluation identifiers.

Derive the stable loss and at least one gradient independently. Evaluate all candidates on the same untouched task criterion and report source-task retention. Deliberately corrupt the teacher or change its confidence and explain the effect. Distinguish this experiment from preference learning or reinforcement learning.

**Keep:** pretraining improvement, saved-base identity, supervision contracts, numerical reference, target/source metrics and teacher-failure diagnosis.

## Task 2: verify bytes and runtime semantics

Export an adapted computation under two declared fixed batch shapes. Serialize and reload the artifact before comparing outputs with independent NumPy calculations.

Build FP32, weight-only INT8 and calibrated weight/activation INT8 policies. Record weight, activation, multiplication, accumulator and output conventions. Use an explicit INT32 reference for the integer accumulation and verify how scales reconstruct the output.

1. Compare at least three changed inputs, including a low-margin decision and an activation beyond the calibrated range.
2. Retain calibration provenance separately from final evaluation inputs.
3. Corrupt a serialized artifact and show that hash validation rejects it.
4. Attempt a malformed shape and nonfinite feature and show an explicit boundary error.
5. Explain why simulated INT4 codes in INT8 containers do not deliver packed four-bit storage.

**Keep:** six actual artifacts, hashes, signatures, precision table, output and task-quality errors, clipping counts and rejected requests. Numerical parity does not prove target-device acceleration.

## Task 3: measure the request that exists

Warm each supported batch shape independently and measure the complete implemented request boundary. Retain first-request time and at least thirty warmed samples, with units and runtime details.

Report p50 and p95 milliseconds per batch and examples per second. Explain the difference between \(t_B\), \(t_B/B\) and \(1000B/t_B\), where \(t_B\) is batch time in milliseconds.

Draw the timing distributions or a clearly labeled summary. Explain the observed values and avoid asserting that a larger batch must improve every metric. Name omitted costs such as networking or server queueing. Do not turn a tiny timing sample into a production tail guarantee.

**Keep:** numerical parity before timing, actual sample arrays, timed-boundary declaration and an interpreted figure.

## Task 4: stress a capacity hypothesis

Use a separately labeled serial queue simulation, with arrival times and service durations in milliseconds.

1. Derive starts, finishes and response times for arrivals \((0,1,4)\) with service duration \(2\).
2. Simulate a simultaneous burst and a sustained overload. Explain the difference between a fixed service time and growing response time.
3. With regular arrivals two milliseconds apart, derive the earliest and average collection wait for a full batch of eight.
4. For hypothetical effective capacity \(200\) requests per second per replica, arrival rate \(300\) per second and target utilization \(0.7\), calculate a first replica estimate.
5. Explain why startup time, uneven routing, variable sequence lengths and failures can invalidate the estimate. Specify a real load-test protocol that would verify the plan.

**Keep:** hand timeline, simulation, units, batch-collection calculation and a clearly labeled unexecuted load-test plan.

## Review the release decision

For each task mark **accept**, **revise**, or **not demonstrated**. Acceptance requires agreement among implementation, independent reference, observed result and explanation. The final decision must distinguish a verified local inference artifact from an externally deployed service or target-device result.

After your attempt, read [reviewer notes](ship-reviewer.md).

## Container delivery extension

After `deployment-07`, use the [engineering release project](../project.html?id=engineering-release) to move an actually tracked model into the supplied container boundary. Record the resolved base image, resulting image identity, artifact checksum and preprocessing/signature. Verify singleton and changed-batch predictions against the source model, readiness after loading, non-root execution and a rejected wrong digest. Retain the real Docker output separately from local Python-process checks.

Describe a CI sequence that builds once, evaluates the same immutable artifact and promotes that identity. Distinguish a registry alias change from a rollout. State the missing concurrency, authentication, overload and accelerator/edge checks before describing the fixture as a production service.
