# Release a trained computation with a deployment audit

Train and adapt a small model, preserve its checkpoint identity, export inference computations, compare explicit precision policies and measure a complete local JSON request. Then build a capacity hypothesis whose assumptions are separate from the measured results.

This project runs on CPU. Its request function is an in-process service boundary; no HTTP server, network load test, autoscaler or edge device is claimed. The six real JAX export artifacts cover fixed batches one and eight under three policies. W8A8 performs explicit integer accumulation, but no optimized low-bit device kernel or speedup is claimed.

## Start from a learner workspace

Complete deployment-01 through deployment-06 and their prerequisites. In the top folder of the course workspace or extracted project ZIP, activate the course CPU environment and install requirements-cpu.txt. Copy the starter:

~~~sh
cp projects/deployment-audit/starter/model.py projects/deployment-audit/my_model.py
~~~

PowerShell:

~~~powershell
Copy-Item projects/deployment-audit/starter/model.py projects/deployment-audit/my_model.py
~~~

Implement each function incrementally. Keep a release report with actual environment versions, command output, artifact and data hashes, chosen tolerances, figure interpretations and failures.

## Stage 1: identify the model being shipped

Implement objective and train. Use stable binary cross-entropy with binary or soft targets. The public checks pretrain a three-parameter logistic model, save and reload the resulting weights, then adapt to a changed target probability rule. This is genuine training on a transparent synthetic fixture.

Compare the gradient with the analytic residual expression using host NumPy. Verify zero-step behavior and that adaptation does not mutate the saved base. A checkpoint without optimizer state is a parameter starting point, not proof of an exact training resume.

~~~sh
python3 projects/deployment-audit/tests/check.py --implementation projects/deployment-audit/my_model.py --stage 1
~~~

**Keep:** source and adaptation data hashes including targets, base checkpoint hash, objective, update count, dtype, task evaluation and source-retention evidence. The fixed checks establish mechanics; use an untouched evaluation split in your report.

## Stage 2: package and audit actual runtime artifacts

Implement quantize, prepare and load. The precision policies are:

| Policy | Stored weights | Activations | Accumulator | Output |
| --- | --- | --- | --- | --- |
| fp32 | FP32 | FP32 | FP32 | FP32 |
| w8a32 | INT8 plus per-output scales | FP32 | FP32 after dequantization | FP32 |
| w8a8 | INT8 plus per-output scales | Symmetric INT8 using calibration scale | INT32 | Dequantized FP32 |

The INT4 exercise stores codes in INT8 containers; no nibble packing or four-bit memory saving is claimed.

Write weights.npy and manifest.json. The manifest records feature count, supported batches, preprocessing, runtime version/platform, complete precision policy, calibration hash, base/source/adaptation provenance and hashes for every serialized computation. Export and deserialize all six policy/batch combinations with the real JAX export API.

Implement hash verification before loading artifacts. Use only training/calibration rows to set activation ranges; evaluation probes remain separate.

~~~sh
python3 projects/deployment-audit/tests/check.py --implementation projects/deployment-audit/my_model.py --stage 2
~~~

**Keep:** actual serialized bytes, reload parity on independent inputs, integer-arithmetic reference, rounding bounds, corruption rejection and clipping counts. The public fixture uses an absolute output error threshold of 0.04; choose task-specific thresholds before testing a real model, and measure quality as well as numeric output error. A matching file hash establishes identity, not model quality.

## Stage 3: measure a declared request boundary

Implement request, benchmark and simulate. The in-process request includes JSON decoding, shape/value validation, host-to-device placement, the restored computation, completion and JSON encoding. Reject oversized payloads, unsupported batch shapes, nonnumeric or nonfinite features and unknown policies. Report clipping for W8A8.

Warm each fixed batch shape independently, retain all timing samples, and report first-request time and warmed p50/p95. Compute example throughput from batch size and elapsed milliseconds. Do not call amortized cost individual request latency.

~~~sh
python3 projects/deployment-audit/tests/check.py --implementation projects/deployment-audit/my_model.py --stage 3
~~~

**Keep:** request parity, actual timing samples, units, boundary declaration and a separate queue simulation. Check arrivals at 0, 1 and 4 milliseconds with two-millisecond service by hand; response times are 2, 3 and 2 milliseconds. Include a burst and an overloaded schedule. A replica estimate remains hypothetical until a real service load test measures routing, networking, startup and sustained concurrency.

For production or edge deployment, follow the runtime/operator/precision and device-validation contract from deployment-06. Desktop integer arithmetic and local export compatibility requires separate verification to establish NPU execution or device latency.

## Reference

~~~sh
python3 projects/deployment-audit/tests/check.py --implementation solution --stage all
~~~

The public reference prints measured local request times and clearly labels missing network/device measurements. Passing reference checks is verified separately from independent learner work or professional readiness. Complete the ship synthesis assessment to review the complete release decision.
