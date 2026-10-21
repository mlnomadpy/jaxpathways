# Bounded Cloud TPU qualification, 2026-10-20

This report records real, isolated TPU runs on one v5e1 device, four v5e devices in a single host, and a four-VM-worker sixteen-device v5e Pod. It qualifies only the targets and configurations listed here. The multi-host result is a small synthetic recovery test, not evidence that every course lesson runs on TPU or that a production model is trained or accelerated.

## Environment and lifecycle

The run used Python 3.12.13, JAX 0.9.2, libtpu 0.0.37 and NumPy 2.4.4 on one physical TPU v5 lite device. The source commit recorded with the evidence is `c0e927650147232ef28b2a1cb5b22a11145862d2`. The pinned minimal environment is `resources/tpu-gcp/requirements-qualification.txt`; it also includes `absl-py` 2.4.0 for the Pallas imports.

The single-device launcher completed training, resumed from its saved state, and completed the precision run. Each target process exited with code 0. The execution manifest is [execution-tpu.json](evidence/tpu-qualification-2026-10-20/v5e1/smoke-v1/execution-tpu.json). The physical VM and queued-resource request were deleted after the evidence was copied.

The successful four-VM-worker recovery run used Python 3.12.13, JAX 0.9.2, libtpu 0.0.37 and NumPy 2.4.4. Its recorded environment also pins Flax 0.12.6, Orbax Checkpoint 0.11.34, gcsfs 2026.10.1 and TensorStore 0.1.82. The source commit was `b71fe83679327da4df0bcc397796cd3d17ba44dc`; every process reported the same source and lesson hashes. The four process receipts are under [v5e16](evidence/tpu-qualification-2026-10-20/v5e16/). The TPU VM workers, queued resource, private synthetic checkpoint bucket and temporary service account were deleted after downloading their evidence.

## Precision target

The precision runner evaluated a synthetic 64×128 by 128×64 dot for balanced inputs and a feature-zero outlier, using FP32, BF16 and symmetric INT8 operands. Outputs were compared against a NumPy float64 dot of the original FP32 inputs. All six target checks passed. Each reported timing is a median of five warmed, synchronized calls; the boundary excludes transfer, quantization and compilation.

| Input | Operand policy | Maximum absolute error | Median time (ms) |
| --- | --- | ---: | ---: |
| Balanced | FP32 | 7.14e-7 | 0.148 |
| Balanced | BF16 | 0.0109 | 0.134 |
| Balanced | INT8 | 0.0352 | 0.135 |
| Outlier | FP32 | 3.21e-6 | 0.132 |
| Outlier | BF16 | 0.0498 | 0.131 |
| Outlier | INT8 | 0.423 | 0.132 |

These results characterize this synthetic dot only. The outlier increases INT8 error; the measured times do not establish a speedup for a model or a different shape. Raw samples, input hashes, compiler reports and lowered HLO are in [smoke-v1/precision](evidence/tpu-qualification-2026-10-20/v5e1/smoke-v1/precision/). The folder also retains an XPlane profile and Perfetto trace.

## Pallas target

The first Pallas attempt is retained in `pallas-v1/` and failed during import because `absl` was unavailable. With `absl-py` 2.4.0 installed, the follow-up run passed the fused and pipelined kernels in FP32 and BF16. All cases used a 257×513 logical shape, an 8×128 block, two input buffers for the pipeline, one TPU v5 lite device and five timing samples. The maximum absolute error was 0.0 against the represented-input NumPy oracle.

| Kernel | Dtype | Candidate median (ms) | Jitted baseline median (ms) |
| --- | --- | ---: | ---: |
| Fused | FP32 | 0.150 | 0.128 |
| Fused | BF16 | 0.152 | 0.118 |
| Pipeline | FP32 | 0.154 | 0.138 |
| Pipeline | BF16 | 0.159 | 0.131 |

The custom-kernel medians were slower than their jitted baselines in all four cases. This tiny, non-tile-aligned workload may be dominated by dispatch and padding. It is not a general performance conclusion. Each raw receipt records the implementation hash, target device, configuration, samples and measurement boundary in [pallas-v2](evidence/tpu-qualification-2026-10-20/v5e1/pallas-v2/).

## Four-device single-host collectives

A separate run at source commit `53f1be134a94858400bdf8331c628e66db6fa26b` used one TPU v5 lite host with four local JAX devices. Python 3.12.13, JAX 0.9.2, libtpu 0.0.37 and NumPy 2.4.4 were recorded in the receipt. The receipt's `sourceSha256` and `baseLessonSha256` match the target script and distributed lesson at that commit; its deterministic fixture hash identifies the host-side inputs.

The test placed 16 rows of eight features across a four-device `data` mesh (four rows per device), sharded the labels the same way, and replicated the length-eight weights. It computed the full-batch gradient using `jax.lax.psum`, then independently computed it with `psum_scatter` followed by `all_gather`. Both matched the NumPy float32 reference at absolute and relative tolerances of 2e-6. The lowered StableHLO contains `all_reduce`, `reduce_scatter` and `all_gather`; both matrix products explicitly request `HIGHEST` precision. Placement checks confirmed four 4×8 input shards, four length-four label shards, replicated length-eight all-reduce output and four length-two reduce-scatter output shards.

The recorded medians below are seven warmed, synchronized calls per operation. They exclude compilation, input placement and trace overhead. The arrays and computation are deliberately small, so these numbers describe this fixture and runtime only; they are not a scaling or speedup result.

| Operation | StableHLO collective | Result placement | Median (ms) |
| --- | --- | --- | ---: |
| Full-batch gradient | `all_reduce` | Replicated length-8 vector | 0.25544 |
| Gradient shards | `reduce_scatter` | Length-2 vector per device | 0.24606 |
| Reconstruct gradient | `all_gather` | Replicated length-8 vector | 0.21951 |

The archived [execution receipt and StableHLO](evidence/tpu-qualification-2026-10-20/v5e4/collectives-v1/) include the tolerances, fixture/source hashes, placement checks and seven raw timing samples. The same folder retains an XPlane profile and Perfetto trace. The physical VM and queued-resource request were deleted after downloading the evidence. This qualifies a single-host four-device collective path only. The separate multi-host checkpoint test below has its own target and receipts.

## Four-host sixteen-device synthetic checkpoint recovery

At source commit `b71fe83679327da4df0bcc397796cd3d17ba44dc`, four TPU v5e VM workers ran one JAX process each. All four process receipts report a TPU backend, four local devices and sixteen global devices. Cloud worker-to-process mapping was worker 0→process 0, worker 1→process 2, worker 2→process 1 and worker 3→process 3. The matching source SHA-256 is `d8dc46e58f869a94c9f0fc3c18b4098427cd44c9fa35ab864f4507270c726820`; all processes also agree on the lesson and fixture hashes.

This was a deterministic synthetic linear workload, not a language or production model. A 64×8 fixture was partitioned into four disjoint 16-row ranges, with four rows per device. The gradient reduction used `jax.lax.Precision.HIGHEST`. Training carried momentum and a step counter; Orbax Checkpoint saved step 4 to a `gs://` location and waited for completion. An abstract sharded restore succeeded, and replay to step 8 matched an independent NumPy calculation for the final weights and momentum on every process. The final receipts record exit status 0 for all four processes.

The synthetic reference loss fell from 0.56445 to 0.01906 by step 6, then rose to 0.05904 at step 8. This is a state-save/restore and numerical replay check, not a quality result. It did not kill a worker, exercise elastic recovery, restore in a fresh process, or test model-scale training, throughput or performance. The four receipts, per-worker logs, environments and failed attempts are retained under [v5e16](evidence/tpu-qualification-2026-10-20/v5e16/). Attempts exposed missing Flax, then missing gcsfs, then a GCS bucket path/layout issue with the extended client. The successful v4 used the standard GCS client and Object Admin access. Temporary identifiers in published logs are documented in `publication-redactions.json`.

## Retained checkpoint and CPU portability

The published [checkpoint files](evidence/tpu-qualification-2026-10-20/v5e16/checkpoint-step-4/) preserve the tensors and Orbax metadata. Only the deleted temporary bucket identifier in the completion marker is replaced. [Original and published hashes](evidence/tpu-qualification-2026-10-20/v5e16/checkpoint-files.json) and the [redaction record](evidence/tpu-qualification-2026-10-20/v5e16/publication-redactions.json) document that change; console logs also replace temporary cloud identifiers.

A fresh local CPU process restored this published checkpoint into explicitly selected CPU sharding. Its step, weights and momentum matched both the recorded restored state and an independently evaluated NumPy calculation. The [CPU receipt](evidence/tpu-qualification-2026-10-20/v5e16/checkpoint-cpu-restore.json) records the source hash and result. Recheck it from the repository root in the course CPU environment:

```sh
python docs/evidence/tpu-qualification-2026-10-20/v5e16/restore-on-cpu.py
```

This establishes portability of this small checkpoint to CPU. The TPU test restored inside the original worker processes; it did not restart a new TPU worker group or inject a worker failure.

## Qualification boundary

This evidence covers one single-device training/resume path, one precision script configuration with six numerical cases, four small Pallas configurations, one four-device single-host collective target and one four-VM-worker sixteen-device synthetic checkpoint-recovery run. It does not cover all 103 lessons: some lesson scripts explicitly select CPU, and others have dependencies or target requirements outside these pinned environments. The multi-host result does not cover worker-loss or elastic recovery, a fresh-process TPU restart, a trained production model, model-scale training, end-to-end throughput, or general performance of BF16, INT8, Pallas or collectives.

Keep CPU simulation evidence and real TPU receipts separate. For a new claim, preserve the source revision, pinned environment, actual backend/device inventory, exact workload and precision policy, independent numerical comparison, synchronized timing boundary and complete trace or failure output.
