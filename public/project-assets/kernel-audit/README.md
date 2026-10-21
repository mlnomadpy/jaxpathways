# Audit a Pallas kernel and its target evidence

Implement real Pallas grid and pipeline operations, defend their numerical contracts, and keep CPU simulation separate from accelerator evidence. Complete the Pallas lessons after synchronized benchmarking and sharding preparation. CPU simulation remains the default local path. A bounded real-target qualification now covers four fused/pipeline cases on one v5e chip; it establishes neither broad shape support nor a performance win. See [the dated TPU qualification report](../../docs/tpu-qualification-2026-10-20.md) and the raw manifests in `docs/evidence/tpu-qualification-2026-10-20/v5e1/pallas-v2/`.

## Prepare the CPU workspace

Install the course `requirements-cpu.txt` in your environment. The tested core uses JAX 0.9.2; Pallas is experimental, so inspect and retest APIs when changing versions. Run from the checkout or extracted bundle's top folder:

```sh
# Copy the starter template into your editable workspace file
cp projects/kernel-audit/starter/model.py projects/kernel-audit/my_model.py
python3 projects/kernel-audit/tests/check.py --implementation projects/kernel-audit/my_model.py --stage 1
```

PowerShell users can use `Copy-Item`. Keep your implementation separate from `solution/model.py`.
Keep this CPU environment for interpretation and local checks. Optional plotting tools can live in a separate CPU plotting environment; neither is the pinned TPU qualification stack.

## Stage 1: grid, ownership and tails

Implement `blocked_axpy`: matching nonempty matrices, actual Pallas Ref arithmetic for twice the first input plus the second, explicit padding, ceiling grid and logical cropping. Support float32 and bfloat16 with float32 arithmetic and one final cast. Reject undocumented input broadcasting and dtype coercion.

The checker uses singleton rows, divisible shapes, nondivisible tails and two block choices. Compare every value with NumPy and the declared output cast; do not settle for an average. Save an ownership diagram and the last logical row/column checks. Explain why zero padding is valid for this elementwise operation but does not automatically work for a reduction.

## Stage 2: a real target-oriented fused operation

Implement `fused_bias_relu`, with matrix activations and a vector bias that follows the column axis. The matrix and bias need different BlockSpecs. Test cancellations at the ReLU boundary, distinct column biases, logical tails and both precisions.

```sh
# Run run command in terminal using the course Python environment
python3 projects/kernel-audit/tests/check.py --implementation projects/kernel-audit/my_model.py --stage 2
```

Expose explicit `interpret` and `tpu` modes. Target mode must require a real TPU backend/placed inputs and pass `interpret=False`; it must never retry through interpretation. The recorded target run covers one logical shape, 257×513, with an 8×128 block and both precisions. Treat other target shapes, tile sizes and dtypes as unqualified until separately checked; this limit does not prohibit exploring them with new correctness evidence.

## Stage 3: pipeline and evidence boundary

Implement `pipelined_axpy` using actual `pltpu.emit_pipeline`. The outer call holds global references; the inner pipeline manages local block buffers and transfers. CPU simulation uses `pltpu.InterpretParams` with race detection and explicit simulated TPU layout metadata. Record the actual backend as CPU.

Compare synchronous copies, two input slots and three input slots while keeping two output slots. Installed JAX 0.9.2 does not support more than two output slots. Preserve representative tails and both dtypes. Reject invalid block/buffer/mode requests.

```sh
# Run run command in terminal using the course Python environment
python3 projects/kernel-audit/tests/check.py --implementation projects/kernel-audit/my_model.py --stage all
```

Implement `target_benchmark` with actual TPU guards, separate compilation, five warmups, repeated synchronized execution and a clear boundary. CPU tests check the refusal path; they do not qualify target compilation or performance. Do not time interpretation and call it a kernel benchmark.

## Run actual target evidence on an available TPU

Use Python 3.12.13 in a fresh configured TPU environment, separate from the CPU test and optional plotting environments. From the repository root, install the pinned minimal stack:

```sh
python3.12 -m venv .tpu-qualification
.tpu-qualification/bin/python -m pip install -r resources/tpu-gcp/requirements-qualification.txt
```

It pins JAX 0.9.2, libtpu 0.0.37, NumPy 2.4.4 and `absl-py` 2.4.0 for the qualification. The first recorded Pallas attempt stopped at import because `absl` was missing; that failure is retained in `pallas-v1/`. Adding `absl-py` produced the passing `pallas-v2/` target receipts. Do not use a process forced to `JAX_PLATFORMS=cpu` for the target lab. This project does not provision hardware or spend cloud credits.

```sh
# Run run command in terminal using the course Python environment
.tpu-qualification/bin/python projects/kernel-audit/target_tpu.py --implementation projects/kernel-audit/my_model.py --kernel fused --shape 257 513 --block 8 128 --dtype float32
.tpu-qualification/bin/python projects/kernel-audit/target_tpu.py --implementation projects/kernel-audit/my_model.py --kernel fused --shape 257 513 --block 8 128 --dtype bfloat16
.tpu-qualification/bin/python projects/kernel-audit/target_tpu.py --implementation projects/kernel-audit/my_model.py --kernel pipeline --shape 257 513 --block 8 128 --buffers 2 --dtype float32
.tpu-qualification/bin/python projects/kernel-audit/target_tpu.py --implementation projects/kernel-audit/my_model.py --kernel pipeline --shape 257 513 --block 8 128 --buffers 2 --dtype bfloat16
```

The runner checks the real backend before allocation, places inputs on a TPU, checks the complete logical output against an independent represented-input oracle, then benchmarks the candidate and a jitted JAX baseline. It prints JSON only after successful checks, including source hash, configuration, actual device, precision, correctness, compilation time and timing samples. On CPU it exits with status two and produces no target receipt.

The measurement boundary starts with already-placed logical inputs and ends with a synchronized logical output. Candidate padding/cropping is included; compilation and explicit host-to-device placement are excluded. The baseline uses the same precision contract. A hand-written kernel may lose to XLA fusion; retain that result rather than selecting only favorable cases.

The recorded one-chip v5e run passed all four combinations with maximum absolute error 0.0 against the represented-input NumPy oracle. Median candidate / jitted-baseline times were 0.150 / 0.128 ms (fused FP32), 0.152 / 0.118 ms (fused BF16), 0.154 / 0.138 ms (pipeline FP32) and 0.159 / 0.131 ms (pipeline BF16), using five samples per case. All four candidate medians were slower in this tiny 257×513 case. Dispatch and padding may dominate at this shape; this observation is not a general speed comparison. The recorded implementation hash and exact configuration are in the `pallas-v2/` JSON receipts.

## Keep and interpret evidence

Save code, environment, commands, numerical references, ownership maps, frozen data seeds, precision policy, a failure/repair, padding counts, modeled buffer costs and any actual target receipts. Distinguish configuration-derived estimates from device counters and timing observations. Connect the local result to a profile and partitioning decision before claiming an end-to-end improvement.

```sh
# Run run command in terminal using the course Python environment
python3 projects/kernel-audit/tests/check.py --implementation solution --stage all
```

Public CPU stage passes are bounded correctness evidence. They are not TPU validation, a performance win, independent expert approval or a credential. Continue to the [scale synthesis assessment](../../assessments/scale.md), which requires profiling and partitioning reasoning in addition to a local kernel.
