# Audit a Pallas kernel and its target evidence

Implement real Pallas grid and pipeline operations, defend their numerical contracts, and keep CPU simulation separate from accelerator evidence. Complete the Pallas lessons after synchronized benchmarking and sharding preparation. The CPU project is runnable now. The included real-TPU runner is implemented but has no executed TPU receipt in this environment.

## Prepare the CPU workspace

Install the course `requirements-cpu.txt` in your environment. The tested core uses JAX 0.9.2; Pallas is experimental, so inspect and retest APIs when changing versions. Run from the checkout or extracted bundle's top folder:

```sh
# Copy the starter template into your editable workspace file
cp projects/kernel-audit/starter/model.py projects/kernel-audit/my_model.py
python3 projects/kernel-audit/tests/check.py --implementation projects/kernel-audit/my_model.py --stage 1
```

PowerShell users can use `Copy-Item`. Keep your implementation separate from `solution/model.py`.

## Stage 1: grid, ownership and tails

Implement `blocked_axpy`: matching nonempty matrices, actual Pallas Ref arithmetic for twice the first input plus the second, explicit padding, ceiling grid and logical cropping. Support float32 and bfloat16 with float32 arithmetic and one final cast. Reject undocumented input broadcasting and dtype coercion.

The checker uses singleton rows, divisible shapes, nondivisible tails and two block choices. Compare every value with NumPy and the declared output cast; do not settle for an average. Save an ownership diagram and the last logical row/column checks. Explain why zero padding is valid for this elementwise operation but does not automatically work for a reduction.

## Stage 2: a real target-oriented fused operation

Implement `fused_bias_relu`, with matrix activations and a vector bias that follows the column axis. The matrix and bias need different BlockSpecs. Test cancellations at the ReLU boundary, distinct column biases, logical tails and both precisions.

```sh
# Run run command in terminal using the course Python environment
python3 projects/kernel-audit/tests/check.py --implementation projects/kernel-audit/my_model.py --stage 2
```

Expose explicit `interpret` and `tpu` modes. Target mode must require a real TPU backend/placed inputs and pass `interpret=False`; it must never retry through interpretation. Use the conservative target tile family with dimensions divisible by eight and one hundred twenty-eight. This project does not establish that all other TPU shapes are forbidden.

## Stage 3: pipeline and evidence boundary

Implement `pipelined_axpy` using actual `pltpu.emit_pipeline`. The outer call holds global references; the inner pipeline manages local block buffers and transfers. CPU simulation uses `pltpu.InterpretParams` with race detection and explicit simulated TPU layout metadata. Record the actual backend as CPU.

Compare synchronous copies, two input slots and three input slots while keeping two output slots. Installed JAX 0.9.2 does not support more than two output slots. Preserve representative tails and both dtypes. Reject invalid block/buffer/mode requests.

```sh
# Run run command in terminal using the course Python environment
python3 projects/kernel-audit/tests/check.py --implementation projects/kernel-audit/my_model.py --stage all
```

Implement `target_benchmark` with actual TPU guards, separate compilation, five warmups, repeated synchronized execution and a clear boundary. CPU tests check the refusal path; they do not qualify target compilation or performance. Do not time interpretation and call it a kernel benchmark.

## Run actual target evidence on an available TPU

Use a fresh configured TPU environment with a compatible pinned JAX/libtpu stack. The CPU dependency file is not a TPU installation recipe. Do not use a process forced to `JAX_PLATFORMS=cpu` for the target lab. This project does not provision hardware or spend cloud credits.

```sh
# Run run command in terminal using the course Python environment
python3 projects/kernel-audit/target_tpu.py --implementation projects/kernel-audit/my_model.py --kernel fused --shape 257 513 --block 8 128 --dtype float32
python3 projects/kernel-audit/target_tpu.py --implementation projects/kernel-audit/my_model.py --kernel pipeline --shape 257 513 --block 8 128 --buffers 2
python3 projects/kernel-audit/target_tpu.py --implementation projects/kernel-audit/my_model.py --kernel pipeline --shape 257 513 --block 16 256 --buffers 3 --synchronous
```

The runner checks the real backend before allocation, places inputs on a TPU, checks the complete logical output against an independent represented-input oracle, then benchmarks the candidate and a jitted JAX baseline. It prints JSON only after successful checks, including source hash, configuration, actual device, precision, correctness, compilation time and timing samples. On CPU it exits with status two and produces no target receipt.

The measurement boundary starts with already-placed logical inputs and ends with a synchronized logical output. Candidate padding/cropping is included; compilation and explicit host-to-device placement are excluded. The baseline uses the same precision contract. A hand-written kernel may lose to XLA fusion; retain that result rather than selecting only favorable cases.

## Keep and interpret evidence

Save code, environment, commands, numerical references, ownership maps, frozen data seeds, precision policy, a failure/repair, padding counts, modeled buffer costs and any actual target receipts. Distinguish configuration-derived estimates from device counters and timing observations. Connect the local result to a profile and partitioning decision before claiming an end-to-end improvement.

```sh
# Run run command in terminal using the course Python environment
python3 projects/kernel-audit/tests/check.py --implementation solution --stage all
```

Public CPU stage passes are bounded correctness evidence. They are not TPU validation, a performance win, independent expert approval or a credential. Continue to the [scale synthesis assessment](../../assessments/scale.md), which requires profiling and partitioning reasoning in addition to a local kernel.
