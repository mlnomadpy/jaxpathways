<!-- jaxpathways-content-audit:2026-10-27:precision-ledger -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P1. Finding type: confirmed-correction.

## Teaching problem

`evaluate_precision_policies` produces BF16-operand/FP32-result, BF16-operand/BF16-result and INT8-storage/FP32-dequantization rows. Every row receives `weight_bytes=int(w_int8.size)` and `scale_bytes=int(scale.size*4)`, so the two BF16 cases are reported as INT8 storage with quantization scales.

For the actual 256x64 weight matrix:

| Policy | Weight storage | Scale storage | Additional distinction |
| --- | ---: | ---: | --- |
| BF16 weights | 32,768 B | 0 B for this fixture | Input/result bytes are separate. |
| INT8 weight codes | 16,384 B | 256 B for 64 FP32 scales | FP32 dequantized weights alone occupy 65,536 B. |

The exact CPU function probe reproduces 16,384 + 256 for all three rows. Its numerical errors and returned-dtype checks are useful; the byte ledger is wrong. Existing prose correctly says this is dequantization, not proof of native INT8 execution.

## Acceptance criteria

- [ ] Calculate policy-specific stored codes/weights, metadata, runtime operands and result bytes from actual shapes/dtypes.
- [ ] Show expanded dequantized workspace separately, with lifetime/peak caveats; do not present compressed file size as runtime peak HBM.
- [ ] Add a hand-computed 256x64 ledger and changed-shape/channel checks including the existing zero channel.
- [ ] Update code, tables, output, exercise and figure labels together, retaining requested-result versus internal-accumulation and native-INT8 caveats.

## Audited source

- [tpu-06](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/21-tpu-systems/02-bfloat16-int8-precision-and-xprof-profiling/lesson.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
