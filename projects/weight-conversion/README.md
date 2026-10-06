# Convert PyTorch weights and audit Flax architecture parity

Convert an actual PyTorch state dictionary into a Flax NNX model and explain where its computation agrees or diverges. The implemented architecture is Linear → LayerNorm → exact GELU → Linear, with dimensions chosen to reveal transposes. This is an explicit architecture mapping, not a universal checkpoint converter.

## Prepare your workspace

Use the top folder of the extracted course or project workspace as your working directory. Activate a Python environment and install `requirements-cpu.txt`. The pinned environment includes JAX, Flax, NumPy and, for actual framework comparison, PyTorch. No GPU is required.

```sh
python3 -m pip install -r requirements-cpu.txt
cp projects/weight-conversion/starter/bridge.py projects/weight-conversion/my_bridge.py
```

On Windows PowerShell, use `Copy-Item projects/weight-conversion/starter/bridge.py projects/weight-conversion/my_bridge.py` and the active environment’s Python executable. If an import fails, compare the active interpreter with the one used to install requirements before changing source code.

Read the connected lessons first:

- [deployment-01](../../phases/15-deployment/01-keras-and-pytorch-bridges-to-explicit-jax/docs/en.md)
- [deployment-08](../../phases/15-deployment/08-pytorch-to-flax-numerical-parity/docs/en.md)

## Build your implementation

The starter retains supporting models and input validation. Replace its named `NotImplementedError` functions. Work on your own file; the solution is a reference to inspect after attempting each stage. Every stage command reruns earlier stages, so a later change cannot silently discard a previously verified capability.

### Stage 1: Map and compare real frameworks

Implement `convert` in `my_bridge.py`.

**Demonstrate:** Multiple seeds, asymmetric layers, input gradients and rejected malformed tensors.

```sh
python3 projects/weight-conversion/tests/check.py --implementation projects/weight-conversion/my_bridge.py --stage 1
```

A successful run prints `PASS stage 1` after the actual assertions. On failure, inspect the named shape, reduction, mapping or numerical comparison before changing a tolerance. Keep the command, observed output and your explanation of a changed case.

### Stage 2: Locate numerical disagreement

Implement `error_report` in `my_bridge.py`.

**Demonstrate:** Independent absolute/relative errors, near-zero tolerances and injected epsilon failure.

```sh
python3 projects/weight-conversion/tests/check.py --implementation projects/weight-conversion/my_bridge.py --stage 2
```

A successful run prints `PASS stage 2` after the actual assertions. On failure, inspect the named shape, reduction, mapping or numerical comparison before changing a tolerance. Keep the command, observed output and your explanation of a changed case.

### Stage 3: Reload converted weights independently

Implement `save_flax, load_flax` in `my_bridge.py`.

**Demonstrate:** Fresh-process inference from the converted artifact and incomplete-archive rejection.

```sh
python3 projects/weight-conversion/tests/check.py --implementation projects/weight-conversion/my_bridge.py --stage 3
```

A successful run prints `PASS stage 3` after the actual assertions. On failure, inspect the named shape, reduction, mapping or numerical comparison before changing a tolerance. Keep the command, observed output and your explanation of a changed case.

## Write the mapping before copying tensors

The source hidden weight is output-by-input with shape \((5,3)\); the Flax kernel is input-by-output with shape \((3,5)\). The output layer similarly transposes \((2,5)\) to \((5,2)\). LayerNorm's scale and bias both have shape \((5,)\). Reject missing or extra names, wrong shapes, non-FP32 tensors and nonfinite values. A successful file load is not a successful conversion.

The supplied model classes specify the computation. Both use feature-wise LayerNorm, the same epsilon, exact GELU and the same operation order. Flax uses the explicit variance calculation rather than its fast-variance option for this audit. Dropout is absent. For another architecture, list training flags, buffers, normalization conventions, masks, positional encoding and tied parameters before writing a mapper.

## Interpret errors before changing tolerances

For each element require \(|y_i-\widehat y_i|\le a+r|y_i|\), using the PyTorch output as the reference. The default forward tolerances are absolute \(2\times10^{-6}\) and relative \(2\times10^{-5}\). These are fixture-specific FP32 CPU acceptance budgets, not universal promises. Input-gradient checks use their own documented tolerance.

Report maximum absolute error and relative L2 error, but keep the elementwise pass criterion. Near zero, relative error alone is unstable; an absolute tolerance keeps small rounding differences interpretable. Never increase a tolerance merely because the candidate fails. First inspect the earliest mismatching operation and compare multiple input scales and shapes.

The injected error changes LayerNorm epsilon while preserving every tensor. The first hidden linear output still agrees, then normalization fails. Later activation and output differences can grow or shrink, so the largest error need not identify the original defect. The matching lesson plots these actual layer errors.

## Audit the derivative and artifact boundary

Compare gradients of the same scalar sum of outputs with respect to identical inputs. Forward parity on a few values cannot by itself establish derivative parity. Save all converted arrays together with epsilon and a schema version. The stage-three checker starts a fresh Python process and obtains predictions from the saved Flax artifact, then deliberately removes a field to verify rejection.

The archive format is a small course-specific NPZ contract with pickling disabled. It is not an Orbax checkpoint or a general deployment format. Keep the original PyTorch checkpoint and its provenance; use trusted artifacts, and record source/target framework versions and hashes for real releases. The lesson uses PyTorch's weights-only loading path for its own locally created checkpoint, not an untrusted remote download.

## Extend without overclaiming

For CNNs add independently checked OIHW-to-HWIO permutations; for attention map fused query/key/value ordering and head axes; for language models preserve vocabulary rows, tied embeddings and position conventions. None of these follows from the MLP pass. Add a golden example and intermediate-output checks for each new operation, then revisit low precision separately from FP32 correctness.

Connect a verified mapping to the [deployment audit](../deployment-audit/README.md) for export/precision and the [engineering release](../engineering-release/README.md) for artifact tracking and container checks. This project does not execute a large pretrained model, GPU kernels or an edge device.

## Review your evidence

Keep your source file, environment versions, exact commands, actual results, one failing case with its repair, and the independent reasoning for each stage. The public reference suite checks bounded correctness; it is not a hidden exam or independent review of your capability.

To inspect the provided implementation after your attempt:

```sh
python3 projects/weight-conversion/tests/check.py --implementation solution --stage all
```
