# bfloat16, int8 precision, and XProf profiling on TPU

Phase 21: TPU Systems: Generations, Memory, Precision & XProf · about 25 minutes · CPU

This worked reference is validated on CPU. Accelerator compatibility requires a separate source review and target execution receipt. A separate precision and profiling runner is implemented in the target instructions. Bounded hardware receipts cover v5e1 launch, resume and precision checks; four single-device Pallas cases; and a four-device v5e collective fixture. Each receipt applies only to its recorded source, environment, cases and hardware. Other lessons and TPU configurations require separate qualification.

## What you will be able to do

- Measure error for BF16 operands with requested FP32 and BF16 dot result dtypes, verify the actual returned dtypes, and avoid inferring internal partial-sum precision from output values.
- Implement per-column INT8 weight quantization and an FP32 dequantization reference; account for scales and test a zero-valued channel.
- Capture a warmed `jax.profiler.trace` (`*.xplane.pb`) on a Cloud TPU VM (or local CPU rehearsal) after synchronizing warmup.
- Download the trace directory with `gcloud compute tpus tpu-vm scp --recurse` and inspect it in TensorBoard / XProf.

## The problem

Measure how BF16 operand rounding and the requested dot result dtype affect a fixed CPU matmul, and compare a per-column INT8 weight-quantization reference. Verify the returned dtypes and distinguish observable outputs from backend arithmetic details. Retain a trace of the actual worked workload in a user-selected directory so its events can be inspected.

## The idea

This lesson measures numerical error for BF16 operands with requested FP32 and BF16 dot result dtypes, plus an FP32 dequantization reference for INT8 weights. The JAX API documents `preferred_element_type` as controlling the returned dtype; for the generic dot path it also serves as a compiler hint for accumulation. A returned dtype and output error do not reveal the precision of every internal partial sum. The FP32 result is a finite-precision reference, not exact arithmetic. The INT8 path stores quantized weights plus scale metadata, then casts and rescales them to FP32 before the dot; it does not demonstrate a native INT8 kernel, packed runtime storage, or reduced device-memory traffic. Keep the generated XProf directory after the example returns.

## What does `preferred_element_type` establish for BF16 dot products?

`jnp.dot(..., preferred_element_type=...)` requests the output element type. The lower-level `jax.lax.dot` documentation also describes this as a compiler hint for accumulation when no explicit `DotAlgorithm` is selected. The lesson verifies the actual returned dtype and compares outputs; neither measurement proves that every partial sum was rounded to BF16 or FP32. FP32 itself is finite precision, so the FP32 JAX dot is a reference rather than exact arithmetic. The error ordering below belongs to the recorded input fixture and this CPU backend only. The retained optimized-HLO file for the installed CPU run shows BF16 operands widened to FP32, an FP32 dot, and a conversion of the returned result to BF16. That describes this compiled CPU graph; HLO does not reveal the physical instruction-level reduction order, and the result does not qualify TPU lowering.

The INT8 example quantizes each output column using `scale = max(abs(W[:, j])) / 127`, stores one signed byte per weight plus one FP32 scale per column, and then dequantizes to FP32 before the dot. A zero column uses a safe scale and must produce a zero quantized column without division by zero. This is a numerical reference only: it does not show a native INT8 dot, packed storage, activation quantization, or reduced device memory traffic.

**Run the source-bound CPU dtype and compiler-IR probe**

```bash
JAX_PLATFORMS=cpu python phases/21-tpu-systems/02-bfloat16-int8-precision-and-xprof-profiling/labs/precision_contract_probe.py --output-dir ./tpu06-precision-output
```

**Expected:** Retains the original and BF16 operands, FP32 reference, both requested-result arrays, explicit FP32-result-to-BF16 cast, seed-zero and seed-42 fixtures, plus seed 7 as a changed-seed check. The receipt records environment and source/data hashes and points to a result summary with returned dtypes, computed errors, and equality counts. StableHLO and optimized HLO are compiler evidence; they do not establish physical partial-sum precision or TPU lowering.

$$
\text{Cost per } 10^6 \text{ tokens} = \frac{C_{\text{chip\_hour}} \times N_{\text{chips}} \times (T_{\text{run}} / 3600)}{\text{Tokens}_{\text{processed}} / 10^6}
$$

### Pause and reason

What can the lesson infer from `jnp.dot(x_bf16, w_bf16, preferred_element_type=jnp.bfloat16)` and its measured output error?

<details><summary>Compare your reasoning</summary>

The call requests a BF16 returned dtype, and this run verifies that dtype and compares the output against a finite-precision reference. For the generic dot path, `preferred_element_type` is also a compiler hint about accumulation. These results do not establish that every partial sum is rounded to BF16; that requires backend-specific implementation evidence.

</details>

## 1. Capture a warmed XProf trace on your Cloud TPU VM and inspect in TensorBoard

The worked CPU example warms the exact function it profiles and writes its trace under `TPU_TRACE_DIR` (default `./tpu-xprof-trace`). The directory remains after the function returns. Open that directory in TensorBoard/XProf and inspect the actual event names, timestamp units, host dispatch, compilation boundary, device work, synchronization, and input stalls that are present in your trace. A CPU trace teaches event reading; it does not establish TPU performance. For timing comparisons, measure synchronized steps outside the profiler.

**Run tpu-06.py and capture a 5-step warmed XProf trace on the Cloud TPU VM**

```bash
gcloud compute tpus tpu-vm scp public/exercises/tpu-06.py "$TPU_NAME":~/jax-tpu-lab/ --zone="$ZONE"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command="set -euo pipefail; source ~/jax-tpu-lab/.venv/bin/activate; export TPU_TRACE_DIR=~/jax-tpu-lab/tpu-xprof-trace; JAX_PLATFORMS=tpu python ~/jax-tpu-lab/tpu-06.py"
mkdir -p ./tpu-xprof-trace
gcloud compute tpus tpu-vm scp --recurse "$TPU_NAME":~/jax-tpu-lab/tpu-xprof-trace ./ --zone="$ZONE"
```

**Expected:** Runs the lesson workload, captures that workload in the persistent TPU_TRACE_DIR, then copies the retained trace directory back. Inspect actual events in XProf; CPU and TPU traces remain separately identified.

**Launch TensorBoard with the XProf profile plugin on your workstation**

```bash
# Launch TensorBoard with the XProf profile plugin on your workstation
pip install tensorboard tensorboard-plugin-profile xprof
tensorboard --logdir ./tpu-xprof-trace --port 6006
```

**Expected:** Serves the XProf Trace Viewer, Overview Page, and Memory Viewer at http://localhost:6006.

## Compare BF16 operands with requested FP32 or BF16 dot result dtypes, plus per-column INT8 reference error

Create main.py with evaluate_precision_policies.

```python
# Step 1 — Compare BF16 operands with requested FP32 or BF16 dot result dtypes, plus per-column INT8 reference error
from pathlib import Path
import os
import tempfile
import time
import jax
import jax.numpy as jnp
import numpy as np


def evaluate_precision_policies(seed=0):
    # Create or split explicit PRNG key(s) (`(k1, k2)`) for reproducible randomness.
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    # Sample deterministic random values into `x` using an explicit PRNG key.
    x = jax.random.normal(k1, (64, 256), dtype=jnp.float32)
    # Sample deterministic random values into `w` using an explicit PRNG key.
    w = jax.random.normal(k2, (256, 64), dtype=jnp.float32) * 0.25
    # Cast or evaluate `ref_fp32` in explicit floating-point precision.
    ref_fp32 = jnp.dot(x, w, preferred_element_type=jnp.float32)
    # Cast or evaluate `out_bf16_requested_fp32` in explicit floating-point precision.
    out_bf16_requested_fp32 = jnp.dot(
        x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), preferred_element_type=jnp.float32
    )
    # Cast or evaluate `out_bf16_requested_bf16` in explicit floating-point precision.
    out_bf16_requested_bf16 = jnp.dot(
        x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16
    )
    # Reduce across the target axis to summarize `scale`.
    column_max = jnp.max(jnp.abs(w), axis=0, keepdims=True)
    scale = jnp.where(column_max == 0, 1.0, column_max / 127.0)
    # The scale of 1 for a zero column avoids division by zero; its quantized values remain zero.
    w_int8 = jnp.clip(jnp.round(w / scale), -127, 127).astype(jnp.int8)
    # Cast or evaluate `out_int8_dequant_requested_fp32` in explicit floating-point precision.
    out_int8_dequant_requested_fp32 = jnp.dot(x, w_int8.astype(jnp.float32) * scale, preferred_element_type=jnp.float32)
    result_dtype_cases = [
        ("BF16 operands / requested FP32 result", out_bf16_requested_fp32),
        ("BF16 operands / requested BF16 result", out_bf16_requested_bf16),
        ("INT8 per-column / FP32 dequantized dot result", out_int8_dequant_requested_fp32),
    ]
    results = []
    # Loop over each requested-result-dtype comparison:
    for label, tensor in result_dtype_cases:
        diff = jnp.abs(tensor - ref_fp32)
        # Append the computed value via `results.append({`.
        results.append({
            "result_dtype_setting": label,
            "returned_dtype": str(tensor.dtype),
            "max_abs_err": round(float(jnp.max(diff)), 5),
            "mean_abs_err": round(float(jnp.mean(diff)), 5),
            "weight_bytes": int(w_int8.size),
            "scale_bytes": int(scale.size * 4),
        })
    return results

```

Measuring both max and mean absolute error across requested result dtypes quantifies why requested FP32 result dtype is standard on TPU MXUs.

## Capture a warmed XProf trace and verify *.xplane.pb output

Append capture_warmed_trace and the verification assertions to main.py.

```python
# Step 2 — Capture a warmed XProf trace and verify *.xplane.pb output
def capture_warmed_trace(steps=4, trace_dir=None):
    x = jnp.ones((256, 256), dtype=jnp.bfloat16)
    # Wrap with `jax.jit` (`step_fn`) so XLA traces and compiles the function.
    # Request an FP32 dot result, then explicitly cast the returned values to BF16.
    step_fn = jax.jit(lambda a: jnp.dot(a, a, preferred_element_type=jnp.float32).astype(jnp.bfloat16))
    t0 = time.perf_counter()
    # Wait for device work to finish before timing it or comparing its result on the host.
    jax.block_until_ready(step_fn(x))
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    # Enter managed runtime/context scope for this block:
    trace_dir = Path(trace_dir or os.environ.get("TPU_TRACE_DIR", "./tpu-xprof-trace")).expanduser().resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_dir)):
        t1 = time.perf_counter()
        # Repeat the update loop over `range(steps)` steps:
        for _ in range(steps):
            x = step_fn(x)
        # Wait for device work to finish before timing it or comparing its result on the host.
        jax.block_until_ready(x)
        steady_ms = (time.perf_counter() - t1) * 1000.0 / steps
    # Collect files from the retained trace directory.
    xplane_files = list(trace_dir.rglob("*.xplane.pb"))
    return {"warmup_ms": warmup_ms, "steady_ms": steady_ms, "xplane_count": len(xplane_files), "trace_dir": str(trace_dir)}



prec_table = evaluate_precision_policies()
trace_info = capture_warmed_trace()
# Assert invariant `trace_info["xplane_count"] >= 1` holds
assert trace_info["xplane_count"] >= 1
assert prec_table[0]["returned_dtype"] == "float32"
assert prec_table[1]["returned_dtype"] == "bfloat16"
assert all(np.isfinite(row["max_abs_err"]) and np.isfinite(row["mean_abs_err"]) for row in prec_table)
print("Precision comparison:", prec_table)
print("Captured *.xplane.pb trace count:", trace_info["xplane_count"], "steady_ms:", round(trace_info["steady_ms"], 4))
```

Warming up before entering jax.profiler.trace ensures the captured *.xplane.pb file records steady-state execution rather than first-call compilation.

## Step 3: Verify invariants on the completed state

Run the final shape and numerical assertions to confirm the state built in Steps 1 and 2.

```python
assert trace_info["xplane_count"] >= 1
assert prec_table[0]["returned_dtype"] == "float32"
assert prec_table[1]["returned_dtype"] == "bfloat16"
assert all(np.isfinite(row["max_abs_err"]) and np.isfinite(row["mean_abs_err"]) for row in prec_table)
```

Checking these invariants confirms the computation is ready for the full worked experiment.

## Run the example

```python
# Step 1 — Compare BF16 operands with requested FP32 or BF16 dot result dtypes, plus per-column INT8 reference error
from pathlib import Path
import os
import time
import jax
import jax.numpy as jnp
import numpy as np


def evaluate_precision_policies(seed=0):
    # Create or split explicit PRNG key(s) (`(k1, k2)`) for reproducible randomness.
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    # Sample deterministic random values into `x` using an explicit PRNG key.
    x = jax.random.normal(k1, (64, 256), dtype=jnp.float32)
    # Sample deterministic random values into `w` using an explicit PRNG key.
    w = jax.random.normal(k2, (256, 64), dtype=jnp.float32) * 0.25
    # Keep the reference matrix product in float32 for comparison.
    ref_fp32 = jnp.dot(x, w, preferred_element_type=jnp.float32)
    # Round both operands to BF16, then request a float32 result dtype from the dot.
    out_bf16_requested_fp32 = jnp.dot(
        x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), preferred_element_type=jnp.float32
    )
    # Request a BF16 dot result, then widen returned values to FP32 for error measurement.
    out_bf16_requested_bf16 = jnp.dot(
        x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16
    )
    # Take one maximum per output column so each column gets its own quantization scale.
    column_max = jnp.max(jnp.abs(w), axis=0, keepdims=True)
    scale = jnp.where(column_max == 0, 1.0, column_max / 127.0)
    # The scale of 1 for a zero column avoids division by zero; its quantized values remain zero.
    w_int8 = jnp.clip(jnp.round(w / scale), -127, 127).astype(jnp.int8)
    # Dequantize int8 weights to FP32, then request an FP32 result from the dot.
    out_int8_dequant_requested_fp32 = jnp.dot(x, w_int8.astype(jnp.float32) * scale, preferred_element_type=jnp.float32)
    result_dtype_cases = [
        ("BF16 operands / requested FP32 result", out_bf16_requested_fp32),
        ("BF16 operands / requested BF16 result", out_bf16_requested_bf16),
        ("INT8 per-column / FP32 dequantized dot result", out_int8_dequant_requested_fp32),
    ]
    results = []
    # Loop over each requested-result-dtype comparison:
    for label, tensor in result_dtype_cases:
        diff = jnp.abs(tensor - ref_fp32)
        # Append the computed value via `results.append({`.
        results.append({
            "result_dtype_setting": label,
            "returned_dtype": str(tensor.dtype),
            "max_abs_err": round(float(jnp.max(diff)), 5),
            "mean_abs_err": round(float(jnp.mean(diff)), 5),
            "weight_bytes": int(w_int8.size),
            "scale_bytes": int(scale.size * 4),
        })
    return results



# Step 2 — Capture a warmed XProf trace and verify *.xplane.pb output
def capture_warmed_trace(steps=4, trace_dir=None):
    x = jnp.ones((256, 256), dtype=jnp.bfloat16)
    # Wrap with `jax.jit` (`step_fn`) so XLA traces and compiles the function.
    # Request an FP32 dot result, then explicitly cast the returned values to BF16.
    step_fn = jax.jit(lambda a: jnp.dot(a, a, preferred_element_type=jnp.float32).astype(jnp.bfloat16))
    t0 = time.perf_counter()
    # Wait for device work to finish before timing it or comparing its result on the host.
    jax.block_until_ready(step_fn(x))
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    trace_dir = Path(trace_dir or os.environ.get("TPU_TRACE_DIR", "./tpu-xprof-trace")).expanduser().resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_dir)):
        t1 = time.perf_counter()
        # Repeat the update loop over `range(steps)` steps:
        for _ in range(steps):
            x = step_fn(x)
        # Wait for device work to finish before timing it or comparing its result on the host.
        jax.block_until_ready(x)
        steady_ms = (time.perf_counter() - t1) * 1000.0 / steps
    # Collect files from the retained trace directory.
    xplane_files = list(trace_dir.rglob("*.xplane.pb"))
    return {"warmup_ms": warmup_ms, "steady_ms": steady_ms, "xplane_count": len(xplane_files), "trace_dir": str(trace_dir)}



prec_table = evaluate_precision_policies()
trace_info = capture_warmed_trace()
# Assert invariant `trace_info["xplane_count"] >= 1` holds
assert trace_info["xplane_count"] >= 1
assert prec_table[0]["returned_dtype"] == "float32"
assert prec_table[1]["returned_dtype"] == "bfloat16"
assert all(np.isfinite(row["max_abs_err"]) and np.isfinite(row["mean_abs_err"]) for row in prec_table)
print("Precision comparison:", prec_table)
print("Retained trace directory:", trace_info["trace_dir"])
print("Captured *.xplane.pb trace count:", trace_info["xplane_count"], "steady_ms:", round(trace_info["steady_ms"], 4))
```

Expected: Prints measured max and mean errors against the finite-precision FP32 reference, reports the requested result settings and observed returned dtypes, and confirms that at least one *.xplane.pb trace file was captured. Any error ordering is specific to the fixed input and backend.

## Measured output error for BF16 inputs and an FP32 INT8 dequantization reference

**Predict:** How do BF16 operands with requested FP32 or BF16 result dtypes, and the per-column INT8 dequantization reference, compare in error against a finite-precision FP32 JAX result?

![Measured output error for BF16 inputs and an FP32 INT8 dequantization reference](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

Errors compare returned outputs with the finite-precision FP32 JAX reference for this seeded CPU fixture. Requested/returned result dtypes do not expose the precision of every internal partial sum.

### Connect it to the computation

The measured INT8 error belongs to this fixed example. Its storage estimate includes one FP32 scale per output column; this reference does not demonstrate runtime memory traffic savings.

```python
# Perform matrix contraction / projection to compute `visual_data`.
visual_data = {
    'kind': 'bar',
    'labels': ['BF16 inputs / FP32 result', 'BF16 inputs / BF16 result', 'INT8 weights / FP32 dequantized dot'],
    'xlabel': 'requested result dtype / operand path',
    'ylabel': 'absolute error vs fp32 reference',
    'series': [
        {'label': 'max abs error', 'y': [float(r['max_abs_err']) for r in prec_table]},
        {'label': 'mean abs error', 'y': [float(r['mean_abs_err']) for r in prec_table]},
    ],
}
```

## Recorded reference execution

CPU run: 2026-10-21T17:46:52.522777+00:00. JAX 0.9.2.

```text
Precision comparison: [{'result_dtype_setting': 'BF16 operands / requested FP32 result', 'returned_dtype': 'float32', 'max_abs_err': 0.03718, 'mean_abs_err': 0.00756, 'weight_bytes': 16384, 'scale_bytes': 256}, {'result_dtype_setting': 'BF16 operands / requested BF16 result', 'returned_dtype': 'bfloat16', 'max_abs_err': 0.05545, 'mean_abs_err': 0.00912, 'weight_bytes': 16384, 'scale_bytes': 256}, {'result_dtype_setting': 'INT8 per-column / FP32 dequantized dot result', 'returned_dtype': 'float32', 'max_abs_err': 0.12516, 'mean_abs_err': 0.02322, 'weight_bytes': 16384, 'scale_bytes': 256}]
Captured *.xplane.pb trace count: 5 steady_ms: 0.1102
Precision comparison: [{'result_dtype_setting': 'BF16 operands / requested FP32 result', 'returned_dtype': 'float32', 'max_abs_err': 0.03718, 'mean_abs_err': 0.00756, 'weight_bytes': 16384, 'scale_bytes': 256}, {'result_dtype_setting': 'BF16 operands / requested BF16 result', 'returned_dtype': 'bfloat16', 'max_abs_err': 0.05545, 'mean_abs_err': 0.00912, 'weight_bytes': 16384, 'scale_bytes': 256}, {'result_dtype_setting': 'INT8 per-column / FP32 dequantized dot result', 'returned_dtype': 'float32', 'max_abs_err': 0.12516, 'mean_abs_err': 0.02322, 'weight_bytes': 16384, 'scale_bytes': 256}]
Retained trace directory: /private/tmp/jaxpathways-reader-review-fix/tpu-xprof-trace
Captured *.xplane.pb trace count: 5 steady_ms: 0.1348
Requested BF16 / FP32 result max-error ratio for seed 0: 1.49
Warmup ms: 25.381 Traced steady step ms: 0.1348
Seed 42 returned dtypes and mean errors: {'BF16 operands / requested FP32 result': {'returned_dtype': 'float32', 'mean_abs_err': 0.0075}, 'BF16 operands / requested BF16 result': {'returned_dtype': 'bfloat16', 'mean_abs_err': 0.00902}, 'INT8 per-column / FP32 dequantized dot result': {'returned_dtype': 'float32', 'mean_abs_err': 0.02182}} xplane files: 5
Mean error (per-channel vs per-tensor): 0.01389 0.0979
xplane.pb bytes: 260182
PASS: tpu-06

```

## Compare requested FP32 vs BF16 result dtype error ratio

**Predict before running:** For the seed-zero CPU fixture, what is the ratio of the maximum error for the requested BF16 result to the requested FP32 result?

```python
# Experiment — Compare requested FP32 vs BF16 result dtype error ratio
requested_fp32_err = prec_table[0]["max_abs_err"]
requested_bf16_err = prec_table[1]["max_abs_err"]
ratio = round(requested_bf16_err / requested_fp32_err, 2)
print("Requested BF16 / FP32 result max-error ratio for seed 0:", ratio)
# This reports the measured ratio for the current backend and fixed seed-zero fixture.
assert np.isfinite(ratio)
```

**Expected:** For this seed-zero CPU fixture, the requested BF16 result has about 1.49 times the maximum error of the requested FP32 result. This is an output comparison, not evidence about every internal partial sum.

The ratio is computed from this fixed seed-zero fixture and the current CPU backend. `preferred_element_type` requests the returned dtype and acts as an accumulation hint for this generic dot path; it does not independently certify backend internals.

## Compare warmup duration vs traced steady-state step duration

**Predict before running:** Does the warmup step outside the profiler take longer than the traced steady-state steps?

```python
# Experiment — Compare warmup duration vs traced steady-state step duration
print("Warmup ms:", round(trace_info["warmup_ms"], 3), "Traced steady step ms:", round(trace_info["steady_ms"], 4))
# Assert invariant `trace_info["warmup_ms"] > trace_info["steady_ms"]` holds
assert trace_info["warmup_ms"] > trace_info["steady_ms"]
```

**Expected:** Warmup takes significantly longer than a single steady-state step inside the profiler block.

Keeping warmup outside `jax.profiler.trace` prevents XLA compilation from dominating the captured XProf timeline.

## Make it yours

Run `evaluate_precision_policies(seed=42)` and `capture_warmed_trace(steps=3)`. Report the observed backend, each returned dtype and the two mean errors; do not infer the physical accumulation path from this comparison. Confirm that `xplane_count >= 1`.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Inputs:**
- The seeded precision-policy evaluator and warmed trace capture helper.

**Outputs:**
- For seed 42, measured max and mean errors, requested-result labels, actual returned dtypes, and the trace receipt.

**Invariants to check:**
- Report the actual backend and returned dtypes; do not treat a requested result dtype or measured error as proof of internal partial-sum precision.
- Keep the error comparison tied to the seed-42 fixture and confirm the trace receipt contains an XPlane file.

**Hints:**
- Compare rows by the literal requested-result-dtype labels, then inspect each output array dtype.
- A CPU trace is a valid local exercise result; reserve TPU implementation and performance claims for matching target evidence.

**Starter code scaffold (fill in the TODOs):**

```python
import jax

def compare_precision_run(evaluate_precision_policies, capture_warmed_trace):
    # TODO: Run the seeded comparison and trace, then validate the requested results and backend.
    raise NotImplementedError
```

<details><summary>Reference solution</summary>

```python
# Exercise solution
p42 = evaluate_precision_policies(seed=42)
t3 = capture_warmed_trace(steps=3)
print("Seed 42 returned dtypes and mean errors:", {r["result_dtype_setting"]: {"returned_dtype": r["returned_dtype"], "mean_abs_err": r["mean_abs_err"]} for r in p42}, "xplane files:", t3["xplane_count"])
assert p42[0]["returned_dtype"] == "float32" and p42[1]["returned_dtype"] == "bfloat16"
assert t3["xplane_count"] >= 1
```

</details>

## Compare per-channel vs per-tensor INT8 quantization error

**Foundations**

Quantize `w` with a single global scalar `scale_global = jnp.max(jnp.abs(w)) / 127.0` and compare its max error against per-channel quantization when one column of `w` is scaled by `10.0`.

<details><summary>Hint</summary>

Multiply `w[:, :1]` by `10.0` and compare per-column `axis=0` scale vs global scalar scale.

</details>

### How to write: Compare per-channel vs per-tensor INT8 quantization error — Step-by-step recipe & starter scaffold

**Inputs:**
- A weight matrix with one column scaled by ten and per-channel and global INT8 quantizers.

**Outputs:**
- Maximum reconstruction errors for both quantization scales.

**Invariants to check:**
- The global scale covers the largest magnitude; per-channel scales are computed per output column (axis 0 for this weight layout).

**Hints:**
- Construct the skewed matrix deterministically by multiplying one column by ten.
- Compare dequantized weights with the same original matrix using global and per-column scales.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp
import numpy as np

def compare_int8_scale_granularity(weights):
    # TODO: skew one column and compare max error from global and per-channel scales.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Compare per-channel vs per-tensor INT8 quantization error (Foundations)
# Create or split explicit PRNG key(s) (`w_skew`) for reproducible randomness.
w_skew = jax.random.normal(jax.random.PRNGKey(9), (128, 32), dtype=jnp.float32) * 0.2
w_skew = w_skew.at[:, 0].multiply(10.0)
# Create or split explicit PRNG key(s) (`x_in`) for reproducible randomness.
x_in = jax.random.normal(jax.random.PRNGKey(10), (32, 128), dtype=jnp.float32)
# Cast or evaluate `ref` in explicit floating-point precision.
ref = jnp.dot(x_in, w_skew, preferred_element_type=jnp.float32)
# Reduce along axis=0 to compute `s_col`.
s_col = jnp.max(jnp.abs(w_skew), axis=0, keepdims=True) / 127.0
s_glb = jnp.max(jnp.abs(w_skew)) / 127.0
q_col = jnp.clip(jnp.round(w_skew / s_col), -127, 127).astype(jnp.int8)
q_glb = jnp.clip(jnp.round(w_skew / s_glb), -127, 127).astype(jnp.int8)
err_col = float(jnp.mean(jnp.abs(jnp.dot(x_in, q_col.astype(jnp.float32) * s_col) - ref)))
err_glb = float(jnp.mean(jnp.abs(jnp.dot(x_in, q_glb.astype(jnp.float32) * s_glb) - ref)))
print("Mean error (per-channel vs per-tensor):", round(err_col, 5), round(err_glb, 5))
# Assert invariant `err_col < err_glb` holds
assert err_col < err_glb
```

One outlier channel stretches a per-tensor scale factor and crushes small channels into a few integer buckets; per-channel scaling isolates each column.

</details>

## Verify non-empty XProf *.xplane.pb byte size

**Transfer / diagnosis**

Capture a trace in a temporary directory and verify that the generated `*.xplane.pb` file has `stat().st_size > 0`.

<details><summary>Hint</summary>

Check `xplane_files[0].stat().st_size > 0`.

</details>

### How to write: Verify non-empty XProf *.xplane.pb byte size — Step-by-step recipe & starter scaffold

**Inputs:**
- A temporary trace directory and the public capture_warmed_trace function.

**Outputs:**
- The captured XPlane file path and its byte size.

**Invariants to check:**
- The reported trace must be an actual file with positive size; a directory count alone is insufficient.

**Hints:**
- Capture a short warmed run.
- Inspect each generated xplane.pb file with stat.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp
import numpy as np

def inspect_xplane_file(capture_warmed_trace, root):
    # TODO: capture a trace and return the nonempty XPlane file evidence.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Verify non-empty XProf *.xplane.pb byte size (Transfer / diagnosis)
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory(prefix="tpu-xplane-size-") as d:
    a = jnp.ones((128, 128), dtype=jnp.bfloat16)
    # Wrap with `jax.jit` (`fn`) so XLA traces and compiles the function.
    fn = jax.jit(lambda z: z @ z)
    # Wait for device work to finish before timing it or comparing its result on the host.
    jax.block_until_ready(fn(a))
    # Enter managed runtime/context scope for this block:
    with jax.profiler.trace(d):
        # Wait for device work to finish before timing it or comparing its result on the host.
        jax.block_until_ready(fn(a))
    pb = list(Path(d).rglob("*.xplane.pb"))[0]
    size_bytes = pb.stat().st_size
print("xplane.pb bytes:", size_bytes)
# Assert invariant `size_bytes > 0` holds
assert size_bytes > 0
```

Verifying a non-empty `*.xplane.pb` file confirms the profiler flushed device and host events before closing.

</details>

## Check your understanding

Why should you run and synchronize at least one warmup step before entering `with jax.profiler.trace(trace_dir):` on a TPU VM?

1. Because `jax.profiler.trace` deletes all model weights if warmup is skipped
2. So the captured `*.xplane.pb` trace measures steady-state TPU execution and collective communication rather than one-time Python tracing and XLA HLO compilation
3. Because `bfloat16` only works inside a profiler context

<details><summary>Answer and explanation</summary>

**1. Because `jax.profiler.trace` deletes all model weights if warmup is skipped**

Tracing observes execution and does not delete model state. Warmup prepares the tested executable before the recorded window.

**2. So the captured `*.xplane.pb` trace measures steady-state TPU execution and collective communication rather than one-time Python tracing and XLA HLO compilation (correct)**

Warming and synchronizing the same signature reduces initial compilation in the recorded window. Inspect the trace to confirm: new shapes or static arguments can still compile, and cache state can still affect timing. Record target latency outside profiler overhead too.

**3. Because `bfloat16` only works inside a profiler context**

Profiling is not restricted to BF16. Record the precision used and the actual operations in the captured target workload.

</details>

## Diagnose the result

If XProf shows large gaps between device steps on your TPU VM, check for un-jitted Python work, host-side NumPy conversions, or synchronous `print(array)` calls inside the training step.

## Carry forward

- Use BF16 operands when appropriate, request the desired result dtype with `preferred_element_type`, and verify the actual returned dtype. Treat internal accumulation behavior as backend-specific unless a supported explicit dot algorithm or target evidence establishes it. Use per-column scales when quantizing weights to `int8`.
- Capture warmed `jax.profiler.trace` (`*.xplane.pb`) artifacts on the TPU VM and copy them back with `gcloud compute tpus tpu-vm scp --recurse` before deleting the VM.

## Keep your evidence

Keep the returned-dtype/error table, seeded cast comparisons, source-bound CPU compiler IR and captured *.xplane.pb trace receipt. State the observed backend and the limits of each artifact.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [Profile JAX programs with XProf and TensorBoard](https://docs.jax.dev/en/latest/profiling.html)
- [Cloud TPU performance guide](https://cloud.google.com/tpu/docs/performance-guide)
- [jax.numpy.dot: preferred_element_type and returned dtype](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.dot.html)
- [jax.lax.dot: output dtype and compiler accumulation hint](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot.html)
- [jax.lax.DotAlgorithm: explicit operand and accumulation types](https://docs.jax.dev/en/latest/jax.lax.html#jax.lax.DotAlgorithm)

