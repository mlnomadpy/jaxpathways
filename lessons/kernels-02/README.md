# A first TPU kernel

Phase 13: Pallas kernels · about 110 minutes · CPU

## What you will be able to do

- Implement a real TPU-targeted Pallas launch with a broadcast BlockSpec.
- Specify target layout and float32 compute/output-cast policies.
- Verify CPU semantics across tails and precision choices.
- Refuse target execution claims when only interpretation is available.

## The problem

An interpreted kernel can be numerically correct and still violate a TPU layout or compilation restriction. How do we build a genuine TPU launch path without pretending that a CPU test ran on an accelerator? We will implement a fused bias-and-ReLU kernel with separate, enforceable execution modes.

## The idea

Once a tile's inputs are identified, a kernel performs local arithmetic and writes the corresponding outputs. Bias broadcasting, activation and dtype rules must preserve the declared shape and value contract.

## Separate tile ownership from the local arithmetic

Take one illustrative row $[-2,1]$ and feature bias $[1,-3]$. Addition gives $[-1,-2]$, and ReLU gives $[0,0]$. A bias indexed by row instead of feature can preserve output shape while computing the wrong result.

Trace loads into the tile, the exact broadcast axis, activation and stores. Keep boundary guards visible beside the arithmetic rather than assuming a correct formula guarantees correct indexing.

The activation plot checks the resulting values. Compare against an independent array implementation on awkward dimensions and negative/positive inputs. CPU interpretation helps test semantics; it does not qualify TPU throughput or prove a particular memory schedule.

### Pause and reason

What can a value reference detect that an output-shape assertion cannot?

<details><summary>Compare your reasoning</summary>

Wrong broadcast axes, incorrect activation order or indexing errors that retain the expected shape. Use distinguishable values so those errors cannot cancel.

</details>

## Map each operand according to its meaning

The activation matrix has shape $(M,N)$; the bias has shape $(N,)$. We reshape the padded bias to $(1,N_p)$ and map its block coordinates to $(0,j)$, so changing the row-program index does not change the bias window. The output uses $(i,j)$ and is written exactly once per tile. This is intentional broadcasting with an explicit contract, unlike accepting accidentally mismatched matrices.

## Use a conservative TPU tile contract

TPU vector-memory layout places restrictions on block dimensions. This wrapper requires block dimensions that are multiples of $(8,128)$ for target mode, then pads and crops arbitrary positive logical shapes. The one-row bias window spans its whole first dimension. This is a deliberately narrow supported contract, not a claim that every other block shape is forbidden on every TPU generation. CPU interpretation alone does not enforce or validate all lowering restrictions.

## State where precision changes

Both inputs may be float32 or bfloat16, but their dtypes must match. The kernel promotes loaded values to float32 for addition and ReLU, then casts once to the output dtype. An independent reference must start from those same represented inputs and apply the same final cast. Comparing bfloat16 output directly to an unrounded original dataset mixes representation error with kernel correctness.

## Make the execution mode inspectable

The wrapper accepts only the strings interpret and tpu. The TPU branch checks the actual default backend and the devices holding the inputs, and passes interpret=False. It never catches a target lowering failure and retries on CPU. A target failure should remain a failed result with its environment, not become a fabricated successful accelerator receipt.

## Run the real target path as a separate lab

On an already configured TPU environment, use `python3 projects/kernel-audit/target_tpu.py --kernel fused --block 8 128`. That runner explicitly places inputs on a TPU, calls the target branch, synchronizes, and prints a JSON evidence record after numerical comparisons. Run it in a fresh TPU process rather than a CPU-forced lesson runner. The command is implemented; no target receipt exists in this authoring environment. CPU interpretation establishes neither target lowering success nor speedup.

## Compare with a strong baseline before optimizing

The ordinary JAX expression can already be fused by XLA. A hand-written fused kernel is not automatically faster. On the target, compare with a warmed jitted baseline using the same dtype, data, padding boundary and output semantics. The final lesson defines the measurement protocol and distinguishes kernel-only timing from end-to-end wrapper cost.

## Define separate interpreted and target-only launch modes

Create main.py for the first block, then append each block in order in your course CPU environment.

```python
# Step 1 — Define separate interpreted and target-only launch modes: The launch path states its execution mode explicitly.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Function `fused_bias_relu(x, bias, block, mode)` implementing this stage's computation:
def fused_bias_relu(x,bias,block=(8,128),mode="interpret"):
    # Guard input contract (`x.ndim != 2 or bias.ndim != 1 or bias.shape[0] != x.shape[1] or (min(x.shape) < 1)`) and fail fast if violated.
    if x.ndim!=2 or bias.ndim!=1 or bias.shape[0]!=x.shape[1] or min(x.shape)<1:
        raise ValueError("x must be (M,N), bias must be (N,), and dimensions nonempty")
    # Guard input contract (`x.dtype != bias.dtype or x.dtype not in (jnp.float32, jnp.bfloat16)`) and fail fast if violated.
    if x.dtype!=bias.dtype or x.dtype not in (jnp.float32,jnp.bfloat16):
        raise ValueError("matching float32 or bfloat16 required")
    # Evaluate `(bm, bn)` from the current inputs and state.
    bm,bn=block
    # Guard input contract (`not isinstance(bm, int) or not isinstance(bn, int) or bm < 1 or (bn < 1)`) and fail fast if violated.
    if not isinstance(bm,int) or not isinstance(bn,int) or bm<1 or bn<1:
        raise ValueError("positive integer block sizes required")
    # Guard input contract (`mode not in ('interpret', 'tpu')`) and fail fast if violated.
    if mode not in ("interpret","tpu"):
        raise ValueError("mode must explicitly be interpret or tpu")
    # Branch on condition `mode == 'tpu'`:
    if mode=="tpu":
        if jax.default_backend()!="tpu" or any(not isinstance(a,jax.core.Tracer) and any(d.platform!="tpu" for d in a.devices()) for a in (x,bias)):
            raise RuntimeError("Real TPU inputs/backend required; CPU fallback is disabled")
        if bm%8 or bn%128:
            raise ValueError("this TPU wrapper requires block multiples of (8,128)")
    # Evaluate `(m, n)` from the current inputs and state.
    # Evaluate `pm` from the current inputs and state.
    # Evaluate `pn` from the current inputs and state.
    m,n=x.shape;pm=(m+bm-1)//bm*bm;pn=(n+bn-1)//bn*bn
    # Combine or mask array elements to form `px`.
    px=jnp.pad(x,((0,pm-m),(0,pn-n)))
    # Combine or mask array elements to form `pb`.
    pb=jnp.pad(bias,(0,pn-n))[None,:]
    # Function `kernel(x_ref, bias_ref, out_ref)` implementing this stage's computation:
    def kernel(x_ref,bias_ref,out_ref):
        # Cast or evaluate `value` in explicit floating-point precision.
        value=x_ref[...].astype(jnp.float32)+bias_ref[...].astype(jnp.float32)
        # Reduce across the target axis to summarize `out_ref[...]`.
        out_ref[...]=jnp.maximum(value,0).astype(out_ref.dtype)
    # Invoke custom Pallas kernel or tile specification (`matrix_spec`).
    matrix_spec=pl.BlockSpec((bm,bn),lambda i,j:(i,j))
    # Invoke custom Pallas kernel or tile specification (`bias_spec`).
    bias_spec=pl.BlockSpec((1,bn),lambda i,j:(0,j))
    # Invoke custom Pallas kernel or tile specification (`output`).
    output=pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((pm,pn),x.dtype),
        grid=(pm//bm,pn//bn),in_specs=(matrix_spec,bias_spec),out_specs=matrix_spec,
        interpret=(mode=="interpret"),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel","parallel")))(px,pb)
    # Return `output[:m, :n]` to the caller.
    return output[:m,:n]
```

The launch path states its execution mode explicitly. TPU mode refuses non-TPU inputs and enforces a conservative target tile contract; interpretation remains a named testing mode.

## Execute the kernel semantics on CPU

Create main.py for the first block, then append each block in order in your course CPU environment.

```python
# Step 2 — Execute the kernel semantics on CPU: The matrix and column bias use different BlockSpecs.
# Construct and reshape `x` into the target tensor dimensions.
x=jnp.linspace(-2,2,9*129,dtype=jnp.float32).reshape(9,129)
# Initialize array `bias` with explicit values and shape.
bias=jnp.linspace(-.3,.4,129,dtype=jnp.float32)
# Run `fused_bias_relu` to compute `actual`.
actual=fused_bias_relu(x,bias)
# Convert `reference` to a host NumPy array for inspection or verification.
reference=np.maximum(np.asarray(x)+np.asarray(bias)[None,:],0)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-6)
# Verify that the output tensor shape matches our prediction.
assert actual.shape==(9,129)
# Print the observed values to compare against the expected result.
print("Interpretation max error:",float(np.max(np.abs(np.asarray(actual)-reference))))
# Print diagnostic summary of the computed outputs.
print("Zero fraction and endpoint:",float(np.mean(reference==0)),float(actual[-1,-1]))
```

The matrix and column bias use different BlockSpecs. The bias window is reused across row blocks, while each output tile has unique ownership.

## Prove the target path cannot silently claim CPU success

Create main.py for the first block, then append each block in order in your course CPU environment.

```python
# Step 3 — Prove the target path cannot silently claim CPU success: The registered lesson executes CPU interpretation only.
if jax.default_backend()=="cpu":
    rejected=False
    try:fused_bias_relu(x,bias,mode="tpu")
    except RuntimeError as error:
        rejected=True
        print("Expected target refusal:",error)
    assert rejected
else:
    print("Reference lesson is intended for CPU; target execution uses the separate project runner")
```

The registered lesson executes CPU interpretation only. On real TPU hardware the project runner explicitly chooses mode=tpu, synchronizes, compares against a reference, and records the actual device.

## Run the example

```python
# Step 1 — Define separate interpreted and target-only launch modes: The launch path states its execution mode explicitly.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Function `fused_bias_relu(x, bias, block, mode)` implementing this stage's computation:
def fused_bias_relu(x,bias,block=(8,128),mode="interpret"):
    # Guard input contract (`x.ndim != 2 or bias.ndim != 1 or bias.shape[0] != x.shape[1] or (min(x.shape) < 1)`) and fail fast if violated.
    if x.ndim!=2 or bias.ndim!=1 or bias.shape[0]!=x.shape[1] or min(x.shape)<1:
        raise ValueError("x must be (M,N), bias must be (N,), and dimensions nonempty")
    # Guard input contract (`x.dtype != bias.dtype or x.dtype not in (jnp.float32, jnp.bfloat16)`) and fail fast if violated.
    if x.dtype!=bias.dtype or x.dtype not in (jnp.float32,jnp.bfloat16):
        raise ValueError("matching float32 or bfloat16 required")
    # Evaluate `(bm, bn)` from the current inputs and state.
    bm,bn=block
    # Guard input contract (`not isinstance(bm, int) or not isinstance(bn, int) or bm < 1 or (bn < 1)`) and fail fast if violated.
    if not isinstance(bm,int) or not isinstance(bn,int) or bm<1 or bn<1:
        raise ValueError("positive integer block sizes required")
    # Guard input contract (`mode not in ('interpret', 'tpu')`) and fail fast if violated.
    if mode not in ("interpret","tpu"):
        raise ValueError("mode must explicitly be interpret or tpu")
    # Branch on condition `mode == 'tpu'`:
    if mode=="tpu":
        if jax.default_backend()!="tpu" or any(not isinstance(a,jax.core.Tracer) and any(d.platform!="tpu" for d in a.devices()) for a in (x,bias)):
            raise RuntimeError("Real TPU inputs/backend required; CPU fallback is disabled")
        if bm%8 or bn%128:
            raise ValueError("this TPU wrapper requires block multiples of (8,128)")
    # Evaluate `(m, n)` from the current inputs and state.
    # Evaluate `pm` from the current inputs and state.
    # Evaluate `pn` from the current inputs and state.
    m,n=x.shape;pm=(m+bm-1)//bm*bm;pn=(n+bn-1)//bn*bn
    # Combine or mask array elements to form `px`.
    px=jnp.pad(x,((0,pm-m),(0,pn-n)))
    # Combine or mask array elements to form `pb`.
    pb=jnp.pad(bias,(0,pn-n))[None,:]
    # Function `kernel(x_ref, bias_ref, out_ref)` implementing this stage's computation:
    def kernel(x_ref,bias_ref,out_ref):
        # Cast or evaluate `value` in explicit floating-point precision.
        value=x_ref[...].astype(jnp.float32)+bias_ref[...].astype(jnp.float32)
        # Reduce across the target axis to summarize `out_ref[...]`.
        out_ref[...]=jnp.maximum(value,0).astype(out_ref.dtype)
    # Invoke custom Pallas kernel or tile specification (`matrix_spec`).
    matrix_spec=pl.BlockSpec((bm,bn),lambda i,j:(i,j))
    # Invoke custom Pallas kernel or tile specification (`bias_spec`).
    bias_spec=pl.BlockSpec((1,bn),lambda i,j:(0,j))
    # Invoke custom Pallas kernel or tile specification (`output`).
    output=pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((pm,pn),x.dtype),
        grid=(pm//bm,pn//bn),in_specs=(matrix_spec,bias_spec),out_specs=matrix_spec,
        interpret=(mode=="interpret"),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel","parallel")))(px,pb)
    # Return `output[:m, :n]` to the caller.
    return output[:m,:n]

# Step 2 — Execute the kernel semantics on CPU: The matrix and column bias use different BlockSpecs.
# Construct and reshape `x` into the target tensor dimensions.
x=jnp.linspace(-2,2,9*129,dtype=jnp.float32).reshape(9,129)
# Initialize array `bias` with explicit values and shape.
bias=jnp.linspace(-.3,.4,129,dtype=jnp.float32)
# Run `fused_bias_relu` to compute `actual`.
actual=fused_bias_relu(x,bias)
# Convert `reference` to a host NumPy array for inspection or verification.
reference=np.maximum(np.asarray(x)+np.asarray(bias)[None,:],0)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-6)
# Verify that the output tensor shape matches our prediction.
assert actual.shape==(9,129)
# Print the observed values to compare against the expected result.
print("Interpretation max error:",float(np.max(np.abs(np.asarray(actual)-reference))))
# Print diagnostic summary of the computed outputs.
print("Zero fraction and endpoint:",float(np.mean(reference==0)),float(actual[-1,-1]))

# Step 3 — Prove the target path cannot silently claim CPU success: The registered lesson executes CPU interpretation only.
if jax.default_backend()=="cpu":
    rejected=False
    try:fused_bias_relu(x,bias,mode="tpu")
    except RuntimeError as error:
        rejected=True
        print("Expected target refusal:",error)
    assert rejected
else:
    print("Reference lesson is intended for CPU; target execution uses the separate project runner")
```

Expected: CPU interpretation agrees with NumPy, including the (9,129) tail. The final value is approximately 2.4. Requesting TPU mode on CPU raises an explicit refusal.

## Bias and ReLU alter the values, not the tile ownership contract

**Predict:** Where will the biased first and last rows cross the zero threshold?

![Bias and ReLU alter the values, not the tile ownership contract](../../phases/13-kernels/02-a-first-tpu-kernel/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis is column index. The vertical axis is output activation after adding that column’s bias and applying ReLU. The two plotted series are the first and last rows from the actual interpreted $(9,129)$ kernel result.

The first row is entirely zero for this fixture because its negative activations remain below zero after bias. The last row stays positive and reaches approximately $2.4$ at column $128$, including the final partial logical tile. These are output values, not timings.

### Connect it to the computation

The per-column NumPy comparison verifies every element, while the displayed rows make the activation boundary and tail visible. The bias window depends on the column program index and is reused across row blocks.

This figure is labeled CPU interpretation. It does not prove TPU lowering, vector-memory scheduling or speed. The separate target-only runner refuses to create those claims without real TPU inputs and completed execution.

```python
# Compute figure data for: Bias and ReLU alter the values, not the tile ownership contract
# Convert `visual_data` to a host NumPy array for inspection or verification.
visual_data={'kind':'line','x':list(range(129)),'xlabel':'logical column','ylabel':'output activation','series':[{'label':'first row, CPU interpretation','y':np.asarray(actual)[0].tolist()},{'label':'last row, CPU interpretation','y':np.asarray(actual)[-1].tolist()}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:05:21.947777+00:00. JAX 0.9.2.

```text
Interpretation max error: 0.0
Zero fraction and endpoint: 0.48578811369509045 2.4000000953674316
Expected target refusal: Real TPU inputs/backend required; CPU fallback is disabled
Interpretation max error: 0.0
Zero fraction and endpoint: 0.48578811369509045 2.4000000953674316
Expected target refusal: Real TPU inputs/backend required; CPU fallback is disabled
bfloat16 represented-input reference matched exactly
Boundary fixture: [[0. 0. 0.]
 [2. 0. 0.]]
Changed tails and two TPU-compatible blocks verified in interpretation
Malformed bias axes rejected
Column/bias pairing survives a joint permutation
PASS: kernels-02

```

## Check the represented bfloat16 contract

**Predict before running:** Will bfloat16 inputs need comparison against their original float32 values or their rounded representations?

```python
# Experiment — Check the represented bfloat16 contract: This checks the declared compute/cast policy.
bx=x.astype(jnp.bfloat16);bb=bias.astype(jnp.bfloat16)
# Run `fused_bias_relu` to compute `bactual`.
bactual=fused_bias_relu(bx,bb)
# Cast or evaluate `bref` in explicit floating-point precision.
bref=jnp.maximum(bx.astype(jnp.float32)+bb.astype(jnp.float32)[None,:],0).astype(jnp.bfloat16)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(bactual),np.asarray(bref))
# Verify contract: `bactual.dtype == jnp.bfloat16`.
assert bactual.dtype==jnp.bfloat16
# Print the observed values to compare against the expected result.
print("bfloat16 represented-input reference matched exactly")
```

**Expected:** The represented-input reference matches exactly for this fixture.

This checks the declared compute/cast policy. Any difference from the original float32 experiment is a separate precision-policy observation.

## Place the ReLU boundary deliberately

**Predict before running:** What should happen when a value exactly cancels its column bias?

```python
# Experiment — Place the ReLU boundary deliberately: Testing activation boundaries catches wrong bias axes and...
# Initialize array `edge_x` with explicit values and shape.
edge_x=jnp.array([[-1.,0.,1.],[1.,-2.,.5]],dtype=jnp.float32)
# Initialize array `edge_bias` with explicit values and shape.
edge_bias=jnp.array([1.,0.,-1.],dtype=jnp.float32)
# Run `fused_bias_relu` to compute `edge`.
edge=fused_bias_relu(edge_x,edge_bias)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(edge,np.array([[0.,0.,0.],[2.,0.,0.]],dtype=np.float32))
# Print the observed values to compare against the expected result.
print("Boundary fixture:",np.asarray(edge))
```

**Expected:** Exact cancellations become zero; the only positive result is 2.

Testing activation boundaries catches wrong bias axes and accidental operations applied before or after casting.

## Make it yours

Change the activation shape to (17,257), freeze a NumPy random seed and compare two target-compatible block choices in interpretation. Keep the bias values distinct by column.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Create device-backed JAX array `changed_x`.
2. Create device-backed JAX array `changed_bias`.
3. Convert `expected` to a host NumPy array for inspection or verification.
4. Iterate over `block` to step through the computation:
5. Run `fused_bias_relu` to compute `result`.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Change the activation shape to (17,257), freeze a NumPy random seed...
rng = np.random.default_rng(...)  # TODO: compute rng
# Create device-backed JAX array `changed_x`.
changed_x = jnp.asarray(...)  # TODO: compute changed_x
# Create device-backed JAX array `changed_bias`.
changed_bias = jnp.asarray(...)  # TODO: compute changed_bias
# Convert `expected` to a host NumPy array for inspection or verification.
expected = np.maximum(...)  # TODO: compute expected
# Iterate over `block` to step through the computation:
for block in [(8,128),(16,256)]:
    # Run `fused_bias_relu` to compute `result`.
    result = fused_bias_relu(...)  # TODO: compute result
    # Verify that computed values match the expected reference within numerical tolerance.
    np.testing.assert_allclose(result,expected,rtol = ...  # TODO: compute np.testing.assert_allclose(result,expected,rtol
# Print the observed values to compare against the expected result.
print("Changed tails and two TPU-compatible blocks verified in interpretation")
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Change the activation shape to (17,257), freeze a NumPy random seed...
rng=np.random.default_rng(5)
# Create device-backed JAX array `changed_x`.
changed_x=jnp.asarray(rng.normal(size=(17,257)),dtype=jnp.float32)
# Create device-backed JAX array `changed_bias`.
changed_bias=jnp.asarray(rng.normal(size=257),dtype=jnp.float32)
# Convert `expected` to a host NumPy array for inspection or verification.
expected=np.maximum(np.asarray(changed_x)+np.asarray(changed_bias)[None,:],0)
# Iterate over `block` to step through the computation:
for block in [(8,128),(16,256)]:
    # Run `fused_bias_relu` to compute `result`.
    result=fused_bias_relu(changed_x,changed_bias,block)
    # Verify that computed values match the expected reference within numerical tolerance.
    np.testing.assert_allclose(result,expected,rtol=1e-6,atol=1e-6)
# Print the observed values to compare against the expected result.
print("Changed tails and two TPU-compatible blocks verified in interpretation")
```

</details>

## Reject hidden bias-axis broadcasting

**Practice**

Give the wrapper a matrix-shaped bias and then a wrong-length vector. Verify both fail before launch.

<details><summary>Hint</summary>

A column bias is a vector with exactly one entry per matrix column.

</details>

### How to write: Reject hidden bias-axis broadcasting — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.zeros / jnp.ones(shape, dtype=...)` — Allocates a tensor of the given `shape` initialized with constants.

**Step-by-step implementation plan:**
1. Iterate over `malformed` to step through the computation:
2. Evaluate `rejected` from the current inputs and state.
3. Run the boundary check and catch the expected exception:
4. Verify contract: `rejected`.
5. Print the observed values to compare against the expected result.

**Starter code scaffold (fill in the TODOs):**

```python
# Reject hidden bias-axis broadcasting (Practice): Explicit validation turns a confusing broadcast result into...
# Iterate over `malformed` to step through the computation:
for malformed in (jnp.ones((1,129),dtype=jnp.float32),jnp.ones(9,dtype=jnp.float32)):
    # Evaluate `rejected` from the current inputs and state.
    rejected = ...  # TODO: compute rejected
    # Run the boundary check and catch the expected exception:
    try:fused_bias_relu(x,malformed)
    except ValueError:rejected=True
    # Verify contract: `rejected`.
    assert rejected  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print("Malformed bias axes rejected")
```

<details><summary>Reference solution and reasoning</summary>

```python
# Reject hidden bias-axis broadcasting (Practice): Explicit validation turns a confusing broadcast result into...
# Iterate over `malformed` to step through the computation:
for malformed in (jnp.ones((1,129),dtype=jnp.float32),jnp.ones(9,dtype=jnp.float32)):
    # Evaluate `rejected` from the current inputs and state.
    rejected=False
    # Run the boundary check and catch the expected exception:
    try:fused_bias_relu(x,malformed)
    except ValueError:rejected=True
    # Verify contract: `rejected`.
    assert rejected
# Print the observed values to compare against the expected result.
print("Malformed bias axes rejected")
```

Explicit validation turns a confusing broadcast result into a local contract error.

</details>

## Check a permutation invariant

**Challenge**

Permute columns and bias entries together, then undo the permutation. Verify the output is unchanged.

<details><summary>Hint</summary>

Use a deterministic reversed column order; do not permute the bias independently.

</details>

### How to write: Check a permutation invariant — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.arange(n, dtype=...)` — Creates a 1-D JAX array of evenly spaced values `[0, 1, ..., n-1]` on the target device.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Initialize array `order` with explicit values and shape.
2. Run `fused_bias_relu` to compute `permuted`.
3. Verify that computed values match the expected reference within numerical tolerance.
4. Print the observed values to compare against the expected result.

**Starter code scaffold (fill in the TODOs):**

```python
# Check a permutation invariant (Challenge): The kernel is columnwise when the corresponding bias travels...
# Initialize array `order` with explicit values and shape.
order = jnp.arange(...)  # TODO: compute order
# Run `fused_bias_relu` to compute `permuted`.
permuted = fused_bias_relu(...)  # TODO: compute permuted
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(permuted[:,order],actual,rtol=1e-6,atol=1e-6)
# Print the observed values to compare against the expected result.
print("Column/bias pairing survives a joint permutation")
```

<details><summary>Reference solution and reasoning</summary>

```python
# Check a permutation invariant (Challenge): The kernel is columnwise when the corresponding bias travels...
# Initialize array `order` with explicit values and shape.
order=jnp.arange(128,-1,-1)
# Run `fused_bias_relu` to compute `permuted`.
permuted=fused_bias_relu(x[:,order],bias[order])
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(permuted[:,order],actual,rtol=1e-6,atol=1e-6)
# Print the observed values to compare against the expected result.
print("Column/bias pairing survives a joint permutation")
```

The kernel is columnwise when the corresponding bias travels with each column. This catches a wrong window map without relying only on one original fixture.

</details>

## Check your understanding

What does a successful CPU interpret run establish for this target-oriented kernel?

1. The TPU kernel has measured high throughput.
2. The tested numerical Ref/grid semantics agree with the reference; target lowering and performance remain separate checks.
3. Every TPU generation accepts all block sizes.

<details><summary>Answer and explanation</summary>

The tested numerical Ref/grid semantics agree with the reference; target lowering and performance remain separate checks.

Interpretation is useful correctness evidence. It does not execute the target compiler or hardware, so those claims require a separate guarded target run.

</details>

## Diagnose the result

If every row has a repeated bias pattern, inspect the bias BlockSpec and its column coordinate. If CPU interpretation passes but TPU lowering fails, preserve the target error and inspect layout/dtype restrictions. If a target command succeeds on a CPU-only environment, its fallback guard is broken and the receipt must not be called TPU validation.

## Carry forward

- Map each operand according to its meaning
- Use a conservative TPU tile contract
- State where precision changes
- Make the execution mode inspectable
- Run the real target path as a separate lab
- Compare with a strong baseline before optimizing

## Keep your evidence

Broadcast/boundary checks, represented-input precision oracle and a tested no-fallback TPU guard; target execution remains separately unverified. Keep the environment, observed outputs and your explanation of the figure.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [Pallas grids and block specifications](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
- [Writing TPU kernels: layout and memory restrictions](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
- [TPU pipelining and emit_pipeline](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)
- [TPU interpretation is simulation, not target execution](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.InterpretParams.html)

