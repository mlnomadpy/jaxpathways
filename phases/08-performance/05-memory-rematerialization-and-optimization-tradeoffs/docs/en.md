# Memory, rematerialization, and optimization tradeoffs

Phase 08: Performance diagnosis · about 100 minutes · CPU

## What you will be able to do

- Explain why reverse-mode differentiation needs forward intermediates.
- Compare ordinary and rematerialized gradients with an independent reverse recurrence.
- Inspect saved residual descriptions and compiled memory estimates without equating them.
- Measure warm gradient calls and explain why a memory benefit is workload-dependent.

## The problem

A model runs out of memory during differentiation. Someone suggests checkpointing every layer. Will that change the answer, and will it necessarily reduce this program’s memory? We will compare gradients, inspect saved residuals and measure compiled estimates and timings before making that decision.

## The idea

Reverse-mode differentiation needs information from forward execution. Rematerialization recomputes some of that information instead of retaining it. This exchanges storage for work inside a computation and is distinct from saving a training checkpoint to disk.

## Follow an intermediate from creation to last use

Imagine a backward rule needs an intermediate from the first operation in a chain. One strategy retains it; another reruns enough forward work to reconstruct it. Draw the value's lifetime and mark which computation is repeated.

Saved autodiff entries, estimated bytes and compiled peak memory are different quantities. Fusion, temporary buffers and reuse can change the compiled memory picture even when a source-level list of saved values changes.

Read each plot with its own units. Fewer entries alone do not establish lower peak device memory or faster execution. Use the storage model to form a hypothesis, then inspect compiler information and actual target behavior.

### Pause and reason

Why might fewer saved values leave compiled memory unchanged?

<details><summary>Compare your reasoning</summary>

The compiler may already eliminate or reuse them, while other temporaries dominate peak usage. Source-level counts do not fully determine compiled buffer lifetimes.

</details>

## Follow one layer backward

Our layer multiplies a vector by a fixed matrix and applies tanh. In the backward pass, the derivative of tanh uses the forward output. If that output is retained, the derivative can reuse it. If selected intermediates are discarded, the program must reconstruct the needed value from retained inputs and weights.

For layer output $h$, the local derivative factor is $1-h^2$. Multiplying the incoming cotangent by that factor, then applying the transposed weight matrix, propagates sensitivity to the preceding vector. The NumPy reference implements this recurrence explicitly instead of calling autodiff.

$$
\bar x=W^\top\bigl(\bar h\odot(1-h^2)\bigr)
$$

## Choose a boundary around pure work

We apply jax.checkpoint to the layer function. The outer objective uses four layers and a sum of squared final outputs. The reference and rematerialized versions share the same weights and input. If a function relies on untracked mutation, hidden randomness or effects, recomputation can have a different meaning; explicit state and keys remain essential.

Checkpointing the entire loss at an unhelpful boundary may simply recompute nearly all work without eliminating the intermediates you expected. Compare policies at meaningful blocks, and start with a small function whose derivative you can verify.

## Inspect what autodiff saves

print_saved_residuals names values retained for differentiation. The output describes a constant weight matrix and selected vector intermediates. Count the entries to compare policies, but read their shapes too: one large matrix may cost more than many scalar entries. The printed list is a diagnostic representation, not a device-memory allocation trace.

In this fixed example, the ordinary policy reports nine residual descriptions and layer rematerialization reports six. Those counts explain that the differentiation strategy changed; they do not guarantee a proportional reduction in peak bytes.

## Read compiled memory estimates separately

Compile the gradient for the same input contract and inspect memory_analysis. Argument, output, temporary and alias bytes answer different questions. Backend support and reporting conventions can vary, so record missing analysis as unavailable instead of substituting zero.

In the tested small CPU example, both policies report the same temporary-buffer estimate. That is a useful result: fewer saved autodiff values need not reduce this compiled program’s allocated temporary storage. Fusion, buffer reuse and compiler decisions stand between a mathematical strategy and a memory allocation.

## Measure the tradeoff you actually care about

Warm both compiled gradients and collect synchronized calls. Keep all samples, not just the fastest. Rematerialization can add arithmetic; on a tiny workload, dispatch and noise can dominate. No strict timing ordering is required.

A training-memory claim requires the complete training step, representative model and batch shapes, and actual peak memory on the target device. Include optimizer state and input buffers. Test whether the policy enables the intended larger batch without changing the objective, then verify quality and runtime again.

## Define two differentiation policies

Create a Python file in the course environment. Add this block after the preceding block, then run the assembled file.

```python
# Step 1 — Define two differentiation policies: Both functions compute the same pure objective; only the...
# Import contextlib for this computation.
import contextlib
import io
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import print_saved_residuals
# Draw pseudorandom samples for `rng` using the explicit RNG state.
rng=np.random.default_rng(18)
# Create device-backed JAX array `w`.
w=jnp.asarray((rng.normal(size=(16,16))*.1).astype(np.float32))
# Initialize array `x` with explicit values and shape.
x=jnp.asarray(np.linspace(-.5,.5,16,dtype=np.float32))
# Function `layer(v)` implementing this stage's computation:
# Return `jnp.tanh(w @ v)` to the caller.
def layer(v):return jnp.tanh(w@v)
# Function `objective(v, rematerialize)` implementing this stage's computation:
def objective(v,rematerialize=False):
    # Run `jax.checkpoint` to compute `operation`.
    operation=jax.checkpoint(layer) if rematerialize else layer
    # Repeat the update loop over `range(4)` steps:
    # Run `operation` to compute `v`.
    for _ in range(4):v=operation(v)
    # Return `jnp.sum(v * v)` to the caller.
    return jnp.sum(v*v)
# Evaluate `plain` from the current inputs and state.
plain=lambda v:objective(v,False)
# Evaluate `remat` from the current inputs and state.
remat=lambda v:objective(v,True)
# Function `residual_report(fn)` implementing this stage's computation:
def residual_report(fn):
    # Run `io.StringIO` to compute `stream`.
    stream=io.StringIO()
    # Enter `contextlib.redirect_stdout(stream)` context block:
    with contextlib.redirect_stdout(stream):print_saved_residuals(fn,x)
    # Evaluate `lines` from the current inputs and state.
    lines=[line for line in stream.getvalue().splitlines() if line.strip()]
    # Return `lines` to the caller.
    return lines
# Evaluate `reports` from the current inputs and state.
reports=[residual_report(plain),residual_report(remat)]
```

Both functions compute the same pure objective; only the recomputation boundary differs. The diagnostic records descriptions rather than claiming physical byte savings.

## Verify with an independent reverse pass

Create a Python file in the course environment. Add this block after the preceding block, then run the assembled file.

```python
# Independent reverse recurrence through the same explicitly specified network.
host_w=np.asarray(w);values=[np.asarray(x)]
# Repeat the update loop over `range(4)` steps:
# Perform matrix contraction / projection to compute ``.
for _ in range(4):values.append(np.tanh(host_w@values[-1]))
# Evaluate `cotangent` from the current inputs and state.
cotangent=2*values[-1]
# Iterate over `i` to step through the computation:
for i in range(4,0,-1):cotangent=host_w.T@(cotangent*(1-values[i]**2))
# Iterate over `fn` to step through the computation:
for fn in (plain,remat):
    # Differentiate the objective to obtain gradients ``.
    np.testing.assert_allclose(jax.grad(fn)(x),cotangent,rtol=3e-5,atol=1e-7)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.grad(plain)(x),jax.grad(remat)(x),rtol=1e-6,atol=1e-7)
# Differentiate the objective to obtain `executables` via automatic differentiation.
executables=[jax.jit(jax.grad(fn)).lower(x).compile() for fn in (plain,remat)]
# Evaluate `memory` from the current inputs and state.
# Evaluate `times` from the current inputs and state.
memory=[];times=[]
# Iterate over `executable` to step through the computation:
for executable in executables:
    # Run `executable.memory_analysis` to compute `analysis`.
    analysis=executable.memory_analysis()
    # Append the current step result to `memory`.
    memory.append(None if analysis is None else {name:getattr(analysis,name) for name in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')})
    # Synchronize host execution until asynchronous device computation completes.
    # Synchronize host execution until asynchronous device computation completes.
    executable(x).block_until_ready();samples=[]
    # Repeat the update loop over `range(5)` steps:
    for _ in range(5):
        # Synchronize host execution until asynchronous device computation completes.
        # Synchronize host execution until asynchronous device computation completes.
        # Synchronize host execution until asynchronous device computation completes.
        start=time.perf_counter();executable(x).block_until_ready();samples.append(time.perf_counter()-start)
    # Append the current step result to `times`.
    times.append(samples)
# Print the observed values to compare against the expected result.
print('Saved residual descriptions:')
# Iterate over `(name, lines)` to step through the computation:
for name,lines in zip(['plain','remat'],reports):print(name,'\n'+'\n'.join(lines))
# Print the observed values to compare against the expected result.
print('Compiler memory estimates:',memory)
# Print diagnostic summary of the computed outputs.
print('Synchronized gradient samples:',times)
# Print diagnostic summary of the computed outputs.
print('Independent reverse recurrence agrees with both gradients.')
```

The NumPy recurrence propagates cotangents through saved host activations, providing an oracle independent of JAX differentiation.

## Run the example

```python
# Step 1 — Define two differentiation policies: Both functions compute the same pure objective; only the...
# Import contextlib for this computation.
import contextlib
import io
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import print_saved_residuals
# Draw pseudorandom samples for `rng` using the explicit RNG state.
rng=np.random.default_rng(18)
# Create device-backed JAX array `w`.
w=jnp.asarray((rng.normal(size=(16,16))*.1).astype(np.float32))
# Initialize array `x` with explicit values and shape.
x=jnp.asarray(np.linspace(-.5,.5,16,dtype=np.float32))
# Function `layer(v)` implementing this stage's computation:
# Return `jnp.tanh(w @ v)` to the caller.
def layer(v):return jnp.tanh(w@v)
# Function `objective(v, rematerialize)` implementing this stage's computation:
def objective(v,rematerialize=False):
    # Run `jax.checkpoint` to compute `operation`.
    operation=jax.checkpoint(layer) if rematerialize else layer
    # Repeat the update loop over `range(4)` steps:
    # Run `operation` to compute `v`.
    for _ in range(4):v=operation(v)
    # Return `jnp.sum(v * v)` to the caller.
    return jnp.sum(v*v)
# Evaluate `plain` from the current inputs and state.
plain=lambda v:objective(v,False)
# Evaluate `remat` from the current inputs and state.
remat=lambda v:objective(v,True)
# Function `residual_report(fn)` implementing this stage's computation:
def residual_report(fn):
    # Run `io.StringIO` to compute `stream`.
    stream=io.StringIO()
    # Enter `contextlib.redirect_stdout(stream)` context block:
    with contextlib.redirect_stdout(stream):print_saved_residuals(fn,x)
    # Evaluate `lines` from the current inputs and state.
    lines=[line for line in stream.getvalue().splitlines() if line.strip()]
    # Return `lines` to the caller.
    return lines
# Evaluate `reports` from the current inputs and state.
reports=[residual_report(plain),residual_report(remat)]

# Independent reverse recurrence through the same explicitly specified network.
host_w=np.asarray(w);values=[np.asarray(x)]
# Repeat the update loop over `range(4)` steps:
# Perform matrix contraction / projection to compute ``.
for _ in range(4):values.append(np.tanh(host_w@values[-1]))
# Evaluate `cotangent` from the current inputs and state.
cotangent=2*values[-1]
# Iterate over `i` to step through the computation:
for i in range(4,0,-1):cotangent=host_w.T@(cotangent*(1-values[i]**2))
# Iterate over `fn` to step through the computation:
for fn in (plain,remat):
    # Differentiate the objective to obtain gradients ``.
    np.testing.assert_allclose(jax.grad(fn)(x),cotangent,rtol=3e-5,atol=1e-7)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.grad(plain)(x),jax.grad(remat)(x),rtol=1e-6,atol=1e-7)
# Differentiate the objective to obtain `executables` via automatic differentiation.
executables=[jax.jit(jax.grad(fn)).lower(x).compile() for fn in (plain,remat)]
# Evaluate `memory` from the current inputs and state.
# Evaluate `times` from the current inputs and state.
memory=[];times=[]
# Iterate over `executable` to step through the computation:
for executable in executables:
    # Run `executable.memory_analysis` to compute `analysis`.
    analysis=executable.memory_analysis()
    # Append the current step result to `memory`.
    memory.append(None if analysis is None else {name:getattr(analysis,name) for name in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')})
    # Synchronize host execution until asynchronous device computation completes.
    # Synchronize host execution until asynchronous device computation completes.
    executable(x).block_until_ready();samples=[]
    # Repeat the update loop over `range(5)` steps:
    for _ in range(5):
        # Synchronize host execution until asynchronous device computation completes.
        # Synchronize host execution until asynchronous device computation completes.
        # Synchronize host execution until asynchronous device computation completes.
        start=time.perf_counter();executable(x).block_until_ready();samples.append(time.perf_counter()-start)
    # Append the current step result to `times`.
    times.append(samples)
# Print the observed values to compare against the expected result.
print('Saved residual descriptions:')
# Iterate over `(name, lines)` to step through the computation:
for name,lines in zip(['plain','remat'],reports):print(name,'\n'+'\n'.join(lines))
# Print the observed values to compare against the expected result.
print('Compiler memory estimates:',memory)
# Print diagnostic summary of the computed outputs.
print('Synchronized gradient samples:',times)
# Print diagnostic summary of the computed outputs.
print('Independent reverse recurrence agrees with both gradients.')
```

Expected: Both gradients agree with the independent reverse recurrence. The saved-residual descriptions differ; the small tested CPU program reports equal temporary-memory estimates. Timings vary and no speedup is asserted.

## Changed autodiff storage does not guarantee changed compiled buffers

**Predict:** Must the two panels tell the same story about memory?

![Changed autodiff storage does not guarantee changed compiled buffers](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The upper panel counts saved-residual descriptions for ordinary differentiation and layer rematerialization. The vertical unit is entries, not bytes. In this fixed network the counts are nine and six, including a shared constant matrix; the decrease shows a changed differentiation strategy.

The lower panel reports compiler-estimated temporary bytes when that analysis is supported. In the recorded CPU example both bars have height $256$. Equal bars mean the compiler reports no reduction in this category for these shapes, even though the residual-entry count decreased.

### Connect it to the computation

The independent reverse recurrence checks the numerical derivative for both policies. The figure then compares two distinct storage views: values named by autodiff and temporary buffers estimated after compilation. Compiler reuse and fusion can make their relationship non-proportional.

These are not measurements of peak accelerator allocation or total training memory. The runtime samples are recorded separately, and a meaningful policy decision needs representative model size, optimizer state and actual target-device measurements. If a backend lacks memory analysis, the plot explicitly substitutes a labeled availability panel rather than treating missing bytes as zero.

```python
# Compute figure data for: Changed autodiff storage does not guarantee changed compiled buffers
# Evaluate `panels` from the current inputs and state.
panels=[{'kind':'bar','title':'Autodiff residual descriptions','labels':['ordinary','layer remat'],'ylabel':'saved entries (not bytes)','series':[{'label':'residual descriptions','y':[len(r) for r in reports]}]}]
# Branch on condition `all((m is not None for m in memory))`:
if all(m is not None for m in memory):
    panels.append({'kind':'bar','title':'Compiler memory estimate','labels':['ordinary','layer remat'],'ylabel':'estimated temporary bytes','series':[{'label':'compiler temporaries','y':[m['temp_size_in_bytes'] for m in memory]}]})
else:
    panels.append({'kind':'bar','title':'Compiler memory analysis unavailable','labels':['ordinary','layer remat'],'ylabel':'analysis available (1=yes)','series':[{'label':'availability, not byte count','y':[int(m is not None) for m in memory]}]})
# Evaluate `visual_data` from the current inputs and state.
visual_data={'kind':'panels','panels':panels}
```

## Recorded reference execution

CPU run: 2026-10-08T14:03:57.545581+00:00. JAX 0.9.2.

```text
Saved residual descriptions:
plain 
f32[16,16] from a constant
f32[16] output of tanh from <string>:14:20 (layer)
f32[16] output of sub from <string>:14:20 (layer)
f32[16] output of tanh from <string>:14:20 (layer)
f32[16] output of sub from <string>:14:20 (layer)
f32[16] output of tanh from <string>:14:20 (layer)
f32[16] output of sub from <string>:14:20 (layer)
f32[16] output of tanh from <string>:14:20 (layer)
f32[16] output of sub from <string>:14:20 (layer)
remat 
f32[16,16] from a constant
f32[16] from the argument v
f32[16] output of tanh from <string>:14:20 (layer)
f32[16] output of tanh from <string>:14:20 (layer)
f32[16] output of tanh from <string>:14:20 (layer)
f32[16] output of tanh from <string>:14:20 (layer)
Compiler memory estimates: [{'argument_size_in_bytes': 64, 'output_size_in_bytes': 64, 'temp_size_in_bytes': 256, 'alias_size_in_bytes': 0}, {'argument_size_in_bytes': 64, 'output_size_in_bytes': 64, 'temp_size_in_bytes': 256, 'alias_size_in_bytes': 0}]
Synchronized gradient samples: [[3.591598942875862e-05, 1.554097980260849e-05, 1.0082963854074478e-05, 9.790994226932526e-06, 1.0209158062934875e-05], [1.8999911844730377e-05, 1.1750031262636185e-05, 1.0792165994644165e-05, 7.2089023888111115e-06, 7.832888513803482e-06]]
Independent reverse recurrence agrees with both gradients.
Saved residual descriptions:
plain 
f32[16,16] from a constant
f32[16] output of tanh from <string>:63:20 (layer)
f32[16] output of sub from <string>:63:20 (layer)
f32[16] output of tanh from <string>:63:20 (layer)
f32[16] output of sub from <string>:63:20 (layer)
f32[16] output of tanh from <string>:63:20 (layer)
f32[16] output of sub from <string>:63:20 (layer)
f32[16] output of tanh from <string>:63:20 (layer)
f32[16] output of sub from <string>:63:20 (layer)
remat 
f32[16,16] from a constant
f32[16] from the argument v
f32[16] output of tanh from <string>:63:20 (layer)
f32[16] output of tanh from <string>:63:20 (layer)
f32[16] output of tanh from <string>:63:20 (layer)
f32[16] output of tanh from <string>:63:20 (layer)
Compiler memory estimates: [{'argument_size_in_bytes': 64, 'output_size_in_bytes': 64, 'temp_size_in_bytes': 256, 'alias_size_in_bytes': 0}, {'argument_size_in_bytes': 64, 'output_size_in_bytes': 64, 'temp_size_in_bytes': 256, 'alias_size_in_bytes': 0}]
Synchronized gradient samples: [[3.0374620109796524e-05, 1.5833880752325058e-05, 1.0499730706214905e-05, 1.020822674036026e-05, 7.166992872953415e-06], [1.67088583111763e-05, 7.541850209236145e-06, 7.291790097951889e-06, 7.457565516233444e-06, 6.875023245811462e-06]]
Independent reverse recurrence agrees with both gradients.
Whole-objective residuals: ['f32[16,16] from a constant', 'f32[16] from the argument v']
Whole-objective memory estimate: CompiledMemoryStats(generated_code_size_in_bytes=0, argument_size_in_bytes=64, output_size_in_bytes=64, alias_size_in_bytes=0, temp_size_in_bytes=256, host_generated_code_size_in_bytes=0, host_argument_size_in_bytes=0, host_output_size_in_bytes=0, host_alias_size_in_bytes=0, host_temp_size_in_bytes=0)
Directional derivative finite/automatic: 0.007404538337141275 0.00740453926846385 epsilon 0.001
Higher-order directional derivatives agree.
PASS: performance-05

```

## Move the boundary to the entire objective

**Predict before running:** Will checkpointing the whole objective preserve the gradient? Does that alone demonstrate useful memory savings?

```python
# Experiment — Move the boundary to the entire objective: Correctness is necessary but does not choose the best policy.
whole=jax.checkpoint(plain)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.grad(whole)(x),cotangent,rtol=3e-5,atol=1e-7)
# Print the observed values to compare against the expected result.
print('Whole-objective residuals:',residual_report(whole))
# Differentiate the objective to obtain `whole_executable` via automatic differentiation.
whole_executable=jax.jit(jax.grad(whole)).lower(x).compile()
# Print the observed values to compare against the expected result.
print('Whole-objective memory estimate:',whole_executable.memory_analysis())
```

**Expected:** The derivative remains correct; residual and compiled-memory reports describe this additional boundary.

Correctness is necessary but does not choose the best policy. The boundary determines what is retained and recomputed; inspect actual memory and timing for the intended workload.

## Make it yours

Use the direction $v=(1,\ldots,1)$ and central differences to independently check a directional derivative of the original objective at the existing input. Compare it with the gradient dot direction and report the perturbation.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.grad(loss_fn)(params, ...)` — Transforms a scalar-output function into a function returning the gradient PyTree with the same structure as `params`.

**Step-by-step implementation plan:**
1. Initialize array `direction` with explicit values and shape.
2. Evaluate `finite` from the current inputs and state.
3. Differentiate the objective to obtain `automatic` via automatic differentiation.
4. Verify that computed values match the expected reference within numerical tolerance.
5. Print the observed values to compare against the expected result.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Use the direction v=(1,\ldots,1) and central differences to...
# Initialize array `direction` with explicit values and shape.
direction = jnp.ones_like(...)  # TODO: compute direction
# Evaluate `finite` from the current inputs and state.
finite = ...  # TODO: compute finite
# Differentiate the objective to obtain `automatic` via automatic differentiation.
automatic = float(...)  # TODO: compute automatic
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(finite,automatic,rtol = ...  # TODO: compute np.testing.assert_allclose(finite,automatic,rtol
# Print the observed values to compare against the expected result.
print('Directional derivative finite/automatic:',finite,automatic,'epsilon',eps)
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Use the direction v=(1,\ldots,1) and central differences to...
# Initialize array `direction` with explicit values and shape.
direction=jnp.ones_like(x);eps=.001
# Evaluate `finite` from the current inputs and state.
finite=(float(plain(x+eps*direction))-float(plain(x-eps*direction)))/(2*eps)
# Differentiate the objective to obtain `automatic` via automatic differentiation.
automatic=float(jnp.vdot(jax.grad(plain)(x),direction))
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(finite,automatic,rtol=3e-3,atol=1e-6)
# Print the observed values to compare against the expected result.
print('Directional derivative finite/automatic:',finite,automatic,'epsilon',eps)
```

</details>

## Check higher-order differentiation

**Transfer**

Compare Hessian-vector products of the ordinary and rematerialized objectives along a nonuniform direction. Explain why equal first derivatives alone would be a weaker test.

<details><summary>Hint</summary>

Apply jvp to each gradient function with the same primal point and tangent.

</details>

### How to write: Check higher-order differentiation — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.linspace(start, stop, num)` — Creates `num` evenly spaced float points across the closed interval `[start, stop]`.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jnp.isfinite(x)` — Returns a boolean mask verifying that no element is `NaN` or `Inf`.
- `jax.grad(loss_fn)(params, ...)` — Transforms a scalar-output function into a function returning the gradient PyTree with the same structure as `params`.

**Step-by-step implementation plan:**
1. Initialize array `direction` with explicit values and shape.
2. Differentiate the objective to obtain `(_, hvp_plain)` via automatic differentiation.
3. Differentiate the objective to obtain `(_, hvp_remat)` via automatic differentiation.
4. Verify that computed values match the expected reference within numerical tolerance.
5. Confirm that all computed values remain finite (no NaN or Inf).

**Starter code scaffold (fill in the TODOs):**

```python
# Check higher-order differentiation (Transfer): Recomputation should preserve the differentiable function in...
# Initialize array `direction` with explicit values and shape.
direction = jnp.linspace(...)  # TODO: compute direction
# Differentiate the objective to obtain `(_, hvp_plain)` via automatic differentiation.
_,hvp_plain = jax.jvp(...)  # TODO: compute _,hvp_plain
# Differentiate the objective to obtain `(_, hvp_remat)` via automatic differentiation.
_,hvp_remat = jax.jvp(...)  # TODO: compute _,hvp_remat
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(hvp_plain,hvp_remat,rtol = ...  # TODO: compute np.testing.assert_allclose(hvp_plain,hvp_remat,rtol
# Confirm that all computed values remain finite (no NaN or Inf).
assert np.isfinite(np.asarray(hvp_plain)).all()  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print('Higher-order directional derivatives agree.')
```

<details><summary>Reference solution and reasoning</summary>

```python
# Check higher-order differentiation (Transfer): Recomputation should preserve the differentiable function in...
# Initialize array `direction` with explicit values and shape.
direction=jnp.linspace(-1.,1.,len(x))
# Differentiate the objective to obtain `(_, hvp_plain)` via automatic differentiation.
_,hvp_plain=jax.jvp(jax.grad(plain),(x,),(direction,))
# Differentiate the objective to obtain `(_, hvp_remat)` via automatic differentiation.
_,hvp_remat=jax.jvp(jax.grad(remat),(x,),(direction,))
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(hvp_plain,hvp_remat,rtol=3e-5,atol=1e-7)
# Confirm that all computed values remain finite (no NaN or Inf).
assert np.isfinite(np.asarray(hvp_plain)).all()
# Print the observed values to compare against the expected result.
print('Higher-order directional derivatives agree.')
```

Recomputation should preserve the differentiable function in this pure example, including its higher-order directional behavior. This still does not establish a device-memory improvement.

</details>

## Check your understanding

Rematerialization reduces the number of saved residual descriptions but the compiled temporary-byte estimates are equal. What is the supported conclusion?

1. The compiler’s report must be wrong.
2. The differentiation strategy changed, but this evidence does not show reduced compiled temporary storage.
3. The model must train faster because fewer values were listed.

<details><summary>Answer and explanation</summary>

The differentiation strategy changed, but this evidence does not show reduced compiled temporary storage.

Residual lists and compiler buffer estimates describe different levels. Report the observed equality and investigate the real target workload before claiming a memory or speed benefit.

</details>

## Diagnose the result

If gradients differ, first compare inputs, weights, random keys and any hidden effects. If memory does not improve, inspect the checkpoint boundary, residual shapes, compiler buffer estimates and full-step peak measurements before adding more checkpoint wrappers.

## Carry forward

- Recomputation trades retained intermediate values for extra work.
- Verify the derivative independently before evaluating a policy.
- Saved values, estimated buffers and measured peak memory must remain distinct.

## Keep your evidence

Keep the plain/rematerialized gradient and HVP comparisons, saved-residual counts, compiler-memory observations and synchronized timings. Explain why fewer residuals do not guarantee lower measured memory.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [JAX gradient checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html)
- [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html)

