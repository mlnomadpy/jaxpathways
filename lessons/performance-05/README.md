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
import contextlib
import io
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import print_saved_residuals
rng=np.random.default_rng(18)
w=jnp.asarray((rng.normal(size=(16,16))*.1).astype(np.float32))
x=jnp.asarray(np.linspace(-.5,.5,16,dtype=np.float32))
def layer(v):return jnp.tanh(w@v)
def objective(v,rematerialize=False):
    operation=jax.checkpoint(layer) if rematerialize else layer
    for _ in range(4):v=operation(v)
    return jnp.sum(v*v)
plain=lambda v:objective(v,False)
remat=lambda v:objective(v,True)
def residual_report(fn):
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):print_saved_residuals(fn,x)
    lines=[line for line in stream.getvalue().splitlines() if line.strip()]
    return lines
reports=[residual_report(plain),residual_report(remat)]
```

Both functions compute the same pure objective; only the recomputation boundary differs. The diagnostic records descriptions rather than claiming physical byte savings.

## Verify with an independent reverse pass

Create a Python file in the course environment. Add this block after the preceding block, then run the assembled file.

```python
# Independent reverse recurrence through the same explicitly specified network.
host_w=np.asarray(w);values=[np.asarray(x)]
for _ in range(4):values.append(np.tanh(host_w@values[-1]))
cotangent=2*values[-1]
for i in range(4,0,-1):cotangent=host_w.T@(cotangent*(1-values[i]**2))
for fn in (plain,remat):
    np.testing.assert_allclose(jax.grad(fn)(x),cotangent,rtol=3e-5,atol=1e-7)
np.testing.assert_allclose(jax.grad(plain)(x),jax.grad(remat)(x),rtol=1e-6,atol=1e-7)
executables=[jax.jit(jax.grad(fn)).lower(x).compile() for fn in (plain,remat)]
memory=[];times=[]
for executable in executables:
    analysis=executable.memory_analysis()
    memory.append(None if analysis is None else {name:getattr(analysis,name) for name in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')})
    executable(x).block_until_ready();samples=[]
    for _ in range(5):
        start=time.perf_counter();executable(x).block_until_ready();samples.append(time.perf_counter()-start)
    times.append(samples)
print('Saved residual descriptions:')
for name,lines in zip(['plain','remat'],reports):print(name,'\n'+'\n'.join(lines))
print('Compiler memory estimates:',memory)
print('Synchronized gradient samples:',times)
print('Independent reverse recurrence agrees with both gradients.')

```

The NumPy recurrence propagates cotangents through saved host activations, providing an oracle independent of JAX differentiation.

## Run the example

```python
import contextlib
import io
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import print_saved_residuals
rng=np.random.default_rng(18)
w=jnp.asarray((rng.normal(size=(16,16))*.1).astype(np.float32))
x=jnp.asarray(np.linspace(-.5,.5,16,dtype=np.float32))
def layer(v):return jnp.tanh(w@v)
def objective(v,rematerialize=False):
    operation=jax.checkpoint(layer) if rematerialize else layer
    for _ in range(4):v=operation(v)
    return jnp.sum(v*v)
plain=lambda v:objective(v,False)
remat=lambda v:objective(v,True)
def residual_report(fn):
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):print_saved_residuals(fn,x)
    lines=[line for line in stream.getvalue().splitlines() if line.strip()]
    return lines
reports=[residual_report(plain),residual_report(remat)]

# Independent reverse recurrence through the same explicitly specified network.
host_w=np.asarray(w);values=[np.asarray(x)]
for _ in range(4):values.append(np.tanh(host_w@values[-1]))
cotangent=2*values[-1]
for i in range(4,0,-1):cotangent=host_w.T@(cotangent*(1-values[i]**2))
for fn in (plain,remat):
    np.testing.assert_allclose(jax.grad(fn)(x),cotangent,rtol=3e-5,atol=1e-7)
np.testing.assert_allclose(jax.grad(plain)(x),jax.grad(remat)(x),rtol=1e-6,atol=1e-7)
executables=[jax.jit(jax.grad(fn)).lower(x).compile() for fn in (plain,remat)]
memory=[];times=[]
for executable in executables:
    analysis=executable.memory_analysis()
    memory.append(None if analysis is None else {name:getattr(analysis,name) for name in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')})
    executable(x).block_until_ready();samples=[]
    for _ in range(5):
        start=time.perf_counter();executable(x).block_until_ready();samples.append(time.perf_counter()-start)
    times.append(samples)
print('Saved residual descriptions:')
for name,lines in zip(['plain','remat'],reports):print(name,'\n'+'\n'.join(lines))
print('Compiler memory estimates:',memory)
print('Synchronized gradient samples:',times)
print('Independent reverse recurrence agrees with both gradients.')

```

Expected: Both gradients agree with the independent reverse recurrence. The saved-residual descriptions differ; the small tested CPU program reports equal temporary-memory estimates. Timings vary and no speedup is asserted.

## Changed autodiff storage does not guarantee changed compiled buffers

**Predict:** Must the two panels tell the same story about memory?

![Changed autodiff storage does not guarantee changed compiled buffers](../../phases/08-performance/05-memory-rematerialization-and-optimization-tradeoffs/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The upper panel counts saved-residual descriptions for ordinary differentiation and layer rematerialization. The vertical unit is entries, not bytes. In this fixed network the counts are nine and six, including a shared constant matrix; the decrease shows a changed differentiation strategy.

The lower panel reports compiler-estimated temporary bytes when that analysis is supported. In the recorded CPU example both bars have height $256$. Equal bars mean the compiler reports no reduction in this category for these shapes, even though the residual-entry count decreased.

### Connect it to the computation

The independent reverse recurrence checks the numerical derivative for both policies. The figure then compares two distinct storage views: values named by autodiff and temporary buffers estimated after compilation. Compiler reuse and fusion can make their relationship non-proportional.

These are not measurements of peak accelerator allocation or total training memory. The runtime samples are recorded separately, and a meaningful policy decision needs representative model size, optimizer state and actual target-device measurements. If a backend lacks memory analysis, the plot explicitly substitutes a labeled availability panel rather than treating missing bytes as zero.

```python
panels=[{'kind':'bar','title':'Autodiff residual descriptions','labels':['ordinary','layer remat'],'ylabel':'saved entries (not bytes)','series':[{'label':'residual descriptions','y':[len(r) for r in reports]}]}]
if all(m is not None for m in memory):
    panels.append({'kind':'bar','title':'Compiler memory estimate','labels':['ordinary','layer remat'],'ylabel':'estimated temporary bytes','series':[{'label':'compiler temporaries','y':[m['temp_size_in_bytes'] for m in memory]}]})
else:
    panels.append({'kind':'bar','title':'Compiler memory analysis unavailable','labels':['ordinary','layer remat'],'ylabel':'analysis available (1=yes)','series':[{'label':'availability, not byte count','y':[int(m is not None) for m in memory]}]})
visual_data={'kind':'panels','panels':panels}
```

## Recorded reference execution

CPU run: 2026-10-06T15:41:05.110057+00:00. JAX 0.9.2.

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
Synchronized gradient samples: [[2.5709159672260284e-05, 2.0499806851148605e-05, 2.533290535211563e-05, 3.2500363886356354e-05, 2.2792257368564606e-05], [2.1459069103002548e-05, 1.4791730791330338e-05, 8.790753781795502e-06, 9.458046406507492e-06, 9.499955922365189e-06]]
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
Synchronized gradient samples: [[2.7791131287813187e-05, 1.9917264580726624e-05, 2.2292137145996094e-05, 2.9708724468946457e-05, 1.0666903108358383e-05], [2.2666994482278824e-05, 1.837499439716339e-05, 1.095794141292572e-05, 9.375158697366714e-06, 9.709037840366364e-06]]
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
whole=jax.checkpoint(plain)
np.testing.assert_allclose(jax.grad(whole)(x),cotangent,rtol=3e-5,atol=1e-7)
print('Whole-objective residuals:',residual_report(whole))
whole_executable=jax.jit(jax.grad(whole)).lower(x).compile()
print('Whole-objective memory estimate:',whole_executable.memory_analysis())
```

**Expected:** The derivative remains correct; residual and compiled-memory reports describe this additional boundary.

Correctness is necessary but does not choose the best policy. The boundary determines what is retained and recomputed; inspect actual memory and timing for the intended workload.

## Make it yours

Use the direction $v=(1,\ldots,1)$ and central differences to independently check a directional derivative of the original objective at the existing input. Compare it with the gradient dot direction and report the perturbation.

<details><summary>Reference solution</summary>

```python
direction=jnp.ones_like(x);eps=.001
finite=(float(plain(x+eps*direction))-float(plain(x-eps*direction)))/(2*eps)
automatic=float(jnp.vdot(jax.grad(plain)(x),direction))
np.testing.assert_allclose(finite,automatic,rtol=3e-3,atol=1e-6)
print('Directional derivative finite/automatic:',finite,automatic,'epsilon',eps)
```

</details>

## Check higher-order differentiation

**Transfer**

Compare Hessian-vector products of the ordinary and rematerialized objectives along a nonuniform direction. Explain why equal first derivatives alone would be a weaker test.

<details><summary>Hint</summary>

Apply jvp to each gradient function with the same primal point and tangent.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
direction=jnp.linspace(-1.,1.,len(x))
_,hvp_plain=jax.jvp(jax.grad(plain),(x,),(direction,))
_,hvp_remat=jax.jvp(jax.grad(remat),(x,),(direction,))
np.testing.assert_allclose(hvp_plain,hvp_remat,rtol=3e-5,atol=1e-7)
assert np.isfinite(np.asarray(hvp_plain)).all()
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

