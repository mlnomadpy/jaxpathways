# Iterate with correctness and performance evidence

Phase 13: Pallas kernels · about 120 minutes · CPU

## What you will be able to do

- Build a representative shape/dtype correctness matrix for an actual Pallas pipeline.
- Separate kernel implementation error from input/output precision differences.
- Implement a guarded compile/warmup/synchronized target benchmark.
- Connect profiling, partitioning, tiling and numerical evidence to an optimization decision.

## The problem

A custom kernel is useful only if it stays correct when shapes and dtypes change and improves an actual workload. What evidence is enough to keep an optimization, and what should we do when no target hardware is available? We will build a repeatable audit and an executable target-only measurement protocol.

## The idea

A low-precision kernel can differ from a high-precision calculation because inputs changed representation, arithmetic changed, or the implementation is wrong. Separate these causes with references that answer different questions.

## Use two references to locate a precision gap

First compare the kernel with an independent computation using the same represented inputs. That isolates implementation behavior under the chosen numeric contract. Then compare the represented-input reference with the original high-precision calculation to estimate representation effects.

If the first comparison passes while the second shows a gap, changing indexing is unlikely to fix that gap. Investigate the precision policy and whether downstream quality tolerates it.

The error bars should keep those comparisons separate and state scale, dtype and tolerance. A passed numerical check also says nothing about speed. Keep workload identity, synchronization and target-device measurements with any optimization claim.

$$
\text{Speedup}(S_{\text{total}}) = \frac{1}{(1 - p) + p / s_{\text{kernel}}}
$$

### Pause and reason

Which comparison should you inspect before blaming quantization for a wrong output?

<details><summary>Compare your reasoning</summary>

Compare against an independent reference on the same represented inputs. A mismatch there may be an implementation error rather than unavoidable representation error.

</details>

## Choose cases that challenge the contract

The fixtures include a singleton matrix, one exact tile, a small tail and a larger tail. The same frozen random data is tested in float32 and bfloat16. A benchmark-sized divisible matrix alone would miss many boundary bugs. Every implementation must return the original logical shape; padding is internal work, not extra observations in the output.

## Use two references for two different questions

The represented-input oracle promotes the actual stored inputs to float32, computes $2X+Y$, then applies the declared output cast. Kernel agreement with that reference checks implementation correctness. Comparing against the original float32 dataset instead reveals input and output rounding. These differences should not be merged into one unexplained error number or hidden by arbitrarily loose tolerances.

## Make target timing a complete operation

The target protocol compiles separately, warms up five times and synchronizes every measured output. Inputs are already placed on the target before timing. The measured boundary includes logical-shape padding, the Pallas call and cropping, because those are part of this wrapper’s user-visible cost. A kernel-only benchmark would need a different explicit boundary and could not be compared directly with these results.

## Compare a strong baseline with the same contract

The baseline is jitted ordinary JAX with float32 arithmetic and the same final dtype cast. XLA can fuse simple expressions, so the custom kernel may lose. Report every sample, median and ninetieth percentile, plus compilation time and actual device. The percentile is a distribution summary, not the maximum. Repeated samples expose variability but do not erase cache, thermal or system-load effects.

## Relate local kernels to the distributed workload

A kernel optimizes an operation on local arrays; it does not automatically reduce all-reduce or resharding costs. Use a profile to determine whether the workload is limited by local compute, memory movement, input staging or communication. Keep the partitioning decision explicit and verify the same global numerical objective. A faster local operation may have little end-to-end benefit when another part dominates.

## Accept, revise or decline the optimization

Keep the change only when representative correctness and the declared target boundary improve enough to justify maintenance and specialization costs. If performance is unchanged or worse, retain the simpler baseline and record the experiment. If target hardware is unavailable, retain the tested kernel and its runnable target protocol, mark performance unverified, and do not fill the gap with interpret-mode timings. The scale synthesis assessment asks for profiling and partitioning evidence as well as the local kernel audit.

## Keep reference semantics, target mode and benchmark boundary together

Create main.py for the first block, then append each block in order in your course CPU environment.

```python
# Step 1 — Keep reference semantics, target mode and benchmark boundary together: The candidate uses the same checked Pallas pipeline.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Function `validate_pair(x, y, block)` implementing this stage's computation:
def validate_pair(x,y,block):
    # Guard input contract (`x.ndim != 2 or x.shape != y.shape or min(x.shape) < 1`) and fail fast if violated.
    if x.ndim!=2 or x.shape!=y.shape or min(x.shape)<1:
        raise ValueError("inputs must be nonempty matrices with identical shapes")
    # Guard input contract (`x.dtype != y.dtype or x.dtype not in (jnp.float32, jnp.bfloat16)`) and fail fast if violated.
    if x.dtype!=y.dtype or x.dtype not in (jnp.float32,jnp.bfloat16):
        raise ValueError("matching float32 or bfloat16 inputs required")
    # Guard input contract (`len(block) != 2 or any((not isinstance(n, int) or n < 1 for n in block))`) and fail fast if violated.
    if len(block)!=2 or any(not isinstance(n,int) or n<1 for n in block):
        raise ValueError("two positive integer block dimensions required")

# Function `pad_pair(x, y, block)` implementing this stage's computation:
def pad_pair(x,y,block):
    # Run `validate_pair` to perform the next check or state transition.
    validate_pair(x,y,block)
    # Evaluate `(m, n)` from the current inputs and state.
    # Compute `m,n` from `x.shape`
    m,n=x.shape
    bm,bn=block
    # Compute `padded` from `((m+bm-1)//bm*bm,(n+bn-1)//bn*bn)`
    padded=((m+bm-1)//bm*bm,(n+bn-1)//bn*bn)
    # Combine or mask array elements to form `pads`.
    pads=((0,padded[0]-m),(0,padded[1]-n))
    # Return `(jnp.pad(x, pads), jnp.pad(y, pads), padded)` to the caller.
    return jnp.pad(x,pads),jnp.pad(y,pads),padded

# Function `axpy_body(x_ref, y_ref, out_ref)` implementing this stage's computation:
def axpy_body(x_ref,y_ref,out_ref):
    # Cast or evaluate `result` in explicit floating-point precision.
    result=2.0*x_ref[...].astype(jnp.float32)+y_ref[...].astype(jnp.float32)
    # Run `result.astype` to compute `out_ref[...]`.
    out_ref[...]=result.astype(out_ref.dtype)

# Function `blocked_axpy(x, y, block)` implementing this stage's computation:
def blocked_axpy(x,y,block=(2,4)):
    # Combine or mask array elements to form `(px, py, padded)`.
    px,py,padded=pad_pair(x,y,block)
    # Compute `bm,bn` from `block`
    bm,bn=block
    # Invoke custom Pallas kernel or tile specification (`spec`).
    spec=pl.BlockSpec(block,lambda i,j:(i,j))
    # Combine or mask array elements to form `out`.
    out=pl.pallas_call(axpy_body,
        out_shape=jax.ShapeDtypeStruct(padded,x.dtype),
        grid=(padded[0]//bm,padded[1]//bn),
        in_specs=(spec,spec),out_specs=spec,interpret=True)(px,py)
    # Return `out[:x.shape[0], :x.shape[1]]` to the caller.
    return out[:x.shape[0],:x.shape[1]]

# Function `pipelined_axpy(x, y, block, buffers, ...)` implementing this stage's computation:
def pipelined_axpy(x,y,block=(8,128),buffers=2,no_pipelining=False,mode="simulate"):
    # Combine or mask array elements to form `(px, py, padded)`.
    px,py,padded=pad_pair(x,y,block)
    # Compute `bm,bn` from `block`
    bm,bn=block
    # Guard input contract (`bm % 8 or bn % 128`) and fail fast if violated.
    if bm%8 or bn%128:
        raise ValueError("pipeline blocks must be multiples of (8,128)")
    # Guard input contract (`buffers not in (2, 3)`) and fail fast if violated.
    if buffers not in (2,3):
        raise ValueError("this lab supports two or three buffers")
    # Guard input contract (`mode not in ('simulate', 'tpu')`) and fail fast if violated.
    if mode not in ("simulate","tpu"):
        raise ValueError("mode must explicitly be simulate or tpu")
    # Guard input contract (`mode == 'tpu' and (jax.default_backend() != 'tpu' or any((not isinstance(a, jax.core.Tracer) and any((d.platform != 'tpu' for d in a.devices())) for a in (x, y))))`) and fail fast if violated.
    if mode=="tpu" and (jax.default_backend()!="tpu" or any(not isinstance(a,jax.core.Tracer) and any(d.platform!="tpu" for d in a.devices()) for a in (x,y))):
        raise RuntimeError("Real TPU inputs/backend required; simulation fallback is disabled")
    # Invoke custom Pallas kernel or tile specification (`spec`).
    spec=pl.BlockSpec(block,lambda i,j:(i,j),pipeline_mode=pl.Buffered(buffer_count=buffers))
    # Invoke custom Pallas kernel or tile specification (`output_spec`).
    output_spec=pl.BlockSpec(block,lambda i,j:(i,j),pipeline_mode=pl.Buffered(buffer_count=2))
    # Function `outer(x_hbm, y_hbm, out_hbm)` implementing this stage's computation:
    def outer(x_hbm,y_hbm,out_hbm):
        # Combine or mask array elements to form ``.
        pltpu.emit_pipeline(axpy_body,grid=(padded[0]//bm,padded[1]//bn),
            in_specs=(spec,spec),out_specs=output_spec,
            no_pipelining=no_pipelining)(x_hbm,y_hbm,out_hbm)
    # Invoke custom Pallas kernel or tile specification (`whole`).
    whole=pl.BlockSpec(memory_space=pl.ANY)
    # Run `pltpu.InterpretParams` to compute `interpretation`.
    interpretation=pltpu.InterpretParams(detect_races=True) if mode=="simulate" else False
    # Combine or mask array elements to form `call`.
    call=pl.pallas_call(outer,out_shape=jax.ShapeDtypeStruct(padded,x.dtype),
        in_specs=(whole,whole),out_specs=whole,interpret=interpretation)
    # Branch on condition `mode == 'simulate'`:
    if mode=="simulate":
        # This describes a SIMULATED TPU layout, not the machine running the code.
        abstract=jax.sharding.AbstractMesh((1,),("simulated_device",),
            abstract_device=jax.sharding.AbstractDevice(device_kind="TPU v5 lite",num_cores=1))
        with jax.sharding.use_abstract_mesh(abstract):
            output=call(px,py)
    else:
        output=call(px,py)
    # Return `output[:x.shape[0], :x.shape[1]]` to the caller.
    return output[:x.shape[0],:x.shape[1]]

# Function `target_benchmark(candidate, baseline, args, repeats)` implementing this stage's computation:
def target_benchmark(candidate,baseline,args,repeats=20):
    # Guard input contract (`jax.default_backend() != 'tpu' or any((d.platform != 'tpu' for a in args for d in a.devices()))`) and fail fast if violated.
    if jax.default_backend()!="tpu" or any(d.platform!="tpu" for a in args for d in a.devices()):
        raise RuntimeError("TPU benchmark requires real TPU inputs; interpretation timings are not accepted")
    # Guard input contract (`repeats < 5`) and fail fast if violated.
    if repeats<5:
        raise ValueError("at least five repeated target measurements required")
    # Import time for this computation.
    import time
    # Compute `records` from `{}`
    records={}
    # Iterate over `(label, function)` to step through the computation:
    for label,function in (("candidate",candidate),("baseline",baseline)):
        # Record execution timing or profiler trace in `started`.
        started=time.perf_counter()
        # Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
        compiled=jax.jit(function).lower(*args).compile()
        # Record execution timing or profiler trace in `compile_seconds`.
        compile_seconds=time.perf_counter()-started
        # Repeat the update loop over `range(5)` steps:
        # Synchronize host execution until asynchronous device computation completes.
        for _ in range(5):compiled(*args).block_until_ready()
        # Compute `samples` from `[]`
        samples=[]
        # Repeat the update loop over `range(repeats)` steps:
        for _ in range(repeats):
            # Record execution timing or profiler trace in `started`.
            started=time.perf_counter()
            # Synchronize host execution until asynchronous device computation completes.
            compiled(*args).block_until_ready()
            # Record execution timing or profiler trace in ``.
            samples.append((time.perf_counter()-started)*1000)
        # Compute `records[label]` from `{"compile_seconds":compile_seconds,"samples_ms":samp...`
        records[label]={"compile_seconds":compile_seconds,"samples_ms":samples,
            "median_ms":float(np.median(samples)),"p90_ms":float(np.percentile(samples,90))}
    # Check which hardware backend (`cpu`, `gpu`, or `tpu`) JAX selected for `records['actual_backend']`.
    records["actual_backend"]=jax.default_backend()
    # Query the active JAX devices into `records['device_kind']`.
    records["device_kind"]=jax.devices()[0].device_kind
    # Combine or mask array elements to form `records['boundary']`.
    records["boundary"]="already-placed logical inputs; complete padded/cropped wrapper output; warm synchronized execution"
    # Return `records` to the caller.
    return records
```

The candidate uses the same checked Pallas pipeline. The benchmark rejects CPU inputs before compiling or timing. It measures the complete wrapper, including padding/cropping, and records compilation separately.

## Audit a matrix of shapes and precisions

Create main.py for the first block, then append each block in order in your course CPU environment.

```python
# Step 2 — Audit a matrix of shapes and precisions: A zero kernel error means the implementation honored the declared...
rng=np.random.default_rng(42)
# Evaluate `correctness` from the current inputs and state.
# Compute `correctness` from `[]`
correctness=[]
precision_gaps=[]
# Iterate over `shape` to step through the computation:
for shape in [(1,1),(8,128),(9,129),(17,257)]:
    # Cast or evaluate `original_x` in explicit floating-point precision.
    original_x=rng.normal(size=shape).astype(np.float32)
    # Cast or evaluate `original_y` in explicit floating-point precision.
    original_y=rng.normal(size=shape).astype(np.float32)
    # Compute `ideal` from `2*original_x+original_y`
    ideal=2*original_x+original_y
    # Evaluate `row` from the current inputs and state.
    # Compute `row` from `[]`
    row=[]
    gaps=[]
    # Loop over `dtype` in `(jnp.float32, jnp.bfloat16)`:
    for dtype in (jnp.float32,jnp.bfloat16):
        # Create device-backed JAX array `a`.
        # Create device-backed JAX array `b`.
        a=jnp.asarray(original_x,dtype=dtype)
        b=jnp.asarray(original_y,dtype=dtype)
        # Run `pipelined_axpy` to compute `actual`.
        actual=pipelined_axpy(a,b)
        # Cast or evaluate `represented` in explicit floating-point precision.
        represented=(2*a.astype(jnp.float32)+b.astype(jnp.float32)).astype(dtype)
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_array_equal(np.asarray(actual),np.asarray(represented))
        # Convert `error` to a host NumPy array for inspection or verification.
        error=float(np.max(np.abs(np.asarray(actual,dtype=np.float32)-np.asarray(represented,dtype=np.float32))))
        # Convert `gap` to a host NumPy array for inspection or verification.
        gap=float(np.max(np.abs(np.asarray(actual,dtype=np.float32)-ideal)))
        # Append the current step result to `row`.
        # Append the current step result to `row`.
        row.append(error)
        gaps.append(gap)
    # Append the current step result to `correctness`.
    # Append the current step result to `correctness`.
    correctness.append(row)
    precision_gaps.append(gaps)
# Print the observed values to compare against the expected result.
print("Kernel errors versus represented-input oracle:",correctness)
# Print diagnostic summary of the computed outputs.
print("Gaps versus original float32 data:",precision_gaps)
```

A zero kernel error means the implementation honored the declared precision contract. A nonzero gap to the original float32 dataset measures representation and output rounding, which must be reported separately.

## Check the evidence boundary before requesting a speed claim

Create main.py for the first block, then append each block in order in your course CPU environment.

```python
# Step 3 — Check the evidence boundary before requesting a speed claim: The runnable target entry point is the project target_tpu.py script.
if jax.default_backend()=="cpu":
    probe=jnp.ones((8,128),dtype=jnp.float32)
    rejected=False
    try:target_benchmark(lambda a,b:pipelined_axpy(a,b,mode="tpu"),lambda a,b:2*a+b,(probe,probe))
    except RuntimeError as error:
        rejected=True
        print("Expected benchmark refusal:",error)
    assert rejected
# Print the observed values to compare against the expected result.
print("CPU correctness audit complete; no TPU latency or speedup measured")
```

The runnable target entry point is the project target_tpu.py script. A missing target receipt remains a missing target receipt; the CPU result is not relabeled as performance data.

## Run the example

```python
# Step 1 — Keep reference semantics, target mode and benchmark boundary together: The candidate uses the same checked Pallas pipeline.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Function `validate_pair(x, y, block)` implementing this stage's computation:
def validate_pair(x,y,block):
    # Guard input contract (`x.ndim != 2 or x.shape != y.shape or min(x.shape) < 1`) and fail fast if violated.
    if x.ndim!=2 or x.shape!=y.shape or min(x.shape)<1:
        raise ValueError("inputs must be nonempty matrices with identical shapes")
    # Guard input contract (`x.dtype != y.dtype or x.dtype not in (jnp.float32, jnp.bfloat16)`) and fail fast if violated.
    if x.dtype!=y.dtype or x.dtype not in (jnp.float32,jnp.bfloat16):
        raise ValueError("matching float32 or bfloat16 inputs required")
    # Guard input contract (`len(block) != 2 or any((not isinstance(n, int) or n < 1 for n in block))`) and fail fast if violated.
    if len(block)!=2 or any(not isinstance(n,int) or n<1 for n in block):
        raise ValueError("two positive integer block dimensions required")

# Function `pad_pair(x, y, block)` implementing this stage's computation:
def pad_pair(x,y,block):
    # Run `validate_pair` to perform the next check or state transition.
    validate_pair(x,y,block)
    # Evaluate `(m, n)` from the current inputs and state.
    # Compute `m,n` from `x.shape`
    m,n=x.shape
    bm,bn=block
    # Compute `padded` from `((m+bm-1)//bm*bm,(n+bn-1)//bn*bn)`
    padded=((m+bm-1)//bm*bm,(n+bn-1)//bn*bn)
    # Combine or mask array elements to form `pads`.
    pads=((0,padded[0]-m),(0,padded[1]-n))
    # Return `(jnp.pad(x, pads), jnp.pad(y, pads), padded)` to the caller.
    return jnp.pad(x,pads),jnp.pad(y,pads),padded

# Function `axpy_body(x_ref, y_ref, out_ref)` implementing this stage's computation:
def axpy_body(x_ref,y_ref,out_ref):
    # Cast or evaluate `result` in explicit floating-point precision.
    result=2.0*x_ref[...].astype(jnp.float32)+y_ref[...].astype(jnp.float32)
    # Run `result.astype` to compute `out_ref[...]`.
    out_ref[...]=result.astype(out_ref.dtype)

# Function `blocked_axpy(x, y, block)` implementing this stage's computation:
def blocked_axpy(x,y,block=(2,4)):
    # Combine or mask array elements to form `(px, py, padded)`.
    px,py,padded=pad_pair(x,y,block)
    # Compute `bm,bn` from `block`
    bm,bn=block
    # Invoke custom Pallas kernel or tile specification (`spec`).
    spec=pl.BlockSpec(block,lambda i,j:(i,j))
    # Combine or mask array elements to form `out`.
    out=pl.pallas_call(axpy_body,
        out_shape=jax.ShapeDtypeStruct(padded,x.dtype),
        grid=(padded[0]//bm,padded[1]//bn),
        in_specs=(spec,spec),out_specs=spec,interpret=True)(px,py)
    # Return `out[:x.shape[0], :x.shape[1]]` to the caller.
    return out[:x.shape[0],:x.shape[1]]

# Function `pipelined_axpy(x, y, block, buffers, ...)` implementing this stage's computation:
def pipelined_axpy(x,y,block=(8,128),buffers=2,no_pipelining=False,mode="simulate"):
    # Combine or mask array elements to form `(px, py, padded)`.
    px,py,padded=pad_pair(x,y,block)
    # Compute `bm,bn` from `block`
    bm,bn=block
    # Guard input contract (`bm % 8 or bn % 128`) and fail fast if violated.
    if bm%8 or bn%128:
        raise ValueError("pipeline blocks must be multiples of (8,128)")
    # Guard input contract (`buffers not in (2, 3)`) and fail fast if violated.
    if buffers not in (2,3):
        raise ValueError("this lab supports two or three buffers")
    # Guard input contract (`mode not in ('simulate', 'tpu')`) and fail fast if violated.
    if mode not in ("simulate","tpu"):
        raise ValueError("mode must explicitly be simulate or tpu")
    # Guard input contract (`mode == 'tpu' and (jax.default_backend() != 'tpu' or any((not isinstance(a, jax.core.Tracer) and any((d.platform != 'tpu' for d in a.devices())) for a in (x, y))))`) and fail fast if violated.
    if mode=="tpu" and (jax.default_backend()!="tpu" or any(not isinstance(a,jax.core.Tracer) and any(d.platform!="tpu" for d in a.devices()) for a in (x,y))):
        raise RuntimeError("Real TPU inputs/backend required; simulation fallback is disabled")
    # Invoke custom Pallas kernel or tile specification (`spec`).
    spec=pl.BlockSpec(block,lambda i,j:(i,j),pipeline_mode=pl.Buffered(buffer_count=buffers))
    # Invoke custom Pallas kernel or tile specification (`output_spec`).
    output_spec=pl.BlockSpec(block,lambda i,j:(i,j),pipeline_mode=pl.Buffered(buffer_count=2))
    # Function `outer(x_hbm, y_hbm, out_hbm)` implementing this stage's computation:
    def outer(x_hbm,y_hbm,out_hbm):
        # Combine or mask array elements to form ``.
        pltpu.emit_pipeline(axpy_body,grid=(padded[0]//bm,padded[1]//bn),
            in_specs=(spec,spec),out_specs=output_spec,
            no_pipelining=no_pipelining)(x_hbm,y_hbm,out_hbm)
    # Invoke custom Pallas kernel or tile specification (`whole`).
    whole=pl.BlockSpec(memory_space=pl.ANY)
    # Run `pltpu.InterpretParams` to compute `interpretation`.
    interpretation=pltpu.InterpretParams(detect_races=True) if mode=="simulate" else False
    # Combine or mask array elements to form `call`.
    call=pl.pallas_call(outer,out_shape=jax.ShapeDtypeStruct(padded,x.dtype),
        in_specs=(whole,whole),out_specs=whole,interpret=interpretation)
    # Branch on condition `mode == 'simulate'`:
    if mode=="simulate":
        # This describes a SIMULATED TPU layout, not the machine running the code.
        abstract=jax.sharding.AbstractMesh((1,),("simulated_device",),
            abstract_device=jax.sharding.AbstractDevice(device_kind="TPU v5 lite",num_cores=1))
        with jax.sharding.use_abstract_mesh(abstract):
            output=call(px,py)
    else:
        output=call(px,py)
    # Return `output[:x.shape[0], :x.shape[1]]` to the caller.
    return output[:x.shape[0],:x.shape[1]]

# Function `target_benchmark(candidate, baseline, args, repeats)` implementing this stage's computation:
def target_benchmark(candidate,baseline,args,repeats=20):
    # Guard input contract (`jax.default_backend() != 'tpu' or any((d.platform != 'tpu' for a in args for d in a.devices()))`) and fail fast if violated.
    if jax.default_backend()!="tpu" or any(d.platform!="tpu" for a in args for d in a.devices()):
        raise RuntimeError("TPU benchmark requires real TPU inputs; interpretation timings are not accepted")
    # Guard input contract (`repeats < 5`) and fail fast if violated.
    if repeats<5:
        raise ValueError("at least five repeated target measurements required")
    # Import time for this computation.
    import time
    # Compute `records` from `{}`
    records={}
    # Iterate over `(label, function)` to step through the computation:
    for label,function in (("candidate",candidate),("baseline",baseline)):
        # Record execution timing or profiler trace in `started`.
        started=time.perf_counter()
        # Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
        compiled=jax.jit(function).lower(*args).compile()
        # Record execution timing or profiler trace in `compile_seconds`.
        compile_seconds=time.perf_counter()-started
        # Repeat the update loop over `range(5)` steps:
        # Synchronize host execution until asynchronous device computation completes.
        for _ in range(5):compiled(*args).block_until_ready()
        # Compute `samples` from `[]`
        samples=[]
        # Repeat the update loop over `range(repeats)` steps:
        for _ in range(repeats):
            # Record execution timing or profiler trace in `started`.
            started=time.perf_counter()
            # Synchronize host execution until asynchronous device computation completes.
            compiled(*args).block_until_ready()
            # Record execution timing or profiler trace in ``.
            samples.append((time.perf_counter()-started)*1000)
        # Compute `records[label]` from `{"compile_seconds":compile_seconds,"samples_ms":samp...`
        records[label]={"compile_seconds":compile_seconds,"samples_ms":samples,
            "median_ms":float(np.median(samples)),"p90_ms":float(np.percentile(samples,90))}
    # Check which hardware backend (`cpu`, `gpu`, or `tpu`) JAX selected for `records['actual_backend']`.
    records["actual_backend"]=jax.default_backend()
    # Query the active JAX devices into `records['device_kind']`.
    records["device_kind"]=jax.devices()[0].device_kind
    # Combine or mask array elements to form `records['boundary']`.
    records["boundary"]="already-placed logical inputs; complete padded/cropped wrapper output; warm synchronized execution"
    # Return `records` to the caller.
    return records

# Step 2 — Audit a matrix of shapes and precisions: A zero kernel error means the implementation honored the declared...
rng=np.random.default_rng(42)
# Evaluate `correctness` from the current inputs and state.
# Compute `correctness` from `[]`
correctness=[]
precision_gaps=[]
# Iterate over `shape` to step through the computation:
for shape in [(1,1),(8,128),(9,129),(17,257)]:
    # Cast or evaluate `original_x` in explicit floating-point precision.
    original_x=rng.normal(size=shape).astype(np.float32)
    # Cast or evaluate `original_y` in explicit floating-point precision.
    original_y=rng.normal(size=shape).astype(np.float32)
    # Compute `ideal` from `2*original_x+original_y`
    ideal=2*original_x+original_y
    # Evaluate `row` from the current inputs and state.
    # Compute `row` from `[]`
    row=[]
    gaps=[]
    # Loop over `dtype` in `(jnp.float32, jnp.bfloat16)`:
    for dtype in (jnp.float32,jnp.bfloat16):
        # Create device-backed JAX array `a`.
        # Create device-backed JAX array `b`.
        a=jnp.asarray(original_x,dtype=dtype)
        b=jnp.asarray(original_y,dtype=dtype)
        # Run `pipelined_axpy` to compute `actual`.
        actual=pipelined_axpy(a,b)
        # Cast or evaluate `represented` in explicit floating-point precision.
        represented=(2*a.astype(jnp.float32)+b.astype(jnp.float32)).astype(dtype)
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_array_equal(np.asarray(actual),np.asarray(represented))
        # Convert `error` to a host NumPy array for inspection or verification.
        error=float(np.max(np.abs(np.asarray(actual,dtype=np.float32)-np.asarray(represented,dtype=np.float32))))
        # Convert `gap` to a host NumPy array for inspection or verification.
        gap=float(np.max(np.abs(np.asarray(actual,dtype=np.float32)-ideal)))
        # Append the current step result to `row`.
        # Append the current step result to `row`.
        row.append(error)
        gaps.append(gap)
    # Append the current step result to `correctness`.
    # Append the current step result to `correctness`.
    correctness.append(row)
    precision_gaps.append(gaps)
# Print the observed values to compare against the expected result.
print("Kernel errors versus represented-input oracle:",correctness)
# Print diagnostic summary of the computed outputs.
print("Gaps versus original float32 data:",precision_gaps)

# Step 3 — Check the evidence boundary before requesting a speed claim: The runnable target entry point is the project target_tpu.py script.
if jax.default_backend()=="cpu":
    probe=jnp.ones((8,128),dtype=jnp.float32)
    rejected=False
    try:target_benchmark(lambda a,b:pipelined_axpy(a,b,mode="tpu"),lambda a,b:2*a+b,(probe,probe))
    except RuntimeError as error:
        rejected=True
        print("Expected benchmark refusal:",error)
    assert rejected
# Print the observed values to compare against the expected result.
print("CPU correctness audit complete; no TPU latency or speedup measured")
```

Expected: All eight shape/dtype cases exactly match the represented-input oracle. bfloat16 differs from original float32 values by a separately reported precision gap. The benchmark refuses CPU execution; no target speedup is reported.

## Precision gaps remain after kernel correctness passes

**Predict:** Can a kernel have zero implementation error and still differ from the original higher-precision data?

![Precision gaps remain after kernel correctness passes](../../phases/13-kernels/04-iterate-with-correctness-and-performance-evidence/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis names four frozen matrix fixtures. The vertical axis is maximum absolute difference between the actual simulated Pallas output and the original float32 dataset calculation. Float32 bars have zero height; bfloat16 bars are positive because stored inputs and final outputs are rounded.

The bfloat16 gaps are approximately $0.000862$, $0.02522$, $0.03079$ and $0.02922$. The largest matrix does not have the largest gap in this run. These are different frozen inputs, so the bar heights are separate from a monotonic law relating shape and error.

### Connect it to the computation

The separate represented-input oracle checks in the code report zero implementation error for every shape and dtype. This plot answers the additional precision question: what changed relative to the original float32 values despite correct execution of the chosen contract?

The figure contains no runtime or target-memory data. It is distinct from throughput, pipeline overlap or TPU correctness. Those claims require the guarded target runner and actual completed accelerator execution.

```python
# Compute figure data for: Precision gaps remain after kernel correctness passes
# Compute `labels` from `['1 x 1','8 x 128','9 x 129','17 x 257']`
labels=['1 x 1','8 x 128','9 x 129','17 x 257']
# Compute `visual_data` from `{'kind':'bar','labels':labels,'ylabel':'max gap vers...`
visual_data={'kind':'bar','labels':labels,'ylabel':'max gap versus original float32 values','series':[{'label':'float32','y':[r[0] for r in precision_gaps]},{'label':'bfloat16','y':[r[1] for r in precision_gaps]}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:05:40.931092+00:00. JAX 0.9.2.

```text
Kernel errors versus represented-input oracle: [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
Gaps versus original float32 data: [[0.0, 0.000862419605255127], [0.0, 0.025217056274414062], [0.0, 0.030788421630859375], [0.0, 0.0292205810546875]]
Expected benchmark refusal: TPU benchmark requires real TPU inputs; interpretation timings are not accepted
CPU correctness audit complete; no TPU latency or speedup measured
Kernel errors versus represented-input oracle: [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
Gaps versus original float32 data: [[0.0, 0.000862419605255127], [0.0, 0.025217056274414062], [0.0, 0.030788421630859375], [0.0, 0.0292205810546875]]
Expected benchmark refusal: TPU benchmark requires real TPU inputs; interpretation timings are not accepted
CPU correctness audit complete; no TPU latency or speedup measured
Joint row permutation preserves output identity
Early/final-only rounding: 1.0 1.0078125
Configuration, padded elements: (8, 128) 2 4096
Configuration, padded elements: (8, 128) 3 4096
Configuration, padded elements: (16, 256) 2 4096
Configuration, padded elements: (16, 256) 3 4096
Mixed and integer dtypes rejected
Modeled twofold-local speedup and zero-cost limit: 1.1764705882352942 1.4285714285714286
PASS: kernels-04

```

## Preserve correctness under a joint row permutation

**Predict before running:** Will permuting both input rows change any value after restoring the order?

```python
# Experiment — Preserve correctness under a joint row permutation: This operation has no coupling between rows.
a=jnp.asarray(rng.normal(size=(9,129)),dtype=jnp.float32)
# Create device-backed JAX array `b`.
b=jnp.asarray(rng.normal(size=(9,129)),dtype=jnp.float32)
# Construct `order` via `jnp.array([8,0,7,1,6,2,5,3,4])`
order=jnp.array([8,0,7,1,6,2,5,3,4])
# Run `jnp.argsort` to compute `inverse`.
inverse=jnp.argsort(order)
# Run `pipelined_axpy` to compute `original`.
original=pipelined_axpy(a,b)
# Run `pipelined_axpy` to compute `permuted`.
permuted=pipelined_axpy(a[order],b[order])[inverse]
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(permuted),np.asarray(original))
# Print the observed values to compare against the expected result.
print("Joint row permutation preserves output identity")
```

**Expected:** The restored output is identical.

This operation has no coupling between rows. A failure suggests ownership or data-pairing errors rather than ordinary floating-point variability.

## Make a precision-policy counterexample

**Predict before running:** Will rounding an intermediate bfloat16 value always equal one final output cast?

```python
# Experiment — Make a precision-policy counterexample: The order of representation changes is part of the numerical...
# Construct `first` via `jnp.array([1.00390625],dtype=jnp.float32)`
first=jnp.array([1.00390625],dtype=jnp.float32)
# Construct `second` via `jnp.array([0.00390625],dtype=jnp.float32)`
second=jnp.array([0.00390625],dtype=jnp.float32)
# Cast or evaluate `early` in explicit floating-point precision.
early=(first.astype(jnp.bfloat16).astype(jnp.float32)+second).astype(jnp.bfloat16)
# Cast or evaluate `late` in explicit floating-point precision.
late=(first+second).astype(jnp.bfloat16)
# Assert invariant `float(early[0])!=float(late[0])` holds
assert float(early[0])!=float(late[0])
# Print the observed values to compare against the expected result.
print("Early/final-only rounding:",float(early[0]),float(late[0]))
```

**Expected:** Early rounding yields 1.0 while one final cast yields 1.0078125.

The order of representation changes is part of the numerical algorithm. A reference must reflect the declared contract rather than a superficially similar expression.

## Make it yours

Add a (15,255) fixture, compare two tile choices and two buffer counts, and retain the same frozen inputs for every candidate. Report errors and padded work separately.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Create device-backed JAX array `b`.
2. Convert `expected` to a host NumPy array for inspection or verification.
3. Iterate over `block` to step through the computation:
4. Loop over `buffers` in `(2, 3)`:
5. Run `pipelined_axpy` to compute `result`.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Add a (15,255) fixture, compare two tile choices and two buffer...
a = jnp.asarray(...)  # TODO: compute a
# Create device-backed JAX array `b`.
b = jnp.asarray(...)  # TODO: compute b
# Convert `expected` to a host NumPy array for inspection or verification.
expected = ...  # TODO: compute expected
# Iterate over `block` to step through the computation:
for block in [(8,128),(16,256)]:
    # Loop over `buffers` in `(2, 3)`:
    for buffers in (2,3):
        # Run `pipelined_axpy` to compute `result`.
        result = pipelined_axpy(...)  # TODO: compute result
        # Check numerical equivalence within tolerance: `np.testing.assert_allclose(result,expected,rtol=1e-6,atol=1e-6)`
        np.testing.assert_allclose(result,expected,rtol = ...  # TODO: compute np.testing.assert_allclose(result,expected,rtol
        # Evaluate `gm` from the current inputs and state.
        # Compute `gm` from `(15+block[0]-1)//block[0]`
        gm = ...  # TODO: compute gm
        gn = ...  # TODO: compute gn
        # Print diagnostic summary of the computed outputs.
        print("Configuration, padded elements:",block,buffers,gm*gn*block[0]*block[1])
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Add a (15,255) fixture, compare two tile choices and two buffer...
a=jnp.asarray(np.random.default_rng(8).normal(size=(15,255)),dtype=jnp.float32)
# Create device-backed JAX array `b`.
b=jnp.asarray(np.random.default_rng(9).normal(size=(15,255)),dtype=jnp.float32)
# Convert `expected` to a host NumPy array for inspection or verification.
expected=2*np.asarray(a)+np.asarray(b)
# Iterate over `block` to step through the computation:
for block in [(8,128),(16,256)]:
    # Loop over `buffers` in `(2, 3)`:
    for buffers in (2,3):
        # Run `pipelined_axpy` to compute `result`.
        result=pipelined_axpy(a,b,block,buffers=buffers)
        # Check numerical equivalence within tolerance: `np.testing.assert_allclose(result,expected,rtol=1e-6,atol=1e-6)`
        np.testing.assert_allclose(result,expected,rtol=1e-6,atol=1e-6)
        # Evaluate `gm` from the current inputs and state.
        # Compute `gm` from `(15+block[0]-1)//block[0]`
        gm=(15+block[0]-1)//block[0]
        gn=(255+block[1]-1)//block[1]
        # Print diagnostic summary of the computed outputs.
        print("Configuration, padded elements:",block,buffers,gm*gn*block[0]*block[1])
```

</details>

## Reject unsupported dtype mixing

**Practice**

Pass a float32 matrix and a bfloat16 matrix together, then pass integer arrays. Verify neither is silently coerced into an undocumented kernel contract.

<details><summary>Hint</summary>

The wrapper promises matching float32 or bfloat16 inputs.

</details>

### How to write: Reject unsupported dtype mixing — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.zeros / jnp.ones(shape, dtype=...)` — Allocates a tensor of the given `shape` initialized with constants.

**Step-by-step implementation plan:**
1. Iterate over `(first, second)` to step through the computation:
2. Compute `rejected` from `False`
3. Run the boundary check and catch the expected exception:
4. Assert invariant `rejected` holds
5. Print the observed values to compare against the expected result.

**Starter code scaffold (fill in the TODOs):**

```python
# Reject unsupported dtype mixing (Practice): Supported dtype boundaries are a correctness decision.
# Iterate over `(first, second)` to step through the computation:
for first,second in [(jnp.ones((8,128),jnp.float32),jnp.ones((8,128),jnp.bfloat16)),(jnp.ones((8,128),jnp.int32),jnp.ones((8,128),jnp.int32))]:
    # Compute `rejected` from `False`
    rejected = ...  # TODO: compute rejected
    # Run the boundary check and catch the expected exception:
    try:pipelined_axpy(first,second)
    except ValueError:rejected=True
    # Assert invariant `rejected` holds
    assert rejected  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print("Mixed and integer dtypes rejected")
```

<details><summary>Reference solution and reasoning</summary>

```python
# Reject unsupported dtype mixing (Practice): Supported dtype boundaries are a correctness decision.
# Iterate over `(first, second)` to step through the computation:
for first,second in [(jnp.ones((8,128),jnp.float32),jnp.ones((8,128),jnp.bfloat16)),(jnp.ones((8,128),jnp.int32),jnp.ones((8,128),jnp.int32))]:
    # Compute `rejected` from `False`
    rejected=False
    # Run the boundary check and catch the expected exception:
    try:pipelined_axpy(first,second)
    except ValueError:rejected=True
    # Assert invariant `rejected` holds
    assert rejected
# Print the observed values to compare against the expected result.
print("Mixed and integer dtypes rejected")
```

Supported dtype boundaries are a correctness decision. Adding a new dtype requires a reference, target support and justified tolerances.

</details>

## Bound the end-to-end opportunity

**Challenge**

A profile attributes $30\%$ of time to a local operation. Derive the best end-to-end speedup if that operation becomes twice as fast, then the impossible-to-exceed limit if its cost vanished.

<details><summary>Hint</summary>

Keep the remaining workload time unchanged in the calculation.

</details>

### How to write: Bound the end-to-end opportunity — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Compute `local_speedup` from `2.0`
2. Compute `end_to_end` from `1/((1-fraction)+fraction/local_speedup)`
3. Compute `limit` from `1/(1-fraction)`
4. Execute `np.testing.assert_allclose(end_to_end,1.1764705882352942)`
5. Execute `np.testing.assert_allclose(limit,1.4285714285714286)`

**Starter code scaffold (fill in the TODOs):**

```python
# Bound the end-to-end opportunity (Challenge): These are conditional analytic estimates, not measured speedups.
fraction = ...  # TODO: compute fraction
# Compute `local_speedup` from `2.0`
local_speedup = ...  # TODO: compute local_speedup
# Compute `end_to_end` from `1/((1-fraction)+fraction/local_speedup)`
end_to_end = ...  # TODO: compute end_to_end
# Compute `limit` from `1/(1-fraction)`
limit = ...  # TODO: compute limit
# Execute `np.testing.assert_allclose(end_to_end,1.1764705882352942)`
np.testing.assert_allclose(end_to_end,1.1764705882352942)
# Execute `np.testing.assert_allclose(limit,1.4285714285714286)`
np.testing.assert_allclose(limit,1.4285714285714286)
# Print the observed values to compare against the expected result.
print("Modeled twofold-local speedup and zero-cost limit:",end_to_end,limit)
```

<details><summary>Reference solution and reasoning</summary>

```python
# Bound the end-to-end opportunity (Challenge): These are conditional analytic estimates, not measured speedups.
fraction=.3
# Compute `local_speedup` from `2.0`
local_speedup=2.0
# Compute `end_to_end` from `1/((1-fraction)+fraction/local_speedup)`
end_to_end=1/((1-fraction)+fraction/local_speedup)
# Compute `limit` from `1/(1-fraction)`
limit=1/(1-fraction)
# Execute `np.testing.assert_allclose(end_to_end,1.1764705882352942)`
np.testing.assert_allclose(end_to_end,1.1764705882352942)
# Execute `np.testing.assert_allclose(limit,1.4285714285714286)`
np.testing.assert_allclose(limit,1.4285714285714286)
# Print the observed values to compare against the expected result.
print("Modeled twofold-local speedup and zero-cost limit:",end_to_end,limit)
```

These are conditional analytic estimates, not measured speedups. Communication or scheduling changes can alter the fraction, so reprofile the complete workload.

</details>

## Check your understanding

The kernel matches the represented-input oracle exactly but differs from the original float32 dataset in bfloat16. What should the report say?

1. The kernel is necessarily incorrect.
2. The interpretation proves TPU throughput.
3. Implementation correctness passed for the declared precision policy; the separate representation gap must still be reported.

<details><summary>Answer and explanation</summary>

Implementation correctness passed for the declared precision policy; the separate representation gap must still be reported.

The references answer different questions. Correct execution of a low-precision contract does not imply identical values to a higher-precision dataset.

</details>

## Diagnose the result

If bfloat16 disagrees only with the original float32 data, check whether the represented-input oracle still passes before calling it a kernel bug. If timings are implausibly tiny, inspect synchronization and device placement. If a local speedup fails to improve the training step, reprofile communication and other unchanged work. Never replace missing target evidence with simulated-device labels.

## Carry forward

- Choose cases that challenge the contract
- Use two references for two different questions
- Make target timing a complete operation
- Compare a strong baseline with the same contract
- Relate local kernels to the distributed workload
- Accept, revise or decline the optimization

## Keep your evidence

Shape/dtype reference checks, separate precision gaps, guarded target benchmark and a bounded end-to-end speedup argument. Keep the environment, observed outputs and your explanation of the figure.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [Pallas grids and block specifications](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
- [Writing TPU kernels: layout and memory restrictions](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
- [TPU pipelining and emit_pipeline](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)
- [TPU interpretation is simulation, not target execution](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.InterpretParams.html)

