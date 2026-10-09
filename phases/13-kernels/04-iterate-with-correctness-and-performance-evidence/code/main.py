"""Iterate with correctness and performance evidence: worked experiments and reference solutions. CPU checks."""

# Keep reference semantics, target mode and benchmark boundary together
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
    # Evaluate `(bm, bn)` from the current inputs and state.
    m,n=x.shape;bm,bn=block
    # Evaluate `padded` from the current inputs and state.
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
    # Evaluate `(bm, bn)` from the current inputs and state.
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
    # Evaluate `(bm, bn)` from the current inputs and state.
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
    # Evaluate `records` from the current inputs and state.
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
        # Evaluate `samples` from the current inputs and state.
        samples=[]
        # Repeat the update loop over `range(repeats)` steps:
        for _ in range(repeats):
            # Record execution timing or profiler trace in `started`.
            started=time.perf_counter()
            # Synchronize host execution until asynchronous device computation completes.
            compiled(*args).block_until_ready()
            # Record execution timing or profiler trace in ``.
            samples.append((time.perf_counter()-started)*1000)
        # Evaluate `records[label]` from the current inputs and state.
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

# Audit a matrix of shapes and precisions
# Step 2 — Audit a matrix of shapes and precisions: A zero kernel error means the implementation honored the declared...
rng=np.random.default_rng(42)
# Evaluate `correctness` from the current inputs and state.
# Evaluate `precision_gaps` from the current inputs and state.
correctness=[];precision_gaps=[]
# Iterate over `shape` to step through the computation:
for shape in [(1,1),(8,128),(9,129),(17,257)]:
    # Cast or evaluate `original_x` in explicit floating-point precision.
    original_x=rng.normal(size=shape).astype(np.float32)
    # Cast or evaluate `original_y` in explicit floating-point precision.
    original_y=rng.normal(size=shape).astype(np.float32)
    # Evaluate `ideal` from the current inputs and state.
    ideal=2*original_x+original_y
    # Evaluate `row` from the current inputs and state.
    # Evaluate `gaps` from the current inputs and state.
    row=[];gaps=[]
    # Loop over `dtype` in `(jnp.float32, jnp.bfloat16)`:
    for dtype in (jnp.float32,jnp.bfloat16):
        # Create device-backed JAX array `a`.
        # Create device-backed JAX array `b`.
        a=jnp.asarray(original_x,dtype=dtype);b=jnp.asarray(original_y,dtype=dtype)
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
        row.append(error);gaps.append(gap)
    # Append the current step result to `correctness`.
    # Append the current step result to `correctness`.
    correctness.append(row);precision_gaps.append(gaps)
# Print the observed values to compare against the expected result.
print("Kernel errors versus represented-input oracle:",correctness)
# Print diagnostic summary of the computed outputs.
print("Gaps versus original float32 data:",precision_gaps)

# Check the evidence boundary before requesting a speed claim
# Step 3 — Check the evidence boundary before requesting a speed claim: The runnable target entry point is the project target_tpu.py script.
if jax.default_backend()=="cpu":
    probe=jnp.ones((8,128),dtype=jnp.float32)
    rejected=False
    try:target_benchmark(lambda a,b:pipelined_axpy(a,b,mode="tpu"),lambda a,b:2*a+b,(probe,probe))
    except RuntimeError as error:
        rejected=True;print("Expected benchmark refusal:",error)
    assert rejected
# Print the observed values to compare against the expected result.
print("CPU correctness audit complete; no TPU latency or speedup measured")

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
    # Evaluate `(bm, bn)` from the current inputs and state.
    m,n=x.shape;bm,bn=block
    # Evaluate `padded` from the current inputs and state.
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
    # Evaluate `(bm, bn)` from the current inputs and state.
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
    # Evaluate `(bm, bn)` from the current inputs and state.
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
    # Evaluate `records` from the current inputs and state.
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
        # Evaluate `samples` from the current inputs and state.
        samples=[]
        # Repeat the update loop over `range(repeats)` steps:
        for _ in range(repeats):
            # Record execution timing or profiler trace in `started`.
            started=time.perf_counter()
            # Synchronize host execution until asynchronous device computation completes.
            compiled(*args).block_until_ready()
            # Record execution timing or profiler trace in ``.
            samples.append((time.perf_counter()-started)*1000)
        # Evaluate `records[label]` from the current inputs and state.
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
# Evaluate `precision_gaps` from the current inputs and state.
correctness=[];precision_gaps=[]
# Iterate over `shape` to step through the computation:
for shape in [(1,1),(8,128),(9,129),(17,257)]:
    # Cast or evaluate `original_x` in explicit floating-point precision.
    original_x=rng.normal(size=shape).astype(np.float32)
    # Cast or evaluate `original_y` in explicit floating-point precision.
    original_y=rng.normal(size=shape).astype(np.float32)
    # Evaluate `ideal` from the current inputs and state.
    ideal=2*original_x+original_y
    # Evaluate `row` from the current inputs and state.
    # Evaluate `gaps` from the current inputs and state.
    row=[];gaps=[]
    # Loop over `dtype` in `(jnp.float32, jnp.bfloat16)`:
    for dtype in (jnp.float32,jnp.bfloat16):
        # Create device-backed JAX array `a`.
        # Create device-backed JAX array `b`.
        a=jnp.asarray(original_x,dtype=dtype);b=jnp.asarray(original_y,dtype=dtype)
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
        row.append(error);gaps.append(gap)
    # Append the current step result to `correctness`.
    # Append the current step result to `correctness`.
    correctness.append(row);precision_gaps.append(gaps)
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
        rejected=True;print("Expected benchmark refusal:",error)
    assert rejected
# Print the observed values to compare against the expected result.
print("CPU correctness audit complete; no TPU latency or speedup measured")

# Figure data experiment
# Compute figure data for: Precision gaps remain after kernel correctness passes
# Evaluate `labels` from the current inputs and state.
labels=['1 x 1','8 x 128','9 x 129','17 x 257']
# Evaluate `visual_data` from the current inputs and state.
visual_data={'kind':'bar','labels':labels,'ylabel':'max gap versus original float32 values','series':[{'label':'float32','y':[r[0] for r in precision_gaps]},{'label':'bfloat16','y':[r[1] for r in precision_gaps]}]}

# Experiment: Preserve correctness under a joint row permutation
# Experiment — Preserve correctness under a joint row permutation: This operation has no coupling between rows.
a=jnp.asarray(rng.normal(size=(9,129)),dtype=jnp.float32)
# Create device-backed JAX array `b`.
b=jnp.asarray(rng.normal(size=(9,129)),dtype=jnp.float32)
# Initialize array `order` with explicit values and shape.
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

# Experiment: Make a precision-policy counterexample
# Experiment — Make a precision-policy counterexample: The order of representation changes is part of the numerical...
# Initialize array `first` with explicit values and shape.
first=jnp.array([1.00390625],dtype=jnp.float32)
# Initialize array `second` with explicit values and shape.
second=jnp.array([0.00390625],dtype=jnp.float32)
# Cast or evaluate `early` in explicit floating-point precision.
early=(first.astype(jnp.bfloat16).astype(jnp.float32)+second).astype(jnp.bfloat16)
# Cast or evaluate `late` in explicit floating-point precision.
late=(first+second).astype(jnp.bfloat16)
# Verify contract: `float(early[0]) != float(late[0])`.
assert float(early[0])!=float(late[0])
# Print the observed values to compare against the expected result.
print("Early/final-only rounding:",float(early[0]),float(late[0]))

# Reference solution. Try the exercise before reading this.
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
        # Verify that computed values match the expected reference within numerical tolerance.
        np.testing.assert_allclose(result,expected,rtol=1e-6,atol=1e-6)
        # Evaluate `gm` from the current inputs and state.
        # Evaluate `gn` from the current inputs and state.
        gm=(15+block[0]-1)//block[0];gn=(255+block[1]-1)//block[1]
        # Print diagnostic summary of the computed outputs.
        print("Configuration, padded elements:",block,buffers,gm*gn*block[0]*block[1])

# Reference practice: Reject unsupported dtype mixing
# Reject unsupported dtype mixing (Practice): Supported dtype boundaries are a correctness decision.
# Iterate over `(first, second)` to step through the computation:
for first,second in [(jnp.ones((8,128),jnp.float32),jnp.ones((8,128),jnp.bfloat16)),(jnp.ones((8,128),jnp.int32),jnp.ones((8,128),jnp.int32))]:
    # Evaluate `rejected` from the current inputs and state.
    rejected=False
    # Run the boundary check and catch the expected exception:
    try:pipelined_axpy(first,second)
    except ValueError:rejected=True
    # Verify contract: `rejected`.
    assert rejected
# Print the observed values to compare against the expected result.
print("Mixed and integer dtypes rejected")

# Reference practice: Bound the end-to-end opportunity
# Bound the end-to-end opportunity (Challenge): These are conditional analytic estimates, not measured speedups.
fraction=.3
# Evaluate `local_speedup` from the current inputs and state.
local_speedup=2.0
# Evaluate `end_to_end` from the current inputs and state.
end_to_end=1/((1-fraction)+fraction/local_speedup)
# Evaluate `limit` from the current inputs and state.
limit=1/(1-fraction)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(end_to_end,1.1764705882352942)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(limit,1.4285714285714286)
# Print the observed values to compare against the expected result.
print("Modeled twofold-local speedup and zero-cost limit:",end_to_end,limit)
print("PASS: kernels-04")
