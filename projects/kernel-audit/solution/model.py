"""Actual Pallas CPU interpretation/simulation and guarded TPU execution paths."""
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
    # Compute `m,n` as `x.shape`.
    m,n=x.shape
    bm,bn=block
    # Evaluate the compound expression for `padded`.
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
    # Compute `bm,bn` as `block`.
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
    # Compute `bm,bn` as `block`.
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
    # Construct dictionary `records` with the structured fields for this stage.
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
        # Initialize list `samples` for the stage values.
        samples=[]
        # Repeat the update loop over `range(repeats)` steps:
        for _ in range(repeats):
            # Record execution timing or profiler trace in `started`.
            started=time.perf_counter()
            # Synchronize host execution until asynchronous device computation completes.
            compiled(*args).block_until_ready()
            # Record execution timing or profiler trace in ``.
            samples.append((time.perf_counter()-started)*1000)
        # Construct dictionary `records[label]` with the structured fields for this stage.
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

# Function `fused_bias_relu(x, bias, block, mode)` implementing this stage's computation:
def fused_bias_relu(x,bias,block=(8,128),mode="interpret"):
    # Guard input contract (`x.ndim != 2 or bias.ndim != 1 or bias.shape[0] != x.shape[1] or (min(x.shape) < 1)`) and fail fast if violated.
    if x.ndim!=2 or bias.ndim!=1 or bias.shape[0]!=x.shape[1] or min(x.shape)<1:
        raise ValueError("x must be (M,N), bias must be (N,), and dimensions nonempty")
    # Guard input contract (`x.dtype != bias.dtype or x.dtype not in (jnp.float32, jnp.bfloat16)`) and fail fast if violated.
    if x.dtype!=bias.dtype or x.dtype not in (jnp.float32,jnp.bfloat16):
        raise ValueError("matching float32 or bfloat16 required")
    # Compute `bm,bn` as `block`.
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
    # Compute `m,n` as `x.shape`.
    m,n=x.shape
    pm=(m+bm-1)//bm*bm
    pn=(n+bn-1)//bn*bn
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
