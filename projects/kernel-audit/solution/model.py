"""Actual Pallas CPU interpretation/simulation and guarded TPU execution paths."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def validate_pair(x,y,block):
    if x.ndim!=2 or x.shape!=y.shape or min(x.shape)<1:
        raise ValueError("inputs must be nonempty matrices with identical shapes")
    if x.dtype!=y.dtype or x.dtype not in (jnp.float32,jnp.bfloat16):
        raise ValueError("matching float32 or bfloat16 inputs required")
    if len(block)!=2 or any(not isinstance(n,int) or n<1 for n in block):
        raise ValueError("two positive integer block dimensions required")

def pad_pair(x,y,block):
    validate_pair(x,y,block)
    m,n=x.shape;bm,bn=block
    padded=((m+bm-1)//bm*bm,(n+bn-1)//bn*bn)
    pads=((0,padded[0]-m),(0,padded[1]-n))
    return jnp.pad(x,pads),jnp.pad(y,pads),padded

def axpy_body(x_ref,y_ref,out_ref):
    result=2.0*x_ref[...].astype(jnp.float32)+y_ref[...].astype(jnp.float32)
    out_ref[...]=result.astype(out_ref.dtype)

def blocked_axpy(x,y,block=(2,4)):
    px,py,padded=pad_pair(x,y,block)
    bm,bn=block
    spec=pl.BlockSpec(block,lambda i,j:(i,j))
    out=pl.pallas_call(axpy_body,
        out_shape=jax.ShapeDtypeStruct(padded,x.dtype),
        grid=(padded[0]//bm,padded[1]//bn),
        in_specs=(spec,spec),out_specs=spec,interpret=True)(px,py)
    return out[:x.shape[0],:x.shape[1]]

def pipelined_axpy(x,y,block=(8,128),buffers=2,no_pipelining=False,mode="simulate"):
    px,py,padded=pad_pair(x,y,block)
    bm,bn=block
    if bm%8 or bn%128:
        raise ValueError("pipeline blocks must be multiples of (8,128)")
    if buffers not in (2,3):
        raise ValueError("this lab supports two or three buffers")
    if mode not in ("simulate","tpu"):
        raise ValueError("mode must explicitly be simulate or tpu")
    if mode=="tpu" and (jax.default_backend()!="tpu" or any(not isinstance(a,jax.core.Tracer) and any(d.platform!="tpu" for d in a.devices()) for a in (x,y))):
        raise RuntimeError("Real TPU inputs/backend required; simulation fallback is disabled")
    spec=pl.BlockSpec(block,lambda i,j:(i,j),pipeline_mode=pl.Buffered(buffer_count=buffers))
    output_spec=pl.BlockSpec(block,lambda i,j:(i,j),pipeline_mode=pl.Buffered(buffer_count=2))
    def outer(x_hbm,y_hbm,out_hbm):
        pltpu.emit_pipeline(axpy_body,grid=(padded[0]//bm,padded[1]//bn),
            in_specs=(spec,spec),out_specs=output_spec,
            no_pipelining=no_pipelining)(x_hbm,y_hbm,out_hbm)
    whole=pl.BlockSpec(memory_space=pl.ANY)
    interpretation=pltpu.InterpretParams(detect_races=True) if mode=="simulate" else False
    call=pl.pallas_call(outer,out_shape=jax.ShapeDtypeStruct(padded,x.dtype),
        in_specs=(whole,whole),out_specs=whole,interpret=interpretation)
    if mode=="simulate":
        # This describes a SIMULATED TPU layout, not the machine running the code.
        abstract=jax.sharding.AbstractMesh((1,),("simulated_device",),
            abstract_device=jax.sharding.AbstractDevice(device_kind="TPU v5 lite",num_cores=1))
        with jax.sharding.use_abstract_mesh(abstract):
            output=call(px,py)
    else:
        output=call(px,py)
    return output[:x.shape[0],:x.shape[1]]

def target_benchmark(candidate,baseline,args,repeats=20):
    if jax.default_backend()!="tpu" or any(d.platform!="tpu" for a in args for d in a.devices()):
        raise RuntimeError("TPU benchmark requires real TPU inputs; interpretation timings are not accepted")
    if repeats<5:
        raise ValueError("at least five repeated target measurements required")
    import time
    records={}
    for label,function in (("candidate",candidate),("baseline",baseline)):
        started=time.perf_counter()
        compiled=jax.jit(function).lower(*args).compile()
        compile_seconds=time.perf_counter()-started
        for _ in range(5):compiled(*args).block_until_ready()
        samples=[]
        for _ in range(repeats):
            started=time.perf_counter()
            compiled(*args).block_until_ready()
            samples.append((time.perf_counter()-started)*1000)
        records[label]={"compile_seconds":compile_seconds,"samples_ms":samples,
            "median_ms":float(np.median(samples)),"p90_ms":float(np.percentile(samples,90))}
    records["actual_backend"]=jax.default_backend()
    records["device_kind"]=jax.devices()[0].device_kind
    records["boundary"]="already-placed logical inputs; complete padded/cropped wrapper output; warm synchronized execution"
    return records

def fused_bias_relu(x,bias,block=(8,128),mode="interpret"):
    if x.ndim!=2 or bias.ndim!=1 or bias.shape[0]!=x.shape[1] or min(x.shape)<1:
        raise ValueError("x must be (M,N), bias must be (N,), and dimensions nonempty")
    if x.dtype!=bias.dtype or x.dtype not in (jnp.float32,jnp.bfloat16):
        raise ValueError("matching float32 or bfloat16 required")
    bm,bn=block
    if not isinstance(bm,int) or not isinstance(bn,int) or bm<1 or bn<1:
        raise ValueError("positive integer block sizes required")
    if mode not in ("interpret","tpu"):
        raise ValueError("mode must explicitly be interpret or tpu")
    if mode=="tpu":
        if jax.default_backend()!="tpu" or any(not isinstance(a,jax.core.Tracer) and any(d.platform!="tpu" for d in a.devices()) for a in (x,bias)):
            raise RuntimeError("Real TPU inputs/backend required; CPU fallback is disabled")
        if bm%8 or bn%128:
            raise ValueError("this TPU wrapper requires block multiples of (8,128)")
    m,n=x.shape;pm=(m+bm-1)//bm*bm;pn=(n+bn-1)//bn*bn
    px=jnp.pad(x,((0,pm-m),(0,pn-n)))
    pb=jnp.pad(bias,(0,pn-n))[None,:]
    def kernel(x_ref,bias_ref,out_ref):
        value=x_ref[...].astype(jnp.float32)+bias_ref[...].astype(jnp.float32)
        out_ref[...]=jnp.maximum(value,0).astype(out_ref.dtype)
    matrix_spec=pl.BlockSpec((bm,bn),lambda i,j:(i,j))
    bias_spec=pl.BlockSpec((1,bn),lambda i,j:(0,j))
    output=pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((pm,pn),x.dtype),
        grid=(pm//bm,pn//bn),in_specs=(matrix_spec,bias_spec),out_specs=matrix_spec,
        interpret=(mode=="interpret"),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel","parallel")))(px,pb)
    return output[:m,:n]
