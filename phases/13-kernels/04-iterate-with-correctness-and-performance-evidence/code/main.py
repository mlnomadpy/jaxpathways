"""Iterate with correctness and performance evidence: worked experiments and reference solutions. CPU checks."""

# Keep reference semantics, target mode and benchmark boundary together
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

# Audit a matrix of shapes and precisions
rng=np.random.default_rng(42)
correctness=[];precision_gaps=[]
for shape in [(1,1),(8,128),(9,129),(17,257)]:
    original_x=rng.normal(size=shape).astype(np.float32)
    original_y=rng.normal(size=shape).astype(np.float32)
    ideal=2*original_x+original_y
    row=[];gaps=[]
    for dtype in (jnp.float32,jnp.bfloat16):
        a=jnp.asarray(original_x,dtype=dtype);b=jnp.asarray(original_y,dtype=dtype)
        actual=pipelined_axpy(a,b)
        represented=(2*a.astype(jnp.float32)+b.astype(jnp.float32)).astype(dtype)
        np.testing.assert_array_equal(np.asarray(actual),np.asarray(represented))
        error=float(np.max(np.abs(np.asarray(actual,dtype=np.float32)-np.asarray(represented,dtype=np.float32))))
        gap=float(np.max(np.abs(np.asarray(actual,dtype=np.float32)-ideal)))
        row.append(error);gaps.append(gap)
    correctness.append(row);precision_gaps.append(gaps)
print("Kernel errors versus represented-input oracle:",correctness)
print("Gaps versus original float32 data:",precision_gaps)

# Check the evidence boundary before requesting a speed claim
if jax.default_backend()=="cpu":
    probe=jnp.ones((8,128),dtype=jnp.float32)
    rejected=False
    try:target_benchmark(lambda a,b:pipelined_axpy(a,b,mode="tpu"),lambda a,b:2*a+b,(probe,probe))
    except RuntimeError as error:
        rejected=True;print("Expected benchmark refusal:",error)
    assert rejected
print("CPU correctness audit complete; no TPU latency or speedup measured")

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

rng=np.random.default_rng(42)
correctness=[];precision_gaps=[]
for shape in [(1,1),(8,128),(9,129),(17,257)]:
    original_x=rng.normal(size=shape).astype(np.float32)
    original_y=rng.normal(size=shape).astype(np.float32)
    ideal=2*original_x+original_y
    row=[];gaps=[]
    for dtype in (jnp.float32,jnp.bfloat16):
        a=jnp.asarray(original_x,dtype=dtype);b=jnp.asarray(original_y,dtype=dtype)
        actual=pipelined_axpy(a,b)
        represented=(2*a.astype(jnp.float32)+b.astype(jnp.float32)).astype(dtype)
        np.testing.assert_array_equal(np.asarray(actual),np.asarray(represented))
        error=float(np.max(np.abs(np.asarray(actual,dtype=np.float32)-np.asarray(represented,dtype=np.float32))))
        gap=float(np.max(np.abs(np.asarray(actual,dtype=np.float32)-ideal)))
        row.append(error);gaps.append(gap)
    correctness.append(row);precision_gaps.append(gaps)
print("Kernel errors versus represented-input oracle:",correctness)
print("Gaps versus original float32 data:",precision_gaps)

if jax.default_backend()=="cpu":
    probe=jnp.ones((8,128),dtype=jnp.float32)
    rejected=False
    try:target_benchmark(lambda a,b:pipelined_axpy(a,b,mode="tpu"),lambda a,b:2*a+b,(probe,probe))
    except RuntimeError as error:
        rejected=True;print("Expected benchmark refusal:",error)
    assert rejected
print("CPU correctness audit complete; no TPU latency or speedup measured")

# Figure data experiment
labels=['1 x 1','8 x 128','9 x 129','17 x 257']
visual_data={'kind':'bar','labels':labels,'ylabel':'max gap versus original float32 values','series':[{'label':'float32','y':[r[0] for r in precision_gaps]},{'label':'bfloat16','y':[r[1] for r in precision_gaps]}]}

# Experiment: Preserve correctness under a joint row permutation
a=jnp.asarray(rng.normal(size=(9,129)),dtype=jnp.float32)
b=jnp.asarray(rng.normal(size=(9,129)),dtype=jnp.float32)
order=jnp.array([8,0,7,1,6,2,5,3,4])
inverse=jnp.argsort(order)
original=pipelined_axpy(a,b)
permuted=pipelined_axpy(a[order],b[order])[inverse]
np.testing.assert_array_equal(np.asarray(permuted),np.asarray(original))
print("Joint row permutation preserves output identity")

# Experiment: Make a precision-policy counterexample
first=jnp.array([1.00390625],dtype=jnp.float32)
second=jnp.array([0.00390625],dtype=jnp.float32)
early=(first.astype(jnp.bfloat16).astype(jnp.float32)+second).astype(jnp.bfloat16)
late=(first+second).astype(jnp.bfloat16)
assert float(early[0])!=float(late[0])
print("Early/final-only rounding:",float(early[0]),float(late[0]))

# Reference solution. Try the exercise before reading this.
a=jnp.asarray(np.random.default_rng(8).normal(size=(15,255)),dtype=jnp.float32)
b=jnp.asarray(np.random.default_rng(9).normal(size=(15,255)),dtype=jnp.float32)
expected=2*np.asarray(a)+np.asarray(b)
for block in [(8,128),(16,256)]:
    for buffers in (2,3):
        result=pipelined_axpy(a,b,block,buffers=buffers)
        np.testing.assert_allclose(result,expected,rtol=1e-6,atol=1e-6)
        gm=(15+block[0]-1)//block[0];gn=(255+block[1]-1)//block[1]
        print("Configuration, padded elements:",block,buffers,gm*gn*block[0]*block[1])

# Reference practice: Reject unsupported dtype mixing
for first,second in [(jnp.ones((8,128),jnp.float32),jnp.ones((8,128),jnp.bfloat16)),(jnp.ones((8,128),jnp.int32),jnp.ones((8,128),jnp.int32))]:
    rejected=False
    try:pipelined_axpy(first,second)
    except ValueError:rejected=True
    assert rejected
print("Mixed and integer dtypes rejected")

# Reference practice: Bound the end-to-end opportunity
fraction=.3
local_speedup=2.0
end_to_end=1/((1-fraction)+fraction/local_speedup)
limit=1/(1-fraction)
np.testing.assert_allclose(end_to_end,1.1764705882352942)
np.testing.assert_allclose(limit,1.4285714285714286)
print("Modeled twofold-local speedup and zero-cost limit:",end_to_end,limit)
print("PASS: kernels-04")
