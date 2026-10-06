"""Tiling, memory, and pipelining: worked experiments and reference solutions. CPU checks."""

# Reuse the checked arithmetic and add a real pipeline
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

# Compare synchronous and buffered pipeline semantics
x=jnp.linspace(-1,1,17*257,dtype=jnp.float32).reshape(17,257)
y=jnp.full_like(x,.25)
reference=2*np.asarray(x)+np.asarray(y)
synchronous=pipelined_axpy(x,y,no_pipelining=True)
buffered=pipelined_axpy(x,y,no_pipelining=False)
np.testing.assert_allclose(synchronous,reference,rtol=1e-6,atol=1e-6)
np.testing.assert_allclose(buffered,reference,rtol=1e-6,atol=1e-6)
np.testing.assert_array_equal(synchronous,buffered)
print("Actual backend:",jax.default_backend(),"simulated layout: TPU v5 lite")
print("Synchronous/buffered endpoints:",float(buffered[0,0]),float(buffered[-1,-1]))

# Quantify tile tradeoffs before measuring hardware
configurations=[(8,128),(16,128),(16,256)]
metadata=[]
for bm,bn in configurations:
    gm=(17+bm-1)//bm;gn=(257+bn-1)//bn
    programs=gm*gn;padded_elements=programs*bm*bn
    # Two input buffers and one output buffer, each double buffered, FP32.
    modeled_buffer_bytes=3*2*bm*bn*4
    metadata.append((programs,padded_elements,modeled_buffer_bytes))
    result=pipelined_axpy(x,y,(bm,bn))
    np.testing.assert_allclose(result,reference,rtol=1e-6,atol=1e-6)
print("programs, padded elements, modeled data-buffer bytes:",metadata)

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

x=jnp.linspace(-1,1,17*257,dtype=jnp.float32).reshape(17,257)
y=jnp.full_like(x,.25)
reference=2*np.asarray(x)+np.asarray(y)
synchronous=pipelined_axpy(x,y,no_pipelining=True)
buffered=pipelined_axpy(x,y,no_pipelining=False)
np.testing.assert_allclose(synchronous,reference,rtol=1e-6,atol=1e-6)
np.testing.assert_allclose(buffered,reference,rtol=1e-6,atol=1e-6)
np.testing.assert_array_equal(synchronous,buffered)
print("Actual backend:",jax.default_backend(),"simulated layout: TPU v5 lite")
print("Synchronous/buffered endpoints:",float(buffered[0,0]),float(buffered[-1,-1]))

configurations=[(8,128),(16,128),(16,256)]
metadata=[]
for bm,bn in configurations:
    gm=(17+bm-1)//bm;gn=(257+bn-1)//bn
    programs=gm*gn;padded_elements=programs*bm*bn
    # Two input buffers and one output buffer, each double buffered, FP32.
    modeled_buffer_bytes=3*2*bm*bn*4
    metadata.append((programs,padded_elements,modeled_buffer_bytes))
    result=pipelined_axpy(x,y,(bm,bn))
    np.testing.assert_allclose(result,reference,rtol=1e-6,atol=1e-6)
print("programs, padded elements, modeled data-buffer bytes:",metadata)

# Figure data experiment
labels=['8 x 128','16 x 128','16 x 256']
visual_data={'panels':[{'kind':'bar','labels':labels,'ylabel':'program count (configuration-derived)','series':[{'label':'grid programs','y':[row[0] for row in metadata]}]},{'kind':'bar','labels':labels,'ylabel':'modeled data buffers (KiB)','series':[{'label':'double-buffered inputs and output','y':[row[2]/1024 for row in metadata]}]}]}

# Experiment: Change input buffer count without changing the result
three=pipelined_axpy(x,y,buffers=3)
np.testing.assert_allclose(three,reference,rtol=1e-6,atol=1e-6)
np.testing.assert_array_equal(three,buffered)
print("Three-buffer schedule matches the two-buffer result")

# Experiment: Check a one-tile pipeline
small_x=jnp.arange(8*128,dtype=jnp.float32).reshape(8,128)/100
small_y=jnp.ones_like(small_x)
small=pipelined_axpy(small_x,small_y)
np.testing.assert_allclose(small,2*np.asarray(small_x)+1,rtol=1e-6,atol=1e-6)
print("Single-tile pipeline checked")

# Reference solution. Try the exercise before reading this.
rng=np.random.default_rng(23)
for dtype in (jnp.float32,jnp.bfloat16):
    a=jnp.asarray(rng.normal(size=(9,129)),dtype=dtype)
    b=jnp.asarray(rng.normal(size=(9,129)),dtype=dtype)
    expected=(2*a.astype(jnp.float32)+b.astype(jnp.float32)).astype(dtype)
    for buffers,sync in [(2,True),(2,False),(3,False)]:
        result=pipelined_axpy(a,b,buffers=buffers,no_pipelining=sync)
        np.testing.assert_array_equal(np.asarray(result),np.asarray(expected))
print("Changed shape, dtype and schedules verified")

# Reference practice: Reject an unsupported pipeline tile
rejected=False
try:pipelined_axpy(x,y,(3,5))
except ValueError:rejected=True
assert rejected
print("Unsupported pipeline tile rejected")

# Reference practice: Estimate the price of a third input buffer
for bm,bn in configurations:
    two=(2*2+2)*bm*bn*4;three=(2*3+2)*bm*bn*4
    assert three*3==two*4
    print("Tile, two/three-input-slot modeled bytes:",(bm,bn),two,three)
print("PASS: kernels-03")
