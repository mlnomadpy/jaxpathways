"""Tiling, memory, and pipelining: worked experiments and reference solutions. CPU checks."""

# Reuse the checked arithmetic and add a real pipeline
# Step 1 — Reuse the checked arithmetic and add a real pipeline: The outer kernel receives global-memory references and emits an...
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

# Compare synchronous and buffered pipeline semantics
# Step 2 — Compare synchronous and buffered pipeline semantics: Both schedules execute real pipeline semantics under CPU simulation.
# Construct and reshape `x` into the target tensor dimensions.
x=jnp.linspace(-1,1,17*257,dtype=jnp.float32).reshape(17,257)
# Compute `y` from `jnp.full_like(x,.25)`
y=jnp.full_like(x,.25)
# Convert `reference` to a host NumPy array for inspection or verification.
reference=2*np.asarray(x)+np.asarray(y)
# Run `pipelined_axpy` to compute `synchronous`.
synchronous=pipelined_axpy(x,y,no_pipelining=True)
# Run `pipelined_axpy` to compute `buffered`.
buffered=pipelined_axpy(x,y,no_pipelining=False)
# Compute `np.testing.assert_allclose(synchronous,reference,rtol` as `1e-6,atol=1e-6)`.
np.testing.assert_allclose(synchronous,reference,rtol=1e-6,atol=1e-6)
# Compute `np.testing.assert_allclose(buffered,reference,rtol` as `1e-6,atol=1e-6)`.
np.testing.assert_allclose(buffered,reference,rtol=1e-6,atol=1e-6)
# Execute `np.testing.assert_array_equal(synchronous,buffered)`
np.testing.assert_array_equal(synchronous,buffered)
# Print the observed values to compare against the expected result.
print("Actual backend:",jax.default_backend(),"simulated layout: TPU v5 lite")
# Print diagnostic summary of the computed outputs.
print("Synchronous/buffered endpoints:",float(buffered[0,0]),float(buffered[-1,-1]))

# Quantify tile tradeoffs before measuring hardware
# Step 3 — Quantify tile tradeoffs before measuring hardware: This footprint counts only the declared data buffers.
configurations=[(8,128),(16,128),(16,256)]
# Compute `metadata` from `[]`
metadata=[]
# Iterate over `(bm, bn)` to step through the computation:
for bm,bn in configurations:
    # Compute `gm` from `(17+bm-1)//bm`
    gm=(17+bm-1)//bm
    gn=(257+bn-1)//bn
    # Compute `programs` from `gm*gn`
    programs=gm*gn
    padded_elements=programs*bm*bn
    # Two input buffers and one output buffer, each double buffered, FP32.
    modeled_buffer_bytes=3*2*bm*bn*4
    # Combine or mask array elements to form ``.
    metadata.append((programs,padded_elements,modeled_buffer_bytes))
    # Run `pipelined_axpy` to compute `result`.
    result=pipelined_axpy(x,y,(bm,bn))
    # Compute `np.testing.assert_allclose(result,reference,rtol` as `1e-6,atol=1e-6)`.
    np.testing.assert_allclose(result,reference,rtol=1e-6,atol=1e-6)
# Print the observed values to compare against the expected result.
print("programs, padded elements, modeled data-buffer bytes:",metadata)

# Step 1 — Reuse the checked arithmetic and add a real pipeline: The outer kernel receives global-memory references and emits an...
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

# Step 2 — Compare synchronous and buffered pipeline semantics: Both schedules execute real pipeline semantics under CPU simulation.
# Construct and reshape `x` into the target tensor dimensions.
x=jnp.linspace(-1,1,17*257,dtype=jnp.float32).reshape(17,257)
# Compute `y` from `jnp.full_like(x,.25)`
y=jnp.full_like(x,.25)
# Convert `reference` to a host NumPy array for inspection or verification.
reference=2*np.asarray(x)+np.asarray(y)
# Run `pipelined_axpy` to compute `synchronous`.
synchronous=pipelined_axpy(x,y,no_pipelining=True)
# Run `pipelined_axpy` to compute `buffered`.
buffered=pipelined_axpy(x,y,no_pipelining=False)
# Compute `np.testing.assert_allclose(synchronous,reference,rtol` as `1e-6,atol=1e-6)`.
np.testing.assert_allclose(synchronous,reference,rtol=1e-6,atol=1e-6)
# Compute `np.testing.assert_allclose(buffered,reference,rtol` as `1e-6,atol=1e-6)`.
np.testing.assert_allclose(buffered,reference,rtol=1e-6,atol=1e-6)
# Execute `np.testing.assert_array_equal(synchronous,buffered)`
np.testing.assert_array_equal(synchronous,buffered)
# Print the observed values to compare against the expected result.
print("Actual backend:",jax.default_backend(),"simulated layout: TPU v5 lite")
# Print diagnostic summary of the computed outputs.
print("Synchronous/buffered endpoints:",float(buffered[0,0]),float(buffered[-1,-1]))

# Step 3 — Quantify tile tradeoffs before measuring hardware: This footprint counts only the declared data buffers.
configurations=[(8,128),(16,128),(16,256)]
# Compute `metadata` from `[]`
metadata=[]
# Iterate over `(bm, bn)` to step through the computation:
for bm,bn in configurations:
    # Compute `gm` from `(17+bm-1)//bm`
    gm=(17+bm-1)//bm
    gn=(257+bn-1)//bn
    # Compute `programs` from `gm*gn`
    programs=gm*gn
    padded_elements=programs*bm*bn
    # Two input buffers and one output buffer, each double buffered, FP32.
    modeled_buffer_bytes=3*2*bm*bn*4
    # Combine or mask array elements to form ``.
    metadata.append((programs,padded_elements,modeled_buffer_bytes))
    # Run `pipelined_axpy` to compute `result`.
    result=pipelined_axpy(x,y,(bm,bn))
    # Compute `np.testing.assert_allclose(result,reference,rtol` as `1e-6,atol=1e-6)`.
    np.testing.assert_allclose(result,reference,rtol=1e-6,atol=1e-6)
# Print the observed values to compare against the expected result.
print("programs, padded elements, modeled data-buffer bytes:",metadata)

# Figure data experiment
# Compute figure data for: Fewer tile programs can require more work and local storage
# Compute `labels` from `['8 x 128','16 x 128','16 x 256']`
labels=['8 x 128','16 x 128','16 x 256']
# Compute `visual_data` from `{'panels':[{'kind':'bar','labels':labels,'ylabel':'p...`
visual_data={'panels':[{'kind':'bar','labels':labels,'ylabel':'program count (configuration-derived)','series':[{'label':'grid programs','y':[row[0] for row in metadata]}]},{'kind':'bar','labels':labels,'ylabel':'modeled data buffers (KiB)','series':[{'label':'double-buffered inputs and output','y':[row[2]/1024 for row in metadata]}]}]}

# Experiment: Change input buffer count without changing the result
# Experiment — Change input buffer count without changing the result: More input buffering changes potential overlap and storage, not...
three=pipelined_axpy(x,y,buffers=3)
# Compute `np.testing.assert_allclose(three,reference,rtol` as `1e-6,atol=1e-6)`.
np.testing.assert_allclose(three,reference,rtol=1e-6,atol=1e-6)
# Execute `np.testing.assert_array_equal(three,buffered)`
np.testing.assert_array_equal(three,buffered)
# Print the observed values to compare against the expected result.
print("Three-buffer schedule matches the two-buffer result")

# Experiment: Check a one-tile pipeline
# Experiment — Check a one-tile pipeline: A one-tile pipeline has no next tile with which to overlap its...
# Construct and reshape `small_x` into the target tensor dimensions.
small_x=jnp.arange(8*128,dtype=jnp.float32).reshape(8,128)/100
# Construct `small_y` via `jnp.ones_like(small_x)`
small_y=jnp.ones_like(small_x)
# Run `pipelined_axpy` to compute `small`.
small=pipelined_axpy(small_x,small_y)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(small,2*np.asarray(small_x)+1,rtol=1e-6,atol=1e-6)
# Print the observed values to compare against the expected result.
print("Single-tile pipeline checked")

# Reference solution. Try the exercise before reading this.
# Exercise solution: Use a (9,129) random fixture and verify synchronous,...
rng=np.random.default_rng(23)
# Iterate over `dtype` to step through the computation:
for dtype in (jnp.float32,jnp.bfloat16):
    # Create device-backed JAX array `a`.
    a=jnp.asarray(rng.normal(size=(9,129)),dtype=dtype)
    # Create device-backed JAX array `b`.
    b=jnp.asarray(rng.normal(size=(9,129)),dtype=dtype)
    # Cast or evaluate `expected` in explicit floating-point precision.
    expected=(2*a.astype(jnp.float32)+b.astype(jnp.float32)).astype(dtype)
    # Loop over `(buffers, sync)` in `[(2, True), (2, False), (3, False)]`:
    for buffers,sync in [(2,True),(2,False),(3,False)]:
        # Run `pipelined_axpy` to compute `result`.
        result=pipelined_axpy(a,b,buffers=buffers,no_pipelining=sync)
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_array_equal(np.asarray(result),np.asarray(expected))
# Print the observed values to compare against the expected result.
print("Changed shape, dtype and schedules verified")

# Reference practice: Reject an unsupported pipeline tile
# Reject an unsupported pipeline tile (Practice): Different wrappers may expose different supported contracts.
rejected=False
# Run the boundary check and catch the expected exception:
try:pipelined_axpy(x,y,(3,5))
except ValueError:rejected=True
# Assert invariant `rejected` holds
assert rejected
# Print the observed values to compare against the expected result.
print("Unsupported pipeline tile rejected")

# Reference practice: Estimate the price of a third input buffer
# Estimate the price of a third input buffer (Challenge): A third slot for each input increases this limited...
# Iterate over `(bm, bn)` to step through the computation:
for bm,bn in configurations:
    # Compute `two` from `(2*2+2)*bm*bn*4`
    two=(2*2+2)*bm*bn*4
    three=(2*3+2)*bm*bn*4
    # Assert invariant `three*3==two*4` holds
    assert three*3==two*4
    # Print the observed values to compare against the expected result.
    print("Tile, two/three-input-slot modeled bytes:",(bm,bn),two,three)
print("PASS: kernels-03")
