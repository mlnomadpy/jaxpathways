# Tiling, memory, and pipelining

Phase 13: Pallas kernels · about 120 minutes · CPU

## What you will be able to do

- Execute actual emit_pipeline semantics under explicit CPU TPU-layout simulation.
- Compare synchronous and buffered schedules against an independent oracle.
- Explain tile count, padding and modeled local-buffer footprint separately.
- Distinguish correctness of scheduling from measured transfer/compute overlap.

## The problem

A larger tile reduces the number of kernel invocations, but it also consumes more local memory and may compute more padded values. Can we make those tradeoffs concrete and execute a real buffered pipeline without claiming that CPU simulation measures TPU overlap?

## The idea

We keep the arithmetic $Z=2X+Y$ fixed and change how data travels through the kernel. An outer Pallas call exposes global references; emit_pipeline copies block windows into local buffers, invokes the arithmetic, and writes results back. The reference run simulates these TPU operations on CPU with race checking enabled.

## Keep one numerical operation while changing data movement

The arithmetic body remains unchanged from the first lesson. The outer kernel receives entire buffers in a global-memory space, and the inner pipeline sees block-local references. This separation lets us change the movement schedule without changing the mathematical result. If a new schedule changes values, it is a correctness regression rather than a performance tradeoff.

## What buffering is trying to overlap

A pipeline can start bringing the next tile into a different buffer while the current tile is being processed. Reusing a buffer too early risks overwriting data still being read; consuming it before its transfer completes risks stale or uninitialized data. The emitter manages copies and waits. We compare two versus three input slots while keeping two output slots: installed JAX 0.9.2 rejects output buffer counts above two. A third input slot consumes more local memory and may or may not hide additional latency.

## Use the appropriate interpreter for memory operations

Ordinary interpret=True is enough for the first lesson’s arithmetic grid, but does not supply all TPU layout information required by this installed pipeline API. Here pltpu.InterpretParams simulates memory spaces and synchronization; an AbstractMesh specifies a simulated TPU v5 lite layout. The actual backend remains CPU. This is JAX 0.9.2 API usage, and its signature should be rechecked when upgrading. The simulated device label must never appear as the actual device in a performance receipt.

## Use synchronous execution as a debugging comparison

no_pipelining=True requests synchronous copies through the same pipeline. Agreement with the buffered result helps isolate scheduling errors while preserving arithmetic and windows. The interpreter enables race detection, but passing these cases does not prove every target interleaving, layout or compiler path. If a TPU interpretation exception occurs, the documented interpreter state-reset procedure is required before reusing that mode in the same process; fresh-process checks avoid carrying failed state into later evidence.

## Count the cost of a tile choice

For a logical $(17,257)$ matrix, blocks $(8,128)$, $(16,128)$ and $(16,256)$ use $9$, $6$ and $4$ programs. Their padded work grows from $9216$ to $12288$ to $16384$ elements. The corresponding modeled double-buffered data storage grows from $24$ to $48$ to $96$ KiB for two inputs and one output in float32. Fewer programs therefore does not automatically mean less work or lower memory use.

$$
B_{\mathrm{model}}=(2n_{\mathrm{input\ buffers}}+2)\,B_mB_n\,\mathrm{bytes\ per\ element}
$$

## Only target measurements can rank pipeline speed

CPU simulation may execute callbacks, bookkeeping and race tracking that have no relation to TPU throughput. Do not time it and call the result a kernel benchmark. The project’s target runner can execute the actual synchronous or buffered TPU branch without fallback; its correctness checks and synchronized baseline comparison must pass before interpreting latency. No such target performance receipt is available here.

## Reuse the checked arithmetic and add a real pipeline

Create main.py for the first block, then append each block in order in your course CPU environment.

```python
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
```

The outer kernel receives global-memory references and emits an actual TPU pipeline. BlockSpecs describe its VMEM windows; the pipeline manages buffer copies and waits. CPU execution uses the TPU interpreter with explicitly simulated layout metadata.

## Compare synchronous and buffered pipeline semantics

Create main.py for the first block, then append each block in order in your course CPU environment.

```python
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
```

Both schedules execute real pipeline semantics under CPU simulation. Equality checks copies, writes and tails; it cannot establish that transfers overlap with computation on a real TPU.

## Quantify tile tradeoffs before measuring hardware

Create main.py for the first block, then append each block in order in your course CPU environment.

```python
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
```

This footprint counts only the declared data buffers. It excludes runtime bookkeeping, semaphores, compiler scratch, register allocation and target-specific layout padding; it is not a measured device-memory statistic.

## Run the example

```python
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
```

Expected: Synchronous and buffered CPU simulations agree with NumPy and each other; endpoints -1.75 and 2.25. Tile program counts [9,6,4], padded elements [9216,12288,16384], modeled data buffers [24576,49152,98304] bytes.

## Fewer tile programs can require more work and local storage

**Predict:** Will the tile configuration with the fewest programs also minimize padding and buffer footprint?

![Fewer tile programs can require more work and local storage](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The first panel shows logical kernel-program counts for three tile shapes on the same $(17,257)$ matrix: $9$, $6$ and $4$. The second panel shows modeled double-buffered data storage: $24$, $48$ and $96$ KiB. These are exact configuration-derived counts, not measured timing or device-memory counters.

The largest tile has fewer programs but four times the modeled data-buffer footprint of the smallest. Its padded element count also rises from $9216$ to $16384$, even though the logical matrix still contains only $4369$ elements.

### Connect it to the computation

Every pictured configuration also runs through the actual Pallas pipeline in CPU simulation and matches the same NumPy result. The figure explains the resource tradeoff that remains after correctness is established.

The footprint omits semaphores, bookkeeping, layout padding, registers and compiler scratch. It cannot establish total VMEM use or speedup. A real TPU trace and synchronized timing are needed to learn whether a larger tile improves reuse enough to offset its costs.

```python
labels=['8 x 128','16 x 128','16 x 256']
visual_data={'panels':[{'kind':'bar','labels':labels,'ylabel':'program count (configuration-derived)','series':[{'label':'grid programs','y':[row[0] for row in metadata]}]},{'kind':'bar','labels':labels,'ylabel':'modeled data buffers (KiB)','series':[{'label':'double-buffered inputs and output','y':[row[2]/1024 for row in metadata]}]}]}
```

## Recorded reference execution

CPU run: 2026-10-06T01:25:48.610526+00:00. JAX 0.9.2.

```text
Actual backend: cpu simulated layout: TPU v5 lite
Synchronous/buffered endpoints: -1.75 2.25
programs, padded elements, modeled data-buffer bytes: [(9, 9216, 24576), (6, 12288, 49152), (4, 16384, 98304)]
Actual backend: cpu simulated layout: TPU v5 lite
Synchronous/buffered endpoints: -1.75 2.25
programs, padded elements, modeled data-buffer bytes: [(9, 9216, 24576), (6, 12288, 49152), (4, 16384, 98304)]
Three-buffer schedule matches the two-buffer result
Single-tile pipeline checked
Changed shape, dtype and schedules verified
Unsupported pipeline tile rejected
Tile, two/three-input-slot modeled bytes: (8, 128) 24576 32768
Tile, two/three-input-slot modeled bytes: (16, 128) 49152 65536
Tile, two/three-input-slot modeled bytes: (16, 256) 98304 131072
PASS: kernels-03

```

## Change input buffer count without changing the result

**Predict before running:** Can adding a third slot to each input change the numerical answer?

```python
three=pipelined_axpy(x,y,buffers=3)
np.testing.assert_allclose(three,reference,rtol=1e-6,atol=1e-6)
np.testing.assert_array_equal(three,buffered)
print("Three-buffer schedule matches the two-buffer result")
```

**Expected:** Two and three input slots produce the same logical result; outputs retain two slots.

More input buffering changes potential overlap and storage, not arithmetic. This installed emitter only supports up to two output slots. A speed benefit remains a target measurement question.

## Check a one-tile pipeline

**Predict before running:** What work can be overlapped when there is only one tile?

```python
small_x=jnp.arange(8*128,dtype=jnp.float32).reshape(8,128)/100
small_y=jnp.ones_like(small_x)
small=pipelined_axpy(small_x,small_y)
np.testing.assert_allclose(small,2*np.asarray(small_x)+1,rtol=1e-6,atol=1e-6)
print("Single-tile pipeline checked")
```

**Expected:** The one-tile result matches the oracle.

A one-tile pipeline has no next tile with which to overlap its initial fetch. Prologue and drain overhead can dominate small workloads; correctness does not imply a benefit.

## Make it yours

Use a (9,129) random fixture and verify synchronous, double-input-buffered and triple-input-buffered pipelines for both float32 and bfloat16 represented-input contracts. Keep output buffering at two slots.

<details><summary>Reference solution</summary>

```python
rng=np.random.default_rng(23)
for dtype in (jnp.float32,jnp.bfloat16):
    a=jnp.asarray(rng.normal(size=(9,129)),dtype=dtype)
    b=jnp.asarray(rng.normal(size=(9,129)),dtype=dtype)
    expected=(2*a.astype(jnp.float32)+b.astype(jnp.float32)).astype(dtype)
    for buffers,sync in [(2,True),(2,False),(3,False)]:
        result=pipelined_axpy(a,b,buffers=buffers,no_pipelining=sync)
        np.testing.assert_array_equal(np.asarray(result),np.asarray(expected))
print("Changed shape, dtype and schedules verified")
```

</details>

## Reject an unsupported pipeline tile

**Practice**

Try a (3,5) block and verify that the pipeline wrapper refuses it rather than using the permissive arithmetic interpreter as a hidden fallback.

<details><summary>Hint</summary>

The pipeline lab deliberately uses a conservative TPU-compatible tile family.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
rejected=False
try:pipelined_axpy(x,y,(3,5))
except ValueError:rejected=True
assert rejected
print("Unsupported pipeline tile rejected")
```

Different wrappers may expose different supported contracts. A narrower explicit contract is safer than implying the generic interpreter established target compatibility.

</details>

## Estimate the price of a third input buffer

**Challenge**

Compute the declared data-buffer footprint for two and three input slots at each tile size, with two output slots in both cases. State at least two omitted memory costs.

<details><summary>Hint</summary>

Count two input arrays and one output array separately because their buffer counts differ.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
for bm,bn in configurations:
    two=(2*2+2)*bm*bn*4;three=(2*3+2)*bm*bn*4
    assert three*3==two*4
    print("Tile, two/three-input-slot modeled bytes:",(bm,bn),two,three)
```

A third slot for each input increases this limited data-buffer model by one third. Output buffering stays at two slots. The estimate excludes registers, semaphores, compiler scratch and layout effects.

</details>

## Check your understanding

The three-buffer simulation is correct and has fewer grid programs after a tile change. Can you conclude the TPU version is faster?

1. Yes; program count determines runtime.
2. Yes; simulation latency is the TPU latency.
3. No; padding, local memory and actual target overlap must be measured with synchronized target execution.

<details><summary>Answer and explanation</summary>

No; padding, local memory and actual target overlap must be measured with synchronized target execution.

Simulation establishes tested semantics. Buffering and tile changes affect several costs, and only actual target measurements can rank performance.

</details>

## Diagnose the result

If the buffered result differs from synchronous execution, keep arithmetic and tile geometry fixed while inspecting copies, ownership and synchronization. If the interpreter cannot infer TPU layout on CPU, choose the documented TPU simulation API and label the metadata as simulated. If a larger tile uses fewer programs but more padded work, neither count alone ranks its performance.

## Carry forward

- Keep one numerical operation while changing data movement
- What buffering is trying to overlap
- Use the appropriate interpreter for memory operations
- Use synchronous execution as a debugging comparison
- Count the cost of a tile choice
- Only target measurements can rank pipeline speed

## Keep your evidence

Synchronous/buffered CPU simulation agreement, two/three input slots, unchanged output buffering, correctness across tiles and modeled footprint counts. Keep the environment, observed outputs and your explanation of the figure.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [Pallas grids and block specifications](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
- [Writing TPU kernels: layout and memory restrictions](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
- [TPU pipelining and emit_pipeline](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)
- [TPU interpretation is simulation, not target execution](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.InterpretParams.html)

