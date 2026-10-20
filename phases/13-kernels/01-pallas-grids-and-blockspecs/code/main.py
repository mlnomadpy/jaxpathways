"""Pallas grids and BlockSpecs: worked experiments and reference solutions. CPU checks."""

# Write a block-local kernel and a boundary-aware wrapper
# Step 1 — Write a block-local kernel and a boundary-aware wrapper: The kernel writes its output Ref and returns nothing.
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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

def pad_pair(x,y,block):
    validate_pair(x,y,block)
    m,n=x.shape
    bm,bn=block
    padded=((m+bm-1)//bm*bm,(n+bn-1)//bn*bn)
    # Combine or mask array elements to form `pads`.
    pads=((0,padded[0]-m),(0,padded[1]-n))
    return jnp.pad(x,pads),jnp.pad(y,pads),padded

def axpy_body(x_ref,y_ref,out_ref):
    result=2.0*x_ref[...].astype(jnp.float32)+y_ref[...].astype(jnp.float32)
    out_ref[...]=result.astype(out_ref.dtype)

def blocked_axpy(x,y,block=(2,4)):
    # Combine or mask array elements to form `(px, py, padded)`.
    px,py,padded=pad_pair(x,y,block)
    bm,bn=block
    # Invoke custom Pallas kernel or tile specification (`spec`).
    spec=pl.BlockSpec(block,lambda i,j:(i,j))
    # Combine or mask array elements to form `out`.
    out=pl.pallas_call(axpy_body,
        out_shape=jax.ShapeDtypeStruct(padded,x.dtype),
        grid=(padded[0]//bm,padded[1]//bn),
        in_specs=(spec,spec),out_specs=spec,interpret=True)(px,py)
    return out[:x.shape[0],:x.shape[1]]

# Check every logical element against NumPy
# Step 2 — Check every logical element against NumPy: The last row and column belong to partial logical tiles.
# Construct and reshape `x` into the target tensor dimensions.
x=jnp.arange(55,dtype=jnp.float32).reshape(5,11)/10-2
y=jnp.full_like(x,0.25)
actual=blocked_axpy(x,y)
# Convert `reference` to a host NumPy array for inspection or verification.
reference=2*np.asarray(x)+np.asarray(y)
# Check tensor shape invariant: `actual.shape==(5,11)`
assert actual.shape==(5,11)
np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-6)
assert float(actual[-1,-1])==float(reference[-1,-1])
print("Shape and boundary:",actual.shape,float(actual[0,0]),float(actual[-1,-1]))

# Derive the ownership map independently
# Step 3 — Derive the ownership map independently: Block coordinates are multiplied by the block shape by Pallas.
rows,cols=np.indices((5,11))
owner=(rows//2)*3+(cols//4)
assert owner[0,0]==0 and owner[4,10]==8
logical=5*11
padded=6*12
print("Grid:",(3,3),"logical/padded elements:",logical,padded)
print("Padding fraction:",(padded-logical)/padded)

# Step 1 — Write a block-local kernel and a boundary-aware wrapper: The kernel writes its output Ref and returns nothing.
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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

def pad_pair(x,y,block):
    validate_pair(x,y,block)
    m,n=x.shape
    bm,bn=block
    padded=((m+bm-1)//bm*bm,(n+bn-1)//bn*bn)
    # Combine or mask array elements to form `pads`.
    pads=((0,padded[0]-m),(0,padded[1]-n))
    return jnp.pad(x,pads),jnp.pad(y,pads),padded

def axpy_body(x_ref,y_ref,out_ref):
    result=2.0*x_ref[...].astype(jnp.float32)+y_ref[...].astype(jnp.float32)
    out_ref[...]=result.astype(out_ref.dtype)

def blocked_axpy(x,y,block=(2,4)):
    # Combine or mask array elements to form `(px, py, padded)`.
    px,py,padded=pad_pair(x,y,block)
    bm,bn=block
    # Invoke custom Pallas kernel or tile specification (`spec`).
    spec=pl.BlockSpec(block,lambda i,j:(i,j))
    # Combine or mask array elements to form `out`.
    out=pl.pallas_call(axpy_body,
        out_shape=jax.ShapeDtypeStruct(padded,x.dtype),
        grid=(padded[0]//bm,padded[1]//bn),
        in_specs=(spec,spec),out_specs=spec,interpret=True)(px,py)
    return out[:x.shape[0],:x.shape[1]]

# Step 2 — Check every logical element against NumPy: The last row and column belong to partial logical tiles.
# Construct and reshape `x` into the target tensor dimensions.
x=jnp.arange(55,dtype=jnp.float32).reshape(5,11)/10-2
y=jnp.full_like(x,0.25)
actual=blocked_axpy(x,y)
# Convert `reference` to a host NumPy array for inspection or verification.
reference=2*np.asarray(x)+np.asarray(y)
# Check tensor shape invariant: `actual.shape==(5,11)`
assert actual.shape==(5,11)
np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-6)
assert float(actual[-1,-1])==float(reference[-1,-1])
print("Shape and boundary:",actual.shape,float(actual[0,0]),float(actual[-1,-1]))

# Step 3 — Derive the ownership map independently: Block coordinates are multiplied by the block shape by Pallas.
rows,cols=np.indices((5,11))
owner=(rows//2)*3+(cols//4)
assert owner[0,0]==0 and owner[4,10]==8
logical=5*11
padded=6*12
print("Grid:",(3,3),"logical/padded elements:",logical,padded)
print("Padding fraction:",(padded-logical)/padded)

# Figure data experiment
# Compute figure data for: Nine programs cover a matrix with partial boundary tiles
visual_data={'kind':'heatmap','values':owner.tolist(),'unit':'program ID (categorical label)','rows':[str(i) for i in range(5)],'columns':[str(i) for i in range(11)],'xlabel':'logical column','ylabel':'logical row'}

# Experiment: Change block geometry without changing values
# Experiment — Change block geometry without changing values: Grid geometry is an implementation choice for this independent...
changed=blocked_axpy(x,y,(3,5))
np.testing.assert_allclose(changed,reference,rtol=1e-6,atol=1e-6)
print("Changed grid:",(2,3),"same numerical result")

# Experiment: Expose the floor-division omission
# Experiment — Expose the floor-division omission: This diagnostic models the ownership bug without reading...
covered=np.zeros((5,11),dtype=bool)
for i in range(5//2):
    for j in range(11//4):
        covered[i*2:(i+1)*2,j*4:(j+1)*4]=True
assert int(covered.sum())==32
assert not covered[-1,-1]
print("Floor-grid missing logical cells:",int((~covered).sum()))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Test a singleton row and a changed nondivisible matrix with negative...
for shape in [(1,7),(7,9)]:
    # Create device-backed JAX array `a`.
    a=jnp.asarray(np.random.default_rng(11).normal(size=shape),dtype=jnp.float32)
    # Create device-backed JAX array `b`.
    b=jnp.asarray(np.random.default_rng(12).normal(size=shape),dtype=jnp.float32)
    for block in [(2,4),(3,5)]:
        result=blocked_axpy(a,b,block)
        # Check tensor shape invariant: `result.shape==shape`
        assert result.shape==shape
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_allclose(result,2*np.asarray(a)+np.asarray(b),rtol=1e-6,atol=1e-6)
print("Singleton and changed tails verified")

# Reference practice: Reject a broadcast that changes the contract
# Reject a broadcast that changes the contract (Practice): The operation promises paired matrices.
rejected=False
# Run the boundary check and catch the expected exception:
try:blocked_axpy(x,jnp.ones((5,1),dtype=jnp.float32))
except ValueError:rejected=True
assert rejected
print("Mismatched global shape rejected")

# Reference practice: Measure padding overhead without claiming runtime
# Measure padding overhead without claiming runtime (Challenge): Program count and padded work expose different costs.
for shape in [(5,11),(33,129)]:
    counts=[]
    for bm,bn in [(2,4),(8,128),(16,256)]:
        gm=(shape[0]+bm-1)//bm
        gn=(shape[1]+bn-1)//bn
        counts.append((bm,bn,gm*gn,gm*gn*bm*bn))
    assert all(padded>=shape[0]*shape[1] for _,_,_,padded in counts)
    print("Shape, block/program/padded counts:",shape,counts)
print("PASS: kernels-01")
