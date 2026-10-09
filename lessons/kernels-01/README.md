# Pallas grids and BlockSpecs

Phase 13: Pallas kernels · about 95 minutes · CPU

## What you will be able to do

- Implement an actual Pallas Ref-writing kernel with grid and BlockSpecs.
- Derive block coordinates and distinguish them from element offsets.
- Handle nondivisible matrix tails explicitly and verify full output coverage.
- Separate CPU interpretation from accelerator execution evidence.

## The problem

A kernel can produce the right answer on a divisible matrix and silently miss the last row on a real workload. How do grid coordinates, block ownership and array boundaries fit together? We will implement an actual Pallas operation and make its tail handling visible.

## The idea

A Pallas grid assigns work to programs, and BlockSpec maps those programs to array regions. Boundary tiles make the distinction between logical data and padded storage important. Verify ownership before optimizing arithmetic.

## Map every logical element to an owning program

The lesson's five-by-eleven matrix is covered by nine program IDs. Those IDs are categories, not measured magnitudes. Tile outlines and a discrete legend show which program owns each region without implying that a higher ID performs more work.

At an edge, the physical tile extends beyond the logical matrix. This implementation pads the inputs before the kernel, making each block access valid, then crops the result back to the logical shape. A padded slot is storage for the computation, not an extra observation or logical output.

Enumerate coordinates independently and verify coverage of every valid element. Then test dimensions that are not multiples of the tile size; perfectly divisible fixtures can conceal boundary mistakes.

### Program coordinates and partial boundary tiles

**Predict:** Why test an awkward shape in addition to a tile-aligned shape?

![Program coordinates and partial boundary tiles](../../phases/13-kernels/01-pallas-grids-and-blockspecs/outputs/mechanism.svg)

*Conceptual / analytic teaching diagram; not a recorded benchmark.*

Two-by-four tiles cover a five-by-eleven logical matrix. Each cell lists a program ID and valid row/column range. Bottom and right tiles are partial; accesses outside the logical range need boundary treatment. Program IDs are categories, matching the corrected discrete ownership heatmap. Here explicit padding makes whole-block accesses valid; cropping removes artificial output positions.

### Pause and reason

Why test an awkward shape in addition to a tile-aligned shape?

<details><summary>Compare your reasoning</summary>

It exercises partial boundary tiles and guards. A tile-aligned case never asks the implementation to distinguish padded positions from valid data.

</details>

## Write the simplest useful contract

Two nonempty matrices have the same shape and dtype; the result has that shape and dtype. The kernel computes in float32 and casts once at its output, supporting float32 and bfloat16 inputs. This explicit precision policy becomes important later. A function returning an array is not the kernel body contract: the body reads input Refs and writes every element of its output Ref.

## A grid coordinate is not an element coordinate

For a $(5,11)$ matrix and $(2,4)$ blocks, the grid is $(3,3)$. Invocation $(i,j)$ owns rows starting at $2i$ and columns starting at $4j$. The BlockSpec index map returns block coordinates $(i,j)$; Pallas applies the block-size scaling. Confusing this with element offsets can produce skipped regions or out-of-bounds windows.

$$
G_m=\left\lceil\frac{M}{B_m}\right\rceil,\quad G_n=\left\lceil\frac{N}{B_n}\right\rceil
$$

## Make boundary behavior explicit

We pad both inputs to $(6,12)$, launch complete blocks and crop the output to $(5,11)$. This avoids relying on implicit out-of-bounds reads or uninitialized output cells. There are $55$ logical elements and $72$ padded elements, so $17$ elements represent extra work. Padding with zeros is harmless here because each output depends only on corresponding input elements. A reduction, softmax or stencil would require a separate padding argument.

## Check ownership and arithmetic separately

The ownership map comes from integer row/column arithmetic, independent of the kernel result. The value reference comes from NumPy. Checking only a mean could hide a missing or swapped tile; compare every element and explicitly inspect the last row and column. A different block size must change work partitioning while preserving the logical result.

## Interpretation proves bounded semantics, not speed

The call sets interpret=True, which runs actual Pallas semantics through a CPU-compatible execution path. This verifies the grid, Ref operations and numerical results for the fixture. It does not validate TPU layout constraints, target lowering, memory overlap or accelerator throughput. The next lesson introduces a separate TPU-only path that refuses to substitute interpretation when target hardware is absent.

## Write a block-local kernel and a boundary-aware wrapper

Create main.py for the first block, then append each block in order in your course CPU environment.

```python
# Step 1 — Write a block-local kernel and a boundary-aware wrapper: The kernel writes its output Ref and returns nothing.
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
```

The kernel writes its output Ref and returns nothing. The wrapper validates and pads the global matrices, defines ownership with BlockSpecs, launches actual Pallas interpretation and crops to the original shape.

## Check every logical element against NumPy

Create main.py for the first block, then append each block in order in your course CPU environment.

```python
# Step 2 — Check every logical element against NumPy: The last row and column belong to partial logical tiles.
# Construct and reshape `x` into the target tensor dimensions.
x=jnp.arange(55,dtype=jnp.float32).reshape(5,11)/10-2
# Initialize array `y` with explicit values and shape.
y=jnp.full_like(x,0.25)
# Run `blocked_axpy` to compute `actual`.
actual=blocked_axpy(x,y)
# Convert `reference` to a host NumPy array for inspection or verification.
reference=2*np.asarray(x)+np.asarray(y)
# Verify that the output tensor shape matches our prediction.
assert actual.shape==(5,11)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-6)
# Verify contract: `float(actual[-1, -1]) == float(reference[-1, -1])`.
assert float(actual[-1,-1])==float(reference[-1,-1])
# Print the observed values to compare against the expected result.
print("Shape and boundary:",actual.shape,float(actual[0,0]),float(actual[-1,-1]))
```

The last row and column belong to partial logical tiles. Checking them explicitly catches floor-divided grids that omit the tail.

## Derive the ownership map independently

Create main.py for the first block, then append each block in order in your course CPU environment.

```python
# Step 3 — Derive the ownership map independently: Block coordinates are multiplied by the block shape by Pallas.
rows,cols=np.indices((5,11))
# Evaluate `owner` from the current inputs and state.
owner=(rows//2)*3+(cols//4)
# Verify contract: `owner[0, 0] == 0 and owner[4, 10] == 8`.
assert owner[0,0]==0 and owner[4,10]==8
# Evaluate `logical` from the current inputs and state.
# Evaluate `padded` from the current inputs and state.
logical=5*11;padded=6*12
# Print the observed values to compare against the expected result.
print("Grid:",(3,3),"logical/padded elements:",logical,padded)
# Print diagnostic summary of the computed outputs.
print("Padding fraction:",(padded-logical)/padded)
```

Block coordinates are multiplied by the block shape by Pallas. An index map returning (i,j) selects the corresponding block; multiplying those indices again would skip data.

## Run the example

```python
# Step 1 — Write a block-local kernel and a boundary-aware wrapper: The kernel writes its output Ref and returns nothing.
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

# Step 2 — Check every logical element against NumPy: The last row and column belong to partial logical tiles.
# Construct and reshape `x` into the target tensor dimensions.
x=jnp.arange(55,dtype=jnp.float32).reshape(5,11)/10-2
# Initialize array `y` with explicit values and shape.
y=jnp.full_like(x,0.25)
# Run `blocked_axpy` to compute `actual`.
actual=blocked_axpy(x,y)
# Convert `reference` to a host NumPy array for inspection or verification.
reference=2*np.asarray(x)+np.asarray(y)
# Verify that the output tensor shape matches our prediction.
assert actual.shape==(5,11)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-6)
# Verify contract: `float(actual[-1, -1]) == float(reference[-1, -1])`.
assert float(actual[-1,-1])==float(reference[-1,-1])
# Print the observed values to compare against the expected result.
print("Shape and boundary:",actual.shape,float(actual[0,0]),float(actual[-1,-1]))

# Step 3 — Derive the ownership map independently: Block coordinates are multiplied by the block shape by Pallas.
rows,cols=np.indices((5,11))
# Evaluate `owner` from the current inputs and state.
owner=(rows//2)*3+(cols//4)
# Verify contract: `owner[0, 0] == 0 and owner[4, 10] == 8`.
assert owner[0,0]==0 and owner[4,10]==8
# Evaluate `logical` from the current inputs and state.
# Evaluate `padded` from the current inputs and state.
logical=5*11;padded=6*12
# Print the observed values to compare against the expected result.
print("Grid:",(3,3),"logical/padded elements:",logical,padded)
# Print diagnostic summary of the computed outputs.
print("Padding fraction:",(padded-logical)/padded)
```

Expected: Logical shape (5,11); endpoints -3.75 and approximately 7.05. Grid (3,3); 55 logical versus 72 padded elements.

## Nine programs cover a matrix with partial boundary tiles

**Predict:** Which program owns the last logical element, and why does the grid need a third row and column?

![Nine programs cover a matrix with partial boundary tiles](../../phases/13-kernels/01-pallas-grids-and-blockspecs/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

Rows and columns are logical matrix coordinates. Each color identifies the program that owns that cell, using the program-ID labels on the colorbar under $(2,4)$ blocks. Program IDs are labels, not execution time or device IDs. Programs zero through two cover the first two rows; programs six through eight cover the last logical row.

The bottom-right cell belongs to program $8$. Its logical region has only one row and three columns, although the launched block has two rows and four columns. The padded cells are intentionally outside this cropped ownership display.

### Connect it to the computation

The independent ownership arithmetic agrees with a $(3,3)$ grid over a padded $(6,12)$ buffer. The numerical Pallas result is separately compared element by element with NumPy, including the bottom-right value of about $7.05$.

Color differences do not imply parallel execution or different speeds. This CPU interpretation figure explains mapping and boundaries; TPU scheduling and layout constraints need their own target-specific evidence.

```python
# Compute figure data for: Nine programs cover a matrix with partial boundary tiles
# Evaluate `visual_data` from the current inputs and state.
visual_data={'kind':'heatmap','values':owner.tolist(),'unit':'program ID (categorical label)','rows':[str(i) for i in range(5)],'columns':[str(i) for i in range(11)],'xlabel':'logical column','ylabel':'logical row'}
```

## Recorded reference execution

CPU run: 2026-10-08T14:05:18.403150+00:00. JAX 0.9.2.

```text
Shape and boundary: (5, 11) -3.75 7.050000190734863
Grid: (3, 3) logical/padded elements: 55 72
Padding fraction: 0.2361111111111111
Shape and boundary: (5, 11) -3.75 7.050000190734863
Grid: (3, 3) logical/padded elements: 55 72
Padding fraction: 0.2361111111111111
Changed grid: (2, 3) same numerical result
Floor-grid missing logical cells: 23
Singleton and changed tails verified
Mismatched global shape rejected
Shape, block/program/padded counts: (5, 11) [(2, 4, 9, 72), (8, 128, 1, 1024), (16, 256, 1, 4096)]
Shape, block/program/padded counts: (33, 129) [(2, 4, 561, 4488), (8, 128, 10, 10240), (16, 256, 3, 12288)]
PASS: kernels-01

```

## Change block geometry without changing values

**Predict before running:** Will blocks of $(3,5)$ change the logical answer?

```python
# Experiment — Change block geometry without changing values: Grid geometry is an implementation choice for this independent...
changed=blocked_axpy(x,y,(3,5))
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(changed,reference,rtol=1e-6,atol=1e-6)
# Print the observed values to compare against the expected result.
print("Changed grid:",(2,3),"same numerical result")
```

**Expected:** The grid becomes (2,3); the logical output is unchanged.

Grid geometry is an implementation choice for this independent elementwise operation. It must not alter the mathematical contract.

## Expose the floor-division omission

**Predict before running:** How many logical cells would a grid built with floor division cover?

```python
# Experiment — Expose the floor-division omission: This diagnostic models the ownership bug without reading...
covered=np.zeros((5,11),dtype=bool)
# Iterate over `i` to step through the computation:
for i in range(5//2):
    # Loop over `j` in `range(11 // 4)`:
    for j in range(11//4):
        # Evaluate `covered[i * 2:(i + 1) * 2, j * 4:(j + 1) * 4]` from the current inputs and state.
        covered[i*2:(i+1)*2,j*4:(j+1)*4]=True
# Verify contract: `int(covered.sum()) == 32`.
assert int(covered.sum())==32
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not covered[-1,-1]
# Print the observed values to compare against the expected result.
print("Floor-grid missing logical cells:",int((~covered).sum()))
```

**Expected:** A floor-divided grid misses 23 of the 55 logical cells.

This diagnostic models the ownership bug without reading uninitialized kernel memory. Ceiling division plus an explicit tail policy repairs it.

## Make it yours

Test a singleton row and a changed nondivisible matrix with negative values. Verify full NumPy agreement and exact output shapes for two block choices.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Iterate over `shape` to step through the computation:
2. Create device-backed JAX array `a`.
3. Create device-backed JAX array `b`.
4. Iterate over `block` to step through the computation:
5. Run `blocked_axpy` to compute `result`.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Test a singleton row and a changed nondivisible matrix with negative...
# Iterate over `shape` to step through the computation:
for shape in [(1,7),(7,9)]:
    # Create device-backed JAX array `a`.
    a = jnp.asarray(...)  # TODO: compute a
    # Create device-backed JAX array `b`.
    b = jnp.asarray(...)  # TODO: compute b
    # Iterate over `block` to step through the computation:
    for block in [(2,4),(3,5)]:
        # Run `blocked_axpy` to compute `result`.
        result = blocked_axpy(...)  # TODO: compute result
        # Verify that the output tensor shape matches our prediction.
        assert result.shape  # TODO: complete assertion check
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_allclose(result,2*np.asarray(a)+np.asarray(b),rtol=1e-6,atol=1e-6)
# Print the observed values to compare against the expected result.
print("Singleton and changed tails verified")
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Test a singleton row and a changed nondivisible matrix with negative...
# Iterate over `shape` to step through the computation:
for shape in [(1,7),(7,9)]:
    # Create device-backed JAX array `a`.
    a=jnp.asarray(np.random.default_rng(11).normal(size=shape),dtype=jnp.float32)
    # Create device-backed JAX array `b`.
    b=jnp.asarray(np.random.default_rng(12).normal(size=shape),dtype=jnp.float32)
    # Iterate over `block` to step through the computation:
    for block in [(2,4),(3,5)]:
        # Run `blocked_axpy` to compute `result`.
        result=blocked_axpy(a,b,block)
        # Verify that the output tensor shape matches our prediction.
        assert result.shape==shape
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_allclose(result,2*np.asarray(a)+np.asarray(b),rtol=1e-6,atol=1e-6)
# Print the observed values to compare against the expected result.
print("Singleton and changed tails verified")
```

</details>

## Reject a broadcast that changes the contract

**Practice**

Pass a column-shaped second input and confirm the wrapper rejects it before launching the kernel. Explain why silent broadcasting would hide an input-contract error.

<details><summary>Hint</summary>

Require identical matrix shapes, not merely broadcast-compatible shapes.

</details>

### How to write: Reject a broadcast that changes the contract — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.zeros / jnp.ones(shape, dtype=...)` — Allocates a tensor of the given `shape` initialized with constants.

**Step-by-step implementation plan:**
1. Run the boundary check and catch the expected exception:
2. Verify contract: `rejected`.
3. Print the observed values to compare against the expected result.

**Starter code scaffold (fill in the TODOs):**

```python
# Reject a broadcast that changes the contract (Practice): The operation promises paired matrices.
rejected = ...  # TODO: compute rejected
# Run the boundary check and catch the expected exception:
try:blocked_axpy(x,jnp.ones((5,1),dtype=jnp.float32))
except ValueError:rejected=True
# Verify contract: `rejected`.
assert rejected  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print("Mismatched global shape rejected")
```

<details><summary>Reference solution and reasoning</summary>

```python
# Reject a broadcast that changes the contract (Practice): The operation promises paired matrices.
rejected=False
# Run the boundary check and catch the expected exception:
try:blocked_axpy(x,jnp.ones((5,1),dtype=jnp.float32))
except ValueError:rejected=True
# Verify contract: `rejected`.
assert rejected
# Print the observed values to compare against the expected result.
print("Mismatched global shape rejected")
```

The operation promises paired matrices. A separate row/column-bias kernel can define an intentional broadcast contract.

</details>

## Measure padding overhead without claiming runtime

**Challenge**

For shapes (5,11) and (33,129), compare logical elements, padded elements and program counts for three block choices. Explain why fewer programs may still mean more work.

<details><summary>Hint</summary>

Use integer ceiling division; do not time interpretation as accelerator performance.

</details>

### How to write: Measure padding overhead without claiming runtime — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `runtime(...)` — Call `runtime` with your updated parameters or inputs from this lesson's workspace.
- `counts.append(...)` — Call `counts.append` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Iterate over `shape` to step through the computation:
2. Evaluate `counts` from the current inputs and state.
3. Iterate over `(bm, bn)` to step through the computation:
4. Evaluate `gm` from the current inputs and state.
5. Evaluate `gn` from the current inputs and state.

**Starter code scaffold (fill in the TODOs):**

```python
# Measure padding overhead without claiming runtime (Challenge): Program count and padded work expose different costs.
# Iterate over `shape` to step through the computation:
for shape in [(5,11),(33,129)]:
    # Evaluate `counts` from the current inputs and state.
    counts = ...  # TODO: compute counts
    # Iterate over `(bm, bn)` to step through the computation:
    for bm,bn in [(2,4),(8,128),(16,256)]:
        # Evaluate `gm` from the current inputs and state.
        # Evaluate `gn` from the current inputs and state.
        gm = ...  # TODO: compute gm
        # Append the current step result to `counts`.
        counts.append((bm,bn,gm*gn,gm*gn*bm*bn))
    # Verify contract: `all((padded >= shape[0] * shape[1] for _, _, _, padded in counts))`.
    assert all(padded  # TODO: complete assertion check
    # Print the observed values to compare against the expected result.
    print("Shape, block/program/padded counts:",shape,counts)
```

<details><summary>Reference solution and reasoning</summary>

```python
# Measure padding overhead without claiming runtime (Challenge): Program count and padded work expose different costs.
# Iterate over `shape` to step through the computation:
for shape in [(5,11),(33,129)]:
    # Evaluate `counts` from the current inputs and state.
    counts=[]
    # Iterate over `(bm, bn)` to step through the computation:
    for bm,bn in [(2,4),(8,128),(16,256)]:
        # Evaluate `gm` from the current inputs and state.
        # Evaluate `gn` from the current inputs and state.
        gm=(shape[0]+bm-1)//bm;gn=(shape[1]+bn-1)//bn
        # Append the current step result to `counts`.
        counts.append((bm,bn,gm*gn,gm*gn*bm*bn))
    # Verify contract: `all((padded >= shape[0] * shape[1] for _, _, _, padded in counts))`.
    assert all(padded>=shape[0]*shape[1] for _,_,_,padded in counts)
    # Print the observed values to compare against the expected result.
    print("Shape, block/program/padded counts:",shape,counts)
```

Program count and padded work expose different costs. Actual target measurements are still needed to rank configurations.

</details>

## Check your understanding

For block shape (2,4), what does a BlockSpec index map returning (i,j) select?

1. A block whose element origin is (2i,4j).
2. One scalar element at (i,j).
3. A block whose origin must be multiplied by block sizes again inside the index map.

<details><summary>Answer and explanation</summary>

A block whose element origin is (2i,4j).

BlockSpec coordinates select blocks. Pallas scales them by the block dimensions; an extra multiplication in the map would skip windows.

</details>

## Diagnose the result

If only the boundary is wrong, inspect ceiling division, padding and cropping before editing arithmetic. If blocks skip input regions, check whether the index map returns block coordinates or incorrectly pre-multiplied offsets. If a mean looks right but full comparison fails, inspect tile permutation or duplicate ownership.

## Carry forward

- Write the simplest useful contract
- A grid coordinate is not an element coordinate
- Make boundary behavior explicit
- Check ownership and arithmetic separately
- Interpretation proves bounded semantics, not speed

## Keep your evidence

Independent full-array comparisons, ownership map, tail fixtures and a floor-grid omission diagnosis. Keep the environment, observed outputs and your explanation of the figure.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [Pallas grids and block specifications](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
- [Writing TPU kernels: layout and memory restrictions](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
- [TPU pipelining and emit_pipeline](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)
- [TPU interpretation is simulation, not target execution](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.InterpretParams.html)

