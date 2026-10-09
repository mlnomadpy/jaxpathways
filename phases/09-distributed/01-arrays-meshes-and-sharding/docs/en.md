# Arrays, meshes, and sharding

Phase 09: Distributed training · about 80 minutes · 4 logical CPU devices

## What you will be able to do

- Translate a PartitionSpec into local shard shapes.
- Inspect shard indices and verify values.
- Explain communication needs of a global reduction.
- Repair uneven batch statistics with padding and masks.

## The problem

We can split examples across devices, split features across devices, or keep a full copy on each one. Let’s make those choices visible on four logical CPU devices. You’ll inspect the pieces and see why some reductions must combine results. No cluster is needed, and these observations do not measure accelerator performance.

## The idea

Sharding places one global array across a named mesh. Global shape, local shard shape and replication describe different views of the same computation. Make ownership explicit before reasoning about communication.

## Compare the global array with each local piece

A four-by-four array on two devices can be split into two-by-four row shards or four-by-two column shards. Both layouts contain the same global values. Their local operations and communication requirements differ.

Replication gives each participant the full value instead of a disjoint portion. Use explicit labels so duplicated ownership cannot be mistaken for partitioning.

Connect each PartitionSpec entry to its array axis and named mesh axis. Inspect actual addressable shards and compare reconstructed values with a global reference. Logical CPU devices validate this placement exercise without reproducing a multi-host network.

### Pause and reason

Must changing row sharding to column sharding change the answer?

<details><summary>Compare your reasoning</summary>

No. A correct compatible computation preserves the intended global result. Storage and communication can change even when the result agrees.

</details>

## Read the mapping from left to right

Start with eight rows and four columns. Our one-dimensional mesh contains four devices along data. `P("data",None)` partitions eight rows into four groups of two and preserves four features per shard. `P(None,"data")` leaves rows intact and assigns one of four columns to each device. The same global shape supports both arrangements.

None means an array dimension is not partitioned by this specification, not that the values are missing. `P()` replicates the whole array. A device mesh axis may not be reused across multiple array dimensions within one PartitionSpec. More complex two-dimensional meshes need distinct axis names and a reason for assigning each dimension.

```text
Array (8,4) → P(data,None) → 4 pieces (2,4)
Array (8,4) → P(None,data) → 4 pieces (8,1)
Array (8,4) → P() → 4 full copies (8,4)
```

## Inspect indices rather than guessing ordering

addressable_shards lists pieces this process can access. Each piece carries a device, global index and data. For row partitioning the indices select successive two-row slices; for column partitioning they select individual columns. Compare host[shard.index] with shard.data. This catches a mismatch between the intended assignment and the actual data.

All four devices are addressable here because we have one process. In a multi-process program, the global device set may include devices another process owns. `np.asarray` on a global array that is not fully addressable is not a general multi-host gathering recipe. This lesson deliberately exercises only the single-process case.

## A reduction has a communication requirement

For rows $0$ through $31$ reshaped to $(8,4)$, the first column is $0$,$4$,$8$,$12$,$16$,$20$,$24$,$28$ and sums to $112$. Each row shard can sum its own two rows, but no one shard initially sees all eight. Producing a replicated global column sum requires combining contributions across the data axis.

We use a jitted ordinary sum and declare replicated output. JAX compiles the placement-aware global operation and arranges necessary communication. The source-level need to combine partial sums is a reasoning argument; exact collective lowering is compiler/version dependent and would require lowered code or profiler evidence to characterize. This CPU lesson does not estimate accelerator communication latency.

```text
partial sums device 0: [4,6,8,10]
partial sums device 1: [20,22,24,26]
partial sums device 2: [36,38,40,42]
partial sums device 3: [52,54,56,58]
combine → [112,120,128,136]
```

## Choose placement for the operation you will perform and run it on a TPU mesh

Elementwise functions naturally preserve local independence. Reductions across a partitioned dimension and matrix multiplications can require communication. Choosing columns instead of rows changes which values are locally available; it does not alter the mathematical answer.

`device_put` from a host NumPy array places that array according to the requested sharding. Moving an already placed JAX array to a different sharding is a resharding operation. Host reconstruction is useful for our tiny verification case but should not be inserted into a real training loop. The experiments compare two layouts for correctness, not speed. Save global shapes, specs and shard tables as the evidence needed to reason about a future workload.

To run this exact mesh and `NamedSharding` contract on a real 4-chip TPU VM (for example `v5litepod-4` or `v6e-4`, provisioned via `lesson-welcome-03.html` and `tpu-gcp.html`), remove the `jax_platforms=cpu` override so `jax.devices()` discovers the four physical TPU chips. On a multi-host TPU slice (`v5litepod-8` or larger), call `jax.distributed.initialize()` at process startup and launch across all workers with `--worker=all`.

**Run the 4-device sharding check on a 4-chip TPU VM (single host)**

```bash
gcloud compute tpus tpu-vm scp exercises/distributed-01.py $TPU_NAME:~/jax-tpu-lab/ --zone=$ZONE
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE \
  --command="sed 's/jax.config.update(\"jax_platforms\", \"cpu\")/# use default TPU backend/; s/jax.devices(\"cpu\")/jax.devices()[:4]/' ~/jax-tpu-lab/distributed-01.py > ~/jax-tpu-lab/distributed-01-tpu.py && JAX_PLATFORMS=tpu ~/jax-tpu-lab/.venv/bin/python ~/jax-tpu-lab/distributed-01-tpu.py"
```

**Expected:** Places the (8, 4) array across four TPU chips, verifies row-shard (2, 4) and column-shard (8, 1) local shapes, and checks the replicated column sum [112, 120, 128, 136].

## Start a fresh CPU runtime

Create main.py in your course workspace. Paste this block first. If using a notebook, restart its kernel before running any cell.

```python
# Step 1 — Start a fresh CPU runtime: Configuration happens before devices() or array creation...
# Import jax for this computation.
import jax
# Run this file in a fresh process; in a notebook restart the kernel first.
jax.config.update("jax_platforms", "cpu")
# Update state in place with the new values.
jax.config.update("jax_num_cpu_devices", 4)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
# Query the active JAX devices into `devices`.
devices = jax.devices("cpu")
# Guard input contract (`len(devices) != 4`) and fail fast if violated.
if len(devices) != 4:
    raise RuntimeError("Expected four CPU devices. Restart the notebook kernel, then Run all; do not run an earlier JAX cell first.")
# Configure multi-device placement / sharding specification (`mesh`).
mesh = Mesh(np.array(devices), ("data",))
# Configure multi-device placement / sharding specification (`rows`).
rows = NamedSharding(mesh, P("data", None))
# Configure multi-device placement / sharding specification (`replicated`).
replicated = NamedSharding(mesh, P())
```

Configuration happens before `devices()` or array creation initializes a backend. This file explicitly chooses CPU even on a GPU machine.

## Place and compute

Append this block in the same file. Predict the shapes and values before running.

```python
# Step 2 — Place and compute: Mesh axis names describe placement.
# Construct and reshape `host` into the target tensor dimensions.
host = np.arange(32, dtype=np.float32).reshape(8,4)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host, rows)
# Configure multi-device placement / sharding specification (`columns`).
columns = NamedSharding(mesh, P(None, "data"))
# Place `x_columns` explicitly onto the target JAX device.
x_columns = jax.device_put(host, columns)
# Wrap with `jax.jit` (`sum_rows`) so XLA traces and compiles the function.
sum_rows = jax.jit(lambda a:a.sum(axis=0), in_shardings=rows, out_shardings=replicated)(x)
# Synchronize host execution until asynchronous device computation completes.
sum_rows.block_until_ready()
```

Mesh axis names describe placement. They are separate from the numerical array dimensions.

## Inspect and verify

Append the checks, save the file and run python main.py with the setup lesson environment.

```python
# Step 3 — Inspect and verify: Assertions compare against host calculations or hand-derived...
np.testing.assert_array_equal(np.asarray(sum_rows), np.array([112,120,128,136],dtype=np.float32))
# Iterate over `shard` to step through the computation:
for shard in x.addressable_shards:
    # Verify that the output tensor shape matches our prediction.
    assert shard.data.shape == (2,4)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), host[shard.index])
# Iterate over `shard` to step through the computation:
for shard in x_columns.addressable_shards:
    # Verify that the output tensor shape matches our prediction.
    assert shard.data.shape == (8,1)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), host[shard.index])
# Print the observed values to compare against the expected result.
print("Row-shard shapes:", [s.data.shape for s in x.addressable_shards])
# Print diagnostic summary of the computed outputs.
print("Column-shard shapes:", [s.data.shape for s in x_columns.addressable_shards])
# Print diagnostic summary of the computed outputs.
print("Column sums:", np.asarray(sum_rows))
```

Assertions compare against host calculations or hand-derived values; printing a sharding object alone does not establish correctness.

## Run the example

```python
# Step 1 — Start a fresh CPU runtime: Configuration happens before devices() or array creation...
# Import jax for this computation.
import jax
# Run this file in a fresh process; in a notebook restart the kernel first.
jax.config.update("jax_platforms", "cpu")
# Update state in place with the new values.
jax.config.update("jax_num_cpu_devices", 4)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
# Query the active JAX devices into `devices`.
devices = jax.devices("cpu")
# Guard input contract (`len(devices) != 4`) and fail fast if violated.
if len(devices) != 4:
    raise RuntimeError("Expected four CPU devices. Restart the notebook kernel, then Run all; do not run an earlier JAX cell first.")
# Configure multi-device placement / sharding specification (`mesh`).
mesh = Mesh(np.array(devices), ("data",))
# Configure multi-device placement / sharding specification (`rows`).
rows = NamedSharding(mesh, P("data", None))
# Configure multi-device placement / sharding specification (`replicated`).
replicated = NamedSharding(mesh, P())

# Step 2 — Place and compute: Mesh axis names describe placement.
# Construct and reshape `host` into the target tensor dimensions.
host = np.arange(32, dtype=np.float32).reshape(8,4)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host, rows)
# Configure multi-device placement / sharding specification (`columns`).
columns = NamedSharding(mesh, P(None, "data"))
# Place `x_columns` explicitly onto the target JAX device.
x_columns = jax.device_put(host, columns)
# Wrap with `jax.jit` (`sum_rows`) so XLA traces and compiles the function.
sum_rows = jax.jit(lambda a:a.sum(axis=0), in_shardings=rows, out_shardings=replicated)(x)
# Synchronize host execution until asynchronous device computation completes.
sum_rows.block_until_ready()

# Step 3 — Inspect and verify: Assertions compare against host calculations or hand-derived...
np.testing.assert_array_equal(np.asarray(sum_rows), np.array([112,120,128,136],dtype=np.float32))
# Iterate over `shard` to step through the computation:
for shard in x.addressable_shards:
    # Verify that the output tensor shape matches our prediction.
    assert shard.data.shape == (2,4)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), host[shard.index])
# Iterate over `shard` to step through the computation:
for shard in x_columns.addressable_shards:
    # Verify that the output tensor shape matches our prediction.
    assert shard.data.shape == (8,1)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), host[shard.index])
# Print the observed values to compare against the expected result.
print("Row-shard shapes:", [s.data.shape for s in x.addressable_shards])
# Print diagnostic summary of the computed outputs.
print("Column-shard shapes:", [s.data.shape for s in x_columns.addressable_shards])
# Print diagnostic summary of the computed outputs.
print("Column sums:", np.asarray(sum_rows))
```

Expected: Four row shards $(2,4)$; four column shards $(8,1)$. Column sums $[112,120,128,136]$.

## Row sharding and column sharding place the same array differently

**Predict:** Which dimension is split across devices in each panel?

![Row sharding and column sharding place the same array differently](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

Both panels represent the same global array with shape $(8,4)$. Each cell’s number and color identify its owning logical device, not the array value stored there. Device colors are categorical and do not rank speed or capacity.

In the upper panel, each device owns two complete rows: device $0$ owns rows $0$ and $1$, for example. In the lower panel, each device owns one complete column across all eight rows. The horizontal bands become vertical bands when the partitioned axis changes.

### Connect it to the computation

Each device owns eight scalar entries in either layout, but their local shapes differ: $(2,4)$ for row partitioning and $(8,1)$ for column partitioning. Equal local element counts do not mean that the same operations will have the same communication needs.

For example, a whole row is locally available in the upper layout but split across devices in the lower one. A row-wise reduction therefore has different placement implications. The figure establishes ownership only; communication volume and execution time are not measured, and these four logical devices share one CPU host.

```python
# Compute figure data for: Row sharding and column sharding place the same array differently
# Evaluate `panels` from the current inputs and state.
panels = []
# Loop over `(name, array)` in `[('Row partition', x), ('Column partition', x_columns)]`:
for name, array in [('Row partition', x), ('Column partition', x_columns)]:
    # Run `np.empty` to compute `owner`.
    owner = np.empty(host.shape, dtype=int)
    # Loop over `s` in `array.addressable_shards`:
    for s in array.addressable_shards:
        # Evaluate `owner[s.index]` from the current inputs and state.
        owner[s.index] = s.device.id
    # Append the current step result to `panels`.
    panels.append({'kind': 'heatmap', 'title': name, 'values': owner.tolist(), 'unit': 'logical CPU device ID'})
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'panels', 'panels': panels}
```

## Recorded reference execution

CPU run: 2026-10-08T14:03:59.243938+00:00. JAX 0.9.2.

```text
Row-shard shapes: [(2, 4), (2, 4), (2, 4), (2, 4)]
Column-shard shapes: [(8, 1), (8, 1), (8, 1), (8, 1)]
Column sums: [112. 120. 128. 136.]
Row-shard shapes: [(2, 4), (2, 4), (2, 4), (2, 4)]
Column-shard shapes: [(8, 1), (8, 1), (8, 1), (8, 1)]
Column sums: [112. 120. 128. 136.]
Resharded values preserved
Row sums: [  6.  22.  38.  54.  70.  86. 102. 118.]
Expected indivisible batch
Expected indivisible batch
PASS: distributed-01

```

## Reshard the same global values

**Predict before running:** Moving $x$ from row partitioning to column partitioning changes local shapes. Does it change the global values?

```python
# Experiment — Reshard the same global values: Resharding changes ownership of values, not their mathematical...
moved = jax.device_put(x, columns)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(moved), host)
# Verify that the output tensor shape matches our prediction.
assert all(s.data.shape == (8,1) for s in moved.addressable_shards)
# Print the observed values to compare against the expected result.
print("Resharded values preserved")
```

**Expected:** Global array unchanged; local shards now $(8,1)$.

Resharding changes ownership of values, not their mathematical identity.

## Reduce a dimension that is not partitioned

**Predict before running:** Sum each row. Predict the output shape and which mesh axis should partition it.

```python
# Experiment — Reduce a dimension that is not partitioned: All four columns for a row are already local in row...
# Configure multi-device placement / sharding specification (`vector_rows`).
vector_rows = NamedSharding(mesh, P("data"))
# Wrap with `jax.jit` (`per_row`) so XLA traces and compiles the function.
per_row = jax.jit(lambda a:a.sum(axis=1), in_shardings=rows, out_shardings=vector_rows)(x)
# Create evenly spaced index values in ``.
np.testing.assert_array_equal(np.asarray(per_row), np.arange(6,119,16,dtype=np.float32))
# Verify that the output tensor shape matches our prediction.
assert all(s.data.shape == (2,) for s in per_row.addressable_shards)
# Print the observed values to compare against the expected result.
print("Row sums:", np.asarray(per_row))
```

**Expected:** Row sums $[6,22,38,54,70,86,102,118]$; local shapes $(2,)$.

All four columns for a row are already local in row partitioning; this reduction does not mathematically require combining data-axis pieces.

## Make it yours

Compare row and column layouts of a $(12,8)$ array, and repair a ten-row mean with padding plus an explicit mask.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `array.reshape(new_shape)` — Reorganizes tensor axes without changing the total element count (`array.size`).
- `x.sum(axis=...) / x.mean(axis=..., keepdims=...)` — Reduces values along the named `axis` (the axis you name is collapsed unless `keepdims=True`).
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.jit(fn) / @jax.jit` — Traces `fn` with abstract shapes and compiles a fused XLA executable cached by input shape and dtype.

**Step-by-step implementation plan:**
1. Construct and reshape `other` into the target tensor dimensions.
2. Iterate over `(spec, expected)` to step through the computation:
3. Place `placed` explicitly onto the target JAX device.
4. Iterate over `shard` to step through the computation:
5. Verify that the output tensor shape matches our prediction.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Compare row and column layouts of a (12,8) array, and repair a ten-row...
# Construct and reshape `other` into the target tensor dimensions.
other = np.arange(...)  # TODO: compute other
# Iterate over `(spec, expected)` to step through the computation:
for spec, expected in [(rows,(3,8)),(columns,(12,2))]:
    # Place `placed` explicitly onto the target JAX device.
    placed = jax.device_put(...)  # TODO: compute placed
    # Iterate over `shard` to step through the computation:
    for shard in placed.addressable_shards:
        # Verify that the output tensor shape matches our prediction.
        assert shard.data.shape  # TODO: complete assertion check
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_array_equal(np.asarray(shard.data),other[shard.index])

# Construct and reshape `original` into the target tensor dimensions.
original = np.arange(...)  # TODO: compute original
# Run the boundary check and catch the expected exception:
try:
    jax.device_put(original, rows)
except ValueError:
    print("Expected indivisible batch")
else:
    raise AssertionError("Expected divisibility failure")
# Place `padded` explicitly onto the target JAX device.
padded = jax.device_put(...)  # TODO: compute padded
# Configure multi-device placement / sharding specification (`mask_sharding`).
mask_sharding = NamedSharding(...)  # TODO: compute mask_sharding
# Initialize array `mask` with explicit values and shape.
mask = jax.device_put(...)  # TODO: compute mask
# Wrap with `jax.jit` (`masked_mean`) so XLA traces and compiles the function.
masked_mean = jax.jit(...)  # TODO: compute masked_mean
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(masked_mean),original.mean(axis = ...  # TODO: compute np.testing.assert_allclose(np.asarray(masked_mean),original.mean(axis
# Verify that the numerical values match the expected reference within tolerance.
assert not np.allclose(np.asarray(padded.mean(axis=0)), original.mean(axis=0))  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Compare row and column layouts of a (12,8) array, and repair a ten-row...
# Construct and reshape `other` into the target tensor dimensions.
other = np.arange(96,dtype=np.float32).reshape(12,8)
# Iterate over `(spec, expected)` to step through the computation:
for spec, expected in [(rows,(3,8)),(columns,(12,2))]:
    # Place `placed` explicitly onto the target JAX device.
    placed = jax.device_put(other,spec)
    # Iterate over `shard` to step through the computation:
    for shard in placed.addressable_shards:
        # Verify that the output tensor shape matches our prediction.
        assert shard.data.shape == expected
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_array_equal(np.asarray(shard.data),other[shard.index])

# Construct and reshape `original` into the target tensor dimensions.
original = np.arange(40,dtype=np.float32).reshape(10,4)
# Run the boundary check and catch the expected exception:
try:
    jax.device_put(original, rows)
except ValueError:
    print("Expected indivisible batch")
else:
    raise AssertionError("Expected divisibility failure")
# Place `padded` explicitly onto the target JAX device.
padded = jax.device_put(np.pad(original,((0,2),(0,0))),rows)
# Configure multi-device placement / sharding specification (`mask_sharding`).
mask_sharding = NamedSharding(mesh,P("data"))
# Initialize array `mask` with explicit values and shape.
mask = jax.device_put(np.array([1]*10+[0]*2,dtype=np.float32),mask_sharding)
# Wrap with `jax.jit` (`masked_mean`) so XLA traces and compiles the function.
masked_mean = jax.jit(lambda a,m:(a*m[:,None]).sum(axis=0)/m.sum(),in_shardings=(rows,mask_sharding),out_shardings=replicated)(padded,mask)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(masked_mean),original.mean(axis=0),rtol=1e-6)
# Verify that the numerical values match the expected reference within tolerance.
assert not np.allclose(np.asarray(padded.mean(axis=0)), original.mean(axis=0))
```

</details>

## Change both dimensions

**Challenge**

Use a $(12,8)$ array. Verify row and column layouts and predict every local shape.

<details><summary>Hint</summary>

Four devices divide twelve rows or eight columns.

</details>

### How to write: Change both dimensions — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `array.reshape(new_shape)` — Reorganizes tensor axes without changing the total element count (`array.size`).
- `Mesh + PartitionSpec + NamedSharding` — Maps logical tensor axes onto physical device mesh axes for SPMD data, tensor, or pipeline parallelism.

**Step-by-step implementation plan:**
1. Construct and reshape `other` into the target tensor dimensions.
2. Iterate over `(spec, expected)` to step through the computation:
3. Place `placed` explicitly onto the target JAX device.
4. Iterate over `shard` to step through the computation:
5. Verify that the output tensor shape matches our prediction.

**Starter code scaffold (fill in the TODOs):**

```python
# Change both dimensions (Challenge): The row layout owns three complete feature vectors per...
# Construct and reshape `other` into the target tensor dimensions.
other = np.arange(...)  # TODO: compute other
# Iterate over `(spec, expected)` to step through the computation:
for spec, expected in [(rows,(3,8)),(columns,(12,2))]:
    # Place `placed` explicitly onto the target JAX device.
    placed = jax.device_put(...)  # TODO: compute placed
    # Iterate over `shard` to step through the computation:
    for shard in placed.addressable_shards:
        # Verify that the output tensor shape matches our prediction.
        assert shard.data.shape  # TODO: complete assertion check
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_array_equal(np.asarray(shard.data),other[shard.index])
```

<details><summary>Reference solution and reasoning</summary>

```python
# Change both dimensions (Challenge): The row layout owns three complete feature vectors per...
# Construct and reshape `other` into the target tensor dimensions.
other = np.arange(96,dtype=np.float32).reshape(12,8)
# Iterate over `(spec, expected)` to step through the computation:
for spec, expected in [(rows,(3,8)),(columns,(12,2))]:
    # Place `placed` explicitly onto the target JAX device.
    placed = jax.device_put(other,spec)
    # Iterate over `shard` to step through the computation:
    for shard in placed.addressable_shards:
        # Verify that the output tensor shape matches our prediction.
        assert shard.data.shape == expected
        # Convert `` to a host NumPy array for inspection or verification.
        np.testing.assert_array_equal(np.asarray(shard.data),other[shard.index])
```

The row layout owns three complete feature vectors per device; the column layout owns two features for every observation.

</details>

## Pad an uneven batch without corrupting its mean

**Challenge**

Ten observations cannot be row-partitioned over four devices. Pad to twelve and use a mask to recover the true column mean. Demonstrate the wrong unmasked mean.

<details><summary>Hint</summary>

The divisor is the sum of the mask, not padded length.

</details>

### How to write: Pad an uneven batch without corrupting its mean — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `array.reshape(new_shape)` — Reorganizes tensor axes without changing the total element count (`array.size`).
- `x.sum(axis=...) / x.mean(axis=..., keepdims=...)` — Reduces values along the named `axis` (the axis you name is collapsed unless `keepdims=True`).
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.jit(fn) / @jax.jit` — Traces `fn` with abstract shapes and compiles a fused XLA executable cached by input shape and dtype.

**Step-by-step implementation plan:**
1. Construct and reshape `original` into the target tensor dimensions.
2. Run the boundary check and catch the expected exception:
3. Place `padded` explicitly onto the target JAX device.
4. Configure multi-device placement / sharding specification (`mask_sharding`).
5. Initialize array `mask` with explicit values and shape.

**Starter code scaffold (fill in the TODOs):**

```python
# Pad an uneven batch without corrupting its mean (Challenge): Padding repairs the shape but adds artificial rows.
# Construct and reshape `original` into the target tensor dimensions.
original = np.arange(...)  # TODO: compute original
# Run the boundary check and catch the expected exception:
try:
    jax.device_put(original, rows)
except ValueError:
    print("Expected indivisible batch")
else:
    raise AssertionError("Expected divisibility failure")
# Place `padded` explicitly onto the target JAX device.
padded = jax.device_put(...)  # TODO: compute padded
# Configure multi-device placement / sharding specification (`mask_sharding`).
mask_sharding = NamedSharding(...)  # TODO: compute mask_sharding
# Initialize array `mask` with explicit values and shape.
mask = jax.device_put(...)  # TODO: compute mask
# Wrap with `jax.jit` (`masked_mean`) so XLA traces and compiles the function.
masked_mean = jax.jit(...)  # TODO: compute masked_mean
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(masked_mean),original.mean(axis = ...  # TODO: compute np.testing.assert_allclose(np.asarray(masked_mean),original.mean(axis
# Verify that the numerical values match the expected reference within tolerance.
assert not np.allclose(np.asarray(padded.mean(axis=0)), original.mean(axis=0))  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Pad an uneven batch without corrupting its mean (Challenge): Padding repairs the shape but adds artificial rows.
# Construct and reshape `original` into the target tensor dimensions.
original = np.arange(40,dtype=np.float32).reshape(10,4)
# Run the boundary check and catch the expected exception:
try:
    jax.device_put(original, rows)
except ValueError:
    print("Expected indivisible batch")
else:
    raise AssertionError("Expected divisibility failure")
# Place `padded` explicitly onto the target JAX device.
padded = jax.device_put(np.pad(original,((0,2),(0,0))),rows)
# Configure multi-device placement / sharding specification (`mask_sharding`).
mask_sharding = NamedSharding(mesh,P("data"))
# Initialize array `mask` with explicit values and shape.
mask = jax.device_put(np.array([1]*10+[0]*2,dtype=np.float32),mask_sharding)
# Wrap with `jax.jit` (`masked_mean`) so XLA traces and compiles the function.
masked_mean = jax.jit(lambda a,m:(a*m[:,None]).sum(axis=0)/m.sum(),in_shardings=(rows,mask_sharding),out_shardings=replicated)(padded,mask)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(masked_mean),original.mean(axis=0),rtol=1e-6)
# Verify that the numerical values match the expected reference within tolerance.
assert not np.allclose(np.asarray(padded.mean(axis=0)), original.mean(axis=0))
```

Padding repairs the shape but adds artificial rows. The mask keeps them out of both numerator and denominator. Checking only divisibility would miss this statistical error.

</details>

## Check your understanding

With row sharding, why does a global column sum need results from other devices?

1. Each device initially holds only some rows.
2. Every sum requires host Python.
3. Partitioning changes the mathematical definition of sum.

<details><summary>Answer and explanation</summary>

Each device initially holds only some rows.

The reduction spans the partitioned row dimension, so partial contributions must be combined.

</details>

## Diagnose the result

Padding repairs the shape but adds artificial rows. The mask keeps them out of both numerator and denominator. Checking only divisibility would miss this statistical error.

## Carry forward

- Read the mapping from left to right
- Inspect indices rather than guessing ordering
- A reduction has a communication requirement
- Choose placement for the operation you will perform

## Keep your evidence

Padding repairs the shape but adds artificial rows. The mask keeps them out of both numerator and denominator. Checking only divisibility would miss this statistical error.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX CPU device configuration (set before initialization)](https://docs.jax.dev/en/latest/config_options.html#num-cpu-devices)
- [Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/201/sharding.html)

