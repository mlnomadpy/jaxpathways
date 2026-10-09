# Practice with four virtual CPU devices

Phase 00: Setup & first steps · about 60 minutes · 4 logical CPU devices

## What you will be able to do

- Start four virtual CPU devices in a fresh process.
- Explain global shape, shard index and local shape.
- Compare row partitioning with replication.
- Diagnose divisibility and distinguish CPU practice from TPU evidence.

## The problem

You can explore how JAX splits an array across devices using just your laptop. We’ll create four logical CPU devices, give each one a piece of an array, and compare the result with NumPy. Start with the setup and array/device lessons. These logical devices share your CPU: they help us learn placement, but they do not reproduce accelerator speed or memory.

## The idea

Logical CPU devices let us practice how a global array is divided and addressed. They share a physical host. Treat them as a way to make placement visible, then use a real accelerator run to investigate accelerator memory, communication and speed.

## Distinguish ownership from extra computing power

Suppose a global array has four rows and each logical device owns one row. The global shape still describes all four rows; a local shard describes the one row stored for that owner. Asking for the global shape and inspecting a local buffer are different operations.

The ownership heatmap assigns a category to each device. A darker device color does not mean more work or a faster device. Follow the row boundary and the device label together.

Set the logical-device configuration before JAX initializes its backend, then inspect the actual device count in a fresh process. Changing a flag after initialization does not retroactively repartition an already-running process.

### Pause and reason

A four-device CPU exercise passes. What can you now claim about four TPU chips?

<details><summary>Compare your reasoning</summary>

You have checked the array-placement exercise on logical CPU devices. TPU execution, memory behavior and performance still require a separate run on that hardware. The ownership reasoning transfers; the measured performance does not.

</details>

## Configure the runtime before it exists

Open the lesson workspace from setup and create main.py. The first block imports JAX, chooses CPU, and requests four CPU devices. Importing jax is allowed before the configuration; querying devices or creating an array can initialize the backend. That initialization is the boundary after which changing the count is too late.

Run python main.py as a fresh process. In Jupyter or Colab, restart the kernel/runtime and use Run all so this configuration is the first JAX operation. If you already ran an earlier lesson in that kernel, merely rerunning this cell cannot reliably rebuild the runtime. The explicit count check gives you a repair instruction instead of silently proceeding with one device. A terminal alternative is `JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 python main.py`; Windows PowerShell uses `$env:JAX_PLATFORMS="cpu"` and `$env:JAX_NUM_CPU_DEVICES="4"` before python main.py.

## Distinguish logical devices from your CPU hardware

Four reported CPU devices do not mean you bought four CPUs or reserved four isolated cores. They share your computer and its resources. Each is a target JAX can place an array on. Keep the actual JAX version, device platform, process count and device count in your report so readers know what was tested.

This exercise has one controller process and four addressable devices. Multi-host JAX involves additional processes and coordination; this exercise does not test those. Even if you time an operation here, the observation belongs to this CPU setup and workload. It cannot establish whether the same partitioning is fast on a TPU.

## A global array contains local pieces

The host array has shape $(4,2)$: four examples with two values each. Mesh assigns the four devices the named axis data. `P("data",None)` maps the first array dimension to that mesh axis and leaves the feature dimension unpartitioned. Each local shard therefore has shape $(1,2)$.

Use shard.index to identify its slice in the global array and shard.data to inspect its values. The global x.shape stays $(4,2)$; it does not become $(1,2)$ just because each device holds one row. Conversion with `np.asarray(x)` gathers the fully addressable global array to the host in this small single-process exercise.

```text
Global rows [0,1] [2,3] [4,5] [6,7]
             ↓     ↓     ↓     ↓
CPU device   0     1     2     3
local shape (1,2) on every device; global shape (4,2)
```

## Check arithmetic separately from placement

The function $2x+1$ is elementwise: a row does not need a value from a different row. For the first row $[0,1]$, predict $[1,3]$; for the final row $[6,7]$, predict $[13,15]$. The jit contract declares input and output placement explicitly. The NumPy check verifies all values, and shard inspections verify the intended partition. Neither alone verifies both.

block_until_ready waits for the result before reporting. It is useful when observing execution and essential for a future synchronized timing experiment, but no timings are claimed here. Four-way row partitioning requires a row count divisible by four. Try five rows deliberately, read the shape error, and choose a repair that preserves what your experiment means.

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
host = np.arange(8, dtype=np.float32).reshape(4, 2)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host, rows)
# Wrap with `jax.jit` (`y`) so XLA traces and compiles the function.
y = jax.jit(lambda a: 2 * a + 1, in_shardings=rows, out_shardings=rows)(x)
# Synchronize host execution until asynchronous device computation completes.
y.block_until_ready()
```

Mesh axis names describe placement. They are separate from the numerical array dimensions.

## Inspect and verify

Append the checks, save the file and run python main.py with the setup lesson environment.

```python
# Step 3 — Inspect and verify: Assertions compare against host calculations or hand-derived...
# Print the observed values to compare against the expected result.
print("JAX version:", jax.__version__)
# Print diagnostic summary of the computed outputs.
print("CPU devices:", len(devices))
# Iterate over `shard` to step through the computation:
for shard in x.addressable_shards:
    # Print diagnostic summary of the computed outputs.
    print("Device", shard.device.id, "index", shard.index, "values", np.asarray(shard.data))
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(y), 2 * host + 1)
# Verify contract: `len(x.addressable_shards) == 4`.
assert len(x.addressable_shards) == 4
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert all(shard.data.shape == (1, 2) for shard in x.addressable_shards)
# Print the observed values to compare against the expected result.
print("Global shape:", x.shape, "result:", np.asarray(y).tolist())
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
host = np.arange(8, dtype=np.float32).reshape(4, 2)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(host, rows)
# Wrap with `jax.jit` (`y`) so XLA traces and compiles the function.
y = jax.jit(lambda a: 2 * a + 1, in_shardings=rows, out_shardings=rows)(x)
# Synchronize host execution until asynchronous device computation completes.
y.block_until_ready()

# Step 3 — Inspect and verify: Assertions compare against host calculations or hand-derived...
# Print the observed values to compare against the expected result.
print("JAX version:", jax.__version__)
# Print diagnostic summary of the computed outputs.
print("CPU devices:", len(devices))
# Iterate over `shard` to step through the computation:
for shard in x.addressable_shards:
    # Print diagnostic summary of the computed outputs.
    print("Device", shard.device.id, "index", shard.index, "values", np.asarray(shard.data))
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(y), 2 * host + 1)
# Verify contract: `len(x.addressable_shards) == 4`.
assert len(x.addressable_shards) == 4
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert all(shard.data.shape == (1, 2) for shard in x.addressable_shards)
# Print the observed values to compare against the expected result.
print("Global shape:", x.shape, "result:", np.asarray(y).tolist())
```

Expected: CPU devices: $4$. Four input shards of shape $(1,2)$. Global shape $(4,2)$; result $[[1,3]$,$[5,7]$,$[9,11]$,$[13,15]]$. Device repr and JAX version vary.

## Four logical devices own four different rows

**Predict:** Does each device hold the full array or one row?

![Four logical devices own four different rows](../../phases/00-welcome/03-virtual-cpu-devices/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

Each cell represents a position in the global array, and its number identifies the logical CPU device that owns that position. The colors are categories: device $3$ is not “more powerful” than device $0$.

Both cells in row $0$ belong to device $0$; both cells in row $1$ belong to device $1$, and the same pattern continues through row $3$. Horizontal bands show that this layout partitions rows while keeping each row’s features together.

### Connect it to the computation

The global shape is $(4,2)$, so splitting its first axis across four devices gives each device a local piece with shape $(1,2)$. This is the spatial meaning of the row partition in the code. The numbers in the figure are ownership labels, not the original array’s contents.

You can check the interpretation by locating row $2$, feature $1$: its owner is device $2$. These logical devices share the host CPU. The figure establishes where values are placed, not a measured speedup or the behavior of four physical accelerators.

```python
# Compute figure data for: Four logical devices own four different rows
# Run `np.empty` to compute `owners`.
owners = np.empty(host.shape, dtype=int)
# Loop over `s` in `x.addressable_shards`:
for s in x.addressable_shards:
    # Evaluate `owners[s.index]` from the current inputs and state.
    owners[s.index] = s.device.id
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'heatmap', 'values': owners.tolist(), 'unit': 'logical CPU device ID', 'rows': ['row ' + str(i) for i in range(4)], 'columns': ['feature 0', 'feature 1']}
```

## Recorded reference execution

CPU run: 2026-10-08T14:01:10.189781+00:00. JAX 0.9.2.

```text
JAX version: 0.9.2
CPU devices: 4
Device 0 index (slice(0, 1, None), slice(None, None, None)) values [[0. 1.]]
Device 1 index (slice(1, 2, None), slice(None, None, None)) values [[2. 3.]]
Device 2 index (slice(2, 3, None), slice(None, None, None)) values [[4. 5.]]
Device 3 index (slice(3, 4, None), slice(None, None, None)) values [[6. 7.]]
Global shape: (4, 2) result: [[1.0, 3.0], [5.0, 7.0], [9.0, 11.0], [13.0, 15.0]]
JAX version: 0.9.2
CPU devices: 4
Device 0 index (slice(0, 1, None), slice(None, None, None)) values [[0. 1.]]
Device 1 index (slice(1, 2, None), slice(None, None, None)) values [[2. 3.]]
Device 2 index (slice(2, 3, None), slice(None, None, None)) values [[4. 5.]]
Device 3 index (slice(3, 4, None), slice(None, None, None)) values [[6. 7.]]
Global shape: (4, 2) result: [[1.0, 3.0], [5.0, 7.0], [9.0, 11.0], [13.0, 15.0]]
Negative input result: [[-3.0, -5.0], [-7.0, -9.0], [-11.0, -13.0], [-15.0, -17.0]]
Replicated local shapes: [(4, 2), (4, 2), (4, 2), (4, 2)]
Expected indivisible row dimension
Expected indivisible row dimension
PASS: welcome-cpu

```

## Keep the shape, change the values

**Predict before running:** If every input value becomes negative, do the number of devices or local shard shapes change?

```python
# Experiment — Keep the shape, change the values: Placement follows the declared shape/specification, not the...
negative = -host - 2
# Place `negative_x` explicitly onto the target JAX device.
negative_x = jax.device_put(negative, rows)
# Wrap with `jax.jit` (`negative_y`) so XLA traces and compiles the function.
negative_y = jax.jit(lambda a: 2*a+1, in_shardings=rows, out_shardings=rows)(negative_x)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(negative_y), 2*negative+1)
# Verify that the output tensor shape matches our prediction.
assert all(s.data.shape == (1,2) for s in negative_x.addressable_shards)
# Print the observed values to compare against the expected result.
print("Negative input result:", np.asarray(negative_y).tolist())
```

**Expected:** Result $[[-3,-5]$,$[-7,-9]$,$[-11,-13]$,$[-15,-17]]$; local shapes remain $(1,2)$.

Placement follows the declared shape/specification, not the signs or magnitudes of array elements.

## Replicate instead of partitioning

**Predict before running:** With `P()`, which values should each CPU device receive?

```python
# Experiment — Replicate instead of partitioning: Replication stores full copies.
copies = jax.device_put(host, replicated)
# Verify contract: `copies.is_fully_replicated`.
assert copies.is_fully_replicated
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert len(copies.addressable_shards) == 4
# Iterate over `shard` to step through the computation:
for shard in copies.addressable_shards:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), host)
# Print the observed values to compare against the expected result.
print("Replicated local shapes:", [s.data.shape for s in copies.addressable_shards])
```

**Expected:** Four local arrays of shape $(4,2)$, each containing the entire input.

Replication stores full copies. A global array shape alone cannot tell you how much of it lives on each device.

## Make it yours

Predict and verify the per-device shapes for twelve rows, then demonstrate how replication repairs an indivisible five-row placement without deleting data.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `array.reshape(new_shape)` — Reorganizes tensor axes without changing the total element count (`array.size`).
- `jax.jit(fn) / @jax.jit` — Traces `fn` with abstract shapes and compiles a fused XLA executable cached by input shape and dtype.
- `Mesh + PartitionSpec + NamedSharding` — Maps logical tensor axes onto physical device mesh axes for SPMD data, tensor, or pipeline parallelism.

**Step-by-step implementation plan:**
1. Construct and reshape `twelve` into the target tensor dimensions.
2. Place `z` explicitly onto the target JAX device.
3. Wrap with `jax.jit` (`z2`) so XLA traces and compiles the function.
4. Verify that the output tensor shape matches our prediction.
5. Iterate over `shard` to step through the computation:

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Predict and verify the per-device shapes for twelve rows, then...
# Construct and reshape `twelve` into the target tensor dimensions.
twelve = np.arange(...)  # TODO: compute twelve
# Place `z` explicitly onto the target JAX device.
z = jax.device_put(...)  # TODO: compute z
# Wrap with `jax.jit` (`z2`) so XLA traces and compiles the function.
z2 = jax.jit(...)  # TODO: compute z2
# Verify that the output tensor shape matches our prediction.
assert all(s.data.shape  # TODO: complete assertion check
# Iterate over `shard` to step through the computation:
for shard in z2.addressable_shards:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), (twelve*twelve)[shard.index])

# Construct and reshape `uneven` into the target tensor dimensions.
uneven = np.arange(...)  # TODO: compute uneven
# Run the boundary check and catch the expected exception:
try:
    jax.device_put(uneven, rows)
except ValueError:
    print("Expected indivisible row dimension")
else:
    raise AssertionError("Expected divisibility failure")
# Place `fixed` explicitly onto the target JAX device.
fixed = jax.device_put(...)  # TODO: compute fixed
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(fixed), uneven)
# Verify contract: `fixed.is_fully_replicated`.
assert fixed.is_fully_replicated  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Predict and verify the per-device shapes for twelve rows, then...
# Construct and reshape `twelve` into the target tensor dimensions.
twelve = np.arange(24,dtype=np.float32).reshape(12,2)
# Place `z` explicitly onto the target JAX device.
z = jax.device_put(twelve, rows)
# Wrap with `jax.jit` (`z2`) so XLA traces and compiles the function.
z2 = jax.jit(lambda a:a*a, in_shardings=rows, out_shardings=rows)(z)
# Verify that the output tensor shape matches our prediction.
assert all(s.data.shape == (3,2) for s in z2.addressable_shards)
# Iterate over `shard` to step through the computation:
for shard in z2.addressable_shards:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), (twelve*twelve)[shard.index])

# Construct and reshape `uneven` into the target tensor dimensions.
uneven = np.arange(10,dtype=np.float32).reshape(5,2)
# Run the boundary check and catch the expected exception:
try:
    jax.device_put(uneven, rows)
except ValueError:
    print("Expected indivisible row dimension")
else:
    raise AssertionError("Expected divisibility failure")
# Place `fixed` explicitly onto the target JAX device.
fixed = jax.device_put(uneven, replicated)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(fixed), uneven)
# Verify contract: `fixed.is_fully_replicated`.
assert fixed.is_fully_replicated
```

</details>

## Move from four rows to twelve

**Challenge**

Create twelve two-feature rows with values $0$ through $23$. Predict shard shape and check every slice of a squared output.

<details><summary>Hint</summary>

Use each shard index to select the corresponding NumPy reference slice.

</details>

### How to write: Move from four rows to twelve — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `array.reshape(new_shape)` — Reorganizes tensor axes without changing the total element count (`array.size`).
- `jax.jit(fn) / @jax.jit` — Traces `fn` with abstract shapes and compiles a fused XLA executable cached by input shape and dtype.
- `Mesh + PartitionSpec + NamedSharding` — Maps logical tensor axes onto physical device mesh axes for SPMD data, tensor, or pipeline parallelism.

**Step-by-step implementation plan:**
1. Construct and reshape `twelve` into the target tensor dimensions.
2. Place `z` explicitly onto the target JAX device.
3. Wrap with `jax.jit` (`z2`) so XLA traces and compiles the function.
4. Verify that the output tensor shape matches our prediction.
5. Iterate over `shard` to step through the computation:

**Starter code scaffold (fill in the TODOs):**

```python
# Move from four rows to twelve (Challenge): The twelve rows split into four groups of three.
# Construct and reshape `twelve` into the target tensor dimensions.
twelve = np.arange(...)  # TODO: compute twelve
# Place `z` explicitly onto the target JAX device.
z = jax.device_put(...)  # TODO: compute z
# Wrap with `jax.jit` (`z2`) so XLA traces and compiles the function.
z2 = jax.jit(...)  # TODO: compute z2
# Verify that the output tensor shape matches our prediction.
assert all(s.data.shape  # TODO: complete assertion check
# Iterate over `shard` to step through the computation:
for shard in z2.addressable_shards:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), (twelve*twelve)[shard.index])
```

<details><summary>Reference solution and reasoning</summary>

```python
# Move from four rows to twelve (Challenge): The twelve rows split into four groups of three.
# Construct and reshape `twelve` into the target tensor dimensions.
twelve = np.arange(24,dtype=np.float32).reshape(12,2)
# Place `z` explicitly onto the target JAX device.
z = jax.device_put(twelve, rows)
# Wrap with `jax.jit` (`z2`) so XLA traces and compiles the function.
z2 = jax.jit(lambda a:a*a, in_shardings=rows, out_shardings=rows)(z)
# Verify that the output tensor shape matches our prediction.
assert all(s.data.shape == (3,2) for s in z2.addressable_shards)
# Iterate over `shard` to step through the computation:
for shard in z2.addressable_shards:
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(np.asarray(shard.data), (twelve*twelve)[shard.index])
```

The twelve rows split into four groups of three. Index-based comparisons establish that the right values occupy each piece, rather than checking only a total.

</details>

## Repair a batch that does not divide evenly

**Challenge**

Place a $(5,2)$ array with the same row sharding. Capture the failure. Repair this placement demonstration by replicating it, and explain why that does not prove the partitioned algorithm supports uneven batches.

<details><summary>Hint</summary>

Do not drop a row to hide the error.

</details>

### How to write: Repair a batch that does not divide evenly — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `array.reshape(new_shape)` — Reorganizes tensor axes without changing the total element count (`array.size`).
- `Mesh + PartitionSpec + NamedSharding` — Maps logical tensor axes onto physical device mesh axes for SPMD data, tensor, or pipeline parallelism.

**Step-by-step implementation plan:**
1. Construct and reshape `uneven` into the target tensor dimensions.
2. Run the boundary check and catch the expected exception:
3. Place `fixed` explicitly onto the target JAX device.
4. Convert `` to a host NumPy array for inspection or verification.
5. Verify contract: `fixed.is_fully_replicated`.

**Starter code scaffold (fill in the TODOs):**

```python
# Repair a batch that does not divide evenly (Challenge): Five rows cannot be evenly split across the four-way data axis.
# Construct and reshape `uneven` into the target tensor dimensions.
uneven = np.arange(...)  # TODO: compute uneven
# Run the boundary check and catch the expected exception:
try:
    jax.device_put(uneven, rows)
except ValueError:
    print("Expected indivisible row dimension")
else:
    raise AssertionError("Expected divisibility failure")
# Place `fixed` explicitly onto the target JAX device.
fixed = jax.device_put(...)  # TODO: compute fixed
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(fixed), uneven)
# Verify contract: `fixed.is_fully_replicated`.
assert fixed.is_fully_replicated  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Repair a batch that does not divide evenly (Challenge): Five rows cannot be evenly split across the four-way data axis.
# Construct and reshape `uneven` into the target tensor dimensions.
uneven = np.arange(10,dtype=np.float32).reshape(5,2)
# Run the boundary check and catch the expected exception:
try:
    jax.device_put(uneven, rows)
except ValueError:
    print("Expected indivisible row dimension")
else:
    raise AssertionError("Expected divisibility failure")
# Place `fixed` explicitly onto the target JAX device.
fixed = jax.device_put(uneven, replicated)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(fixed), uneven)
# Verify contract: `fixed.is_fully_replicated`.
assert fixed.is_fully_replicated
```

Five rows cannot be evenly split across the four-way data axis. Replication preserves all observations but changes the memory/communication plan; padding requires a mask when computing means.

</details>

## Check your understanding

What does a $(12,2)$ global array sharded with `P("data",None)` over four devices mean?

1. Each device holds a $(3,2)$ slice and the global array remains $(12,2)$.
2. Each device holds the complete array.
3. Each device represents a separate TPU chip.

<details><summary>Answer and explanation</summary>

Each device holds a $(3,2)$ slice and the global array remains $(12,2)$.

Partitioning divides the mapped dimension; it does not change the global shape or create accelerator hardware.

</details>

## Diagnose the result

Five rows cannot be evenly split across the four-way data axis. Replication preserves all observations but changes the memory/communication plan; padding requires a mask when computing means.

## Carry forward

- Configure the runtime before it exists
- Distinguish logical devices from your CPU hardware
- A global array contains local pieces
- Check arithmetic separately from placement

## Keep your evidence

Keep package/device reports, global and local shapes, shard index/value tables, NumPy comparisons, changed-shape predictions, and the five-row failure/repair. State that the run uses one host and does not validate TPU performance.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [JAX CPU device configuration (set before initialization)](https://docs.jax.dev/en/latest/config_options.html#num-cpu-devices)
- [Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/201/sharding.html)

