# Communication-efficient algorithms

Phase 09: Distributed training · about 100 minutes · 4 logical CPU devices

This reference configures logical CPU devices to practice placement and sharding. Logical devices do not measure TPU performance. Use the execution matrix to review device selection and data ownership before adapting a separate target program.

**Before this lesson:** [A sharded training step](https://www.tahabouhsine.com/jaxpathways/lesson-distributed-02.html).

## What you will be able to do

- Compare all-reduce and reduce-scatter against the same independent global gradient.
- Explain global result shape versus per-device storage.
- Account for a subsequent all-gather before claiming a communication saving.

## The problem

Every device computed a partial gradient. Does every device need the complete result immediately? We will implement an all-reduce and a reduce-scatter on four logical CPU devices, prove they represent the same global gradient, and account for when a later all-gather gives the communication back.

## The idea

A collective combines or redistributes values among participants. Track chunk identity and reduction state at each stage to understand what the operation achieves. Communication volume also depends on the assumed algorithm.

## Track reduced chunks through communication

The eight-value gradient is the same global mathematical result in each program, but its ownership differs. All-reduce gives every device all eight reduced values. Reduce-scatter first sums corresponding contributions, then leaves each device with its assigned two-value slice; an all-gather restores the full vector on each device.

The diagram uses two ranks and four entries to expose the sequence: $[1,2]+[3,4]=[4,6]$, then each rank owns one reduced entry before gathering. In the executed four-device example, the gradient has eight entries, so each reduce-scatter shard has two. A transmitted contribution is not yet the reduced chunk; the reduction happens before ownership is divided.

### Reduce-scatter followed by all-gather

**Predict:** Does every participant hold the complete reduced array after reduce-scatter?

![Reduce-scatter followed by all-gather](../../phases/09-distributed/03-communication-efficient-algorithms/outputs/mechanism.svg)

*Architecture and dataflow mechanism diagram.*

The table's two participants contribute $[1,2]$ and $[3,4]$. Reduce-scatter sums matching entries to $[4,6]$ and assigns one entry to each participant; neither local result is the full vector yet. All-gather then gives both participants $[4,6]$. In the lesson's actual four-rank, eight-value case, the corresponding reduced pieces have two entries each. The table explains ownership and order, not a physical schedule or measured traffic.

### Pause and reason

Does every participant hold the complete reduced array after reduce-scatter?

<details><summary>Compare your reasoning</summary>

No. Each holds its reduced portion. Replicating the complete result requires the appropriate gather stage or another collective contract.

</details>

## Keep normalization independent of device count

There are sixteen observations and eight weights, with four observations on each of four devices. Each local gradient contribution uses the global denominator $N=16$, so summing the four contributions gives the derivative of the global mean loss. If each device divided by its local count 4 and those local means were then summed, the result would be four times too large. A changed batch size must update the global count, not redefine the objective as a sum of local means.

$$
g_r=\frac{2}{N}X_r^\mathsf{T}(X_rw-y_r),\qquad g=\sum_{r=0}^{p-1}g_r,\qquad N=16,\quad p=4
$$

## Read the local result before interpreting the global array

Both programs return a global vector with shape $(8,)$. With all-reduce, each device holds all eight values; the output specification is `P()`. With reduce-scatter, each holds two consecutive values; `P("data")` describes how these pieces form the global vector. The full-array NumPy comparison checks values, while inspecting addressable shards checks storage. A shape print alone cannot distinguish these results.

## Derive a communication model with its assumptions visible

Let $G$ be one gradient's byte size and $p$ the rank count. The idealized ring model gives $(p-1)G/p$ bytes sent per rank for reduce-scatter, and the same for all-gather. Here eight float32 values give $G=32$ bytes; with four ranks the model predicts 24 bytes for reduce-scatter and 48 for the two phases together. The chart's 48/24/48 bars encode those calculated payloads, not network counters. If the same vector were split across eight ranks, the modeled scatter payload would be 28 bytes per rank and the two-phase total 56, while each rank would own one value. Actual timings can still reverse that apparent tradeoff because the model omits latency, implementation and topology.

$$
B_{\mathrm{RS}}=\frac{p-1}{p}G,\qquad B_{\mathrm{AG}}=\frac{p-1}{p}G,\qquad B_{\mathrm{AR}}=2\frac{p-1}{p}G
$$

## Follow the consumer of the gradient

An elementwise optimizer can update matching parameter and momentum slices using a gradient slice. This can reduce persistent per-device optimizer storage. But if the next forward pass expects replicated weights, those updated weights still need gathering. Reducing one collective does not automatically reduce the communication of a whole training step.

Our example implements the gather explicitly and checks the reconstructed vector. In installed JAX 0.9.2, the default all_gather variation rule does not infer the replicated output required here. Only this small wrapper sets check_vma=False; its proof is that every rank gathers the same ordered slices. The code verifies every local replica as well as the global reference. Do not disable variation checks to conceal a genuinely rank-dependent output.

## Separate observed execution from the payload model and run on a TPU mesh

The recorded CPU medians vary across repeated runs: in one execution all-reduce was about 123 microseconds and reduce-scatter about 105; another had all-reduce about 94 and reduce-scatter about 103. These small-job measurements do not establish a stable winner, and neither corresponds to the idealized byte bars. The timed functions are warmed and synchronized, but compilation is excluded and all four logical devices share one host.

Choose the consumer before choosing the collective. If the optimizer can update each parameter and momentum slice locally, reduce-scatter can leave the gradient partitioned; if the next operation requires replicated weights, all-gather returns the storage and communication. Benchmark the complete consuming step on the target topology before claiming a system benefit.

**Run all-reduce and reduce-scatter across 4 TPU chips on a TPU VM**

```bash
gcloud compute tpus tpu-vm scp exercises/distributed-03.py $TPU_NAME:~/jax-tpu-lab/ --zone=$ZONE
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE \
  --command="sed \"s/jax.config.update('jax_platforms', 'cpu')/# use default TPU backend/; s/jax.config.update('jax_num_cpu_devices', 4)/# use physical TPU chips/\" ~/jax-tpu-lab/distributed-03.py > ~/jax-tpu-lab/distributed-03-tpu.py && JAX_PLATFORMS=tpu ~/jax-tpu-lab/.venv/bin/python ~/jax-tpu-lab/distributed-03-tpu.py"
```

**Expected:** Verifies that all_reduce, reduce_scatter, and all_gather match the NumPy reference across 4 physical TPU devices and prints completed TPU median microseconds.

## Configure four logical devices

Add this block after the preceding block in a fresh Python file, then run the complete file.

```python
# Step 1 — Configure four logical devices
import jax
# Select the CPU backend and virtual device count before initializing JAX.
jax.config.update('jax_platforms', 'cpu')
# Select the CPU backend and virtual device count before initializing JAX.
jax.config.update('jax_num_cpu_devices', 4)
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from time import perf_counter
```

Compare both the numerical global result and the placement needed by its next consumer.

## Place the data and model

Add this block after the preceding block in a fresh Python file, then run the complete file.

```python
# Step 2 — Place the data and model
assert jax.local_device_count() == 4, 'Restart with four logical CPU devices'
mesh = Mesh(np.asarray(jax.devices()), ('data',))
rows = NamedSharding(mesh, P('data', None))
replicated = NamedSharding(mesh, P())
features = NamedSharding(mesh, P('data'))
xh = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 128 - 0.5
yh = xh @ np.linspace(-0.4, 0.3, 8, dtype=np.float32)
wh = np.linspace(0.1, -0.2, 8, dtype=np.float32)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(xh, rows)
y = jax.device_put(yh, NamedSharding(mesh, P('data')))
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(wh, replicated)
```

Compare both the numerical global result and the placement needed by its next consumer.

## Implement partial gradients and collectives

Add this block after the preceding block in a fresh Python file, then run the complete file.

```python
# Step 3 — Implement partial gradients and collectives
def local_contribution(a, b, weights):
    return 2 * a.T @ (a @ weights - b) / 16
```

Compare both the numerical global result and the placement needed by its next consumer.

## Verify the result and placement

Add this block after the preceding block in a fresh Python file, then run the complete file.

```python
# Step 4 — Verify the result and placement
def reduce_all(a, b, weights):
    return jax.lax.psum(local_contribution(a, b, weights), 'data')
```

Compare both the numerical global result and the placement needed by its next consumer.

## Measure completed calls

Add this block after the preceding block in a fresh Python file, then run the complete file.

```python
# Step 5 — Measure completed calls
def reduce_shard(a, b, weights):
    return jax.lax.psum_scatter(local_contribution(a, b, weights), 'data', tiled=True)
```

Compare both the numerical global result and the placement needed by its next consumer.

## Measure completed calls

Add this block after the preceding block in a fresh Python file, then run the complete file.

```python
# Step 6 — Measure completed calls
all_gradient = jax.jit(jax.shard_map(reduce_all, mesh=mesh,
    in_specs=(P('data', None), P('data'), P()), out_specs=P()))
shard_gradient = jax.jit(jax.shard_map(reduce_shard, mesh=mesh,
    in_specs=(P('data', None), P('data'), P()), out_specs=P('data')))
gather = jax.jit(jax.shard_map(lambda g: jax.lax.all_gather(g, 'data', tiled=True),
    mesh=mesh, in_specs=P('data'), out_specs=P(), check_vma=False))
ga = all_gradient(x, y, w)
gs = shard_gradient(x, y, w)
gg = gather(gs)
reference = 2 * xh.T @ (xh @ wh - yh) / 16
for result in [ga, gs, gg]:
    # Transfer the device result to the host for an independent NumPy comparison.
    np.testing.assert_allclose(np.asarray(result), reference, atol=2e-6, rtol=2e-6)
# Check tensor shape invariant: `all(s.data.shape == (8,) for s in ga.addressable_shards)`
assert all(s.data.shape == (8,) for s in ga.addressable_shards)
# Check tensor shape invariant: `all(s.data.shape == (2,) for s in gs.addressable_shards)`
assert all(s.data.shape == (2,) for s in gs.addressable_shards)
# Check tensor shape invariant: `all(s.data.shape == (8,) for s in gg.addressable_shards)`
assert all(s.data.shape == (8,) for s in gg.addressable_shards)
# Trace or lower the function to inspect its compiler representation (`hlo_all`).
hlo_all = str(all_gradient.lower(x, y, w).compiler_ir(dialect='stablehlo'))
hlo_shard = str(shard_gradient.lower(x, y, w).compiler_ir(dialect='stablehlo'))
assert 'all_reduce' in hlo_all and 'reduce_scatter' in hlo_shard
```

Compare both the numerical global result and the placement needed by its next consumer.

## Measure completed calls

Add this block after the preceding block in a fresh Python file, then run the complete file.

```python
# Step 7 — Measure completed calls
def completed_samples(fn, *args):
    # Wait for device work to finish before timing it or comparing its result on the host.
    fn(*args).block_until_ready()
    samples = []
    for _ in range(7):
        start = perf_counter()
        # Wait for device work to finish before timing it or comparing its result on the host.
        fn(*args).block_until_ready()
        samples.append((perf_counter() - start) * 1e6)
    return samples
all_us = completed_samples(all_gradient, x, y, w)
shard_us = completed_samples(shard_gradient, x, y, w)
gather_us = completed_samples(gather, gs)
# Idealized ring payload per rank; not a network measurement.
ranks, gradient_bytes = 4, wh.nbytes
ring_scatter_bytes = (ranks - 1) / ranks * gradient_bytes
ring_all_bytes = 2 * ring_scatter_bytes
print('Gradient reference:', reference.tolist())
print('Local result shapes: replicated (8,), reduce-scatter (2,), gathered (8,)')
print('Idealized ring bytes per rank:', ring_all_bytes, ring_scatter_bytes)
print('Completed CPU median microseconds:', {
    'all_reduce': float(np.median(all_us)), 'reduce_scatter': float(np.median(shard_us)),
    'all_gather_only': float(np.median(gather_us))})
```

Compare both the numerical global result and the placement needed by its next consumer.

## Run the example

```python
# Step 1 — Configure four logical devices
import jax
# Select the CPU backend and virtual device count before initializing JAX.
jax.config.update('jax_platforms', 'cpu')
# Select the CPU backend and virtual device count before initializing JAX.
jax.config.update('jax_num_cpu_devices', 4)
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from time import perf_counter

# Step 2 — Place the data and model
assert jax.local_device_count() == 4, 'Restart with four logical CPU devices'
mesh = Mesh(np.asarray(jax.devices()), ('data',))
rows = NamedSharding(mesh, P('data', None))
replicated = NamedSharding(mesh, P())
features = NamedSharding(mesh, P('data'))
xh = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 128 - 0.5
yh = xh @ np.linspace(-0.4, 0.3, 8, dtype=np.float32)
wh = np.linspace(0.1, -0.2, 8, dtype=np.float32)
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(xh, rows)
y = jax.device_put(yh, NamedSharding(mesh, P('data')))
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(wh, replicated)

# Step 3 — Implement partial gradients and collectives
def local_contribution(a, b, weights):
    return 2 * a.T @ (a @ weights - b) / 16

# Step 4 — Verify the result and placement
def reduce_all(a, b, weights):
    return jax.lax.psum(local_contribution(a, b, weights), 'data')

# Step 5 — Measure completed calls
def reduce_shard(a, b, weights):
    return jax.lax.psum_scatter(local_contribution(a, b, weights), 'data', tiled=True)

# Step 6 — Measure completed calls
all_gradient = jax.jit(jax.shard_map(reduce_all, mesh=mesh,
    in_specs=(P('data', None), P('data'), P()), out_specs=P()))
shard_gradient = jax.jit(jax.shard_map(reduce_shard, mesh=mesh,
    in_specs=(P('data', None), P('data'), P()), out_specs=P('data')))
gather = jax.jit(jax.shard_map(lambda g: jax.lax.all_gather(g, 'data', tiled=True),
    mesh=mesh, in_specs=P('data'), out_specs=P(), check_vma=False))
ga = all_gradient(x, y, w)
gs = shard_gradient(x, y, w)
gg = gather(gs)
reference = 2 * xh.T @ (xh @ wh - yh) / 16
for result in [ga, gs, gg]:
    # Transfer the device result to the host for an independent NumPy comparison.
    np.testing.assert_allclose(np.asarray(result), reference, atol=2e-6, rtol=2e-6)
# Check tensor shape invariant: `all(s.data.shape == (8,) for s in ga.addressable_shards)`
assert all(s.data.shape == (8,) for s in ga.addressable_shards)
# Check tensor shape invariant: `all(s.data.shape == (2,) for s in gs.addressable_shards)`
assert all(s.data.shape == (2,) for s in gs.addressable_shards)
# Check tensor shape invariant: `all(s.data.shape == (8,) for s in gg.addressable_shards)`
assert all(s.data.shape == (8,) for s in gg.addressable_shards)
# Trace or lower the function to inspect its compiler representation (`hlo_all`).
hlo_all = str(all_gradient.lower(x, y, w).compiler_ir(dialect='stablehlo'))
hlo_shard = str(shard_gradient.lower(x, y, w).compiler_ir(dialect='stablehlo'))
assert 'all_reduce' in hlo_all and 'reduce_scatter' in hlo_shard

# Step 7 — Measure completed calls
def completed_samples(fn, *args):
    # Wait for device work to finish before timing it or comparing its result on the host.
    fn(*args).block_until_ready()
    samples = []
    for _ in range(7):
        start = perf_counter()
        # Wait for device work to finish before timing it or comparing its result on the host.
        fn(*args).block_until_ready()
        samples.append((perf_counter() - start) * 1e6)
    return samples
all_us = completed_samples(all_gradient, x, y, w)
shard_us = completed_samples(shard_gradient, x, y, w)
gather_us = completed_samples(gather, gs)
# Idealized ring payload per rank; not a network measurement.
ranks, gradient_bytes = 4, wh.nbytes
ring_scatter_bytes = (ranks - 1) / ranks * gradient_bytes
ring_all_bytes = 2 * ring_scatter_bytes
print('Gradient reference:', reference.tolist())
print('Local result shapes: replicated (8,), reduce-scatter (2,), gathered (8,)')
print('Idealized ring bytes per rank:', ring_all_bytes, ring_scatter_bytes)
print('Completed CPU median microseconds:', {
    'all_reduce': float(np.median(all_us)), 'reduce_scatter': float(np.median(shard_us)),
    'all_gather_only': float(np.median(gather_us))})
```

Expected: All three global gradients agree with NumPy. The replicated and reconstructed local results contain eight values each; reduce-scatter shards contain two. The idealized ring payload is 48 bytes per rank for all-reduce and 24 for reduce-scatter. Actual CPU timing samples vary.

## One mathematical gradient, different local storage

**Predict:** Will equal global gradients imply equal numbers of stored values on each device?

![One mathematical gradient, different local storage](../../phases/09-distributed/03-communication-efficient-algorithms/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

In the upper panel, the vertical axis counts locally stored gradient values per device: 8 after all-reduce, 2 after reduce-scatter, and 8 after the following gather. These counts come from the executed shard shapes, and each state represents the same global eight-value gradient. In the lower panel, the vertical axis is modeled ring payload in bytes per rank: 48 for all-reduce, 24 for reduce-scatter alone, and 48 for scatter plus gather. These are calculated model values, not observed traffic.

### Connect it to the computation

The middle upper bar shows what the optimizer can consume locally when parameters and momentum are sharded the same way. The final bar returns to eight values per device because gathering restores replication. Thus lower local storage during one step does not imply lower communication for the full training step. The CPU timing output is a separate observation and varies between runs; use the payload model to form a hypothesis, then measure the whole algorithm on the intended hardware.

```python
visual_data={'panels':[{'kind':'bar','labels':['all-reduce','reduce-scatter','after gather'],'ylabel':'values stored per device','series':[{'label':'observed local shape','y':[ga.addressable_shards[0].data.size,gs.addressable_shards[0].data.size,gg.addressable_shards[0].data.size]}]},{'kind':'bar','labels':['all-reduce','reduce-scatter','scatter + gather'],'ylabel':'modeled bytes per rank','series':[{'label':'idealized ring payload','y':[ring_all_bytes,ring_scatter_bytes,ring_scatter_bytes*2]}]}]}
```

## Recorded reference execution

CPU run: 2026-10-27T17:06:05.104234+00:00. JAX 0.9.2.

```text
Gradient reference: [0.0029296851716935635, 0.00219726306386292, 0.0014648411888629198, 0.0007324193138629198, -2.6193447411060333e-09, -0.000732424552552402, -0.001464846427552402, -0.002197268418967724]
Local result shapes: replicated (8,), reduce-scatter (2,), gathered (8,)
Idealized ring bytes per rank: 48.0 24.0
Completed CPU median microseconds: {'all_reduce': 146.91567048430443, 'reduce_scatter': 140.70887118577957, 'all_gather_only': 121.54225260019302}
Gradient reference: [0.0029296851716935635, 0.00219726306386292, 0.0014648411888629198, 0.0007324193138629198, -2.6193447411060333e-09, -0.000732424552552402, -0.001464846427552402, -0.002197268418967724]
Local result shapes: replicated (8,), reduce-scatter (2,), gathered (8,)
Idealized ring bytes per rank: 48.0 24.0
Completed CPU median microseconds: {'all_reduce': 104.37518358230591, 'reduce_scatter': 143.50004494190216, 'all_gather_only': 126.37488543987274}
Sharded momentum matches the independent elementwise update.
Joint permutation preserves the gradient; label-only reversal does not.
Idealized payload/storage budget, not observed network traffic: [{'ranks': 2, 'all_reduce_bytes': 1048576, 'scatter_gather_bytes': 1048576, 'shard_storage_bytes': 524288}, {'ranks': 4, 'all_reduce_bytes': 1572864, 'scatter_gather_bytes': 1572864, 'shard_storage_bytes': 262144}, {'ranks': 8, 'all_reduce_bytes': 1835008, 'scatter_gather_bytes': 1835008, 'shard_storage_bytes': 131072}]
PASS: distributed-03

```

## Update slices before gathering

**Predict before running:** Can elementwise momentum produce the same update while its state remains partitioned?

```python
# Experiment — Update slices before gathering: An elementwise update can consume the partitioned gradient.
v_host = np.linspace(-0.02, 0.03, 8, dtype=np.float32)
# Place `ws` explicitly onto the target JAX device.
ws = jax.device_put(wh, features)
# Place `vs` explicitly onto the target JAX device.
vs = jax.device_put(v_host, features)
new_v = 0.9 * vs + gs
new_w = ws - 0.1 * new_v
updated = gather(new_w)
expected_w = wh - 0.1 * (0.9 * v_host + reference)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(updated), expected_w, atol=2e-6, rtol=2e-6)
for shard in updated.addressable_shards:
    # Transfer the device result to the host for an independent NumPy comparison.
    np.testing.assert_allclose(np.asarray(shard.data), expected_w, atol=2e-6, rtol=2e-6)
# Check tensor shape invariant: `all(s.data.shape == (2,) for s in new_v.addressable_shards)`
assert all(s.data.shape == (2,) for s in new_v.addressable_shards)
print('Sharded momentum matches the independent elementwise update.')
```

**Expected:** The gathered update matches the independent momentum formula, while local momentum retains two values.

An elementwise update can consume the partitioned gradient. Gathering the updated parameter later is still real communication, so include it when evaluating the full algorithm.

## Make it yours

Repeat the gradient comparison after reversing both observation rows and labels and changing the initial weights. Verify every all-reduce and reconstructed local replica, then explain why reversing only the labels changes the problem.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Inputs:**
- Reversed observation rows and labels, changed initial weights, and the distributed gradient routine from the lesson.

**Outputs:**
- Gradients and updated parameters after the changed data ordering.
- Reconstructed local replicas and all-reduce comparisons.
- A comparison showing why reversing labels without the corresponding rows changes the gradient problem.

**Invariants to check:**
- Reversing both paired arrays preserves examples; reversing labels alone changes targets.
- Each local replica matches the globally updated parameter vector.

**Hints:**
- Apply the same row permutation to features and labels.
- Compare local contributions, reduced gradients, and replica state independently.

**Starter code scaffold (fill in the TODOs):**

```python
import numpy as np

def verify_changed_distributed_step(features, labels, initial_weights, update):
    # TODO: reverse paired observations, change the initial state, and audit replicas.
    raise NotImplementedError("Check pairing, reduction, and replica agreement")
```

<details><summary>Reference solution</summary>

```python
# Exercise solution
x2 = jax.device_put(xh[::-1].copy(), rows)
y2 = jax.device_put(yh[::-1].copy(), NamedSharding(mesh, P('data')))
w2h = wh + np.float32(0.17)
# Place `w2` explicitly onto the target JAX device.
w2 = jax.device_put(w2h, replicated)
r2 = 2 * xh.T @ (xh @ w2h - yh) / 16
for answer in [all_gradient(x2, y2, w2), gather(shard_gradient(x2, y2, w2))]:
    # Transfer the device result to the host for an independent NumPy comparison.
    np.testing.assert_allclose(np.asarray(answer), r2, atol=2e-6, rtol=2e-6)
    for shard in answer.addressable_shards:
        # Transfer the device result to the host for an independent NumPy comparison.
        np.testing.assert_allclose(np.asarray(shard.data), r2, atol=2e-6, rtol=2e-6)
wrong_labels = 2 * xh.T @ (xh @ w2h - yh[::-1]) / 16
assert np.max(np.abs(wrong_labels-r2)) > 0.01
print('Joint permutation preserves the gradient; label-only reversal does not.')
```

</details>

## Account for the complete communication cycle

**Transfer**

Compare a complete ring all-reduce with reduce-scatter followed by all-gather for a gradient of one mebibyte at two, four and eight ranks. Also compute per-rank gradient storage. Explain why lower storage does not guarantee fewer communicated bytes.

<details><summary>Hint</summary>

Use the same full gradient byte count for both collectives; one mebibyte is 2**20 bytes. Divide storage by ranks only for the sharded state.

</details>

### How to write: Account for the complete communication cycle — Step-by-step recipe & starter scaffold

**Inputs:**
- A one-mebibyte gradient payload and rank counts 2, 4, and 8.
- A ring collective communication model.

**Outputs:**
- Per-rank storage and communicated bytes for each complete strategy.

**Invariants to check:**
- The all-reduce comparison includes both reduce-scatter and all-gather phases.
- Per-rank storage is distinct from network bytes transferred.

**Hints:**
- For a ring, account for (P-1) chunks in each phase.
- Keep byte units explicit and divide payload by rank count only for shard storage.

**Starter code scaffold (fill in the TODOs):**

```python
def compare_ring_collectives(payload_bytes, rank_counts):
    # TODO: calculate complete all-reduce and reduce-scatter/all-gather costs.
    # Return storage and communication in separately named fields.
    raise NotImplementedError("Count both ring phases and per-rank storage")
```

<details><summary>Reference solution and reasoning</summary>

```python
# Account for the complete communication cycle (Transfer)
budget = []
for p in [2, 4, 8]:
    size = 2**20
    rs = size * (p - 1) // p
    ar = 2 * rs
    cycle = rs + rs
    assert ar == cycle
    assert size // p * p == size
    budget.append({'ranks': p, 'all_reduce_bytes': ar,
                   'scatter_gather_bytes': cycle, 'shard_storage_bytes': size // p})
assert [row['all_reduce_bytes'] for row in budget] == [1048576, 1572864, 1835008]
print('Idealized payload/storage budget, not observed network traffic:', budget)
```

The two-phase payload equals the all-reduce payload under this model, even though the persistent sharded state is smaller. Timing and algorithm selection still need measurements on the complete target workload.

</details>

## Check your understanding

A program replaces all-reduce with reduce-scatter and immediately gathers the full vector again. What does the idealized ring model predict?

1. Half the communication of the original all-reduce.
2. The same two-phase payload, despite a temporarily smaller local result.
3. No communication because the global shape is unchanged.

<details><summary>Answer and explanation</summary>

**1. Half the communication of the original all-reduce.**

Count bytes in both collective phases. Replacing an all-reduce with reduce-scatter plus all-gather need not reduce the ideal ring payload.

**2. The same two-phase payload, despite a temporarily smaller local result. (correct)**

The scatter and gather each contribute one phase. The advantage depends on what consumes the partitioned value and when reconstruction is required.

**3. No communication because the global shape is unchanged.**

A global shape describes array semantics, not how replicas or shards become consistent. Communication still has to move the contributions.

</details>

## Diagnose the result

A gradient four times too large suggests local-mean normalization was summed. Correct global values with wrong local shapes suggest an output placement mismatch. A variation error in a replicated output needs a mathematical replication argument before changing the check.

## Carry forward

- Check both global values and local storage.
- Derive collective costs for the complete consumer path.
- Label ring payload arithmetic separately from measured CPU timing.

## Keep your evidence

Keep global NumPy gradient checks, local shard shapes, lowered collectives, changed-value replicas and complete scatter/gather payload accounting. Separate the idealized ring model from observed CPU timings.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX shard_map and collectives](https://docs.jax.dev/en/latest/notebooks/shard_map.html)
- [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html)

