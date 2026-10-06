# A sharded training step

Phase 09: Distributed training · about 90 minutes · 4 logical CPU devices

## What you will be able to do

- Derive the global mean loss and gradient.
- Specify aligned data and replicated parameter placement.
- Verify sharded updates against independent host arithmetic.
- Diagnose normalization errors in equal and unequal local batches.

## The problem

If we split a batch across four devices, should the model learn four times as much in one step? It should still follow the same global loss. We’ll compare a sharded regression update with a formula we can check on the host, then reproduce the scaling mistake caused by summing local mean gradients.

## The idea

Data parallelism partitions examples while replicating small model parameters. Every shard contributes to the same global objective. Here the loss is a global mean over eight observations, so the gradient is $\frac{2}{8}X^\mathsf{T}(Xw-y)$. Automatic differentiation of the global sharded-array program preserves that mathematical objective; explicit placement defines where its data and results live.

## Write the objective before splitting the data

Our model predicts $Xw$ with two weights and no bias. The synthetic labels use true weights $[2,-1]$, but the initial weights are zero. The squared-error loss averages eight residuals. Its gradient is $\frac{2}{N}X^\mathsf{T}r$, where $r=Xw-y$ and $N=8$. This host formula is independent of JAX autodiff and supplies a useful reference for the first update.

At $w=0$ the targets are $[2,-1,1,4,-2,3,0,2]$. Squared targets sum to $39$, so initial loss is $39/8=4.875$. $X^\mathsf{T}y=[21,3]$, so the gradient is $[-5.25,-0.75]$. With learning rate $0.1$ the new weights are $[0.525,0.075]$. Check these values before looking at a training curve.

$$
\begin{aligned}L(w)&=\frac{1}{8}\sum_{i=1}^{8}(x_i^\mathsf{T}w-y_i)^2\\g&=\frac{2}{8}X^\mathsf{T}(Xw-y)\\w_{\mathrm{next}}&=w-0.1g\end{aligned}
$$

## Give batch and parameters different placements

The feature matrix has shape $(8,2)$ and uses `P("data",None)`; each of four devices owns two examples. Labels have shape $(8,)$ and use `P("data")`, so their shards align with the feature rows. Parameters have shape $(2,)$ and use `P()`, giving every device both weights.

The input sharding tree supplied to jit has one entry per argument. Outputs are weights, scalar loss and a two-element gradient; all three are replicated in this small example. Replication does not mean four independently trained models. The replicas represent the same global parameters and must receive the same global update.

```text
CPU 0: X rows 0:2, y rows 0:2, full w
CPU 1: X rows 2:4, y rows 2:4, full w
CPU 2: X rows 4:6, y rows 4:6, full w
CPU 3: X rows 6:8, y rows 6:8, full w
partial contributions → global gradient → same w_next
```

## Differentiate the global function rather than inventing a local objective

loss operates on global JAX arrays. value_and_grad evaluates the objective and obtains its parameter derivative together. jit compiles that function with explicit placement contracts; you do not call a Python training loop separately on each device. Combining contributions is necessary because each local batch alone cannot determine the global mean gradient.

This lesson uses automatic global-array parallelization, not manual shard_map or an explicit pmean. The compiler arranges required operations. Do not add an extra collective to an already global reduction merely because the input is sharded. If you later implement a manual local program, you must derive its aggregation and normalization from the same objective.

## A correct first update is stronger evidence than a falling curve

A loss curve can decrease even with a normalization bug, especially at a small learning rate. Compare the initial loss, every gradient component and updated weights with NumPy. Then change the batch contents and order: a correct global mean should not depend on which device receives a row. Check replicated weight shards separately.

A ten-step loop is included as a transfer experiment, but this is not a complete resilient training system. It has no data pipeline, optimizer moments, checkpoint or multi-host failure handling. CPU virtual devices demonstrate placement and numerical equivalence; accelerator efficiency requires measurement on that accelerator. Float32 reduction order may differ, so use stated tolerances instead of bitwise comparisons.

## Start a fresh CPU runtime

Create main.py in your course workspace. Paste this block first. If using a notebook, restart its kernel before running any cell.

```python
import jax
# Run this file in a fresh process; in a notebook restart the kernel first.
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 4)
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
devices = jax.devices("cpu")
if len(devices) != 4:
    raise RuntimeError("Expected four CPU devices. Restart the notebook kernel, then Run all; do not run an earlier JAX cell first.")
mesh = Mesh(np.array(devices), ("data",))
rows = NamedSharding(mesh, P("data", None))
replicated = NamedSharding(mesh, P())
```

Configuration happens before `devices()` or array creation initializes a backend. This file explicitly chooses CPU even on a GPU machine.

## Place and compute

Append this block in the same file. Predict the shapes and values before running.

```python
host_x = np.array([[1,0],[0,1],[1,1],[2,0],[0,2],[2,1],[1,2],[2,2]],dtype=np.float32)
true_w = np.array([2.,-1.],dtype=np.float32)
host_y = host_x @ true_w
labels_sharding = NamedSharding(mesh,P("data"))
x = jax.device_put(host_x, rows)
y = jax.device_put(host_y, labels_sharding)
w = jax.device_put(np.zeros(2,dtype=np.float32),replicated)
def loss(weights, features, labels):
    return jnp.mean((features @ weights-labels)**2)
def update(weights, features, labels):
    value, gradient = jax.value_and_grad(loss)(weights,features,labels)
    return weights - 0.1*gradient, value, gradient
train_step = jax.jit(update, in_shardings=(replicated,rows,labels_sharding),out_shardings=(replicated,replicated,replicated))
new_w, before, gradient = train_step(w,x,y)
```

Mesh axis names describe placement. They are separate from the numerical array dimensions.

## Inspect and verify

Append the checks, save the file and run python main.py with the setup lesson environment.

```python
expected_gradient = 2 * host_x.T @ (-host_y) / len(host_y)
np.testing.assert_allclose(np.asarray(gradient),expected_gradient,rtol=1e-6,atol=1e-6)
np.testing.assert_allclose(np.asarray(new_w),-0.1*expected_gradient,rtol=1e-6,atol=1e-6)
expected_loss = np.mean(host_y**2)
np.testing.assert_allclose(np.asarray(before), expected_loss,rtol=1e-6)
assert float(loss(new_w,x,y)) < float(before)
for shard in new_w.addressable_shards:
    np.testing.assert_allclose(np.asarray(shard.data),np.asarray(new_w))
print("Initial loss:",float(before))
print("Gradient:",np.asarray(gradient),"updated weights:",np.asarray(new_w))
print("Updated loss:",float(loss(new_w,x,y)))
```

Assertions compare against host calculations or hand-derived values; printing a sharding object alone does not establish correctness.

## Run the example

```python
import jax
# Run this file in a fresh process; in a notebook restart the kernel first.
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 4)
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
devices = jax.devices("cpu")
if len(devices) != 4:
    raise RuntimeError("Expected four CPU devices. Restart the notebook kernel, then Run all; do not run an earlier JAX cell first.")
mesh = Mesh(np.array(devices), ("data",))
rows = NamedSharding(mesh, P("data", None))
replicated = NamedSharding(mesh, P())

host_x = np.array([[1,0],[0,1],[1,1],[2,0],[0,2],[2,1],[1,2],[2,2]],dtype=np.float32)
true_w = np.array([2.,-1.],dtype=np.float32)
host_y = host_x @ true_w
labels_sharding = NamedSharding(mesh,P("data"))
x = jax.device_put(host_x, rows)
y = jax.device_put(host_y, labels_sharding)
w = jax.device_put(np.zeros(2,dtype=np.float32),replicated)
def loss(weights, features, labels):
    return jnp.mean((features @ weights-labels)**2)
def update(weights, features, labels):
    value, gradient = jax.value_and_grad(loss)(weights,features,labels)
    return weights - 0.1*gradient, value, gradient
train_step = jax.jit(update, in_shardings=(replicated,rows,labels_sharding),out_shardings=(replicated,replicated,replicated))
new_w, before, gradient = train_step(w,x,y)

expected_gradient = 2 * host_x.T @ (-host_y) / len(host_y)
np.testing.assert_allclose(np.asarray(gradient),expected_gradient,rtol=1e-6,atol=1e-6)
np.testing.assert_allclose(np.asarray(new_w),-0.1*expected_gradient,rtol=1e-6,atol=1e-6)
expected_loss = np.mean(host_y**2)
np.testing.assert_allclose(np.asarray(before), expected_loss,rtol=1e-6)
assert float(loss(new_w,x,y)) < float(before)
for shard in new_w.addressable_shards:
    np.testing.assert_allclose(np.asarray(shard.data),np.asarray(new_w))
print("Initial loss:",float(before))
print("Gradient:",np.asarray(gradient),"updated weights:",np.asarray(new_w))
print("Updated loss:",float(loss(new_w,x,y)))
```

Expected: Initial loss $4.875$; gradient $[-5.25,-0.75]$; weights $[0.525,0.075]$; updated loss is lower. Small float32 printing differences are expected.

## Sharded differentiation matches the global reference

**Predict:** Should placing examples on different devices change the gradient?

![Sharded differentiation matches the global reference](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

For each weight coordinate, compare the sharded JAX gradient with the NumPy reference gradient beside it. Both methods give approximately $(-5.25,-0.75)$, so the paired bars have equal heights. They remain separate bars even where their edges touch.

The bars extend below zero because the derivatives are negative. This is not a plot of negative loss. The larger magnitude for weight $0$ indicates a stronger local sensitivity of the loss to that coordinate.

### Connect it to the computation

A gradient descent step subtracts these values. From zero weights with rate $0.1$, the next weights would be $(0.525,0.075)$. This connects the signs in the picture to an actual update rather than treating taller or shorter bars as inherently better.

Agreement with the global reference checks that sharding and reduction preserved the intended gradient on this batch. It does not establish a speedup, or imply that a coordinate’s current gradient sign reveals its final optimal weight. Input correlations and subsequent updates can change that sign.

```python
visual_data = {'kind': 'bar', 'labels': ['weight 0', 'weight 1'], 'ylabel': 'gradient coordinate', 'series': [{'label': 'sharded JAX', 'y': np.asarray(gradient).tolist()}, {'label': 'NumPy reference', 'y': expected_gradient.tolist()}]}
```

## Recorded reference execution

CPU run: 2026-10-06T01:24:35.449035+00:00. JAX 0.9.2.

```text
Initial loss: 4.875
Gradient: [-5.25 -0.75] updated weights: [0.52500004 0.075     ]
Updated loss: 2.6784372329711914
Initial loss: 4.875
Gradient: [-5.25 -0.75] updated weights: [0.52500004 0.075     ]
Updated loss: 2.6784372329711914
Permutation preserves the global update
Ten-step weights: [ 1.704636  -0.7047408]
Summed local means are four times too large; weighted aggregation repaired
Summed local means are four times too large; weighted aggregation repaired
PASS: distributed-02

```

## Permutation across shards

**Predict before running:** Reversing example order changes which device owns each example. Should the global gradient change?

```python
reverse_x = jax.device_put(host_x[::-1].copy(),rows)
reverse_y = jax.device_put(host_y[::-1].copy(),labels_sharding)
reverse_w,reverse_loss,reverse_gradient = train_step(w,reverse_x,reverse_y)
np.testing.assert_allclose(np.asarray(reverse_gradient),expected_gradient,rtol=1e-6,atol=1e-6)
np.testing.assert_allclose(np.asarray(reverse_w),np.asarray(new_w),rtol=1e-6,atol=1e-6)
print("Permutation preserves the global update")
```

**Expected:** Gradient and update agree within $10^{-6}$.

The global mean is invariant to row order. This check exposes accidental dependence on a particular local subset.

## Ten updates against a host training loop

**Predict before running:** Predict whether the two implementations should track each other, not whether loss reaches zero in ten steps.

```python
cpu_weights = w
reference_weights = np.zeros(2,dtype=np.float32)
for step_index in range(10):
    cpu_weights,_,_ = train_step(cpu_weights,x,y)
    residual = host_x @ reference_weights-host_y
    reference_weights -= 0.1*(2*host_x.T @ residual/len(host_y))
    np.testing.assert_allclose(np.asarray(cpu_weights),reference_weights,rtol=1e-5,atol=1e-6)
print("Ten-step weights:",np.asarray(cpu_weights))
```

**Expected:** Each of ten parameter updates agrees with the independent host loop within the stated tolerance.

Checking every update catches transient mismatches that a single final loss comparison could hide.

## Make it yours

Run a changed target/initial-state update, then diagnose the factor-of-four local-mean bug and repair uneven-batch aggregation.

<details><summary>Reference solution</summary>

```python
changed_y = host_x @ np.array([-1.,3.],dtype=np.float32)
start = np.array([0.2,-0.1],dtype=np.float32)
def changed_update(weights,features,labels):
    return weights-0.05*jax.grad(loss)(weights,features,labels)
changed_step = jax.jit(changed_update,in_shardings=(replicated,rows,labels_sharding),out_shardings=replicated)
result = changed_step(jax.device_put(start,replicated),x,jax.device_put(changed_y,labels_sharding))
reference_gradient = 2*host_x.T @ (host_x@start-changed_y)/len(changed_y)
np.testing.assert_allclose(np.asarray(result),start-0.05*reference_gradient,rtol=1e-6,atol=1e-6)

local_gradients = []
for features,labels in zip(np.split(host_x,4),np.split(host_y,4)):
    local_gradients.append(2*features.T @ (-labels)/len(labels))
wrong = np.sum(local_gradients,axis=0)
right = np.mean(local_gradients,axis=0)
np.testing.assert_allclose(wrong,4*expected_gradient,rtol=1e-6)
np.testing.assert_allclose(right,expected_gradient,rtol=1e-6)
# Unequal partitions: weight local mean gradients by their observation counts.
parts = [(host_x[:3],host_y[:3]),(host_x[3:],host_y[3:])]
weighted = sum(len(b)*(2*a.T @ (-b)/len(b)) for a,b in parts)/len(host_y)
np.testing.assert_allclose(weighted,expected_gradient,rtol=1e-6)
print("Summed local means are four times too large; weighted aggregation repaired")
```

</details>

## Transfer to a different target model

**Challenge**

Change true weights to $[-1,3]$, start at $[0.2,-0.1]$ and use learning rate $0.05$. Derive and check the new update without reusing the original numeric answer.

<details><summary>Hint</summary>

The original train_step hardcodes $0.1$; define a new step with the changed rate.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
changed_y = host_x @ np.array([-1.,3.],dtype=np.float32)
start = np.array([0.2,-0.1],dtype=np.float32)
def changed_update(weights,features,labels):
    return weights-0.05*jax.grad(loss)(weights,features,labels)
changed_step = jax.jit(changed_update,in_shardings=(replicated,rows,labels_sharding),out_shardings=replicated)
result = changed_step(jax.device_put(start,replicated),x,jax.device_put(changed_y,labels_sharding))
reference_gradient = 2*host_x.T @ (host_x@start-changed_y)/len(changed_y)
np.testing.assert_allclose(np.asarray(result),start-0.05*reference_gradient,rtol=1e-6,atol=1e-6)
```

The independent formula follows the new target, initial state and learning rate. Reusing a hard-coded expected weight vector would not test transfer.

</details>

## Diagnose summed local means

**Challenge**

Compute four host local mean gradients. Show why summing them gives four times the global mean, then repair the aggregation. Also explain the unequal-batch generalization.

<details><summary>Hint</summary>

A mean of equal-size local means works; unequal sizes require weighting by local counts.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
local_gradients = []
for features,labels in zip(np.split(host_x,4),np.split(host_y,4)):
    local_gradients.append(2*features.T @ (-labels)/len(labels))
wrong = np.sum(local_gradients,axis=0)
right = np.mean(local_gradients,axis=0)
np.testing.assert_allclose(wrong,4*expected_gradient,rtol=1e-6)
np.testing.assert_allclose(right,expected_gradient,rtol=1e-6)
# Unequal partitions: weight local mean gradients by their observation counts.
parts = [(host_x[:3],host_y[:3]),(host_x[3:],host_y[3:])]
weighted = sum(len(b)*(2*a.T @ (-b)/len(b)) for a,b in parts)/len(host_y)
np.testing.assert_allclose(weighted,expected_gradient,rtol=1e-6)
print("Summed local means are four times too large; weighted aggregation repaired")
```

Each local gradient already divides by its local count. Summing four equal local means omits division by four. For unequal counts, an unweighted mean also misrepresents examples; sum count-weighted local means and divide by total count.

</details>

## Check your understanding

Four equal-size shards compute local mean gradients. Which aggregation gives the global mean gradient?

1. Sum them without scaling.
2. Average them, or sum local gradient numerators and divide by total examples.
3. Choose the device with the lowest local loss.

<details><summary>Answer and explanation</summary>

Average them, or sum local gradient numerators and divide by total examples.

A global mean gives each observation equal weight. With equal local counts, averaging local means preserves that weighting.

</details>

## Diagnose the result

Each local gradient already divides by its local count. Summing four equal local means omits division by four. For unequal counts, an unweighted mean also misrepresents examples; sum count-weighted local means and divide by total count.

## Carry forward

- Write the objective before splitting the data
- Give batch and parameters different placements
- Differentiate the global function rather than inventing a local objective
- A correct first update is stronger evidence than a falling curve

## Keep your evidence

Each local gradient already divides by its local count. Summing four equal local means omits division by four. For unequal counts, an unweighted mean also misrepresents examples; sum count-weighted local means and divide by total count.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [JAX CPU device configuration (set before initialization)](https://docs.jax.dev/en/latest/config_options.html#num-cpu-devices)
- [Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/201/sharding.html)

