# Compile a function with jit

Phase 02: JAX transformations · about 55 minutes · CPU

## What you will be able to do

- Explain tracing, compilation, dispatch, and synchronization as separate work.
- Validate a compiled numerical function against its eager and analytic results.
- Measure first-call latency and repeated-call latency with explicit completion.
- Compile a value-and-gradient computation without claiming a universal speedup.

## The problem

Your calculation is correct, and now you want to run it many times. What does compilation actually change? We’ll compare the original function with its jitted version, separate the first call from later calls, and wait for the result before stopping the timer. A small CPU example is enough to learn a fair comparison; a speedup is not guaranteed.

## The idea

Compilation prepares a program that can be reused for compatible calls. The first call can include tracing and compilation as well as execution; later compatible calls can reuse that work. Numerical equality and performance are separate checks.

## Separate changing data from preparing a program

Suppose you call a compiled function on a float32 vector, then call it on different values with the same shape and dtype. Those new values can flow through the prepared computation. If the next call changes a relevant shape or static choice, JAX may need a different specialization.

Think of the first-call path as preparation followed by execution. A warm-call timing should describe the repeated workload after preparation, and it must wait for device work to finish. Timing only dispatch can make a slow computation look fast.

First compare compiled and ordinary outputs on several inputs. Then, if speed is your question, state the input size, device, warmup and synchronization boundary. Three equal bars demonstrate agreement; they do not demonstrate acceleration.

### Pause and reason

Why can a tiny function be slower on its first compiled call?

<details><summary>Compare your reasoning</summary>

The call includes preparation costs that may exceed the work itself. Reuse can amortize those costs, but a speed claim still needs measurements of the intended workload.

</details>

## Separate the work hidden in one call

The first call can involve tracing, lowering, compilation, dispatch, and execution. Calling the same compiled function on compatible inputs can avoid much of the setup, but still involves a Python invocation, dispatch, and execution. A persistent cache or earlier compilation can also affect what first means.

Define the scope: first call to this callable in this process with these input shapes/dtypes. Do not label it cold-machine startup, pure compilation time, or kernel execution time unless your experiment actually isolates those quantities.

```text
first relevant call → trace / compile / dispatch / execute
compatible repeated call → reuse / dispatch / execute
```

## Check values before timing

Our input matrix has shape $(4, 3)$, weights have shape $(3,)$, and targets have shape $(4,)$. Predictions are $[0, 0.6, 1.2, 1.8]$. Residuals are $[-1, -0.4, 0.2, 0.8]$. Squaring and averaging yields $0.46$. Compute this independently so that two implementations agreeing on an unintended broadcast is not mistaken for correctness.

Compilation should preserve the intended numerical calculation within suitable tolerance. Floating-point transformations and reduction ordering can change last bits; exact equality is often too strong. Choose tolerance using dtype and scale, and do not suppress large discrepancies with a very loose allclose check.

## Wait for the quantity you are timing

JAX can return an array object while device computation is still pending. Timing only the Python call can therefore measure dispatch rather than completed work. block_until_ready waits for the array result without needing to print it or convert it to NumPy.

A function returning a tuple or a parameter tree needs a completion strategy for all relevant leaves. A scalar loss is convenient for this introductory experiment because waiting on it marks that result complete. Keep data creation and transfers outside the interval when measuring a prepared numerical invocation; include them deliberately when measuring end-to-end latency.

## Use repetitions and report a distribution

One repeated call is easy to distort with scheduling noise. Warm the exact callable and input signature, collect multiple completion-synchronized durations, and report a median plus range and repeat count. Use the same prepared arrays and comparable waiting behavior for eager and compiled calls.

This measures invocation latency under the stated conditions, including Python overhead. It is not a throughput measurement for queued training steps, and it does not isolate accelerator kernels. A median for a small toy workload cannot justify a performance claim about your future model or a different device.

## Compile the computation you intend to repeat

For training, loss and parameter gradients are often consumed together. `jit(value_and_grad(loss))` expresses a compiled transformed computation that returns both. Check the value against the loss and the gradient against an independent formula before using it as part of an update.

Avoid creating a fresh jitted lambda inside every iteration. Reusing a stable transformed callable gives compilation reuse a chance. A useful compilation boundary often surrounds a complete step rather than every scalar operation, but the correct boundary depends on the workload, state, and integration constraints. We will make state explicit before building that larger step.

## Decide when compilation is worth its setup

A long compilation can be worthwhile if a computation runs many times. A one-off small calculation might never recover its setup cost. You need both setup latency and steady-state measurements to reason about break-even behavior.

This lesson does not measure TPU execution, multi-device scaling, or a model training speedup. Its artifact is a reproducible correctness/timing report that records the backend, shapes, dtype, synchronization policy, warmup, repeat count, and observed durations.

## Run the example

```python
import time
import jax
import jax.numpy as jnp
def loss(w, x, y):
    return jnp.mean((x @ w - y) ** 2)
x = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10.
w = jnp.array([1., 2., -1.])
y = jnp.ones(4)
compiled = jax.jit(loss)
t0 = time.perf_counter()
first = compiled(w, x, y).block_until_ready()
first_seconds = time.perf_counter() - t0
t0 = time.perf_counter()
second = compiled(w, x, y).block_until_ready()
repeat_seconds = time.perf_counter() - t0
print("Loss:", float(second))
print("First/repeat seconds:", first_seconds, repeat_seconds)
assert jnp.allclose(first, loss(w, x, y))
assert jnp.allclose(second, first)
```

Expected: Loss ≈ $0.46$. Timings depend on the device, process, and workload; no speed threshold is required.

## Compilation preserves the result

**Predict:** Should compilation change the loss value?

![Compilation preserves the result](../../phases/02-transforms/04-compile-a-function-with-jit/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The three categories are eager execution, the first compiled call, and a repeated compiled call. The vertical axis is mean squared loss. Each bar reaches approximately $0.46$, so all three execution paths return the same numerical result for this input.

These are separate bars with equal heights. The first compiled call has not disappeared, and its bar is not a measurement of compilation time.

### Connect it to the computation

Compilation changes how the computation runs while preserving its intended result. The agreement here is the correctness check to make before asking whether compilation improved performance. Small floating-point differences should be assessed with the example’s numerical tolerance.

To measure speed, you need a time axis, synchronization, warm-up, and repeated measurements. None of those can be inferred from these loss bars. A useful next check is to change the input values and confirm that eager and compiled results still agree.

```python
visual_data = {'kind': 'bar', 'labels': ['eager', 'first compiled', 'repeat compiled'], 'ylabel': 'mean squared loss', 'series': [{'label': 'evaluated loss', 'y': [float(loss(w, x, y)), float(first), float(second)]}]}
```

## Recorded reference execution

CPU run: 2026-10-06T21:56:19.298722+00:00. JAX 0.9.2.

```text
Loss: 0.46000000834465027
First/repeat seconds: 0.023624374996870756 2.39592045545578e-05
eager median / min / max seconds: 3.9103906601667404e-05 3.7457793951034546e-05 6.258394569158554e-05
compiled median / min / max seconds: 6.333226338028908e-06 5.958136171102524e-06 8.458271622657776e-06
backend / shapes / dtype: cpu (4, 3) (3,) (4,) float32
compiled value-and-grad median seconds: 1.1312542483210564e-05
PASS: transforms-04

```

## Derive predictions and gradients outside the transform

**Predict before running:** Predict the loss $0.46$ and derive $\frac{2}{N}X^\mathsf{T}(Xw-y)$. Does the compiled result match both independent quantities?

```python
expected_predictions = jnp.array([0., 0.6, 1.2, 1.8])
assert jnp.allclose(x @ w, expected_predictions, atol=1e-6)
assert jnp.allclose(compiled(w, x, y), 0.46, atol=1e-6)
manual_gradient = (2. / len(y)) * x.T @ (x @ w - y)
compiled_value_gradient = jax.jit(jax.value_and_grad(loss))
checked_value, checked_gradient = compiled_value_gradient(w, x, y)
assert jnp.allclose(checked_gradient, manual_gradient, rtol=1e-5, atol=1e-6)
```

**Expected:** The loss is $0.46$; the gradient matches the independent matrix expression.

The analytic value check detects mistakes that eager-versus-compiled equality alone would miss.

## Collect a synchronized latency sample

**Predict before running:** Which costs are included in these measurements? Predict whether the tiny compiled loss must be faster, then report what you actually observe.

```python
import statistics
def completed_samples(function, repeats=20):
    function(w, x, y).block_until_ready()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(w, x, y).block_until_ready()
        durations.append(time.perf_counter() - start)
    return durations
for name, function in (("eager", loss), ("compiled", compiled)):
    samples = completed_samples(function)
    print(name, "median / min / max seconds:", statistics.median(samples), min(samples), max(samples))
print("backend / shapes / dtype:", jax.default_backend(), x.shape, w.shape, y.shape, x.dtype)
```

**Expected:** Twenty completed calls are measured after a warmup for each callable. Durations vary; no speed threshold is asserted.

The report includes Python invocation and completed numerical work on already prepared data. It is a bounded latency observation, not a universal accelerator benchmark.

## Make it yours

Foundation · Compile `value_and_grad(loss)`, check the loss and gradient against both the eager transform and the manual matrix formula, and explain how you would synchronize a benchmark of its two outputs.

<details><summary>Reference solution</summary>

```python
compiled_step = jax.jit(jax.value_and_grad(loss))
v, g = compiled_step(w, x, y)
expected_v, expected_g = jax.value_and_grad(loss)(w, x, y)
assert jnp.allclose(v, expected_v)
assert jnp.allclose(g, expected_g)
```

</details>

## Check that a new value remains runtime data

**Practice**

Keep shapes and dtypes fixed, change $w$, and compare the compiled loss against an independently computed residual mean. Explain why a different numerical answer does not itself imply a new compilation.

<details><summary>Hint</summary>

Compilation specializes to relevant properties, not each ordinary array element value.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
for new_w in (w, w + 0.25):
    expected = jnp.mean((x @ new_w - y) ** 2)
    observed = compiled(new_w, x, y)
    assert jnp.allclose(observed, expected, rtol=1e-5, atol=1e-6)
```

The same computation accepts different runtime values. The next lesson inspects tracing explicitly rather than inferring it from timing noise.

</details>

## Build a report whose timing claim can be checked

**Challenge**

Benchmark the compiled value-and-gradient function for $20$ calls after warmup, waiting on every output leaf. Include backend, input shapes, dtype, and median. State what your measurement excludes.

<details><summary>Hint</summary>

Use `jax.tree.map` to block on every returned array, then stop the timer.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
def wait_for_outputs(outputs):
    return jax.tree.map(lambda array: array.block_until_ready(), outputs)
wait_for_outputs(compiled_value_gradient(w, x, y))
gradient_times = []
for _ in range(20):
    started = time.perf_counter()
    wait_for_outputs(compiled_value_gradient(w, x, y))
    gradient_times.append(time.perf_counter() - started)
print("compiled value-and-grad median seconds:", statistics.median(gradient_times))
assert len(gradient_times) == 20
```

The measurement excludes input construction and first compilation, and includes repeated invocation and completion. It cannot be compared with another report without matching those boundaries.

</details>

## Check your understanding

Why wait with block_until_ready when timing a computation?

1. To change its gradient
2. To include completion of device work in elapsed time
3. To force a different result

<details><summary>Answer and explanation</summary>

To include completion of device work in elapsed time

Without waiting, elapsed time can primarily measure dispatch rather than the completion of the numerical work.

</details>

## Diagnose the result

A suspiciously tiny time may measure dispatch only; inspect synchronization. A slow first call followed by faster calls may reflect setup, but timing alone does not identify which setup stage. Inconsistent eager/compiled results call for shape, dtype, and tolerance checks before performance tuning. Repeatedly slow setup can come from recreating transformed functions or changing specialization inputs; the next lesson tests those hypotheses directly.

## Carry forward

- Compilation setup and repeated invocation are different measurements.
- Correctness needs independent numerical checks before performance interpretation.
- Asynchronous execution requires completion synchronization.
- A useful report records workload, backend, dtype, warmup, repetitions, and timing boundaries.

## Keep your evidence

Keep the independent loss/gradient calculation, backend/shapes/dtype, first-call interval, warmup/repeat policy, median/range samples, and synchronization of every output leaf. State what the benchmark excludes.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [JAX: just-in-time compilation](https://docs.jax.dev/en/latest/jit-compilation.html)
- [JAX: benchmarking](https://docs.jax.dev/en/latest/benchmarking.html)

