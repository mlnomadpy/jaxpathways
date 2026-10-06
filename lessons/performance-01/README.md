# Benchmark asynchronous work correctly

Phase 08: Performance diagnosis · about 80 minutes · CPU

## What you will be able to do

- Draw the interval a timing actually measures.
- Exclude initial placement from a device-resident benchmark.
- Check an independent NumPy result before comparing timings.
- Produce a report containing first-call and warmed sample distributions.

## The problem

A first timing suggests a tenfold speedup. Before celebrating, let’s ask when the timer started and stopped. We’ll separate the first call from warm calls, wait for JAX to finish, and compare numerical results. By the end, you’ll have a timing report another learner can reproduce.

## The idea

An array handle and a completed calculation are different events. JAX may dispatch work asynchronously; block_until_ready waits for the requested result without requiring a NumPy copy. A first call can also include tracing, lowering and compilation. Time first-call latency separately from repeated calls to a warmed function.

This lesson explicitly places a modest matrix workload on CPU. It teaches measurement design and numerical equivalence, not an accelerator speed claim. The JSON report retains every measured sample, the workload, the environment and the synchronization boundary.

## Draw the boundary before starting the clock

For our workload, the input has shape $(64, 32)$, the weights have shape $(32, 16)$, and the result has shape $(64, 16)$. Matrix multiplication contracts the feature dimension, then tanh transforms each prediction. The timer starts after both input arrays are placed on CPU and ready. It stops after the output is ready.

That interval includes Python dispatch, execution and the final wait. It is not an isolated kernel duration. Data generation, initial placement and output conversion to NumPy are outside it. A serving request or data-loading benchmark would deliberately include different boundaries; specify which question your measurement answers.

```text
prepare host arrays → place → wait | start → call → result handle → wait → stop | inspect host values
```

## A handle is not proof of completion

Python can receive a JAX array before its contents finish computing. Reading shape or dtype can use metadata without reading the values. Calling block_until_ready on the result waits for readiness. Converting it to NumPy also requires the values, but adds a host-materialization boundary.

CPU dispatch behavior and workload size may make the difference difficult to see. If a handle-only interval is close to a synchronized interval on this machine, that does not justify removing synchronization from a portable benchmark. Timing two independent calls cannot be subtracted to obtain an exact queue or execution cost.

## First call and steady state answer different questions

A fresh jitted function can trace its Python body, lower the resulting program, obtain an executable and run it. The first-call timer combines these stages. Persistent caches and prior process activity can alter the result, so call it first-call latency for this process rather than an exact compilation measurement.

We then run seven synchronized repeats with identical shapes and dtypes. The list reveals variation; the median gives a compact center. Seven samples are an instructional start, not a confidence interval. Increase repetitions and document contention, power state and process initialization before drawing deployment conclusions.

## Correctness comes before a speed ratio

The NumPy reference uses the same float32 inputs and computes the same matrix product and tanh. Floating-point reduction order can vary, so the comparison permits small absolute and relative differences rather than demanding bit equality. We record maximum absolute error as well.

Comparing a float64 reference against an unreported float32 JAX computation, or comparing different matrix sizes, would mix numerical work with implementation differences. A faster answer to a different problem is not evidence of optimization. Retain both workload contracts and both checks in the report.

## Prepare and place the workload

Create benchmark.py in your activated course environment and add this block.

```python
import json
import platform
import time
import statistics
import numpy as np
import jax
import jax.numpy as jnp
cpu = jax.devices("cpu")[0]
rng = np.random.default_rng(4)
x_host = rng.normal(size=(64, 32)).astype(np.float32)
w_host = rng.normal(size=(32, 16)).astype(np.float32)
x = jax.device_put(x_host, cpu)
w = jax.device_put(w_host, cpu)
jax.block_until_ready((x, w))

```

CPU placement and waiting occur before timing. The deterministic host inputs make the numerical check reproducible.

## Define the contract and timer

Append this block to the same file. Run the assembled file with python benchmark.py.

```python
def predict(x, w):
    return jnp.tanh(x @ w)
compiled_predict = jax.jit(predict)
def synchronized_samples(fn, args, repeats=7):
    if repeats < 1:
        raise ValueError("at least one repeat is required")
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return result, samples

```

The timer accepts arbitrary result pytrees because `jax.block_until_ready` waits across their array leaves. It collects observations without asserting which implementation is faster.

## Separate first call from warm calls

Append this block to the same file. Run the assembled file with python benchmark.py.

```python
start = time.perf_counter()
first = compiled_predict(x, w)
first.block_until_ready()
first_call_seconds = time.perf_counter() - start
result, samples = synchronized_samples(compiled_predict, (x, w))
reference = np.tanh(x_host @ w_host)
np.testing.assert_allclose(np.asarray(result), reference, atol=2e-5, rtol=2e-5)
report = {
    "backend": str(cpu), "jax": jax.__version__, "numpy": np.__version__,
    "python": platform.python_version(), "input_shape": list(x.shape),
    "weight_shape": list(w.shape), "dtype": str(x.dtype),
    "boundary": "input already placed; call plus output synchronization",
    "first_call_seconds": first_call_seconds,
    "warm_samples_seconds": samples, "warm_median_seconds": statistics.median(samples),
    "max_abs_error": float(np.max(np.abs(np.asarray(result) - reference))),
}
assert len(samples) == 7 and all(t >= 0 for t in samples)
print(json.dumps(report, indent=2))

```

The report prints variable measured times. Redirect the full run output to a text file and keep the benchmark script with it.

## Run the example

```python
import json
import platform
import time
import statistics
import numpy as np
import jax
import jax.numpy as jnp
cpu = jax.devices("cpu")[0]
rng = np.random.default_rng(4)
x_host = rng.normal(size=(64, 32)).astype(np.float32)
w_host = rng.normal(size=(32, 16)).astype(np.float32)
x = jax.device_put(x_host, cpu)
w = jax.device_put(w_host, cpu)
jax.block_until_ready((x, w))

def predict(x, w):
    return jnp.tanh(x @ w)
compiled_predict = jax.jit(predict)
def synchronized_samples(fn, args, repeats=7):
    if repeats < 1:
        raise ValueError("at least one repeat is required")
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return result, samples

start = time.perf_counter()
first = compiled_predict(x, w)
first.block_until_ready()
first_call_seconds = time.perf_counter() - start
result, samples = synchronized_samples(compiled_predict, (x, w))
reference = np.tanh(x_host @ w_host)
np.testing.assert_allclose(np.asarray(result), reference, atol=2e-5, rtol=2e-5)
report = {
    "backend": str(cpu), "jax": jax.__version__, "numpy": np.__version__,
    "python": platform.python_version(), "input_shape": list(x.shape),
    "weight_shape": list(w.shape), "dtype": str(x.dtype),
    "boundary": "input already placed; call plus output synchronization",
    "first_call_seconds": first_call_seconds,
    "warm_samples_seconds": samples, "warm_median_seconds": statistics.median(samples),
    "max_abs_error": float(np.max(np.abs(np.asarray(result) - reference))),
}
assert len(samples) == 7 and all(t >= 0 for t in samples)
print(json.dumps(report, indent=2))

```

Expected: A JSON benchmark report: input_shape $[64, 32]$, weight_shape $[32, 16]$, dtype float32, seven nonnegative warmed samples, and a checked maximum error. Times vary by run; there is no required speedup.

## Separate the first call from warm measurements

**Predict:** Why should the first call not be averaged into steady-state latency?

![Separate the first call from warm measurements](../../phases/08-performance/01-benchmark-asynchronous-work-correctly/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The upper panel measures the first synchronized call. The lower panel measures repeated warm calls, with sample number on the horizontal axis. Both report seconds, but they have independent vertical scales. Compare the tick values and any scientific-notation multiplier, not the physical heights of the panels.

The recorded first call is much longer than the warm calls. For orientation, $10^{-2}$ seconds is $10$ milliseconds, while $10^{-5}$ seconds is $10$ microseconds. A multiplier printed beside an axis applies to all its tick labels; overlooking it can hide a difference of orders of magnitude.

### Connect it to the computation

The first measurement includes work needed to prepare this specialization, including tracing and compilation. Compatible warm calls reuse it. Each measurement waits for the result, so it includes completed computation rather than only the time needed to dispatch work. Inputs are already placed before this measured boundary.

The warm line shows variation between individual samples. An isolated higher point is a slower sample, not proof of recompilation or a changing model. Read the recorded median alongside the full sample list, and rerun before comparing implementations. These measurements describe this small CPU workload; they do not establish accelerator speed or end-to-end request latency.

```python
visual_data = {'kind': 'panels', 'panels': [{'kind': 'bar', 'title': 'First call', 'labels': ['first'], 'ylabel': 'seconds', 'series': [{'label': 'synchronized call', 'y': [first_call_seconds]}]}, {'kind': 'line', 'title': 'Warm calls', 'x': list(range(1, len(samples) + 1)), 'xlabel': 'warm sample', 'ylabel': 'seconds', 'series': [{'label': 'synchronized call', 'y': samples}]}]}
```

## Recorded reference execution

CPU run: 2026-10-06T01:24:27.970616+00:00. JAX 0.9.2.

```text
{
  "backend": "TFRT_CPU_0",
  "jax": "0.9.2",
  "numpy": "2.4.4",
  "python": "3.14.3",
  "input_shape": [
    64,
    32
  ],
  "weight_shape": [
    32,
    16
  ],
  "dtype": "float32",
  "boundary": "input already placed; call plus output synchronization",
  "first_call_seconds": 0.027212874963879585,
  "warm_samples_seconds": [
    3.5542063415050507e-05,
    1.1541880667209625e-05,
    8.417060598731041e-06,
    9.082956239581108e-06,
    8.499948307871819e-06,
    8.125090971589088e-06,
    8.041970431804657e-06
  ],
  "warm_median_seconds": 8.499948307871819e-06,
  "max_abs_error": 2.384185791015625e-07
}
{
  "backend": "TFRT_CPU_0",
  "jax": "0.9.2",
  "numpy": "2.4.4",
  "python": "3.14.3",
  "input_shape": [
    64,
    32
  ],
  "weight_shape": [
    32,
    16
  ],
  "dtype": "float32",
  "boundary": "input already placed; call plus output synchronization",
  "first_call_seconds": 0.017281959066167474,
  "warm_samples_seconds": [
    2.7082860469818115e-05,
    9.00006853044033e-06,
    6.542075425386429e-06,
    9.499955922365189e-06,
    7.209135219454765e-06,
    7.58306123316288e-06,
    8.040806278586388e-06
  ],
  "warm_median_seconds": 8.040806278586388e-06,
  "max_abs_error": 2.384185791015625e-07
}
Handle-only seconds: 7.166992872953415e-06
Complete-call samples: [3.6999816074967384e-05, 2.3250002413988113e-05, 1.4124903827905655e-05]
Structured samples: [3.1624920666217804e-05, 1.9916798919439316e-05, 1.7666956409811974e-05]
Changed workload: (96, 32) [2.9959017410874367e-05, 1.3417098671197891e-05, 8.292030543088913e-06, 1.0833144187927246e-05, 1.033279113471508e-05]
Energy shape and samples: (64,) [2.8125010430812836e-05, 1.3999873772263527e-05, 8.916947990655899e-06, 8.999835699796677e-06, 1.1499971151351929e-05]
{'boundary': 'placed inputs; warm call plus output wait', 'samples': [1.2666918337345123e-05, 8.417060598731041e-06, 8.00006091594696e-06, 9.041046723723412e-06, 8.041039109230042e-06]}
PASS: performance-01

```

## Compare handle-only and synchronized boundaries

**Predict before running:** Will the handle-only number necessarily be smaller on this CPU? Which interval guarantees output completion?

```python
start = time.perf_counter()
handle = compiled_predict(x, w)
handle_seconds = time.perf_counter() - start
handle.block_until_ready()  # drain this call before the separate comparison
_, complete_samples = synchronized_samples(compiled_predict, (x, w), repeats=3)
print("Handle-only seconds:", handle_seconds)
print("Complete-call samples:", complete_samples)

```

**Expected:** One handle-only interval and three complete-call intervals, all observed rather than fixed.

The separate intervals have different measurement boundaries and noise. Only synchronized samples guarantee completed output before the timer stops; no strict timing ordering is asserted.

## Return structured results without host copies

**Predict before running:** If a function returns predictions and their mean, does waiting for the prediction alone clearly document the readiness of every reported result?

```python
structured = jax.jit(lambda a, b: {"prediction": jnp.tanh(a @ b), "mean": jnp.mean(jnp.tanh(a @ b))})
jax.block_until_ready(structured(x, w))
structured_result, structured_times = synchronized_samples(structured, (x, w), 3)
np.testing.assert_allclose(np.asarray(structured_result["prediction"]), reference, atol=2e-5, rtol=2e-5)
np.testing.assert_allclose(np.asarray(structured_result["mean"]), reference.mean(), atol=2e-5, rtol=2e-5)
print("Structured samples:", structured_times)

```

**Expected:** Three timings and independently verified prediction/mean leaves.

A tree-level wait makes the benchmark contract explicit for every array leaf. Host inspection occurs after the timer, so the report describes device-ready computation rather than transfer-inclusive output consumption.

## Make it yours

Repeat the benchmark with $96$ rows while preserving the feature and output dimensions. Keep a separate report rather than overwriting the original workload label.

<details><summary>Reference solution</summary>

```python
larger_host = rng.normal(size=(96, 32)).astype(np.float32)
larger = jax.device_put(larger_host, cpu)
larger.block_until_ready()
compiled_predict(larger, w).block_until_ready()
larger_result, larger_times = synchronized_samples(compiled_predict, (larger, w), 5)
np.testing.assert_allclose(np.asarray(larger_result), np.tanh(larger_host @ w_host), atol=2e-5, rtol=2e-5)
print("Changed workload:", larger.shape, larger_times)

```

</details>

## Benchmark a different equivalent workload

**Transfer**

Replace tanh with a row-wise sum of squared predictions, returning one scalar per row. Derive the NumPy reference, warm this new function, collect five synchronized samples and record the changed output shape.

<details><summary>Hint</summary>

Square before reducing the output-feature axis; the result has shape $(64,)$.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
energy = jax.jit(lambda a, b: jnp.sum((a @ b)**2, axis=1))
energy(x, w).block_until_ready()
energy_result, energy_times = synchronized_samples(energy, (x, w), 5)
energy_reference = np.sum((x_host @ w_host)**2, axis=1)
assert energy_result.shape == (64,)
np.testing.assert_allclose(np.asarray(energy_result), energy_reference, atol=2e-3, rtol=2e-5)
print("Energy shape and samples:", energy_result.shape, energy_times)

```

Changing the computation changes the performance question. The shape and independent numerical check prevent a benchmark from accidentally timing an unrelated scalar reduction.

</details>

## Repair a timer that includes setup but omits completion

**Diagnosis**

Inspect the broken timer below conceptually: start; create inputs; call `jit(predict)(inputs)`; stop. Write a repaired runner and show its explicit boundary. Explain which stages the broken interval mixes and misses.

<details><summary>Hint</summary>

Prepare and wait for inputs, reuse one compiled callable, warm it, then synchronize the returned output inside every measured interval.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
def repaired_benchmark(fn, args):
    jax.block_until_ready(args)
    jax.block_until_ready(fn(*args))
    result, measured = synchronized_samples(fn, args, 5)
    return {"boundary": "placed inputs; warm call plus output wait", "samples": measured}, result
fixed_report, fixed_result = repaired_benchmark(compiled_predict, (x, w))
np.testing.assert_allclose(np.asarray(fixed_result), reference, atol=2e-5, rtol=2e-5)
assert len(fixed_report["samples"]) == 5
print(fixed_report)

```

The repaired boundary excludes setup, separates warmup and waits before stopping. A speedup cannot be reconstructed from the flawed interval; rerun both implementations under the same contract.

</details>

## Check your understanding

What does the synchronized warmed timer in this lesson measure?

1. Only the accelerator kernel, with all dispatch overhead removed
2. A warmed call including dispatch and waiting for output readiness, excluding prior input placement
3. Exactly the compiler duration

<details><summary>Answer and explanation</summary>

A warmed call including dispatch and waiting for output readiness, excluding prior input placement

The boundary includes the call and its completion wait. It excludes input placement and does not isolate compiler or kernel stages.

</details>

## Diagnose the result

If samples vary, first check workload identity, dtype, the warmup call, where synchronization occurs and system contention. Do not change a timer until its start/stop boundary answers the intended question.

## Carry forward

- Keep first-call latency separate from warmed samples.
- Wait for results before reporting completed computation.
- Report numerical equivalence and workload details beside times.

## Keep your evidence

Keep benchmark.py, complete observed JSON/sample output, workload shapes/dtypes, numerical tolerance/error, first-call versus warmed boundaries, changed-workload report and repaired timer explanation.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [Benchmarking JAX code](https://docs.jax.dev/en/latest/benchmarking.html)
- [Asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)

