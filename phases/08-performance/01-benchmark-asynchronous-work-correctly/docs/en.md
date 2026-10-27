# Benchmark asynchronous work correctly

Phase 08: Performance diagnosis · about 80 minutes · CPU

This reference explicitly selects CPU. Uploading the same file or setting JAX_PLATFORMS does not adapt it to TPU. Use the execution matrix to review device selection and data ownership before adapting a separate target program.

## What you will be able to do

- Draw the interval a timing actually measures.
- Exclude initial placement from a device-resident benchmark.
- Check an independent NumPy result before comparing timings.
- Produce a report containing first-call and warmed sample distributions.

## The problem

A first timing suggests a tenfold speedup. Before celebrating, let’s ask when the timer started and stopped. We’ll separate the first call from warm calls, wait for JAX to finish, and compare numerical results. By the end, you’ll have a timing report another learner can reproduce.

## The idea

JAX can return an array handle before device work finishes. A benchmark must define what starts the interval and what counts as completion. First-call preparation, warm execution and end-to-end requests answer different performance questions.

## Time a completed result

A JAX call can return a handle before its device work is complete, so a timer that stops at return may measure dispatch rather than the finished result. Wait on the output before stopping the clock when the question is completed-result latency.

Separate first-call preparation from repeated calls, and name the boundary you care about. This lesson starts with inputs already placed; an end-to-end request would need a different interval that includes whichever preparation and transfers belong to that request.

$$
T_{\rm first}=t_{\rm first,ready}-t_{\rm first,start},\qquad T_{\rm warm}=\operatorname{median}_r(t_{r,\rm ready}-t_{r,\rm start})
$$

### Time the completed result

**Predict:** Where should synchronization occur in completed-result latency measurement?

![Time the completed result](../outputs/mechanism.svg)

*Architecture and dataflow mechanism diagram.*

Read left to right: already-ready inputs reach the host, the host dispatches a call, and the device-work lane proceeds while the host waits. The timer bracket starts at dispatch and ends at result readiness, so the wait is inside the measurement. The arrows show dependency and elapsed ordering, not measured durations; input preparation and optional compilation need their own boundary if the question includes them.

### Pause and reason

Where should synchronization occur in completed-result latency measurement?

<details><summary>Compare your reasoning</summary>

Before stopping the timer. Synchronizing afterward can omit unfinished work from the reported interval.

</details>

## Draw the boundary before starting the clock

The workload maps $(64,32)$ times $(32,16)$ to $(64,16)$, then applies tanh. Both inputs are placed on CPU and ready before the timer starts; the timer stops after the output is ready. Thus the interval includes host dispatch and completion waiting, while input creation, initial placement and conversion to NumPy are outside it.

Use this boundary for the question “how long until this placed-input result is ready?” It is not a kernel-only duration or a full serving request.

```text
prepare host arrays → place → wait | start → call → result handle → wait → stop | inspect host values
```

## A handle is not proof of completion

Python can receive a JAX array while its values are still being computed. Shape and dtype are metadata; `block_until_ready` waits for the result, while conversion to NumPy also materializes values on the host. Put the wait inside the timed interval when reporting completed-result latency.

This CPU workload may make an unsynchronized and synchronized timing look similar. That observation does not turn a handle-return timer into a portable completion measurement, and subtracting separate calls cannot isolate exact queue or execution time.

## First call and steady state answer different questions

The first jitted call can trace Python, lower the program, obtain an executable and run it. The first-call timer includes that combined interval. Later calls with the same signature can reuse the executable, so the warm samples answer a different question: how long does repeated completed work take in this process?

For a worked conversion, suppose a first call took $40$ milliseconds and a warm call took $20$ microseconds. Converting both to seconds gives $0.04$ and $0.00002$, a ratio of $2000$. These illustrative values explain the unit conversion, not a promised speedup. Nor would their difference isolate compilation: the first interval can include other first-use work.

Keep every warm sample beside the median. Seven observations show variation but do not establish a confidence interval or deployment behavior. The plot and the script receipt are separate captures; use each capture's own samples when calculating its summary.

## Correctness comes before a speed ratio

The NumPy reference uses the same float32 inputs, matrix product and tanh. The recorded maximum absolute error is about $2.38\times10^{-7}$, within the declared tolerance. Check this equivalence before comparing timing.

If dtype, shape or operation differs, a lower time answers a different question. Keep the workload contract and numerical error next to the timing report.

## Prepare and place the workload

Create benchmark.py in your activated course environment and add this block.

```python
# Step 1 — Prepare and place the workload: CPU placement and waiting occur before timing.
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
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(x_host, cpu)
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(w_host, cpu)
# Wait for device work to finish before timing it or comparing its result on the host.
jax.block_until_ready((x, w))
```

CPU placement and waiting occur before timing. The deterministic host inputs make the numerical check reproducible.

## Define the contract and timer

Append this block to the same file. Run the assembled file with python benchmark.py.

```python
# Step 2 — Define the contract and timer
def predict(x, w):
    return jnp.tanh(x @ w)
# Wrap with `jax.jit` (`compiled_predict`) so XLA traces and compiles the function.
compiled_predict = jax.jit(predict)
def synchronized_samples(fn, args, repeats=7):
    if repeats < 1:
        raise ValueError("at least one repeat is required")
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args)
        # Wait for device work to finish before timing it or comparing its result on the host.
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return result, samples
```

The timer accepts arbitrary result pytrees because `jax.block_until_ready` waits across their array leaves. It collects observations without asserting which implementation is faster.

## Separate first call from warm calls

Append this block to the same file. Run the assembled file with python benchmark.py.

```python
# Step 3 — Separate first call from warm calls: The report prints variable measured times.
start = time.perf_counter()
first = compiled_predict(x, w)
# Wait for device work to finish before timing it or comparing its result on the host.
first.block_until_ready()
first_call_seconds = time.perf_counter() - start
result, samples = synchronized_samples(compiled_predict, (x, w))
reference = np.tanh(x_host @ w_host)
# Transfer the device result to the host for an independent NumPy comparison.
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
# Step 1 — Prepare and place the workload: CPU placement and waiting occur before timing.
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
# Place `x` explicitly onto the target JAX device.
x = jax.device_put(x_host, cpu)
# Place `w` explicitly onto the target JAX device.
w = jax.device_put(w_host, cpu)
# Wait for device work to finish before timing it or comparing its result on the host.
jax.block_until_ready((x, w))
# Step 2 — Define the contract and timer
def predict(x, w):
    return jnp.tanh(x @ w)
# Wrap with `jax.jit` (`compiled_predict`) so XLA traces and compiles the function.
compiled_predict = jax.jit(predict)
def synchronized_samples(fn, args, repeats=7):
    if repeats < 1:
        raise ValueError("at least one repeat is required")
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args)
        # Wait for device work to finish before timing it or comparing its result on the host.
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return result, samples
# Step 3 — Separate first call from warm calls: The report prints variable measured times.
start = time.perf_counter()
first = compiled_predict(x, w)
# Wait for device work to finish before timing it or comparing its result on the host.
first.block_until_ready()
first_call_seconds = time.perf_counter() - start
result, samples = synchronized_samples(compiled_predict, (x, w))
reference = np.tanh(x_host @ w_host)
# Transfer the device result to the host for an independent NumPy comparison.
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

![Separate the first call from warm measurements](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The upper panel has one bar for the first synchronized call; the lower panel has seven points for repeated synchronized calls of the same signature. Both vertical axes are seconds, but they use independent scales. Read the tick values and any scientific-notation multiplier before comparing heights across panels.

The first-call interval is on the order of hundredths of a second in this local capture, while the warm points are much smaller. Each lower-panel dot is a repeated timing sample, not a training step, and its vertical movement is timing variation rather than an improving objective. Find the minimum and maximum dots, then locate the median among all seven values. One unusually low sample is not a repeatable latency estimate.

### Connect it to the computation

Inputs were already placed before both measurements, and each interval ends when the output is ready. The first interval can include preparation plus execution; the warm interval measures reuse of the same compiled signature. The separate execution receipt is another run of that protocol, so its individual timings need not equal the plotted capture.

Now change the question: if a user supplies a fresh NumPy input for every request, which preparation or placement costs belong inside the timer? Draw that new boundary before comparing it with this resident-input result. A valid timing comparison needs equivalent work as well as a common unit.

```python
visual_data = {'kind': 'panels', 'panels': [{'kind': 'bar', 'title': 'First call', 'labels': ['first'], 'ylabel': 'seconds', 'series': [{'label': 'synchronized call', 'y': [first_call_seconds]}]}, {'kind': 'line', 'title': 'Warm calls', 'x': list(range(1, len(samples) + 1)), 'xlabel': 'warm sample', 'ylabel': 'seconds', 'series': [{'label': 'synchronized call', 'y': samples}]}]}
```

## Recorded reference execution

CPU run: 2026-10-27T17:13:51.978305+00:00. JAX 0.9.2.

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
  "first_call_seconds": 0.057142124976962805,
  "warm_samples_seconds": [
    0.00012704217806458473,
    5.745887756347656e-05,
    4.9666035920381546e-05,
    1.4457851648330688e-05,
    1.058308407664299e-05,
    1.0292045772075653e-05,
    9.458977729082108e-06
  ],
  "warm_median_seconds": 1.4457851648330688e-05,
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
  "first_call_seconds": 0.030218500178307295,
  "warm_samples_seconds": [
    0.00024641724303364754,
    0.00013183290138840675,
    0.0004402496851980686,
    0.00034912489354610443,
    0.00017945794388651848,
    9.325006976723671e-05,
    0.0001166253350675106
  ],
  "warm_median_seconds": 0.00017945794388651848,
  "max_abs_error": 2.384185791015625e-07
}
Handle-only seconds: 3.370782360434532e-05
Complete-call samples: [3.7457793951034546e-05, 2.9458198696374893e-05, 1.7083249986171722e-05]
Structured samples: [0.0005777501501142979, 0.00021449988707900047, 0.00011362507939338684]
Changed workload: (96, 32) [7.704179733991623e-05, 2.133287489414215e-05, 3.7250109016895294e-05, 8.712522685527802e-05, 4.124967381358147e-05]
Energy shape and samples: (64,) [9.020790457725525e-05, 8.579203858971596e-05, 3.191595897078514e-05, 1.8458813428878784e-05, 1.4250166714191437e-05]
{'boundary': 'placed inputs; warm call plus output wait', 'samples': [4.637520760297775e-05, 0.00023041712120175362, 0.0006031668744981289, 0.00011795805767178535, 7.745902985334396e-05]}
PASS: performance-01

```

## Compare handle-only and synchronized boundaries

**Predict before running:** Will the handle-only number necessarily be smaller on this CPU? Which interval guarantees output completion?

```python
# Experiment — Compare handle-only and synchronized boundaries
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
# Experiment — Return structured results without host copies
# Wrap with `jax.jit` (`structured`) so XLA traces and compiles the function.
structured = jax.jit(lambda a, b: {"prediction": jnp.tanh(a @ b), "mean": jnp.mean(jnp.tanh(a @ b))})
# Wait for device work to finish before timing it or comparing its result on the host.
jax.block_until_ready(structured(x, w))
structured_result, structured_times = synchronized_samples(structured, (x, w), 3)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(structured_result["prediction"]), reference, atol=2e-5, rtol=2e-5)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(structured_result["mean"]), reference.mean(), atol=2e-5, rtol=2e-5)
print("Structured samples:", structured_times)
```

**Expected:** Three timings and independently verified prediction/mean leaves.

A tree-level wait makes the benchmark contract explicit for every array leaf. Host inspection occurs after the timer, so the report describes device-ready computation rather than transfer-inclusive output consumption.

## Make it yours

Repeat the benchmark with $96$ rows while preserving the feature and output dimensions. Keep a separate report rather than overwriting the original workload label.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Inputs:**
- A JAX function, fixed workload, warmup count, and repeat count.

**Outputs:**
- Raw synchronized timings with compilation separated from warmed execution.

**Invariants to check:**
- Every sample waits for completed output; workload shape/dtype and timing boundary are recorded.

**Hints:**
- Call block_until_ready inside each measured interval.
- Use an independent NumPy result for the changed equivalent workload.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp
import time
import statistics

def synchronized_benchmark(function, arguments, warmup, repeats):
    """Raw synchronized timings with compilation separated from warmed execution."""
    # TODO: Warm the selected JAX function and synchronize every timed result while retaining raw samples.
    raise NotImplementedError
```

<details><summary>Reference solution</summary>

```python
# Exercise solution
larger_host = rng.normal(size=(96, 32)).astype(np.float32)
# Place `larger` explicitly onto the target JAX device.
larger = jax.device_put(larger_host, cpu)
# Wait for device work to finish before timing it or comparing its result on the host.
larger.block_until_ready()
# Wait for device work to finish before timing it or comparing its result on the host.
compiled_predict(larger, w).block_until_ready()
larger_result, larger_times = synchronized_samples(compiled_predict, (larger, w), 5)
# Transfer the device result to the host for an independent NumPy comparison.
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

### How to write: Benchmark a different equivalent workload — Step-by-step recipe & starter scaffold

**Inputs:**
- Replace tanh with a row-wise sum of squared predictions, returning one scalar per row. Derive the NumPy reference, warm this new function, collect five synchronized samples and record the changed output shape.

**Outputs:**
- Per-row scalar outputs, NumPy parity, and five completed timing samples.

**Invariants to check:**
- The benchmark synchronizes each output and records the changed output shape.

**Hints:**
- Square before reducing the output-feature axis; the result has shape $(64,)$.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp
import numpy as np
import time
import statistics

def practice_benchmark_a_different_equivalent_workload(features, weights):
    """Implement the benchmark a different equivalent workload check."""
    # TODO: Implement row-wise squared-sum predictions, derive a NumPy reference, warm the changed function, and collect five synchronized samples.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Benchmark a different equivalent workload (Transfer): Changing the computation changes the performance question.
# Wrap with `jax.jit` (`energy`) so XLA traces and compiles the function.
energy = jax.jit(lambda a, b: jnp.sum((a @ b)**2, axis=1))
# Wait for device work to finish before timing it or comparing its result on the host.
energy(x, w).block_until_ready()
energy_result, energy_times = synchronized_samples(energy, (x, w), 5)
# Reduce along axis=1 to compute `energy_reference`.
energy_reference = np.sum((x_host @ w_host)**2, axis=1)
# Check tensor shape invariant: `energy_result.shape == (64,)`
assert energy_result.shape == (64,)
# Transfer the device result to the host for an independent NumPy comparison.
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

### How to write: Repair a timer that includes setup but omits completion — Step-by-step recipe & starter scaffold

**Inputs:**
- Inspect the broken timer below conceptually: start; create inputs; call `jit(predict)(inputs)`; stop. Write a repaired runner and show its explicit boundary. Explain which stages the broken interval mixes and misses.

**Outputs:**
- A runner with separately stated setup and completed-work timing boundaries.

**Invariants to check:**
- The measured interval contains the declared workload and waits for its result.

**Hints:**
- Prepare and wait for inputs, reuse one compiled callable, warm it, then synchronize the returned output inside every measured interval.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp
import time

def practice_repair_a_timer_that_includes_setup_but_omits_completion(function, arguments):
    """Implement the repair a timer that includes setup but omits completion check."""
    # TODO: Move input creation outside the timed region and block on the output before stopping the clock.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Repair a timer that includes setup but omits completion (Diagnosis)
def repaired_benchmark(fn, args):
    # Wait for device work to finish before timing it or comparing its result on the host.
    jax.block_until_ready(args)
    # Wait for device work to finish before timing it or comparing its result on the host.
    jax.block_until_ready(fn(*args))
    result, measured = synchronized_samples(fn, args, 5)
    return {"boundary": "placed inputs; warm call plus output wait", "samples": measured}, result
fixed_report, fixed_result = repaired_benchmark(compiled_predict, (x, w))
# Transfer the device result to the host for an independent NumPy comparison.
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

**1. Only the accelerator kernel, with all dispatch overhead removed**

The measured boundary includes host dispatch and the wait. Calling it pure device-kernel time requires a device trace.

**2. A warmed call including dispatch and waiting for output readiness, excluding prior input placement (correct)**

The boundary includes the call and its completion wait. It excludes input placement and does not isolate compiler or kernel stages.

**3. Exactly the compiler duration**

Warmup has already performed compilation for the tested signature. A new shape or static argument may still introduce another compilation.

</details>

## Diagnose the result

If samples vary, first check workload identity, dtype, the warmup call, where synchronization occurs and system contention. Do not change a timer until its start/stop boundary answers the intended question.

## Carry forward

- Keep first-call latency separate from warmed samples.
- Wait for results before reporting completed computation.
- Report numerical equivalence and workload details beside times.

## Keep your evidence

Keep benchmark.py, complete observed JSON/sample output, workload shapes/dtypes, numerical tolerance/error, first-call versus warmed boundaries, changed-workload report and repaired timer explanation.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [Benchmarking JAX code](https://docs.jax.dev/en/latest/benchmarking.html)
- [Asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)

