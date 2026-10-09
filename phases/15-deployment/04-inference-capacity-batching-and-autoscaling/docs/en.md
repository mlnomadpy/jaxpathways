# Inference capacity, batching, and autoscaling

Phase 15: Deployment, interoperability & edge AI · about 130 minutes · CPU

## What you will be able to do

- Measure completed inference after warming each batch shape
- Report batch latency and example throughput with correct units
- Implement and independently check a serial queue timeline
- Distinguish utilization, batch-collection delay and hypothetical replica planning from measured service evidence

## The problem

A model call looks fast in isolation, yet users may wait when requests arrive together. We will measure actual completed inference at several batch sizes, then use those observations in a clearly labeled queue simulation. The aim is to connect milliseconds per batch, examples per second and response latency without confusing a local benchmark with a deployed service.

## The idea

Inference capacity depends on arrivals, queueing, batching, service time and response work. A fast model call is one component. Define the request boundary before estimating throughput or deciding when to add capacity.

## A request spends time before and after computation

A request may wait for a batch, wait behind existing work, execute and then be encoded for response. Larger batches can improve work per call while increasing the first waiting request's latency. These objectives can conflict.

Draw arrival, enqueue, dispatch and completion on one timeline. Then measure or model each interval with its source stated. Tail latency depends on the arrival pattern and concurrency, not just the mean of isolated calls.

This lesson combines measured CPU batch computation with a separately simulated queue. Keep those panels distinct. A queue simulation explores stated assumptions; it is not an observed autoscaling system or a production traffic measurement.

### Pause and reason

Can higher batch throughput coexist with worse user latency?

<details><summary>Compare your reasoning</summary>

Yes. Requests can wait longer to form a batch or reach service. Report both completed throughput and the declared end-to-end latency distribution.

</details>

## Choose the clock boundary first

JAX dispatch may return before numerical work has completed. The timer calls block_until_ready inside the measured interval. Inputs and weights are already resident; preprocessing, device placement, request decoding and response encoding are excluded. The first call for each shape can include compilation and is recorded separately from $40$ warm calls. A p95 from $40$ samples is a small descriptive statistic, not a reliable production tail estimate. Timings change across hosts and repeated runs; correctness checks must not assert a fixed speedup.

## Keep request units and example units separate

Let a batch of $B$ examples take $t_B$ milliseconds. A serial worker completing full batches has a compute-throughput proxy $1000B/t_B$ examples per second. The batch still takes $t_B$ milliseconds; dividing by $B$ describes amortized cost per example, not the response time experienced by one example in that batch. Partial batches, memory pressure, shape recompilation and preprocessing can change both numbers.

$$
q_B=\frac{1000B}{t_B}\quad\text{examples/second}
$$

## Trace a queue before using a rate formula

For one first-come-first-served worker, a request starts at the later of its arrival and the previous finish. With arrivals at $0,1,4$ milliseconds and service duration $2$ milliseconds, starts are $0,2,4$, finishes are $2,4,6$, and response times are $2,3,2$. The second request waits one millisecond because the worker is busy. This independent hand timeline checks our simulation before we feed it measured service durations.

$$
s_i=\max(a_i,f_{i-1}),\qquad f_i=s_i+t,\qquad r_i=f_i-a_i
$$

## A utilization estimate is not a latency guarantee

If arrival rate is $\lambda$ requests per second and each request consumes $t$ milliseconds on one serial worker, the utilization proxy is $\rho=\lambda t/1000$. Sustained offered work above capacity makes backlog grow in this unlimited-queue model. Even below capacity, bursts and service-time variation can cause long waits. Our stable arrivals are spaced $1.5t$ apart, so no queue forms. Overloaded arrivals are $0.6t$ apart; waiting grows by $0.4t$ per request and the sixtieth response takes $24.6t$. These are simulated consequences of declared inputs, not measured user latencies.

## Batch collection trades waiting for throughput

With regularly spaced arrivals separated by $\delta$, filling a batch of $B$ from empty makes its first example wait $(B-1)\delta$ before execution; the average collection wait is $(B-1)\delta/2$ if all positions are equally represented. A timeout permits partial batches and limits intentional collection delay, but a busy worker can still add queueing. A production experiment should replay realistic arrivals, batch sizes and sequence lengths, record queue, collection and execution times separately, and compare tail latency against an explicit service-level objective. **TODO (Dynamic Batching Server Extension):** Replay this arrival trace against a live dynamic-batching HTTP server and compare p99 queue + execution latency.

## Plan replicas with headroom, then test the plan

For a measured effective capacity $q$ requests per second per replica and target utilization $u$, a first estimate is $R=\lceil\lambda/(u q)\rceil$. This assumes compatible units, balanced routing and an actual capacity measurement for the workload. Autoscaling also has observation windows, startup time, warming and cooldown; it cannot erase a burst that arrives before new capacity is ready. Keep the formula labeled as a planning assumption until a load test includes the real request boundary, concurrency limits, failures and startup behavior.

## A percentile is a sample summary, not a latency promise

Sort twenty observed request times. If nineteen are $1$ millisecond and one is $100$ milliseconds, NumPy's default linear percentile interpolation reports p95 as $5.95$ milliseconds. The maximum remains $100$. The interpolated value need not be an observed request time.

That summary does not guarantee that future requests will stay below p95, and a small sample gives weak evidence about rare slow requests. State the sample size and percentile method, retain the full samples, and repeat the run under the intended workload. Do not average percentiles from separate workers; combine request samples or use an aggregation method appropriate to the monitoring system.

### Pause and reason

Would reporting only p95 reveal the actual worst observed request in this example?

<details><summary>Compare your reasoning</summary>

No. The worst observed request is $100$ milliseconds, while interpolated p95 is $5.95$. Report the distribution and maximum alongside the requested percentile when those distinctions matter.

</details>

## 1. Define and check the inference workload

Create a fresh main.py. This fixed dense-network fixture measures an inference boundary; its random weights are not presented as a pretrained or accurate model.

```python
# Step 1 — 1. Define and check the inference workload: The timer includes dispatch and completion for already resident...
# Import math for this computation.
import math
import time
import numpy as np
import jax
import jax.numpy as jnp
# Draw pseudorandom samples for `rng` using the explicit RNG state.
rng=np.random.default_rng(82)
# Create device-backed JAX array `W1`.
W1=jnp.asarray(rng.normal(0,.1,(64,32)).astype(np.float32))
# Create device-backed JAX array `W2`.
W2=jnp.asarray(rng.normal(0,.1,(32,8)).astype(np.float32))
# Define and JIT-compile `infer(inputs)` so XLA traces and fuses the operations:
@jax.jit
# Function `infer(inputs)` implementing this stage's computation:
def infer(inputs):
    # Return `jnp.tanh(inputs @ W1) @ W2` to the caller.
    return jnp.tanh(inputs@W1)@W2
# Function `measure(batch_size, repeats)` implementing this stage's computation:
def measure(batch_size,repeats=40):
    # Cast or evaluate `host` in explicit floating-point precision.
    host=rng.normal(size=(batch_size,64)).astype(np.float32)
    # Create device-backed JAX array `device`.
    device=jnp.asarray(host)
    # Record execution timing or profiler trace in `began`.
    began=time.perf_counter()
    # Synchronize host execution until asynchronous device computation completes.
    first=infer(device).block_until_ready()
    # Record execution timing or profiler trace in `first_ms`.
    first_ms=(time.perf_counter()-began)*1000
    # Convert `expected` to a host NumPy array for inspection or verification.
    expected=np.tanh(host@np.asarray(W1))@np.asarray(W2)
    # Check numerical equivalence within tolerance: `np.testing.assert_allclose(first,expected,rtol=2e-5,atol=2e-6)`
    np.testing.assert_allclose(first,expected,rtol=2e-5,atol=2e-6)
    # Compute `samples` from `[]`
    samples=[]
    # Repeat the update loop over `range(repeats)` steps:
    for _ in range(repeats):
        # Record execution timing or profiler trace in `began`.
        began=time.perf_counter()
        # Synchronize host execution until asynchronous device computation completes.
        infer(device).block_until_ready()
        # Record execution timing or profiler trace in ``.
        samples.append((time.perf_counter()-began)*1000)
    # Evaluate `np.percentile(samples, 50)` and convert the result into Python scalar/collection `p50`.
    p50=float(np.percentile(samples,50))
    # Return `{'batch': batch_size, 'first_call_ms': first_ms, 'samples_ms': samples, 'p50_ms': p50, 'p95_ms': float(np.percentile(samples, 95)), 'steady_examples_per_second': 1000 * batch_size / p50}` to the caller.
    return {'batch':batch_size,'first_call_ms':first_ms,'samples_ms':samples,
            'p50_ms':p50,'p95_ms':float(np.percentile(samples,95)),
            'steady_examples_per_second':1000*batch_size/p50}
```

The timer includes dispatch and completion for already resident inputs. JSON, networking, queueing and host-to-device placement are outside this boundary.

## 2. Measure each batch shape separately

Append three batch shapes and print all timing boundaries. Each shape has its own first call and warmed sample distribution.

```python
# Step 2 — 2. Measure each batch shape separately: Do not assume a larger batch is faster per request.
measurements=[measure(batch) for batch in (1,8,32)]
# Iterate over `report` to step through the computation:
for report in measurements:
    # Assert invariant `len(report['samples_ms'])==40` holds
    assert len(report['samples_ms'])==40
    # Ensure all array elements remain finite: `all(value>0 and np.isfinite(value) for value in report['samples_m...`
    assert all(value>0 and np.isfinite(value) for value in report['samples_ms'])
    # Print the observed values to compare against the expected result.
    print({key:value for key,value in report.items() if key!='samples_ms'})
# Compute `service_ms` from `measurements[0]['p50_ms']`
service_ms=measurements[0]['p50_ms']
```

Do not assume a larger batch is faster per request. Examples per second and milliseconds per batch answer different questions.

## 3. Model a queue with an independent hand-check

Append the FCFS simulator and two hypothetical arrival schedules. Times are milliseconds throughout the simulation.

```python
# Step 3 — 3. Model a queue with an independent hand-check: This simulator assumes one serial worker and a constant service...
def simulate_queue(arrival_ms,service_ms):
    # Convert `arrivals` to a host NumPy array for inspection or verification.
    arrivals=np.asarray(arrival_ms,dtype=float)
    # Guard input contract (`arrivals.ndim != 1 or np.any(~np.isfinite(arrivals)) or np.any(np.diff(arrivals) < 0)`) and fail fast if violated.
    if arrivals.ndim!=1 or np.any(~np.isfinite(arrivals)) or np.any(np.diff(arrivals)<0):
        raise ValueError('arrivals must be a finite ordered vector')
    # Guard input contract (`not np.isfinite(service_ms) or service_ms <= 0`) and fail fast if violated.
    if not np.isfinite(service_ms) or service_ms<=0:
        raise ValueError('service duration must be positive milliseconds')
    # Compute `ready` from `0.`
    ready=0.
    # Evaluate `starts` from the current inputs and state.
    # Compute `starts` from `[]`
    starts=[]
    finishes=[]
    # Iterate over `arrival` to step through the computation:
    for arrival in arrivals:
        # Run `max` to compute `start`.
        start=max(float(arrival),ready)
        # Compute `ready` from `start+service_ms`
        ready=start+service_ms
        # Append the current step result to `starts`.
        # Append the current step result to `starts`.
        starts.append(start)
        finishes.append(ready)
    # Return `(np.array(starts), np.array(finishes), np.array(finishes) - arrivals)` to the caller.
    return np.array(starts),np.array(finishes),np.array(finishes)-arrivals
# Run `simulate_queue` to compute `(hand_start, hand_finish, hand_latency)`.
hand_start,hand_finish,hand_latency=simulate_queue([0.,1.,4.],2.)
# Execute `np.testing.assert_allclose(hand_start,[0.,2.,4.])`
np.testing.assert_allclose(hand_start,[0.,2.,4.])
# Execute `np.testing.assert_allclose(hand_finish,[2.,4.,6.])`
np.testing.assert_allclose(hand_finish,[2.,4.,6.])
# Execute `np.testing.assert_allclose(hand_latency,[2.,3.,2.])`
np.testing.assert_allclose(hand_latency,[2.,3.,2.])
# Compute `request_index` from `np.arange(60)`
request_index=np.arange(60)
# Compute `stable_arrivals` from `request_index*1.5*service_ms`
stable_arrivals=request_index*1.5*service_ms
# Compute `overloaded_arrivals` from `request_index*.6*service_ms`
overloaded_arrivals=request_index*.6*service_ms
# Run `simulate_queue` to compute `(_, _, stable_latency)`.
_,_,stable_latency=simulate_queue(stable_arrivals,service_ms)
# Run `simulate_queue` to compute `(_, _, overloaded_latency)`.
_,_,overloaded_latency=simulate_queue(overloaded_arrivals,service_ms)
# Check numerical equivalence within tolerance: `np.allclose(stable_latency,service_ms)`
assert np.allclose(stable_latency,service_ms)
# Check numerical equivalence within tolerance: `np.isclose(overloaded_latency[-1],24.6*service_ms)`
assert np.isclose(overloaded_latency[-1],24.6*service_ms)
# Print diagnostic summary of the computed outputs.
print('measured batch-one service proxy (ms):',service_ms)
# Print diagnostic summary of the computed outputs.
print('simulated last stable/overloaded response (ms):',stable_latency[-1],overloaded_latency[-1])
```

This simulator assumes one serial worker and a constant service duration derived from the measured median. It is not a load test, an HTTP server or an autoscaler.

## Run the example

```python
# Step 1 — 1. Define and check the inference workload: The timer includes dispatch and completion for already resident...
# Import math for this computation.
import math
import time
import numpy as np
import jax
import jax.numpy as jnp
# Draw pseudorandom samples for `rng` using the explicit RNG state.
rng=np.random.default_rng(82)
# Create device-backed JAX array `W1`.
W1=jnp.asarray(rng.normal(0,.1,(64,32)).astype(np.float32))
# Create device-backed JAX array `W2`.
W2=jnp.asarray(rng.normal(0,.1,(32,8)).astype(np.float32))
# Define and JIT-compile `infer(inputs)` so XLA traces and fuses the operations:
@jax.jit
# Function `infer(inputs)` implementing this stage's computation:
def infer(inputs):
    # Return `jnp.tanh(inputs @ W1) @ W2` to the caller.
    return jnp.tanh(inputs@W1)@W2
# Function `measure(batch_size, repeats)` implementing this stage's computation:
def measure(batch_size,repeats=40):
    # Cast or evaluate `host` in explicit floating-point precision.
    host=rng.normal(size=(batch_size,64)).astype(np.float32)
    # Create device-backed JAX array `device`.
    device=jnp.asarray(host)
    # Record execution timing or profiler trace in `began`.
    began=time.perf_counter()
    # Synchronize host execution until asynchronous device computation completes.
    first=infer(device).block_until_ready()
    # Record execution timing or profiler trace in `first_ms`.
    first_ms=(time.perf_counter()-began)*1000
    # Convert `expected` to a host NumPy array for inspection or verification.
    expected=np.tanh(host@np.asarray(W1))@np.asarray(W2)
    # Check numerical equivalence within tolerance: `np.testing.assert_allclose(first,expected,rtol=2e-5,atol=2e-6)`
    np.testing.assert_allclose(first,expected,rtol=2e-5,atol=2e-6)
    # Compute `samples` from `[]`
    samples=[]
    # Repeat the update loop over `range(repeats)` steps:
    for _ in range(repeats):
        # Record execution timing or profiler trace in `began`.
        began=time.perf_counter()
        # Synchronize host execution until asynchronous device computation completes.
        infer(device).block_until_ready()
        # Record execution timing or profiler trace in ``.
        samples.append((time.perf_counter()-began)*1000)
    # Evaluate `np.percentile(samples, 50)` and convert the result into Python scalar/collection `p50`.
    p50=float(np.percentile(samples,50))
    # Return `{'batch': batch_size, 'first_call_ms': first_ms, 'samples_ms': samples, 'p50_ms': p50, 'p95_ms': float(np.percentile(samples, 95)), 'steady_examples_per_second': 1000 * batch_size / p50}` to the caller.
    return {'batch':batch_size,'first_call_ms':first_ms,'samples_ms':samples,
            'p50_ms':p50,'p95_ms':float(np.percentile(samples,95)),
            'steady_examples_per_second':1000*batch_size/p50}

# Step 2 — 2. Measure each batch shape separately: Do not assume a larger batch is faster per request.
measurements=[measure(batch) for batch in (1,8,32)]
# Iterate over `report` to step through the computation:
for report in measurements:
    # Assert invariant `len(report['samples_ms'])==40` holds
    assert len(report['samples_ms'])==40
    # Ensure all array elements remain finite: `all(value>0 and np.isfinite(value) for value in report['samples_m...`
    assert all(value>0 and np.isfinite(value) for value in report['samples_ms'])
    # Print the observed values to compare against the expected result.
    print({key:value for key,value in report.items() if key!='samples_ms'})
# Compute `service_ms` from `measurements[0]['p50_ms']`
service_ms=measurements[0]['p50_ms']

# Step 3 — 3. Model a queue with an independent hand-check: This simulator assumes one serial worker and a constant service...
def simulate_queue(arrival_ms,service_ms):
    # Convert `arrivals` to a host NumPy array for inspection or verification.
    arrivals=np.asarray(arrival_ms,dtype=float)
    # Guard input contract (`arrivals.ndim != 1 or np.any(~np.isfinite(arrivals)) or np.any(np.diff(arrivals) < 0)`) and fail fast if violated.
    if arrivals.ndim!=1 or np.any(~np.isfinite(arrivals)) or np.any(np.diff(arrivals)<0):
        raise ValueError('arrivals must be a finite ordered vector')
    # Guard input contract (`not np.isfinite(service_ms) or service_ms <= 0`) and fail fast if violated.
    if not np.isfinite(service_ms) or service_ms<=0:
        raise ValueError('service duration must be positive milliseconds')
    # Compute `ready` from `0.`
    ready=0.
    # Evaluate `starts` from the current inputs and state.
    # Compute `starts` from `[]`
    starts=[]
    finishes=[]
    # Iterate over `arrival` to step through the computation:
    for arrival in arrivals:
        # Run `max` to compute `start`.
        start=max(float(arrival),ready)
        # Compute `ready` from `start+service_ms`
        ready=start+service_ms
        # Append the current step result to `starts`.
        # Append the current step result to `starts`.
        starts.append(start)
        finishes.append(ready)
    # Return `(np.array(starts), np.array(finishes), np.array(finishes) - arrivals)` to the caller.
    return np.array(starts),np.array(finishes),np.array(finishes)-arrivals
# Run `simulate_queue` to compute `(hand_start, hand_finish, hand_latency)`.
hand_start,hand_finish,hand_latency=simulate_queue([0.,1.,4.],2.)
# Execute `np.testing.assert_allclose(hand_start,[0.,2.,4.])`
np.testing.assert_allclose(hand_start,[0.,2.,4.])
# Execute `np.testing.assert_allclose(hand_finish,[2.,4.,6.])`
np.testing.assert_allclose(hand_finish,[2.,4.,6.])
# Execute `np.testing.assert_allclose(hand_latency,[2.,3.,2.])`
np.testing.assert_allclose(hand_latency,[2.,3.,2.])
# Compute `request_index` from `np.arange(60)`
request_index=np.arange(60)
# Compute `stable_arrivals` from `request_index*1.5*service_ms`
stable_arrivals=request_index*1.5*service_ms
# Compute `overloaded_arrivals` from `request_index*.6*service_ms`
overloaded_arrivals=request_index*.6*service_ms
# Run `simulate_queue` to compute `(_, _, stable_latency)`.
_,_,stable_latency=simulate_queue(stable_arrivals,service_ms)
# Run `simulate_queue` to compute `(_, _, overloaded_latency)`.
_,_,overloaded_latency=simulate_queue(overloaded_arrivals,service_ms)
# Check numerical equivalence within tolerance: `np.allclose(stable_latency,service_ms)`
assert np.allclose(stable_latency,service_ms)
# Check numerical equivalence within tolerance: `np.isclose(overloaded_latency[-1],24.6*service_ms)`
assert np.isclose(overloaded_latency[-1],24.6*service_ms)
# Print diagnostic summary of the computed outputs.
print('measured batch-one service proxy (ms):',service_ms)
# Print diagnostic summary of the computed outputs.
print('simulated last stable/overloaded response (ms):',stable_latency[-1],overloaded_latency[-1])
```

Expected: The program prints first-call, warm p50/p95 and examples-per-second measurements for each batch size. Absolute timings are host-dependent. The independent queue example yields response times of 2, 3 and 2 milliseconds.

## Measured batch computation and a separately simulated queue

**Predict:** Can a queue grow even when the duration of each individual model call is unchanged?

![Measured batch computation and a separately simulated queue](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The upper panel shows measured CPU warm p50 and p95 milliseconds per batch for batch sizes $1,8,32$. Each timing waits for completion; inputs are already resident. These are sample percentiles, not confidence bounds or a speedup guarantee. The lower panel uses request index on the horizontal axis and simulated response time divided by the measured batch-one median on the vertical axis. Stable arrivals stay at $1$; overloaded arrivals rise from $1$ to $24.6$.

### Connect it to the computation

The lower panel’s growth comes entirely from the chosen arrival schedule and serial queue recurrence. The service duration stays fixed in the simulation. Normalizing by the measured service duration makes the buildup visible without freezing host-dependent milliseconds into the prose. Real requests can add networking, preprocessing, variable service times and batch collection; those costs are not inferred from this graph.

```python
# Compute figure data for: Measured batch computation and a separately simulated queue
# Compute `visual_data` from `{"kind":"panels","panels":[{"kind":"bar","x":[0,1,2]...`
visual_data={"kind":"panels","panels":[{"kind":"bar","x":[0,1,2],"labels":["batch 1","batch 8","batch 32"],"xlabel":"resident-input batch shape","ylabel":"measured CPU milliseconds per batch","series":[{"label":"warm p50","y":[r["p50_ms"] for r in measurements]},{"label":"warm p95","y":[r["p95_ms"] for r in measurements]}]},{"kind":"line","x":request_index.tolist(),"xlabel":"request index (simulation)","ylabel":"simulated response / measured service","series":[{"label":"spaced arrivals","y":(stable_latency/service_ms).tolist()},{"label":"overloaded arrivals","y":(overloaded_latency/service_ms).tolist()}]}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:06:17.910117+00:00. JAX 0.9.2.

```text
{'batch': 1, 'first_call_ms': 36.89387487247586, 'p50_ms': 0.008020550012588501, 'p95_ms': 0.011514918878674488, 'steady_examples_per_second': 124679.72875058059}
{'batch': 8, 'first_call_ms': 21.6712080873549, 'p50_ms': 0.01529185101389885, 'p95_ms': 0.02352311275899409, 'steady_examples_per_second': 523154.4561040227}
{'batch': 32, 'first_call_ms': 20.841915626078844, 'p50_ms': 0.01591700129210949, 'p95_ms': 0.037331623025238514, 'steady_examples_per_second': 2010428.937758729}
measured batch-one service proxy (ms): 0.008020550012588501
simulated last stable/overloaded response (ms): 0.008020550012588501 0.19730553030967712
{'batch': 1, 'first_call_ms': 23.563208058476448, 'p50_ms': 0.007708091288805008, 'p95_ms': 0.010602222755551317, 'steady_examples_per_second': 129733.80341931977}
{'batch': 8, 'first_call_ms': 21.455375012010336, 'p50_ms': 0.013895798474550247, 'p95_ms': 0.044068414717912674, 'steady_examples_per_second': 575713.5881505312}
{'batch': 32, 'first_call_ms': 21.58812526613474, 'p50_ms': 0.017375219613313675, 'p95_ms': 0.04005841910839081, 'steady_examples_per_second': 1841703.3402835473}
measured batch-one service proxy (ms): 0.007708091288805008
simulated last stable/overloaded response (ms): 0.007708091288805008 0.1896190457046032
hypothetical planned replicas: 3
Analytic sample p95 / max (ms): 5.95000000000007 100.0
Requests beyond the declared deadline: 2
PASS: deployment-04

```

## Compute a replica estimate with explicit units

**Predict before running:** At 300 requests per second, 200 requests per second of effective capacity per replica, and 70% target utilization, how many replicas does the planning formula suggest?

```python
# Experiment — Compute a replica estimate with explicit units: This is arithmetic on hypothetical effective capacities, not an...
arrival_per_s=300.
capacity_per_s=200.
target_utilization=.7
# Run `math.ceil` to compute `replicas`.
replicas=math.ceil(arrival_per_s/(target_utilization*capacity_per_s))
# Assert invariant `replicas==3` holds
assert replicas==3
# Print the observed values to compare against the expected result.
print("hypothetical planned replicas:",replicas)
```

**Expected:** The formula suggests three replicas under the stated assumptions.

This is arithmetic on hypothetical effective capacities, not an executed autoscaling test.

## Estimate batch-collection delay

**Predict before running:** For eight requests arriving two milliseconds apart, how long do the earliest and average request wait for a full batch before computation?

```python
# Experiment — Estimate batch-collection delay: A throughput improvement can be overwhelmed by waiting for a...
batch=8
spacing_ms=2.
# Compute `collection_wait` from `(batch-1-np.arange(batch))*spacing_ms`
collection_wait=(batch-1-np.arange(batch))*spacing_ms
# Assert invariant `collection_wait[0]==14.` holds
assert collection_wait[0]==14.
# Assert invariant `collection_wait.mean()==7.` holds
assert collection_wait.mean()==7.
```

**Expected:** The first waits fourteen milliseconds; mean collection delay is seven milliseconds.

A throughput improvement can be overwhelmed by waiting for a batch at low arrival rates.

## Inspect the interpolation behind p95

**Predict before running:** With nineteen $1$-millisecond samples and one $100$-millisecond sample, must p95 equal an observed sample?

```python
# Experiment — Inspect the interpolation behind p95: These deliberately constructed values explain the percentile...
# Compute `illustrative_ms` from `np.array([1.]*19+[100.])`
illustrative_ms=np.array([1.]*19+[100.])
# Evaluate `np.percentile(illustrative_ms, 95, method='linear')` and convert the result into Python scalar/collection `interpolated_p95`.
interpolated_p95=float(np.percentile(illustrative_ms,95,method='linear'))
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(interpolated_p95,5.95,atol=1e-10)`
np.testing.assert_allclose(interpolated_p95,5.95,atol=1e-10)
# Assert invariant `interpolated_p95 not in illustrative_ms and illustrative_ms.max()...` holds
assert interpolated_p95 not in illustrative_ms and illustrative_ms.max()==100.
# Print the observed values to compare against the expected result.
print('Analytic sample p95 / max (ms):',interpolated_p95,illustrative_ms.max())
```

**Expected:** The interpolated p95 is $5.95$ milliseconds and the maximum is $100$ milliseconds.

These deliberately constructed values explain the percentile calculation. They are not measurements of the course model or an accelerator. The timed figure retains its actual measured samples.

## Make it yours

Simulate a burst of four requests arriving at time zero with a two-millisecond serial service duration. Derive all finish times and response times before executing.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Execute `np.testing.assert_allclose(burst_finish,[2.,4.,6.,8.])`
2. Execute `np.testing.assert_allclose(burst_latency,[2.,4.,6.,8.])`

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Simulate a burst of four requests arriving at time zero with a...
_,burst_finish,burst_latency = simulate_queue(...)  # TODO: compute _,burst_finish,burst_latency
# Execute `np.testing.assert_allclose(burst_finish,[2.,4.,6.,8.])`
np.testing.assert_allclose(burst_finish,[2.,4.,6.,8.])
# Execute `np.testing.assert_allclose(burst_latency,[2.,4.,6.,8.])`
np.testing.assert_allclose(burst_latency,[2.,4.,6.,8.])
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Simulate a burst of four requests arriving at time zero with a...
_,burst_finish,burst_latency=simulate_queue([0.,0.,0.,0.],2.)
# Execute `np.testing.assert_allclose(burst_finish,[2.,4.,6.,8.])`
np.testing.assert_allclose(burst_finish,[2.,4.,6.,8.])
# Execute `np.testing.assert_allclose(burst_latency,[2.,4.,6.,8.])`
np.testing.assert_allclose(burst_latency,[2.,4.,6.,8.])
```

</details>

## Expose burst sensitivity

**Transfer / diagnosis**

Compare five requests spaced three milliseconds apart with five simultaneous requests. Use two-millisecond service and compare maximum response latency.

<details><summary>Hint</summary>

The worker is idle between spaced requests but never idle during the burst.

</details>

### How to write: Expose burst sensitivity — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `sensitivity(...)` — Call `sensitivity` with your updated parameters or inputs from this lesson's workspace.
- `simulate_queue(...)` — Call `simulate_queue` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Compute `_,_,smooth` from `simulate_queue(np.arange(5)*3.,2.)`
2. Allocate initialized array `(_, _, burst)` with the specified shape and dtype.
3. Assert invariant `smooth.max()==2. and burst.max()==10.` holds

**Starter code scaffold (fill in the TODOs):**

```python
# Expose burst sensitivity (Transfer / diagnosis): An average arrival rate alone does not describe a burst or...
# Compute `_,_,smooth` from `simulate_queue(np.arange(5)*3.,2.)`
_,_,smooth = simulate_queue(...)  # TODO: compute _,_,smooth
# Allocate initialized array `(_, _, burst)` with the specified shape and dtype.
_,_,burst = simulate_queue(...)  # TODO: compute _,_,burst
# Assert invariant `smooth.max()==2. and burst.max()==10.` holds
assert smooth.max()  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Expose burst sensitivity (Transfer / diagnosis): An average arrival rate alone does not describe a burst or...
# Compute `_,_,smooth` from `simulate_queue(np.arange(5)*3.,2.)`
_,_,smooth=simulate_queue(np.arange(5)*3.,2.)
# Allocate initialized array `(_, _, burst)` with the specified shape and dtype.
_,_,burst=simulate_queue(np.zeros(5),2.)
# Assert invariant `smooth.max()==2. and burst.max()==10.` holds
assert smooth.max()==2. and burst.max()==10.
```

An average arrival rate alone does not describe a burst or tail latency.

</details>

## Reject a broken timeline

**Transfer / diagnosis**

Pass an out-of-order arrival vector and a zero service duration. Confirm both are rejected instead of returning a plausible chart.

<details><summary>Hint</summary>

Public simulation inputs need finite, ordered values and positive service time.

</details>

### How to write: Reject a broken timeline — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `timeline(...)` — Call `timeline` with your updated parameters or inputs from this lesson's workspace.
- `simulate_queue(...)` — Call `simulate_queue` with your updated parameters or inputs from this lesson's workspace.

**Step-by-step implementation plan:**
1. Iterate over `(arrivals, duration)` to step through the computation:

**Starter code scaffold (fill in the TODOs):**

```python
# Reject a broken timeline (Transfer / diagnosis): Validating assumptions prevents a simulation from silently...
# Iterate over `(arrivals, duration)` to step through the computation:
for arrivals,duration in [([2.,1.],2.),([0.,1.],0.)]:
    try:
        simulate_queue(arrivals,duration)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid timeline was accepted")
```

<details><summary>Reference solution and reasoning</summary>

```python
# Reject a broken timeline (Transfer / diagnosis): Validating assumptions prevents a simulation from silently...
# Iterate over `(arrivals, duration)` to step through the computation:
for arrivals,duration in [([2.,1.],2.),([0.,1.],0.)]:
    try:
        simulate_queue(arrivals,duration)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid timeline was accepted")
```

Validating assumptions prevents a simulation from silently supporting a false capacity story.

</details>

## Measure the margin below a response deadline

**Transfer / diagnosis**

For five simultaneous requests and a $2$-millisecond serial service, count responses that exceed a $6$-millisecond deadline. Explain why adding a faster model alone is not the only intervention.

<details><summary>Hint</summary>

Compute finish minus arrival for each request. Here a response exactly at the deadline is accepted; declare that convention.

</details>

### How to write: Measure the margin below a response deadline — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `deadline(...)` — Call `deadline` with your updated parameters or inputs from this lesson's workspace.
- `simulate_queue(...)` — Call `simulate_queue` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Execute `np.testing.assert_array_equal(deadline_latencies,[2.,4.,6.,8`
2. Evaluate `np.count_nonzero(deadline_latencies > 6.0)` and convert the result into Python scalar/collection `missed_deadlines`.
3. Assert invariant `missed_deadlines==2` holds
4. Print the observed values to compare against the expected result.

**Starter code scaffold (fill in the TODOs):**

```python
# Measure the margin below a response deadline (Transfer / diagnosis): Two responses miss the deadline in this serial simulation.
_,_,deadline_latencies = simulate_queue(...)  # TODO: compute _,_,deadline_latencies
# Execute `np.testing.assert_array_equal(deadline_latencies,[2.,4.,6.,8`
np.testing.assert_array_equal(deadline_latencies,[2.,4.,6.,8.,10.])
# Evaluate `np.count_nonzero(deadline_latencies > 6.0)` and convert the result into Python scalar/collection `missed_deadlines`.
missed_deadlines = int(...)  # TODO: compute missed_deadlines
# Assert invariant `missed_deadlines==2` holds
assert missed_deadlines  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print('Requests beyond the declared deadline:',missed_deadlines)
```

<details><summary>Reference solution and reasoning</summary>

```python
# Measure the margin below a response deadline (Transfer / diagnosis): Two responses miss the deadline in this serial simulation.
_,_,deadline_latencies=simulate_queue(np.zeros(5),2.)
# Execute `np.testing.assert_array_equal(deadline_latencies,[2.,4.,6.,8`
np.testing.assert_array_equal(deadline_latencies,[2.,4.,6.,8.,10.])
# Evaluate `np.count_nonzero(deadline_latencies > 6.0)` and convert the result into Python scalar/collection `missed_deadlines`.
missed_deadlines=int(np.count_nonzero(deadline_latencies>6.))
# Assert invariant `missed_deadlines==2` holds
assert missed_deadlines==2
# Print the observed values to compare against the expected result.
print('Requests beyond the declared deadline:',missed_deadlines)
```

Two responses miss the deadline in this serial simulation. Admission control, batching policy or additional workers could change the queue, but each requires a new model or measurement. The example does not execute a real autoscaler.

</details>

## Check your understanding

A batch of eight finishes in four milliseconds. Which statement follows for a fully occupied serial worker?

1. Every example experiences half a millisecond of response latency
2. The compute-throughput proxy is two thousand examples per second, while batch execution still takes four milliseconds
3. The service can guarantee the same throughput for every arrival pattern

<details><summary>Answer and explanation</summary>

The compute-throughput proxy is two thousand examples per second, while batch execution still takes four milliseconds

Eight examples per four milliseconds gives two thousand per second when batches remain full. A request still waits for collection and the full execution, plus any other boundary costs.

</details>

## Diagnose the result

If timings look impossibly small, check that completion is inside the timer. If a larger batch improves examples per second but misses a latency target, inspect collection and queue delay before choosing replicas. If a capacity calculation differs by a thousand, label milliseconds versus seconds at every conversion.

## Carry forward

- Measure completion and declare the boundary.
- Batch throughput, amortized cost and individual response time are different quantities.
- A queue simulation and replica formula guide experiments; neither replaces a real service load test.

## Keep your evidence

Warm timing samples, latency/throughput units, independent queue timeline and labeled simulation/replica assumptions. Keep the environment, observed outputs and your explanation of the figure.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html)
- [JAX asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)

