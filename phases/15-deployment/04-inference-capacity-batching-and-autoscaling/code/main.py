"""Inference capacity, batching, and autoscaling: worked experiments and reference solutions. CPU checks."""

# 1. Define and check the inference workload
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
    # Compute `np.testing.assert_allclose(first,expected,rtol` as `2e-5,atol=2e-6)`.
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

# 2. Measure each batch shape separately
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

# 3. Model a queue with an independent hand-check
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
# Assert that `np.allclose(stable_latency,service_ms)`.
assert np.allclose(stable_latency,service_ms)
# Assert that `np.isclose(overloaded_latency[-1],24.6*service_ms)`.
assert np.isclose(overloaded_latency[-1],24.6*service_ms)
# Print diagnostic summary of the computed outputs.
print('measured batch-one service proxy (ms):',service_ms)
# Print diagnostic summary of the computed outputs.
print('simulated last stable/overloaded response (ms):',stable_latency[-1],overloaded_latency[-1])

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
    # Compute `np.testing.assert_allclose(first,expected,rtol` as `2e-5,atol=2e-6)`.
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
# Assert that `np.allclose(stable_latency,service_ms)`.
assert np.allclose(stable_latency,service_ms)
# Assert that `np.isclose(overloaded_latency[-1],24.6*service_ms)`.
assert np.isclose(overloaded_latency[-1],24.6*service_ms)
# Print diagnostic summary of the computed outputs.
print('measured batch-one service proxy (ms):',service_ms)
# Print diagnostic summary of the computed outputs.
print('simulated last stable/overloaded response (ms):',stable_latency[-1],overloaded_latency[-1])

# Figure data experiment
# Compute figure data for: Measured batch computation and a separately simulated queue
# Compute `visual_data` from `{"kind":"panels","panels":[{"kind":"bar","x":[0,1,2]...`
visual_data={"kind":"panels","panels":[{"kind":"bar","x":[0,1,2],"labels":["batch 1","batch 8","batch 32"],"xlabel":"resident-input batch shape","ylabel":"measured CPU milliseconds per batch","series":[{"label":"warm p50","y":[r["p50_ms"] for r in measurements]},{"label":"warm p95","y":[r["p95_ms"] for r in measurements]}]},{"kind":"line","x":request_index.tolist(),"xlabel":"request index (simulation)","ylabel":"simulated response / measured service","series":[{"label":"spaced arrivals","y":(stable_latency/service_ms).tolist()},{"label":"overloaded arrivals","y":(overloaded_latency/service_ms).tolist()}]}]}

# Experiment: Compute a replica estimate with explicit units
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

# Experiment: Estimate batch-collection delay
# Experiment — Estimate batch-collection delay: A throughput improvement can be overwhelmed by waiting for a...
batch=8
spacing_ms=2.
# Compute `collection_wait` from `(batch-1-np.arange(batch))*spacing_ms`
collection_wait=(batch-1-np.arange(batch))*spacing_ms
# Assert invariant `collection_wait[0]==14.` holds
assert collection_wait[0]==14.
# Assert invariant `collection_wait.mean()==7.` holds
assert collection_wait.mean()==7.

# Experiment: Inspect the interpolation behind p95
# Experiment — Inspect the interpolation behind p95: These deliberately constructed values explain the percentile...
# Compute `illustrative_ms` from `np.array([1.]*19+[100.])`
illustrative_ms=np.array([1.]*19+[100.])
# Evaluate `np.percentile(illustrative_ms, 95, method='linear')` and convert the result into Python scalar/collection `interpolated_p95`.
interpolated_p95=float(np.percentile(illustrative_ms,95,method='linear'))
# Compute `np.testing.assert_allclose(interpolated_p95,5.95,atol` as `1e-10)`.
np.testing.assert_allclose(interpolated_p95,5.95,atol=1e-10)
# Assert that `interpolated_p95 not in illustrative_ms and illustrative_ms.max()==100.`.
assert interpolated_p95 not in illustrative_ms and illustrative_ms.max()==100.
# Print the observed values to compare against the expected result.
print('Analytic sample p95 / max (ms):',interpolated_p95,illustrative_ms.max())

# Reference solution. Try the exercise before reading this.
# Exercise solution: Simulate a burst of four requests arriving at time zero with a...
_,burst_finish,burst_latency=simulate_queue([0.,0.,0.,0.],2.)
# Execute `np.testing.assert_allclose(burst_finish,[2.,4.,6.,8.])`
np.testing.assert_allclose(burst_finish,[2.,4.,6.,8.])
# Execute `np.testing.assert_allclose(burst_latency,[2.,4.,6.,8.])`
np.testing.assert_allclose(burst_latency,[2.,4.,6.,8.])

# Reference practice: Expose burst sensitivity
# Expose burst sensitivity (Transfer / diagnosis): An average arrival rate alone does not describe a burst or...
# Compute `_,_,smooth` from `simulate_queue(np.arange(5)*3.,2.)`
_,_,smooth=simulate_queue(np.arange(5)*3.,2.)
# Allocate initialized array `(_, _, burst)` with the specified shape and dtype.
_,_,burst=simulate_queue(np.zeros(5),2.)
# Assert invariant `smooth.max()==2. and burst.max()==10.` holds
assert smooth.max()==2. and burst.max()==10.

# Reference practice: Reject a broken timeline
# Reject a broken timeline (Transfer / diagnosis): Validating assumptions prevents a simulation from silently...
# Iterate over `(arrivals, duration)` to step through the computation:
for arrivals,duration in [([2.,1.],2.),([0.,1.],0.)]:
    try:
        simulate_queue(arrivals,duration)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid timeline was accepted")

# Reference practice: Measure the margin below a response deadline
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
print("PASS: deployment-04")
