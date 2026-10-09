"""Read an XProf trace: worked experiments and reference solutions. CPU checks."""

# Prepare a traceable workload
# Step 1 — Prepare a traceable workload: Inputs and compilation precede the measured steps.
# Import gzip for this computation.
import gzip
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import jax
import jax.numpy as jnp
# Read `trace_root` from environment configuration (with a default fallback).
trace_root=Path(os.environ.get('JAX_COURSE_TRACE_DIR') or tempfile.mkdtemp(prefix='jax-course-trace-')).resolve()
# Run `trace_root.mkdir` to perform the next check or state transition.
trace_root.mkdir(parents=True,exist_ok=True)
# Draw pseudorandom samples for `rng` using the explicit RNG state.
rng=np.random.default_rng(31)
# Cast or evaluate `host` in explicit floating-point precision.
host=rng.normal(size=(48,32)).astype(np.float32)
# Cast or evaluate `weights` in explicit floating-point precision.
weights=rng.normal(size=(32,16)).astype(np.float32)
# Create device-backed JAX array `(x, w)`.
x,w=jnp.asarray(host),jnp.asarray(weights)
# Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
compiled=jax.jit(lambda a,b:jnp.tanh(a@b))
# Synchronize host execution until asynchronous device computation completes.
compiled(x,w).block_until_ready()

# Capture named intervals
# Step 2 — Capture named intervals: The outer annotation provides context, while two nonoverlapping...
# Enter `jax.profiler.trace(trace_root, create_perfett` context block:
with jax.profiler.trace(trace_root,create_perfetto_trace=True):
    # Loop over `index` in `range(4)`:
    for index in range(4):
        # Enter managed runtime/context scope for this block:
        with jax.profiler.StepTraceAnnotation('learner_step',step_num=index):
            # Enter managed runtime/context scope for this block:
            with jax.profiler.TraceAnnotation('input_wait'):
                time.sleep(.001)  # injected host delay; not a measured real input pipeline
            # Enter managed runtime/context scope for this block:
            with jax.profiler.TraceAnnotation('model_and_wait'):
                # Run `compiled` to compute `result`.
                result=compiled(x,w)
                # Synchronize host execution until asynchronous device computation completes.
                result.block_until_ready()
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(result,np.tanh(host@weights),rtol=2e-5,atol=2e-5)
# Run `sorted` to compute `traces`.
traces=sorted(trace_root.rglob('perfetto_trace.json.gz'),key=lambda p:p.stat().st_mtime_ns)
# Verify contract: `traces and list(trace_root.rglob('*.xplane.pb'))`.
assert traces and list(trace_root.rglob('*.xplane.pb'))
# Read or serialize artifact data on disk (`events`).
events=json.loads(gzip.decompress(traces[-1].read_bytes()))['traceEvents']
# Function `complete(name)` implementing this stage's computation:
def complete(name):
    # Return `sorted([e for e in events if e.get('ph') == 'X' and e.get('name') == name], key=lambda e: e['ts'])` to the caller.
    return sorted([e for e in events if e.get('ph')=='X' and e.get('name')==name],key=lambda e:e['ts'])
# Run `complete` to compute `steps`.
# Run `complete` to compute `input_events`.
# Run `complete` to compute `model_events`.
steps=complete('learner_step');input_events=complete('input_wait');model_events=complete('model_and_wait')
# Verify contract: `len(steps) == len(input_events) == len(model_events) == 4`.
assert len(steps)==len(input_events)==len(model_events)==4
# Iterate over `(outer, waiting, compute)` to step through the computation:
for outer,waiting,compute in zip(steps,input_events,model_events):
    # Iterate over `inner` to step through the computation:
    for inner in (waiting,compute):
        # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
        assert outer['ts']<=inner['ts']<=inner['ts']+inner['dur']<=outer['ts']+outer['dur']+1
    # Verify contract: `waiting['ts'] + waiting['dur'] <= compute['ts'] + 1`.
    assert waiting['ts']+waiting['dur']<=compute['ts']+1
# Evaluate `input_us` from the current inputs and state.
input_us=[e['dur'] for e in input_events]
# Evaluate `model_us` from the current inputs and state.
model_us=[e['dur'] for e in model_events]
# Evaluate `step_us` from the current inputs and state.
step_us=[e['dur'] for e in steps]
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert all(t>=0 for t in input_us+model_us+step_us)
# Print diagnostic summary of the computed outputs.
print('Trace directory:',trace_root)
# Print diagnostic summary of the computed outputs.
print(json.dumps({'input_wait_us':input_us,'model_and_wait_us':model_us,'step_us':step_us},indent=2))
# Print diagnostic summary of the computed outputs.
print('Four named steps verified; host annotations are not isolated device kernels.')

# Step 1 — Prepare a traceable workload: Inputs and compilation precede the measured steps.
# Import gzip for this computation.
import gzip
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import jax
import jax.numpy as jnp
# Read `trace_root` from environment configuration (with a default fallback).
trace_root=Path(os.environ.get('JAX_COURSE_TRACE_DIR') or tempfile.mkdtemp(prefix='jax-course-trace-')).resolve()
# Run `trace_root.mkdir` to perform the next check or state transition.
trace_root.mkdir(parents=True,exist_ok=True)
# Draw pseudorandom samples for `rng` using the explicit RNG state.
rng=np.random.default_rng(31)
# Cast or evaluate `host` in explicit floating-point precision.
host=rng.normal(size=(48,32)).astype(np.float32)
# Cast or evaluate `weights` in explicit floating-point precision.
weights=rng.normal(size=(32,16)).astype(np.float32)
# Create device-backed JAX array `(x, w)`.
x,w=jnp.asarray(host),jnp.asarray(weights)
# Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
compiled=jax.jit(lambda a,b:jnp.tanh(a@b))
# Synchronize host execution until asynchronous device computation completes.
compiled(x,w).block_until_ready()

# Step 2 — Capture named intervals: The outer annotation provides context, while two nonoverlapping...
# Enter `jax.profiler.trace(trace_root, create_perfett` context block:
with jax.profiler.trace(trace_root,create_perfetto_trace=True):
    # Loop over `index` in `range(4)`:
    for index in range(4):
        # Enter managed runtime/context scope for this block:
        with jax.profiler.StepTraceAnnotation('learner_step',step_num=index):
            # Enter managed runtime/context scope for this block:
            with jax.profiler.TraceAnnotation('input_wait'):
                time.sleep(.001)  # injected host delay; not a measured real input pipeline
            # Enter managed runtime/context scope for this block:
            with jax.profiler.TraceAnnotation('model_and_wait'):
                # Run `compiled` to compute `result`.
                result=compiled(x,w)
                # Synchronize host execution until asynchronous device computation completes.
                result.block_until_ready()
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(result,np.tanh(host@weights),rtol=2e-5,atol=2e-5)
# Run `sorted` to compute `traces`.
traces=sorted(trace_root.rglob('perfetto_trace.json.gz'),key=lambda p:p.stat().st_mtime_ns)
# Verify contract: `traces and list(trace_root.rglob('*.xplane.pb'))`.
assert traces and list(trace_root.rglob('*.xplane.pb'))
# Read or serialize artifact data on disk (`events`).
events=json.loads(gzip.decompress(traces[-1].read_bytes()))['traceEvents']
# Function `complete(name)` implementing this stage's computation:
def complete(name):
    # Return `sorted([e for e in events if e.get('ph') == 'X' and e.get('name') == name], key=lambda e: e['ts'])` to the caller.
    return sorted([e for e in events if e.get('ph')=='X' and e.get('name')==name],key=lambda e:e['ts'])
# Run `complete` to compute `steps`.
# Run `complete` to compute `input_events`.
# Run `complete` to compute `model_events`.
steps=complete('learner_step');input_events=complete('input_wait');model_events=complete('model_and_wait')
# Verify contract: `len(steps) == len(input_events) == len(model_events) == 4`.
assert len(steps)==len(input_events)==len(model_events)==4
# Iterate over `(outer, waiting, compute)` to step through the computation:
for outer,waiting,compute in zip(steps,input_events,model_events):
    # Iterate over `inner` to step through the computation:
    for inner in (waiting,compute):
        # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
        assert outer['ts']<=inner['ts']<=inner['ts']+inner['dur']<=outer['ts']+outer['dur']+1
    # Verify contract: `waiting['ts'] + waiting['dur'] <= compute['ts'] + 1`.
    assert waiting['ts']+waiting['dur']<=compute['ts']+1
# Evaluate `input_us` from the current inputs and state.
input_us=[e['dur'] for e in input_events]
# Evaluate `model_us` from the current inputs and state.
model_us=[e['dur'] for e in model_events]
# Evaluate `step_us` from the current inputs and state.
step_us=[e['dur'] for e in steps]
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert all(t>=0 for t in input_us+model_us+step_us)
# Print diagnostic summary of the computed outputs.
print('Trace directory:',trace_root)
# Print diagnostic summary of the computed outputs.
print(json.dumps({'input_wait_us':input_us,'model_and_wait_us':model_us,'step_us':step_us},indent=2))
# Print diagnostic summary of the computed outputs.
print('Four named steps verified; host annotations are not isolated device kernels.')

# Figure data experiment
# Compute figure data for: Where four annotated steps spent host time
# Evaluate `visual_data` from the current inputs and state.
visual_data={'kind':'line','x':[0,1,2,3],'xlabel':'captured step','ylabel':'host annotation duration (microseconds)','series':[{'label':'injected input wait','y':input_us},{'label':'model call and completion wait','y':model_us}]}

# Experiment: Account for the enclosing step
# Experiment — Account for the enclosing step: Outer and child intervals describe nested boundaries.
overhead=np.asarray(step_us)-np.asarray(input_us)-np.asarray(model_us)
# Verify contract: `np.all(overhead >= -2)`.
assert np.all(overhead>=-2)
# Convert `wrong` to a host NumPy array for inspection or verification.
wrong=np.asarray(step_us)+np.asarray(input_us)+np.asarray(model_us)
# Verify contract: `np.all(wrong >= np.asarray(step_us))`.
assert np.all(wrong>=np.asarray(step_us))
# Print the observed values to compare against the expected result.
print('Unattributed outer time, microseconds:',overhead.tolist())
# Print diagnostic summary of the computed outputs.
print('Wrong double-counted totals:',wrong.tolist())

# Reference solution. Try the exercise before reading this.
# Exercise solution: Convert every model interval to milliseconds and summarize its...
model_ms=np.asarray(model_us)/1000
# Aggregate array values to compute `summary`.
summary={'minimum_ms':float(model_ms.min()),'median_ms':float(np.median(model_ms)),'maximum_ms':float(model_ms.max())}
# Verify contract: `summary['minimum_ms'] <= summary['median_ms'] <= summary['maximum_ms...`.
assert summary['minimum_ms']<=summary['median_ms']<=summary['maximum_ms']
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(model_ms*1000,model_us)
# Print the observed values to compare against the expected result.
print(summary)

# Reference practice: Compute a union instead of summing overlap
# Compute a union instead of summing overlap (Transfer): A union describes time covered by any interval.
def union_duration(intervals):
    # Evaluate `total` from the current inputs and state.
    # Evaluate `left, right` from the current inputs and state.
    total=0;left=right=None
    # Iterate over `(start, end)` to step through the computation:
    for start,end in sorted(intervals):
        # Guard input contract (`end < start`) and fail fast if violated.
        if end<start:raise ValueError('negative interval')
        # Branch on condition `right is None`:
        if right is None:left,right=start,end
        elif start<=right:right=max(right,end)
        else:total+=right-left;left,right=start,end
    # Return `total + (0 if right is None else right - left)` to the caller.
    return total+(0 if right is None else right-left)
# Verify contract: `union_duration([(0, 10), (2, 7), (8, 12)]) == 12`.
assert union_duration([(0,10),(2,7),(8,12)])==12
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert sum(b-a for a,b in [(0,10),(2,7),(8,12)])==19
# Print the observed values to compare against the expected result.
print('Hypothetical union: 12 microseconds; summed duration: 19.')
print("PASS: performance-03")
