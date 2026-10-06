# Read an XProf trace

Phase 08: Performance diagnosis · about 100 minutes · CPU

## What you will be able to do

- Capture an actual warmed CPU trace with named steps.
- Identify nested intervals and avoid double-counting their durations.
- Separate an injected host wait from a completed model call.
- Explain how to inspect the trace in XProf and bound a bottleneck claim.

## The problem

Your timer says a training step is slow, but it cannot show whether the time was spent waiting for input or computing. We will capture a real JAX trace, add names that match the code, and read four steps from the recorded events before proposing an optimization.

## The idea

A trace places events on a shared timeline so we can inspect order, waiting and overlap. Begin with a specific question about a slow step and identify which recorded events can answer it. Host annotations and device operations are different evidence.

## Do not add nested durations twice

An outer step lasting $10$ milliseconds can contain a preprocessing region lasting $3$ milliseconds. Their sum is not $13$ milliseconds of elapsed work: the smaller region is already inside the larger one.

Inspect event starts, ends, nesting and lanes before aggregating durations. A gap may be waiting or uninstrumented work; it needs context before being assigned a cause.

The host-region plot summarizes recorded annotations, not GPU kernel execution. State a hypothesis, change one workload component and collect a comparable trace. A useful explanation predicts which interval changes and why, rather than only celebrating a shorter bar.

### Pause and reason

What should accompany a claimed trace-based optimization?

<details><summary>Compare your reasoning</summary>

The event boundary, comparable workload, specific intervention and repeated completed measurements showing its predicted effect. A changed unrelated interval is insufficient.

</details>

## Give the timeline a question

We want to know where this particular step spends time. Input creation and compilation happen before tracing. Each recorded step then contains an explicitly injected host sleep and a compiled model call that waits for its output. The sleep is a controlled stand-in for a stalled input stage; it is not a measurement of Grain, network storage or a real dataset.

Before running, predict which named interval should contain the wait. If the annotation surrounds the wrong code, a polished timeline will answer the wrong question. Keep the code region and annotation name together.

## Capture warmed work without hiding completion

Warm the exact shape and dtype before the trace. A model annotation ends after block_until_ready, so it includes host dispatch and waiting for the requested result. Without that wait, the host region might end while device work remains queued.

The four step annotations surround both inner regions. We check that each inner interval falls inside its step and that input wait precedes model execution. These temporal checks establish a relationship in the actual trace, not a simulated schedule.

## Read a trace without adding the same time twice

An outer step includes its inner wait and model regions. Adding all three durations would double-count most of the step. Instead compare the inner intervals and calculate the remaining outer time as annotation and host overhead. Small timing rounding differences deserve an explicit tolerance; a large negative remainder means your events were not matched correctly.

The exported trace’s complete events express timestamps and durations in microseconds. Convert to seconds before comparing with a perf_counter report. An event lasting a thousand microseconds lasts one millisecond; a unit mistake can overwhelm any optimization effect.

## Open the same evidence in XProf

The program prints the trace directory and retains the actual XPlane and Perfetto files there. To choose a durable folder, set JAX_COURSE_TRACE_DIR before running. Open the official XProf viewer in a compatible separately installed environment, select this log directory/profile, then open Trace Viewer and locate learner_step. Expand its host thread and inspect input_wait and model_and_wait.

Select one step and record its start, duration, enclosing thread and child intervals. On accelerator hardware, inspect device lanes and correlate them with host submission; an empty host region does not prove an idle device. This CPU run does not claim an XProf UI session or accelerator trace was executed.

**Optional viewer in a separate compatible Python environment**

```sh
python -m pip install xprof
xprof --port 8791 /absolute/path/to/your/trace-directory
```

**Expected:** Replace the final path with the trace directory printed by the lesson. The viewer prints a local URL. Select the recorded run and trace_viewer. The viewer dependency and GUI are not covered by the course CPU receipt.

## Make one change and repeat the same question

To investigate a real input bottleneck, replace the injected sleep with the actual input fetch while retaining the annotation boundary. Capture both versions under comparable load. Do not report a speedup from deleting a synthetic delay as if it optimized real model computation.

For compilation problems, capture a separate first-call trace and keep it labeled. Mixing cold and warm steps hides the distinction the previous lessons established. Small traces reduce observer overhead, but any profiler can perturb timings; confirm an improvement with an independent synchronized benchmark.

## Prepare a traceable workload

Create a Python file in the course environment. Add this block after the preceding block, then run the assembled file.

```python
import gzip
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import jax
import jax.numpy as jnp
trace_root=Path(os.environ.get('JAX_COURSE_TRACE_DIR') or tempfile.mkdtemp(prefix='jax-course-trace-')).resolve()
trace_root.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(31)
host=rng.normal(size=(48,32)).astype(np.float32)
weights=rng.normal(size=(32,16)).astype(np.float32)
x,w=jnp.asarray(host),jnp.asarray(weights)
compiled=jax.jit(lambda a,b:jnp.tanh(a@b))
compiled(x,w).block_until_ready()
```

Inputs and compilation precede the measured steps. The output remains independently checked against NumPy.

## Capture named intervals

Create a Python file in the course environment. Add this block after the preceding block, then run the assembled file.

```python
with jax.profiler.trace(trace_root,create_perfetto_trace=True):
    for index in range(4):
        with jax.profiler.StepTraceAnnotation('learner_step',step_num=index):
            with jax.profiler.TraceAnnotation('input_wait'):
                time.sleep(.001)  # injected host delay; not a measured real input pipeline
            with jax.profiler.TraceAnnotation('model_and_wait'):
                result=compiled(x,w)
                result.block_until_ready()
np.testing.assert_allclose(result,np.tanh(host@weights),rtol=2e-5,atol=2e-5)
traces=sorted(trace_root.rglob('perfetto_trace.json.gz'),key=lambda p:p.stat().st_mtime_ns)
assert traces and list(trace_root.rglob('*.xplane.pb'))
events=json.loads(gzip.decompress(traces[-1].read_bytes()))['traceEvents']
def complete(name):
    return sorted([e for e in events if e.get('ph')=='X' and e.get('name')==name],key=lambda e:e['ts'])
steps=complete('learner_step');input_events=complete('input_wait');model_events=complete('model_and_wait')
assert len(steps)==len(input_events)==len(model_events)==4
for outer,waiting,compute in zip(steps,input_events,model_events):
    for inner in (waiting,compute):
        assert outer['ts']<=inner['ts']<=inner['ts']+inner['dur']<=outer['ts']+outer['dur']+1
    assert waiting['ts']+waiting['dur']<=compute['ts']+1
input_us=[e['dur'] for e in input_events]
model_us=[e['dur'] for e in model_events]
step_us=[e['dur'] for e in steps]
assert all(t>=0 for t in input_us+model_us+step_us)
print('Trace directory:',trace_root)
print(json.dumps({'input_wait_us':input_us,'model_and_wait_us':model_us,'step_us':step_us},indent=2))
print('Four named steps verified; host annotations are not isolated device kernels.')

```

The outer annotation provides context, while two nonoverlapping child regions distinguish injected waiting from completed computation.

## Run the example

```python
import gzip
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import jax
import jax.numpy as jnp
trace_root=Path(os.environ.get('JAX_COURSE_TRACE_DIR') or tempfile.mkdtemp(prefix='jax-course-trace-')).resolve()
trace_root.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(31)
host=rng.normal(size=(48,32)).astype(np.float32)
weights=rng.normal(size=(32,16)).astype(np.float32)
x,w=jnp.asarray(host),jnp.asarray(weights)
compiled=jax.jit(lambda a,b:jnp.tanh(a@b))
compiled(x,w).block_until_ready()

with jax.profiler.trace(trace_root,create_perfetto_trace=True):
    for index in range(4):
        with jax.profiler.StepTraceAnnotation('learner_step',step_num=index):
            with jax.profiler.TraceAnnotation('input_wait'):
                time.sleep(.001)  # injected host delay; not a measured real input pipeline
            with jax.profiler.TraceAnnotation('model_and_wait'):
                result=compiled(x,w)
                result.block_until_ready()
np.testing.assert_allclose(result,np.tanh(host@weights),rtol=2e-5,atol=2e-5)
traces=sorted(trace_root.rglob('perfetto_trace.json.gz'),key=lambda p:p.stat().st_mtime_ns)
assert traces and list(trace_root.rglob('*.xplane.pb'))
events=json.loads(gzip.decompress(traces[-1].read_bytes()))['traceEvents']
def complete(name):
    return sorted([e for e in events if e.get('ph')=='X' and e.get('name')==name],key=lambda e:e['ts'])
steps=complete('learner_step');input_events=complete('input_wait');model_events=complete('model_and_wait')
assert len(steps)==len(input_events)==len(model_events)==4
for outer,waiting,compute in zip(steps,input_events,model_events):
    for inner in (waiting,compute):
        assert outer['ts']<=inner['ts']<=inner['ts']+inner['dur']<=outer['ts']+outer['dur']+1
    assert waiting['ts']+waiting['dur']<=compute['ts']+1
input_us=[e['dur'] for e in input_events]
model_us=[e['dur'] for e in model_events]
step_us=[e['dur'] for e in steps]
assert all(t>=0 for t in input_us+model_us+step_us)
print('Trace directory:',trace_root)
print(json.dumps({'input_wait_us':input_us,'model_and_wait_us':model_us,'step_us':step_us},indent=2))
print('Four named steps verified; host annotations are not isolated device kernels.')

```

Expected: A real trace directory, four input-wait intervals, four model-and-wait intervals and four step intervals in microseconds. Values vary by run and include profiler overhead.

## Where four annotated steps spent host time

**Predict:** Which line should reflect the deliberately injected host wait?

![Where four annotated steps spent host time](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis identifies the four captured steps. The vertical axis reports microseconds from actual trace events. The input-wait series contains a requested one-millisecond host sleep, so its duration should be interpreted around that scale with scheduler and profiler overhead; it is not a fixed measured constant.

The model-and-wait series covers dispatch through result readiness. Compare the numeric ticks and the recorded sample list to see which region takes longer in this run. Variation between steps can reflect scheduling or profiling overhead; it does not by itself demonstrate changing model work.

### Connect it to the computation

Both plotted series are child regions within a learner_step annotation. They do not include warmup and they are not two isolated accelerator kernels. The code verifies their temporal order and checks the model output independently before interpreting the trace.

The outer step also contains small amounts of other host work. Do not add the outer duration to these inner durations, or call the injected sleep a measured production data stall. Replace the sleep with a real input fetch to investigate an actual pipeline.

```python
visual_data={'kind':'line','x':[0,1,2,3],'xlabel':'captured step','ylabel':'host annotation duration (microseconds)','series':[{'label':'injected input wait','y':input_us},{'label':'model call and completion wait','y':model_us}]}
```

## Recorded reference execution

CPU run: 2026-10-06T23:01:12.175848+00:00. JAX 0.9.2.

```text
Trace directory: /private/var/folders/mp/_mq7srqx2y10p5nhjz9m7hj40000gn/T/jax-course-trace-dfxhhvyp
{
  "input_wait_us": [
    1261,
    1256,
    1264,
    1257
  ],
  "model_and_wait_us": [
    63,
    34,
    59,
    42
  ],
  "step_us": [
    1331,
    1294,
    1328,
    1302
  ]
}
Four named steps verified; host annotations are not isolated device kernels.
Trace directory: /private/var/folders/mp/_mq7srqx2y10p5nhjz9m7hj40000gn/T/jax-course-trace-xf6o01xs
{
  "input_wait_us": [
    1260,
    1257,
    1258,
    1257
  ],
  "model_and_wait_us": [
    57,
    47,
    39,
    36
  ],
  "step_us": [
    1322,
    1309,
    1300,
    1296
  ]
}
Four named steps verified; host annotations are not isolated device kernels.
Unattributed outer time, microseconds: [5, 5, 3, 3]
Wrong double-counted totals: [2639, 2613, 2597, 2589]
{'minimum_ms': 0.036, 'median_ms': 0.043, 'maximum_ms': 0.057}
Hypothetical union: 12 microseconds; summed duration: 19.
PASS: performance-03

```

## Account for the enclosing step

**Predict before running:** What happens if you sum outer and inner durations as though they were independent work?

```python
overhead=np.asarray(step_us)-np.asarray(input_us)-np.asarray(model_us)
assert np.all(overhead>=-2)
wrong=np.asarray(step_us)+np.asarray(input_us)+np.asarray(model_us)
assert np.all(wrong>=np.asarray(step_us))
print('Unattributed outer time, microseconds:',overhead.tolist())
print('Wrong double-counted totals:',wrong.tolist())
```

**Expected:** The remainder is nonnegative within rounding; the double-counted totals exceed the step durations.

Outer and child intervals describe nested boundaries. The remainder is host/annotation work inside the step but outside its two children; it is not automatically device idle time.

## Make it yours

Convert every model interval to milliseconds and summarize its minimum, median and maximum. Keep the individual samples and explain why four observations cannot establish a tail-latency service guarantee.

<details><summary>Reference solution</summary>

```python
model_ms=np.asarray(model_us)/1000
summary={'minimum_ms':float(model_ms.min()),'median_ms':float(np.median(model_ms)),'maximum_ms':float(model_ms.max())}
assert summary['minimum_ms']<=summary['median_ms']<=summary['maximum_ms']
np.testing.assert_allclose(model_ms*1000,model_us)
print(summary)
```

</details>

## Compute a union instead of summing overlap

**Transfer**

For hypothetical intervals [0,10], [2,7] and [8,12] microseconds, calculate their union duration and compare with their summed durations. Explain why this matters for nested host events.

<details><summary>Hint</summary>

Sort endpoints and merge intervals that intersect.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
def union_duration(intervals):
    total=0;left=right=None
    for start,end in sorted(intervals):
        if end<start:raise ValueError('negative interval')
        if right is None:left,right=start,end
        elif start<=right:right=max(right,end)
        else:total+=right-left;left,right=start,end
    return total+(0 if right is None else right-left)
assert union_duration([(0,10),(2,7),(8,12)])==12
assert sum(b-a for a,b in [(0,10),(2,7),(8,12)])==19
print('Hypothetical union: 12 microseconds; summed duration: 19.')
```

A union describes time covered by any interval. A sum counts overlap repeatedly. Neither alone says which device was busy; lane and dependency information still matter.

</details>

## Check your understanding

A step annotation lasts longer than its model annotation. What can this alone establish?

1. The accelerator was idle for the entire difference.
2. The whole difference is input loading.
3. The step includes time outside the model region; inspect the other regions and lanes to explain it.

<details><summary>Answer and explanation</summary>

The step includes time outside the model region; inspect the other regions and lanes to explain it.

The annotation boundaries establish inclusion, not a universal attribution of the remaining time. Our injected wait is known from code; an actual workload needs trace evidence.

</details>

## Diagnose the result

If expected events are absent, verify that the trace scope surrounds execution, warmup is outside it, names match exactly, and the selected file belongs to the latest capture. If nested durations do not fit, inspect timestamps and thread IDs rather than adding unrelated lanes.

## Carry forward

- Name trace regions for a concrete diagnostic question.
- Read units, threads and nesting before attributing time.
- Confirm trace-driven changes with an independent benchmark.

## Keep your evidence

Keep the actual XPlane and Perfetto paths, named interval durations and nesting check, units and an interpretation separating injected input delay from completed computation.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [JAX profiling guide and XProf workflow](https://docs.jax.dev/en/latest/profiling.html)

