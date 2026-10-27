"""Read an XProf trace: worked experiments and reference solutions. CPU checks."""

# Prepare a traceable workload
# Step 1 — Prepare a traceable workload: Inputs and compilation precede the measured steps.
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
trace_root.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(31)
host=rng.normal(size=(48,32)).astype(np.float32)
weights=rng.normal(size=(32,16)).astype(np.float32)
x,w=jnp.asarray(host),jnp.asarray(weights)
# Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
compiled=jax.jit(lambda a,b:jnp.tanh(a@b))
# Wait for device work to finish before timing it or comparing its result on the host.
compiled(x,w).block_until_ready()

# Capture named intervals
# Step 2 — Capture named intervals
with jax.profiler.trace(trace_root,create_perfetto_trace=True):
    for index in range(4):
        with jax.profiler.StepTraceAnnotation('learner_step',step_num=index):
            with jax.profiler.TraceAnnotation('input_wait'):
                time.sleep(.001)  # injected host delay; not a measured real input pipeline
            with jax.profiler.TraceAnnotation('model_and_wait'):
                result=compiled(x,w)
                # Wait for device work to finish before timing it or comparing its result on the host.
                result.block_until_ready()
np.testing.assert_allclose(result,np.tanh(host@weights),rtol=2e-5,atol=2e-5)
traces=sorted(trace_root.rglob('perfetto_trace.json.gz'),key=lambda p:p.stat().st_mtime_ns)
assert traces and list(trace_root.rglob('*.xplane.pb'))
events=json.loads(gzip.decompress(traces[-1].read_bytes()))['traceEvents']
def complete(name):
    return sorted([e for e in events if e.get('ph')=='X' and e.get('name')==name],key=lambda e:e['ts'])
steps=complete('learner_step')
input_events=complete('input_wait')
model_events=complete('model_and_wait')
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

# Step 3: Verify invariants on the completed state
step_us=[e['dur'] for e in steps]
assert all(t>=0 for t in input_us+model_us+step_us)

# Step 1 — Prepare a traceable workload: Inputs and compilation precede the measured steps.
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
trace_root.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(31)
host=rng.normal(size=(48,32)).astype(np.float32)
weights=rng.normal(size=(32,16)).astype(np.float32)
x,w=jnp.asarray(host),jnp.asarray(weights)
# Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
compiled=jax.jit(lambda a,b:jnp.tanh(a@b))
# Wait for device work to finish before timing it or comparing its result on the host.
compiled(x,w).block_until_ready()

# Step 2 — Capture named intervals
with jax.profiler.trace(trace_root,create_perfetto_trace=True):
    for index in range(4):
        with jax.profiler.StepTraceAnnotation('learner_step',step_num=index):
            with jax.profiler.TraceAnnotation('input_wait'):
                time.sleep(.001)  # injected host delay; not a measured real input pipeline
            with jax.profiler.TraceAnnotation('model_and_wait'):
                result=compiled(x,w)
                # Wait for device work to finish before timing it or comparing its result on the host.
                result.block_until_ready()
np.testing.assert_allclose(result,np.tanh(host@weights),rtol=2e-5,atol=2e-5)
traces=sorted(trace_root.rglob('perfetto_trace.json.gz'),key=lambda p:p.stat().st_mtime_ns)
assert traces and list(trace_root.rglob('*.xplane.pb'))
events=json.loads(gzip.decompress(traces[-1].read_bytes()))['traceEvents']
def complete(name):
    return sorted([e for e in events if e.get('ph')=='X' and e.get('name')==name],key=lambda e:e['ts'])
steps=complete('learner_step')
input_events=complete('input_wait')
model_events=complete('model_and_wait')
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

# Figure data experiment
from pathlib import Path
import json
retained_events = [{'condition': '2 ms injected wait', 'name': 'learner_step', 'start_us': 122, 'duration_us': 2665, 'end_us': 2787, 'pid': 701, 'tid': 164376313}, {'condition': '2 ms injected wait', 'name': 'input_wait', 'start_us': 125, 'duration_us': 2520, 'end_us': 2645, 'pid': 701, 'tid': 164376313}, {'condition': '2 ms injected wait', 'name': 'model_and_wait', 'start_us': 2668, 'duration_us': 119, 'end_us': 2787, 'pid': 701, 'tid': 164376313}, {'condition': '0.5 ms injected wait', 'name': 'learner_step', 'start_us': 2796, 'duration_us': 732, 'end_us': 3528, 'pid': 701, 'tid': 164376313}, {'condition': '0.5 ms injected wait', 'name': 'input_wait', 'start_us': 2798, 'duration_us': 642, 'end_us': 3440, 'pid': 701, 'tid': 164376313}, {'condition': '0.5 ms injected wait', 'name': 'model_and_wait', 'start_us': 3448, 'duration_us': 79, 'end_us': 3527, 'pid': 701, 'tid': 164376313}]
# Use embedded measured rows for downloaded standalone lesson scripts. When the
# workspace excerpt is present, verify that its event records match exactly.
excerpt_path = Path('phases/08-performance/03-read-an-xprof-trace/data/retained_cpu_timeline_p2_81.json')
if excerpt_path.exists():
    excerpt = json.loads(excerpt_path.read_text())
    assert excerpt['events'] == retained_events
conditions = ['2 ms injected wait', '0.5 ms injected wait']
lookup = {(e['condition'], e['name']): e['duration_us'] for e in retained_events}
visual_data = {
    'kind': 'line',
    'x': [2.0, 0.5],
    'xlabel': 'requested injected sleep (milliseconds)',
    'ylabel': 'actual host annotation duration (microseconds)',
    'series': [
        {'label': 'input_wait', 'y': [lookup[(condition, 'input_wait')] for condition in conditions]},
        {'label': 'model_and_wait', 'y': [lookup[(condition, 'model_and_wait')] for condition in conditions]},
    ],
}

# Experiment: Account for the enclosing step
# Experiment — Account for the enclosing step: Outer and child intervals describe nested boundaries.
overhead=np.asarray(step_us)-np.asarray(input_us)-np.asarray(model_us)
assert np.all(overhead>=-2)
# Transfer the device result to the host for an independent NumPy comparison.
wrong=np.asarray(step_us)+np.asarray(input_us)+np.asarray(model_us)
assert np.all(wrong>=np.asarray(step_us))
print('Unattributed outer time, microseconds:',overhead.tolist())
print('Wrong double-counted totals:',wrong.tolist())

# Experiment: Predict and check a changed input wait
baseline_wait_us=2520
changed_wait_us=642
baseline_model_us=119
changed_model_us=79
measured_wait_drop_us=baseline_wait_us-changed_wait_us
assert measured_wait_drop_us==1878
assert abs(baseline_model_us-changed_model_us)<100
print({'input_wait_drop_us':measured_wait_drop_us,'model_before_us':baseline_model_us,'model_after_us':changed_model_us})

# Reference solution. Try the exercise before reading this.
# Exercise solution
model_ms=np.asarray(model_us)/1000
summary={'minimum_ms':float(model_ms.min()),'median_ms':float(np.median(model_ms)),'maximum_ms':float(model_ms.max())}
assert summary['minimum_ms']<=summary['median_ms']<=summary['maximum_ms']
np.testing.assert_allclose(model_ms*1000,model_us)
print(summary)

# Reference practice: Account for the actual nested timeline
intervals=[('learner_step',122,2787),('input_wait',125,2645),('model_and_wait',2668,2787)]
children=[(125,2645),(2668,2787)]
child_union=(2645-125)+(2787-2668)
outer=2787-122
remainder=outer-child_union
negative_control=outer+(2645-125)+(2787-2668)
assert (outer,child_union,remainder,negative_control)==(2665,2639,26,5304)
assert max(children[0][1]-children[0][0],children[1][1]-children[1][0])==2520
print({'outer_us':outer,'children_union_us':child_union,'remainder_us':remainder,'wrong_parent_plus_children_us':negative_control})
print("PASS: performance-03")
