"""Read an XProf trace: worked experiments and reference solutions. CPU checks."""

# Prepare a traceable workload
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

# Capture named intervals
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


# Figure data experiment
visual_data={'kind':'line','x':[0,1,2,3],'xlabel':'captured step','ylabel':'host annotation duration (microseconds)','series':[{'label':'injected input wait','y':input_us},{'label':'model call and completion wait','y':model_us}]}

# Experiment: Account for the enclosing step
overhead=np.asarray(step_us)-np.asarray(input_us)-np.asarray(model_us)
assert np.all(overhead>=-2)
wrong=np.asarray(step_us)+np.asarray(input_us)+np.asarray(model_us)
assert np.all(wrong>=np.asarray(step_us))
print('Unattributed outer time, microseconds:',overhead.tolist())
print('Wrong double-counted totals:',wrong.tolist())

# Reference solution. Try the exercise before reading this.
model_ms=np.asarray(model_us)/1000
summary={'minimum_ms':float(model_ms.min()),'median_ms':float(np.median(model_ms)),'maximum_ms':float(model_ms.max())}
assert summary['minimum_ms']<=summary['median_ms']<=summary['maximum_ms']
np.testing.assert_allclose(model_ms*1000,model_us)
print(summary)

# Reference practice: Compute a union instead of summing overlap
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
print("PASS: performance-03")
