"""Memory, rematerialization, and optimization tradeoffs: worked experiments and reference solutions. CPU checks."""

# Define two differentiation policies
import contextlib
import io
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import print_saved_residuals
rng=np.random.default_rng(18)
w=jnp.asarray((rng.normal(size=(16,16))*.1).astype(np.float32))
x=jnp.asarray(np.linspace(-.5,.5,16,dtype=np.float32))
def layer(v):return jnp.tanh(w@v)
def objective(v,rematerialize=False):
    operation=jax.checkpoint(layer) if rematerialize else layer
    for _ in range(4):v=operation(v)
    return jnp.sum(v*v)
plain=lambda v:objective(v,False)
remat=lambda v:objective(v,True)
def residual_report(fn):
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):print_saved_residuals(fn,x)
    lines=[line for line in stream.getvalue().splitlines() if line.strip()]
    return lines
reports=[residual_report(plain),residual_report(remat)]

# Verify with an independent reverse pass
# Independent reverse recurrence through the same explicitly specified network.
host_w=np.asarray(w);values=[np.asarray(x)]
for _ in range(4):values.append(np.tanh(host_w@values[-1]))
cotangent=2*values[-1]
for i in range(4,0,-1):cotangent=host_w.T@(cotangent*(1-values[i]**2))
for fn in (plain,remat):
    np.testing.assert_allclose(jax.grad(fn)(x),cotangent,rtol=3e-5,atol=1e-7)
np.testing.assert_allclose(jax.grad(plain)(x),jax.grad(remat)(x),rtol=1e-6,atol=1e-7)
executables=[jax.jit(jax.grad(fn)).lower(x).compile() for fn in (plain,remat)]
memory=[];times=[]
for executable in executables:
    analysis=executable.memory_analysis()
    memory.append(None if analysis is None else {name:getattr(analysis,name) for name in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')})
    executable(x).block_until_ready();samples=[]
    for _ in range(5):
        start=time.perf_counter();executable(x).block_until_ready();samples.append(time.perf_counter()-start)
    times.append(samples)
print('Saved residual descriptions:')
for name,lines in zip(['plain','remat'],reports):print(name,'\n'+'\n'.join(lines))
print('Compiler memory estimates:',memory)
print('Synchronized gradient samples:',times)
print('Independent reverse recurrence agrees with both gradients.')


import contextlib
import io
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import print_saved_residuals
rng=np.random.default_rng(18)
w=jnp.asarray((rng.normal(size=(16,16))*.1).astype(np.float32))
x=jnp.asarray(np.linspace(-.5,.5,16,dtype=np.float32))
def layer(v):return jnp.tanh(w@v)
def objective(v,rematerialize=False):
    operation=jax.checkpoint(layer) if rematerialize else layer
    for _ in range(4):v=operation(v)
    return jnp.sum(v*v)
plain=lambda v:objective(v,False)
remat=lambda v:objective(v,True)
def residual_report(fn):
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):print_saved_residuals(fn,x)
    lines=[line for line in stream.getvalue().splitlines() if line.strip()]
    return lines
reports=[residual_report(plain),residual_report(remat)]

# Independent reverse recurrence through the same explicitly specified network.
host_w=np.asarray(w);values=[np.asarray(x)]
for _ in range(4):values.append(np.tanh(host_w@values[-1]))
cotangent=2*values[-1]
for i in range(4,0,-1):cotangent=host_w.T@(cotangent*(1-values[i]**2))
for fn in (plain,remat):
    np.testing.assert_allclose(jax.grad(fn)(x),cotangent,rtol=3e-5,atol=1e-7)
np.testing.assert_allclose(jax.grad(plain)(x),jax.grad(remat)(x),rtol=1e-6,atol=1e-7)
executables=[jax.jit(jax.grad(fn)).lower(x).compile() for fn in (plain,remat)]
memory=[];times=[]
for executable in executables:
    analysis=executable.memory_analysis()
    memory.append(None if analysis is None else {name:getattr(analysis,name) for name in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')})
    executable(x).block_until_ready();samples=[]
    for _ in range(5):
        start=time.perf_counter();executable(x).block_until_ready();samples.append(time.perf_counter()-start)
    times.append(samples)
print('Saved residual descriptions:')
for name,lines in zip(['plain','remat'],reports):print(name,'\n'+'\n'.join(lines))
print('Compiler memory estimates:',memory)
print('Synchronized gradient samples:',times)
print('Independent reverse recurrence agrees with both gradients.')


# Figure data experiment
panels=[{'kind':'bar','title':'Autodiff residual descriptions','labels':['ordinary','layer remat'],'ylabel':'saved entries (not bytes)','series':[{'label':'residual descriptions','y':[len(r) for r in reports]}]}]
if all(m is not None for m in memory):
    panels.append({'kind':'bar','title':'Compiler memory estimate','labels':['ordinary','layer remat'],'ylabel':'estimated temporary bytes','series':[{'label':'compiler temporaries','y':[m['temp_size_in_bytes'] for m in memory]}]})
else:
    panels.append({'kind':'bar','title':'Compiler memory analysis unavailable','labels':['ordinary','layer remat'],'ylabel':'analysis available (1=yes)','series':[{'label':'availability, not byte count','y':[int(m is not None) for m in memory]}]})
visual_data={'kind':'panels','panels':panels}

# Experiment: Move the boundary to the entire objective
whole=jax.checkpoint(plain)
np.testing.assert_allclose(jax.grad(whole)(x),cotangent,rtol=3e-5,atol=1e-7)
print('Whole-objective residuals:',residual_report(whole))
whole_executable=jax.jit(jax.grad(whole)).lower(x).compile()
print('Whole-objective memory estimate:',whole_executable.memory_analysis())

# Reference solution. Try the exercise before reading this.
direction=jnp.ones_like(x);eps=.001
finite=(float(plain(x+eps*direction))-float(plain(x-eps*direction)))/(2*eps)
automatic=float(jnp.vdot(jax.grad(plain)(x),direction))
np.testing.assert_allclose(finite,automatic,rtol=3e-3,atol=1e-6)
print('Directional derivative finite/automatic:',finite,automatic,'epsilon',eps)

# Reference practice: Check higher-order differentiation
direction=jnp.linspace(-1.,1.,len(x))
_,hvp_plain=jax.jvp(jax.grad(plain),(x,),(direction,))
_,hvp_remat=jax.jvp(jax.grad(remat),(x,),(direction,))
np.testing.assert_allclose(hvp_plain,hvp_remat,rtol=3e-5,atol=1e-7)
assert np.isfinite(np.asarray(hvp_plain)).all()
print('Higher-order directional derivatives agree.')
print("PASS: performance-05")
