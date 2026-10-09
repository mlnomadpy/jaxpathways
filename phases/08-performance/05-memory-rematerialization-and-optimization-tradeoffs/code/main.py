"""Memory, rematerialization, and optimization tradeoffs: worked experiments and reference solutions. CPU checks."""

# Define two differentiation policies
# Step 1 — Define two differentiation policies: Both functions compute the same pure objective; only the...
# Import contextlib for this computation.
import contextlib
import io
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import print_saved_residuals
# Draw pseudorandom samples for `rng` using the explicit RNG state.
rng=np.random.default_rng(18)
# Create device-backed JAX array `w`.
w=jnp.asarray((rng.normal(size=(16,16))*.1).astype(np.float32))
# Compute `x` from `jnp.asarray(np.linspace(-.5,.5,16,dtype=np.float32))`
x=jnp.asarray(np.linspace(-.5,.5,16,dtype=np.float32))
# Function `layer(v)` implementing this stage's computation:
# Return `jnp.tanh(w @ v)` to the caller.
def layer(v):return jnp.tanh(w@v)
# Function `objective(v, rematerialize)` implementing this stage's computation:
def objective(v,rematerialize=False):
    # Run `jax.checkpoint` to compute `operation`.
    operation=jax.checkpoint(layer) if rematerialize else layer
    # Repeat the update loop over `range(4)` steps:
    # Run `operation` to compute `v`.
    for _ in range(4):v=operation(v)
    # Return `jnp.sum(v * v)` to the caller.
    return jnp.sum(v*v)
# Compute `plain` from `lambda v:objective(v,False)`
plain=lambda v:objective(v,False)
# Compute `remat` from `lambda v:objective(v,True)`
remat=lambda v:objective(v,True)
# Function `residual_report(fn)` implementing this stage's computation:
def residual_report(fn):
    # Run `io.StringIO` to compute `stream`.
    stream=io.StringIO()
    # Enter `contextlib.redirect_stdout(stream)` context block:
    with contextlib.redirect_stdout(stream):print_saved_residuals(fn,x)
    # Compute `lines` from `[line for line in stream.getvalue().splitlines() if ...`
    lines=[line for line in stream.getvalue().splitlines() if line.strip()]
    # Return `lines` to the caller.
    return lines
# Compute `reports` from `[residual_report(plain),residual_report(remat)]`
reports=[residual_report(plain),residual_report(remat)]

# Verify with an independent reverse pass
# Independent reverse recurrence through the same explicitly specified network.
host_w=np.asarray(w)
values=[np.asarray(x)]
# Repeat the update loop over `range(4)` steps:
# Perform matrix contraction / projection to compute ``.
for _ in range(4):values.append(np.tanh(host_w@values[-1]))
# Compute `cotangent` from `2*values[-1]`
cotangent=2*values[-1]
# Iterate over `i` to step through the computation:
for i in range(4,0,-1):cotangent=host_w.T@(cotangent*(1-values[i]**2))
# Iterate over `fn` to step through the computation:
for fn in (plain,remat):
    # Differentiate the objective to obtain gradients ``.
    np.testing.assert_allclose(jax.grad(fn)(x),cotangent,rtol=3e-5,atol=1e-7)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.grad(plain)(x),jax.grad(remat)(x),rtol=1e-6,atol=1e-7)
# Differentiate the objective to obtain `executables` via automatic differentiation.
executables=[jax.jit(jax.grad(fn)).lower(x).compile() for fn in (plain,remat)]
# Compute `memory` from `[]`
memory=[]
times=[]
# Iterate over `executable` to step through the computation:
for executable in executables:
    # Run `executable.memory_analysis` to compute `analysis`.
    analysis=executable.memory_analysis()
    # Append the current step result to `memory`.
    memory.append(None if analysis is None else {name:getattr(analysis,name) for name in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')})
    # Synchronize host execution until asynchronous device computation completes.
    # Synchronize host execution until asynchronous device computation completes.
    executable(x).block_until_ready()
    samples=[]
    # Repeat the update loop over `range(5)` steps:
    for _ in range(5):
        # Synchronize host execution until asynchronous device computation completes.
        # Synchronize host execution until asynchronous device computation completes.
        # Synchronize host execution until asynchronous device computation completes.
        start=time.perf_counter()
        executable(x).block_until_ready()
        samples.append(time.perf_counter()-start)
    # Append the current step result to `times`.
    times.append(samples)
# Print the observed values to compare against the expected result.
print('Saved residual descriptions:')
# Iterate over `(name, lines)` to step through the computation:
for name,lines in zip(['plain','remat'],reports):print(name,'\n'+'\n'.join(lines))
# Print the observed values to compare against the expected result.
print('Compiler memory estimates:',memory)
# Print diagnostic summary of the computed outputs.
print('Synchronized gradient samples:',times)
# Print diagnostic summary of the computed outputs.
print('Independent reverse recurrence agrees with both gradients.')

# Step 3: Verify invariants on the completed state
    times.append(samples)
for name,lines in zip(['plain','remat'],reports):print(name,'\n'+'\n'.join(lines))

# Step 1 — Define two differentiation policies: Both functions compute the same pure objective; only the...
# Import contextlib for this computation.
import contextlib
import io
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import print_saved_residuals
# Draw pseudorandom samples for `rng` using the explicit RNG state.
rng=np.random.default_rng(18)
# Create device-backed JAX array `w`.
w=jnp.asarray((rng.normal(size=(16,16))*.1).astype(np.float32))
# Compute `x` from `jnp.asarray(np.linspace(-.5,.5,16,dtype=np.float32))`
x=jnp.asarray(np.linspace(-.5,.5,16,dtype=np.float32))
# Function `layer(v)` implementing this stage's computation:
# Return `jnp.tanh(w @ v)` to the caller.
def layer(v):return jnp.tanh(w@v)
# Function `objective(v, rematerialize)` implementing this stage's computation:
def objective(v,rematerialize=False):
    # Run `jax.checkpoint` to compute `operation`.
    operation=jax.checkpoint(layer) if rematerialize else layer
    # Repeat the update loop over `range(4)` steps:
    # Run `operation` to compute `v`.
    for _ in range(4):v=operation(v)
    # Return `jnp.sum(v * v)` to the caller.
    return jnp.sum(v*v)
# Compute `plain` from `lambda v:objective(v,False)`
plain=lambda v:objective(v,False)
# Compute `remat` from `lambda v:objective(v,True)`
remat=lambda v:objective(v,True)
# Function `residual_report(fn)` implementing this stage's computation:
def residual_report(fn):
    # Run `io.StringIO` to compute `stream`.
    stream=io.StringIO()
    # Enter `contextlib.redirect_stdout(stream)` context block:
    with contextlib.redirect_stdout(stream):print_saved_residuals(fn,x)
    # Compute `lines` from `[line for line in stream.getvalue().splitlines() if ...`
    lines=[line for line in stream.getvalue().splitlines() if line.strip()]
    # Return `lines` to the caller.
    return lines
# Compute `reports` from `[residual_report(plain),residual_report(remat)]`
reports=[residual_report(plain),residual_report(remat)]

# Independent reverse recurrence through the same explicitly specified network.
host_w=np.asarray(w)
values=[np.asarray(x)]
# Repeat the update loop over `range(4)` steps:
# Perform matrix contraction / projection to compute ``.
for _ in range(4):values.append(np.tanh(host_w@values[-1]))
# Compute `cotangent` from `2*values[-1]`
cotangent=2*values[-1]
# Iterate over `i` to step through the computation:
for i in range(4,0,-1):cotangent=host_w.T@(cotangent*(1-values[i]**2))
# Iterate over `fn` to step through the computation:
for fn in (plain,remat):
    # Differentiate the objective to obtain gradients ``.
    np.testing.assert_allclose(jax.grad(fn)(x),cotangent,rtol=3e-5,atol=1e-7)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.grad(plain)(x),jax.grad(remat)(x),rtol=1e-6,atol=1e-7)
# Differentiate the objective to obtain `executables` via automatic differentiation.
executables=[jax.jit(jax.grad(fn)).lower(x).compile() for fn in (plain,remat)]
# Compute `memory` from `[]`
memory=[]
times=[]
# Iterate over `executable` to step through the computation:
for executable in executables:
    # Run `executable.memory_analysis` to compute `analysis`.
    analysis=executable.memory_analysis()
    # Append the current step result to `memory`.
    memory.append(None if analysis is None else {name:getattr(analysis,name) for name in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')})
    # Synchronize host execution until asynchronous device computation completes.
    # Synchronize host execution until asynchronous device computation completes.
    executable(x).block_until_ready()
    samples=[]
    # Repeat the update loop over `range(5)` steps:
    for _ in range(5):
        # Synchronize host execution until asynchronous device computation completes.
        # Synchronize host execution until asynchronous device computation completes.
        # Synchronize host execution until asynchronous device computation completes.
        start=time.perf_counter()
        executable(x).block_until_ready()
        samples.append(time.perf_counter()-start)
    # Append the current step result to `times`.
    times.append(samples)
# Print the observed values to compare against the expected result.
print('Saved residual descriptions:')
# Iterate over `(name, lines)` to step through the computation:
for name,lines in zip(['plain','remat'],reports):print(name,'\n'+'\n'.join(lines))
# Print the observed values to compare against the expected result.
print('Compiler memory estimates:',memory)
# Print diagnostic summary of the computed outputs.
print('Synchronized gradient samples:',times)
# Print diagnostic summary of the computed outputs.
print('Independent reverse recurrence agrees with both gradients.')

# Figure data experiment
# Compute figure data for: Changed autodiff storage does not guarantee changed compiled buffers
# Compute `panels` from `[{'kind':'bar','title':'Autodiff residual descriptio...`
panels=[{'kind':'bar','title':'Autodiff residual descriptions','labels':['ordinary','layer remat'],'ylabel':'saved entries (not bytes)','series':[{'label':'residual descriptions','y':[len(r) for r in reports]}]}]
# Branch on condition `all((m is not None for m in memory))`:
if all(m is not None for m in memory):
    panels.append({'kind':'bar','title':'Compiler memory estimate','labels':['ordinary','layer remat'],'ylabel':'estimated temporary bytes','series':[{'label':'compiler temporaries','y':[m['temp_size_in_bytes'] for m in memory]}]})
else:
    panels.append({'kind':'bar','title':'Compiler memory analysis unavailable','labels':['ordinary','layer remat'],'ylabel':'analysis available (1=yes)','series':[{'label':'availability, not byte count','y':[int(m is not None) for m in memory]}]})
# Compute `visual_data` from `{'kind':'panels','panels':panels}`
visual_data={'kind':'panels','panels':panels}

# Experiment: Move the boundary to the entire objective
# Experiment — Move the boundary to the entire objective: Correctness is necessary but does not choose the best policy.
whole=jax.checkpoint(plain)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.grad(whole)(x),cotangent,rtol=3e-5,atol=1e-7)
# Print the observed values to compare against the expected result.
print('Whole-objective residuals:',residual_report(whole))
# Differentiate the objective to obtain `whole_executable` via automatic differentiation.
whole_executable=jax.jit(jax.grad(whole)).lower(x).compile()
# Print the observed values to compare against the expected result.
print('Whole-objective memory estimate:',whole_executable.memory_analysis())

# Reference solution. Try the exercise before reading this.
# Exercise solution: Use the direction v=(1,\ldots,1) and central differences to...
# Construct `direction` via `jnp.ones_like(x)`
direction=jnp.ones_like(x)
eps=.001
# Compute `finite` from `(float(plain(x+eps*direction))-float(plain(x-eps*dir...`
finite=(float(plain(x+eps*direction))-float(plain(x-eps*direction)))/(2*eps)
# Differentiate the objective to obtain `automatic` via automatic differentiation.
automatic=float(jnp.vdot(jax.grad(plain)(x),direction))
# Compute `np.testing.assert_allclose(finite,automatic,rtol` as `3e-3,atol=1e-6)`.
np.testing.assert_allclose(finite,automatic,rtol=3e-3,atol=1e-6)
# Print the observed values to compare against the expected result.
print('Directional derivative finite/automatic:',finite,automatic,'epsilon',eps)

# Reference practice: Check higher-order differentiation
# Check higher-order differentiation (Transfer): Recomputation should preserve the differentiable function in...
# Construct `direction` via `jnp.linspace(-1.,1.,len(x))`
direction=jnp.linspace(-1.,1.,len(x))
# Differentiate the objective to obtain `(_, hvp_plain)` via automatic differentiation.
_,hvp_plain=jax.jvp(jax.grad(plain),(x,),(direction,))
# Differentiate the objective to obtain `(_, hvp_remat)` via automatic differentiation.
_,hvp_remat=jax.jvp(jax.grad(remat),(x,),(direction,))
# Compute `np.testing.assert_allclose(hvp_plain,hvp_remat,rtol` as `3e-5,atol=1e-7)`.
np.testing.assert_allclose(hvp_plain,hvp_remat,rtol=3e-5,atol=1e-7)
# Confirm that all computed values remain finite (no NaN or Inf).
assert np.isfinite(np.asarray(hvp_plain)).all()
# Print the observed values to compare against the expected result.
print('Higher-order directional derivatives agree.')
print("PASS: performance-05")
