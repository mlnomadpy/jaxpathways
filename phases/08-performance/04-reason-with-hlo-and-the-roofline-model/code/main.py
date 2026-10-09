"""Reason with HLO and the roofline model: worked experiments and reference solutions. CPU checks."""

# Inspect a specialized lowered computation
# Step 1 — Inspect a specialized lowered computation: StableHLO establishes the operation and shape contract.
# Import time for this computation.
import time
import numpy as np
import jax
import jax.numpy as jnp

# Calculate an explicitly hypothetical model
# Step 2 — Calculate an explicitly hypothetical model: The square-matrix simplification supplies an independent...
def workload(a,b):return jnp.tanh(a@b)
# Construct and reshape `a` into the target tensor dimensions.
a=jnp.arange(32*16,dtype=jnp.float32).reshape(32,16)/512
# Construct and reshape `b` into the target tensor dimensions.
b=jnp.arange(16*8,dtype=jnp.float32).reshape(16,8)/128
# Wrap with `jax.jit` (`lowered`) so XLA traces and compiles the function.
lowered=jax.jit(workload).lower(a,b)
# Trace or lower the function to inspect its compiler representation (`stablehlo`).
stablehlo=str(lowered.compiler_ir(dialect='stablehlo'))
# Verify contract: `'dot_general' in stablehlo and 'tanh' in stablehlo`.
assert 'dot_general' in stablehlo and 'tanh' in stablehlo
# Trace or lower the function to inspect its compiler representation (`executable`).
executable=lowered.compile()
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(executable(a,b),np.tanh(np.asarray(a)@np.asarray(b)),rtol=2e-5,atol=2e-5)
# Print the observed values to compare against the expected result.
print('StableHLO operations present: dot_general and tanh')
# Print diagnostic summary of the computed outputs.
print('Compiler estimates (backend-specific):',executable.cost_analysis())

# Keep runtime observations separate
# An analytic traffic model, not a measurement of this CPU.
def contract(m,k,n,itemsize=4):
    # Guard input contract (`min(m, k, n, itemsize) <= 0`) and fail fast if violated.
    if min(m,k,n,itemsize)<=0:raise ValueError('positive dimensions and itemsize required')
    flops=2*m*k*n  # conventional multiply-add count; excludes tanh
    # Evaluate `minimum_bytes` from the current inputs and state.
    minimum_bytes=itemsize*(m*k+k*n+m*n)
    # Return `(flops, minimum_bytes, flops / minimum_bytes)` to the caller.
    return flops,minimum_bytes,flops/minimum_bytes
# Initialize array `sizes` with explicit values and shape.
sizes=np.array([8,16,32,64,128])
# Initialize array `intensity` with explicit values and shape.
intensity=np.array([contract(int(n),int(n),int(n))[2] for n in sizes])
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(intensity,sizes/6)
peak=100e9;bandwidth=20e9  # hypothetical budgets, not device specifications
# Evaluate `memory_ceiling` from the current inputs and state.
memory_ceiling=bandwidth*intensity
# Reduce across the target axis to summarize `roofline`.
roofline=np.minimum(peak,memory_ceiling)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(roofline[:2]/1e9,[80/3,160/3])
# Verify contract: `np.all(roofline[2:] == peak)`.
assert np.all(roofline[2:]==peak)

# Keep runtime observations separate
# Step 4 — Keep runtime observations separate: Warm samples measure the current CPU, while the work count remains...
executable(a,b).block_until_ready()
# Evaluate `times` from the current inputs and state.
times=[]
# Repeat the update loop over `range(5)` steps:
for _ in range(5):
    # Synchronize host execution until asynchronous device computation completes.
    # Synchronize host execution until asynchronous device computation completes.
    # Synchronize host execution until asynchronous device computation completes.
    start=time.perf_counter();executable(a,b).block_until_ready();times.append(time.perf_counter()-start)
# Run `contract` to compute `(flops, minimum_bytes, ai)`.
flops,minimum_bytes,ai=contract(32,16,8)
# Print the observed values to compare against the expected result.
print('Local CPU samples seconds:',times)
# Print diagnostic summary of the computed outputs.
print('Matmul-only accounting:',{'flops':flops,'minimum_bytes':minimum_bytes,'flops_per_byte':ai})
# Print diagnostic summary of the computed outputs.
print('Hypothetical ceilings GFLOP/s:',(roofline/1e9).tolist())

# Step 1 — Inspect a specialized lowered computation: StableHLO establishes the operation and shape contract.
# Import time for this computation.
import time
import numpy as np
import jax
import jax.numpy as jnp

# Step 2 — Calculate an explicitly hypothetical model: The square-matrix simplification supplies an independent...
def workload(a,b):return jnp.tanh(a@b)
# Construct and reshape `a` into the target tensor dimensions.
a=jnp.arange(32*16,dtype=jnp.float32).reshape(32,16)/512
# Construct and reshape `b` into the target tensor dimensions.
b=jnp.arange(16*8,dtype=jnp.float32).reshape(16,8)/128
# Wrap with `jax.jit` (`lowered`) so XLA traces and compiles the function.
lowered=jax.jit(workload).lower(a,b)
# Trace or lower the function to inspect its compiler representation (`stablehlo`).
stablehlo=str(lowered.compiler_ir(dialect='stablehlo'))
# Verify contract: `'dot_general' in stablehlo and 'tanh' in stablehlo`.
assert 'dot_general' in stablehlo and 'tanh' in stablehlo
# Trace or lower the function to inspect its compiler representation (`executable`).
executable=lowered.compile()
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(executable(a,b),np.tanh(np.asarray(a)@np.asarray(b)),rtol=2e-5,atol=2e-5)
# Print the observed values to compare against the expected result.
print('StableHLO operations present: dot_general and tanh')
# Print diagnostic summary of the computed outputs.
print('Compiler estimates (backend-specific):',executable.cost_analysis())

# An analytic traffic model, not a measurement of this CPU.
def contract(m,k,n,itemsize=4):
    # Guard input contract (`min(m, k, n, itemsize) <= 0`) and fail fast if violated.
    if min(m,k,n,itemsize)<=0:raise ValueError('positive dimensions and itemsize required')
    flops=2*m*k*n  # conventional multiply-add count; excludes tanh
    # Evaluate `minimum_bytes` from the current inputs and state.
    minimum_bytes=itemsize*(m*k+k*n+m*n)
    # Return `(flops, minimum_bytes, flops / minimum_bytes)` to the caller.
    return flops,minimum_bytes,flops/minimum_bytes
# Initialize array `sizes` with explicit values and shape.
sizes=np.array([8,16,32,64,128])
# Initialize array `intensity` with explicit values and shape.
intensity=np.array([contract(int(n),int(n),int(n))[2] for n in sizes])
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(intensity,sizes/6)
peak=100e9;bandwidth=20e9  # hypothetical budgets, not device specifications
# Evaluate `memory_ceiling` from the current inputs and state.
memory_ceiling=bandwidth*intensity
# Reduce across the target axis to summarize `roofline`.
roofline=np.minimum(peak,memory_ceiling)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(roofline[:2]/1e9,[80/3,160/3])
# Verify contract: `np.all(roofline[2:] == peak)`.
assert np.all(roofline[2:]==peak)

# Step 4 — Keep runtime observations separate: Warm samples measure the current CPU, while the work count remains...
executable(a,b).block_until_ready()
# Evaluate `times` from the current inputs and state.
times=[]
# Repeat the update loop over `range(5)` steps:
for _ in range(5):
    # Synchronize host execution until asynchronous device computation completes.
    # Synchronize host execution until asynchronous device computation completes.
    # Synchronize host execution until asynchronous device computation completes.
    start=time.perf_counter();executable(a,b).block_until_ready();times.append(time.perf_counter()-start)
# Run `contract` to compute `(flops, minimum_bytes, ai)`.
flops,minimum_bytes,ai=contract(32,16,8)
# Print the observed values to compare against the expected result.
print('Local CPU samples seconds:',times)
# Print diagnostic summary of the computed outputs.
print('Matmul-only accounting:',{'flops':flops,'minimum_bytes':minimum_bytes,'flops_per_byte':ai})
# Print diagnostic summary of the computed outputs.
print('Hypothetical ceilings GFLOP/s:',(roofline/1e9).tolist())

# Figure data experiment
# Compute figure data for: A hypothetical roofline bends at the compute budget
# Run `np.sort` to compute `plot_intensity`.
plot_intensity=np.sort(np.append(intensity,peak/bandwidth))
# Reduce across the target axis to summarize `visual_data`.
visual_data={'kind':'line','x':plot_intensity.tolist(),'xlabel':'minimum-traffic operations per byte','ylabel':'hypothetical ceiling (GFLOP/s)','series':[{'label':'bandwidth budget','y':(bandwidth*plot_intensity/1e9).tolist()},{'label':'compute budget','y':[peak/1e9]*len(plot_intensity)},{'label':'combined ceiling','y':(np.minimum(peak,bandwidth*plot_intensity)/1e9).tolist()}]}

# Experiment: Double the assumed data movement
# Experiment — Double the assumed data movement: The work is unchanged; doubling transferred bytes halves intensity.
reduced=intensity/2
# Reduce across the target axis to summarize `revised`.
revised=np.minimum(peak,bandwidth*reduced)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(revised[:2],roofline[:2]/2)
# Verify contract: `revised[-1] == peak`.
assert revised[-1]==peak
# Print the observed values to compare against the expected result.
print('Ceilings with twice minimum traffic GFLOP/s:',(revised/1e9).tolist())

# Reference solution. Try the exercise before reading this.
# Exercise solution: Derive the work and minimum traffic for shapes (48,24) and (24,12).
work,traffic,ratio=contract(48,24,12)
# Verify contract: `work == 27648 and traffic == 8064`.
assert work==27648 and traffic==8064
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(ratio,24/7)
# Verify contract: `bandwidth * ratio < peak`.
assert bandwidth*ratio<peak
# Print the observed values to compare against the expected result.
print('Changed contract:',work,traffic,ratio,'bandwidth-limited in the hypothetical model')

# Reference practice: Keep the same shapes but change storage precision
# Keep the same shapes but change storage precision (Transfer): Changing storage bytes in an accounting model does not...
f32,b32,i32=contract(32,16,8,4)
# Run `contract` to compute `(f16, b16, i16)`.
f16,b16,i16=contract(32,16,8,2)
# Verify contract: `f32 == f16 and b32 == 2 * b16`.
assert f32==f16 and b32==2*b16
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(i16,2*i32)
# Print the observed values to compare against the expected result.
print('Arithmetic unchanged; modeled traffic halved and intensity doubled.')
print("PASS: performance-04")
