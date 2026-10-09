"""Branches with lax.cond: worked experiments and reference solutions. CPU checks."""

# Prepare the inputs
# Step 1 — Prepare the inputs: These explicit inputs define the case that the later checks will...
# Import jax for this computation.
import jax
import jax.numpy as jnp

# Build the computation
# Step 2 — Build the computation: Both branches return the same scalar type.
# Define and JIT-compile `magnitude(x)` so XLA traces and fuses the operations:
@jax.jit
# Function `magnitude(x)` implementing this stage's computation:
def magnitude(x):
    # Return `jax.lax.cond(x >= 0.0, lambda v: v, lambda v: -v, x)` to the caller.
    return jax.lax.cond(x >= 0., lambda v: v, lambda v: -v, x)

# Run and check the result
# Step 3 — Run and check the result: Compare the output to the expected result below before making the...
# Print the observed values to compare against the expected result.
print(float(magnitude(-3.)))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(magnitude(-3.), 3.)
# Check numerical equivalence within tolerance: `jnp.allclose(magnitude(2.), 2.)`
assert jnp.allclose(magnitude(2.), 2.)

# Step 1 — Prepare the inputs: These explicit inputs define the case that the later checks will...
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Step 2 — Build the computation: Both branches return the same scalar type.
# Define and JIT-compile `magnitude(x)` so XLA traces and fuses the operations:
@jax.jit
# Function `magnitude(x)` implementing this stage's computation:
def magnitude(x):
    # Return `jax.lax.cond(x >= 0.0, lambda v: v, lambda v: -v, x)` to the caller.
    return jax.lax.cond(x >= 0., lambda v: v, lambda v: -v, x)
# Step 3 — Run and check the result: Compare the output to the expected result below before making the...
# Print the observed values to compare against the expected result.
print(float(magnitude(-3.)))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(magnitude(-3.), 3.)
# Check numerical equivalence within tolerance: `jnp.allclose(magnitude(2.), 2.)`
assert jnp.allclose(magnitude(2.), 2.)

# Figure data experiment
# Compute figure data for: Both branches form one magnitude function
# Generate a uniform grid of points in `grid`.
grid = jnp.linspace(-3.0, 3.0, 25)
# Vectorize across the batch dimension without a Python loop (`visual_data`).
visual_data = {'kind': 'line', 'x': grid.tolist(), 'xlabel': 'input', 'ylabel': 'magnitude', 'series': [{'label': 'lax.cond output', 'y': jax.vmap(magnitude)(grid).tolist()}]}

# Experiment: Reproduce the Python truth-test failure
# Experiment — Reproduce the Python truth-test failure: The repair changes where the decision is represented.
def eager_branch(v):
    # Branch on condition `v >= 0.0`:
    if v>=0.:
        return v
    # Return `-v` to the caller.
    return -v
# Assert invariant `eager_branch(-3.)==3.` holds
assert eager_branch(-3.)==3.
# Run the boundary check and catch the expected exception:
try:
    jax.jit(eager_branch)(jnp.array(-3.))
except jax.errors.TracerBoolConversionError:
    print("Expected traced Python Boolean failure")
else:
    raise AssertionError("Expected branch error")
# Iterate over `v` to step through the computation:
for v in [-3.,0.,2.]:
    # Check numerical equivalence within tolerance: `jnp.allclose(magnitude(v),abs(v))`
    assert jnp.allclose(magnitude(v),abs(v))

# Experiment: Check a batched numerical selection
# Experiment — Check a batched numerical selection: Correct values under batching do not imply a conditional branch...
# Construct `values` via `jnp.array([-3.,-1.,2.,4.])`
values=jnp.array([-3.,-1.,2.,4.])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.vmap(magnitude)(values),jnp.abs(values))
# Check numerical equivalence within tolerance: `jnp.allclose(jax.vmap(jax.grad(magnitude))(values),jnp.array([-1....`
assert jnp.allclose(jax.vmap(jax.grad(magnitude))(values),jnp.array([-1.,-1.,1.,1.]))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Write a compiled function that applies 2x when x is positive and x-2...
# Define and JIT-compile `choose_update(x)` so XLA traces and fuses the operations:
@jax.jit
# Function `choose_update(x)` implementing this stage's computation:
def choose_update(x):
    # Return `jax.lax.cond(x > 0.0, lambda v: v * 2.0, lambda v: v - 2.0, x)` to the caller.
    return jax.lax.cond(x > 0., lambda v: v * 2., lambda v: v - 2., x)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(choose_update(3.), 6.)
# Check numerical equivalence within tolerance: `jnp.allclose(choose_update(-3.), -5.)`
assert jnp.allclose(choose_update(-3.), -5.)
# Check numerical equivalence within tolerance: `jnp.allclose(choose_update(0.), -2.)`
assert jnp.allclose(choose_update(0.), -2.)

# Reference practice: Match a changed piecewise rule
# Match a changed piecewise rule (Practice): The values check the exact boundary and both formulas; the...
def piecewise(v):
    # Return `jax.lax.cond(v > 1.0, lambda z: z * z, lambda z: 2 * z, v)` to the caller.
    return jax.lax.cond(v>1.,lambda z:z*z,lambda z:2*z,v)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.vmap(piecewise)(jnp.array([-1.,1.,2.])),jnp.array([-2.,2.,4.]))
# Check numerical equivalence within tolerance: `jnp.allclose(jax.grad(piecewise)(-1.),2.)`
assert jnp.allclose(jax.grad(piecewise)(-1.),2.)
# Check numerical equivalence within tolerance: `jnp.allclose(jax.grad(piecewise)(2.),4.)`
assert jnp.allclose(jax.grad(piecewise)(2.),4.)

# Reference practice: Repair incompatible branch shapes
# Repair incompatible branch shapes (Challenge): A branch mismatch is a result-contract error even when the...
# Run the boundary check and catch the expected exception:
try:
    jax.lax.cond(True,lambda x:x,lambda x:jnp.stack([x,x]),jnp.array(2.))
except TypeError:
    print("Expected branch shape mismatch")
else:
    raise AssertionError("Expected result type error")
# Function `pair(v)` implementing this stage's computation:
def pair(v):
    # Return `jax.lax.cond(v >= 0.0, lambda x: jnp.stack([x, x * x]), lambda x: jnp.stack([-x, x * x]), v)` to the caller.
    return jax.lax.cond(v>=0.,lambda x:jnp.stack([x,x*x]),lambda x:jnp.stack([-x,x*x]),v)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(pair(jnp.array(-2.)),jnp.array([2.,4.]))
print("PASS: state-04")
