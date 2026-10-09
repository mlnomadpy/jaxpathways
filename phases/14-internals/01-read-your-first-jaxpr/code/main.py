"""Read your first jaxpr: worked experiments and reference solutions. CPU checks."""

# 1. Establish the input signature
# Step 1 — 1. Establish the input signature: The vector has shape (3,) and dtype float32; these determine the...
# Import jax for this computation.
import jax
import jax.numpy as jnp

# Construct `x` via `jnp.array([1., 2., 3.], dtype=jnp.float32)`
x = jnp.array([1., 2., 3.], dtype=jnp.float32)

# 2. Write the pure function
# Step 2 — 2. Write the pure function: Each assignment names a new value.
def shifted_square_sum(x):
    # Compute `shifted` from `x + 1.`
    shifted = x + 1.
    # Compute `squared` from `shifted * shifted`
    squared = shifted * shifted
    # Return `jnp.sum(squared)` to the caller.
    return jnp.sum(squared)

# 3. Inspect and independently verify
# Step 3 — 3. Inspect and independently verify: Forward operations include add, mul and reduce_sum, with output...
# Trace or lower the function to inspect its compiler representation (`closed`).
closed = jax.make_jaxpr(shifted_square_sum)(x)
# Print the observed values to compare against the expected result.
print("Forward program:")
# Print diagnostic summary of the computed outputs.
print(closed)
# Iterate over `equation` to step through the computation:
for equation in closed.jaxpr.eqns:
    # Print diagnostic summary of the computed outputs.
    print("Operation:", equation.primitive.name,
          "output shapes:", [v.aval.shape for v in equation.outvars])
# Run `shifted_square_sum` to compute `value`.
value = shifted_square_sum(x)
# Differentiate the objective to obtain `gradient` via automatic differentiation.
gradient = jax.grad(shifted_square_sum)(x)
# Print the observed values to compare against the expected result.
print("Value / gradient:", value, gradient)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(value, 29.)
# Check numerical equivalence within tolerance: `jnp.allclose(gradient, jnp.array([4., 6., 8.]))`
assert jnp.allclose(gradient, jnp.array([4., 6., 8.]))

# Step 1 — 1. Establish the input signature: The vector has shape (3,) and dtype float32; these determine the...
# Import jax for this computation.
import jax
import jax.numpy as jnp

# Construct `x` via `jnp.array([1., 2., 3.], dtype=jnp.float32)`
x = jnp.array([1., 2., 3.], dtype=jnp.float32)
# Step 2 — 2. Write the pure function: Each assignment names a new value.
def shifted_square_sum(x):
    # Compute `shifted` from `x + 1.`
    shifted = x + 1.
    # Compute `squared` from `shifted * shifted`
    squared = shifted * shifted
    # Return `jnp.sum(squared)` to the caller.
    return jnp.sum(squared)

# Step 3 — 3. Inspect and independently verify: Forward operations include add, mul and reduce_sum, with output...
# Trace or lower the function to inspect its compiler representation (`closed`).
closed = jax.make_jaxpr(shifted_square_sum)(x)
# Print the observed values to compare against the expected result.
print("Forward program:")
# Print diagnostic summary of the computed outputs.
print(closed)
# Iterate over `equation` to step through the computation:
for equation in closed.jaxpr.eqns:
    # Print diagnostic summary of the computed outputs.
    print("Operation:", equation.primitive.name,
          "output shapes:", [v.aval.shape for v in equation.outvars])
# Run `shifted_square_sum` to compute `value`.
value = shifted_square_sum(x)
# Differentiate the objective to obtain `gradient` via automatic differentiation.
gradient = jax.grad(shifted_square_sum)(x)
# Print the observed values to compare against the expected result.
print("Value / gradient:", value, gradient)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(value, 29.)
# Check numerical equivalence within tolerance: `jnp.allclose(gradient, jnp.array([4., 6., 8.]))`
assert jnp.allclose(gradient, jnp.array([4., 6., 8.]))

# Figure data experiment
# Compute figure data for: Follow values through the jaxpr operations
# Combine or mask array elements to form `visual_data`.
visual_data = {'kind': 'heatmap', 'values': jnp.stack([x, x + 1, (x + 1) ** 2]).tolist(), 'rows': ['input', 'add 1', 'multiply by self'], 'columns': ['coordinate 0', 'coordinate 1', 'coordinate 2'], 'unit': 'intermediate value'}

# Experiment: Trace and verify the derivative
# Experiment — Trace and verify the derivative: The new input checks the derivative independently of the...
# Differentiate the objective to obtain `derivative_program` via automatic differentiation.
derivative_program = jax.make_jaxpr(jax.grad(shifted_square_sum))(x)
# Print the observed values to compare against the expected result.
print("Derivative program:", derivative_program)
# Construct `other` via `jnp.array([-1., 0., 2.])`
other = jnp.array([-1., 0., 2.])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(shifted_square_sum(other), 10.)
# Check numerical equivalence within tolerance: `jnp.allclose(jax.grad(shifted_square_sum)(other), jnp.array([0., ...`
assert jnp.allclose(jax.grad(shifted_square_sum)(other), jnp.array([0., 2., 6.]))
# Check numerical equivalence within tolerance: `jnp.allclose(jax.grad(shifted_square_sum)(other), 2*(other+1))`
assert jnp.allclose(jax.grad(shifted_square_sum)(other), 2*(other+1))

# Experiment: Inspect a captured array
# Experiment — Inspect a captured array: The function signature determines ordinary inputs; closed...
# Construct `offset` via `jnp.array([1., 2., 3.])`
offset = jnp.array([1., 2., 3.])
# Function `captured(z)` implementing this stage's computation:
def captured(z):
    # Return `jnp.sum(z + offset)` to the caller.
    return jnp.sum(z + offset)
# Function `explicit(z, offset)` implementing this stage's computation:
def explicit(z, offset):
    # Return `jnp.sum(z + offset)` to the caller.
    return jnp.sum(z + offset)
# Trace or lower the function to inspect its compiler representation (`captured_program`).
captured_program = jax.make_jaxpr(captured)(x)
# Trace or lower the function to inspect its compiler representation (`explicit_program`).
explicit_program = jax.make_jaxpr(explicit)(x, offset)
# Print the observed values to compare against the expected result.
print("Captured:", captured_program)
# Print diagnostic summary of the computed outputs.
print("Explicit:", explicit_program)
# Assert invariant `len(captured_program.jaxpr.invars) == 1` holds
assert len(captured_program.jaxpr.invars) == 1
# Assert invariant `len(explicit_program.jaxpr.invars) == 2` holds
assert len(explicit_program.jaxpr.invars) == 2
# Check numerical equivalence within tolerance: `jnp.allclose(captured(x), explicit(x, offset))`
assert jnp.allclose(captured(x), explicit(x, offset))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Change the scalar offset from 1 to 2.
def shifted_two(z):
    # Compute `shifted` from `z + 2.`
    shifted = z + 2.
    # Return `jnp.sum(shifted * shifted)` to the caller.
    return jnp.sum(shifted * shifted)
# Print the observed values to compare against the expected result.
print(jax.make_jaxpr(shifted_two)(x))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(shifted_two(x), 50.)
# Check numerical equivalence within tolerance: `jnp.allclose(jax.grad(shifted_two)(x), jnp.array([6., 8., 10.]))`
assert jnp.allclose(jax.grad(shifted_two)(x), jnp.array([6., 8., 10.]))

# Reference practice: Compare unrolled and scanned recurrences
# Compare unrolled and scanned recurrences (Transfer / diagnosis): The outer scan equation has a nested body.
def unrolled(z):
    # Repeat the update loop over `range(4)` steps:
    for _ in range(4):
        # Compute `z` from `0.5*z + 1.`
        z = 0.5*z + 1.
    # Return `z` to the caller.
    return z
# Define `scanned(z)` to carry state across steps with `jax.lax.scan`:
def scanned(z):
    # Function `step(carry, _)` implementing this stage's computation:
    def step(carry, _):
        # Compute `new` from `0.5*carry + 1.`
        new = 0.5*carry + 1.
        # Return `(new, new)` to the caller.
        return new, new
    # Return `jax.lax.scan(step, z, None, length=4)[0]` to the caller.
    return jax.lax.scan(step, z, None, length=4)[0]
# Print the observed values to compare against the expected result.
print("Unrolled:", jax.make_jaxpr(unrolled)(jnp.array(0.)))
# Print diagnostic summary of the computed outputs.
print("Scanned:", jax.make_jaxpr(scanned)(jnp.array(0.)))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(unrolled(jnp.array(0.)), 1.875)
# Check numerical equivalence within tolerance: `jnp.allclose(scanned(jnp.array(0.)), 1.875)`
assert jnp.allclose(scanned(jnp.array(0.)), 1.875)

# Reference practice: Reproduce and repair a traced boolean failure
# Reproduce and repair a traced boolean failure (Transfer / diagnosis): The repair makes runtime control flow explicit and verifies...
def bad_branch(z):
    # Branch on condition `z[0] > 0`:
    if z[0] > 0:
        return jnp.sum(z)
    # Return `-jnp.sum(z)` to the caller.
    return -jnp.sum(z)
# Run the boundary check and catch the expected exception:
try:
    jax.make_jaxpr(bad_branch)(x)
except jax.errors.TracerBoolConversionError:
    print("Expected failure: traced boolean used by Python if")
else:
    raise AssertionError("expected tracing failure")
# Function `repaired(z)` implementing this stage's computation:
def repaired(z):
    # Return `jax.lax.cond(z[0] > 0, lambda v: jnp.sum(v), lambda v: -jnp.sum(v), z)` to the caller.
    return jax.lax.cond(z[0] > 0, lambda v: jnp.sum(v), lambda v: -jnp.sum(v), z)
# Print the observed values to compare against the expected result.
print("Repaired:", jax.make_jaxpr(repaired)(x))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.jit(repaired)(x), 6.)
# Check numerical equivalence within tolerance: `jnp.allclose(jax.jit(repaired)(-x), 6.)`
assert jnp.allclose(jax.jit(repaired)(-x), 6.)
print("PASS: internals-01")
