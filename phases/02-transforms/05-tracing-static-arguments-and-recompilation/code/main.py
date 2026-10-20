"""Tracing, static arguments, and recompilation: worked experiments and reference solutions. CPU checks."""

# Step 1: Set up imports and input tensors
import jax
import jax.numpy as jnp

# Step 2: Apply the core JAX transformation
def reduce_values(x, mode):
    if mode == "sum":
        return x.sum()
    if mode == "mean":
        return x.mean()
    raise ValueError("mode must be sum or mean")
# Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
compiled = jax.jit(reduce_values, static_argnames=("mode",))

# Step 3: Verify shapes and numerical invariants
x = jnp.array([1., 2., 3.])
assert jnp.allclose(compiled(x, mode="sum"), 6.)
assert jnp.allclose(compiled(x, mode="mean"), 2.)

# Tracing, static arguments, and recompilation: Tracing observes how Python builds an array computation from...
import jax
import jax.numpy as jnp
def reduce_values(x, mode):
    if mode == "sum":
        return x.sum()
    if mode == "mean":
        return x.mean()
    raise ValueError("mode must be sum or mean")
# Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
compiled = jax.jit(reduce_values, static_argnames=("mode",))
x = jnp.array([1., 2., 3.])
print("Sum:", float(compiled(x, mode="sum")))
print("Mean:", float(compiled(x, mode="mean")))
assert jnp.allclose(compiled(x, mode="sum"), 6.)
assert jnp.allclose(compiled(x, mode="mean"), 2.)

# Figure data experiment
# Compare compiled execution in 'sum' mode and 'mean' mode:
out_sum = float(compiled(x, 'sum'))
out_mean = float(compiled(x, 'mean'))
visual_data = {'kind': 'bar', 'x': [0, 1], 'labels': ['sum', 'mean'], 'xlabel': 'static mode argument', 'ylabel': 'compiled(x, mode) output', 'series': [{'label': 'compiled(x, mode)', 'y': [out_sum, out_mean]}]}

# Experiment: Look at the staged numerical work
# Experiment — Look at the staged numerical work: The second call supplies new values with the same shape/dtype.
def sum_squares(values):
    print("Tracing sum_squares; shape:", values.shape)
    return jnp.sum(values * values)
print(jax.make_jaxpr(sum_squares)(x))
# Wrap with `jax.jit` (`staged_squares`) so XLA traces and compiles the function.
staged_squares = jax.jit(sum_squares)
assert jnp.allclose(staged_squares(x), 14.)
assert jnp.allclose(staged_squares(x + 1.), 29.)

# Experiment: Reproduce and repair a dynamic branch
# Experiment — Reproduce and repair a dynamic branch: The test expects a particular category of error and confirms the...
def python_branch(value):
    if value > 0.:
        return value ** 2
    return -value
assert python_branch(-2.) == 2.
# Run the boundary check and catch the expected exception:
try:
    jax.jit(python_branch)(jnp.array(-2.))
except jax.errors.TracerBoolConversionError:
    print("Expected dynamic Boolean conversion error")
else:
    raise AssertionError("Expected a traced Python branch to fail")
def runtime_branch(value):
    return jax.lax.cond(value > 0., lambda z: z ** 2, lambda z: -z, value)
# Wrap with `jax.jit` (`compiled_branch`) so XLA traces and compiles the function.
compiled_branch = jax.jit(runtime_branch)
assert jnp.allclose(compiled_branch(jnp.array(-2.)), 2.)
assert jnp.allclose(compiled_branch(jnp.array(3.)), 9.)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Foundation · Extend reduce_values with a static max mode while...
def choose_reduction(values, mode):
    if mode == "sum":
        return values.sum()
    if mode == "mean":
        return values.mean()
    if mode == "max":
        return values.max()
    raise ValueError("unknown mode")
# Wrap with `jax.jit` (`choose`) so XLA traces and compiles the function.
choose = jax.jit(choose_reduction, static_argnames=("mode",))
for values in (x, jnp.array([-4., -1., -2.])):
    for mode, reference in (("sum", jnp.sum), ("mean", jnp.mean), ("max", jnp.max)):
        assert jnp.allclose(choose(values, mode=mode), reference(values))
# Run the boundary check and catch the expected exception:
try:
    choose(x, mode="typo")
except ValueError:
    pass
else:
    raise AssertionError("Unknown mode should fail")

# Reference practice: Keep runtime values dynamic
# Keep runtime values dynamic (Practice): The vector values are runtime data.
def elementwise_absolute(values):
    return jnp.where(values >= 0., values, -values)
# Wrap with `jax.jit` (`compiled_absolute`) so XLA traces and compiles the function.
compiled_absolute = jax.jit(elementwise_absolute)
for values in (jnp.array([-2., 0., 3.]), jnp.array([5., -4., -1.])):
    assert jnp.allclose(compiled_absolute(values), jnp.abs(values))

# Reference practice: Audit a shape-dependent specialization
# Audit a shape-dependent specialization (Challenge): Changing the length changes shape metadata and can need a...
def plain_sum_squares(values):
    return jnp.sum(values ** 2)
# Wrap with `jax.jit` (`compiled_squares`) so XLA traces and compiles the function.
compiled_squares = jax.jit(plain_sum_squares)
for length in (3, 5):
    for shift in (0., 1.):
        # Create evenly spaced index values in `values`.
        values = jnp.arange(length, dtype=jnp.float32) + shift
        expected = sum(float(v) ** 2 for v in values)
        assert jnp.allclose(compiled_squares(values), expected)
print("PASS: transforms-05")
