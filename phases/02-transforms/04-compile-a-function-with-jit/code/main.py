"""Compile a function with jit: worked experiments and reference solutions. CPU checks."""

# Step 1: Set up imports and input tensors
import time
import jax
import jax.numpy as jnp
# Function `loss(w, x, y)` implementing this stage's computation:
def loss(w, x, y):
    # Return `jnp.mean((x @ w - y) ** 2)` to the caller.
    return jnp.mean((x @ w - y) ** 2)
# Construct and reshape `x` into the target tensor dimensions.
x = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10.

# Step 2: Apply the core JAX transformation
w = jnp.array([1., 2., -1.])
# Construct `y` via `jnp.ones(4)`
y = jnp.ones(4)
# Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
compiled = jax.jit(loss)
# Record execution timing or profiler trace in `t0`.
t0 = time.perf_counter()
# Synchronize host execution until asynchronous device computation completes.
first = compiled(w, x, y).block_until_ready()

# Step 3: Verify shapes and numerical invariants
first_seconds = time.perf_counter() - t0
# Record execution timing or profiler trace in `t0`.
t0 = time.perf_counter()
# Synchronize host execution until asynchronous device computation completes.
second = compiled(w, x, y).block_until_ready()
# Record execution timing or profiler trace in `repeat_seconds`.
repeat_seconds = time.perf_counter() - t0
# Print diagnostic summary of the computed outputs.
# Print diagnostic summary of the computed outputs.
# Assert that `jnp.allclose(first, loss(w, x, y))`.
assert jnp.allclose(first, loss(w, x, y))
# Assert that `jnp.allclose(second, first)`.
assert jnp.allclose(second, first)

# Compile a function with jit: Compilation prepares a program that can be reused for compatible...
# Import time for this computation.
import time
import jax
import jax.numpy as jnp
# Function `loss(w, x, y)` implementing this stage's computation:
def loss(w, x, y):
    # Return `jnp.mean((x @ w - y) ** 2)` to the caller.
    return jnp.mean((x @ w - y) ** 2)
# Construct and reshape `x` into the target tensor dimensions.
x = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10.
# Construct `w` via `jnp.array([1., 2., -1.])`
w = jnp.array([1., 2., -1.])
# Construct `y` via `jnp.ones(4)`
y = jnp.ones(4)
# Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
compiled = jax.jit(loss)
# Record execution timing or profiler trace in `t0`.
t0 = time.perf_counter()
# Synchronize host execution until asynchronous device computation completes.
first = compiled(w, x, y).block_until_ready()
# Record execution timing or profiler trace in `first_seconds`.
first_seconds = time.perf_counter() - t0
# Record execution timing or profiler trace in `t0`.
t0 = time.perf_counter()
# Synchronize host execution until asynchronous device computation completes.
second = compiled(w, x, y).block_until_ready()
# Record execution timing or profiler trace in `repeat_seconds`.
repeat_seconds = time.perf_counter() - t0
# Print diagnostic summary of the computed outputs.
print("Loss:", float(second))
# Print diagnostic summary of the computed outputs.
print("First/repeat seconds:", first_seconds, repeat_seconds)
# Assert that `jnp.allclose(first, loss(w, x, y))`.
assert jnp.allclose(first, loss(w, x, y))
# Assert that `jnp.allclose(second, first)`.
assert jnp.allclose(second, first)

# Figure data experiment
# Compute figure data for: Compilation preserves the result
# Compute `visual_data` from `{'kind': 'bar', 'labels': ['eager', 'first compiled'...`
visual_data = {'kind': 'bar', 'labels': ['eager', 'first compiled', 'repeat compiled'], 'ylabel': 'mean squared loss', 'series': [{'label': 'evaluated loss', 'y': [float(loss(w, x, y)), float(first), float(second)]}]}

# Experiment: Derive predictions and gradients outside the transform
# Experiment — Derive predictions and gradients outside the transform: The analytic value check detects mistakes that...
# Construct `expected_predictions` via `jnp.array([0., 0.6, 1.2, 1.8])`
expected_predictions = jnp.array([0., 0.6, 1.2, 1.8])
# Assert that `jnp.allclose(x @ w, expected_predictions, atol=1e-6)`.
assert jnp.allclose(x @ w, expected_predictions, atol=1e-6)
# Assert that `jnp.allclose(compiled(w, x, y), 0.46, atol=1e-6)`.
assert jnp.allclose(compiled(w, x, y), 0.46, atol=1e-6)
# Perform matrix / vector contraction (`@`) to compute `manual_gradient`.
manual_gradient = (2. / len(y)) * x.T @ (x @ w - y)
# Differentiate the objective to obtain `compiled_value_gradient` via automatic differentiation.
compiled_value_gradient = jax.jit(jax.value_and_grad(loss))
# Run `compiled_value_gradient` to compute `(checked_value, checked_gradient)`.
checked_value, checked_gradient = compiled_value_gradient(w, x, y)
# Assert that `jnp.allclose(checked_gradient, manual_gradient, rtol=1e-5, atol=1e-6)`.
assert jnp.allclose(checked_gradient, manual_gradient, rtol=1e-5, atol=1e-6)

# Experiment: Collect a synchronized latency sample
# Experiment — Collect a synchronized latency sample: The report includes Python invocation and completed numerical...
# Import statistics for this computation.
import statistics
# Function `completed_samples(function, repeats)` implementing this stage's computation:
def completed_samples(function, repeats=20):
    # Synchronize host execution until asynchronous device computation completes.
    function(w, x, y).block_until_ready()
    # Compute `durations` from `[]`
    durations = []
    # Repeat the update loop over `range(repeats)` steps:
    for _ in range(repeats):
        # Record execution timing or profiler trace in `start`.
        start = time.perf_counter()
        # Synchronize host execution until asynchronous device computation completes.
        function(w, x, y).block_until_ready()
        # Record execution timing or profiler trace in ``.
        durations.append(time.perf_counter() - start)
    # Return `durations` to the caller.
    return durations
# Iterate over `(name, function)` to step through the computation:
for name, function in (("eager", loss), ("compiled", compiled)):
    # Run `completed_samples` to compute `samples`.
    samples = completed_samples(function)
    # Print the observed values to compare against the expected result.
    print(name, "median / min / max seconds:", statistics.median(samples), min(samples), max(samples))
# Print the observed values to compare against the expected result.
print("backend / shapes / dtype:", jax.default_backend(), x.shape, w.shape, y.shape, x.dtype)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Foundation · Compile value_and_grad(loss), check the loss and gradient...
# Differentiate the objective to obtain `compiled_step` via automatic differentiation.
compiled_step = jax.jit(jax.value_and_grad(loss))
# Run `compiled_step` to compute `(v, g)`.
v, g = compiled_step(w, x, y)
# Differentiate the objective to obtain `(expected_v, expected_g)` via automatic differentiation.
expected_v, expected_g = jax.value_and_grad(loss)(w, x, y)
# Assert that `jnp.allclose(v, expected_v)`.
assert jnp.allclose(v, expected_v)
# Assert that `jnp.allclose(g, expected_g)`.
assert jnp.allclose(g, expected_g)

# Reference practice: Check that a new value remains runtime data
# Check that a new value remains runtime data (Practice): The same computation accepts different runtime values.
# Iterate over `new_w` to step through the computation:
for new_w in (w, w + 0.25):
    # Aggregate array values to compute `expected`.
    expected = jnp.mean((x @ new_w - y) ** 2)
    # Run `compiled` to compute `observed`.
    observed = compiled(new_w, x, y)
    # Assert that `jnp.allclose(observed, expected, rtol=1e-5, atol=1e-6)`.
    assert jnp.allclose(observed, expected, rtol=1e-5, atol=1e-6)

# Reference practice: Build a report whose timing claim can be checked
# Build a report whose timing claim can be checked (Challenge): The measurement excludes input construction and first...
def wait_for_outputs(outputs):
    # Return `jax.tree.map(lambda array: array.block_until_ready(), outputs)` to the caller.
    return jax.tree.map(lambda array: array.block_until_ready(), outputs)
# Run `wait_for_outputs` to perform the next check or state transition.
wait_for_outputs(compiled_value_gradient(w, x, y))
# Compute `gradient_times` from `[]`
gradient_times = []
# Repeat the update loop over `range(20)` steps:
for _ in range(20):
    # Record execution timing or profiler trace in `started`.
    started = time.perf_counter()
    # Run `wait_for_outputs` to perform the next check or state transition.
    wait_for_outputs(compiled_value_gradient(w, x, y))
    # Record execution timing or profiler trace in ``.
    gradient_times.append(time.perf_counter() - started)
# Print the observed values to compare against the expected result.
print("compiled value-and-grad median seconds:", statistics.median(gradient_times))
# Assert invariant `len(gradient_times) == 20` holds
assert len(gradient_times) == 20
print("PASS: transforms-04")
