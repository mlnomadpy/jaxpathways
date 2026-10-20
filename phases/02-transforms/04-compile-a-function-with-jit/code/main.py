"""Compile a function with jit: worked experiments and reference solutions. CPU checks."""

# Step 1: Set up imports and input tensors
import time
import jax
import jax.numpy as jnp
def loss(w, x, y):
    return jnp.mean((x @ w - y) ** 2)
# Construct and reshape `x` into the target tensor dimensions.
x = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10.

# Step 2: Apply the core JAX transformation
w = jnp.array([1., 2., -1.])
y = jnp.ones(4)
# Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
compiled = jax.jit(loss)
t0 = time.perf_counter()
# Synchronize host execution until asynchronous device computation completes.
first = compiled(w, x, y).block_until_ready()

# Step 3: Verify shapes and numerical invariants
first_seconds = time.perf_counter() - t0
t0 = time.perf_counter()
# Synchronize host execution until asynchronous device computation completes.
second = compiled(w, x, y).block_until_ready()
repeat_seconds = time.perf_counter() - t0
assert jnp.allclose(first, loss(w, x, y))
assert jnp.allclose(second, first)

# Compile a function with jit: Compilation prepares a program that can be reused for compatible...
import time
import jax
import jax.numpy as jnp
def loss(w, x, y):
    return jnp.mean((x @ w - y) ** 2)
# Construct and reshape `x` into the target tensor dimensions.
x = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10.
w = jnp.array([1., 2., -1.])
y = jnp.ones(4)
# Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
compiled = jax.jit(loss)
t0 = time.perf_counter()
# Synchronize host execution until asynchronous device computation completes.
first = compiled(w, x, y).block_until_ready()
first_seconds = time.perf_counter() - t0
t0 = time.perf_counter()
# Synchronize host execution until asynchronous device computation completes.
second = compiled(w, x, y).block_until_ready()
repeat_seconds = time.perf_counter() - t0
print("Loss:", float(second))
print("First/repeat seconds:", first_seconds, repeat_seconds)
assert jnp.allclose(first, loss(w, x, y))
assert jnp.allclose(second, first)

# Figure data experiment
# Compute figure data for: Compilation preserves the result
visual_data = {'kind': 'bar', 'labels': ['eager', 'first compiled', 'repeat compiled'], 'ylabel': 'mean squared loss', 'series': [{'label': 'evaluated loss', 'y': [float(loss(w, x, y)), float(first), float(second)]}]}

# Experiment: Derive predictions and gradients outside the transform
# Experiment — Derive predictions and gradients outside the transform: The analytic value check detects mistakes that...
expected_predictions = jnp.array([0., 0.6, 1.2, 1.8])
assert jnp.allclose(x @ w, expected_predictions, atol=1e-6)
assert jnp.allclose(compiled(w, x, y), 0.46, atol=1e-6)
# Perform matrix / vector contraction (`@`) to compute `manual_gradient`.
manual_gradient = (2. / len(y)) * x.T @ (x @ w - y)
# Differentiate the objective to obtain `compiled_value_gradient` via automatic differentiation.
compiled_value_gradient = jax.jit(jax.value_and_grad(loss))
checked_value, checked_gradient = compiled_value_gradient(w, x, y)
assert jnp.allclose(checked_gradient, manual_gradient, rtol=1e-5, atol=1e-6)

# Experiment: Collect a synchronized latency sample
# Experiment — Collect a synchronized latency sample: The report includes Python invocation and completed numerical...
import statistics
def completed_samples(function, repeats=20):
    # Synchronize host execution until asynchronous device computation completes.
    function(w, x, y).block_until_ready()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        # Synchronize host execution until asynchronous device computation completes.
        function(w, x, y).block_until_ready()
        durations.append(time.perf_counter() - start)
    return durations
for name, function in (("eager", loss), ("compiled", compiled)):
    samples = completed_samples(function)
    print(name, "median / min / max seconds:", statistics.median(samples), min(samples), max(samples))
print("backend / shapes / dtype:", jax.default_backend(), x.shape, w.shape, y.shape, x.dtype)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Foundation · Compile value_and_grad(loss), check the loss and gradient...
# Differentiate the objective to obtain `compiled_step` via automatic differentiation.
compiled_step = jax.jit(jax.value_and_grad(loss))
v, g = compiled_step(w, x, y)
# Differentiate the objective to obtain `(expected_v, expected_g)` via automatic differentiation.
expected_v, expected_g = jax.value_and_grad(loss)(w, x, y)
assert jnp.allclose(v, expected_v)
assert jnp.allclose(g, expected_g)

# Reference practice: Check that a new value remains runtime data
# Check that a new value remains runtime data (Practice): The same computation accepts different runtime values.
for new_w in (w, w + 0.25):
    # Aggregate array values to compute `expected`.
    expected = jnp.mean((x @ new_w - y) ** 2)
    observed = compiled(new_w, x, y)
    assert jnp.allclose(observed, expected, rtol=1e-5, atol=1e-6)

# Reference practice: Build a report whose timing claim can be checked
# Build a report whose timing claim can be checked (Challenge): The measurement excludes input construction and first...
def wait_for_outputs(outputs):
    return jax.tree.map(lambda array: array.block_until_ready(), outputs)
wait_for_outputs(compiled_value_gradient(w, x, y))
gradient_times = []
for _ in range(20):
    started = time.perf_counter()
    wait_for_outputs(compiled_value_gradient(w, x, y))
    gradient_times.append(time.perf_counter() - started)
print("compiled value-and-grad median seconds:", statistics.median(gradient_times))
assert len(gradient_times) == 20
print("PASS: transforms-04")
