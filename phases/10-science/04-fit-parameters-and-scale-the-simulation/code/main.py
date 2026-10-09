"""Fit parameters and scale the simulation: worked experiments and reference solutions. CPU checks."""

# Define the simulator and immutable observations
# Step 1 — Define the simulator and immutable observations: Observations come from the analytic physical model, not the same...
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update("jax_enable_x64", True)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np

# Function `rk4_step(rate, value, dt)` implementing this stage's computation:
def rk4_step(rate, value, dt):
    # Evaluate `a` from the current inputs and state.
    a = -rate*value
    # Evaluate `b` from the current inputs and state.
    b = -rate*(value + dt*a/2)
    # Evaluate `c` from the current inputs and state.
    c = -rate*(value + dt*b/2)
    # Evaluate `d` from the current inputs and state.
    d = -rate*(value + dt*c)
    # Return `value + dt * (a + 2 * b + 2 * c + d) / 6` to the caller.
    return value + dt*(a + 2*b + 2*c + d)/6

# Define `solve(rate, initial, steps, dt)` to carry state across steps with `jax.lax.scan`:
def solve(rate, initial, steps=40, dt=0.05):
    # Function `advance(value, unused)` implementing this stage's computation:
    def advance(value, unused):
        # Run `rk4_step` to compute `next_value`.
        next_value = rk4_step(rate, value, dt)
        # Return `(next_value, next_value)` to the caller.
        return next_value, next_value
    # Run compiled structured control flow via `jax.lax` (`(_, tail)`).
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    # Return `jnp.concatenate((jnp.atleast_1d(initial), tail))` to the caller.
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

# Initialize array `train_initials` with explicit values and shape.
train_initials = jnp.array([0.7,1.5,2.3])
# Initialize array `times` with explicit values and shape.
times = jnp.arange(41)*0.05
# Evaluate `true_rate` from the current inputs and state.
true_rate = 0.7
# Evaluate `observations` from the current inputs and state.
observations = train_initials[:,None]*jnp.exp(-true_rate*times)
# Function `predict(rate, initials)` implementing this stage's computation:
def predict(rate, initials):
    # Return `jax.vmap(lambda initial: solve(rate, initial))(initials)` to the caller.
    return jax.vmap(lambda initial: solve(rate,initial))(initials)
# Function `objective(log_rate)` implementing this stage's computation:
def objective(log_rate):
    # Run `predict` to compute `residual`.
    residual = predict(jnp.exp(log_rate),train_initials)-observations
    # Return `jnp.mean(residual ** 2)` to the caller.
    return jnp.mean(residual**2)

# Fit a positive rate with an explicit update
# Step 2 — Fit a positive rate with an explicit update: The logarithm parameterization keeps the rate positive.
# Define and JIT-compile `update(log_rate)` so XLA traces and fuses the operations:
@jax.jit
# Function `update(log_rate)` implementing this stage's computation:
def update(log_rate):
    # Return `log_rate - 0.4 * jax.grad(objective)(log_rate)` to the caller.
    return log_rate - 0.4*jax.grad(objective)(log_rate)
# Run `jnp.log` to compute `log_rate`.
log_rate = jnp.log(1.4)
# Evaluate `loss_history` from the current inputs and state.
loss_history = [float(objective(log_rate))]
# Evaluate `rate_history` from the current inputs and state.
rate_history = [float(jnp.exp(log_rate))]
# Repeat the update loop over `range(160)` steps:
for _ in range(160):
    # Run `update` to compute `log_rate`.
    log_rate = update(log_rate)
    # Append the current step result to `loss_history`.
    loss_history.append(float(objective(log_rate)))
    # Append the current step result to `rate_history`.
    rate_history.append(float(jnp.exp(log_rate)))
# Evaluate `jnp.exp(log_rate)` and convert the result into Python scalar/collection `fitted_rate`.
fitted_rate = float(jnp.exp(log_rate))
# Verify contract: `abs(fitted_rate - true_rate) < 2e-07`.
assert abs(fitted_rate-true_rate) < 2e-7
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert loss_history[-1] < 1e-20
# Print the observed values to compare against the expected result.
print("Initial/fitted rate:",rate_history[0],fitted_rate)
# Print diagnostic summary of the computed outputs.
print("Initial/final loss:",loss_history[0],loss_history[-1])

# Validate on unseen initial conditions and a finer grid
# Step 3 — Validate on unseen initial conditions and a finer grid: Held-out starts test transfer within this known equation.
# Initialize array `held_initials` with explicit values and shape.
held_initials=jnp.array([0.4,1.1,3.2])
# Convert `held_truth` to a host NumPy array for inspection or verification.
held_truth=np.asarray(held_initials)[:,None]*np.exp(-true_rate*np.asarray(times))
# Convert `held_prediction` to a host NumPy array for inspection or verification.
held_prediction=np.asarray(predict(fitted_rate,held_initials))
# Aggregate array values to compute `held_rmse`.
held_rmse=float(np.sqrt(np.mean((held_prediction-held_truth)**2)))
# Verify contract: `held_rmse < 1e-07`.
assert held_rmse < 1e-7
# Vectorize across the batch dimension with `jax.vmap` (`fine`).
fine=np.asarray(jax.vmap(lambda initial:solve(fitted_rate,initial,80,0.025))(held_initials))[:,::2]
# Aggregate array values to compute `fine_rmse`.
fine_rmse=float(np.sqrt(np.mean((fine-held_truth)**2)))
# Verify contract: `fine_rmse < 1e-07`.
assert fine_rmse < 1e-7
# Differentiate the objective to obtain `initial_only_gradient` via automatic differentiation.
initial_only_gradient=jax.grad(lambda p:jnp.mean((predict(jnp.exp(p),train_initials)[:,0]-observations[:,0])**2))(jnp.log(1.4))
# Verify contract: `float(initial_only_gradient) == 0.0`.
assert float(initial_only_gradient)==0.0
# Print diagnostic summary of the computed outputs.
print("Held-out RMSE:",held_rmse,"finer-grid RMSE:",fine_rmse)
# Print diagnostic summary of the computed outputs.
print("Initial-time-only gradient:",float(initial_only_gradient))

# Step 1 — Define the simulator and immutable observations: Observations come from the analytic physical model, not the same...
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update("jax_enable_x64", True)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np

# Function `rk4_step(rate, value, dt)` implementing this stage's computation:
def rk4_step(rate, value, dt):
    # Evaluate `a` from the current inputs and state.
    a = -rate*value
    # Evaluate `b` from the current inputs and state.
    b = -rate*(value + dt*a/2)
    # Evaluate `c` from the current inputs and state.
    c = -rate*(value + dt*b/2)
    # Evaluate `d` from the current inputs and state.
    d = -rate*(value + dt*c)
    # Return `value + dt * (a + 2 * b + 2 * c + d) / 6` to the caller.
    return value + dt*(a + 2*b + 2*c + d)/6

# Define `solve(rate, initial, steps, dt)` to carry state across steps with `jax.lax.scan`:
def solve(rate, initial, steps=40, dt=0.05):
    # Function `advance(value, unused)` implementing this stage's computation:
    def advance(value, unused):
        # Run `rk4_step` to compute `next_value`.
        next_value = rk4_step(rate, value, dt)
        # Return `(next_value, next_value)` to the caller.
        return next_value, next_value
    # Run compiled structured control flow via `jax.lax` (`(_, tail)`).
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    # Return `jnp.concatenate((jnp.atleast_1d(initial), tail))` to the caller.
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

# Initialize array `train_initials` with explicit values and shape.
train_initials = jnp.array([0.7,1.5,2.3])
# Initialize array `times` with explicit values and shape.
times = jnp.arange(41)*0.05
# Evaluate `true_rate` from the current inputs and state.
true_rate = 0.7
# Evaluate `observations` from the current inputs and state.
observations = train_initials[:,None]*jnp.exp(-true_rate*times)
# Function `predict(rate, initials)` implementing this stage's computation:
def predict(rate, initials):
    # Return `jax.vmap(lambda initial: solve(rate, initial))(initials)` to the caller.
    return jax.vmap(lambda initial: solve(rate,initial))(initials)
# Function `objective(log_rate)` implementing this stage's computation:
def objective(log_rate):
    # Run `predict` to compute `residual`.
    residual = predict(jnp.exp(log_rate),train_initials)-observations
    # Return `jnp.mean(residual ** 2)` to the caller.
    return jnp.mean(residual**2)

# Step 2 — Fit a positive rate with an explicit update: The logarithm parameterization keeps the rate positive.
# Define and JIT-compile `update(log_rate)` so XLA traces and fuses the operations:
@jax.jit
# Function `update(log_rate)` implementing this stage's computation:
def update(log_rate):
    # Return `log_rate - 0.4 * jax.grad(objective)(log_rate)` to the caller.
    return log_rate - 0.4*jax.grad(objective)(log_rate)
# Run `jnp.log` to compute `log_rate`.
log_rate = jnp.log(1.4)
# Evaluate `loss_history` from the current inputs and state.
loss_history = [float(objective(log_rate))]
# Evaluate `rate_history` from the current inputs and state.
rate_history = [float(jnp.exp(log_rate))]
# Repeat the update loop over `range(160)` steps:
for _ in range(160):
    # Run `update` to compute `log_rate`.
    log_rate = update(log_rate)
    # Append the current step result to `loss_history`.
    loss_history.append(float(objective(log_rate)))
    # Append the current step result to `rate_history`.
    rate_history.append(float(jnp.exp(log_rate)))
# Evaluate `jnp.exp(log_rate)` and convert the result into Python scalar/collection `fitted_rate`.
fitted_rate = float(jnp.exp(log_rate))
# Verify contract: `abs(fitted_rate - true_rate) < 2e-07`.
assert abs(fitted_rate-true_rate) < 2e-7
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert loss_history[-1] < 1e-20
# Print the observed values to compare against the expected result.
print("Initial/fitted rate:",rate_history[0],fitted_rate)
# Print diagnostic summary of the computed outputs.
print("Initial/final loss:",loss_history[0],loss_history[-1])

# Step 3 — Validate on unseen initial conditions and a finer grid: Held-out starts test transfer within this known equation.
# Initialize array `held_initials` with explicit values and shape.
held_initials=jnp.array([0.4,1.1,3.2])
# Convert `held_truth` to a host NumPy array for inspection or verification.
held_truth=np.asarray(held_initials)[:,None]*np.exp(-true_rate*np.asarray(times))
# Convert `held_prediction` to a host NumPy array for inspection or verification.
held_prediction=np.asarray(predict(fitted_rate,held_initials))
# Aggregate array values to compute `held_rmse`.
held_rmse=float(np.sqrt(np.mean((held_prediction-held_truth)**2)))
# Verify contract: `held_rmse < 1e-07`.
assert held_rmse < 1e-7
# Vectorize across the batch dimension with `jax.vmap` (`fine`).
fine=np.asarray(jax.vmap(lambda initial:solve(fitted_rate,initial,80,0.025))(held_initials))[:,::2]
# Aggregate array values to compute `fine_rmse`.
fine_rmse=float(np.sqrt(np.mean((fine-held_truth)**2)))
# Verify contract: `fine_rmse < 1e-07`.
assert fine_rmse < 1e-7
# Differentiate the objective to obtain `initial_only_gradient` via automatic differentiation.
initial_only_gradient=jax.grad(lambda p:jnp.mean((predict(jnp.exp(p),train_initials)[:,0]-observations[:,0])**2))(jnp.log(1.4))
# Verify contract: `float(initial_only_gradient) == 0.0`.
assert float(initial_only_gradient)==0.0
# Print diagnostic summary of the computed outputs.
print("Held-out RMSE:",held_rmse,"finer-grid RMSE:",fine_rmse)
# Print diagnostic summary of the computed outputs.
print("Initial-time-only gradient:",float(initial_only_gradient))

# Figure data experiment
# Compute figure data for: Recover a physical rate, then check what the fit means
# Evaluate `range(61)` and convert the result into Python scalar/collection `indices`.
indices=list(range(61))
# Evaluate `visual_data` from the current inputs and state.
visual_data={'panels':[{'kind':'line','x':indices,'xlabel':'completed updates','ylabel':'rate (inverse seconds)','series':[{'label':'fitted rate','y':rate_history[:61]},{'label':'known synthetic rate','y':[true_rate]*61}]},{'kind':'line','x':indices,'xlabel':'completed updates','ylabel':'mean squared trajectory error','yscale':'log','series':[{'label':'training error','y':loss_history[:61]}]}]}

# Experiment: A perfect Euler fit can infer the wrong rate
# Experiment — A perfect Euler fit can infer the wrong rate: The model parameter absorbs discretization error.
h=0.5
# Evaluate `biased_rate` from the current inputs and state.
biased_rate=(1-np.exp(-true_rate*h))/h
# Initialize array `coarse_times` with explicit values and shape.
coarse_times=np.arange(5)*h
# Initialize array `matched` with explicit values and shape.
matched=2*(1-biased_rate*h)**np.arange(5)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(matched,2*np.exp(-true_rate*coarse_times),rtol=1e-12)
# Verify contract: `abs(biased_rate - true_rate) > 0.1`.
assert abs(biased_rate-true_rate)>0.1
# Print the observed values to compare against the expected result.
print("True rate:",true_rate,"perfect coarse Euler rate:",biased_rate)

# Experiment: Measure fixed-shape CPU batches
# Experiment — Measure fixed-shape CPU batches: The warmed, synchronized boundary excludes compilation and...
# Import time for this computation.
import time
# Evaluate `measurements` from the current inputs and state.
measurements=[]
# Iterate over `batch_size` to step through the computation:
for batch_size in (1,16,256):
    # Initialize array `starts` with explicit values and shape.
    starts=jnp.linspace(0.4,3.2,batch_size)
    # Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
    compiled=jax.jit(lambda values:predict(fitted_rate,values))
    # Synchronize host execution until asynchronous device computation completes.
    compiled(starts).block_until_ready()
    # Evaluate `samples` from the current inputs and state.
    samples=[]
    # Repeat the update loop over `range(7)` steps:
    for _ in range(7):
        # Record execution timing or profiler trace in `beginning`.
        beginning=time.perf_counter()
        # Synchronize host execution until asynchronous device computation completes.
        compiled(starts).block_until_ready()
        # Record execution timing or profiler trace in ``.
        samples.append((time.perf_counter()-beginning)*1000)
    # Evaluate `np.median(samples)` and convert the result into Python scalar/collection `median_ms`.
    median_ms=float(np.median(samples))
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
    assert median_ms>0
    # Append the current step result to `measurements`.
    measurements.append((batch_size,median_ms,median_ms/batch_size))
# Print the observed values to compare against the expected result.
print("batch, median milliseconds, milliseconds per trajectory:",measurements)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Repeat inference for a new true rate 0.4 using analytically generated...
changed_truth=0.4
# Evaluate `changed_observations` from the current inputs and state.
changed_observations=train_initials[:,None]*jnp.exp(-changed_truth*times)
# Function `changed_objective(theta)` implementing this stage's computation:
def changed_objective(theta):
    # Return `jnp.mean((predict(jnp.exp(theta), train_initials) - changed_observations) ** 2)` to the caller.
    return jnp.mean((predict(jnp.exp(theta),train_initials)-changed_observations)**2)
# Differentiate the objective to obtain `changed_update` via automatic differentiation.
changed_update=jax.jit(lambda theta:theta-0.4*jax.grad(changed_objective)(theta))
# Run `jnp.log` to compute `changed_theta`.
changed_theta=jnp.log(0.9)
# Repeat the update loop over `range(200)` steps:
for _ in range(200):
    # Run `changed_update` to compute `changed_theta`.
    changed_theta=changed_update(changed_theta)
# Evaluate `jnp.exp(changed_theta)` and convert the result into Python scalar/collection `changed_rate`.
changed_rate=float(jnp.exp(changed_theta))
# Verify contract: `abs(changed_rate - changed_truth) < 1e-06`.
assert abs(changed_rate-changed_truth)<1e-6
# Convert `changed_held` to a host NumPy array for inspection or verification.
changed_held=np.asarray(held_initials)[:,None]*np.exp(-changed_truth*np.asarray(times))
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(predict(changed_rate,held_initials),changed_held,rtol=2e-6,atol=1e-7)
# Print the observed values to compare against the expected result.
print("Changed true/recovered rate:",changed_truth,changed_rate)

# Reference practice: Infer a different physical quantity
# Infer a different physical quantity (Practice): The inverse problem becomes linear when rate is fixed.
basis=np.exp(-0.7*np.asarray(times))
# Evaluate `unknown_initial` from the current inputs and state.
unknown_initial=2.8
# Evaluate `measured` from the current inputs and state.
measured=unknown_initial*basis
# Evaluate `np.dot(basis, measured) / np.dot(basis, basis)` and convert the result into Python scalar/collection `estimate`.
estimate=float(np.dot(basis,measured)/np.dot(basis,basis))
# Verify contract: `abs(estimate - unknown_initial) < 1e-12`.
assert abs(estimate-unknown_initial)<1e-12
# Function `amplitude_loss(amplitude)` implementing this stage's computation:
def amplitude_loss(amplitude):
    # Return `jnp.mean((amplitude * jnp.asarray(basis) - jnp.asarray(measured)) ** 2)` to the caller.
    return jnp.mean((amplitude*jnp.asarray(basis)-jnp.asarray(measured))**2)
# Verify contract: `abs(float(jax.grad(amplitude_loss)(estimate))) < 1e-12`.
assert abs(float(jax.grad(amplitude_loss)(estimate)))<1e-12
# Print the observed values to compare against the expected result.
print("Recovered initial amplitude:",estimate)

# Reference practice: Compare noisy inference with an independent search
# Compare noisy inference with an independent search (Challenge): A noisy optimum need not equal the generating parameter exactly.
noisy=np.asarray(observations)+0.005*np.random.default_rng(19).normal(size=observations.shape)
# Initialize array `candidates` with explicit values and shape.
candidates=np.linspace(0.65,0.75,1001)
# Convert `reference_predictions` to a host NumPy array for inspection or verification.
reference_predictions=np.asarray(train_initials)[None,:,None]*np.exp(-candidates[:,None,None]*np.asarray(times)[None,None,:])
# Reduce along axis=(1 to compute `reference_losses`.
reference_losses=np.mean((reference_predictions-noisy[None,:,:])**2,axis=(1,2))
# Evaluate `candidates[np.argmin(reference_losses)]` and convert the result into Python scalar/collection `best`.
best=float(candidates[np.argmin(reference_losses)])
# Verify contract: `abs(best - 0.7) < 0.01`.
assert abs(best-0.7)<0.01
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert float(np.min(reference_losses))>0
# Print the observed values to compare against the expected result.
print("Noisy grid-search estimate:",best,"residual MSE:",float(np.min(reference_losses)))
print("PASS: science-04")
