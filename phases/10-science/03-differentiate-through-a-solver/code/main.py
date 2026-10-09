"""Differentiate through a solver: worked experiments and reference solutions. CPU checks."""

# Rebuild the differentiable solver
# Step 1 — Rebuild the differentiable solver: The solver remains a pure JAX function with a static number of...
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

# Check the derivative of the discrete endpoint
# Step 2 — Check the derivative of the discrete endpoint: Autodiff gives the derivative of the actual RK4 program.
# Initialize array `(rate, initial, steps, dt)` with explicit values and shape.
rate, initial, steps, dt = 0.7, jnp.array(2.0), 40, 0.05
# Evaluate `endpoint` from the current inputs and state.
endpoint = lambda k: solve(k,initial,steps,dt)[-1]
# Differentiate the objective to obtain `autodiff` via automatic differentiation.
autodiff = float(jax.grad(endpoint)(rate))
# Evaluate `z` from the current inputs and state.
z = -rate*dt
# Evaluate `r` from the current inputs and state.
r = 1+z+z*z/2+z**3/6+z**4/24
# Evaluate `dr_dk` from the current inputs and state.
dr_dk = -dt*(1+z+z*z/2+z**3/6)
# Evaluate `discrete_gradient` from the current inputs and state.
discrete_gradient = 2.0*steps*r**(steps-1)*dr_dk
# Evaluate `continuous_gradient` from the current inputs and state.
continuous_gradient = -2.0*(steps*dt)*np.exp(-rate*steps*dt)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(autodiff,discrete_gradient,rtol=1e-11,atol=1e-12)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(autodiff,continuous_gradient,rtol=1e-7)
# Print the observed values to compare against the expected result.
print("Autodiff:",autodiff,"discrete oracle:",discrete_gradient,"continuous oracle:",continuous_gradient)

# Differentiate an observation loss and check its direction
# Step 3 — Differentiate an observation loss and check its direction: The candidate rate is too large.
# Initialize array `times` with explicit values and shape.
times = jnp.arange(steps+1)*dt
# Evaluate `observations` from the current inputs and state.
observations = 2.0*jnp.exp(-0.7*times)
# Function `objective(k)` implementing this stage's computation:
def objective(k):
    # Return `jnp.mean((solve(k, initial, steps, dt) - observations) ** 2)` to the caller.
    return jnp.mean((solve(k,initial,steps,dt)-observations)**2)
# Evaluate `probe` from the current inputs and state.
probe = 1.0
# Differentiate the objective to obtain `(value, derivative)` via automatic differentiation.
value, derivative = jax.value_and_grad(objective)(probe)
# Evaluate `epsilon` from the current inputs and state.
epsilon = 1e-4
# Evaluate `finite_difference` from the current inputs and state.
finite_difference = (float(objective(probe+epsilon))-float(objective(probe-epsilon)))/(2*epsilon)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(derivative,finite_difference,rtol=2e-7,atol=1e-10)
# Verify contract: `float(derivative) > 0`.
assert float(derivative) > 0
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert float(objective(probe-0.1*derivative)) < float(value)
# Print the observed values to compare against the expected result.
print("Loss:",float(value),"gradient:",float(derivative),"finite difference:",finite_difference)

# Step 1 — Rebuild the differentiable solver: The solver remains a pure JAX function with a static number of...
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

# Step 2 — Check the derivative of the discrete endpoint: Autodiff gives the derivative of the actual RK4 program.
# Initialize array `(rate, initial, steps, dt)` with explicit values and shape.
rate, initial, steps, dt = 0.7, jnp.array(2.0), 40, 0.05
# Evaluate `endpoint` from the current inputs and state.
endpoint = lambda k: solve(k,initial,steps,dt)[-1]
# Differentiate the objective to obtain `autodiff` via automatic differentiation.
autodiff = float(jax.grad(endpoint)(rate))
# Evaluate `z` from the current inputs and state.
z = -rate*dt
# Evaluate `r` from the current inputs and state.
r = 1+z+z*z/2+z**3/6+z**4/24
# Evaluate `dr_dk` from the current inputs and state.
dr_dk = -dt*(1+z+z*z/2+z**3/6)
# Evaluate `discrete_gradient` from the current inputs and state.
discrete_gradient = 2.0*steps*r**(steps-1)*dr_dk
# Evaluate `continuous_gradient` from the current inputs and state.
continuous_gradient = -2.0*(steps*dt)*np.exp(-rate*steps*dt)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(autodiff,discrete_gradient,rtol=1e-11,atol=1e-12)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(autodiff,continuous_gradient,rtol=1e-7)
# Print the observed values to compare against the expected result.
print("Autodiff:",autodiff,"discrete oracle:",discrete_gradient,"continuous oracle:",continuous_gradient)

# Step 3 — Differentiate an observation loss and check its direction: The candidate rate is too large.
# Initialize array `times` with explicit values and shape.
times = jnp.arange(steps+1)*dt
# Evaluate `observations` from the current inputs and state.
observations = 2.0*jnp.exp(-0.7*times)
# Function `objective(k)` implementing this stage's computation:
def objective(k):
    # Return `jnp.mean((solve(k, initial, steps, dt) - observations) ** 2)` to the caller.
    return jnp.mean((solve(k,initial,steps,dt)-observations)**2)
# Evaluate `probe` from the current inputs and state.
probe = 1.0
# Differentiate the objective to obtain `(value, derivative)` via automatic differentiation.
value, derivative = jax.value_and_grad(objective)(probe)
# Evaluate `epsilon` from the current inputs and state.
epsilon = 1e-4
# Evaluate `finite_difference` from the current inputs and state.
finite_difference = (float(objective(probe+epsilon))-float(objective(probe-epsilon)))/(2*epsilon)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(derivative,finite_difference,rtol=2e-7,atol=1e-10)
# Verify contract: `float(derivative) > 0`.
assert float(derivative) > 0
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert float(objective(probe-0.1*derivative)) < float(value)
# Print the observed values to compare against the expected result.
print("Loss:",float(value),"gradient:",float(derivative),"finite difference:",finite_difference)

# Figure data experiment
# Compute figure data for: A solver gradient converges toward the physical sensitivity
# Evaluate `counts` from the current inputs and state.
counts=[5,10,20,40]
# Create device-backed JAX array `errors`.
errors=[abs(float(jax.grad(lambda k: solve(k,jnp.array(2.0),n,2.0/n)[-1])(0.7))+4*np.exp(-1.4)) for n in counts]
# Evaluate `visual_data` from the current inputs and state.
visual_data={'kind':'line','x':counts,'xlabel':'RK4 steps over two seconds','ylabel':'absolute endpoint sensitivity error','yscale':'log','series':[{'label':'autodiff versus continuous derivative','y':errors}]}

# Experiment: Watch derivative discretization error shrink
# Experiment — Watch derivative discretization error shrink: Fourth-order convergence becomes visible as an error ratio near...
# Initialize array `step_counts` with explicit values and shape.
step_counts = np.array([5,10,20,40])
# Evaluate `gradient_errors` from the current inputs and state.
gradient_errors = []
# Iterate over `n` to step through the computation:
for n in step_counts:
    # Differentiate the objective to obtain `numerical` via automatic differentiation.
    numerical = jax.grad(lambda k: solve(k,jnp.array(2.0),int(n),2.0/int(n))[-1])(0.7)
    # Append the current step result to `gradient_errors`.
    gradient_errors.append(abs(float(numerical)-continuous_gradient))
# Verify contract: `np.all(np.diff(gradient_errors) < 0)`.
assert np.all(np.diff(gradient_errors) < 0)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert 12 < gradient_errors[-2]/gradient_errors[-1] < 20
# Print the observed values to compare against the expected result.
print("Sensitivity errors:",gradient_errors)

# Experiment: Sweep central-difference perturbations
# Experiment — Sweep central-difference perturbations: Use the measured errors to select a tolerance.
epsilons = [1e-2,1e-3,1e-4,1e-5,1e-6]
# Evaluate `fd_errors` from the current inputs and state.
fd_errors=[]
# Iterate over `eps` to step through the computation:
for eps in epsilons:
    # Evaluate `fd` from the current inputs and state.
    fd=(float(objective(probe+eps))-float(objective(probe-eps)))/(2*eps)
    # Append the current step result to `fd_errors`.
    fd_errors.append(abs(fd-float(derivative)))
# Verify contract: `min(fd_errors) < 1e-08`.
assert min(fd_errors) < 1e-8
# Print the observed values to compare against the expected result.
print("Finite-difference absolute errors:",fd_errors)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Change the initial value to 3 and the rate to 0.4.
changed_rate, changed_initial = 0.4, 3.0
# Differentiate the objective to obtain `g_rate` via automatic differentiation.
g_rate = jax.grad(lambda k: solve(k,jnp.array(changed_initial))[-1])(changed_rate)
# Differentiate the objective to obtain `g_initial` via automatic differentiation.
g_initial = jax.grad(lambda u: solve(changed_rate,u)[-1])(changed_initial)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(g_rate,-2*changed_initial*np.exp(-0.8),rtol=1e-7)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(g_initial,np.exp(-0.8),rtol=1e-8)
# Print the observed values to compare against the expected result.
print("Rate and initial-state sensitivities:",float(g_rate),float(g_initial))

# Reference practice: Weight observations without losing normalization
# Weight observations without losing normalization (Practice): The independent expression checks reduction normalization...
# Initialize array `weights` with explicit values and shape.
weights=np.linspace(0.2,2.0,41)
# Function `weighted_objective(k)` implementing this stage's computation:
def weighted_objective(k):
    # Initialize array `residual` with explicit values and shape.
    residual=solve(k,jnp.array(2.0))-observations
    # Return `jnp.sum(jnp.asarray(weights) * residual ** 2) / np.sum(weights)` to the caller.
    return jnp.sum(jnp.asarray(weights)*residual**2)/np.sum(weights)
# Initialize array `n` with explicit values and shape.
k=1.0; h=0.05; n=np.arange(41); z=-k*h
# Evaluate `r` from the current inputs and state.
r=1+z+z*z/2+z**3/6+z**4/24
# Evaluate `dr` from the current inputs and state.
dr=-h*(1+z+z*z/2+z**3/6)
# Evaluate `u` from the current inputs and state.
u=2*r**n
# Reduce across the target axis to summarize `sensitivity`.
sensitivity=2*n*r**np.maximum(n-1,0)*dr
# Aggregate array values to compute `reference`.
reference=np.sum(2*weights*(u-np.asarray(observations))*sensitivity)/weights.sum()
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.grad(weighted_objective)(k),reference,rtol=1e-11,atol=1e-12)
# Print the observed values to compare against the expected result.
print("Weighted derivative reference:",reference)

# Reference practice: Diagnose a detached backward path
# Diagnose a detached backward path (Challenge): A plausible forward curve does not validate backward behavior.
# Define `detached_objective(k)` to evaluate the objective and its automatic derivatives:
def detached_objective(k):
    # Initialize array `prediction` with explicit values and shape.
    prediction=jax.lax.stop_gradient(solve(k,jnp.array(2.0)))
    # Return `jnp.mean((prediction - observations) ** 2)` to the caller.
    return jnp.mean((prediction-observations)**2)
# Verify contract: `float(jax.grad(detached_objective)(1.0)) == 0.0`.
assert float(jax.grad(detached_objective)(1.0)) == 0.0
# Evaluate `fd` from the current inputs and state.
fd=(float(detached_objective(1.0001))-float(detached_objective(0.9999)))/0.0002
# Verify contract: `abs(fd) > 0.001`.
assert abs(fd)>1e-3
# Print the observed values to compare against the expected result.
print("Detached autodiff is zero; actual value sensitivity:",fd)
print("PASS: science-03")
