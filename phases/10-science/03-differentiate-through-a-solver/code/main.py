"""Differentiate through a solver: worked experiments and reference solutions. CPU checks."""

# Rebuild the differentiable solver
# Step 1 — Rebuild the differentiable solver
import jax
# Enable 64-bit arithmetic before importing jax.numpy or creating arrays.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def rk4_step(rate, value, dt):
    a = -rate*value
    b = -rate*(value + dt*a/2)
    c = -rate*(value + dt*b/2)
    d = -rate*(value + dt*c)
    return value + dt*(a + 2*b + 2*c + d)/6

# Define `solve(rate, initial, steps, dt)` to carry state across steps with `jax.lax.scan`:
def solve(rate, initial, steps=40, dt=0.05):
    def advance(value, unused):
        next_value = rk4_step(rate, value, dt)
        return next_value, next_value
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

# Check the derivative of the discrete endpoint
# Step 2 — Check the derivative of the discrete endpoint: Autodiff gives the derivative of the actual RK4 program.
rate, initial, steps, dt = 0.7, jnp.array(2.0), 40, 0.05
endpoint = lambda k: solve(k,initial,steps,dt)[-1]
autodiff = float(jax.grad(endpoint)(rate))
z = -rate*dt
r = 1+z+z*z/2+z**3/6+z**4/24
dr_dk = -dt*(1+z+z*z/2+z**3/6)
discrete_gradient = 2.0*steps*r**(steps-1)*dr_dk
continuous_gradient = -2.0*(steps*dt)*np.exp(-rate*steps*dt)
np.testing.assert_allclose(autodiff,discrete_gradient,rtol=1e-11,atol=1e-12)
np.testing.assert_allclose(autodiff,continuous_gradient,rtol=1e-7)
print("Autodiff:",autodiff,"discrete oracle:",discrete_gradient,"continuous oracle:",continuous_gradient)

# Differentiate an observation loss and check its direction
# Step 3 — Differentiate an observation loss and check its direction: The candidate rate is too large.
times = jnp.arange(steps+1)*dt
observations = 2.0*jnp.exp(-0.7*times)
def objective(k):
    return jnp.mean((solve(k,initial,steps,dt)-observations)**2)
probe = 1.0
value, derivative = jax.value_and_grad(objective)(probe)
epsilon = 1e-4
finite_difference = (float(objective(probe+epsilon))-float(objective(probe-epsilon)))/(2*epsilon)
np.testing.assert_allclose(derivative,finite_difference,rtol=2e-7,atol=1e-10)
assert float(derivative) > 0
assert float(objective(probe-0.1*derivative)) < float(value)
print("Loss:",float(value),"gradient:",float(derivative),"finite difference:",finite_difference)

# Step 1 — Rebuild the differentiable solver
import jax
# Enable 64-bit arithmetic before importing jax.numpy or creating arrays.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def rk4_step(rate, value, dt):
    a = -rate*value
    b = -rate*(value + dt*a/2)
    c = -rate*(value + dt*b/2)
    d = -rate*(value + dt*c)
    return value + dt*(a + 2*b + 2*c + d)/6

# Define `solve(rate, initial, steps, dt)` to carry state across steps with `jax.lax.scan`:
def solve(rate, initial, steps=40, dt=0.05):
    def advance(value, unused):
        next_value = rk4_step(rate, value, dt)
        return next_value, next_value
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

# Step 2 — Check the derivative of the discrete endpoint: Autodiff gives the derivative of the actual RK4 program.
rate, initial, steps, dt = 0.7, jnp.array(2.0), 40, 0.05
endpoint = lambda k: solve(k,initial,steps,dt)[-1]
autodiff = float(jax.grad(endpoint)(rate))
z = -rate*dt
r = 1+z+z*z/2+z**3/6+z**4/24
dr_dk = -dt*(1+z+z*z/2+z**3/6)
discrete_gradient = 2.0*steps*r**(steps-1)*dr_dk
continuous_gradient = -2.0*(steps*dt)*np.exp(-rate*steps*dt)
np.testing.assert_allclose(autodiff,discrete_gradient,rtol=1e-11,atol=1e-12)
np.testing.assert_allclose(autodiff,continuous_gradient,rtol=1e-7)
print("Autodiff:",autodiff,"discrete oracle:",discrete_gradient,"continuous oracle:",continuous_gradient)

# Step 3 — Differentiate an observation loss and check its direction: The candidate rate is too large.
times = jnp.arange(steps+1)*dt
observations = 2.0*jnp.exp(-0.7*times)
def objective(k):
    return jnp.mean((solve(k,initial,steps,dt)-observations)**2)
probe = 1.0
value, derivative = jax.value_and_grad(objective)(probe)
epsilon = 1e-4
finite_difference = (float(objective(probe+epsilon))-float(objective(probe-epsilon)))/(2*epsilon)
np.testing.assert_allclose(derivative,finite_difference,rtol=2e-7,atol=1e-10)
assert float(derivative) > 0
assert float(objective(probe-0.1*derivative)) < float(value)
print("Loss:",float(value),"gradient:",float(derivative),"finite difference:",finite_difference)

# Figure data experiment
counts=[5,10,20,40]
errors=[abs(float(jax.grad(lambda k: solve(k,jnp.array(2.0),n,2.0/n)[-1])(0.7))+4*np.exp(-1.4)) for n in counts]
visual_data={'kind':'line','x':counts,'xlabel':'RK4 steps over two seconds','ylabel':'absolute endpoint sensitivity error','yscale':'log','series':[{'label':'autodiff versus continuous derivative','y':errors}]}

# Experiment: Watch derivative discretization error shrink
# Experiment — Watch derivative discretization error shrink
step_counts = np.array([5,10,20,40])
gradient_errors = []
for n in step_counts:
    numerical = jax.grad(lambda k: solve(k,jnp.array(2.0),int(n),2.0/int(n))[-1])(0.7)
    gradient_errors.append(abs(float(numerical)-continuous_gradient))
assert np.all(np.diff(gradient_errors) < 0)
assert 12 < gradient_errors[-2]/gradient_errors[-1] < 20
print("Sensitivity errors:",gradient_errors)

# Experiment: Sweep central-difference perturbations
# Experiment — Sweep central-difference perturbations: Use the measured errors to select a tolerance.
epsilons = [1e-2,1e-3,1e-4,1e-5,1e-6]
fd_errors=[]
for eps in epsilons:
    fd=(float(objective(probe+eps))-float(objective(probe-eps)))/(2*eps)
    fd_errors.append(abs(fd-float(derivative)))
assert min(fd_errors) < 1e-8
print("Finite-difference absolute errors:",fd_errors)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Change the initial value to 3 and the rate to 0.4.
changed_rate, changed_initial = 0.4, 3.0
g_rate = jax.grad(lambda k: solve(k,jnp.array(changed_initial))[-1])(changed_rate)
g_initial = jax.grad(lambda u: solve(changed_rate,u)[-1])(changed_initial)
np.testing.assert_allclose(g_rate,-2*changed_initial*np.exp(-0.8),rtol=1e-7)
np.testing.assert_allclose(g_initial,np.exp(-0.8),rtol=1e-8)
print("Rate and initial-state sensitivities:",float(g_rate),float(g_initial))

# Reference practice: Weight observations without losing normalization
# Weight observations without losing normalization (Practice)
weights=np.linspace(0.2,2.0,41)
def weighted_objective(k):
    residual=solve(k,jnp.array(2.0))-observations
    return jnp.sum(jnp.asarray(weights)*residual**2)/np.sum(weights)
k=1.0
h=0.05
n=np.arange(41)
z=-k*h
r=1+z+z*z/2+z**3/6+z**4/24
dr=-h*(1+z+z*z/2+z**3/6)
u=2*r**n
sensitivity=2*n*r**np.maximum(n-1,0)*dr
reference=np.sum(2*weights*(u-np.asarray(observations))*sensitivity)/weights.sum()
np.testing.assert_allclose(jax.grad(weighted_objective)(k),reference,rtol=1e-11,atol=1e-12)
print("Weighted derivative reference:",reference)

# Reference practice: Diagnose a detached backward path
# Diagnose a detached backward path (Challenge): A plausible forward curve does not validate backward behavior.
def detached_objective(k):
    prediction=jax.lax.stop_gradient(solve(k,jnp.array(2.0)))
    return jnp.mean((prediction-observations)**2)
assert float(jax.grad(detached_objective)(1.0)) == 0.0
fd=(float(detached_objective(1.0001))-float(detached_objective(0.9999)))/0.0002
assert abs(fd)>1e-3
print("Detached autodiff is zero; actual value sensitivity:",fd)

# Reference practice: Check adaptive log-rate sensitivity at multiple tolerances
import numpy as np

def compare_adaptive_sensitivities(log_rate_sensitivity, solve_forced_decay, rate, initial, times, tolerances):
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("times must be a finite vector starting at zero")
    analytic = -rate * times * (initial - 1.0) * np.exp(-rate * times)
    epsilon = 2e-5
    results = {}
    for tolerance in tolerances:
        values = np.asarray(log_rate_sensitivity(
            np.log(rate), initial, times, rtol=tolerance, atol=tolerance * 1e-2
        ))
        plus = solve_forced_decay(
            rate * np.exp(epsilon), initial, times, rtol=tolerance, atol=tolerance * 1e-2
        )
        minus = solve_forced_decay(
            rate * np.exp(-epsilon), initial, times, rtol=tolerance, atol=tolerance * 1e-2
        )
        if not plus["success"] or not minus["success"]:
            raise RuntimeError("finite-difference solve failed")
        finite_difference = (plus["values"] - minus["values"]) / (2.0 * epsilon)
        if values.shape != times.shape or not np.isfinite(values).all():
            raise ValueError("sensitivity API returned invalid values")
        results[float(tolerance)] = {
            "values": values,
            "continuous_max_error": float(np.max(np.abs(values - analytic))),
            "finite_difference_max_error": float(np.max(np.abs(values - finite_difference))),
        }
    return {"analytic": analytic, "by_tolerance": results}

print("PASS: science-03")
