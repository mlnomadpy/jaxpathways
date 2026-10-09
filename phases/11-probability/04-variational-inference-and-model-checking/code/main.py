"""Variational inference and model checking: worked experiments and reference solutions. CPU checks."""

# 1. Specify a correlated target and a simpler approximation
# Step 1 — 1. Specify a correlated target and a simpler approximation: KL here means KL(q || p).
# Import numpy for this computation.
import numpy as np
import jax
import jax.numpy as jnp
# Initialize array `mu` with explicit values and shape.
mu=jnp.array([1.,-1.])
# Initialize array `Sigma` with explicit values and shape.
Sigma=jnp.array([[1.,.8],[.8,1.]])
# Initialize array `P` with explicit values and shape.
P=jnp.linalg.solve(Sigma,jnp.eye(2))
# Function `kl(params)` implementing this stage's computation:
def kl(params):
    # Evaluate `(location, log_scale)` from the current inputs and state.
    location,log_scale=params
    # Run `jnp.exp` to compute `variance`.
    variance=jnp.exp(2*log_scale)
    # Evaluate `delta` from the current inputs and state.
    delta=location-mu
    # Return `0.5 * (jnp.sum(jnp.diag(P) * variance) + delta @ P @ delta - 2 + jnp.linalg.slogdet(Sigma)[1] - jnp.sum(2 * log_scale))` to the caller.
    return .5*(jnp.sum(jnp.diag(P)*variance)+delta@P@delta-2+jnp.linalg.slogdet(Sigma)[1]-jnp.sum(2*log_scale))

# 2. Optimize and compare to the exact variational optimum
# Step 2 — 2. Optimize and compare to the exact variational optimum: Optimization has found the best diagonal Gaussian in this KL...
# Define `update(params, _)` to evaluate the objective and its automatic derivatives:
def update(params,_):
    # Differentiate the objective to obtain `grads` via automatic differentiation.
    grads=jax.grad(kl)(params)
    # Transform every leaf of the parameter PyTree (`new`).
    new=jax.tree.map(lambda value,grad:value-.05*grad,params,grads)
    # Return `(new, kl(new))` to the caller.
    return new,kl(new)
# Run compiled structured control flow via `jax.lax` (`(params, history)`).
params,history=jax.lax.scan(update,(jnp.zeros(2),jnp.zeros(2)),None,length=600)
# Evaluate `(location, log_scale)` from the current inputs and state.
location,log_scale=params
# Run `jnp.exp` to compute `variance`.
variance=jnp.exp(2*log_scale)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(location,mu,atol=1e-4)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(variance,1/jnp.diag(P),atol=1e-5)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(variance,jnp.array([.36,.36]),atol=1e-5)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert kl(params)>0.4
# Print the observed values to compare against the expected result.
print('VI mean:',location,'variances:',variance,'remaining KL:',float(kl(params)))

# 3. Check a prediction under a changed data-generating process
# Step 3 — 3. Check a prediction under a changed data-generating process: This is a deterministic stress fixture with nine targets, not an...
# Initialize array `x` with explicit values and shape.
x=jnp.linspace(-2,2,9)
# Initialize array `linear_mean` with explicit values and shape.
linear_mean=jnp.zeros_like(x)
# Evaluate `noise_scale` from the current inputs and state.
noise_scale=.25
# Evaluate `curved_targets` from the current inputs and state.
curved_targets=x**2-1.
# Evaluate `standardized_residual` from the current inputs and state.
standardized_residual=(curved_targets-linear_mean)/noise_scale
# Aggregate array values to compute `coverage`.
coverage=jnp.mean(jnp.abs(standardized_residual)<=1.96)
# Verify contract: `coverage < 0.4`.
assert coverage<.4
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert standardized_residual[0]>10
# Print the observed values to compare against the expected result.
print('synthetic nominal-95% interval coverage:',float(coverage))
# Print diagnostic summary of the computed outputs.
print('standardized residuals:',standardized_residual)

# Step 1 — 1. Specify a correlated target and a simpler approximation: KL here means KL(q || p).
# Import numpy for this computation.
import numpy as np
import jax
import jax.numpy as jnp
# Initialize array `mu` with explicit values and shape.
mu=jnp.array([1.,-1.])
# Initialize array `Sigma` with explicit values and shape.
Sigma=jnp.array([[1.,.8],[.8,1.]])
# Initialize array `P` with explicit values and shape.
P=jnp.linalg.solve(Sigma,jnp.eye(2))
# Function `kl(params)` implementing this stage's computation:
def kl(params):
    # Evaluate `(location, log_scale)` from the current inputs and state.
    location,log_scale=params
    # Run `jnp.exp` to compute `variance`.
    variance=jnp.exp(2*log_scale)
    # Evaluate `delta` from the current inputs and state.
    delta=location-mu
    # Return `0.5 * (jnp.sum(jnp.diag(P) * variance) + delta @ P @ delta - 2 + jnp.linalg.slogdet(Sigma)[1] - jnp.sum(2 * log_scale))` to the caller.
    return .5*(jnp.sum(jnp.diag(P)*variance)+delta@P@delta-2+jnp.linalg.slogdet(Sigma)[1]-jnp.sum(2*log_scale))

# Step 2 — 2. Optimize and compare to the exact variational optimum: Optimization has found the best diagonal Gaussian in this KL...
# Define `update(params, _)` to evaluate the objective and its automatic derivatives:
def update(params,_):
    # Differentiate the objective to obtain `grads` via automatic differentiation.
    grads=jax.grad(kl)(params)
    # Transform every leaf of the parameter PyTree (`new`).
    new=jax.tree.map(lambda value,grad:value-.05*grad,params,grads)
    # Return `(new, kl(new))` to the caller.
    return new,kl(new)
# Run compiled structured control flow via `jax.lax` (`(params, history)`).
params,history=jax.lax.scan(update,(jnp.zeros(2),jnp.zeros(2)),None,length=600)
# Evaluate `(location, log_scale)` from the current inputs and state.
location,log_scale=params
# Run `jnp.exp` to compute `variance`.
variance=jnp.exp(2*log_scale)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(location,mu,atol=1e-4)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(variance,1/jnp.diag(P),atol=1e-5)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(variance,jnp.array([.36,.36]),atol=1e-5)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert kl(params)>0.4
# Print the observed values to compare against the expected result.
print('VI mean:',location,'variances:',variance,'remaining KL:',float(kl(params)))

# Step 3 — 3. Check a prediction under a changed data-generating process: This is a deterministic stress fixture with nine targets, not an...
# Initialize array `x` with explicit values and shape.
x=jnp.linspace(-2,2,9)
# Initialize array `linear_mean` with explicit values and shape.
linear_mean=jnp.zeros_like(x)
# Evaluate `noise_scale` from the current inputs and state.
noise_scale=.25
# Evaluate `curved_targets` from the current inputs and state.
curved_targets=x**2-1.
# Evaluate `standardized_residual` from the current inputs and state.
standardized_residual=(curved_targets-linear_mean)/noise_scale
# Aggregate array values to compute `coverage`.
coverage=jnp.mean(jnp.abs(standardized_residual)<=1.96)
# Verify contract: `coverage < 0.4`.
assert coverage<.4
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert standardized_residual[0]>10
# Print the observed values to compare against the expected result.
print('synthetic nominal-95% interval coverage:',float(coverage))
# Print diagnostic summary of the computed outputs.
print('standardized residuals:',standardized_residual)

# Figure data experiment
# Compute figure data for: An accurate mean can coexist with underestimated uncertainty
# Reduce across the target axis to summarize `visual_data`.
visual_data={"kind":"bar","x":[0,1,2],"labels":["weight 1","weight 2","sum of weights"],"xlabel":"quantity","ylabel":"variance (squared parameter units)","series":[{"label":"exact target","y":[1.,1.,3.6]},{"label":"optimized diagonal q","y":[float(variance[0]),float(variance[1]),float(variance.sum())]}]}

# Experiment: Monte Carlo KL through reparameterization
# Experiment — Monte Carlo KL through reparameterization: Matching the exact KL validates this estimator on the fixture;...
# Create or split explicit PRNG key(s) (`eps`) for reproducible randomness.
eps=jax.random.normal(jax.random.key(52),(40000,2))
# Evaluate `z` from the current inputs and state.
z=location+jnp.exp(log_scale)*eps
# Evaluate `delta` from the current inputs and state.
delta=z-mu
# Perform matrix contraction / projection to compute `logp`.
logp=-.5*(jnp.einsum('ni,ij,nj->n',delta,P,delta)+2*jnp.log(2*jnp.pi)+jnp.linalg.slogdet(Sigma)[1])
# Reduce along axis=1 to compute `logq`.
logq=-.5*jnp.sum(eps**2,axis=1)-jnp.sum(log_scale)-jnp.log(2*jnp.pi)
# Aggregate array values to compute `estimate`.
estimate=jnp.mean(logq-logp)
# Verify contract: `abs(float(estimate - kl(params))) < 0.03`.
assert abs(float(estimate-kl(params)))<.03
# Print the observed values to compare against the expected result.
print('Monte Carlo KL:',float(estimate))

# Experiment: Count model-generated intervals
# Experiment — Count model-generated intervals: Self-coverage checks interval arithmetic under the model.
# Create or split explicit PRNG key(s) (`replicated`) for reproducible randomness.
replicated=.25*jax.random.normal(jax.random.key(83),(40000,))
# Aggregate array values to compute `model_coverage`.
model_coverage=jnp.mean(jnp.abs(replicated)<=1.96*.25)
# Verify contract: `0.94 < model_coverage < 0.96`.
assert .94<model_coverage<.96
# Print the observed values to compare against the expected result.
print("model self-coverage:",float(model_coverage))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Replace the target correlation by zero and rederive the optimal...
def independent_kl(location,log_scale):
    # Return `0.5 * (jnp.sum(jnp.exp(2 * log_scale)) + jnp.sum((location - mu) ** 2) - 2 - jnp.sum(2 * log_scale))` to the caller.
    return .5*(jnp.sum(jnp.exp(2*log_scale))+jnp.sum((location-mu)**2)-2-jnp.sum(2*log_scale))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(independent_kl(mu,jnp.zeros(2)),0.)

# Reference practice: A correct diagonal is not a correct covariance
# A correct diagonal is not a correct covariance (Transfer / diagnosis): Joint predictions can reveal approximation error that...
# Initialize array `direction` with explicit values and shape.
direction=jnp.ones(2)
# Perform matrix / vector contraction (`@`) to compute `true_sum_var`.
true_sum_var=direction@Sigma@direction
# Aggregate array values to compute `approx_sum_var`.
approx_sum_var=jnp.sum(variance)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(true_sum_var,3.6)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(approx_sum_var,.72,atol=1e-5)

# Reference practice: Repair the changed mean without hiding noise
# Repair the changed mean without hiding noise (Transfer / diagnosis): Correcting the mean removes systematic residual structure in...
# Initialize array `truth` with explicit values and shape.
truth=jnp.linspace(-2,2,20000)**2-1
# Create or split explicit PRNG key(s) (`fresh`) for reproducible randomness.
fresh=truth+.25*jax.random.normal(jax.random.key(95),(20000,))
# Evaluate `residual` from the current inputs and state.
residual=fresh-truth
# Verify contract: `abs(float(residual.std()) - 0.25) < 0.01`.
assert abs(float(residual.std())-.25)<.01
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert abs(float(residual.mean()))<.01
print("PASS: probability-04")
