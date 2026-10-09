"""Build a Bayesian regression: worked experiments and reference solutions. CPU checks."""

# 1. Construct a small linear model
# Step 1 — 1. Construct a small linear model: The design has three rows and two columns.
# Import numpy for this computation.
import numpy as np
import jax
import jax.numpy as jnp
# Initialize array `X` with explicit values and shape.
X = jnp.array([[1.,-1.],[1.,0.],[1.,1.]])
# Initialize array `y` with explicit values and shape.
y = jnp.array([-1.,1.,3.])
# Evaluate `(sigma, prior_scale)` from the current inputs and state.
sigma, prior_scale = 1., 2.

# 2. Solve the posterior system
# Step 2 — 2. Solve the posterior system: The independent fractions follow because centered inputs make the...
def posterior(X,y,sigma,prior_scale):
    # Initialize array `precision` with explicit values and shape.
    precision = jnp.eye(X.shape[1])/prior_scale**2 + X.T@X/sigma**2
    # Perform matrix contraction / projection to compute `mean`.
    mean = jnp.linalg.solve(precision,X.T@y/sigma**2)
    # Initialize array `covariance` with explicit values and shape.
    covariance = jnp.linalg.solve(precision,jnp.eye(X.shape[1]))
    # Return `(mean, covariance)` to the caller.
    return mean,covariance
# Run `posterior` to compute `(mean, cov)`.
mean,cov = posterior(X,y,sigma,prior_scale)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(mean,jnp.array([12/13,16/9]),atol=1e-6)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(cov,jnp.diag(jnp.array([4/13,4/9])),atol=1e-6)

# 3. Predict both the latent mean and a new observation
# Step 3 — 3. Predict both the latent mean and a new observation: The variance at the extrapolation input is larger because slope...
def predictive(design,mean,cov,sigma):
    # Perform matrix / vector contraction (`@`) to compute `center`.
    center = design@mean
    # Perform matrix contraction / projection to compute `latent_variance`.
    latent_variance = jnp.einsum('ni,ij,nj->n',design,cov,design)
    # Return `(center, latent_variance, latent_variance + sigma ** 2)` to the caller.
    return center,latent_variance,latent_variance+sigma**2
# Initialize array `query` with explicit values and shape.
query = jnp.array([[1.,0.],[1.,2.]])
# Run `predictive` to compute `(center, latent_var, observation_var)`.
center,latent_var,observation_var = predictive(query,mean,cov,sigma)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(latent_var,jnp.array([4/13,4/13+16/9]),atol=1e-6)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(observation_var-latent_var,1.)
# Print the observed values to compare against the expected result.
print('posterior mean:',mean,'covariance:',cov)
# Print diagnostic summary of the computed outputs.
print('latent variance:',latent_var,'observation variance:',observation_var)

# Step 1 — 1. Construct a small linear model: The design has three rows and two columns.
# Import numpy for this computation.
import numpy as np
import jax
import jax.numpy as jnp
# Initialize array `X` with explicit values and shape.
X = jnp.array([[1.,-1.],[1.,0.],[1.,1.]])
# Initialize array `y` with explicit values and shape.
y = jnp.array([-1.,1.,3.])
# Evaluate `(sigma, prior_scale)` from the current inputs and state.
sigma, prior_scale = 1., 2.

# Step 2 — 2. Solve the posterior system: The independent fractions follow because centered inputs make the...
def posterior(X,y,sigma,prior_scale):
    # Initialize array `precision` with explicit values and shape.
    precision = jnp.eye(X.shape[1])/prior_scale**2 + X.T@X/sigma**2
    # Perform matrix contraction / projection to compute `mean`.
    mean = jnp.linalg.solve(precision,X.T@y/sigma**2)
    # Initialize array `covariance` with explicit values and shape.
    covariance = jnp.linalg.solve(precision,jnp.eye(X.shape[1]))
    # Return `(mean, covariance)` to the caller.
    return mean,covariance
# Run `posterior` to compute `(mean, cov)`.
mean,cov = posterior(X,y,sigma,prior_scale)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(mean,jnp.array([12/13,16/9]),atol=1e-6)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(cov,jnp.diag(jnp.array([4/13,4/9])),atol=1e-6)

# Step 3 — 3. Predict both the latent mean and a new observation: The variance at the extrapolation input is larger because slope...
def predictive(design,mean,cov,sigma):
    # Perform matrix / vector contraction (`@`) to compute `center`.
    center = design@mean
    # Perform matrix contraction / projection to compute `latent_variance`.
    latent_variance = jnp.einsum('ni,ij,nj->n',design,cov,design)
    # Return `(center, latent_variance, latent_variance + sigma ** 2)` to the caller.
    return center,latent_variance,latent_variance+sigma**2
# Initialize array `query` with explicit values and shape.
query = jnp.array([[1.,0.],[1.,2.]])
# Run `predictive` to compute `(center, latent_var, observation_var)`.
center,latent_var,observation_var = predictive(query,mean,cov,sigma)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(latent_var,jnp.array([4/13,4/13+16/9]),atol=1e-6)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(observation_var-latent_var,1.)
# Print the observed values to compare against the expected result.
print('posterior mean:',mean,'covariance:',cov)
# Print diagnostic summary of the computed outputs.
print('latent variance:',latent_var,'observation variance:',observation_var)

# Figure data experiment
# Compute figure data for: Prediction uncertainty grows away from the observed center
# Generate a uniform grid of points in `grid`.
grid=jnp.linspace(-3,3,61)
# Allocate initialized array `(_, lv, ov)` with the specified shape and dtype.
_,lv,ov=predictive(jnp.stack([jnp.ones_like(grid),grid],axis=1),mean,cov,sigma)
# Evaluate `visual_data` from the current inputs and state.
visual_data={"kind":"line","x":grid.tolist(),"xlabel":"input value","ylabel":"predictive standard deviation (target units)","series":[{"label":"latent line","y":jnp.sqrt(lv).tolist()},{"label":"new observation","y":jnp.sqrt(ov).tolist()}]}

# Experiment: Repeat independent observations
# Experiment — Repeat independent observations: The calculation treats rows as independent new evidence.
repeat_mean,repeat_cov = posterior(jnp.tile(X,(2,1)),jnp.tile(y,2),sigma,prior_scale)
# Verify contract: `jnp.all(jnp.diag(repeat_cov) < jnp.diag(cov))`.
assert jnp.all(jnp.diag(repeat_cov)<jnp.diag(cov))
# Print the observed values to compare against the expected result.
print(jnp.diag(repeat_cov))

# Experiment: Check the prior-only boundary
# Experiment — Check the prior-only boundary: The posterior recovers the prior when no likelihood information...
empty_mean,empty_cov = posterior(jnp.empty((0,2)),jnp.empty((0,)),sigma,prior_scale)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(empty_mean,0.)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(empty_cov,4*jnp.eye(2))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Strengthen the prior by changing its scale from two to one half.
strong_mean,strong_cov = posterior(X,y,sigma,.5)
# Verify contract: `jnp.linalg.norm(strong_mean) < jnp.linalg.norm(mean)`.
assert jnp.linalg.norm(strong_mean)<jnp.linalg.norm(mean)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.all(jnp.diag(strong_cov)<jnp.diag(cov))

# Reference practice: Move away from centered inputs
# Move away from centered inputs (Transfer / diagnosis): Nonzero covariance affects prediction uncertainty; keeping...
# Initialize array `shift_X` with explicit values and shape.
shift_X = jnp.array([[1.,0.],[1.,1.],[1.,2.]])
# Run `posterior` to compute `(shift_m, shift_c)`.
shift_m,shift_c = posterior(shift_X,y,sigma,prior_scale)
# Convert `host_X` to a host NumPy array for inspection or verification.
host_X=np.asarray(shift_X,dtype=np.float64)
# Construct an identity matrix `host_P`.
host_P=np.eye(2)/4+host_X.T@host_X
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(shift_m,np.linalg.solve(host_P,host_X.T@np.asarray(y)),rtol=1e-5,atol=1e-6)
# Verify contract: `shift_c[0, 1] < 0`.
assert shift_c[0,1]<0

# Reference practice: Verify total predictive variance
# Verify total predictive variance (Transfer / diagnosis): A simulation verifies that the independent noise term...
# Create or split explicit PRNG key(s) (`(kw, kn)`) for reproducible randomness.
kw,kn=jax.random.split(jax.random.key(37))
# Sample deterministic random values into `weights` using an explicit PRNG key.
weights=mean+jax.random.normal(kw,(30000,2))@jnp.linalg.cholesky(cov).T
# Sample deterministic random values into `replicated` using an explicit PRNG key.
replicated=weights[:,0]+sigma*jax.random.normal(kn,(30000,))
# Verify contract: `abs(float(replicated.var()) - float(observation_var[0])) < 0.07`.
assert abs(float(replicated.var())-float(observation_var[0]))<.07
print("PASS: probability-02")
