"""Distributions and sampling: worked experiments and reference solutions. CPU checks."""

# 1. Define a density and its contract
# Step 1 — 1. Define a density and its contract: The normalization term matters when comparing different scales;...
# Import math for this computation.
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

# Function `normal_logpdf(value, loc, scale)` implementing this stage's computation:
def normal_logpdf(value, loc, scale):
    # Return `-0.5 * ((value - loc) / scale) ** 2 - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi)` to the caller.
    return -0.5*((value-loc)/scale)**2 - jnp.log(scale) - 0.5*jnp.log(2*jnp.pi)

# 2. Sample with explicit ownership
# Step 2 — 2. Sample with explicit ownership: The broad mean bound is five analytic standard errors for this...
# Create or split explicit PRNG key(s) (`(key_a, key_b)`) for reproducible randomness.
key_a, key_b = jax.random.split(jax.random.key(23))
# Sample deterministic random values into `a` using an explicit PRNG key.
a = 2. + 3.*jax.random.normal(key_a, (20000,))
# Sample deterministic random values into `b` using an explicit PRNG key.
b = 2. + 3.*jax.random.normal(key_b, (20000,))
# Evaluate `(expected_mean, expected_variance)` from the current inputs and state.
expected_mean, expected_variance = 2., 9.
# Verify contract: `abs(float(a.mean()) - expected_mean) < 5 * 3 / math.sqrt(a.size)`.
assert abs(float(a.mean())-expected_mean) < 5*3/math.sqrt(a.size)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert abs(float(a.var())-expected_variance) < .4
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not jnp.array_equal(a,b)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.array_equal(a, 2.+3.*jax.random.normal(key_a,(20000,)))

# 3. Compare an independent calculation
# Step 3 — 3. Compare an independent calculation: A log density near negative one thousand remains representable...
# Initialize array `actual` with explicit values and shape.
actual = normal_logpdf(jnp.array([2.,5.]),2.,3.)
# Evaluate `reference` from the current inputs and state.
reference = [-math.log(3*math.sqrt(2*math.pi)), -.5-math.log(3*math.sqrt(2*math.pi))]
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(actual,reference,rtol=1e-6)
# Initialize array `components` with explicit values and shape.
components = jnp.array([-1000.,-1001.])
# Evaluate numerically stable log-space cross-entropy/likelihood (`stable`).
stable = logsumexp(components)-jnp.log(2.)
# Confirm that all computed values remain finite (no NaN or Inf).
assert jnp.isfinite(stable)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.isneginf(jnp.log(jnp.exp(components).mean()))
# Print the observed values to compare against the expected result.
print('sample mean, variance:',float(a.mean()),float(a.var()))
# Print diagnostic summary of the computed outputs.
print('stable equal-mixture log density:',float(stable))

# Step 1 — 1. Define a density and its contract: The normalization term matters when comparing different scales;...
# Import math for this computation.
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

# Function `normal_logpdf(value, loc, scale)` implementing this stage's computation:
def normal_logpdf(value, loc, scale):
    # Return `-0.5 * ((value - loc) / scale) ** 2 - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi)` to the caller.
    return -0.5*((value-loc)/scale)**2 - jnp.log(scale) - 0.5*jnp.log(2*jnp.pi)

# Step 2 — 2. Sample with explicit ownership: The broad mean bound is five analytic standard errors for this...
# Create or split explicit PRNG key(s) (`(key_a, key_b)`) for reproducible randomness.
key_a, key_b = jax.random.split(jax.random.key(23))
# Sample deterministic random values into `a` using an explicit PRNG key.
a = 2. + 3.*jax.random.normal(key_a, (20000,))
# Sample deterministic random values into `b` using an explicit PRNG key.
b = 2. + 3.*jax.random.normal(key_b, (20000,))
# Evaluate `(expected_mean, expected_variance)` from the current inputs and state.
expected_mean, expected_variance = 2., 9.
# Verify contract: `abs(float(a.mean()) - expected_mean) < 5 * 3 / math.sqrt(a.size)`.
assert abs(float(a.mean())-expected_mean) < 5*3/math.sqrt(a.size)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert abs(float(a.var())-expected_variance) < .4
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not jnp.array_equal(a,b)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.array_equal(a, 2.+3.*jax.random.normal(key_a,(20000,)))

# Step 3 — 3. Compare an independent calculation: A log density near negative one thousand remains representable...
# Initialize array `actual` with explicit values and shape.
actual = normal_logpdf(jnp.array([2.,5.]),2.,3.)
# Evaluate `reference` from the current inputs and state.
reference = [-math.log(3*math.sqrt(2*math.pi)), -.5-math.log(3*math.sqrt(2*math.pi))]
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(actual,reference,rtol=1e-6)
# Initialize array `components` with explicit values and shape.
components = jnp.array([-1000.,-1001.])
# Evaluate numerically stable log-space cross-entropy/likelihood (`stable`).
stable = logsumexp(components)-jnp.log(2.)
# Confirm that all computed values remain finite (no NaN or Inf).
assert jnp.isfinite(stable)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.isneginf(jnp.log(jnp.exp(components).mean()))
# Print the observed values to compare against the expected result.
print('sample mean, variance:',float(a.mean()),float(a.var()))
# Print diagnostic summary of the computed outputs.
print('stable equal-mixture log density:',float(stable))

# Figure data experiment
# Compute figure data for: A precise mean does not remove observation noise
# Convert `sizes` to a host NumPy array for inspection or verification.
sizes = np.array([1,4,16,64,100,400])
# Evaluate `visual_data` from the current inputs and state.
visual_data = {"kind":"line","x":sizes.tolist(),"xlabel":"independent observations N","ylabel":"standard deviation (reading units)","xscale":"log","series":[{"label":"individual reading","y":[3.]*len(sizes)},{"label":"average of N readings","y":(3/np.sqrt(sizes)).tolist()}]}

# Experiment: Density can exceed one
# Experiment — Density can exceed one: A narrow continuous density can exceed one; its area, not its...
peak = jnp.exp(normal_logpdf(0.,0.,.1))
# Verify contract: `peak > 1`.
assert peak > 1
# Print the observed values to compare against the expected result.
print("peak density:",float(peak))

# Experiment: Separate sum and mixture
# Experiment — Separate sum and mixture: Independence multiplies probabilities, while alternative mixture...
# Initialize array `values` with explicit values and shape.
values = jnp.array([-2.,-3.])
# Aggregate array values to compute `independent_log`.
independent_log = values.sum()
# Evaluate numerically stable log-space cross-entropy/likelihood (`mixture_log`).
mixture_log = logsumexp(values)-jnp.log(2.)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(independent_log,-5.)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert mixture_log > -3.
# Print the observed values to compare against the expected result.
print(float(independent_log),float(mixture_log))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Change the sensor to mean -1, standard deviation 0.5, and draw 30000...
# Create or split explicit PRNG key(s) (`changed`) for reproducible randomness.
changed = -1.+.5*jax.random.normal(jax.random.key(51),(30000,))
# Verify contract: `abs(float(changed.mean()) + 1.0) < 5 * 0.5 / math.sqrt(30000)`.
assert abs(float(changed.mean())+1.) < 5*.5/math.sqrt(30000)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert abs(float(changed.var())-.25) < .015

# Reference practice: Detect a missing normalization
# Detect a missing normalization (Transfer / diagnosis): Scale changes density even with identical zero residuals.
# Initialize array `scores` with explicit values and shape.
scores = normal_logpdf(0.,0.,jnp.array([1.,2.]))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(scores[0]-scores[1],jnp.log(2.))

# Reference practice: Demonstrate key ownership
# Demonstrate key ownership (Transfer / diagnosis): Different split keys avoid accidental replay; one differing...
# Create or split explicit PRNG key(s) (`(left, right)`) for reproducible randomness.
left,right = jax.random.split(jax.random.key(17))
# Sample deterministic random values into `first` using an explicit PRNG key.
first = jax.random.normal(left,(32,))
# Sample deterministic random values into `second` using an explicit PRNG key.
second = jax.random.normal(right,(32,))
# Verify contract: `jnp.array_equal(first, jax.random.normal(left, (32,)))`.
assert jnp.array_equal(first,jax.random.normal(left,(32,)))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not jnp.array_equal(first,second)
print("PASS: probability-01")
