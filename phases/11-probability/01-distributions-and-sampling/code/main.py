"""Distributions and sampling: worked experiments and reference solutions. CPU checks."""

# 1. Define a density and its contract
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

def normal_logpdf(value, loc, scale):
    return -0.5*((value-loc)/scale)**2 - jnp.log(scale) - 0.5*jnp.log(2*jnp.pi)

# 2. Sample with explicit ownership
key_a, key_b = jax.random.split(jax.random.key(23))
a = 2. + 3.*jax.random.normal(key_a, (20000,))
b = 2. + 3.*jax.random.normal(key_b, (20000,))
expected_mean, expected_variance = 2., 9.
assert abs(float(a.mean())-expected_mean) < 5*3/math.sqrt(a.size)
assert abs(float(a.var())-expected_variance) < .4
assert not jnp.array_equal(a,b)
assert jnp.array_equal(a, 2.+3.*jax.random.normal(key_a,(20000,)))

# 3. Compare an independent calculation
actual = normal_logpdf(jnp.array([2.,5.]),2.,3.)
reference = [-math.log(3*math.sqrt(2*math.pi)), -.5-math.log(3*math.sqrt(2*math.pi))]
np.testing.assert_allclose(actual,reference,rtol=1e-6)
components = jnp.array([-1000.,-1001.])
stable = logsumexp(components)-jnp.log(2.)
assert jnp.isfinite(stable)
assert jnp.isneginf(jnp.log(jnp.exp(components).mean()))
print('sample mean, variance:',float(a.mean()),float(a.var()))
print('stable equal-mixture log density:',float(stable))

import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

def normal_logpdf(value, loc, scale):
    return -0.5*((value-loc)/scale)**2 - jnp.log(scale) - 0.5*jnp.log(2*jnp.pi)

key_a, key_b = jax.random.split(jax.random.key(23))
a = 2. + 3.*jax.random.normal(key_a, (20000,))
b = 2. + 3.*jax.random.normal(key_b, (20000,))
expected_mean, expected_variance = 2., 9.
assert abs(float(a.mean())-expected_mean) < 5*3/math.sqrt(a.size)
assert abs(float(a.var())-expected_variance) < .4
assert not jnp.array_equal(a,b)
assert jnp.array_equal(a, 2.+3.*jax.random.normal(key_a,(20000,)))

actual = normal_logpdf(jnp.array([2.,5.]),2.,3.)
reference = [-math.log(3*math.sqrt(2*math.pi)), -.5-math.log(3*math.sqrt(2*math.pi))]
np.testing.assert_allclose(actual,reference,rtol=1e-6)
components = jnp.array([-1000.,-1001.])
stable = logsumexp(components)-jnp.log(2.)
assert jnp.isfinite(stable)
assert jnp.isneginf(jnp.log(jnp.exp(components).mean()))
print('sample mean, variance:',float(a.mean()),float(a.var()))
print('stable equal-mixture log density:',float(stable))

# Figure data experiment
sizes = np.array([1,4,16,64,100,400])
visual_data = {"kind":"line","x":sizes.tolist(),"xlabel":"independent observations N","ylabel":"standard deviation (reading units)","xscale":"log","series":[{"label":"individual reading","y":[3.]*len(sizes)},{"label":"average of N readings","y":(3/np.sqrt(sizes)).tolist()}]}

# Experiment: Density can exceed one
peak = jnp.exp(normal_logpdf(0.,0.,.1))
assert peak > 1
print("peak density:",float(peak))

# Experiment: Separate sum and mixture
values = jnp.array([-2.,-3.])
independent_log = values.sum()
mixture_log = logsumexp(values)-jnp.log(2.)
assert jnp.allclose(independent_log,-5.)
assert mixture_log > -3.
print(float(independent_log),float(mixture_log))

# Reference solution. Try the exercise before reading this.
changed = -1.+.5*jax.random.normal(jax.random.key(51),(30000,))
assert abs(float(changed.mean())+1.) < 5*.5/math.sqrt(30000)
assert abs(float(changed.var())-.25) < .015

# Reference practice: Detect a missing normalization
scores = normal_logpdf(0.,0.,jnp.array([1.,2.]))
assert jnp.allclose(scores[0]-scores[1],jnp.log(2.))

# Reference practice: Demonstrate key ownership
left,right = jax.random.split(jax.random.key(17))
first = jax.random.normal(left,(32,))
second = jax.random.normal(right,(32,))
assert jnp.array_equal(first,jax.random.normal(left,(32,)))
assert not jnp.array_equal(first,second)
print("PASS: probability-01")
