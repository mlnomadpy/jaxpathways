"""Build a Bayesian regression: worked experiments and reference solutions. CPU checks."""

# 1. Construct a small linear model
import numpy as np
import jax
import jax.numpy as jnp
X = jnp.array([[1.,-1.],[1.,0.],[1.,1.]])
y = jnp.array([-1.,1.,3.])
sigma, prior_scale = 1., 2.

# 2. Solve the posterior system
def posterior(X,y,sigma,prior_scale):
    precision = jnp.eye(X.shape[1])/prior_scale**2 + X.T@X/sigma**2
    mean = jnp.linalg.solve(precision,X.T@y/sigma**2)
    covariance = jnp.linalg.solve(precision,jnp.eye(X.shape[1]))
    return mean,covariance
mean,cov = posterior(X,y,sigma,prior_scale)
assert jnp.allclose(mean,jnp.array([12/13,16/9]),atol=1e-6)
assert jnp.allclose(cov,jnp.diag(jnp.array([4/13,4/9])),atol=1e-6)

# 3. Predict both the latent mean and a new observation
def predictive(design,mean,cov,sigma):
    center = design@mean
    latent_variance = jnp.einsum('ni,ij,nj->n',design,cov,design)
    return center,latent_variance,latent_variance+sigma**2
query = jnp.array([[1.,0.],[1.,2.]])
center,latent_var,observation_var = predictive(query,mean,cov,sigma)
assert jnp.allclose(latent_var,jnp.array([4/13,4/13+16/9]),atol=1e-6)
assert jnp.allclose(observation_var-latent_var,1.)
print('posterior mean:',mean,'covariance:',cov)
print('latent variance:',latent_var,'observation variance:',observation_var)

import numpy as np
import jax
import jax.numpy as jnp
X = jnp.array([[1.,-1.],[1.,0.],[1.,1.]])
y = jnp.array([-1.,1.,3.])
sigma, prior_scale = 1., 2.

def posterior(X,y,sigma,prior_scale):
    precision = jnp.eye(X.shape[1])/prior_scale**2 + X.T@X/sigma**2
    mean = jnp.linalg.solve(precision,X.T@y/sigma**2)
    covariance = jnp.linalg.solve(precision,jnp.eye(X.shape[1]))
    return mean,covariance
mean,cov = posterior(X,y,sigma,prior_scale)
assert jnp.allclose(mean,jnp.array([12/13,16/9]),atol=1e-6)
assert jnp.allclose(cov,jnp.diag(jnp.array([4/13,4/9])),atol=1e-6)

def predictive(design,mean,cov,sigma):
    center = design@mean
    latent_variance = jnp.einsum('ni,ij,nj->n',design,cov,design)
    return center,latent_variance,latent_variance+sigma**2
query = jnp.array([[1.,0.],[1.,2.]])
center,latent_var,observation_var = predictive(query,mean,cov,sigma)
assert jnp.allclose(latent_var,jnp.array([4/13,4/13+16/9]),atol=1e-6)
assert jnp.allclose(observation_var-latent_var,1.)
print('posterior mean:',mean,'covariance:',cov)
print('latent variance:',latent_var,'observation variance:',observation_var)

# Figure data experiment
grid=jnp.linspace(-3,3,61)
_,lv,ov=predictive(jnp.stack([jnp.ones_like(grid),grid],axis=1),mean,cov,sigma)
visual_data={"kind":"line","x":grid.tolist(),"xlabel":"input value","ylabel":"predictive standard deviation (target units)","series":[{"label":"latent line","y":jnp.sqrt(lv).tolist()},{"label":"new observation","y":jnp.sqrt(ov).tolist()}]}

# Experiment: Repeat independent observations
repeat_mean,repeat_cov = posterior(jnp.tile(X,(2,1)),jnp.tile(y,2),sigma,prior_scale)
assert jnp.all(jnp.diag(repeat_cov)<jnp.diag(cov))
print(jnp.diag(repeat_cov))

# Experiment: Check the prior-only boundary
empty_mean,empty_cov = posterior(jnp.empty((0,2)),jnp.empty((0,)),sigma,prior_scale)
assert jnp.allclose(empty_mean,0.)
assert jnp.allclose(empty_cov,4*jnp.eye(2))

# Reference solution. Try the exercise before reading this.
strong_mean,strong_cov = posterior(X,y,sigma,.5)
assert jnp.linalg.norm(strong_mean)<jnp.linalg.norm(mean)
assert jnp.all(jnp.diag(strong_cov)<jnp.diag(cov))

# Reference practice: Move away from centered inputs
shift_X = jnp.array([[1.,0.],[1.,1.],[1.,2.]])
shift_m,shift_c = posterior(shift_X,y,sigma,prior_scale)
host_X=np.asarray(shift_X,dtype=np.float64)
host_P=np.eye(2)/4+host_X.T@host_X
np.testing.assert_allclose(shift_m,np.linalg.solve(host_P,host_X.T@np.asarray(y)),rtol=1e-5,atol=1e-6)
assert shift_c[0,1]<0

# Reference practice: Verify total predictive variance
kw,kn=jax.random.split(jax.random.key(37))
weights=mean+jax.random.normal(kw,(30000,2))@jnp.linalg.cholesky(cov).T
replicated=weights[:,0]+sigma*jax.random.normal(kn,(30000,))
assert abs(float(replicated.var())-float(observation_var[0]))<.07
print("PASS: probability-02")
