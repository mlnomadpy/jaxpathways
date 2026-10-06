"""CPU teaching implementation: Gaussian regression, fixed-trajectory HMC, prediction."""
import jax
import jax.numpy as jnp
import numpy as np


def fit(X, y, noise_scale=1., prior_scale=2.):
    """Zero-mean isotropic Gaussian prior; known positive observation-noise scale."""
    X, y = jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0] or X.shape[1] == 0:
        raise ValueError("X must be (observations, features), y must be (observations,)")
    if not np.isfinite(np.asarray(X)).all() or not np.isfinite(np.asarray(y)).all():
        raise ValueError("observations must be finite")
    if not np.isfinite([noise_scale, prior_scale]).all() or min(noise_scale, prior_scale) <= 0:
        raise ValueError("scales must be positive and finite")
    precision = jnp.eye(X.shape[1])/prior_scale**2 + X.T@X/noise_scale**2
    mean = jnp.linalg.solve(precision, X.T@y/noise_scale**2)
    covariance = jnp.linalg.solve(precision, jnp.eye(X.shape[1]))
    return mean, covariance


def log_joint(weights, X, y, noise_scale=1., prior_scale=2.):
    """Normalized prior and independent Gaussian likelihood; weights is one vector."""
    residual = y-X@weights
    n, d = X.shape
    return (-.5*jnp.sum(residual**2)/noise_scale**2-n*jnp.log(noise_scale)
            -.5*jnp.sum(weights**2)/prior_scale**2-d*jnp.log(prior_scale)
            -.5*(n+d)*jnp.log(2*jnp.pi))


def sample(key, mean, covariance, starts, step_size=.15, leapfrog_steps=9,
           warmup=300, draws=1800):
    """Return retained chains, acceptance flags and signed energy errors.
    Fixed warmup discard; no adaptive step size, NUTS or mass-matrix adaptation.
    """
    mean, covariance, starts = map(jnp.asarray, (mean, covariance, starts))
    d = mean.size
    if mean.ndim != 1 or covariance.shape != (d,d) or starts.ndim != 2 or starts.shape[1] != d:
        raise ValueError("mean, covariance and chain-start shapes disagree")
    if draws < 4 or warmup < 0 or leapfrog_steps < 1 or not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("invalid sampler configuration")
    host_cov = np.asarray(covariance)
    if (not np.isfinite(host_cov).all()
        or not np.allclose(host_cov, host_cov.T, atol=1e-6)
        or np.linalg.eigvalsh(host_cov).min() <= 0):
        raise ValueError("covariance must be finite, symmetric and positive definite")
    if not np.isfinite(np.asarray(mean)).all() or not np.isfinite(np.asarray(starts)).all():
        raise ValueError("mean and starts must be finite")
    precision = jnp.linalg.solve(covariance, jnp.eye(d))
    def energy(q):
        delta = q-mean
        return .5*delta@precision@delta
    force = jax.grad(energy)

    def one_chain(chain_key, start):
        def transition(state, _):
            q, chain_key = state
            chain_key, momentum_key, accept_key = jax.random.split(chain_key,3)
            initial_p = jax.random.normal(momentum_key,q.shape)
            p = initial_p-.5*step_size*force(q)
            def leap(i,state):
                position,momentum = state
                position = position+step_size*momentum
                momentum = momentum-jnp.where(i<leapfrog_steps-1,step_size,.5*step_size)*force(position)
                return position,momentum
            proposed_q,proposed_p = jax.lax.fori_loop(0,leapfrog_steps,leap,(q,p))
            error = energy(proposed_q)+.5*jnp.sum(proposed_p**2)-energy(q)-.5*jnp.sum(initial_p**2)
            accepted = jnp.isfinite(error)&(jnp.log(jax.random.uniform(accept_key)) < -error)
            q = jnp.where(accepted,proposed_q,q)
            return (q,chain_key),(q,accepted,error)
        _,records = jax.lax.scan(transition,(start,chain_key),None,length=warmup+draws)
        return tuple(a[warmup:] for a in records)
    chains,acceptance,energy_error = jax.vmap(one_chain)(jax.random.split(key,len(starts)),starts)
    return {"chains":chains,"accepted":acceptance,"energy_error":energy_error}


def diagnose(chains):
    """Classical split R-hat, not rank-normalized R-hat or an ESS estimate."""
    values=np.asarray(chains)
    if values.ndim!=3 or values.shape[0]<2 or values.shape[1]<4 or not np.isfinite(values).all():
        raise ValueError("diagnostics require finite (chains, draws, parameters) with multiple chains")
    half=values.shape[1]//2
    split=np.concatenate([values[:,:half],values[:,-half:]],axis=0)
    within=split.var(axis=1,ddof=1).mean(axis=0)
    if np.any(within<=0):
        raise ValueError("zero within-chain variation; R-hat is not meaningful")
    between=half*split.mean(axis=1).var(axis=0,ddof=1)
    rhat=np.sqrt(((half-1)*within/half+between/half)/within)
    return {"mean":values.mean(axis=(0,1)),"classical_split_rhat":rhat}


def predict(X, mean, covariance, noise_scale=1.):
    """Exact conditional Gaussian mean, latent variance and observation variance."""
    X,mean,covariance=map(jnp.asarray,(X,mean,covariance))
    if X.ndim!=2 or mean.ndim!=1 or X.shape[1]!=mean.size or covariance.shape!=(mean.size,mean.size):
        raise ValueError("prediction shapes disagree")
    if not np.isfinite(noise_scale) or noise_scale<=0:
        raise ValueError("noise scale must be positive and finite")
    if any(not np.isfinite(np.asarray(value)).all() for value in (X,mean,covariance)):
        raise ValueError("prediction inputs must be finite")
    host_cov=np.asarray(covariance)
    if not np.allclose(host_cov,host_cov.T,atol=1e-6) or np.linalg.eigvalsh(host_cov).min() < -1e-6:
        raise ValueError("predictive covariance must be symmetric positive semidefinite")
    latent=jnp.einsum("ni,ij,nj->n",X,covariance,X)
    return {"mean":X@mean,"latent_variance":latent,"observation_variance":latent+noise_scale**2}
