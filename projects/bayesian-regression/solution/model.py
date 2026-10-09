"""CPU teaching implementation: Gaussian regression, fixed-trajectory HMC, prediction."""
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np


# Function `fit(X, y, noise_scale, prior_scale)` implementing this stage's computation:
def fit(X, y, noise_scale=1., prior_scale=2.):
    """Zero-mean isotropic Gaussian prior; known positive observation-noise scale."""
    # Create device-backed JAX array `(X, y)`.
    X, y = jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)
    # Guard input contract (`X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0] or (X.shape[1] == 0)`) and fail fast if violated.
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0] or X.shape[1] == 0:
        raise ValueError("X must be (observations, features), y must be (observations,)")
    # Guard input contract (`not np.isfinite(np.asarray(X)).all() or not np.isfinite(np.asarray(y)).all()`) and fail fast if violated.
    if not np.isfinite(np.asarray(X)).all() or not np.isfinite(np.asarray(y)).all():
        raise ValueError("observations must be finite")
    # Guard input contract (`not np.isfinite([noise_scale, prior_scale]).all() or min(noise_scale, prior_scale) <= 0`) and fail fast if violated.
    if not np.isfinite([noise_scale, prior_scale]).all() or min(noise_scale, prior_scale) <= 0:
        raise ValueError("scales must be positive and finite")
    # Initialize array `precision` with explicit values and shape.
    precision = jnp.eye(X.shape[1])/prior_scale**2 + X.T@X/noise_scale**2
    # Perform matrix contraction / projection to compute `mean`.
    mean = jnp.linalg.solve(precision, X.T@y/noise_scale**2)
    # Construct an identity matrix `covariance`.
    covariance = jnp.linalg.solve(precision, jnp.eye(X.shape[1]))
    # Return `(mean, covariance)` to the caller.
    return mean, covariance


# Function `log_joint(weights, X, y, noise_scale, ...)` implementing this stage's computation:
def log_joint(weights, X, y, noise_scale=1., prior_scale=2.):
    """Normalized prior and independent Gaussian likelihood; weights is one vector."""
    # Perform matrix contraction / projection to compute `residual`.
    residual = y-X@weights
    # Compute `n, d` as `X.shape`.
    n, d = X.shape
    # Return `-0.5 * jnp.sum(residual ** 2) / noise_scale ** 2 - n * jnp.log(noise_scale) - 0.5 * jnp.sum(weights ** 2) / prior_scale ** 2 - d * jnp.log(prior_scale) - 0.5 * (n + d) * jnp.log(2 * jnp.pi)` to the caller.
    return (-.5*jnp.sum(residual**2)/noise_scale**2-n*jnp.log(noise_scale)
            -.5*jnp.sum(weights**2)/prior_scale**2-d*jnp.log(prior_scale)
            -.5*(n+d)*jnp.log(2*jnp.pi))


# Define `sample(key, mean, covariance, starts...)` to evaluate the objective and its automatic derivatives:
def sample(key, mean, covariance, starts, step_size=.15, leapfrog_steps=9,
           warmup=300, draws=1800):
    """Return retained chains, acceptance flags and signed energy errors.
    Fixed warmup discard
    no adaptive step size, NUTS or mass-matrix adaptation.
    """
    # Create device-backed JAX array `(mean, covariance, starts)`.
    mean, covariance, starts = map(jnp.asarray, (mean, covariance, starts))
    # Compute `d` as `mean.size`.
    d = mean.size
    # Guard input contract (`mean.ndim != 1 or covariance.shape != (d, d) or starts.ndim != 2 or (starts.shape[1] != d)`) and fail fast if violated.
    if mean.ndim != 1 or covariance.shape != (d,d) or starts.ndim != 2 or starts.shape[1] != d:
        raise ValueError("mean, covariance and chain-start shapes disagree")
    # Guard input contract (`draws < 4 or warmup < 0 or leapfrog_steps < 1 or (not np.isfinite(step_size)) or (step_size <= 0)`) and fail fast if violated.
    if draws < 4 or warmup < 0 or leapfrog_steps < 1 or not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("invalid sampler configuration")
    # Convert `host_cov` to a host NumPy array for inspection or verification.
    host_cov = np.asarray(covariance)
    # Guard input contract (`not np.isfinite(host_cov).all() or not np.allclose(host_cov, host_cov.T, atol=1e-06) or np.linalg.eigvalsh(host_cov).min() <= 0`) and fail fast if violated.
    if (not np.isfinite(host_cov).all()
        or not np.allclose(host_cov, host_cov.T, atol=1e-6)
        or np.linalg.eigvalsh(host_cov).min() <= 0):
        raise ValueError("covariance must be finite, symmetric and positive definite")
    # Guard input contract (`not np.isfinite(np.asarray(mean)).all() or not np.isfinite(np.asarray(starts)).all()`) and fail fast if violated.
    if not np.isfinite(np.asarray(mean)).all() or not np.isfinite(np.asarray(starts)).all():
        raise ValueError("mean and starts must be finite")
    # Initialize array `precision` with explicit values and shape.
    precision = jnp.linalg.solve(covariance, jnp.eye(d))
    # Function `energy(q)` implementing this stage's computation:
    def energy(q):
        # Compute `delta` as `q-mean`.
        delta = q-mean
        # Return `0.5 * delta @ precision @ delta` to the caller.
        return .5*delta@precision@delta
    # Differentiate the objective to obtain gradients `force`.
    force = jax.grad(energy)

    # Function `one_chain(chain_key, start)` implementing this stage's computation:
    def one_chain(chain_key, start):
        # Function `transition(state, _)` implementing this stage's computation:
        def transition(state, _):
            # Compute `q, chain_key` as `state`.
            q, chain_key = state
            # Split the PRNG key deterministically into independent subkeys (`(chain_key, momentum_key, accept_key)`).
            chain_key, momentum_key, accept_key = jax.random.split(chain_key,3)
            # Draw pseudorandom samples for `initial_p` using the explicit RNG state.
            initial_p = jax.random.normal(momentum_key,q.shape)
            # Compute `p` as `initial_p-.5*step_size*force(q)`.
            p = initial_p-.5*step_size*force(q)
            # Function `leap(i, state)` implementing this stage's computation:
            def leap(i,state):
                # Compute `position,momentum` as `state`.
                position,momentum = state
                # Compute `position` as `position+step_size*momentum`.
                position = position+step_size*momentum
                # Combine or mask array elements to form `momentum`.
                momentum = momentum-jnp.where(i<leapfrog_steps-1,step_size,.5*step_size)*force(position)
                # Return `(position, momentum)` to the caller.
                return position,momentum
            # Run `jax.lax.fori_loop` to compute `(proposed_q, proposed_p)`.
            proposed_q,proposed_p = jax.lax.fori_loop(0,leapfrog_steps,leap,(q,p))
            # Reduce across the target axis to summarize `error`.
            error = energy(proposed_q)+.5*jnp.sum(proposed_p**2)-energy(q)-.5*jnp.sum(initial_p**2)
            # Draw pseudorandom samples for `accepted` using the explicit RNG state.
            accepted = jnp.isfinite(error)&(jnp.log(jax.random.uniform(accept_key)) < -error)
            # Combine or mask array elements to form `q`.
            q = jnp.where(accepted,proposed_q,q)
            # Return `((q, chain_key), (q, accepted, error))` to the caller.
            return (q,chain_key),(q,accepted,error)
        # Run a compiled sequential scan over the time/step axis (`(_, records)`).
        _,records = jax.lax.scan(transition,(start,chain_key),None,length=warmup+draws)
        # Return `tuple((a[warmup:] for a in records))` to the caller.
        return tuple(a[warmup:] for a in records)
    # Split the PRNG key deterministically into independent subkeys (`(chains, acceptance, energy_error)`).
    chains,acceptance,energy_error = jax.vmap(one_chain)(jax.random.split(key,len(starts)),starts)
    # Return `{'chains': chains, 'accepted': acceptance, 'energy_error': energy_error}` to the caller.
    return {"chains":chains,"accepted":acceptance,"energy_error":energy_error}


# Function `diagnose(chains)` implementing this stage's computation:
def diagnose(chains):
    """Classical split R-hat, not rank-normalized R-hat or an ESS estimate."""
    # Convert `values` to a host NumPy array for inspection or verification.
    values=np.asarray(chains)
    # Guard input contract (`values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 4 or (not np.isfinite(values).all())`) and fail fast if violated.
    if values.ndim!=3 or values.shape[0]<2 or values.shape[1]<4 or not np.isfinite(values).all():
        raise ValueError("diagnostics require finite (chains, draws, parameters) with multiple chains")
    # Compute `half` as `values.shape[1]//2`.
    half=values.shape[1]//2
    # Combine or mask array elements to form `split`.
    split=np.concatenate([values[:,:half],values[:,-half:]],axis=0)
    # Reduce across the target axis to summarize `within`.
    within=split.var(axis=1,ddof=1).mean(axis=0)
    # Guard input contract (`np.any(within <= 0)`) and fail fast if violated.
    if np.any(within<=0):
        raise ValueError("zero within-chain variation; R-hat is not meaningful")
    # Reduce across the target axis to summarize `between`.
    between=half*split.mean(axis=1).var(axis=0,ddof=1)
    # Run `np.sqrt` to compute `rhat`.
    rhat=np.sqrt(((half-1)*within/half+between/half)/within)
    # Return `{'mean': values.mean(axis=(0, 1)), 'classical_split_rhat': rhat}` to the caller.
    return {"mean":values.mean(axis=(0,1)),"classical_split_rhat":rhat}


# Function `predict(X, mean, covariance, noise_scale)` implementing this stage's computation:
def predict(X, mean, covariance, noise_scale=1.):
    """Exact conditional Gaussian mean, latent variance and observation variance."""
    # Create device-backed JAX array `(X, mean, covariance)`.
    X,mean,covariance=map(jnp.asarray,(X,mean,covariance))
    # Guard input contract (`X.ndim != 2 or mean.ndim != 1 or X.shape[1] != mean.size or (covariance.shape != (mean.size, mean.size))`) and fail fast if violated.
    if X.ndim!=2 or mean.ndim!=1 or X.shape[1]!=mean.size or covariance.shape!=(mean.size,mean.size):
        raise ValueError("prediction shapes disagree")
    # Guard input contract (`not np.isfinite(noise_scale) or noise_scale <= 0`) and fail fast if violated.
    if not np.isfinite(noise_scale) or noise_scale<=0:
        raise ValueError("noise scale must be positive and finite")
    # Guard input contract (`any((not np.isfinite(np.asarray(value)).all() for value in (X, mean, covariance)))`) and fail fast if violated.
    if any(not np.isfinite(np.asarray(value)).all() for value in (X,mean,covariance)):
        raise ValueError("prediction inputs must be finite")
    # Convert `host_cov` to a host NumPy array for inspection or verification.
    host_cov=np.asarray(covariance)
    # Guard input contract (`not np.allclose(host_cov, host_cov.T, atol=1e-06) or np.linalg.eigvalsh(host_cov).min() < -1e-06`) and fail fast if violated.
    if not np.allclose(host_cov,host_cov.T,atol=1e-6) or np.linalg.eigvalsh(host_cov).min() < -1e-6:
        raise ValueError("predictive covariance must be symmetric positive semidefinite")
    # Perform matrix contraction / projection to compute `latent`.
    latent=jnp.einsum("ni,ij,nj->n",X,covariance,X)
    # Return `{'mean': X @ mean, 'latent_variance': latent, 'observation_variance': latent + noise_scale ** 2}` to the caller.
    return {"mean":X@mean,"latent_variance":latent,"observation_variance":latent+noise_scale**2}
