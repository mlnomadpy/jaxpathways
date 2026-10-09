"""Implement one stage at a time. Public tests are not a professional credential."""

def fit(X, y, noise_scale=1., prior_scale=2.):
    """Stage 1: validate arrays; return Gaussian posterior (mean, covariance).
    X: (observations, features), y: (observations,). Scales must be positive.
    Empty observations recover the isotropic zero-mean prior.
    """
    # Key APIs to use: `jnp.asarray`, `contract`, `or`, `be`, `np.isfinite`
    # Step 1: Create device-backed JAX array `(X, y)`.
    # Step 2: Guard input contract (`X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0] or (X.shape[1] == 0)`) and fail fast if violated.
    # Step 3: Guard input contract (`not np.isfinite(np.asarray(X)).all() or not np.isfinite(np.asarray(y)).all()`) and fail fast if violated.
    # Step 4: Guard input contract (`not np.isfinite([noise_scale, prior_scale]).all() or min(noise_scale, prior_scale) <= 0`) and fail fast if violated.
    # Step 5: Construct an identity matrix `precision`.
    # Step 6: Perform matrix contraction / projection to compute `mean`.
    raise NotImplementedError("Derive and solve the posterior precision system")


def log_joint(weights, X, y, noise_scale=1., prior_scale=2.):
    """Stage 1: scalar normalized Gaussian prior + independent log likelihood.
    Must support JAX differentiation with respect to the weights.
    """
    # Key APIs to use: `jnp.sum`, `jnp.log`, `return`
    # Step 1: Perform matrix contraction / projection to compute `residual`.
    # Step 2: Evaluate `(n, d)` from the current inputs and state.
    # Step 3: Return `-0.5 * jnp.sum(residual ** 2) / noise_scale ** 2 - n * jnp.log(noise_scale) - 0.5 * jnp.sum(weights ** 2) / prior_scale ** 2 - d * jnp.log(prior_scale) - 0.5 * (n + d) * jnp.log(2 * jnp.pi)` to the caller.
    raise NotImplementedError("Keep scale-dependent normalization terms")


def sample(key, mean, covariance, starts, step_size=.15, leapfrog_steps=9,
           warmup=300, draws=1800):
    """Stage 2: HMC with a fixed identity mass matrix, independent chain keys.
    Return dict: chains (chains, draws, parameters), accepted (chains, draws),
    energy_error (chains, draws). Discard warmup before returning.
    Keep rejected positions. Reject nonfinite energy errors.
    """
    # Key APIs to use: `map`, `contract`, `or`, `np.isfinite`, `np.asarray`
    # Step 1: Create device-backed JAX array `(mean, covariance, starts)`.
    # Step 2: Evaluate `d` from the current inputs and state.
    # Step 3: Guard input contract (`mean.ndim != 1 or covariance.shape != (d, d) or starts.ndim != 2 or (starts.shape[1] != d)`) and fail fast if violated.
    # Step 4: Guard input contract (`draws < 4 or warmup < 0 or leapfrog_steps < 1 or (not np.isfinite(step_size)) or (step_size <= 0)`) and fail fast if violated.
    # Step 5: Convert `host_cov` to a host NumPy array for inspection or verification.
    # Step 6: Guard input contract (`not np.isfinite(host_cov).all() or not np.allclose(host_cov, host_cov.T, atol=1e-06) or np.linalg.eigvalsh(host_cov).min() <= 0`) and fail fast if violated.
    raise NotImplementedError("Implement leapfrog, log acceptance and state ownership")


def diagnose(chains):
    """Stage 2: return mean and classical_split_rhat per parameter.
    Require multiple finite chains
    reject zero within-chain variance.
    This is classical split R-hat, not rank-normalized R-hat or ESS.
    """
    # Key APIs to use: `np.asarray`, `contract`, `or`, `np.isfinite`, `all`
    # Step 1: Convert `values` to a host NumPy array for inspection or verification.
    # Step 2: Guard input contract (`values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 4 or (not np.isfinite(values).all())`) and fail fast if violated.
    # Step 3: Evaluate `half` from the current inputs and state.
    # Step 4: Combine or mask array elements to form `split`.
    # Step 5: Reduce across the target axis to summarize `within`.
    # Step 6: Guard input contract (`np.any(within <= 0)`) and fail fast if violated.
    raise NotImplementedError("Split chains and compare between/within variance")


def predict(X, mean, covariance, noise_scale=1.):
    """Stage 3: return mean, latent_variance and observation_variance vectors.
    Validate shapes, finite inputs, positive noise scale and covariance.
    """
    # Key APIs to use: `map`, `contract`, `or`, `np.isfinite`, `any`
    # Step 1: Create device-backed JAX array `(X, mean, covariance)`.
    # Step 2: Guard input contract (`X.ndim != 2 or mean.ndim != 1 or X.shape[1] != mean.size or (covariance.shape != (mean.size, mean.size))`) and fail fast if violated.
    # Step 3: Guard input contract (`not np.isfinite(noise_scale) or noise_scale <= 0`) and fail fast if violated.
    # Step 4: Guard input contract (`any((not np.isfinite(np.asarray(value)).all() for value in (X, mean, covariance)))`) and fail fast if violated.
    # Step 5: Convert `host_cov` to a host NumPy array for inspection or verification.
    # Step 6: Guard input contract (`not np.allclose(host_cov, host_cov.T, atol=1e-06) or np.linalg.eigvalsh(host_cov).min() < -1e-06`) and fail fast if violated.
    raise NotImplementedError("Propagate weight covariance and add observation noise")
