"""Implement one stage at a time. Public tests are not a professional credential."""

def fit(X, y, noise_scale=1., prior_scale=2.):
    """Stage 1: validate arrays; return Gaussian posterior (mean, covariance).
    X: (observations, features), y: (observations,). Scales must be positive.
    Empty observations recover the isotropic zero-mean prior.
    """
    raise NotImplementedError("Derive and solve the posterior precision system")


def log_joint(weights, X, y, noise_scale=1., prior_scale=2.):
    """Stage 1: scalar normalized Gaussian prior + independent log likelihood.
    Must support JAX differentiation with respect to the weights.
    """
    raise NotImplementedError("Keep scale-dependent normalization terms")


def sample(key, mean, covariance, starts, step_size=.15, leapfrog_steps=9,
           warmup=300, draws=1800):
    """Stage 2: HMC with a fixed identity mass matrix, independent chain keys.
    Return dict: chains (chains, draws, parameters), accepted (chains, draws),
    energy_error (chains, draws). Discard warmup before returning.
    Keep rejected positions. Reject nonfinite energy errors.
    """
    raise NotImplementedError("Implement leapfrog, log acceptance and state ownership")


def diagnose(chains):
    """Stage 2: return mean and classical_split_rhat per parameter.
    Require multiple finite chains; reject zero within-chain variance.
    This is classical split R-hat, not rank-normalized R-hat or ESS.
    """
    raise NotImplementedError("Split chains and compare between/within variance")


def predict(X, mean, covariance, noise_scale=1.):
    """Stage 3: return mean, latent_variance and observation_variance vectors.
    Validate shapes, finite inputs, positive noise scale and covariance.
    """
    raise NotImplementedError("Propagate weight covariance and add observation noise")
