"""Optional NumPyro extension scaffold. Install the project-local requirements first."""


def regression_model(X, y=None, *, noise_scale=0.35, prior_scale=1.5):
    """Declare a vector-weight Gaussian regression model in NumPyro."""
    raise NotImplementedError("TODO: express the prior and observation model")


def run_nuts(
    X,
    y,
    *,
    seed=2026,
    noise_scale=0.35,
    prior_scale=1.5,
    warmup=400,
    draws=600,
    chains=4
):
    """Run independent NUTS chains and return the NumPyro MCMC object."""
    raise NotImplementedError("TODO: configure and run MCMC with independent chains")


def modern_diagnostics(chains, divergences):
    """Return rank-normalized split R-hat, bulk/tail ESS and divergence count."""
    raise NotImplementedError("TODO: compute and interpret multi-chain diagnostics")


def exact_posterior_numpy(X, y, *, noise_scale, prior_scale):
    """Return the independent NumPy posterior mean and covariance oracle."""
    raise NotImplementedError("TODO: solve the Gaussian precision system on the host")


def audit_gaussian_case(
    X, y, *, seed, noise_scale=0.35, prior_scale=1.5, warmup=400, draws=600, chains=4
):
    """Compare sample moments with the exact posterior using Monte Carlo error."""
    raise NotImplementedError("TODO: compare chains and the independent oracle")


def posterior_predictive_discrepancy(
    X, y, *, seed, prior_scale=1.5, noise_scale=0.25, warmup=300, draws=500, chains=4
):
    """Compare a linear mean with a quadratic remedy using replicated data."""
    raise NotImplementedError(
        "TODO: simulate and evaluate the preregistered discrepancy"
    )


def run_reference_workflow(output_dir, *, seed=2026, warmup=400, draws=600, chains=4):
    """Write inspectable diagnostic/PPC data and figures into output_dir."""
    raise NotImplementedError("TODO: compose checks and save run evidence")
