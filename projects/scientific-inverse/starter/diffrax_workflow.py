"""Optional Diffrax inverse-problem workflow starter; install requirements-science.txt first."""


def solve_forced_decay(
    rate,
    initial,
    times,
    *,
    solver="tsit5",
    rtol=1e-8,
    atol=1e-10,
    max_steps=100_000,
):
    """Return saved values and explicit Diffrax status/step counts."""
    raise NotImplementedError("TODO: validate the solve contract and call Diffrax")


def log_rate_sensitivity(
    log_rate,
    initial,
    times,
    *,
    solver="tsit5",
    rtol=1e-8,
    atol=1e-10,
    max_steps=100_000,
):
    """Differentiate saved solution values with respect to log(rate)."""
    raise NotImplementedError("TODO: differentiate the numerical solve in reverse mode")


def fit_rate(
    times,
    observations,
    initial,
    *,
    initial_rate=0.5,
    noise_scale=0.025,
    solver="tsit5",
    rtol=1e-8,
    atol=1e-10,
    max_steps=100_000,
    updates=300,
    learning_rate=0.03,
):
    """Fit a positive rate from fixed-noise observations; return a scalar rate."""
    raise NotImplementedError("TODO: minimize the known-noise Gaussian objective")
