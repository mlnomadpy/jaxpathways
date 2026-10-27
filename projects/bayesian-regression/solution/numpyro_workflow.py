"""Optional CPU NumPyro workflow for the Bayesian regression project."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import ndtri
from numpyro.diagnostics import effective_sample_size
from numpyro.infer import MCMC, NUTS, Predictive


def regression_model(X, y=None, *, noise_scale=0.35, prior_scale=1.5):
    """Normal linear model with a known observation scale and isotropic prior."""
    weights = numpyro.sample(
        "weights",
        dist.Normal(jnp.zeros(X.shape[1]), prior_scale).to_event(1),
    )
    numpyro.sample("obs", dist.Normal(X @ weights, noise_scale).to_event(1), obs=y)


def run_nuts(
    X,
    y,
    *,
    seed=2026,
    noise_scale=0.35,
    prior_scale=1.5,
    warmup=400,
    draws=600,
    chains=4,
):
    """Run independent NUTS chains sequentially, suitable for one CPU device."""
    X = jnp.asarray(X, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)
    if X.ndim != 2 or y.shape != (X.shape[0],) or X.shape[1] == 0:
        raise ValueError(
            "X and y must have shapes (observations, features) and (observations,)"
        )
    if not np.isfinite(np.asarray(X)).all() or not np.isfinite(np.asarray(y)).all():
        raise ValueError("X and y must be finite")
    if noise_scale <= 0 or prior_scale <= 0 or warmup < 0 or draws < 4 or chains < 2:
        raise ValueError("scales, chain count, warmup and draw count are invalid")
    sampler = MCMC(
        NUTS(regression_model),
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        chain_method="sequential",
        progress_bar=False,
    )
    sampler.run(
        jax.random.key(seed),
        X,
        y,
        noise_scale=noise_scale,
        prior_scale=prior_scale,
        extra_fields=("diverging",),
    )
    return sampler


def _split(draws):
    half = draws.shape[1] // 2
    return np.concatenate((draws[:, :half], draws[:, -half:]), axis=0)


def _rank_normalize(values):
    """Blom rank-normal scores; average ranks preserve ties."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    sorted_values = flat[order]
    ranks = np.empty(flat.size, dtype=np.float64)
    start = 0
    while start < flat.size:
        stop = start + 1
        while stop < flat.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    probabilities = (ranks - 0.375) / (flat.size + 0.25)
    return np.asarray(ndtri(jnp.asarray(probabilities))).reshape(values.shape)


def _rhat(split_draws):
    chain_count, draw_count, parameter_count = split_draws.shape
    chain_means = split_draws.mean(axis=1)
    within = split_draws.var(axis=1, ddof=1).mean(axis=0)
    between = draw_count * chain_means.var(axis=0, ddof=1)
    variance = (draw_count - 1) / draw_count * within + between / draw_count
    if np.any(within <= 0):
        raise ValueError("R-hat is undefined for a constant parameter")
    return np.sqrt(variance / within)


def modern_diagnostics(chains, divergences):
    """Compute rank-normalized split R-hat and rank bulk / indicator tail ESS."""
    chains = np.asarray(chains, dtype=np.float64)
    divergences = np.asarray(divergences, dtype=bool)
    if chains.ndim != 3 or chains.shape[0] < 2 or chains.shape[1] < 8:
        raise ValueError(
            "diagnostics require (chains, draws, parameters), multiple chains and 8+ draws"
        )
    if not np.isfinite(chains).all() or divergences.shape != chains.shape[:2]:
        raise ValueError(
            "diagnostic inputs must be finite and match chain/draw dimensions"
        )

    split = _split(chains)
    rank_values = np.stack(
        [_rank_normalize(split[:, :, i]) for i in range(split.shape[2])], axis=-1
    )
    rank_rhat = _rhat(rank_values)
    pooled_median = np.median(split, axis=(0, 1), keepdims=True)
    folded_values = np.abs(split - pooled_median)
    folded_ranks = np.stack(
        [_rank_normalize(folded_values[:, :, i]) for i in range(split.shape[2])],
        axis=-1,
    )
    folded_rhat = _rhat(folded_ranks)
    rhat = np.maximum(rank_rhat, folded_rhat)

    bulk_ess = np.asarray(effective_sample_size(jnp.asarray(rank_values)))
    lower = np.quantile(split, 0.05, axis=(0, 1), keepdims=True)
    upper = np.quantile(split, 0.95, axis=(0, 1), keepdims=True)
    lower_ess = np.asarray(
        effective_sample_size(jnp.asarray((split <= lower).astype(np.float32)))
    )
    upper_ess = np.asarray(
        effective_sample_size(jnp.asarray((split >= upper).astype(np.float32)))
    )
    tail_ess = np.minimum(lower_ess, upper_ess)
    return {
        "rank_normalized_rhat": rhat.tolist(),
        "bulk_ess": bulk_ess.tolist(),
        "tail_ess": tail_ess.tolist(),
        "divergences": int(divergences.sum()),
        "chains": int(chains.shape[0]),
        "draws_per_chain": int(chains.shape[1]),
    }


def exact_posterior_numpy(X, y, *, noise_scale, prior_scale):
    """Independent host NumPy oracle using a precision solve."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    precision = np.eye(X.shape[1]) / prior_scale**2 + X.T @ X / noise_scale**2
    covariance = np.linalg.solve(precision, np.eye(X.shape[1]))
    mean = np.linalg.solve(precision, X.T @ y / noise_scale**2)
    return mean, covariance


def _batch_means_mcse(values):
    """Estimate standard error from nonoverlapping batches within each chain.

    The final incomplete batch is discarded. At least four batches per chain
    are required so this remains a sampling-based uncertainty estimate rather
    than an IID ESS approximation.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or not np.isfinite(values).all():
        raise ValueError("batch-means input must be finite with chain and draw axes")
    chain_count, draw_count = values.shape[:2]
    batch_size = max(20, draw_count // 10)
    batch_count = draw_count // batch_size
    if chain_count < 2 or batch_count < 4:
        raise ValueError("batch-means MCSE needs two chains and four batches per chain")
    grouped = values[:, : batch_count * batch_size].reshape(
        chain_count, batch_count, batch_size, *values.shape[2:]
    )
    batch_averages = grouped.mean(axis=2).reshape(
        chain_count * batch_count, *values.shape[2:]
    )
    return batch_averages.std(axis=0, ddof=1) / np.sqrt(chain_count * batch_count)


def audit_gaussian_case(
    X, y, *, seed, noise_scale=0.35, prior_scale=1.5, warmup=400, draws=600, chains=4
):
    """Sample a known conjugate target and compare moments on MCSE scale."""
    sampler = run_nuts(
        X,
        y,
        seed=seed,
        noise_scale=noise_scale,
        prior_scale=prior_scale,
        warmup=warmup,
        draws=draws,
        chains=chains,
    )
    chains_array = np.asarray(sampler.get_samples(group_by_chain=True)["weights"])
    divergence = np.asarray(sampler.get_extra_fields(group_by_chain=True)["diverging"])
    diagnostics = modern_diagnostics(chains_array, divergence)
    exact_mean, exact_covariance = exact_posterior_numpy(
        X, y, noise_scale=noise_scale, prior_scale=prior_scale
    )
    flat = chains_array.reshape(-1, chains_array.shape[-1])
    sample_mean = flat.mean(axis=0)
    sample_covariance = np.cov(flat, rowvar=False, ddof=1)
    mean_mcse = _batch_means_mcse(chains_array)
    mean_z = np.abs(sample_mean - exact_mean) / np.maximum(mean_mcse, 1e-12)
    covariance_mcse = np.empty_like(exact_covariance)
    cov_z = np.empty_like(exact_covariance)
    for i in range(exact_covariance.shape[0]):
        for j in range(exact_covariance.shape[1]):
            product_series = (chains_array[:, :, i] - sample_mean[i]) * (
                chains_array[:, :, j] - sample_mean[j]
            )
            covariance_mcse[i, j] = float(_batch_means_mcse(product_series))
            cov_z[i, j] = abs(sample_covariance[i, j] - exact_covariance[i, j]) / max(
                covariance_mcse[i, j], 1e-12
            )
    return {
        "chains": chains_array,
        "diverging": divergence,
        "diagnostics": diagnostics,
        "exact_mean": exact_mean,
        "exact_covariance": exact_covariance,
        "sample_mean": sample_mean,
        "sample_covariance": sample_covariance,
        "mean_mcse": mean_mcse,
        "covariance_product_batch_mcse": covariance_mcse,
        "mcse_method": (
            "nonoverlapping batch means of raw parameter and centered product series"
        ),
        "mean_mcse_z": mean_z,
        "covariance_mcse_z": cov_z,
    }


def posterior_predictive_discrepancy(
    X, y, *, seed, prior_scale=1.5, noise_scale=0.25, warmup=300, draws=500, chains=4
):
    """Check residual quadratic structure under a linear and quadratic mean."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x = X[:, 1]
    q = x**2 - np.mean(x**2)
    design_by_order = {
        "linear": X,
        "quadratic": np.column_stack((X, q)).astype(np.float32),
    }
    results = {}
    for offset, (name, design) in enumerate(design_by_order.items()):
        sampler = run_nuts(
            design,
            y,
            seed=seed + offset,
            noise_scale=noise_scale,
            prior_scale=prior_scale,
            warmup=warmup,
            draws=draws,
            chains=chains,
        )
        posterior = sampler.get_samples(group_by_chain=False)["weights"]
        posterior_mean = np.asarray(posterior).mean(axis=0)
        observed_residual = y - design @ posterior_mean
        observed_statistic = float(q @ observed_residual / (q @ q))

        predictive = Predictive(
            regression_model, posterior_samples={"weights": posterior}
        )
        replicated = np.asarray(
            predictive(
                jax.random.key(seed + 100 + offset),
                jnp.asarray(design),
                None,
                noise_scale=noise_scale,
                prior_scale=prior_scale,
            )["obs"]
        )
        replicated_mean = np.asarray(posterior) @ design.T
        replicated_residual = replicated - replicated_mean
        replicated_statistic = replicated_residual @ q / (q @ q)
        lower, upper = np.quantile(replicated_statistic, [0.05, 0.95])
        tail_probability = float(
            np.mean(np.abs(replicated_statistic) >= abs(observed_statistic))
        )
        results[name] = {
            "observed_quadratic_residual": observed_statistic,
            "replicated_quadratic_residuals": replicated_statistic,
            "replicated_90_interval": [float(lower), float(upper)],
            "tail_probability": tail_probability,
            "replications": int(replicated_statistic.size),
        }
    return results


def _save_diagnostic_figure(path, diagnostics):
    labels = ["intercept", "slope"]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    axes[0].bar(x, diagnostics["rank_normalized_rhat"], color="#5865f2")
    axes[0].axhline(1.01, color="#bd493b", linestyle="--", label="1.01 guide")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("rank-normalized split R-hat")
    axes[0].set_title("Between-chain agreement")
    axes[0].legend(frameon=False)
    width = 0.36
    axes[1].bar(
        x - width / 2, diagnostics["bulk_ess"], width, label="bulk ESS", color="#1f9b8f"
    )
    axes[1].bar(
        x + width / 2, diagnostics["tail_ess"], width, label="tail ESS", color="#e4a340"
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("effective draws (ESS)")
    axes[1].set_title("Information in the chains")
    axes[1].legend(frameon=False)
    fig.suptitle(
        f"NumPyro NUTS · {diagnostics['chains']} chains × {diagnostics['draws_per_chain']} draws · {diagnostics['divergences']} divergences"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_ppc_figure(path, ppc):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    plotted = {}
    for axis, name, label, color in zip(
        axes,
        ("linear", "quadratic"),
        ("linear model", "quadratic remedy"),
        ("#bd493b", "#1f9b8f"),
    ):
        result = ppc[name]
        values = result["replicated_quadratic_residuals"]
        counts, edges, _ = axis.hist(
            values, bins=28, color="#aab4c5", edgecolor="white"
        )
        plotted[name] = {"counts": counts.astype(int).tolist(), "edges": edges.tolist()}
        axis.axvline(
            result["observed_quadratic_residual"],
            color=color,
            linewidth=2.2,
            label=f"observed = {result['observed_quadratic_residual']:.3f}",
        )
        axis.set_xlabel("quadratic residual coefficient (y / x²)")
        axis.set_ylabel("posterior predictive replications")
        axis.set_title(label)
        axis.legend(frameon=False)
    fig.suptitle("Posterior predictive check: residual curvature")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return plotted


def run_reference_workflow(output_dir, *, seed=2026, warmup=400, draws=600, chains=4):
    """Run exact-posterior validation, transfer fixture, and misspecified PPC."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.5, 1.5, 36, dtype=np.float32)
    X = np.column_stack((np.ones_like(x), x))
    truth = np.array([0.8, -0.55])
    noise_scale, prior_scale = 0.35, 1.5
    y = X @ truth + rng.normal(0, noise_scale, size=x.size).astype(np.float32)
    oracle = audit_gaussian_case(
        X,
        y,
        seed=seed,
        noise_scale=noise_scale,
        prior_scale=prior_scale,
        warmup=warmup,
        draws=draws,
        chains=chains,
    )

    x_transfer = np.linspace(-1.2, 1.2, 29, dtype=np.float32)
    X_transfer = np.column_stack(
        (np.ones_like(x_transfer), x_transfer, x_transfer**2 - 0.48)
    )
    y_transfer = X_transfer @ np.array([0.4, 0.3, -0.2]) + np.random.default_rng(
        seed + 1
    ).normal(0, noise_scale, size=x_transfer.size).astype(np.float32)
    transfer = audit_gaussian_case(
        X_transfer,
        y_transfer,
        seed=seed + 17,
        noise_scale=noise_scale,
        prior_scale=prior_scale,
        warmup=warmup,
        draws=draws,
        chains=chains,
    )

    x_ppc = np.linspace(-2, 2, 41, dtype=np.float32)
    X_ppc = np.column_stack((np.ones_like(x_ppc), x_ppc))
    centered_square = x_ppc**2 - np.mean(x_ppc**2)
    y_ppc = (
        1.0
        + 0.65 * x_ppc
        + 0.85 * centered_square
        + np.random.default_rng(seed + 2).normal(0, 0.25, size=x_ppc.size)
    ).astype(np.float32)
    ppc = posterior_predictive_discrepancy(
        X_ppc,
        y_ppc,
        seed=seed + 31,
        prior_scale=prior_scale,
        noise_scale=0.25,
        warmup=max(200, warmup // 2),
        draws=max(300, draws // 2),
        chains=chains,
    )

    serializable = {
        "environment": {
            "jax": jax.__version__,
            "numpy": np.__version__,
            "numpyro": numpyro.__version__,
            "platform": jax.default_backend(),
        },
        "configuration": {
            "seed": seed,
            "warmup": warmup,
            "draws": draws,
            "chains": chains,
        },
        "exact_case": {
            "shape": list(X.shape),
            "exact_mean": oracle["exact_mean"].tolist(),
            "sample_mean": oracle["sample_mean"].tolist(),
            "exact_covariance": oracle["exact_covariance"].tolist(),
            "sample_covariance": oracle["sample_covariance"].tolist(),
            "mean_mcse_z": oracle["mean_mcse_z"].tolist(),
            "covariance_mcse_z": oracle["covariance_mcse_z"].tolist(),
            "mean_mcse": oracle["mean_mcse"].tolist(),
            "covariance_product_batch_mcse": oracle[
                "covariance_product_batch_mcse"
            ].tolist(),
            "mcse_method": oracle["mcse_method"],
            "diagnostics": oracle["diagnostics"],
        },
        "changed_shape_case": {
            "shape": list(X_transfer.shape),
            "exact_mean": transfer["exact_mean"].tolist(),
            "sample_mean": transfer["sample_mean"].tolist(),
            "exact_covariance": transfer["exact_covariance"].tolist(),
            "sample_covariance": transfer["sample_covariance"].tolist(),
            "mean_mcse_z": transfer["mean_mcse_z"].tolist(),
            "covariance_mcse_z": transfer["covariance_mcse_z"].tolist(),
            "mean_mcse": transfer["mean_mcse"].tolist(),
            "covariance_product_batch_mcse": transfer[
                "covariance_product_batch_mcse"
            ].tolist(),
            "mcse_method": transfer["mcse_method"],
            "diagnostics": transfer["diagnostics"],
            "changed_seed": seed + 17,
        },
        "posterior_predictive": {
            name: {
                key: value
                for key, value in result.items()
                if key != "replicated_quadratic_residuals"
            }
            for name, result in ppc.items()
        },
    }
    (output_dir / "numpyro-diagnostics.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )
    figure_data = {
        "title": "NumPyro NUTS diagnostics against an exact Gaussian posterior",
        "x": ["intercept", "slope"],
        "xlabel": "regression parameter",
        "y_left": "rank-normalized split R-hat (unitless)",
        "y_right": "effective sample size (draws)",
        "diagnostics": oracle["diagnostics"],
        "scope": "Measured CPU execution for the pinned NumPyro environment; ESS is an estimated information count, not independent stored draws.",
    }
    (output_dir / "numpyro-diagnostics-figure.json").write_text(
        json.dumps(figure_data, indent=2) + "\n", encoding="utf-8"
    )
    ppc_histograms = _save_ppc_figure(output_dir / "numpyro-ppc.svg", ppc)
    ppc_figure_data = {
        "title": "Posterior predictive discrepancy under a misspecified mean",
        "x": "quadratic residual coefficient (y / x²)",
        "y": "posterior predictive replications per bin",
        "panels": {
            name: {
                **{
                    key: value
                    for key, value in result.items()
                    if key != "replicated_quadratic_residuals"
                },
                "histogram": ppc_histograms[name],
            }
            for name, result in ppc.items()
        },
        "scope": "The curved data are a deliberately generated stress fixture with 41 fixed x locations and known Gaussian noise. This does not estimate real-data coverage.",
    }
    (output_dir / "numpyro-ppc-figure.json").write_text(
        json.dumps(ppc_figure_data, indent=2) + "\n", encoding="utf-8"
    )
    _save_diagnostic_figure(
        output_dir / "numpyro-diagnostics.svg", oracle["diagnostics"]
    )
    return serializable


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(
            Path(tempfile.gettempdir()) / "bayesian-regression-numpyro-reference"
        ),
        help="output directory (default: a directory under the system temporary path)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--draws", type=int, default=600)
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()
    result = run_reference_workflow(
        args.output,
        seed=args.seed,
        warmup=args.warmup,
        draws=args.draws,
        chains=args.chains,
    )
    print(json.dumps(result, indent=2))
