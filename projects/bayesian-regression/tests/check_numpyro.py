"""Independent optional stage 4 checks for the project-local NumPyro workflow."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.diagnostics import effective_sample_size
from numpyro.infer import MCMC, NUTS

CHAINS = 4
WARMUP = 250
DRAWS = 400
NOISE_SCALE = 0.35
PRIOR_SCALE = 1.5


def load_implementation(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("bayesian_numpyro_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(seed, rows, truth):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.5, 1.5, rows, dtype=np.float32)
    columns = [np.ones_like(x), x]
    if len(truth) >= 3:
        columns.append(x**2 - np.mean(x**2))
    if len(truth) >= 4:
        columns.append(np.sin(1.7 * x))
    X = np.column_stack(columns).astype(np.float32)
    y = X @ np.asarray(truth, dtype=np.float32) + rng.normal(
        0, NOISE_SCALE, size=rows
    ).astype(np.float32)
    return X, y


def _oracle_numpy(X, y, noise_scale, prior_scale):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    precision = np.eye(X.shape[1]) / prior_scale**2 + X.T @ X / noise_scale**2
    covariance = np.linalg.solve(precision, np.eye(X.shape[1]))
    mean = np.linalg.solve(precision, X.T @ y / noise_scale**2)
    return mean, covariance


def _split(chains):
    half = chains.shape[1] // 2
    return np.concatenate((chains[:, :half], chains[:, -half:]), axis=0)


def _rank_normalize(values):
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(flat).all():
        raise ValueError("rank normalization requires finite values")
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
    u = (ranks - 0.375) / (flat.size + 0.25)
    from jax.scipy.special import ndtri

    return np.asarray(ndtri(jnp.asarray(u))).reshape(values.shape)


def independent_diagnostics(chains):
    """Compute split rank/folded R-hat plus rank-bulk and indicator-tail ESS."""
    chains = np.asarray(chains, dtype=np.float64)
    if (
        chains.ndim != 3
        or chains.shape[0] < 2
        or chains.shape[1] < 8
        or chains.shape[2] < 1
    ):
        raise ValueError(
            "need finite (chains, draws, parameters), at least two chains and eight draws"
        )
    if not np.isfinite(chains).all():
        raise ValueError("chain values must be finite")
    split = _split(chains)
    draws = split.shape[1]

    def basic_rhat(values):
        within = values.var(axis=1, ddof=1).mean(axis=0)
        between = draws * values.mean(axis=1).var(axis=0, ddof=1)
        if np.any(within <= 0):
            raise ValueError("R-hat is undefined for a constant parameter")
        variance = (draws - 1) / draws * within + between / draws
        return np.sqrt(variance / within)

    ranked = np.stack(
        [_rank_normalize(split[:, :, i]) for i in range(split.shape[-1])], axis=-1
    )
    rank_rhat = basic_rhat(ranked)
    folded = np.abs(split - np.median(split, axis=(0, 1), keepdims=True))
    folded_ranked = np.stack(
        [_rank_normalize(folded[:, :, i]) for i in range(split.shape[-1])], axis=-1
    )
    rhat = np.maximum(rank_rhat, basic_rhat(folded_ranked))
    bulk = np.asarray(effective_sample_size(jnp.asarray(ranked)))
    lower = np.quantile(split, 0.05, axis=(0, 1), keepdims=True)
    upper = np.quantile(split, 0.95, axis=(0, 1), keepdims=True)
    lower_ess = np.asarray(
        effective_sample_size(jnp.asarray((split <= lower).astype(np.float32)))
    )
    upper_ess = np.asarray(
        effective_sample_size(jnp.asarray((split >= upper).astype(np.float32)))
    )
    tail = np.minimum(lower_ess, upper_ess)
    return {"rhat": rhat, "bulk_ess": bulk, "tail_ess": tail}


def batch_means_mcse(series):
    """Estimate mean MCSE from nonoverlapping within-chain batch means."""
    values = np.asarray(series, dtype=np.float64)
    if values.ndim < 2 or not np.isfinite(values).all():
        raise ValueError("batch-means input must be finite with chain and draw axes")
    chains, draws = values.shape[:2]
    batch = max(20, draws // 10)
    count = draws // batch
    if chains < 2 or count < 4:
        raise ValueError("batch-means MCSE needs at least four batches per chain")
    grouped = values[:, : count * batch].reshape(
        chains, count, batch, *values.shape[2:]
    )
    batch_means = grouped.mean(axis=2).reshape(chains * count, *values.shape[2:])
    return batch_means.std(axis=0, ddof=1) / np.sqrt(chains * count)


def _sample_raw(
    module,
    X,
    y,
    *,
    seed,
    draws=DRAWS,
    warmup=WARMUP,
    chains=CHAINS,
    noise_scale=NOISE_SCALE,
    prior_scale=PRIOR_SCALE,
):
    sampler = module.run_nuts(
        X,
        y,
        seed=seed,
        noise_scale=noise_scale,
        prior_scale=prior_scale,
        warmup=warmup,
        draws=draws,
        chains=chains,
    )
    if not isinstance(sampler, MCMC) or not isinstance(sampler.sampler, NUTS):
        raise AssertionError("run_nuts must return a completed NumPyro MCMC using NUTS")
    if (
        sampler._last_state is None
        or sampler.num_chains != chains
        or sampler.num_samples != draws
        or sampler.num_warmup != warmup
        or sampler.chain_method != "sequential"
    ):
        raise AssertionError(
            "MCMC did not complete the requested sequential multi-chain run"
        )
    if len(sampler._args) < 2 or not np.array_equal(np.asarray(sampler._args[0]), X):
        raise AssertionError(
            "the recorded NUTS input differs from the checker-owned design matrix"
        )
    if not np.array_equal(np.asarray(sampler._args[1]), y):
        raise AssertionError(
            "the recorded NUTS input differs from the checker-owned outcomes"
        )
    if (
        sampler._kwargs.get("noise_scale") != noise_scale
        or sampler._kwargs.get("prior_scale") != prior_scale
    ):
        raise AssertionError(
            "the sampler did not use the checker-owned likelihood/prior scales"
        )

    sample_map = sampler.get_samples(group_by_chain=True)
    if set(sample_map) != {"weights"}:
        raise AssertionError(f"unexpected latent sample sites: {sorted(sample_map)}")
    chain_array = np.asarray(sample_map["weights"], dtype=np.float64)
    divergence_map = sampler.get_extra_fields(group_by_chain=True)
    if "diverging" not in divergence_map:
        raise AssertionError("NUTS divergence indicators were not retained")
    diverging = np.asarray(divergence_map["diverging"], dtype=bool)
    expected = (chains, draws, X.shape[1])
    if chain_array.shape != expected or diverging.shape != expected[:2]:
        raise AssertionError(
            f"raw chain/divergence shapes are {chain_array.shape}/{diverging.shape}, expected {expected}/{expected[:2]}"
        )
    if not np.isfinite(chain_array).all():
        raise AssertionError("raw posterior chain contains NaN or infinity")
    return sampler, chain_array, diverging


def _case_result(chains, diverging, X, y, noise_scale, prior_scale):
    diagnostics = independent_diagnostics(chains)
    exact_mean, exact_covariance = _oracle_numpy(X, y, noise_scale, prior_scale)
    flat = chains.reshape(-1, chains.shape[-1])
    sample_mean = flat.mean(axis=0)
    sample_covariance = np.cov(flat, rowvar=False, ddof=1)
    mean_mcse = batch_means_mcse(chains)
    mean_z = np.abs(sample_mean - exact_mean) / np.maximum(mean_mcse, 1e-12)
    covariance_mcse = np.empty_like(exact_covariance)
    covariance_z = np.empty_like(exact_covariance)
    for i in range(chains.shape[-1]):
        for j in range(chains.shape[-1]):
            product = (chains[:, :, i] - sample_mean[i]) * (
                chains[:, :, j] - sample_mean[j]
            )
            covariance_mcse[i, j] = float(batch_means_mcse(product))
            covariance_z[i, j] = abs(
                sample_covariance[i, j] - exact_covariance[i, j]
            ) / max(covariance_mcse[i, j], 1e-12)
    return {
        "shape": list(X.shape),
        "exact_mean": exact_mean,
        "sample_mean": sample_mean,
        "exact_covariance": exact_covariance,
        "sample_covariance": sample_covariance,
        "mean_batch_mcse": mean_mcse,
        "covariance_product_batch_mcse": covariance_mcse,
        "mean_mcse_z": mean_z,
        "covariance_mcse_z": covariance_z,
        "diagnostics": diagnostics,
        "divergences": int(diverging.sum()),
        "chains": chains,
        "diverging": diverging,
    }


def _check_gaussian_case(module, name, X, y, *, seed):
    sampler, chains, diverging = _sample_raw(module, X, y, seed=seed)
    case = _case_result(chains, diverging, X, y, NOISE_SCALE, PRIOR_SCALE)
    if np.max(case["mean_mcse_z"]) > 6 or np.max(case["covariance_mcse_z"]) > 6:
        raise AssertionError(
            f"{name} posterior differs from exact NumPy oracle by >6 batch MCSE"
        )
    d = X.shape[1]
    diag = case["diagnostics"]
    if any(
        np.asarray(diag[key]).shape != (d,) for key in ("rhat", "bulk_ess", "tail_ess")
    ):
        raise AssertionError(
            f"{name} diagnostic dimensions do not match {d} parameters"
        )
    if (
        not all(np.isfinite(diag[key]).all() for key in diag)
        or np.max(diag["rhat"]) > 1.05
        or np.min(diag["bulk_ess"]) < 100
        or np.min(diag["tail_ess"]) < 50
    ):
        raise AssertionError(f"{name} chain diagnostics are inadequate: {diag}")
    if case["divergences"] != 0:
        raise AssertionError(f"{name} conjugate Gaussian run had NUTS divergences")
    return sampler, case


def _paired_ppc(module, X, y, *, seed, noise_scale=0.25, prior_scale=PRIOR_SCALE):
    """Run the model and calculate T(y, theta_s), T(yrep_s, theta_s) in pairs."""
    sampler, chains, diverging = _sample_raw(
        module, X, y, seed=seed, noise_scale=noise_scale, prior_scale=prior_scale
    )
    if diverging.sum() > 0:
        raise AssertionError("posterior-predictive fit had NUTS divergences")
    flat = chains.reshape(-1, chains.shape[-1])
    x = X[:, 1]
    q = x**2 - np.mean(x**2)
    design = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fitted = flat @ design.T
    # Both statistics use the same posterior draw theta_s and the same fixed discrepancy.
    observed = (y[None, :] - fitted) @ q / (q @ q)
    ppc_rng = np.random.default_rng(seed + 100_003)
    replicated = fitted + ppc_rng.normal(0, noise_scale, size=fitted.shape)
    replicated_statistic = (replicated - fitted) @ q / (q @ q)
    paired_tail_probability = float(
        np.mean(np.abs(replicated_statistic) >= np.abs(observed))
    )
    return sampler, {
        "chains": chains,
        "diverging": diverging,
        "observed_statistic_by_draw": observed,
        "replicated_statistic_by_draw": replicated_statistic,
        "observed_median": float(np.median(observed)),
        "observed_interval_90": np.quantile(observed, [0.05, 0.95]),
        "replicated_interval_90": np.quantile(replicated_statistic, [0.05, 0.95]),
        "paired_tail_probability": paired_tail_probability,
    }


def _diagnostic_controls():
    rng = np.random.default_rng(5001)
    iid = rng.normal(size=(4, 400, 2))
    base = independent_diagnostics(iid)
    shifted = iid.copy()
    shifted[0, :, 0] += 4.0
    unequal = iid.copy()
    unequal[0, :, 1] *= 4.0
    shift_rhat = independent_diagnostics(shifted)["rhat"]
    scale_rhat = independent_diagnostics(unequal)["rhat"]
    if shift_rhat[0] < 1.1 or scale_rhat[1] < 1.05:
        raise AssertionError(
            "rank/folded R-hat failed shifted-mean or unequal-scale control"
        )
    noise = rng.normal(size=(4, 600, 1))
    ar = np.empty_like(noise)
    ar[:, 0, :] = noise[:, 0, :]
    for i in range(1, noise.shape[1]):
        ar[:, i, :] = 0.97 * ar[:, i - 1, :] + np.sqrt(1 - 0.97**2) * noise[:, i, :]
    iid_ess = independent_diagnostics(rng.normal(size=ar.shape))["bulk_ess"][0]
    ar_ess = independent_diagnostics(ar)["bulk_ess"][0]
    if ar_ess > iid_ess * 0.55:
        raise AssertionError(
            f"bulk ESS did not flag strong autocorrelation: iid={iid_ess}, AR={ar_ess}"
        )
    for bad in (
        np.ones((4, 100, 2)),
        np.where(np.indices((4, 100, 2))[0] == 0, np.nan, 0.0),
    ):
        try:
            independent_diagnostics(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("diagnostics accepted constant or nonfinite input")
    return {
        "shifted_mean_rhat": shift_rhat.tolist(),
        "unequal_scale_rhat": scale_rhat.tolist(),
        "iid_bulk_ess": float(iid_ess),
        "autocorrelated_bulk_ess": float(ar_ess),
        "constant_and_nonfinite_rejected": True,
    }


def _histogram(values, bins=28):
    counts, edges = np.histogram(np.asarray(values), bins=bins)
    return {"counts": counts.tolist(), "edges": edges.tolist()}


def _render_svg(path, cases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    base_diag = cases["base"]["diagnostics"]
    labels = ["intercept", "slope"]
    x = np.arange(len(labels))
    axes[0].bar(x, base_diag["rhat"], color="#5865f2")
    axes[0].axhline(1.01, color="#bd493b", linestyle="--", label="1.01 guide")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("rank-normalized split R-hat")
    axes[0].set_title("Between-chain agreement")
    axes[0].legend(frameon=False)
    width = 0.36
    axes[1].bar(
        x - width / 2, base_diag["bulk_ess"], width, label="bulk ESS", color="#1f9b8f"
    )
    axes[1].bar(
        x + width / 2, base_diag["tail_ess"], width, label="tail ESS", color="#e4a340"
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("effective draws (ESS)")
    axes[1].set_title("Information in the chains")
    axes[1].legend(frameon=False)
    fig.savefig(path, format="svg")
    plt.close(fig)


def validate_evidence(evidence):
    """Reject malformed raw arrays, diagnostic dimensions, histograms, or SVGs."""
    cases = evidence["cases"]
    for name, case in cases.items():
        chains = np.asarray(case["chains"])
        divergence = np.asarray(case["diverging"])
        if chains.ndim != 3 or chains.shape[0] != CHAINS or chains.shape[1] != DRAWS:
            raise ValueError(f"{name}: wrong raw chain shape {chains.shape}")
        if not np.isfinite(chains).all() or divergence.shape != chains.shape[:2]:
            raise ValueError(f"{name}: invalid raw samples or divergence shape")
        d = chains.shape[-1]
        diagnostics = case["diagnostics"]
        for key in ("rhat", "bulk_ess", "tail_ess"):
            values = np.asarray(diagnostics[key])
            if values.shape != (d,) or not np.isfinite(values).all():
                raise ValueError(f"{name}: invalid {key} shape or values")
        if np.any(np.asarray(diagnostics["rhat"]) <= 0):
            raise ValueError(f"{name}: R-hat must be positive")
    for ppc_case in evidence["ppc"].values():
        for key in ("observed_statistic_by_draw", "replicated_statistic_by_draw"):
            values = np.asarray(ppc_case[key])
            if values.shape != (CHAINS * DRAWS,) or not np.isfinite(values).all():
                raise ValueError(
                    f"PPC {key} must retain one finite paired value per posterior draw"
                )
        counts = np.asarray(ppc_case["histogram"]["counts"])
        edges = np.asarray(ppc_case["histogram"]["edges"])
        if counts.ndim != 1 or edges.shape != (counts.size + 1,) or np.any(counts < 0):
            raise ValueError("PPC histogram has invalid counts or edges")
    for svg in evidence["svg"].values():
        root = ET.fromstring(svg)
        if root.tag.split("}")[-1] != "svg":
            raise ValueError("figure source is not an SVG document")
    return True


def _mutation_regressions():
    svg = '<svg xmlns="http://www.w3.org/2000/svg"/>'
    base = {
        "cases": {
            "base": {
                "chains": np.zeros((CHAINS, DRAWS, 2)),
                "diverging": np.zeros((CHAINS, DRAWS), dtype=bool),
                "diagnostics": {
                    "rhat": np.ones(2),
                    "bulk_ess": np.ones(2),
                    "tail_ess": np.ones(2),
                },
            }
        },
        "ppc": {
            "linear": {
                "observed_statistic_by_draw": np.zeros(CHAINS * DRAWS),
                "replicated_statistic_by_draw": np.zeros(CHAINS * DRAWS),
                "histogram": {"counts": [1, 2], "edges": [0, 1, 2]},
            }
        },
        "svg": {"diagnostic": svg, "ppc": svg},
    }
    validate_evidence(base)
    mutations = (
        lambda bad: bad["cases"]["base"]["diagnostics"].__setitem__("rhat", np.ones(1)),
        lambda bad: bad["svg"].__setitem__("ppc", "<svg><malformed"),
        lambda bad: bad["ppc"]["linear"]["histogram"].__setitem__("counts", [-1, 2]),
        lambda bad: bad["cases"]["base"].__setitem__("chains", np.zeros((3, DRAWS, 2))),
    )
    for mutate in mutations:
        bad = copy.deepcopy(base)
        mutate(bad)
        try:
            validate_evidence(bad)
        except (ValueError, ET.ParseError):
            continue
        raise AssertionError("evidence validator accepted a deliberate mutation")

    class AggregateOnly:
        def run_reference_workflow(self, *args, **kwargs):
            return {
                "exact_case": {},
                "changed_shape_case": {},
                "posterior_predictive": {},
            }

    try:
        _sample_raw(
            AggregateOnly(),
            np.ones((8, 2)),
            np.ones(8),
            seed=1,
            draws=DRAWS,
            warmup=WARMUP,
            chains=CHAINS,
        )
    except (AttributeError, AssertionError):
        aggregate_only_rejected = True
    else:
        raise AssertionError(
            "checker accepted an aggregate-only fake with no raw sampler"
        )
    return {
        "one_element_rhat": True,
        "invalid_svg": True,
        "negative_histogram": True,
        "wrong_chain_shape": True,
        "aggregate_only_fake": aggregate_only_rejected,
    }


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def verify(module, evidence_dir, implementation_path):
    """Checker owns fixtures, exact solves, diagnostics, PPC and saved evidence."""
    out = Path(evidence_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    X, y = _fixture(411, 36, [0.8, -0.55])
    sampler, base = _check_gaussian_case(module, "base", X, y, seed=411)
    X_changed, y_changed = _fixture(428, 29, [0.4, 0.3, -0.2])
    changed_sampler, changed = _check_gaussian_case(
        module, "changed", X_changed, y_changed, seed=428
    )
    if np.array_equal(base["chains"], changed["chains"][:, :, :2]):
        raise AssertionError(
            "changed seed/data/shape unexpectedly replayed the baseline chains"
        )

    ppc_x = np.linspace(-2, 2, 41, dtype=np.float32)
    ppc_X = np.column_stack((np.ones_like(ppc_x), ppc_x)).astype(np.float32)
    centered_square = ppc_x**2 - np.mean(ppc_x**2)
    ppc_y = (
        1
        + 0.65 * ppc_x
        + 0.85 * centered_square
        + np.random.default_rng(413).normal(0, 0.25, size=ppc_x.size)
    ).astype(np.float32)
    quad_X = np.column_stack((ppc_X, centered_square)).astype(np.float32)
    linear_sampler, linear = _paired_ppc(module, ppc_X, ppc_y, seed=442)
    quad_sampler, quadratic = _paired_ppc(module, quad_X, ppc_y, seed=443)
    if linear["paired_tail_probability"] > 0.02:
        raise AssertionError(
            "paired PPC did not expose the designed linear-mean mismatch"
        )
    if not 0.05 < quadratic["paired_tail_probability"] < 0.95:
        raise AssertionError("paired PPC did not support the quadratic remedy")

    controls = _diagnostic_controls()
    cases = {"base": base, "changed": changed}
    ppc = {"linear": linear, "quadratic": quadratic}
    for name, result in ppc.items():
        result["histogram"] = _histogram(result["replicated_statistic_by_draw"])

    diagnostic_svg = out / "numpyro-diagnostics.svg"
    ppc_svg = out / "numpyro-ppc.svg"
    _render_svg(diagnostic_svg, cases)
    ppc_figure = out / "numpyro-ppc-only.svg"
    _render_ppc_svg(ppc_figure, ppc)
    ppc_figure.replace(ppc_svg)
    if not diagnostic_svg.is_file() or not ppc_svg.is_file():
        raise AssertionError("checker did not save both SVG figures")

    evidence = {
        "cases": cases,
        "ppc": ppc,
        "svg": {"diagnostic": diagnostic_svg.read_text(), "ppc": ppc_svg.read_text()},
    }
    validate_evidence(evidence)
    mutation_results = _mutation_regressions()

    raw_file = out / "numpyro-raw-chains.npz"
    np.savez_compressed(
        raw_file,
        base_X=X,
        base_y=y,
        changed_X=X_changed,
        changed_y=y_changed,
        ppc_linear_X=ppc_X,
        ppc_y=ppc_y,
        ppc_quadratic_X=quad_X,
        base_chains=base["chains"],
        base_diverging=base["diverging"],
        changed_chains=changed["chains"],
        changed_diverging=changed["diverging"],
        ppc_linear_chains=linear["chains"],
        ppc_linear_diverging=linear["diverging"],
        ppc_quadratic_chains=quadratic["chains"],
        ppc_quadratic_diverging=quadratic["diverging"],
        ppc_linear_observed=linear["observed_statistic_by_draw"],
        ppc_linear_replicated=linear["replicated_statistic_by_draw"],
        ppc_quadratic_observed=quadratic["observed_statistic_by_draw"],
        ppc_quadratic_replicated=quadratic["replicated_statistic_by_draw"],
    )

    def case_summary(case):
        return {
            key: val for key, val in case.items() if key not in ("chains", "diverging")
        }

    def ppc_summary(result):
        return {
            key: val
            for key, val in result.items()
            if key
            not in (
                "chains",
                "diverging",
                "observed_statistic_by_draw",
                "replicated_statistic_by_draw",
            )
        }

    requirements = Path(__file__).resolve().parents[1] / "requirements-numpyro.txt"
    script = Path(implementation_path).resolve()
    receipt = {
        "environment": {
            "python": __import__("platform").python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "numpyro": numpyro.__version__,
            "backend": jax.default_backend(),
        },
        "configuration": {
            "chains": CHAINS,
            "warmup": WARMUP,
            "draws": DRAWS,
            "noise_scale": NOISE_SCALE,
            "prior_scale": PRIOR_SCALE,
            "chain_method": "sequential",
        },
        "seeds": {"base": 411, "changed": 428, "ppc_linear": 442, "ppc_quadratic": 443},
        "hashes_sha256": {
            "implementation": hashlib.sha256(script.read_bytes()).hexdigest(),
            "checker": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "requirements": hashlib.sha256(requirements.read_bytes()).hexdigest(),
            "base_data_X_y": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(),
            "changed_data_X_y": hashlib.sha256(
                X_changed.tobytes() + y_changed.tobytes()
            ).hexdigest(),
            "ppc_data_X_y": hashlib.sha256(
                ppc_X.tobytes() + ppc_y.tobytes()
            ).hexdigest(),
            "raw_arrays_npz": hashlib.sha256(raw_file.read_bytes()).hexdigest(),
        },
    }
    summary = {
        "receipt": receipt,
        "base": case_summary(base),
        "changed": case_summary(changed),
        "posterior_predictive": {
            name: ppc_summary(result) for name, result in ppc.items()
        },
        "diagnostic_controls": controls,
        "mutation_regressions_rejected": mutation_results,
        "assets": {
            "raw_arrays": raw_file.name,
            "diagnostic_svg": diagnostic_svg.name,
            "ppc_svg": ppc_svg.name,
        },
    }
    figure_data = {
        "diagnostics": {
            "title": "NumPyro NUTS diagnostics against the conjugate Gaussian posterior",
            "x": ["intercept", "slope"],
            "xlabel": "regression parameter",
            "left_axis": {
                "label": "rank-normalized split R-hat (unitless)",
                "guide": 1.01,
                "values": base["diagnostics"]["rhat"].tolist(),
            },
            "right_axis": {
                "label": "effective sample size (effective draws)",
                "bulk_ess": base["diagnostics"]["bulk_ess"].tolist(),
                "tail_ess": base["diagnostics"]["tail_ess"].tolist(),
            },
            "chains": CHAINS,
            "draws_per_chain": DRAWS,
            "divergences": base["divergences"],
            "scope": "Measured CPU execution of the pinned environment. ESS estimates information after autocorrelation adjustment; it is not independent stored rows. Diagnostics do not prove convergence or rule out missed modes.",
        },
        "posterior_predictive": {
            "title": "Paired posterior predictive discrepancy for residual curvature",
            "xlabel": "quadratic residual coefficient (y / x²)",
            "ylabel": "paired posterior predictive replications per histogram bin",
            "panels": {
                name: {
                    "observed_median": result["observed_median"],
                    "observed_interval_90": result["observed_interval_90"],
                    "replicated_interval_90": result["replicated_interval_90"],
                    "paired_tail_probability": result["paired_tail_probability"],
                    "histogram": result["histogram"],
                }
                for name, result in ppc.items()
            },
            "scope": "Each T(y, theta_s) is paired with T(y_rep_s, theta_s) using the same posterior draw. The 41-point curved fixture is synthetic and does not estimate real-data coverage.",
        },
    }
    summary["figure_data"] = figure_data
    summary_path = out / "numpyro-check-evidence.json"
    summary_path.write_text(json.dumps(_plain(summary), indent=2) + "\n")
    (out / "numpyro-diagnostics-figure.json").write_text(
        json.dumps(_plain(figure_data["diagnostics"]), indent=2) + "\n"
    )
    (out / "numpyro-ppc-figure.json").write_text(
        json.dumps(_plain(figure_data["posterior_predictive"]), indent=2) + "\n"
    )
    return _plain(summary)


def _render_ppc_svg(path, ppc):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for axis, name, title, color in zip(
        axes,
        ("linear", "quadratic"),
        ("linear model", "quadratic remedy"),
        ("#bd493b", "#1f9b8f"),
    ):
        result = ppc[name]
        vals = result["replicated_statistic_by_draw"]
        axis.hist(vals, bins=28, color="#aab4c5", edgecolor="white")
        obs = result["observed_statistic_by_draw"]
        axis.axvline(
            np.median(obs),
            color=color,
            linewidth=2.2,
            label=f"median observed T = {np.median(obs):.3f}",
        )
        axis.set_xlabel("quadratic residual coefficient (y / x²)")
        axis.set_ylabel("paired posterior predictive replications per bin")
        axis.set_title(title)
        axis.legend(frameon=False)
    fig.suptitle("Paired posterior predictive check: residual curvature")
    fig.savefig(path, format="svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation",
        required=True,
        help="starter/numpyro_workflow.py or a completed implementation",
    )
    parser.add_argument(
        "--evidence-dir",
        default=str(Path(tempfile.gettempdir()) / "bayesian-regression-numpyro-check"),
        help="directory for raw arrays, receipt and figures (default: system temporary directory)",
    )
    args = parser.parse_args()
    module = load_implementation(args.implementation)
    try:
        result = verify(module, args.evidence_dir, args.implementation)
    except Exception as exc:
        print(f"FAIL optional NumPyro stage: {type(exc).__name__}: {exc}")
        raise
    print(
        "PASS optional NumPyro stage: raw NUTS chains, independent exact solve, batch-means MCSE, rank R-hat, bulk/tail ESS, divergences, paired PPC and evidence mutations"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
