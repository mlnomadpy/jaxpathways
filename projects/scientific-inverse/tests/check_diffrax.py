"""Independent acceptance checks for the optional Diffrax extension."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "jaxpathways-mpl-cache")
)

import jax

jax.config.update("jax_enable_x64", True)

import diffrax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def exact(rate, initial, times):
    """Independent host oracle for the continuous forced-decay equation."""
    t = np.asarray(times, dtype=np.float64)
    return np.cos(t) + (float(initial) - 1.0) * np.exp(-float(rate) * t)


def exact_sensitivity(rate, initial, times):
    t = np.asarray(times, dtype=np.float64)
    return -float(rate) * t * (float(initial) - 1.0) * np.exp(-float(rate) * t)


def profile_nll(rate_grid, times, observations, initial, sigma):
    predicted = np.stack([exact(rate, initial, times) for rate in rate_grid])
    return 0.5 * np.sum(((predicted - observations[None, :]) / sigma) ** 2, axis=1)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def require_close(actual, expected, *, atol, rtol, message):
    require(np.shape(actual) == np.shape(expected), f"{message}: shape mismatch")
    require(np.isfinite(actual).all(), f"{message}: nonfinite actual values")
    require(np.allclose(actual, expected, atol=atol, rtol=rtol), message)


def load_module(path):
    spec = importlib.util.spec_from_file_location("diffrax_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_result(result, times, expected, *, label, atol, rtol):
    require(isinstance(result, dict), f"{label}: solve must return a result mapping")
    for key in (
        "times",
        "values",
        "success",
        "status",
        "num_steps",
        "num_accepted_steps",
        "num_rejected_steps",
    ):
        require(key in result, f"{label}: missing {key}")
    actual_times = np.asarray(result["times"])
    values = np.asarray(result["values"])
    require(actual_times.shape == times.shape, f"{label}: wrong saved time shape")
    require_close(
        actual_times, times, atol=0, rtol=0, message=f"{label}: wrong observation times"
    )
    require(values.shape == expected.shape, f"{label}: wrong trajectory shape")
    require(
        bool(result["success"]),
        f"{label}: solver did not report success: {result['status']}",
    )
    require(
        result["status"] == "successful", f"{label}: success status is not explicit"
    )
    require(np.isfinite(values).all(), f"{label}: solution is nonfinite")
    steps = [
        int(result[k])
        for k in ("num_steps", "num_accepted_steps", "num_rejected_steps")
    ]
    require(
        steps[0] > 0 and steps[1] + steps[2] == steps[0],
        f"{label}: inconsistent step accounting {steps}",
    )
    require_close(
        values,
        expected,
        atol=atol,
        rtol=rtol,
        message=f"{label}: trajectory disagrees with analytic oracle",
    )
    return values, steps


def check_sensitivity(module, times, rate, initial, *, tolerance):
    got = module.log_rate_sensitivity(
        np.log(rate),
        initial,
        times,
        solver="tsit5",
        rtol=tolerance,
        atol=tolerance * 1e-2,
    )
    oracle = exact_sensitivity(rate, initial, times)
    require_close(
        got,
        oracle,
        atol=max(2e-8, tolerance * 15),
        rtol=max(2e-7, tolerance * 15),
        message=f"log-rate sensitivity mismatch at rtol={tolerance:g}",
    )
    epsilon = 2e-5
    finite_difference = (
        exact(rate * np.exp(epsilon), initial, times)
        - exact(rate * np.exp(-epsilon), initial, times)
    ) / (2 * epsilon)
    require_close(
        got,
        finite_difference,
        atol=max(3e-8, tolerance * 18),
        rtol=max(3e-7, tolerance * 18),
        message=f"sensitivity disagrees with independent centered finite difference at rtol={tolerance:g}",
    )
    return np.asarray(got), float(np.max(np.abs(np.asarray(got) - oracle)))


def require_rejects(call, label):
    try:
        call()
    except (ValueError, TypeError):
        return
    raise AssertionError(f"invalid input was accepted: {label}")


def classify_identifiability(profile_range, prediction_spread):
    """Checker-owned rule: exact flat predictions and likelihood are unidentifiable."""
    flat = profile_range <= 1e-10 and prediction_spread <= 1e-12
    return not flat


def check_false_identifiability_mutation(profile_range, prediction_spread):
    candidate_claim = True
    checker_claim = classify_identifiability(profile_range, prediction_spread)
    require(
        candidate_claim == checker_claim,
        "mutation control: rejected a false positive identifiability claim for a flat profile",
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_figures(output, raw, report):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key in zip(axes, ("base", "stiff")):
        item = raw["trajectory_" + key]
        ax.plot(
            item["times"], item["analytic"], color="#222222", label="analytic solution"
        )
        ax.plot(item["times"], item["values"], "o", ms=3, label=item["solver"])
        ax.set_title(f"{item['solver']} (k={item['rate']:g})")
        ax.set_xlabel("time t")
        ax.set_ylabel("state y(t)")
        ax.legend(frameon=False)
    fig.suptitle("Adaptive solver trajectories at requested observation times")
    fig.savefig(output / "trajectories.svg", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for key, values in raw["sensitivities"].items():
        ax.plot(raw["sensitivity_times"], values, label=key)
    ax.set_xlabel("time t")
    ax.set_ylabel("dy / d log(k)")
    ax.set_title("Numerical reverse-mode sensitivity and exact continuous derivative")
    ax.legend(frameon=False)
    fig.savefig(output / "sensitivities.svg", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, name in zip(axes, ("early", "on_manifold", "late")):
        values = raw["profiles"][name]["nll"]
        ax.plot(raw["profile_rates"], values - np.min(values), color="#5865f2")
        ax.set_title(raw["profiles"][name]["label"])
        ax.set_xlabel("rate k")
        ax.set_ylabel("profile NLL minus minimum")
        ax.set_ylim(bottom=0)
    fig.suptitle("Known-noise analytic profile: observation timing and identifiability")
    fig.savefig(output / "rate-profiles.svg", dpi=150)
    plt.close(fig)
    report["figures"] = {
        "trajectories.svg": {
            "x": "requested time t",
            "y": "state y(t)",
            "axes": "two panels: nonstiff Tsit5 and stiff Kvaerno5, each overlaid with analytic solution",
            "limits": "only the saved synthetic cases; solver cost is recorded in report",
        },
        "sensitivities.svg": {
            "x": "requested time t",
            "y": "dy/d log(k)",
            "axes": "three Diffrax tolerances and the exact continuous derivative",
            "limits": "reverse derivative of the numerical solve; agreement is measured at tested tolerances",
        },
        "rate-profiles.svg": {
            "x": "positive rate k",
            "y": "known-noise Gaussian profile NLL minus its grid minimum",
            "axes": "early off-manifold, on-manifold y0=1, and late off-manifold observations",
            "limits": "synthetic fixed-noise profile, not a confidence interval for real measurements",
        },
    }


def run_checks(module, implementation_path, output):
    output.mkdir(parents=True, exist_ok=True)
    raw = {}
    cases = [
        ("base", 0.7, 2.3, np.linspace(0.0, 2.0, 41), "tsit5", 2e-6),
        (
            "changed",
            2.4,
            -0.4,
            np.array([0.0, 0.03, 0.16, 0.49, 0.8, 1.4]),
            "tsit5",
            2e-6,
        ),
        ("stiff3", 80.0, 2.0, np.linspace(0.0, 1.0, 31), "kvaerno3", 2e-5),
        ("stiff", 80.0, 2.0, np.linspace(0.0, 1.0, 31), "kvaerno5", 2e-5),
    ]
    results = {}
    for name, rate, initial, times, solver, tolerance in cases:
        before = times.copy()
        result = module.solve_forced_decay(
            rate, initial, times, solver=solver, rtol=1e-8, atol=1e-10
        )
        require(
            np.array_equal(times, before), f"{name}: solver mutated caller time input"
        )
        expected = exact(rate, initial, times)
        values, steps = check_result(
            result, times, expected, label=name, atol=tolerance, rtol=tolerance
        )
        results[name] = result
        raw["trajectory_" + name] = {
            "times": times,
            "values": values,
            "analytic": expected,
            "rate": rate,
            "initial": initial,
            "solver": solver,
            "steps": steps,
        }

    base_t = cases[0][3]
    loose = module.solve_forced_decay(
        0.7, 2.3, base_t, solver="tsit5", rtol=1e-4, atol=1e-6
    )
    tight = module.solve_forced_decay(
        0.7, 2.3, base_t, solver="tsit5", rtol=1e-10, atol=1e-12
    )
    err_loose = float(
        np.max(np.abs(np.asarray(loose["values"]) - exact(0.7, 2.3, base_t)))
    )
    err_tight = float(
        np.max(np.abs(np.asarray(tight["values"]) - exact(0.7, 2.3, base_t)))
    )
    require(
        tight["success"] and err_tight <= err_loose + 2e-12,
        "tightened tolerance did not preserve/improve trajectory error",
    )
    raw["refinement"] = {
        "loose_error": err_loose,
        "tight_error": err_tight,
        "loose_steps": int(loose["num_steps"]),
        "tight_steps": int(tight["num_steps"]),
    }

    sensitivity_times = np.linspace(0.0, 2.0, 41)
    sensitivities = {}
    sensitivity_errors = {}
    for tolerance in (1e-5, 1e-8, 1e-10):
        got, error = check_sensitivity(
            module, sensitivity_times, 1.1, 2.4, tolerance=tolerance
        )
        sensitivities[f"Diffrax rtol={tolerance:g}"] = got
        sensitivity_errors[str(tolerance)] = error
    sensitivities["exact continuous"] = exact_sensitivity(1.1, 2.4, sensitivity_times)
    raw["sensitivity_times"] = sensitivity_times
    raw["sensitivities"] = sensitivities

    # A mutation that returns a detached zero gradient must fail the same contract.
    class DetachedSensitivity:
        @staticmethod
        def log_rate_sensitivity(*args, **kwargs):
            return np.zeros_like(sensitivity_times)

    try:
        check_sensitivity(
            DetachedSensitivity, sensitivity_times, 1.1, 2.4, tolerance=1e-8
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("mutation control: detached sensitivity was accepted")

    try:
        module.log_rate_sensitivity(
            np.log(80.0),
            2.0,
            np.linspace(0.0, 2.0, 21),
            solver="tsit5",
            rtol=1e-12,
            atol=1e-14,
            max_steps=1,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "mutation control: gradient from an unsuccessful primal solve was accepted"
        )

    # A fabricated zero trajectory must fail the independent analytic oracle.
    fake = dict(results["base"])
    fake["values"] = np.zeros_like(base_t)
    try:
        check_result(
            fake,
            base_t,
            exact(0.7, 2.3, base_t),
            label="fake-output mutation",
            atol=2e-6,
            rtol=2e-6,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("mutation control: fabricated trajectory was accepted")

    sigma = 0.025
    fit_times = np.linspace(0.0, 1.2, 41)
    true_rate, fit_initial = 1.25, 2.6
    rng = np.random.default_rng(81021)
    observations = exact(true_rate, fit_initial, fit_times) + rng.normal(
        0.0, sigma, fit_times.size
    )
    original_times, original_obs = fit_times.copy(), observations.copy()
    fitted = float(
        module.fit_rate(
            fit_times,
            observations,
            fit_initial,
            initial_rate=0.45,
            noise_scale=sigma,
            solver="tsit5",
            rtol=1e-8,
            atol=1e-10,
            updates=220,
            learning_rate=0.03,
        )
    )
    require(
        np.isfinite(fitted) and fitted > 0,
        "fit_rate must return a finite positive scalar",
    )
    require(
        np.array_equal(fit_times, original_times)
        and np.array_equal(observations, original_obs),
        "fit_rate mutated caller arrays",
    )
    rate_grid = np.linspace(0.2, 2.5, 4001)
    nll = profile_nll(rate_grid, fit_times, observations, fit_initial, sigma)
    oracle_fit = float(rate_grid[np.argmin(nll)])
    fitted_nll = float(
        profile_nll(np.array([fitted]), fit_times, observations, fit_initial, sigma)[0]
    )
    best_nll = float(np.min(nll))
    require(
        abs(fitted - oracle_fit) < 0.035 and fitted_nll - best_nll < 0.15,
        f"numerical fit {fitted:.6g} disagrees with independent analytic profile optimum {oracle_fit:.6g}",
    )
    holdout_times = np.linspace(0.0, 2.0, 61)
    heldout_initials = np.array([0.7, 2.0])
    heldout_results = [
        module.solve_forced_decay(
            fitted,
            y0,
            holdout_times,
            solver="tsit5",
            rtol=1e-8,
            atol=1e-10,
        )
        for y0 in heldout_initials
    ]
    heldout = np.stack([np.asarray(result["values"]) for result in heldout_results])
    for y0, result in zip(heldout_initials, heldout_results):
        check_result(
            result,
            holdout_times,
            exact(fitted, y0, holdout_times),
            label="held-out solver replay",
            atol=2e-6,
            rtol=2e-6,
        )
    heldout_truth = np.stack(
        [exact(true_rate, y0, holdout_times) for y0 in heldout_initials]
    )

    profile_grid = np.linspace(0.05, 3.0, 601)
    profiles = {}
    for name, y0, times in (
        ("early", 2.6, np.linspace(0.0, 1.2, 41)),
        ("on_manifold", 1.0, np.linspace(0.0, 1.2, 41)),
        ("late", 2.6, np.linspace(6.0, 9.0, 41)),
    ):
        obs = exact(true_rate, y0, times) + rng.normal(0.0, sigma, times.size)
        profile = profile_nll(profile_grid, times, obs, y0, sigma)
        predictions = np.stack(
            [exact(k, y0, times) for k in (0.2, 0.6, 1.25, 2.0, 3.0)]
        )
        entry = {
            "label": {
                "early": "early off-manifold",
                "on_manifold": "on-manifold y0=1",
                "late": "late off-manifold",
            }[name],
            "times": times,
            "observations": obs,
            "nll": profile,
            "prediction_spread": float(np.max(np.ptp(predictions, axis=0))),
            "profile_range": float(np.ptp(profile)),
            "identified": classify_identifiability(
                float(np.ptp(profile)), float(np.max(np.ptp(predictions, axis=0)))
            ),
        }
        profiles[name] = entry
    require(
        profiles["early"]["identified"],
        "early off-manifold observations should carry rate information",
    )
    require(
        not profiles["on_manifold"]["identified"],
        "on-manifold predictions/profile are structurally rate-flat",
    )
    require(
        profiles["on_manifold"]["prediction_spread"] <= 1e-12,
        "on-manifold predictions must be flat across rate grid",
    )
    require(
        profiles["late"]["prediction_spread"] < profiles["early"]["prediction_spread"],
        "late observations should have less rate sensitivity",
    )
    try:
        check_false_identifiability_mutation(
            profiles["on_manifold"]["profile_range"],
            profiles["on_manifold"]["prediction_spread"],
        )
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "mutation control: false certainty on a flat profile was accepted"
        )
    raw["fit"] = {
        "times": fit_times,
        "observations": observations,
        "fitted_rate": fitted,
        "oracle_rate": oracle_fit,
        "profile_rates": rate_grid,
        "profile_nll": nll,
        "holdout_times": holdout_times,
        "heldout_initials": heldout_initials,
        "heldout_predictions": heldout,
        "heldout_truth": heldout_truth,
    }
    raw["profile_rates"] = profile_grid
    raw["profiles"] = profiles

    # Configuration errors and explicit unsuccessful-step status.
    require_rejects(
        lambda: module.solve_forced_decay(1.0, 2.0, [0.0, 0.2, 0.2]), "duplicate times"
    )
    require_rejects(
        lambda: module.solve_forced_decay(1.0, 2.0, [0.0, 0.3, 0.2]), "decreasing times"
    )
    require_rejects(
        lambda: module.solve_forced_decay(1.0, 2.0, [0.0, np.nan, 1.0]),
        "nonfinite times",
    )
    require_rejects(
        lambda: module.solve_forced_decay(0.0, 2.0, [0.0, 1.0]), "nonpositive rate"
    )
    require_rejects(
        lambda: module.solve_forced_decay(1.0, 2.0, [0.0, 1.0], rtol=0), "zero rtol"
    )
    require_rejects(
        lambda: module.solve_forced_decay(1.0, 2.0, [0.0, 1.0], atol=-1e-3),
        "negative atol",
    )
    require_rejects(
        lambda: module.solve_forced_decay(1.0, 2.0, [0.0, 1.0], max_steps=0),
        "zero max_steps",
    )
    require_rejects(
        lambda: module.solve_forced_decay(1.0, 2.0, [1.0, 1.2, 1.5]),
        "nonzero initial observation time",
    )
    short = module.solve_forced_decay(
        80.0,
        2.0,
        np.linspace(0, 2, 21),
        solver="tsit5",
        rtol=1e-12,
        atol=1e-14,
        max_steps=1,
    )
    require(
        not short["success"]
        and short["num_steps"] == 1
        and short["num_rejected_steps"] == 1,
        "max_steps=1 must be exposed as an unsuccessful solve with accounting",
    )
    require(
        "maximum number of solver steps" in short["status"],
        "failed status should explain that max_steps was reached",
    )
    try:
        module.fit_rate(
            np.linspace(0.0, 2.0, 21),
            exact(80.0, 2.0, np.linspace(0.0, 2.0, 21)),
            2.0,
            initial_rate=80.0,
            solver="tsit5",
            rtol=1e-12,
            atol=1e-14,
            max_steps=1,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("fit from an unsuccessful primal solve was accepted")
    euler_t = np.linspace(0.0, 1.0, 26)
    euler_y = np.empty_like(euler_t)
    euler_y[0] = 2.0
    for i, (left, right) in enumerate(zip(euler_t[:-1], euler_t[1:])):
        dt = right - left
        euler_y[i + 1] = euler_y[i] + dt * (
            -80.0 * (euler_y[i] - np.cos(left)) - np.sin(left)
        )
    euler_error = float(np.max(np.abs(euler_y - exact(80.0, 2.0, euler_t))))
    require(
        euler_error > 1e6,
        "coarse explicit Euler should exhibit diagnosed stiff instability",
    )
    raw["failure"] = {
        "insufficient_steps": {
            k: short[k]
            for k in (
                "success",
                "status",
                "num_steps",
                "num_accepted_steps",
                "num_rejected_steps",
            )
        },
        "coarse_euler_max_error": euler_error,
        "coarse_euler_max_state": float(np.max(np.abs(euler_y))),
    }

    # Complete raw arrays are retained for independent replay and figure inspection.
    flattened = {}
    for name, case in raw.items():
        if name.startswith("trajectory_"):
            flattened[name + "_times"] = case["times"]
            flattened[name + "_values"] = case["values"]
            flattened[name + "_analytic"] = case["analytic"]
    flattened["sensitivity_times"] = sensitivity_times
    flattened.update(
        {
            "sensitivity_" + key.replace(" ", "_").replace("=", "_"): value
            for key, value in sensitivities.items()
        }
    )
    flattened.update(
        {
            "fit_times": fit_times,
            "fit_observations": observations,
            "fit_profile_rates": rate_grid,
            "fit_profile_nll": nll,
            "heldout_times": holdout_times,
            "heldout_initials": heldout_initials,
            "heldout_predictions": heldout,
            "heldout_truth": heldout_truth,
            "profile_rates": profile_grid,
        }
    )
    for name, entry in profiles.items():
        flattened[f"profile_{name}_times"] = entry["times"]
        flattened[f"profile_{name}_observations"] = entry["observations"]
        flattened[f"profile_{name}_nll"] = entry["nll"]
    np.savez_compressed(output / "raw-evidence.npz", **flattened)
    report = {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "diffrax": diffrax.__version__,
            "backend": jax.default_backend(),
            "dtype": "float64",
        },
        "solver_cases": {
            name: {
                "rate": case["rate"],
                "initial": case["initial"],
                "solver": case["solver"],
                "max_abs_error": float(
                    np.max(np.abs(case["values"] - case["analytic"]))
                ),
                "steps": case["steps"],
            }
            for name, case in (
                (key, raw["trajectory_" + key])
                for key in ("base", "changed", "stiff3", "stiff")
            )
        },
        "refinement": raw["refinement"],
        "sensitivity_max_abs_error": sensitivity_errors,
        "inverse_fit": {
            "seed": 81021,
            "noise_scale": sigma,
            "true_rate": true_rate,
            "fitted_rate": fitted,
            "independent_grid_rate": oracle_fit,
            "profile_delta_nll": fitted_nll - best_nll,
            "heldout_initials": heldout_initials.tolist(),
            "heldout_rmse": np.sqrt(
                np.mean((heldout - heldout_truth) ** 2, axis=1)
            ).tolist(),
        },
        "identifiability": {
            name: {
                "prediction_spread": entry["prediction_spread"],
                "profile_nll_range": entry["profile_range"],
                "identified_by_checker": entry["identified"],
            }
            for name, entry in profiles.items()
        },
        "failure_controls": raw["failure"],
        "checks": [
            "changed rate/initial/nonuniform times",
            "Tsit5 nonstiff",
            "Kvaerno3/5 stiff",
            "requested-time saving",
            "tolerance refinement",
            "log-rate autodiff and finite difference",
            "noise-owned inference and analytic profile oracle",
            "structural and practical identifiability",
            "invalid input and input mutation",
            "max-step status",
            "coarse Euler instability",
            "detached-gradient/fake-output/false-identifiability mutation rejection",
        ],
    }
    save_figures(output, raw, report)
    np.savez_compressed(
        output / "figure-data.npz",
        **{k: v for k, v in flattened.items() if isinstance(v, np.ndarray)},
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    data_hash = hashlib.sha256((output / "raw-evidence.npz").read_bytes()).hexdigest()
    receipt = {
        "implementation": str(Path(implementation_path).resolve()),
        "implementation_sha256": sha256(implementation_path),
        "checker_sha256": sha256(__file__),
        "requirements_sha256": sha256(
            Path(__file__).parents[1] / "requirements-science.txt"
        ),
        "fixture_data_sha256": hashlib.sha256(
            np.asarray(observations, dtype="<f8").tobytes()
            + np.asarray(fit_times, dtype="<f8").tobytes()
        ).hexdigest(),
        "raw_evidence_sha256": data_hash,
        "report_sha256": sha256(output / "report.json"),
        "environment": report["environment"],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "implementation": str(Path(implementation_path).resolve()),
                "evidence_dir": str(output.resolve()),
                "receipt": receipt,
                "fit_rate": fitted,
                "independent_profile_rate": oracle_fit,
                "max_sensitivity_error": max(sensitivity_errors.values()),
                "max_euler_error": euler_error,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--evidence-dir", default=None)
    args = parser.parse_args()
    output = (
        Path(args.evidence_dir)
        if args.evidence_dir
        else Path(tempfile.mkdtemp(prefix="jaxpathways-science-diffrax-"))
    )
    run_checks(load_module(args.implementation), args.implementation, output)


if __name__ == "__main__":
    main()
