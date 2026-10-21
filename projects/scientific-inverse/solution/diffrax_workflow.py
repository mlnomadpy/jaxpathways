"""Optional Diffrax inverse-problem workflow, pinned and tested on CPU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import jax

jax.config.update("jax_enable_x64", True)

import diffrax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOLVERS = {
    "tsit5": diffrax.Tsit5,
    "kvaerno3": diffrax.Kvaerno3,
    "kvaerno5": diffrax.Kvaerno5,
}


def forced_decay_vector_field(t, y, rate):
    """RHS with exact solution cos(t) + (initial - 1) exp(-rate*t)."""
    return -rate * (y - jnp.cos(t)) - jnp.sin(t)


def _validate_configuration(initial, times, solver, rtol, atol, max_steps):
    y0 = np.asarray(initial, dtype=np.float64)
    ts = np.asarray(times, dtype=np.float64)
    if y0.ndim != 0 or not np.isfinite(y0):
        raise ValueError("initial must be one finite scalar")
    if ts.ndim != 1 or ts.size < 2 or not np.isfinite(ts).all():
        raise ValueError("times must be a finite one-dimensional array with 2+ entries")
    if ts[0] != 0.0:
        raise ValueError("times must start at zero because initial is y(0)")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("times must be strictly increasing")
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {tuple(SOLVERS)}")
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol <= 0 or atol <= 0:
        raise ValueError("rtol and atol must be finite and positive")
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    return ts


def _raw_solve(rate, initial, times, solver, rtol, atol, max_steps):
    ts = jnp.asarray(times, dtype=jnp.float64)
    dt0 = (ts[-1] - ts[0]) / 20.0
    return diffrax.diffeqsolve(
        terms=diffrax.ODETerm(forced_decay_vector_field),
        solver=SOLVERS[solver](),
        t0=ts[0],
        t1=ts[-1],
        dt0=dt0,
        y0=jnp.asarray(initial, dtype=jnp.float64),
        args=jnp.asarray(rate, dtype=jnp.float64),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=max_steps,
        throw=False,
    )


def _steps_summary(stats):
    return {
        "num_steps": int(np.asarray(stats["num_steps"])),
        "num_accepted_steps": int(np.asarray(stats["num_accepted_steps"])),
        "num_rejected_steps": int(np.asarray(stats["num_rejected_steps"])),
    }


def _require_success(solution, context):
    success = bool(np.asarray(solution.result == diffrax.RESULTS.successful))
    values = np.asarray(solution.ys, dtype=np.float64)
    if not success:
        raise RuntimeError(f"{context} failed: {solution.result}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{context} returned nonfinite saved values")
    return values


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
    """Solve at the requested observation times and expose success and work counts."""
    ts = _validate_configuration(initial, times, solver, rtol, atol, max_steps)
    rate_host = np.asarray(rate, dtype=np.float64)
    if rate_host.ndim != 0 or not np.isfinite(rate_host) or rate_host <= 0:
        raise ValueError("rate must be a finite positive scalar")
    solution = _raw_solve(rate, initial, ts, solver, rtol, atol, max_steps)
    result = solution.result
    success = bool(np.asarray(result == diffrax.RESULTS.successful))
    return {
        "times": ts.copy(),
        "values": np.asarray(solution.ys, dtype=np.float64),
        "success": success,
        "status": "successful" if success else str(result),
        **_steps_summary(solution.stats),
        "solver": solver,
        "rtol": float(rtol),
        "atol": float(atol),
        "max_steps": max_steps,
    }


def _validate_differentiable_inputs(initial, times, solver, rtol, atol, max_steps):
    ts = _validate_configuration(initial, times, solver, rtol, atol, max_steps)
    return jnp.asarray(ts, dtype=jnp.float64)


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
    """Return reverse-mode derivative of saved values with respect to log(rate)."""
    ts = _validate_differentiable_inputs(initial, times, solver, rtol, atol, max_steps)
    theta_host = np.asarray(log_rate, dtype=np.float64)
    if theta_host.ndim != 0 or not np.isfinite(theta_host):
        raise ValueError("log_rate must be a finite scalar")
    rate_host = np.exp(theta_host)
    if not np.isfinite(rate_host) or rate_host <= 0:
        raise ValueError("exp(log_rate) must be finite and positive")
    _require_success(
        _raw_solve(rate_host, initial, ts, solver, rtol, atol, max_steps),
        "sensitivity primal solve",
    )
    theta = jnp.asarray(log_rate, dtype=jnp.float64)

    def values_from_log_rate(parameter):
        return _raw_solve(
            jnp.exp(parameter), initial, ts, solver, rtol, atol, max_steps
        ).ys

    result = jax.jacrev(values_from_log_rate)(theta)
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("sensitivity derivative returned nonfinite values")
    return result


def _validate_fit_inputs(
    times,
    observations,
    initial,
    initial_rate,
    noise_scale,
    solver,
    rtol,
    atol,
    max_steps,
    updates,
    learning_rate,
):
    ts = _validate_configuration(initial, times, solver, rtol, atol, max_steps)
    y = np.asarray(observations, dtype=np.float64)
    if y.shape != ts.shape or not np.isfinite(y).all():
        raise ValueError("observations must be finite and match the time vector")
    if not np.isfinite(initial_rate) or initial_rate <= 0:
        raise ValueError("initial_rate must be finite and positive")
    if not np.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("noise_scale must be finite and positive")
    if not isinstance(updates, int) or updates < 1:
        raise ValueError("updates must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    return ts, y


def _fit_with_history(
    times,
    observations,
    initial,
    *,
    initial_rate,
    noise_scale,
    solver,
    rtol,
    atol,
    max_steps,
    updates,
    learning_rate,
):
    ts, observed = _validate_fit_inputs(
        times,
        observations,
        initial,
        initial_rate,
        noise_scale,
        solver,
        rtol,
        atol,
        max_steps,
        updates,
        learning_rate,
    )
    ts_device = jnp.asarray(ts, dtype=jnp.float64)
    observed_device = jnp.asarray(observed, dtype=jnp.float64)
    sigma = jnp.asarray(noise_scale, dtype=jnp.float64)

    def objective(log_rate):
        predicted = _raw_solve(
            jnp.exp(log_rate), initial, ts_device, solver, rtol, atol, max_steps
        ).ys
        return jnp.mean(0.5 * ((predicted - observed_device) / sigma) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    theta = jnp.log(jnp.asarray(initial_rate, dtype=jnp.float64))
    primal = _raw_solve(initial_rate, initial, ts, solver, rtol, atol, max_steps)
    _require_success(primal, "fit initial primal solve")
    first_loss, first_gradient = value_and_grad(theta)
    if not np.isfinite(float(first_loss)) or not np.isfinite(float(first_gradient)):
        raise RuntimeError("fit initial objective or gradient is nonfinite")
    history = [float(first_loss)]
    first = jnp.asarray(0.9, dtype=jnp.float64)
    second = jnp.asarray(0.999, dtype=jnp.float64)
    m = jnp.asarray(0.0, dtype=jnp.float64)
    v = jnp.asarray(0.0, dtype=jnp.float64)
    gradient = first_gradient
    for step in range(1, updates + 1):
        if not np.isfinite(float(gradient)):
            raise RuntimeError("fit gradient became nonfinite")
        m = first * m + (1 - first) * gradient
        v = second * v + (1 - second) * gradient**2
        m_hat = m / (1 - first**step)
        v_hat = v / (1 - second**step)
        theta = theta - learning_rate * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        theta = jnp.clip(theta, jnp.log(1e-4), jnp.log(1e3))
        rate = float(jnp.exp(theta))
        _require_success(
            _raw_solve(rate, initial, ts, solver, rtol, atol, max_steps),
            f"fit primal solve at update {step}",
        )
        loss_value, gradient = value_and_grad(theta)
        if not np.isfinite(float(loss_value)) or not np.isfinite(float(gradient)):
            raise RuntimeError(
                f"fit objective or gradient is nonfinite at update {step}"
            )
        history.append(float(loss_value))
    fitted_rate = float(jnp.exp(theta))
    return fitted_rate, np.asarray(history, dtype=np.float64)


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
    """Fit one positive rate using known Gaussian observation noise; return only rate."""
    rate, _ = _fit_with_history(
        times,
        observations,
        initial,
        initial_rate=initial_rate,
        noise_scale=noise_scale,
        solver=solver,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        updates=updates,
        learning_rate=learning_rate,
    )
    return rate


def exact_solution_numpy(rate, initial, times):
    """Host NumPy oracle, independent of Diffrax/JAX solver output."""
    t = np.asarray(times, dtype=np.float64)
    return np.cos(t) + (float(initial) - 1.0) * np.exp(-float(rate) * t)


def exact_log_rate_sensitivity(rate, initial, times):
    """Analytic derivative of the continuous solution with respect to log(rate)."""
    t = np.asarray(times, dtype=np.float64)
    return -float(rate) * t * (float(initial) - 1.0) * np.exp(-float(rate) * t)


def fixed_euler(rate, initial, times):
    """Deliberately elementary explicit-Euler reference for instability diagnosis."""
    ts = np.asarray(times, dtype=np.float64)
    ys = np.empty_like(ts)
    ys[0] = initial
    for i, (left, right) in enumerate(zip(ts[:-1], ts[1:])):
        dt = right - left
        ys[i + 1] = ys[i] + dt * (-rate * (ys[i] - np.cos(left)) - np.sin(left))
    return ys


def _profile_nll(rate_grid, times, observations, initial, noise_scale):
    predictions = np.stack(
        [exact_solution_numpy(rate, initial, times) for rate in rate_grid]
    )
    residual = predictions - np.asarray(observations, dtype=np.float64)[None, :]
    return 0.5 * np.sum((residual / noise_scale) ** 2, axis=1)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_trajectory_figure(path, cases):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, case in zip(axes.flat, cases):
        t = case["times"]
        axis.plot(t, case["exact"], color="#252A34", lw=1.8, label="analytic")
        axis.plot(t, case["values"], "o", color="#267C78", ms=2.8, label=case["method"])
        axis.set_title(
            f"{case['title']} · k={case['rate']:g}, y₀={case['initial']:g}\n"
            f"{case['num_steps']:,} steps ({case['num_accepted_steps']:,} accepted, "
            f"{case['num_rejected_steps']:,} rejected); max error {case['max_abs_error']:.2e}",
            fontsize=8.5,
        )
        axis.set_xlabel("time (s)")
        axis.set_ylabel("state y (unitless model state)")
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.18)
    for axis in axes.flat[len(cases) :]:
        axis.set_visible(False)
    fig.suptitle(
        "Adaptive solver trajectories at requested observations",
        fontsize=14,
    )
    fig.savefig(path, dpi=160)
    fig.savefig(Path(path).with_suffix(".png"), dpi=180)
    plt.close(fig)


def _save_sensitivity_figure(path, times, series):
    fig, (axis, error_axis) = plt.subplots(
        1, 2, figsize=(12, 4.5), constrained_layout=True
    )
    exact = np.asarray(series[-1]["values"], dtype=np.float64)
    for item in series:
        axis.plot(times, item["values"], label=item["label"])
    colors = [line.get_color() for line in axis.lines]
    for item, color in zip(series, colors):
        absolute_error = np.abs(np.asarray(item["values"], dtype=np.float64) - exact)
        error_axis.plot(times, absolute_error, color=color, label=item["label"])
    axis.set_xlabel("time (s)")
    axis.set_ylabel("dy / d log(rate)")
    axis.set_title("Sensitivity curves")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.2)
    error_axis.set_xlabel("time (s)")
    error_axis.set_ylabel("absolute error in dy / d log(rate)")
    error_axis.set_title("Absolute error from continuous derivative")
    error_axis.set_ylim(bottom=0)
    error_axis.grid(alpha=0.2)
    error_axis.text(
        0.03,
        0.96,
        "Linear scale; analytic series is exactly zero error",
        transform=error_axis.transAxes,
        va="top",
        fontsize=8,
    )
    error_axis.legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Discretized log-rate sensitivity and absolute error")
    fig.savefig(path, dpi=160)
    fig.savefig(Path(path).with_suffix(".png"), dpi=180)
    plt.close(fig)


def _save_profile_figure(path, rate_grid, profiles):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for axis, (name, profile) in zip(axes, profiles.items()):
        shifted = profile["nll"] - np.min(profile["nll"])
        axis.plot(rate_grid, shifted, color="#267C78", lw=1.8)
        axis.axhline(0.0, color="#65727E", lw=0.8, ls="--")
        axis.set_title(profile["title"])
        axis.set_xlabel("rate k")
        axis.set_ylabel("profile NLL minus minimum")
        axis.grid(axis="y", alpha=0.18)
        if name == "on_manifold":
            # Keep the true flat zero curve above the frame edge without
            # modifying its data or substituting a plotting epsilon.
            axis.set_ylim(-0.05, 0.5)
            axis.annotate(
                "exactly 0 over all 4,001 rates",
                xy=(float(rate_grid[len(rate_grid) // 2]), 0.0),
                xytext=(0.12, 0.68),
                textcoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#52616B"},
                fontsize=8,
            )
        else:
            axis.set_ylim(bottom=0)
    fig.suptitle(
        "Known-noise analytic rate profiles: identifiability depends on observations"
    )
    fig.savefig(path, dpi=160)
    fig.savefig(Path(path).with_suffix(".png"), dpi=180)
    plt.close(fig)


def run_reference_workflow(output_dir=None, *, seed=341, rtol=1e-9, atol=1e-11):
    """Run solver, sensitivity and inverse-problem experiments into an evidence folder."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="scientific-inverse-diffrax-")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    base_t = np.linspace(0.0, 2.0, 41, dtype=np.float64)
    changed_t = np.linspace(0.0, 1.4, 29, dtype=np.float64) ** 1.1
    stiff_t = np.linspace(0.0, 1.0, 51, dtype=np.float64)
    requested_cases = [
        ("nonstiff Tsit5", 0.7, 2.3, base_t, "tsit5"),
        ("nonstiff Kvaerno5", 0.7, 2.3, base_t, "kvaerno5"),
        ("changed k/initial/grid", 2.4, -0.4, changed_t, "tsit5"),
        ("stiff Kvaerno3", 80.0, 2.0, stiff_t, "kvaerno3"),
        ("stiff Kvaerno5", 80.0, 2.0, stiff_t, "kvaerno5"),
    ]
    cases = []
    for title, rate, initial, times, solver in requested_cases:
        result = solve_forced_decay(
            rate,
            initial,
            times,
            solver=solver,
            rtol=rtol,
            atol=atol,
        )
        exact = exact_solution_numpy(rate, initial, times)
        cases.append(
            {
                "title": title,
                "method": f"{solver} adaptive",
                "rate": rate,
                "initial": initial,
                "times": times,
                "values": result["values"],
                "exact": exact,
                "max_abs_error": float(np.max(np.abs(result["values"] - exact))),
                "status": result["status"],
                "success": result["success"],
                **{
                    key: result[key]
                    for key in ("num_steps", "num_accepted_steps", "num_rejected_steps")
                },
            }
        )

    sensitivity_t = np.linspace(0.0, 2.0, 41, dtype=np.float64)
    sensitivity_rate, sensitivity_initial = 1.1, 2.4
    analytic_sensitivity = exact_log_rate_sensitivity(
        sensitivity_rate, sensitivity_initial, sensitivity_t
    )
    sensitivity = []
    for tolerance in (1e-5, 1e-8, 1e-10):
        values = log_rate_sensitivity(
            np.log(sensitivity_rate),
            sensitivity_initial,
            sensitivity_t,
            rtol=tolerance,
            atol=tolerance * 1e-2,
        )
        sensitivity.append(
            {
                "label": f"Diffrax rtol={tolerance:g}",
                "values": values,
                "max_abs_error": float(np.max(np.abs(values - analytic_sensitivity))),
            }
        )
    sensitivity.append(
        {
            "label": "continuous analytic derivative",
            "values": analytic_sensitivity,
            "max_abs_error": 0.0,
        }
    )
    epsilon = 1e-5
    finite_difference = (
        exact_solution_numpy(
            sensitivity_rate * np.exp(epsilon), sensitivity_initial, sensitivity_t
        )
        - exact_solution_numpy(
            sensitivity_rate * np.exp(-epsilon), sensitivity_initial, sensitivity_t
        )
    ) / (2 * epsilon)

    rng = np.random.default_rng(seed)
    sigma = 0.025
    train_t = np.linspace(0.0, 1.2, 41, dtype=np.float64)
    truth_rate, train_initial = 1.25, 2.6
    train_observations = exact_solution_numpy(
        truth_rate, train_initial, train_t
    ) + rng.normal(0, sigma, train_t.size)
    fitted_rate, loss_history = _fit_with_history(
        train_t,
        train_observations,
        train_initial,
        initial_rate=0.45,
        noise_scale=sigma,
        solver="tsit5",
        rtol=1e-8,
        atol=1e-10,
        max_steps=100_000,
        updates=300,
        learning_rate=0.03,
    )
    holdout_t = np.linspace(0.0, 2.0, 61, dtype=np.float64)
    heldout_initials = np.asarray([0.7, 2.0], dtype=np.float64)
    heldout_predictions = np.stack(
        [
            solve_forced_decay(fitted_rate, y0, holdout_t, rtol=rtol, atol=atol)[
                "values"
            ]
            for y0 in heldout_initials
        ]
    )
    heldout_exact = np.stack(
        [exact_solution_numpy(truth_rate, y0, holdout_t) for y0 in heldout_initials]
    )
    profile_grid = np.linspace(0.05, 3.0, 4001, dtype=np.float64)
    profiles = {}
    profile_data = {}
    profile_scenarios = (
        ("early_off_manifold", train_initial, train_t, train_observations),
        ("on_manifold", 1.0, np.linspace(0.0, 1.2, 41), None),
        ("late_off_manifold", 2.6, np.linspace(6.0, 9.0, 41), None),
    )
    for name, y0, times, observations in profile_scenarios:
        times = np.asarray(times, dtype=np.float64)
        if observations is None:
            noise = rng.normal(0, sigma, len(times))
            observations = exact_solution_numpy(truth_rate, y0, times) + noise
        else:
            observations = np.asarray(observations, dtype=np.float64)
        if name == "early_off_manifold" and not (
            np.array_equal(times, train_t)
            and np.array_equal(observations, train_observations)
        ):
            raise RuntimeError("early profile must reuse the exact fit training data")
        nll = _profile_nll(profile_grid, times, observations, y0, sigma)
        predictions_across_rates = np.stack(
            [
                exact_solution_numpy(rate, y0, times)
                for rate in (0.2, 0.6, 1.25, 2.0, 3.0)
            ]
        )
        entry = {
            "title": {
                "early_off_manifold": "early, off-manifold",
                "on_manifold": "on-manifold y0=1",
                "late_off_manifold": "late, off-manifold",
            }[name],
            "times": np.asarray(times),
            "initial": y0,
            "observations": observations,
            "nll": nll,
            "profile_range": float(np.ptp(nll)),
            "prediction_spread_across_rates": float(
                np.max(np.ptp(predictions_across_rates, axis=0))
            ),
        }
        profile_data[name] = entry
        profiles[name] = entry
    training_profile = profiles["early_off_manifold"]["nll"]
    training_profile_min_index = int(np.argmin(training_profile))
    training_profile_min_rate = float(profile_grid[training_profile_min_index])
    fitted_training_profile_nll = float(
        _profile_nll(
            np.asarray([fitted_rate]),
            train_t,
            train_observations,
            train_initial,
            sigma,
        )[0]
    )
    training_profile_delta_nll = fitted_training_profile_nll - float(
        training_profile[training_profile_min_index]
    )
    training_data_sha256 = hashlib.sha256(
        np.asarray(train_t, dtype="<f8").tobytes()
        + np.asarray(train_observations, dtype="<f8").tobytes()
    ).hexdigest()

    unstable_t = np.linspace(0.0, 1.0, 26)
    unstable = fixed_euler(80.0, 2.0, unstable_t)
    unstable_exact = exact_solution_numpy(80.0, 2.0, unstable_t)
    too_few = solve_forced_decay(
        80.0,
        2.0,
        np.linspace(0.0, 2.0, 21),
        solver="tsit5",
        rtol=1e-12,
        atol=1e-14,
        max_steps=1,
    )

    _save_trajectory_figure(output / "trajectories.svg", cases)
    _save_sensitivity_figure(output / "sensitivity.svg", sensitivity_t, sensitivity)
    _save_profile_figure(output / "likelihood-profiles.svg", profile_grid, profiles)
    raw = {
        "trajectory_times": np.concatenate([case["times"] for case in cases]),
        "trajectory_values": np.concatenate([case["values"] for case in cases]),
        "trajectory_exact": np.concatenate([case["exact"] for case in cases]),
        "sensitivity_times": sensitivity_t,
        "sensitivity_analytic": analytic_sensitivity,
        "sensitivity_finite_difference": finite_difference,
        "train_times": train_t,
        "train_observations": train_observations,
        "loss_history": loss_history,
        "profile_rates": profile_grid,
        "profile_early_nll": profiles["early_off_manifold"]["nll"],
        "profile_on_nll": profiles["on_manifold"]["nll"],
        "profile_late_nll": profiles["late_off_manifold"]["nll"],
        "unstable_euler_times": unstable_t,
        "unstable_euler_values": unstable,
        "unstable_euler_exact": unstable_exact,
    }
    for index, case in enumerate(cases):
        raw[f"trajectory_case_{index}_times"] = case["times"]
        raw[f"trajectory_case_{index}_values"] = case["values"]
        raw[f"trajectory_case_{index}_analytic"] = case["exact"]
    for index, item in enumerate(sensitivity):
        raw[f"sensitivity_curve_{index}"] = np.asarray(item["values"], dtype=np.float64)
        raw[f"sensitivity_absolute_error_{index}"] = np.abs(
            np.asarray(item["values"], dtype=np.float64) - analytic_sensitivity
        )
    for name, item in profile_data.items():
        raw[f"profile_{name}_times"] = np.asarray(item["times"], dtype=np.float64)
        raw[f"profile_{name}_observations"] = np.asarray(
            item["observations"], dtype=np.float64
        )
        raw[f"profile_{name}_nll"] = np.asarray(item["nll"], dtype=np.float64)
    raw["fit_rate"] = np.asarray(fitted_rate, dtype=np.float64)
    raw["fit_training_prediction"] = exact_solution_numpy(
        fitted_rate, train_initial, train_t
    )
    raw["fit_training_residual"] = raw["fit_training_prediction"] - train_observations
    raw["heldout_times"] = holdout_t
    raw["heldout_initials"] = heldout_initials
    raw["heldout_predictions"] = heldout_predictions
    raw["heldout_exact_at_truth"] = heldout_exact
    np.savez_compressed(output / "raw-evidence.npz", **raw)
    figure_data = {
        key: value
        for key, value in raw.items()
        if key.startswith(("trajectory_", "sensitivity_", "profile_", "heldout_"))
    }
    np.savez_compressed(output / "figure-data.npz", **figure_data)
    report = {
        "evidence_dir": str(output.resolve()),
        "environment": {
            "python": __import__("platform").python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "diffrax": diffrax.__version__,
            "backend": jax.default_backend(),
            "dtype": "float64",
        },
        "configuration": {
            "seed": seed,
            "rtol": rtol,
            "atol": atol,
            "max_steps": 100_000,
            "noise_scale": sigma,
        },
        "solver_cases": [
            {
                key: value
                for key, value in case.items()
                if key not in ("times", "values", "exact")
            }
            for case in cases
        ],
        "sensitivity": {
            "max_abs_error_by_rtol": {
                str(t): item["max_abs_error"]
                for t, item in zip((1e-5, 1e-8, 1e-10), sensitivity[:3])
            },
            "finite_difference_epsilon": epsilon,
            "finite_difference_max_abs_error": float(
                np.max(np.abs(finite_difference - analytic_sensitivity))
            ),
            "formula": "-k*t*(y0-1)*exp(-k*t)",
        },
        "inverse_fit": {
            "known_rate": truth_rate,
            "fitted_rate": fitted_rate,
            "initial": train_initial,
            "training_times": [float(train_t[0]), float(train_t[-1])],
            "heldout_initials": heldout_initials.tolist(),
            "heldout_rmse_against_truth": np.sqrt(
                np.mean((heldout_predictions - heldout_exact) ** 2, axis=1)
            ).tolist(),
            "updates": len(loss_history) - 1,
            "initial_objective": float(loss_history[0]),
            "final_objective": float(loss_history[-1]),
            "profile_nll_grid_min_rate": training_profile_min_rate,
            "profile_nll_grid_minimum": float(
                training_profile[training_profile_min_index]
            ),
            "profile_nll_at_fitted_rate": fitted_training_profile_nll,
            "profile_nll_delta_from_grid_minimum": training_profile_delta_nll,
            "profile_grid": [
                float(profile_grid[0]),
                float(profile_grid[-1]),
                len(profile_grid),
            ],
            "profile_uses_fit_training_data": True,
            "training_data_sha256": training_data_sha256,
            "scope": "Synthetic observations with fixed known Gaussian noise; no instrument calibration claim.",
        },
        "identifiability": {
            name: {
                "profile_nll_range": entry["profile_range"],
                "prediction_spread_across_rates": entry[
                    "prediction_spread_across_rates"
                ],
                "identified_by_this_fixture": name == "early_off_manifold",
            }
            for name, entry in profile_data.items()
        },
        "failure_controls": {
            "explicit_euler_max_abs_error": float(
                np.max(np.abs(unstable - unstable_exact))
            ),
            "explicit_euler_max_abs_state": float(np.max(np.abs(unstable))),
            "insufficient_steps_success": too_few["success"],
            "insufficient_steps_status": too_few["status"],
            **{
                key: too_few[key]
                for key in ("num_steps", "num_accepted_steps", "num_rejected_steps")
            },
        },
        "figures": {
            "trajectories": {
                "file": "trajectories.svg",
                "png_file": "trajectories.png",
                "x": "time (s)",
                "y": "state y",
                "limits": "Five executed synthetic cases: two nonstiff solver comparisons, a changed-parameter/grid case, and stiff Kvaerno3/Kvaerno5. Each panel labels solver, rate, initial state, accepted/rejected step counts, and maximum error.",
                "values": "actual executed solutions and independent analytic values at requested output times",
            },
            "sensitivity": {
                "file": "sensitivity.svg",
                "png_file": "sensitivity.png",
                "x": "time (s)",
                "y": "dy/d log(rate), then absolute error from continuous derivative",
                "limits": "Left: three Diffrax tolerances and the continuous derivative. Right: linear absolute-error axis; exact zero for the analytic series is retained as zero, not replaced with a positive plotting value.",
                "values": "Diffrax gradients, continuous derivative, and absolute errors computed from the retained gradient arrays",
            },
            "profiles": {
                "file": "likelihood-profiles.svg",
                "png_file": "likelihood-profiles.png",
                "x": "positive rate k",
                "y": "known-noise profile NLL minus minimum",
                "limits": "Synthetic fixed-noise diagnostic. The on-manifold profile retains its exact all-zero values and is annotated; negative axis space only lifts the zero line from the frame edge.",
                "values": "Analytic predictions compared with fixed-seed observations; the early panel reuses the exact fit observations.",
            },
            "figure_data": {
                "file": "figure-data.npz",
                "values": "Time, trajectory, sensitivity, profile, and held-out arrays used to make the figures.",
            },
        },
        "sha256": {"raw_evidence": _sha256(output / "raw-evidence.npz")},
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help="evidence output directory (defaults to a fresh system temporary folder)",
    )
    parser.add_argument("--seed", type=int, default=341)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-11)
    args = parser.parse_args()
    print(
        json.dumps(
            run_reference_workflow(
                args.output, seed=args.seed, rtol=args.rtol, atol=args.atol
            ),
            indent=2,
        )
    )
