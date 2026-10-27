"""Differentiate through a solver · learner-only exercises (4/4 contracts ready)"""

# Exercise: Change the initial value to \(3\) and the rate to \(0.4\). Verify the endpoint gradient against the continuous derivative, then check the derivative with respect to the initial value as well.

# Inputs:
# - RK4 endpoint solver, initial value 3.0, rate 0.4, and the lesson final time.

# Outputs:
# - Endpoint derivatives with respect to rate and initial value.
# - An analytic continuous-time comparison for both sensitivities.

# Invariants to check:
# - The rate derivative matches -t*u0*exp(-rate*t) within numerical tolerance.
# - The initial-value derivative matches exp(-rate*t).
# - Derivatives are taken through the discrete solver before comparison with the continuous reference.

# Hints:
# - Differentiate the actual discrete endpoint with respect to each input separately.
# - Use the same final time in the two analytic sensitivity formulas.

import numpy as np
import jax
import jax.numpy as jnp

def endpoint_sensitivities(solve_endpoint, final_time, initial=3.0, rate=0.4):
    # TODO: differentiate the discrete endpoint with respect to both inputs.
    # Compare each result with its continuous-time sensitivity.
    raise NotImplementedError("Check rate and initial-state endpoint sensitivities")



# Weight observations without losing normalization (Practice):
# Differentiate a weighted observation loss that emphasizes late times. Compare with a NumPy chain-rule reference built from the RK4 polynomial.

# Inputs:
# - Observation times, weights emphasizing later observations, initial value, and a rate parameter.

# Outputs:
# - Weighted observation-loss gradient and an independent NumPy chain-rule value.

# Invariants to check:
# - Normalization uses the sum of observation weights.
# - The independent path differentiates the RK4 polynomial rather than calling JAX autodiff.

# Hints:
# - Write the weighted residual loss explicitly.
# - Propagate a scalar derivative through each RK4 stage in the reference calculation.

import numpy as np
import jax
import jax.numpy as jnp

def weighted_loss_gradient(times, observations, weights, initial, rate):
    # TODO: define the weighted loss and build an independent RK4 derivative check.
    raise NotImplementedError("Preserve weighted-loss normalization and independent derivative check")



# Diagnose a detached backward path (Challenge):
# Reproduce an accidentally detached simulation and explain why a finite-difference test catches it.

# Inputs:
# - The decay solver, observations generated for initial value 2.0, and scalar rate parameter.
# - A differentiable objective and the same objective with its simulated prediction detached.
# - Parameter point 1.0 and centered finite-difference step 1e-4.

# Outputs:
# - Autodiff gradients for attached and detached objectives.
# - Centered finite-difference estimate exposing the detached-path error.

# Invariants to check:
# - The detached autodiff path can be zero while the objective value still changes with the parameter.
# - Centered finite differences use the same held observations and initial state.

# Hints:
# - Keep the forward objective identical and detach only its simulated prediction in the intentionally broken variant.
# - Compare autodiff with a centered difference around the stated parameter.

import jax
import jax.numpy as jnp

def diagnose_detached_path(solve, observations, parameter=1.0, initial=2.0, epsilon=1e-4):
    # TODO: compare attached and detached autodiff with a centered difference.
    raise NotImplementedError("Find the gradient path severed by stop_gradient")



# Check adaptive log-rate sensitivity at multiple tolerances (Optional extension):
# In the isolated Diffrax environment, evaluate the numerical derivative with respect to $\log k$ at $k=1.1$, $y_0=2.4$, and the stated observation grid. Compare all requested tolerances with the continuous derivative and a centered finite difference. Explain why agreement can improve as tolerances tighten without making the finite-tolerance solve identical to the continuous system.

# Inputs:
# - Rate 1.1, initial state 2.4, observation times from 0 to 2, and tolerances 1e-5, 1e-8, and 1e-10.
# - The optional project API `log_rate_sensitivity(log_rate, initial, times, solver, rtol, atol)`.
# - The optional project API `solve_forced_decay(rate, initial, times, solver, rtol, atol)` for numerical centered differences.

# Outputs:
# - Numerical dy/dlog(k) at each requested time for every tolerance.
# - Maximum error against the continuous derivative and centered finite difference for each tolerance.

# Invariants to check:
# - The differentiated parameter is log(k), so the positive rate equals exp(log_rate).
# - All evaluations return one finite sensitivity per requested time in the input order.
# - The continuous reference is -k*t*(y0-1)*exp(-k*t) and is calculated independently of Diffrax.

# Hints:
# - Use the same observation grid for all tolerance settings.
# - Use a symmetric perturbation in log(k) for the finite-difference check.

import numpy as np

def compare_adaptive_sensitivities(log_rate_sensitivity, solve_forced_decay, rate, initial, times, tolerances):
    # TODO: evaluate the submitted sensitivity API and compare with independent references.
    raise NotImplementedError("Compare numerical and continuous log-rate sensitivities")
