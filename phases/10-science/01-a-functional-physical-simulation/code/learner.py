"""A functional physical simulation · learner-only exercises (3/3 contracts ready)"""

# Exercise: Change the rate to \(0.3\), the initial excess to \(5\), and use twenty steps of \(0.1\) seconds. Verify the full discrete trajectory and bound the endpoint error against the exponential.

# Inputs:
# - Initial excess 5.0, rate 0.3, step size 0.1, and twenty explicit-Euler steps.

# Outputs:
# - Full discrete state trajectory and the analytic exponential trajectory.
# - An endpoint-error bound relative to the exponential solution at the matching final time.

# Invariants to check:
# - Each update uses the previous state.
# - The endpoint uses the same physical time for both methods.

# Hints:
# - Record the initial value before applying any step.
# - Compare discrete updates with exp(-rate * time), including the final time.

import numpy as np

def compare_euler_and_exponential(initial, rate, dt, steps):
    # TODO: return times, discrete trajectory, and analytic values.
    raise NotImplementedError("Build both trajectories on the same time grid")



# Energy does not check every invariant (Practice):
# Construct an energy diagnostic \(E_n=u_n^2/2\). Test a monotone stable grid and an unstable grid, and explain why decreasing energy alone does not guarantee a nonnegative state.

# Inputs:
# - A stable and an unstable explicit-Euler step size for scalar decay.

# Outputs:
# - Energy histories and state-sign checks for both runs.

# Invariants to check:
# - Energy E=u²/2 can decrease even after the state crosses below zero.
# - Stability and nonnegativity are separate properties.

# Hints:
# - Track signed state as well as its square.
# - Use a step large enough to cross zero but remain within a simple decay example.

import numpy as np

def energy_diagnostics(initial, rate, step_sizes, steps):
    # TODO: simulate each grid and return state and energy histories.
    raise NotImplementedError("Check signed-state and energy invariants separately")



# Keep the physical horizon fixed (Challenge):
# A colleague reports lower error after reducing both the step and number of updates. Reconstruct the final time for each run and repair the comparison.

# Inputs:
# - Two step-size/update-count pairs and a fixed physical decay problem.

# Outputs:
# - Final physical time and endpoint error for each configuration.

# Invariants to check:
# - A fair accuracy comparison uses the same final time.
# - Changing the step count changes the horizon unless dt*steps stays fixed.

# Hints:
# - Compute the horizon before comparing endpoint values.
# - Use the exponential solution at each run’s actual horizon.

import numpy as np

def compare_horizons(initial, rate, runs):
    # TODO: for each (dt, steps), compute its horizon and endpoint error.
    raise NotImplementedError("Separate discretization error from a changed horizon")
