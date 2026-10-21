"""Capacity, utilization, and operating cost · learner-only exercises (3/3 contracts ready)"""

# Exercise: Using the measured service-time input, calculate the minimum worker count for \(1200\) arrivals per hour with \(40\)% reserve, then independently verify that the chosen count satisfies the load bound.

# Inputs:
# - Arrival demand, measured service duration, reserve fraction, and proposed worker count.

# Outputs:
# - A minimum worker estimate and an independent utilization/load-bound check.

# Invariants to check:
# - Units are consistent; full reserve, nonpositive workers, and nonfinite service measurements are rejected.

# Hints:
# - Convert hourly arrivals to per-second demand before multiplying by service time.
# - Label price and demand as assumptions unless directly measured.

import math
import numpy as np

def required_worker_count(arrivals_per_hour, service_seconds, reserve_fraction, workers):
    """A minimum worker estimate and an independent utilization/load-bound check."""
    # TODO: Convert demand, service time, reserve, and worker count to consistent units before checking feasibility.
    raise NotImplementedError




# Account for failed attempts (Transfer / diagnosis):
# Assume \(100\) completed jobs require \(125\) equal-duration attempts. Compare active-time cost per attempt with active-time cost per completion.

# Inputs:
# - Assume \(100\) completed jobs require \(125\) equal-duration attempts. Compare active-time cost per attempt with active-time cost per completion.

# Outputs:
# - Cost per attempt and per successful completion.

# Invariants to check:
# - The two calculations differ only by their declared denominator.

# Hints:
# - Every attempt consumes resources under this hypothetical model.
# - Count all 125 attempts for attempt cost and 100 completions for completion cost.

import math

def practice_account_for_failed_attempts(completed_jobs, total_attempts, attempt_seconds):
    """Complete the account for failed attempts check."""
    # TODO: ('Compute active-time cost per attempt and per completion from 125 equal-duration attempts for 100 completed jobs.', 'Both cost denominators and their distinct totals.', 'Attempt cost uses all attempts; completion cost uses only successful completions.')
    raise NotImplementedError




# Reject impossible capacity inputs (Transfer / diagnosis):
# Try a full reserve, zero workers and a nonfinite measured duration.

# Inputs:
# - Try a full reserve, zero workers and a nonfinite measured duration.

# Outputs:
# - Input errors for infeasible reserve, worker count, and service duration.

# Invariants to check:
# - A capacity report requires reserve below one and finite positive work/service values.

# Hints:
# - A plan should not silently divide by zero or emit a meaningless worker count.
# - Check all three invalid cases before division.

import math

def practice_reject_impossible_capacity_inputs(arrival_rate, service_seconds, reserve, workers):
    """Complete the reject impossible capacity inputs check."""
    # TODO: ('Reject reserve fraction 1, worker count 0, and a nonfinite service duration before computing a plan.', 'Three explicit input rejections.', 'A finite feasible capacity estimate requires positive workers and reserve below one.')
    raise NotImplementedError

