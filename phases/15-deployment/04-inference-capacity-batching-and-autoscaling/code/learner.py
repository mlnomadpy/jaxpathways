"""Inference capacity, batching, and autoscaling · learner-only exercises (4/4 contracts ready)"""

# Exercise: Simulate a burst of four requests arriving at time zero with a two-millisecond serial service duration. Derive all finish times and response times before executing.

# Inputs:
# - Four arrival times all equal to zero and a two-millisecond serial service time.

# Outputs:
# - The four finish times and four response times in arrival order.

# Invariants to check:
# - Each request starts when the prior serial request finishes; response time is finish time minus arrival time.

# Hints:
# - Derive each finish as the previous finish plus service time.
# - Keep queue arithmetic separate from model execution timing.

import numpy as np

def simulate_four_request_burst(service_ms):
    # TODO: Calculate serial finish and response times for four arrivals at time zero.
    raise NotImplementedError




# Expose burst sensitivity (Transfer / diagnosis):
# Compare five requests spaced three milliseconds apart with five simultaneous requests. Use two-millisecond service and compare maximum response latency.

# Inputs:
# - Compare five requests spaced three milliseconds apart with five simultaneous requests. Use two-millisecond service and compare maximum response latency.

# Outputs:
# - Two simulated response-time vectors and their maxima.

# Invariants to check:
# - Both cases use five arrivals and the same 2 ms service assumption.

# Hints:
# - The worker is idle between spaced requests but never idle during the burst.
# - Keep spaced and simultaneous arrival vectors separate.

import math

def practice_expose_burst_sensitivity(spaced_arrivals_ms, simultaneous_arrivals_ms, service_ms):
    """Complete the expose burst sensitivity check."""
    # TODO: ('Simulate five requests spaced 3 ms apart and five simultaneous requests with 2 ms service.', 'Two response-time vectors and their maximum latencies.', 'Service duration is fixed at 2 ms and the two scenarios differ only in arrivals.')
    raise NotImplementedError




# Reject a broken timeline (Transfer / diagnosis):
# Pass an out-of-order arrival vector and a zero service duration. Confirm both are rejected instead of returning a plausible chart.

# Inputs:
# - Pass an out-of-order arrival vector and a zero service duration. Confirm both are rejected instead of returning a plausible chart.

# Outputs:
# - Explicit input errors for unsorted arrivals and nonpositive service.

# Invariants to check:
# - Queue calculation runs only for ordered arrivals and positive service duration.

# Hints:
# - Public simulation inputs need finite, ordered values and positive service time.
# - Validate the timeline before deriving finish times.

import math

def practice_reject_a_broken_timeline(arrival_ms, service_ms):
    """Complete the reject a broken timeline check."""
    # TODO: ('Validate monotone arrival times and strictly positive service duration before queue simulation.', 'Clear errors for out-of-order arrivals and zero service time.', 'Invalid timing inputs are rejected instead of yielding a chart or capacity estimate.')
    raise NotImplementedError




# Measure the margin below a response deadline (Transfer / diagnosis):
# For five simultaneous requests and a \(2\)-millisecond serial service, count responses that exceed a \(6\)-millisecond deadline. Explain why adding a faster model alone is not the only intervention.

# Inputs:
# - For five simultaneous requests and a \(2\)-millisecond serial service, count responses that exceed a \(6\)-millisecond deadline. Explain why adding a faster model alone is not the only intervention.

# Outputs:
# - A simulated count of requests past the deadline.

# Invariants to check:
# - Deadline violations use completion-minus-arrival response time.

# Hints:
# - Compute finish minus arrival for each request. Here a response exactly at the deadline is accepted; declare that convention.
# - Label the result as serial queue simulation, not a measured service SLO.

import math

def practice_measure_the_margin_below_a_response_deadline(arrival_ms, service_ms, deadline_ms):
    """Complete the measure the margin below a response deadline check."""
    # TODO: ('Count serial burst responses exceeding the declared 6 ms deadline and compare with an alternative queue intervention.', 'Deadline violation count and an explicitly labeled simulated response-time result.', 'The deadline and service-time assumptions remain fixed across the comparison.')
    raise NotImplementedError

