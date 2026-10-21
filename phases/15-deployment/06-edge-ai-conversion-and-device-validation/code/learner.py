"""Deploy at the edge: conversion, budgets and device checks · learner-only exercises (3/3 contracts ready)"""

# Exercise: Send a second raw input \([0, 255, 128]\) through the full request, and verify its independent normalized NumPy result.

# Inputs:
# - The edge request pipeline and the second raw input [0, 255, 128].

# Outputs:
# - The normalized request, pipeline result, and an independent NumPy result for that raw input.

# Invariants to check:
# - Normalization follows the request contract and the pipeline output matches the independent normalized calculation.

# Hints:
# - Apply the same normalization used by the request path before calculating the host reference.
# - Keep target-device readiness evidence separate from this numerical request check.

import numpy as np

def check_second_edge_request(request_pipeline, raw_input):
    # TODO: Send the second raw vector through the full request and compare with independent normalized arithmetic.
    raise NotImplementedError




# Make incomplete deployment evidence explicit (Transfer):
# Create a device report template with required hardware and measurement fields. Reject it as ready while those fields are missing.

# Inputs:
# - Create a device report template with required hardware and measurement fields. Reject it as ready while those fields are missing.

# Outputs:
# - A report-validation result with missing target fields listed.

# Invariants to check:
# - Absent device measurements remain pending rather than receiving synthetic pass values.

# Hints:
# - List the hardware, runtime, delegate and measurement fields before testing completeness. A missing value is different from a measured zero or an explicitly unsupported capability.
# - Validate required fields before considering readiness.

import json
import time
import numpy as np
import jax
import jax.numpy as jnp

def practice_make_incomplete_deployment_evidence_explicit(report, required_fields):
    """Complete the make incomplete deployment evidence explicit check."""
    # TODO: ('Validate that required target model/runtime/device/measurement fields are present and keep the report pending if any are absent.', 'A report schema and explicit not-ready result for missing device fields.', 'Missing target-device evidence never becomes a measured pass through default values.')
    raise NotImplementedError




# Decide whether a model speedup meets the request budget (Transfer / diagnosis):
# Use the illustrative serial times above. A \(19\)-millisecond deadline is declared before optimization. Does halving model time meet it? What is the best possible total if model time tends to zero?

# Inputs:
# - Use the illustrative serial times above. A \(19\)-millisecond deadline is declared before optimization. Does halving model time meet it? What is the best possible total if model time tends to zero?

# Outputs:
# - A deadline decision and fixed-overhead lower bound.

# Invariants to check:
# - Only the model component changes while fixed overhead remains constant.

# Hints:
# - Keep the fixed work unchanged. Compare the total request with the deadline, not the model-only duration.
# - Take the model-time-to-zero limit to find the best possible total.

import json
import time
import numpy as np
import jax
import jax.numpy as jnp

def practice_decide_whether_a_model_speedup_meets_the_request_budget(fixed_overhead_ms, model_ms, deadline_ms):
    """Complete the decide whether a model speedup meets the request budget check."""
    # TODO: ('Compute fixed overhead plus halved model time against the predeclared 19 ms deadline, then take the zero-model-time lower bound.', 'Deadline decision and irreducible total-latency floor.', 'Only model time changes; other serial costs remain fixed.')
    raise NotImplementedError

