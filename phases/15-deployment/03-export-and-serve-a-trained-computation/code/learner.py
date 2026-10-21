"""Export a computation and verify its serving contract · learner-only exercises (3/3 contracts ready)"""

# Exercise: Verify a new input \([-2, 0, 3]\) against an independent NumPy expression.

# Inputs:
# - The exported computation and the new input vector [-2, 0, 3].

# Outputs:
# - The exported result and an independent NumPy result for the same input.

# Invariants to check:
# - The fresh request matches the declared numerical expression within the lesson tolerance.

# Hints:
# - Evaluate the request through the exported computation.
# - Build the host reference directly from the declared expression rather than reusing the export wrapper.

import numpy as np

def check_exported_new_input(exported_function, inputs):
    # TODO: Run the requested input through the export and compare with independent NumPy arithmetic.
    raise NotImplementedError




# Validate a request before inference (Transfer):
# Write a host-side request checker that rejects nonfinite values and the wrong shape. Test a NaN input and a wrong feature count.

# Inputs:
# - Write a host-side request checker that rejects nonfinite values and the wrong shape. Test a NaN input and a wrong feature count.

# Outputs:
# - Host-side results for malformed and valid requests.

# Invariants to check:
# - A malformed request is rejected before the exported function is called.

# Hints:
# - Validate shape and finiteness on the host before calling inference. Include a singleton batch, the wrong feature count and a NaN; specify which cases your serving signature permits.
# - Check finiteness and feature count at the serving boundary.

import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import export

def practice_validate_a_request_before_inference(infer, request, expected_features):
    """Complete the validate a request before inference check."""
    # TODO: ('Reject nonfinite values and wrong feature counts on the host before invoking the exported computation.', 'NaN and wrong-width rejections plus a valid request check.', 'Malformed requests do not reach inference.')
    raise NotImplementedError




# Prove that request validation precedes exported inference (Transfer / diagnosis):
# Build a request wrapper that records how often inference runs. Reject both a malformed feature row and a nonfinite row before calling the exported computation. Then verify one valid request.

# Inputs:
# - Build a request wrapper that records how often inference runs. Reject both a malformed feature row and a nonfinite row before calling the exported computation. Then verify one valid request.

# Outputs:
# - Call-count evidence for rejected and accepted requests.

# Invariants to check:
# - Only a fully valid finite request increments the inference counter.

# Hints:
# - Increment the counter only after validation succeeds. An error message alone does not prove which boundary rejected the input.
# - Test shape and numeric validity separately.

import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import export

def practice_prove_that_request_validation_precedes_exported_inference(infer, validator, requests):
    """Complete the prove that request validation precedes exported inference check."""
    # TODO: ('Wrap inference with a call counter and submit malformed-shape, nonfinite, and valid rows.', 'Two rejections with no calls and one valid request with exactly one call.', 'Only a validated finite row increments the inference call count.')
    raise NotImplementedError

